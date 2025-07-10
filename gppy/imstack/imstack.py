import os
import sys
import re
import time
import shutil
import subprocess
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.time import Time
from typing import Any, List, Dict, Tuple, Optional, Union
from contextlib import nullcontext
import warnings

from ..const import REF_DIR, PipelineError
from ..config import SciProcConfiguration
from .. import external
from ..services.setup import BaseSetup
from ..utils import collapse, get_header, add_suffix, swap_ext, time_diff_in_seconds, get_basename, flatten
from ..path.path import PathHandler, NameHandler

from .utils import move_file  # inputlist_parser, move_file
from .const import ZP_KEY, IC_KEYS, CORE_KEYS
from .weight import calc_weight
from .interpolate import interpolate_masked_pixels, add_bpx_method
from .convolve import convolve_fft, add_conv_method, get_edge_mask

warnings.filterwarnings("ignore")


class ImStack(BaseSetup):
    def __init__(
        self,
        config=None,
        logger=None,
        queue=None,
        overwrite=False,
        use_gpu: bool = True,
    ) -> None:

        super().__init__(config, logger, queue)
        self.overwrite = overwrite
        self._device_id = None
        self._use_gpu = use_gpu
        self._flag_name = "combine"

        if self.config.settings.is_pipeline:
            self.config.imstack.convolve = False

    @classmethod
    def from_list(cls, input_images, working_dir=None):
        """use soft link if files are from different directories"""

        image_list = []
        for image in input_images:
            path = Path(image)
            if not path.is_file():
                print(f"{image} does not exist.")
                return None
            image_list.append(path.parts[-1])  # str

        working_dir = working_dir or os.getcwd()  # str(path.parent.absolute()

        config = SciProcConfiguration.base_config(working_dir=working_dir)
        # config.path.path_processed = working_dir
        config.config.input.calibrated_images = image_list

        self = cls(config=config)
        self.path = PathHandler(input_images)
        return self

    @property
    def sequential_task(self):
        """[(number, name, use_gpu), ...]"""
        return [(1, "run", False)]
        # return [
        #     (1, "initialize", False),
        #     (2, "bkgsub", Filse),
        #     (3, "zpscale", False),
        #     (4, "calculate_weight_map", True),
        #     (5, "apply_bpmask", True),
        #     (6, "joint_registration", False),
        #     (7, "prepare_convolution", False),
        #     (8, "run_convolution", True),
        #     (9, "save_convolved_images", False),
        #     (10, "stack_with_swarp", False),
        # ]

    def get_device_id(self, device_id):

        if self._use_gpu:
            if device_id is None:
                from ..services.utils import get_best_gpu_device

                return get_best_gpu_device()
            elif device_id == "CPU":
                return "CPU"
            from ..services.utils import check_gpu_activity

            if check_gpu_activity(device_id):
                return "CPU"
            else:
                return device_id
        else:
            return "CPU"

    def run(self, use_gpu: bool = True, device_id=None):
        try:
            self._use_gpu = all([use_gpu, self.config.imstack.gpu, self._use_gpu])

            self.initialize()

            # background subtraction
            self.bkgsub()

            # zero point scaling
            self.zpscale()

            if self.config.imstack.weight_map:
                self.calculate_weight_map(device_id=device_id)

            # replace hot pixels
            self.apply_bpmask(device_id=device_id)

            # re-registration
            if self.config.imstack.joint_wcs:
                self.joint_registration()

            # seeing convolution
            if self.config.imstack.convolve:
                self.prepare_convolution()
                self.run_convolution(device_id=device_id)
                self.run_convolution(device_id=device_id, weight=True)

            # swarp imcombine
            self.stack_with_swarp()
        except Exception as e:
            self.logger.error(f"Error during imstack processing: {str(e)}")
            raise
        # self.logger.debug(MemoryMonitor.log_memory_usage)

    def initialize(self):
        self._st = time.time()
        self.logger.info(f"Start 'ImStack'")
        # use common input if imstack.input_files override is not set
        if not (hasattr(self.config.imstack, "input_images") and self.config.imstack.input_images):
            self.config.imstack.input_images = self.config.input.calibrated_images

        self.input_images = self.config.imstack.input_images
        if not self.input_images:
            raise PipelineError("No Input for ImStack")

        self.zpkey = self.config.imstack.zp_key or ZP_KEY
        # self.ic_keys = IC_KEYS
        self.keys_to_propagate = CORE_KEYS

        # self.define_paths(working_dir=self.config.path.path_processed)
        self.path_tmp = self.path.imstack.tmp_dir

        # self.set_metadata()
        self.set_metadata_without_ccdproc()

        # Output stacked image file name
        self.config.imstack.stacked_image = (
            self.path.imstack.daily_stacked_image
            if self.config.settings.is_pipeline
            else self.path.imstack.stacked_image
        )
        self.config.input.stacked_image = self.config.imstack.stacked_image
        self.logger.debug(f"Stacked Image: {self.config.imstack.stacked_image}")

        self.logger.info(f"Initialization for ImStack is completed")

    def set_metadata_without_ccdproc(self):
        """
        Use OBJCTRA and OBJCTDEC of the first image as the deprojection center.
        The code currently does not check existence and uniqueness of the keys.

        The image with the highest ZP value gives the refernce ZP. This
        is a conservative choice for saturation levels across images.
        """
        self.n_stack = len(self.input_images)

        header_list = [fits.getheader(f) for f in self.input_images]

        self.total_exptime = np.sum([hdr["EXPTIME"] for hdr in header_list])  # sec
        self.zpvalues = [hdr[self.zpkey] for hdr in header_list]
        self.skyvalues = [hdr["SKYVAL"] for hdr in header_list]
        self.mjd_stacked = np.mean([hdr["MJD"] for hdr in header_list])
        self.satur_level = np.min([hdr["SATURATE"] * hdr["FLXSCALE"] for hdr in header_list])
        self.coadd_egain = np.sum([hdr["EGAIN"] / hdr["FLXSCALE"] for hdr in header_list])

        objs = list(set([hdr["OBJECT"] for hdr in header_list]))
        filters = list(set([hdr["FILTER"] for hdr in header_list]))
        egains = list(set([hdr["EGAIN"] for hdr in header_list]))
        camera_gains = list(set([hdr["GAIN"] for hdr in header_list]))
        self.obj = objs[0]
        self.camera_gain = camera_gains[0]

        if len(objs) != 1:
            self.logger.warning("Multiple OBJECT found. Using the first one.")
        self.filte = filters[0]
        if len(filters) != 1:
            self.logger.warning("Multiple FILTER found. Using the first one.")
        if len(egains) != 1:
            self.logger.warning("Multiple EGAIN found. Using the first one.")
        if len(camera_gains) != 1:
            self.logger.warning("Multiple GAIN found. Using the first one.")

        #   Hard coding for the UDS field
        # self.gain_default = 0.78

        self.header_ref_img = self.input_images[0]

        # Determine Deprojection Center
        # 	Tile object (e.g. T01026)
        if bool(re.match(r"T\d{5}", self.obj)):
            self.logger.info(f"Using predefined deprojection center for {self.obj}.")
            from astropy.table import Table
            from astropy.coordinates import Angle

            skygrid_table = Table.read(f"{REF_DIR}/skygrid.fits")
            indx_skygrid = skygrid_table["tile"] == self.obj

            ra = Angle(skygrid_table["ra"][indx_skygrid][0], unit="deg")
            objra = ra.to_string(unit="hourangle", sep=":", pad=True)

            dec = Angle(skygrid_table["dec"][indx_skygrid][0], unit="deg")
            objdec = dec.to_string(unit="degree", sep=":", pad=True, alwayssign=True)

        # 	Non-Tile object
        else:
            self.logger.info(f"Using the center of the first image as the deprojection center.")
            objra = header_list[0]["OBJCTRA"]
            objdec = header_list[0]["OBJCTDEC"]
            # objra = objra.replace(' ', ':')
            # objdec = objdec.replace(' ', ':')

        self.center = f"{objra},{objdec}"
        self.logger.debug(f"Deprojection center: {self.center}")

        # base zero point for flux scaling
        # base = np.where(self.zpvalues == np.max(self.zpvalues))[0][0]
        # self.zp_base = self.zpvalues[base]
        self.zp_base = 23.9  # uJy
        # if self.zp_base < np.max(self.zpvalues):
        #     self.logger.warning(
        #         f"Scaline downward: destination ZP: ({self.zp_base}), "
        #         f"max image ZP: ({np.max(self.zpvalues)})"
        #     )
        self.logger.debug(f"Reference zero point: {self.zp_base}")

    def bkgsub(self):
        # ------------------------------------------------------------
        # 	Global Background Subtraction
        # ------------------------------------------------------------
        st = time.time()

        self.path_bkgsub = os.path.join(self.path_tmp, "bkgsub")
        os.makedirs(self.path_bkgsub, exist_ok=True)

        self.config.imstack.bkgsub_images = [
            f"{self.path_bkgsub}/{add_suffix(get_basename(f), 'bkgsub')}" for f in self.input_images
        ]

        if self.config.imstack.bkgsub_type.lower() == "dynamic":
            self.logger.info("Start dynamic background subtraction")
            self._dynamic_bkgsub()
        else:
            self.logger.info("Start constant background subtraction")
            self._const_bkgsub()

        self.logger.info(f"Background subtraction is completed in {time_diff_in_seconds(st)} seconds")

        self.images_to_stack = self.config.imstack.bkgsub_images

    def _const_bkgsub(self):
        for ii, (inim, _bkg, outim) in enumerate(
            zip(self.input_images, self.skyvalues, self.config.imstack.bkgsub_images)
        ):
            self.logger.debug(f"[{ii:>6}] {get_basename(inim)}")
            if not os.path.exists(outim) or self.overwrite:
                with fits.open(inim, memmap=True) as hdul:
                    _data = hdul[0].data
                    _hdr = hdul[0].header
                    _data -= _bkg
                    self.logger.debug(f"Using SKYVAL: {_bkg:.3f}")
                    fits.writeto(outim, _data, header=_hdr, overwrite=True)

    def _dynamic_bkgsub(self):
        """
        Later to be refined using iterations
        """
        from ..external import sextractor

        bkg_images = [f"{self.path_bkgsub}/{add_suffix(get_basename(f), 'bkg')}" for f in self.input_images]
        bkg_rms_images = [f"{self.path_bkgsub}/{add_suffix(get_basename(f), 'bkgrms')}" for f in self.input_images]

        self.config.imstack.bkg_images = bkg_images
        self.config.imstack.bkg_rms_images = bkg_rms_images

        for inim, outim, bkg, bkg_rms in zip(
            self.input_images, self.config.imstack.bkgsub_images, bkg_images, bkg_rms_images
        ):

            # if result is already in factory, use that
            if os.path.exists(outim) and os.path.exists(bkg) and os.path.exists(bkg_rms):
                self.logger.info(f"Background subtraction result exists; skipping: {get_basename(outim)}")
                continue

            sex_args = [
                "-CATALOG_TYPE", "NONE",  # save no source catalog
                "-CHECKIMAGE_TYPE", "BACKGROUND,BACKGROUND_RMS",
                "-CHECKIMAGE_NAME", f"{bkg},{bkg_rms}"
            ]  # fmt: skip
            sex_log = os.path.join(
                self.path_bkgsub,
                os.path.splitext(get_basename(outim))[0] + "_sextractor.log",
            )
            sextractor(inim, sex_args=sex_args, log_file=sex_log, logger=self.logger)

            with fits.open(inim, memmap=True) as hdul:
                _data = hdul[0].data
                _hdr = hdul[0].header
                bkg = fits.getdata(bkg)
                _data -= bkg
                fits.writeto(outim, _data, header=_hdr, overwrite=True)

    @staticmethod
    def _group_IMCMB(images):
        calibs = []
        for image in images:
            header = get_header(image)
            zdf = [v for k, v in header.items() if "IMCMB" in k]  # [z, d, f]
            calibs.append(zdf)

        groups = dict()
        for image, zdf in zip(images, calibs):
            key = tuple(zdf)
            if key not in groups:
                groups[key] = []
            groups.setdefault(key, []).append(image)

        # groups = NameHandler.get_grouped_files(bkgsub_images)
        return groups

    def calculate_weight_map(self, device_id=None, use_gpu: bool = True, overwrite: bool = False):
        """Retains cpu support as a code template"""
        st = time.time()
        self._use_gpu = all([use_gpu, self.config.imstack.gpu, self._use_gpu])
        # pick xp and device‐context based on GPU flag
        device_id = self.get_device_id(device_id)

        self.logger.info(f"Start weight-map calculation with device_id: {device_id}")

        bkgsub_images = self.config.imstack.bkgsub_images
        self.config.imstack.bkgsub_weight_images = []

        groups = self._group_IMCMB(bkgsub_images)
        self.logger.debug(f"{len(groups)} groups for weight map calculation.")
        self.logger.debug(f"{groups}")

        for i, ((z_m_file, d_m_file, f_m_file), images) in enumerate(groups.items()):
            st_loop = time.time()

            header = fits.getheader(images[0])  # same cailb in a group
            calibs = [v for k, v in header.items() if "IMCMB" in k]  # must be ordered mbias, mdark, mflat
            self.logger.debug(f"Group {i} calibs: {calibs}")
            d_m_file, f_m_file, sig_z_file, sig_f_file = PathHandler.weight_map_input(calibs)

            self.logger.debug(f"{time_diff_in_seconds(st_loop)} seconds for group {i} preparation")

            uncalculated_images = []

            for img in images:
                weight_image_file = add_suffix(img, "weight")
                self.config.imstack.bkgsub_weight_images.append(weight_image_file)
                if os.path.exists(weight_image_file) and not self.overwrite:
                    self.logger.debug(f"Already exists; skip generating {weight_image_file}")
                    continue
                else:
                    uncalculated_images.append(img)

            # output = calc_weight(
            #     uncalculated_images, d_m, f_m, sig_z, sig_f, p_d, p_z, p_f, egain, weight=True, device=device_id
            # )

            if uncalculated_images:
                script = os.path.join(os.path.dirname(__file__), "weight_map_runner.py")
                cmd = [
                    sys.executable, script,
                    "--d_m_file",   d_m_file,
                    "--f_m_file",   f_m_file,
                    "--sig_z_file", sig_z_file,
                    "--sig_f_file", sig_f_file,
                    # "--p_d",        str(p_d),
                    # "--p_z",        str(p_z),
                    # "--p_f",        str(p_f),
                    "--device",     f"{device_id}",
                ] + uncalculated_images  # fmt: skip

                self.logger.debug(f"ImStack weight map command: {cmd}")

                st_image = time.time()
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    print("COMMAND:", e.cmd)
                    print("RETURN CODE:", e.returncode)
                    print("STDOUT:\n", e.stdout)
                    print("STDERR:\n", e.stderr)

                self.logger.debug(
                    f"Weight-map calculation (device={device_id}) for group {i} is completed in {time_diff_in_seconds(st_image)} seconds"
                )
            else:
                self.logger.info("All weight images already exist. Skipping weight map calculation")

            self.logger.info(
                f"Weight-map calculation is completed in {time_diff_in_seconds(st)} seconds ({time_diff_in_seconds(st, return_float=True)/len(images):.1f} s/image)"
            )

    def apply_bpmask(self, badpix=0, device_id=None, use_gpu: bool = True):
        st = time.time()
        self._use_gpu = all([use_gpu, self.config.imstack.gpu, self._use_gpu])

        device_id = self.get_device_id(device_id)

        st = time.time()
        self.logger.info("Start the interpolation for bad pixels")

        path_interp = os.path.join(self.path_tmp, "interp")
        os.makedirs(path_interp, exist_ok=True)

        self.config.imstack.interp_images = [
            os.path.join(path_interp, add_suffix(get_basename(f), "interp")) for f in self.input_images
        ]

        # bpmask_array, header = fits.getdata(self.config.preprocess.bpmask_file, header=True)
        # ad-hoc. Todo: group input_files for different bpmask files
        bpmask_array, header = fits.getdata(PathHandler.get_bpmask(self.config.imstack.bkgsub_images[0]), header=True)

        if "BADPIX" in header.keys():
            badpix = header["BADPIX"]
            self.logger.debug(f"BADPIX found in header. Using badpix {badpix}.")
        else:
            self.logger.warning("BADPIX not found in header. Using default value 0.")

        # pre-bind to avoid UnboundLocalError
        # interp = image = weight = interp_weight = None

        method = self.config.imstack.interp_type
        weight = self.config.imstack.weight_map

        groups = self._group_IMCMB(self.input_images)
        self.logger.warning(f"{len(groups)} groups detected: multi-group bpmask not implemented. Using one bpmask")

        uncalculated_images = []
        for i in range(len(self.config.imstack.bkgsub_images)):
            input_image_file = self.config.imstack.bkgsub_images[i]
            output_file = self.config.imstack.interp_images[i]

            if os.path.exists(output_file) and not self.overwrite:
                self.logger.debug(f"Already exists; skip generating {output_file}")
                continue
            else:
                if weight:
                    uncalculated_images.append([input_image_file, add_suffix(input_image_file, "weight")])
                else:
                    uncalculated_images.append([input_image_file, None])

                    interp, interp_weight = interpolate_masked_pixels(
                        image, mask, weight=weight, method=method, badpix=badpix
                    )

                    if hasattr(interp_weight, "get"):  # if CuPy array
                        interp_weight = xp.asnumpy(interp_weight)  # Convert to NumPy array

                    fits.writeto(
                        output_weight_file,
                        data=interp_weight,
                        header=add_bpx_method(fits.getheader(input_weight_file), method),
                        overwrite=True,
                    )

                else:
                    interp = interpolate_masked_pixels(image, mask, method=method, badpix=badpix)

                if hasattr(interp, "get"):  # if CuPy array
                    interp = xp.asnumpy(interp)  # Convert to NumPy array

                fits.writeto(
                    output_file,
                    data=interp,
                    header=add_bpx_method(fits.getheader(input_file), method),
                    overwrite=True,
                )

        self.images_to_stack = self.config.imstack.interp_images

        for attr in ("outputs", "output_weights"):
            if attr in self.__dict__:
                del self.__dict__[attr]

        self.logger.info(
            f"Interpolation for bad pixels is completed in {time_diff_in_seconds(st)} seconds ({time_diff_in_seconds(st, return_float=True)/len(self.images_to_stack):.1f} s/image)"
        )

    def zpscale(self):
        """
        Store the value in header as FLXSCALE, and use it in stacking.
        Keep FSCALE_KEYWORD = FLXSCALE in SWarp config.
        The headers of the last processed images are modified.
        """
        st = time.time()
        for file, zp in zip(self.images_to_stack, self.zpvalues):
            flxscale = 10 ** (0.4 * (self.zp_base - zp))
            with fits.open(file, mode="update") as hdul:
                hdul[0].header["FLXSCALE"] = (
                    flxscale,
                    "flux scaling factor by 7DT Pipeline (ImStack)",
                )
                hdul.flush()
            self.logger.debug(f"{get_basename(file)} FLXSCALE: {flxscale:.3f}")

        self.logger.info(f"ZP scaling is completed in {time_diff_in_seconds(st)} seconds")

        # ------------------------------------------------------------
        # 	ZP Scale
        # ------------------------------------------------------------
        # self.path_scaled = f"{path_output}/scaled"
        # os.makedirs(self.path_scaled, exist_ok=True)

        # self.logger.debug(f"Flux Scale to ZP={self.zp_base}")
        # zpscaled_images = []
        # _st = time.time()
        # for ii, (inim, _zp) in enumerate(
        #     zip(self.config.imstack.bkgsub_files, self.zpvalues)
        # ):
        #     self.logger.debug(f"[{ii:>6}] {get_basename(inim)}")
        #     _fscaled_image = f"{self.path_scaled}/{get_basename(inim).replace('fits', 'zpscaled.fits')}"
        #     if not os.path.exists(_fscaled_image):
        #         with fits.open(inim, memmap=True) as hdul:
        #             _data = hdul[0].data
        #             _hdr = hdul[0].header
        #             _fscale = 10 ** (0.4 * (self.zp_base - _zp))
        #             _fscaled_data = _data * _fscale
        #             self.logger.debug(
        #                 f"x {_fscale:.3f}",
        #             )
        #             fits.writeto(_fscaled_image, _fscaled_data, _hdr, overwrite=True)
        #     zpscaled_images.append(_fscaled_image)
        # self.zpscaled_images = zpscaled_images
        # _delt = time.time() - _st
        # self.logger.debug(f"--> Done ({_delt:.1f}sec)")

    def joint_registration(self):
        """
        It can address cross-filter registration when given just the image paths.
        Just give the new joint WCS to all images and let individual ImStack
        handle the rest of the process.
        """
        pass

    def prepare_convolution(self):
        """
        This is ad-hoc. Change it to convolve after resampling and take
        advantage of uniform pixel scale.
        """

        method = self.config.imstack.convolve.lower()
        self.logger.info(f"Prepare the convolution with {method} method")

        self.kernel = []
        self._convolved_images = []
        self._convolved_wht_images = []

        if method == "gaussian":
            from astropy.convolution import Gaussian2DKernel
            from ..utils import force_symlink

            # Define output path
            path_conv = os.path.join(self.path_tmp, "conv")
            os.makedirs(path_conv, exist_ok=True)
            self.config.imstack.conv_files = [
                os.path.join(path_conv, add_suffix(get_basename(f), "conv")) for f in self.input_images
            ]

            # Get peeings for convolution
            peeings = [fits.getheader(inim)["PEEING"] for inim in self.images_to_stack]
            # max_peeing = np.max(peeings)
            self._max_peeing = (
                self.config.imstack.target_seeing / collapse(self.path.pixscale, raise_error=True)
                if hasattr(self.config.imstack, "target_seeing")
                and isinstance(self.config.imstack.target_seeing, (int, float))
                else np.max(peeings)
            )
            delta_peeings = [self._calc_delta_peeing(peeing) for peeing in peeings]
            self.logger.debug(f"PEEINGs: {peeings}")

            for i, delta_peeing in enumerate(delta_peeings):
                if delta_peeing is None:
                    force_symlink(self.images_to_stack[i], self.config.imstack.conv_files[i])
                    if self.config.imstack.weight_map:
                        force_symlink(
                            add_suffix(self.images_to_stack[i], "weight"),
                            add_suffix(self.config.imstack.conv_files[i], "weight"),
                        )
                    self.kernel.append(None)
                else:
                    self.kernel.append(
                        Gaussian2DKernel(x_stddev=delta_peeing / (np.sqrt(8 * np.log(2))))
                    )  # 8*sig + 1 sized

        else:
            self.logger.info("Undefined convolution method. Skipping seeing match")

    def run_convolution(self, device_id=None, use_gpu: bool = True, weight=False):
        from .convolve import convolve_fft, get_edge_mask

        st = time.time()
        self._use_gpu = all([use_gpu, self.config.imstack.gpu, self._use_gpu])
        device_id = self.get_device_id(device_id)
        if self._use_gpu:
            self.logger.info(f"Using GPU {device_id} for convolution")
        else:
            self.logger.info("Using CPU for convolution")

        self._convolved_images = []
        self._convolved_wht_images = []

        for i in range(len(self.images_to_stack)):
            if self.kernel[i] is None:
                self._convolved_images.append(None)
                self._convolved_wht_images.append(None)
                self.logger.info(
                    f"Convolution is skipped for images due to no kernel [{i+1}/{len(self.images_to_stack)}]"
                )
                continue
            inim = self.images_to_stack[i]
            im = fits.getdata(inim)
            convolved_im = convolve_fft(im, self.kernel[i], device_id=device_id)
            self._convolved_images.append(convolved_im)
            if self.config.imstack.weight_map:
        self._convolved_images = []
        self._convolved_wht_images = []

        for i in range(len(self.images_to_stack)):
            if self.kernel[i] is None:
                self._convolved_images.append(None)
                self._convolved_wht_images.append(None)
                self.logger.info(
                    f"Convolution is skipped for images due to no kernel [{i+1}/{len(self.images_to_stack)}]"
                )
                continue
            inim = self.images_to_stack[i]
            im = fits.getdata(inim)
            convolved_im = convolve_fft(im, self.kernel[i], device_id=device_id)
            self._convolved_images.append(convolved_im)
            if self.config.imstack.weight_map:
                inim_wht = add_suffix(inim, "weight")
                if os.path.exists(inim_wht):
                    image_list.append((inim, inim_wht))
                else:
                    self.logger.warning(f"Weight map not found for {inim}. Skipping.")
        else:
            image_list = self.images_to_stack

        output = convolve_fft(image_list, self.kernel, device_id=device_id, apply_edge_mask=weight)

        for img, out in zip(image_list, output):
            peeing = fits.getheader(self.images_to_stack[i])["PEEING"]
            delta_peeing = self._calc_delta_peeing(peeing)
            if delta_peeing is None:
                continue
            header = add_conv_method(fits.getheader(self.images_to_stack[i]), delta_peeing, method)
            fits.writeto(
                img,
                out,
                header=header,
                overwrite=True,
            )
        self.logger.info(
            f"Convolution is completed in {time_diff_in_seconds(st)} seconds ({time_diff_in_seconds(st, return_float=True)/len(self.images_to_stack):.1f} s/image)"
        )

    def _calc_delta_peeing(self, peeing):
        delta_peeing = np.sqrt(self._max_peeing**2 - peeing**2)
        if delta_peeing == 0:
            self.logger.debug(f"Skipping calculating delta peeing.")
            return None
        else:
            return delta_peeing

    def stack_with_swarp(self):
        st = time.time()
        self.logger.info("Start to run swarp for stacking images")

        # Write target images to a text file
        self.path_imagelist = os.path.join(self.path_tmp, "images_to_stack.txt")

        with open(self.path_imagelist, "w") as f:
            for inim in self.images_to_stack:  # self.zpscaled_images:
                f.write(f"{inim}\n")

        self.logger.debug(f"Total Exptime: {self.total_exptime}")

        if not self.config.imstack.weight_map:
            # if no weight map, just stack the images
            self._run_swarp("")

        else:
            # science images
            self._run_swarp("sci", args=["-RESAMPLING_TYPE", "LANCZOS3"])

            # weight images
            self._run_swarp("wht", args=["-RESAMPLING_TYPE", "NEAREST"])

            # Update/Todo: consider uncollapsed bpmask files
            if self.config.imstack.propagate_mask:
                # bpmask_file = self.config.preprocess.bpmask_file
                bpmask_file = PathHandler.get_bpmask(self.images_to_stack)
                bpmask_inverted = 1 - fits.getdata(bpmask_file)
                bpmask_inverted_file = os.path.join(self.path_tmp, get_basename(bpmask_file))
                fits.writeto(bpmask_inverted_file, bpmask_inverted, overwrite=True)
                self.logger.debug(f"Inverted bpmask saved as {bpmask_inverted_file}")
                args = ["-WEIGHT_IMAGE", bpmask_file, "-RESAMPLING_TYPE", "LANCZOS3"]
                self._run_swarp("bpm", args=args)

        self._update_header()

        self.logger.info(f"Running swarp is completed in {time_diff_in_seconds(st)} seconds")

        self.config.flag.combine = True
        self.logger.info(f"'ImStack' is Completed in {time_diff_in_seconds(self._st)} seconds")

    def _run_swarp(self, type="", args=None):
        """Pass type='' for no weight"""
        working_dir = os.path.join(self.path_tmp, type)
        resample_dir = os.path.join(working_dir, "resamp")
        log_file = os.path.join(working_dir, "_".join([self.config.name, type, "swarp.log"]))

        if type == "":
            output_file = self.config.imstack.stacked_image  # output to output_dir directly
        else:
            # output to tmp dir, and then selectively move to output_dir
            output_file = os.path.join(working_dir, get_basename(self.config.imstack.stacked_image))

        external.swarp(
            input=self.path_imagelist,
            output=output_file,
            center=self.center,
            resample_dir=resample_dir,
            log_file=log_file,
            logger=self.logger,
            weight_map=self.config.imstack.weight_map,
            swarp_args=args,
        )

        # move files to the final directory
        if type == "sci":
            shutil.move(output_file, self.config.imstack.stacked_image)
        elif type == "wht":
            shutil.move(
                add_suffix(output_file, "weight"),
                add_suffix(self.config.imstack.stacked_image, "weight"),
            )
        elif type == "bpm":
            shutil.move(
                add_suffix(output_file, "weight"),
                add_suffix(self.config.imstack.stacked_image, "bpmask"),
            )

        return

    def stack_with_cupy(self):
        pass

    def _update_header(self):
        # 	Get Header info
        exptime_stacked = self.total_exptime
        mjd_stacked = self.mjd_stacked
        jd_stacked = Time(mjd_stacked, format="mjd").jd
        dateobs_stacked = Time(mjd_stacked, format="mjd").isot
        # gain = (2 / 3) * self.n_stack * self.egain
        # airmass_stacked = np.mean(airmasslist)
        # dateloc_stacked = calc_mean_dateloc(dateloclist)
        # alt_stacked = np.mean(altlist)
        # az_stacked = np.mean(azlist)

        # datestr, timestr = extract_date_and_time(dateobs_stacked)
        # comim = f"{self.path_save}/calib_{self.config.unit}_{self.obj}_{datestr}_{timestr}_{self.filte}_{exptime_stacked:g}.com.fits"

        # 	Get Select Header Keys from Base Image
        with fits.open(self.header_ref_img) as hdulist:
            header = hdulist[0].header
            select_header_dict = {key: header.get(key, None) for key in self.keys_to_propagate}

        # 	Put them into stacked image
        with fits.open(self.config.imstack.stacked_image) as hdulist:
            header = hdulist[0].header
            for key in select_header_dict.keys():
                header[key] = select_header_dict[key]

        # 	Additional Header Information
        keywords_to_update = {
            "DATE-OBS": (dateobs_stacked, "Time of observation (UTC) for combined image"),
            # 'DATE-LOC': (dateloc_stacked, 'Time of observation (local) for combined image'),
            "EXPTIME": (exptime_stacked, "[s] Total exposure duration for combined image"),
            "EXPOSURE": (exptime_stacked, "[s] Total exposure duration for combined image"),
            # 'CENTALT' : (alt_stacked,     '[deg] Average altitude of telescope for combined image'),
            # 'CENTAZ'  : (az_stacked,      '[deg] Average azimuth of telescope for combined image'),
            # 'AIRMASS' : (airmass_stacked, 'Average airmass at frame center for combined image (Gueymard 1993)'),
            "MJD": (mjd_stacked, "Modified Julian Date at start of observations for combined image"),
            "JD": (jd_stacked, "Julian Date at start of observations for combined image"),
            "SKYVAL": (0, "SKY MEDIAN VALUE (Subtracted)"),
            "EGAIN": (self.coadd_egain, "Effective EGAIN for combined image (e-/ADU)"),  # swarp calculates it as GAIN, but irreproducible.
            "GAIN": (self.camera_gain, "Gain from the camera configuration"),
            "SATURATE": (self.satur_level, "Conservative saturation level for combined image"),  # let swarp handle this
        }  # fmt: skip

        # 	Update Header
        with fits.open(self.config.imstack.stacked_image, mode="update") as hdul:
            header = hdul[0].header

            for key, (value, comment) in keywords_to_update.items():
                header[key] = (value, comment)

            # 	Names of stacked single images
            for nn, inim in enumerate(self.input_images):
                header[f"IMG{nn:0>5}"] = (get_basename(inim), "")

            hdul.flush()


#   Example
if __name__ == "__main__":
    image_lists = [
        #   Broad bands
        "/large_data/Commission/NGC0253/g/select_median.txt",
        "/large_data/Commission/NGC0253/r/select_median.txt",
        "/large_data/Commission/NGC0253/i/select_median.txt",
        "/large_data/Commission/NGC0253/z/select_median.txt",
        #   Medium bands
        "/large_data/Commission/NGC0253/m400/select_median.txt",
        "/large_data/Commission/NGC0253/m425/select_median.txt",
        "/large_data/Commission/NGC0253/m450/select_median.txt",
        "/large_data/Commission/NGC0253/m475/select_median.txt",
        "/large_data/Commission/NGC0253/m500/select_median.txt",
        "/large_data/Commission/NGC0253/m525/select_median.txt",
        "/large_data/Commission/NGC0253/m550/select_median.txt",
        "/large_data/Commission/NGC0253/m575/select_median.txt",
        "/large_data/Commission/NGC0253/m600/select_median.txt",
        "/large_data/Commission/NGC0253/m625/select_median.txt",
        "/large_data/Commission/NGC0253/m650/select_median.txt",
        "/large_data/Commission/NGC0253/m675/select_median.txt",
        "/large_data/Commission/NGC0253/m700/select_median.txt",
        "/large_data/Commission/NGC0253/m725/select_median.txt",
        "/large_data/Commission/NGC0253/m750/select_median.txt",
        "/large_data/Commission/NGC0253/m775/select_median.txt",
        "/large_data/Commission/NGC0253/m800/select_median.txt",
        "/large_data/Commission/NGC0253/m825/select_median.txt",
        "/large_data/Commission/NGC0253/m850/select_median.txt",
        "/large_data/Commission/NGC0253/m875/select_median.txt",
    ]

    for image_list in image_lists:
        ImStack.from_text_file(image_list).run()
