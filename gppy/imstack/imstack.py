import os
import re
import time
import shutil
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.time import Time
from typing import Any, List, Dict, Tuple, Optional, Union
from contextlib import nullcontext

import warnings

warnings.filterwarnings("ignore")

from ..const import REF_DIR
from ..config import SciProcConfiguration
from .. import external
from ..services.setup import BaseSetup
from ..utils import collapse, get_header, add_suffix, swap_ext, define_output_dir, time_diff_in_seconds, get_basename
from .utils import move_file  # inputlist_parser, move_file
from .const import ZP_KEY, IC_KEYS, CORE_KEYS
from ..const import PipelineError
from ..path.path import PathHandler, NameHandler


class ImStack(BaseSetup):
    def __init__(self, config=None, logger=None, queue=None, overwrite=False) -> None:

        super().__init__(config, logger, queue)
        self.overwrite = overwrite
        self._device_id = None
        self._flag_name = "combine"


    @classmethod
    def from_list(cls, input_images):
        """use soft link if files are from different directories"""

        image_list = []
        for image in input_images:
            path = Path(image)
            if not path.is_file():
                print(f"{image} does not exist.")
                return None
            image_list.append(path.parts[-1])  # str

        working_dir = str(path.parent.absolute())

        config = SciProcConfiguration.base_config(working_dir=working_dir)
        # config.path.path_processed = working_dir
        config.input.calibrated_images = image_list

        return cls(config=config)

    @property
    def sequential_task(self):
        """[(number, name, use_gpu), ...]"""
        return [
            (1, "initialize", False),
            (2, "bkgsub", False),
            (3, "zpscale", False),
            (4, "calculate_weight_map", False),
            (5, "apply_bpmask", False),
            (6, "joint_registration", False),
            (7, "convolve", False),
            (8, "stack_with_swarp", False),
        ]

    def get_device_id(self, device_id):
        if device_id is not None:
            self._device_id = device_id
            return device_id
        elif self._device_id is not None:
            self.config.imstack.device = self._device_id
            return self._device_id
        elif self.config.imstack.device is None:
            from ..services.utils import get_best_gpu_device
            self.config.imstack.device = get_best_gpu_device()
            
        return self.config.imstack.device

    def run(self):
        try:
            self.initialize()

            # background subtraction
            self.bkgsub()

            # zero point scaling
            self.zpscale()

            if self.config.imstack.weight_map:
                self.calculate_weight_map()

            # replace hot pixels
            self.apply_bpmask()

            # re-registration
            if self.config.imstack.joint_wcs:
                self.joint_registration()

            # seeing convolution
            if self.config.imstack.convolve:
                self.convolve()

            # swarp imcombine
            self.stack_with_swarp()
        except Exception as e:
            self.logger.error(f"Error during imstack processing: {str(e)}", exc_info=True)
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

    def calculate_weight_map(self, device_id=None):
        """Retains cpu support as a code template"""
        st = time.time()
        use_gpu = self.config.imstack.gpu
        self.logger.info(f"Start weight-map calculation with {'GPU' if use_gpu else 'CPU'}")
        # pick xp and device‐context based on GPU flag
        if use_gpu:
            import cupy as xp
            device_id = self.get_device_id(device_id)
            device_ctx = xp.cuda.Device(device_id)
        else:
            import numpy as xp

            device_ctx = nullcontext()  # no‐op context manager for CPU

        from .weight import pix_err

        # self.config.imstack.input_images  # if you want to save single frame weights
        bkgsub_images = self.config.imstack.bkgsub_images
        self.config.imstack.bkgsub_weight_images = []
        # bkgsub_weight_images = [add_suffix(f, "weight") for f in bkgsub_images]
        # self.config.imstack.bkgsub_weight_images = bkgsub_weight_images

        # DEPRECATED: same groups may have different calibs, though unlikely
        # groups = NameHandler.get_grouped_files(bkgsub_images)
        # for (typ_tuple, obs_params_tuple), images in groups.items():

        groups = self._group_IMCMB(bkgsub_images)
        self.logger.debug(f"{len(groups)} groups for weight map calculation.")
        self.logger.debug(f"{groups}")

        with device_ctx:
            for (z_m_file, d_m_file, f_m_file), images in groups.items():

                header = fits.getheader(images[0])  # same cailb in a group
                calibs = [v for k, v in header.items() if "IMCMB" in k]  # must be ordered mbias, mdark, mflat
                d_m_file, f_m_file, sig_z_file, sig_f_file = PathHandler.weight_map_input(calibs)

                d_m = xp.asarray(fits.getdata(d_m_file))
                f_m = xp.asarray(fits.getdata(f_m_file))
                sig_z = xp.asarray(fits.getdata(sig_z_file))
                sig_f = xp.asarray(fits.getdata(sig_f_file))
                p_z = fits.getheader(sig_z_file)["NFRAMES"]
                p_d = fits.getheader(d_m_file)["NFRAMES"]
                p_f = fits.getheader(f_m_file)["NFRAMES"]

                # Pick the master dark as the source to read EGAIN.
                # The choice shouldn't matter as long as you're in the same group
                egain = fits.getheader(d_m_file)["EGAIN"]  # e-/ADU

                for i in range(len(images)):
                    r_p_file = images[i]
                    r_p = xp.asarray(fits.getdata(r_p_file))

                    # bkg_file = self.config.imstack.bkg_files[i]
                    # bkgsub_file = self.config.imstack.bkgsub_files[i]
                    sig_b = xp.zeros_like(r_p)
                    # sig_b = calculate_background_sigma(bkg_file, egain)
                    # sig_b = xp.asarray(fits.getdata(sig_b_file))

                    weight_image = pix_err(xp, r_p, d_m, f_m, sig_b, sig_z, sig_f, p_d, p_z, p_f, egain, weight=True)

                    if hasattr(weight_image, "get"):  # if CuPy array
                        weight_image = weight_image.get()  # Convert to NumPy array

                    weight_image_file = add_suffix(r_p_file, "weight")
                    self.config.imstack.bkgsub_weight_images.append(weight_image_file)

                    fits.writeto(
                        # os.path.join(config.path.path_processed, weight_file),
                        weight_image_file,
                        data=weight_image,
                        overwrite=True,
                    )
        del r_p, d_m, f_m, sig_b, sig_z, sig_f, p_d, p_z, p_f, egain, weight_image
        cp.get_default_memory_pool().free_all_blocks()
        self.logger.info(f"Weight-map calculation is completed in {time_diff_in_seconds(st)} seconds")

    def apply_bpmask(self, badpix=0, device_id=None):
        st = time.time()
        device_id = self.get_device_id(device_id)

        import cupy as cp
        from .interpolate import (
            interpolate_masked_pixels_gpu_vectorized_weight,
            add_bpx_method,
        )

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

        with cp.cuda.Device(device_id):
            mask = cp.asarray(bpmask_array)
            if "BADPIX" in header.keys():
                badpix = header["BADPIX"]
                self.logger.debug(f"BADPIX found in header. Using badpix {badpix}.")
            else:
                self.logger.warning("BADPIX not found in header. Using default value 0.")
            method = self.config.imstack.interp_type

            for i in range(len(self.config.imstack.bkgsub_images)):
                input_file = self.config.imstack.bkgsub_images[i]
                output_file = self.config.imstack.interp_images[i]

                if os.path.exists(output_file) and not self.overwrite:
                    self.logger.debug(f"Already exists; skip generating {output_file}")
                    continue

                image = cp.asarray(fits.getdata(input_file))

                if self.config.imstack.weight_map:
                    input_weight_file = add_suffix(input_file, "weight")
                    output_weight_file = add_suffix(output_file, "weight")
                    weight = cp.asarray(fits.getdata(input_weight_file))  # as 1/VARIANCE

                    interp, interp_weight = interpolate_masked_pixels_gpu_vectorized_weight(
                        image, mask, weight=weight, method=method, badpix=badpix
                    )
                    fits.writeto(
                        output_weight_file,
                        interp_weight.get(),
                        header=add_bpx_method(fits.getheader(input_weight_file), method),
                        overwrite=True,
                    )

                else:
                    interp = interpolate_masked_pixels_gpu_vectorized_weight(image, mask, method=method, badpix=badpix)

                fits.writeto(
                    output_file,
                    interp.get(),
                    header=add_bpx_method(fits.getheader(input_file), method),
                    overwrite=True,
                )

        self.images_to_stack = self.config.imstack.interp_images
        del mask, interp, image, weight, interp_weight
        cp.get_default_memory_pool().free_all_blocks()
        self.logger.info(f"Interpolation for bad pixels is completed in {time_diff_in_seconds(st)} seconds")

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

    def convolve(self, device_id=None):
        """
        This is ad-hoc. Change it to convolve after resampling and take
        advantage of uniform pixel scale.
        """
        st = time.time()
        device_id = self.get_device_id(device_id)

        method = self.config.imstack.convolve.lower()
        self.logger.info(f"Start the convolution with {method} method")

        if method == "gaussian":
            from astropy.convolution import Gaussian2DKernel
            from .convolve import convolve_fft_gpu, get_edge_mask, add_conv_method
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
            max_peeing = (
                self.config.imstack.target_seeing / collapse(self.path.pixscale, raise_error=True)
                if hasattr(self.config.imstack, "target_seeing")
                and isinstance(self.config.imstack.target_seeing, (int, float))
                else np.max(peeings)
            )
            delta_peeings = [np.sqrt(max_peeing**2 - peeing**2) for peeing in peeings]
            self.logger.debug(f"PEEINGs: {peeings}")

            # convolve
            for i in range(len(self.images_to_stack)):
                inim = self.images_to_stack[i]
                delta_peeing = delta_peeings[i]
                outim = self.config.imstack.conv_files[i]

                # Skip the seeing reference image
                if delta_peeing == 0:
                    self.logger.debug(f"Skipping conv for max peeing image {get_basename(inim)}")
                    force_symlink(inim, outim)
                    if self.config.imstack.weight_map:
                        force_symlink(add_suffix(inim, "weight"), add_suffix(outim, "weight"))
                    continue

                kernel = Gaussian2DKernel(x_stddev=delta_peeing / (np.sqrt(8 * np.log(2))))  # 8*sig + 1 sized

                # sci
                im = fits.getdata(inim)
                convolved_im = convolve_fft_gpu(im, kernel, device_id=device_id)
                fits.writeto(
                    outim,
                    convolved_im,
                    header=add_conv_method(fits.getheader(inim), delta_peeing, method),
                    overwrite=True,
                )

                # wht
                if self.config.imstack.weight_map:
                    inim_wht = add_suffix(inim, "weight")
                    outim_wht = add_suffix(outim, "weight")
                    if os.path.exists(inim_wht):
                        wht = fits.getdata(inim_wht)
                        convolved_wht = convolve_fft_gpu(wht, kernel)

                        # Expand zero-padding by the size of the kernel
                        mask = get_edge_mask(wht, kernel)
                        convolved_wht[~mask] = 0

                        fits.writeto(
                            outim_wht,
                            convolved_wht,
                            header=add_conv_method(fits.getheader(inim_wht), delta_peeing, method),
                            overwrite=True,
                        )
                    else:
                        self.logger.warning(f"Weight map not found for {inim}")

            self.images_to_stack = self.config.imstack.conv_files

        else:
            self.logger.info("Undefined convolution method. Skipping seeing match")

        del kernel, mask, convolved_wht, wht
        cp.get_default_memory_pool().free_all_blocks()
        
        self.logger.info(f"Convolution is completed in {time_diff_in_seconds(st)} seconds")

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
