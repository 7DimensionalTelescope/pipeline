import os
import re
import time
import shutil
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.time import Time
from typing import Any, List, Dict, Tuple, Optional, Union

import warnings

# for ccdproc imagefilecollection
# warnings.filterwarnings(
#     "ignore", category=UserWarning, message=".*multiple entries for.*"
# )
warnings.filterwarnings("ignore")

# import warnings
# warnings.filterwarnings("ignore")

from ..const import REF_DIR
from ..config import SciProcConfiguration
from .. import external
from ..services.setup import BaseSetup
from ..utils import collapse, add_suffix, swap_ext, define_output_dir
from .utils import move_file  # inputlist_parser, move_file
from .const import ZP_KEY, IC_KEYS, CORE_KEYS
from ..const import PipelineError
from ..path.path import PathHandler


class ImStack(BaseSetup):
    def __init__(
        self,
        config=None,
        logger=None,
        queue=None,
        overwrite=False,
        daily=True,
    ) -> None:

        super().__init__(config, logger, queue)
        self.overwrite = overwrite
        self.daily = daily
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
        return [
            (1, "initialize", False),
            (2, "bkgsub", False),
            (3, "zpscale", False),
            (4, "calculate_weight_map", True),
            (5, "apply_bpmask", True),
            (6, "joint_registration", False),
            (7, "convolve", True),
            (8, "stack_with_swarp", False),
            (9, "flagging", False),
        ]

    def run(self):

        self.logger.info("-" * 80)
        self.logger.info(f"Start imstack for {self.config.name}")

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

        self.config.flag.combine = True

        self.logger.info(f"Imstack Done for {self.config.name}")
        # self.logger.debug(MemoryMonitor.log_memory_usage)

    def initialize(self):
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
        self.path = PathHandler(self.input_images)
        self.path_tmp = self.path.imstack.tmp_dir

        # self.set_metadata()
        self.set_metadata_without_ccdproc()

        # Output stacked image file name
        self.config.imstack.stacked_image = (
            self.path.imstack.daily_stacked_image if self.daily else self.path.imstack.stacked_image
        )
        self.config.input.stacked_image = self.config.imstack.stacked_image
        self.logger.debug(f"Stacked Image: {self.config.imstack.stacked_image}")

        self.logger.info(f"ImStack Initialized for {self.config.name}")

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
            self.logger.info(f"Using deprojection center of first image.")
            objra = header_list[0]["OBJCTRA"]
            objdec = header_list[0]["OBJCTDEC"]
            # objra = objra.replace(' ', ':')
            # objdec = objdec.replace(' ', ':')

        self.center = f"{objra},{objdec}"
        self.logger.debug(f"Deprojection Center: {self.center}")

        # base zero point for flux scaling
        # base = np.where(self.zpvalues == np.max(self.zpvalues))[0][0]
        # self.zp_base = self.zpvalues[base]
        self.zp_base = 23.9  # uJy
        # if self.zp_base < np.max(self.zpvalues):
        #     self.logger.warning(
        #         f"Scaline downward: destination ZP: ({self.zp_base}), "
        #         f"max image ZP: ({np.max(self.zpvalues)})"
        #     )
        self.logger.debug(f"Reference ZP: {self.zp_base}")

    def bkgsub(self):
        # ------------------------------------------------------------
        # 	Global Background Subtraction
        # ------------------------------------------------------------
        _st = time.time()

        self.path_bkgsub = os.path.join(self.path_tmp, "bkgsub")
        os.makedirs(self.path_bkgsub, exist_ok=True)

        self.config.imstack.bkgsub_files = [
            f"{self.path_bkgsub}/{add_suffix(os.path.basename(f), 'bkgsub')}" for f in self.input_images
        ]

        if self.config.imstack.bkgsub_type.lower() == "dynamic":
            self.logger.info("Starting Dynamic Background Subtraction")
            self._dynamic_bkgsub()
        else:
            self.logger.info("Starting Constant Background Subtraction")
            self._const_bkgsub()
        _delt = time.time() - _st
        self.logger.debug(f"--> Done ({_delt:.1f}sec)")

        self.images_to_stack = self.config.imstack.bkgsub_files

    def _const_bkgsub(self):
        for ii, (inim, _bkg, outim) in enumerate(
            zip(self.input_images, self.skyvalues, self.config.imstack.bkgsub_files)
        ):
            self.logger.debug(f"[{ii:>6}] {os.path.basename(inim)}")
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

        bkg_files = [f"{self.path_bkgsub}/{add_suffix(os.path.basename(f), 'bkg')}" for f in self.input_images]
        bkg_rms_files = [f"{self.path_bkgsub}/{add_suffix(os.path.basename(f), 'bkgrms')}" for f in self.input_images]

        self.config.imstack.bkg_files = bkg_files
        self.config.imstack.bkg_rms_files = bkg_rms_files

        for inim, outim, bkg, bkg_rms in zip(
            self.input_images, self.config.imstack.bkgsub_files, bkg_files, bkg_rms_files
        ):
            sex_args = [
                "-CATALOG_TYPE", "NONE",  # save no source catalog
                "-CHECKIMAGE_TYPE", "BACKGROUND,BACKGROUND_RMS",
                "-CHECKIMAGE_NAME", f"{bkg},{bkg_rms}"
            ]  # fmt: skip
            sex_log = os.path.join(
                self.path_bkgsub,
                os.path.splitext(os.path.basename(outim))[0] + "_sextractor.log",
            )
            sextractor(inim, sex_args=sex_args, log_file=sex_log, logger=self.logger)

            with fits.open(inim, memmap=True) as hdul:
                _data = hdul[0].data
                _hdr = hdul[0].header
                bkg = fits.getdata(bkg)
                _data -= bkg
                fits.writeto(outim, _data, header=_hdr, overwrite=True)

    def calculate_weight_map(self):
        if self.config.imstack.gpu:
            import cupy as xp
        else:
            import numpy as xp
        from .weight import pix_err

        self.config.file.bkgsub_weight_files = [add_suffix(f, "weight") for f in self.config.imstack.bkgsub_files]
        # self.config.file.processed_files  # if you want to save single frame weights

        d_m = xp.asarray(fits.getdata(self.config.preprocess.mdark_file))
        f_m = xp.asarray(fits.getdata(self.config.preprocess.mflat_file))
        sig_z = xp.asarray(fits.getdata(self.config.preprocess.biassig_file))
        sig_f = xp.asarray(fits.getdata(self.config.preprocess.flatsig_file))
        p_z = fits.getheader(self.config.preprocess.biassig_file)["NFRAMES"]
        p_d = fits.getheader(self.config.preprocess.mdark_file)["NFRAMES"]
        p_f = fits.getheader(self.config.preprocess.flatsig_file)["NFRAMES"]
        egain = fits.getheader(self.config.preprocess.mdark_file)["EGAIN"]  # e-/ADU

        for i in range(len(self.config.file.processed_files)):
            # r_p_file = os.path.join(self.config.path.path_processed, self.config.file.processed_files[i])
            r_p_file = self.path.processed_images[i]
            r_p = xp.asarray(fits.getdata(r_p_file))

            # bkg_file = self.config.imstack.bkg_files[i]
            bkgsub_file = self.config.imstack.bkgsub_files[i]
            sig_b = xp.zeros_like(r_p)
            # sig_b = calculate_background_sigma(bkg_file, egain)
            # sig_b = xp.asarray(fits.getdata(sig_b_file))

            weight_image = pix_err(xp, r_p, d_m, f_m, sig_b, sig_z, sig_f, p_d, p_z, p_f, egain, weight=True)

            if hasattr(weight_image, "get"):  # if CuPy array
                weight_image = weight_image.get()  # Convert to NumPy array

            fits.writeto(
                # os.path.join(config.path.path_processed, weight_file),
                self.config.file.bkgsub_weight_files[i],
                data=weight_image,
                overwrite=True,
            )

    def apply_bpmask(self, badpix=0):
        import cupy as cp
        from .interpolate import (
            interpolate_masked_pixels_gpu_vectorized_weight,
            add_bpx_method,
        )

        self.logger.info("Initiating Interpolating Bad Pixels")

        path_interp = os.path.join(self.path_tmp, "interp")
        os.makedirs(path_interp, exist_ok=True)

        self.config.imstack.interp_files = [
            os.path.join(path_interp, add_suffix(os.path.basename(f), "interp")) for f in self.input_images
        ]

        bpmask_array, header = fits.getdata(self.config.preprocess.bpmask_file, header=True)
        mask = cp.asarray(bpmask_array)
        if "BADPIX" in header.keys():
            badpix = header["BADPIX"]
            self.logger.debug("BADPIX found in header. Using default value 0.")
        else:
            self.logger.warning("BADPIX not found in header. Using default value 0.")
        method = self.config.imstack.interp_type

        for i in range(len(self.config.imstack.bkgsub_files)):
            input_file = self.config.imstack.bkgsub_files[i]
            output_file = self.config.imstack.interp_files[i]

            if os.path.exists(output_file) and not self.overwrite:
                self.logger.debug(f"Skipping {output_file}")
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

        self.images_to_stack = self.config.imstack.interp_files

        self.logger.info("Completed Interpolation of Bad Pixels")

    def zpscale(self):
        """
        Store the value in header as FLXSCALE, and use it in stacking.
        Keep FSCALE_KEYWORD = FLXSCALE in SWarp config.
        The headers of the last processed images are modified.
        """
        for file, zp in zip(self.images_to_stack, self.zpvalues):
            flxscale = 10 ** (0.4 * (self.zp_base - zp))
            with fits.open(file, mode="update") as hdul:
                hdul[0].header["FLXSCALE"] = (
                    flxscale,
                    "flux scaling factor by 7DT Pipeline (ImStack)",
                )
                hdul.flush()
            self.logger.debug(f"{os.path.basename(file)} FLXSCALE: {flxscale:.3f}")

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
        #     self.logger.debug(f"[{ii:>6}] {os.path.basename(inim)}")
        #     _fscaled_image = f"{self.path_scaled}/{os.path.basename(inim).replace('fits', 'zpscaled.fits')}"
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

    def convolve(self):
        """
        This is ad-hoc. Change it to convolve after resampling and take
        advantage of uniform pixel scale.
        """
        method = self.config.imstack.convolve.lower()
        if method == "gaussian":
            from astropy.convolution import Gaussian2DKernel
            from .convolve import convolve_fft_gpu, get_edge_mask, add_conv_method
            from ..utils import force_symlink

            self.logger.info("Initiating Convolution")

            # Define output path
            path_conv = os.path.join(self.path_tmp, "conv")
            os.makedirs(path_conv, exist_ok=True)
            self.config.imstack.conv_files = [
                os.path.join(path_conv, add_suffix(os.path.basename(f), "conv")) for f in self.input_images
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
                    self.logger.debug(f"Skipping conv for max peeing image {os.path.basename(inim)}")
                    force_symlink(inim, outim)
                    if self.config.imstack.weight_map:
                        force_symlink(add_suffix(inim, "weight"), add_suffix(outim, "weight"))
                    continue

                kernel = Gaussian2DKernel(x_stddev=delta_peeing / (np.sqrt(8 * np.log(2))))  # 8*sig + 1 sized

                # sci
                im = fits.getdata(inim)
                convolved_im = convolve_fft_gpu(im, kernel)
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
            self.logger.info("Completed Gaussian convolution to match seeing")

        else:
            self.logger.info("Undefined Convolution Method. Skipping Seeing Match")

    def stack_with_swarp(self):
        self.logger.info("Initiating Stacking Images")

        # Write target images to a text file
        self.path_imagelist = os.path.join(self.path_tmp, "images_to_stack.txt")

        with open(self.path_imagelist, "w") as f:
            for inim in self.images_to_stack:  # self.zpscaled_images:
                f.write(f"{inim}\n")

        t0_stack = time.time()
        self.logger.debug(f"Total Exptime: {self.total_exptime}")

        if not self.config.imstack.weight_map:
            # self.swarp_log_file = os.path.join(
            #     self.path_tmp, f"{self.config.name}_swarp.log"
            # )

            # external.swarp(
            #     input=self.path_imagelist,
            #     output=self.config.file.stacked_file,
            #     center=self.center,
            #     resample_dir=os.path.join(self.path_tmp, "resamp"),
            #     log_file=self.swarp_log_file,
            #     logger=self.logger,
            #     weight_map=self.config.imstack.weight_map,
            # )
            self._run_swarp("")
        else:
            # science images
            self._run_swarp("sci", args=["-RESAMPLING_TYPE", "LANCZOS3"])

            # weight images
            self._run_swarp("wht", args=["-RESAMPLING_TYPE", "NEAREST"])

            if self.config.imstack.propagate_mask:
                bpmask_file = self.config.preprocess.bpmask_file
                bpmask_inverted = 1 - fits.getdata(bpmask_file)
                bpmask_inverted_file = os.path.join(self.path_tmp, os.path.basename(bpmask_file))
                fits.writeto(bpmask_inverted_file, bpmask_inverted, overwrite=True)
                self.logger.debug(f"Inverted bpmask saved as {bpmask_inverted_file}")
                args = ["-WEIGHT_IMAGE", bpmask_file, "-RESAMPLING_TYPE", "LANCZOS3"]
                self._run_swarp("bpm", args=args)

        self._update_header()

        delt_stack = time.time() - t0_stack
        self.logger.debug(f"Stacking {self.n_stack} images took {delt_stack:.3f} sec")

    def _run_swarp(self, type="", args=None):
        """pass type='' for no weight"""
        working_dir = os.path.join(self.path_tmp, type)
        resample_dir = os.path.join(working_dir, "resamp")
        log_file = os.path.join(working_dir, "_".join([self.config.name, type, "swarp.log"]))

        if type == "":
            output_file = self.config.imstack.stacked_image  # to processed directly
        else:
            output_file = os.path.join(working_dir, os.path.basename(self.config.imstack.stacked_image))

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
            shutil.move(output_file, self.config.file.stacked_file)
        elif type == "wht":
            shutil.move(
                add_suffix(output_file, "weight"),
                add_suffix(self.config.file.stacked_file, "weight"),
            )
        elif type == "bpm":
            shutil.move(
                add_suffix(output_file, "weight"),
                add_suffix(self.config.file.stacked_file, "bpmask"),
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
        with fits.open(self.config.file.stacked_file) as hdulist:
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
        with fits.open(self.config.file.stacked_file, mode="update") as hdul:
            header = hdul[0].header

            for key, (value, comment) in keywords_to_update.items():
                header[key] = (value, comment)

            # 	Names of stacked single images
            for nn, inim in enumerate(self.input_images):
                header[f"IMG{nn:0>5}"] = (os.path.basename(inim), "")

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
