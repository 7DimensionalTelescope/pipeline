import os
import re
import time
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.time import Time
from ccdproc import ImageFileCollection
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
from ..config import Configuration
from ..base import BaseSetup
from ..utils import swap_ext, define_output_dir
from .utils import extract_date_and_time, calc_mean_dateloc, inputlist_parser, unpack
from .const import ZP_KEY, IC_KEYS, KEYS_TO_ADD


class ImStack(BaseSetup):
    def __init__(
        self,
        config=None,
        logger=None,
        queue=None,
        overwrite=False,
    ) -> None:

        super().__init__(config, logger, queue)
        self.overwrite = overwrite

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

        config = Configuration.base_config(working_dir)
        config.path.path_processed = working_dir
        config.file.processed_files = image_list

        return cls(config=config)

    @classmethod
    def from_file(cls, imagelist_file):
        input_images = inputlist_parser(imagelist_file)
        cls.from_list(input_images)

    def run(self):

        self.initialize()

        # background subtraction
        self.bkgsub()

        if self.config.imstack.weight_map:
            self.calculate_weight_map()

        # replace hot pixels
        self.apply_bpmask()

        # zero point scaling
        self.zpscale()

        # re-registration
        if self.config.imstack.joint_wcs:
            self.joint_registration()

        # swarp imcombine
        self.stack_with_swarp()

    def initialize(self):
        self.files = [
            os.path.join(self.config.path.path_processed, f)
            for f in self.config.file.processed_files
        ]

        self.zpkey = self.config.imstack.zp_key or ZP_KEY
        self.ic_keys = IC_KEYS
        self.keywords_to_add = KEYS_TO_ADD

        # self.define_paths(working_dir=self.config.path.path_processed)
        self.define_paths(dump_dir=self.config.path.path_factory)

        # self.set_metadata()
        self.set_metadata_without_ccdproc()

        # Output stacked image file name
        parts = os.path.basename(self.files[-1]).split("_")
        parts[-1] = f"{self.total_exptime:.0f}s.com.fits"
        self.config.file.stacked_file = "_".join(parts)

        self.logger.info(f"ImStack Initialized for {self.config.name}")

    def define_paths(self, dump_dir=None):
        # ------------------------------------------------------------
        #   Define Paths
        # ------------------------------------------------------------
        path_tmp = dump_dir or "./tmp_imstack"
        # path_save = f"/lyman/data1/Commission/{self.obj}/{self.filte}"

        # 	Image List for SWarp
        try:
            self.path_imagelist = os.path.join(path_tmp, "images_to_stack.txt")
        except:
            self.logger.info("config.path.path_factory not defined. Using CWD")
            self.path_imagelist = os.path.join(path_tmp, "images_to_stack.txt")

        # 	Background Subtracted
        self.path_bkgsub = os.path.join(path_tmp, "bkgsub")

        # 	Flux Scaled
        # self.path_scaled = f"{path_output}/scaled"
        # os.makedirs(self.path_scaled, exist_ok=True)

        #   Interpolate Bad Pixels
        self.path_interp = os.path.join(path_tmp, "interp")

        # 	Resampled (temp. files from SWarp)
        self.path_resamp = os.path.join(path_tmp, "resamp")

        self.swarp_log_file = os.path.join(path_tmp, f"{self.config.name}_swarp.log")

    def set_metadata(self):
        """This interferes with logger due to ccdproc"""
        # ------------------------------------------------------------
        # 	Setting Metadata
        # ------------------------------------------------------------
        self.logger.debug(f"Reading images... (takes a few mins)")

        # 	Get Image Collection (takes some time)
        ic = ImageFileCollection(filenames=self.files, keywords=self.ic_keys)
        self.ic = ic

        filtered_table = ic.summary[~ic.summary[self.zpkey].mask]
        self.logger.debug(f"{len(ic.summary)}, {len(filtered_table)}")

        #   Former def
        # self.files = ic.files
        # self.n_stack = len(self.files)
        # self.zpvalues = ic.summary[self.zpkey].data
        # self.skyvalues = ic.summary["SKYVAL"].data
        # self.objra = ic.summary["OBJCTRA"].data[0].replace(" ", ":")
        # self.objdec = ic.summary["OBJCTDEC"].data[0].replace(" ", ":")
        # self.mjd_stacked = np.mean(ic.summary["MJD"].data)

        #   New def
        self.files = [filename for filename in filtered_table["file"]]
        self.n_stack = len(self.files)
        self.zpvalues = filtered_table[self.zpkey].data
        self.skyvalues = filtered_table["SKYVAL"].data
        self.objra = filtered_table["OBJCTRA"].data[0].replace(" ", ":")
        self.objdec = filtered_table["OBJCTDEC"].data[0].replace(" ", ":")
        self.mjd_stacked = np.mean(filtered_table["MJD"].data)

        # 	Total Exposure Time [sec]
        self.total_exptime = np.sum(filtered_table["EXPTIME"])

        objs = np.unique(filtered_table["OBJECT"].data)
        filters = np.unique(filtered_table["FILTER"].data)
        egains = np.unique(filtered_table["EGAIN"].data)
        self.logger.debug(f"OBJECT(s): {objs} (N={len(objs)})")
        self.logger.debug(f"FILTER(s): {filters} (N={len(filters)})")
        self.logger.debug(f"EGAIN(s): {egains} (N={len(egains)})")
        self.obj = unpack(objs, "object")
        self.filte = unpack(filters, "filter", ex="m650")
        self.gain_default = unpack(egains, "egain", ex="0.256190478801727")
        #   Hard coding for the UDS field
        # self.gain_default = 0.78

        # ------------------------------------------------------------
        # 	Base Image for Image Alignment
        # ------------------------------------------------------------
        self.header_ref_img = self.files[0]
        # self.zp_base = ic.summary[self.zpkey][0]
        self.zp_base = filtered_table[self.zpkey][0]

        # ------------------------------------------------------------
        # 	Print Input Summary
        # ------------------------------------------------------------
        self.logger.debug(f"Input Images to Stack ({len(self.files):_}):")
        for ii, inim in enumerate(self.files):
            self.logger.debug(f"[{ii:>6}] {os.path.basename(inim)}")
            if ii > 10:
                self.logger.debug("...")
                break

        # return ic

    def set_metadata_without_ccdproc(self):
        """
        Use OBJCTRA and OBJCTDEC of the first image as the deprojection center.
        The code currently does not check existence and uniqueness of the keys.

        The image with the highest ZP value gives the refernce ZP. This
        is a conservative choice for saturation levels across images.
        """
        self.n_stack = len(self.files)
        header_list = [fits.getheader(f) for f in self.files]

        self.zpvalues = [hdr[self.zpkey] for hdr in header_list]
        self.skyvalues = [hdr["SKYVAL"] for hdr in header_list]
        self.mjd_stacked = np.mean([hdr["MJD"] for hdr in header_list])

        # 	Total Exposure Time [sec]
        self.total_exptime = np.sum([hdr["EXPTIME"] for hdr in header_list])

        objs = list(set([hdr["OBJECT"] for hdr in header_list]))
        filters = list(set([hdr["FILTER"] for hdr in header_list]))
        egains = list(set([hdr["EGAIN"] for hdr in header_list]))
        self.obj = objs[0]
        if len(objs) != 1:
            self.logger.warning("Multiple OBJECTs found. Using the first one.")
        self.filte = filters[0]
        if len(filters) != 1:
            self.logger.warning("Multiple FILTERs found. Using the first one.")
        self.gain_default = float(egains[0])
        if len(egains) != 1:
            self.logger.warning("Multiple EGAINs found. Using the first one.")

        #   Hard coding for the UDS field
        # self.gain_default = 0.78

        self.header_ref_img = self.files[0]

        # Deprojection Center
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
        base = np.where(self.zpvalues == np.max(self.zpvalues))[0][0]
        self.zp_base = self.zpvalues[base]
        self.logger.debug(f"Reference ZP: {self.zp_base}")

    def bkgsub(self):
        # ------------------------------------------------------------
        # 	Global Background Subtraction
        # ------------------------------------------------------------
        _st = time.time()

        os.makedirs(self.path_bkgsub, exist_ok=True)

        self.config.imstack.bkgsub_files = [
            f"{self.path_bkgsub}/{swap_ext(os.path.basename(f), 'bkgsub.fits')}"
            for f in self.files
        ]

        if self.config.imstack.bkgsub_mode.lower() == "dynamic":
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
            zip(self.files, self.skyvalues, self.config.imstack.bkgsub_files)
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

        bkg_files = [
            f"{self.path_bkgsub}/{swap_ext(os.path.basename(f), 'bkg.fits')}"
            for f in self.files
        ]
        bkg_rms_files = [
            f"{self.path_bkgsub}/{swap_ext(os.path.basename(f), 'bkgrms.fits')}"
            for f in self.files
        ]

        self.config.imstack.bkg_files = bkg_files
        self.config.imstack.bkg_rms_files = bkg_rms_files

        for inim, outim, bkg, bkg_rms in zip(
            self.files, self.config.imstack.bkgsub_files, bkg_files, bkg_rms_files
        ):
            sex_args = [
                "-CATALOG_TYPE", "NONE",  # save no source catalog
                "-CHECKIMAGE_TYPE", "BACKGROUND,BACKGROUND_RMS",
                "-CHECKIMAGE_NAME", f"{bkg},{bkg_rms}"
            ]  # fmt: skip
            sex_log = os.path.join(
                self.config.path.path_factory,
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
        from .weight import calculate_weight

        # self.config.file.weight_files = [
        #     swap_ext(f, "weight.fits") for f in self.config.file.processed_files
        # ]

        calculate_weight(self.config)

    def apply_bpmask(self):
        import cupy as cp
        from .interpolate import interpolate_masked_pixels_gpu_batched

        os.makedirs(self.path_interp, exist_ok=True)

        self.config.imstack.interp_files = [
            os.path.join(self.path_interp, swap_ext(os.path.basename(f), "interp.fits"))
            for f in self.files
        ]
        # bpmask_files = self.config.preprocess.bpmask_file

        mask = cp.asarray(fits.getdata(self.config.preprocess.bpmask_file))

        for inim, outim in zip(
            self.files,
            self.config.imstack.interp_files,
        ):
            image = cp.asarray(fits.getdata(inim))

            interp = interpolate_masked_pixels_gpu_batched(image, mask)
            fits.writeto(
                outim, interp.get(), header=fits.getheader(inim), overwrite=True
            )

        self.images_to_stack = self.config.imstack.interp_files

    def zpscale(self):
        """
        Store the value in header as FLXSCALE, and use it in stacking.
        Keep FSCALE_KEYWORD = FLXSCALE in SWarp config.
        The headers of the last processed images are modified.
        """
        # ------------------------------------------------------------
        # 	ZP Scale
        # ------------------------------------------------------------
        for file, zp in zip(self.images_to_stack, self.zpvalues):
            flxscale = 10 ** (0.4 * (self.zp_base - zp))
            with fits.open(file, mode="update") as hdul:
                hdul[0].header["FLXSCALE"] = flxscale
                hdul.flush()
                self.logger.debug(f"{os.path.basename(file)} FLXSCALE: {flxscale:.3f}")

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
        pass

    def stack_with_swarp(self):
        os.makedirs(self.path_resamp, exist_ok=True)

        # Write target images to a text file
        with open(self.path_imagelist, "w") as f:
            for inim in self.images_to_stack:  # self.zpscaled_images:
                f.write(f"{inim}\n")

        # 	Get Header info
        exptime_stacked = self.total_exptime
        mjd_stacked = self.mjd_stacked
        jd_stacked = Time(mjd_stacked, format="mjd").jd
        dateobs_stacked = Time(mjd_stacked, format="mjd").isot
        # airmass_stacked = np.mean(airmasslist)
        # dateloc_stacked = calc_mean_dateloc(dateloclist)
        # alt_stacked = np.mean(altlist)
        # az_stacked = np.mean(azlist)

        # datestr, timestr = extract_date_and_time(dateobs_stacked)
        # comim = f"{self.path_save}/calib_{self.config.unit}_{self.obj}_{datestr}_{timestr}_{self.filte}_{exptime_stacked:g}.com.fits"
        comim = os.path.join(
            self.config.path.path_stacked, self.config.file.stacked_file
        )
        weightim = comim.replace("com", "weight")

        # ------------------------------------------------------------
        # 	Image Combine
        # ------------------------------------------------------------
        t0_stack = time.time()
        self.logger.debug(f"Total Exptime: {self.total_exptime}")

        gain = (2 / 3) * self.n_stack * self.gain_default
        # 	SWarp
        # swarpcom = f"swarp -c {path_config}/7dt.swarp @{path_imagelist} -IMAGEOUT_NAME {comim} -CENTER_TYPE MANUAL -CENTER {center} -SUBTRACT_BACK N -RESAMPLE_DIR {path_resamp} -GAIN_KEYWORD EGAIN -GAIN_DEFAULT {gain_default} -FSCALE_KEYWORD FAKE -WEIGHTOUT_NAME {weightim}"
        swarpcom = [
            "swarp", f"@{self.path_imagelist}",
            "-c", os.path.join(REF_DIR, '7dt.swarp'),
            "-IMAGEOUT_NAME", f"{comim}",
            "-WEIGHTOUT_NAME", f"{weightim}",
            "-CENTER_TYPE", "MANUAL",
            "-CENTER", f"{self.center} ",
            "-SUBTRACT_BACK", "N",
            "-RESAMPLE_DIR", f"{self.path_resamp}",
            # f"-GAIN_KEYWORD EGAIN -GAIN_DEFAULT {self.gain_default} "
            # f"-FSCALE_KEYWORD FAKE"
        ]  # fmt: skip

        if not self.config.imstack.weight_map:
            swarpcom.extend(["-WEIGHT_TYPE", "NONE"])

        swarpcom = " ".join(swarpcom)
        self.logger.debug(f"SWarp Command: {swarpcom}")
        # os.system(f"{swarpcom} > {self.swarp_log_file} 2>&1")
        os.system(swarpcom)

        # 	Get Select Header Keys from Base Image
        with fits.open(self.header_ref_img) as hdulist:
            header = hdulist[0].header
            new_header = {key: header.get(key, None) for key in self.keywords_to_add}

        # 	Put them into stacked image
        with fits.open(comim) as hdulist:
            header = hdulist[0].header
            for key in new_header.keys():
                header[key] = new_header[key]

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
            "EGAIN": (gain, "Effective EGAIN for combined image (e-/ADU)"),
        }  # fmt: skip

        # 	Header Update
        with fits.open(comim, mode="update") as hdul:
            header = hdul[0].header

            for key, (value, comment) in keywords_to_update.items():
                header[key] = (value, comment)

            # 	Stacked Single images
            for nn, inim in enumerate(self.files):
                header[f"IMG{nn:0>5}"] = (os.path.basename(inim), "")

            hdul.flush()

        delt_stack = time.time() - t0_stack

        self.logger.debug(f"Time to stack {self.n_stack} images: {delt_stack:.3f} sec")


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
        ImStack.from_file(image_list).run()
