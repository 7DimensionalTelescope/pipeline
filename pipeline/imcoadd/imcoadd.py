import os
import time
import shutil
import warnings
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.table import Table
from astropy.coordinates import Angle

from ..const import REF_DIR
from ..errors import PipelineError, CoaddError
from ..config import SciProcConfiguration
from ..path.path import PathHandler
from ..services.setup import BaseSetup
from ..services.utils import acquire_available_gpu
from ..config.utils import get_key
from ..utils import collapse, add_suffix, time_diff_in_seconds, get_basename, atleast_1d, swap_ext
from ..preprocess.utils import get_zdf_from_header_IMCMB
from ..preprocess.plotting import save_fits_as_figures
from .. import external
from ..utils.tile import is_ris_tile, find_ris_tile
from ..utils.header import update_padded_header

from ..services.database.handler import DatabaseHandler
from ..services.database.image_qa import ImageQATable
from ..services.checker import Checker, SanityFilterMixin

from .const import ZP_KEY, CORE_KEYS


warnings.filterwarnings("ignore")


class ImCoadd(BaseSetup, DatabaseHandler, Checker, SanityFilterMixin):
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
        # self._flag_name = "coadd"
        self.logger.process_error = CoaddError

        if self.config_node.settings.is_pipeline:
            self.config_node.imcoadd.convolve = False

        self.qa_id = None
        db_handler = DatabaseHandler.__init__(
            self, add_database=self.config_node.settings.is_pipeline, is_too=self.config_node.settings.is_too
        )

        if self.is_connected:
            self.logger.database = db_handler
            self.process_status_id = self.create_process_data(self.config_node)
            if self.too_id is not None:
                self.logger.debug(f"Initialized DatabaseHandler for ToO data management, ToO ID: {self.too_id}")
            else:
                self.logger.debug(
                    f"Initialized DatabaseHandler for pipeline and QA data management, Pipeline ID: {self.process_status_id}"
                )
            self.update_progress(60, "imcoadd-configured")

    @classmethod
    def from_list(cls, input_images, working_dir=None):
        """use soft link if files are from different directories"""

        for image in input_images:
            if not os.path.exists(image):
                raise FileNotFoundError(f"Input file does not exist: {image}")

        config = SciProcConfiguration.user_config(input_images=input_images, working_dir=working_dir, logger=True)
        return cls(config=config)

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
        #     (10, "coadd_with_swarp", False),
        # ]

    def run(self, use_gpu: bool = False, device_id=None):
        try:
            self._use_gpu = all([use_gpu, self.config_node.imcoadd.gpu, self._use_gpu])

            self.initialize()

            # background subtraction
            self.bkgsub()
            self.update_progress(61, "imcoadd-bkgsub-completed")
            # zero point scaling
            self.zpscale()
            self.update_progress(62, "imcoadd-zpscale-completed")

            if self.config_node.imcoadd.weight_map:
                self.calculate_weight_map(device_id=device_id)
                self.update_progress(63, "imcoadd-calculate-weight-map-completed")

            # replace hot pixels
            if self.config_node.imcoadd.apply_bpmask:
                self.apply_bpmask(device_id=device_id)
                self.update_progress(64, "imcoadd-apply-bpmask-completed")

            # re-registration
            if self.config_node.imcoadd.joint_wcs:
                self.joint_registration()
                self.update_progress(65, "imcoadd-joint-registration-completed")

            # seeing convolution
            if self.config_node.imcoadd.convolve:
                self.prepare_convolution()
                self.run_convolution(device_id=device_id)
                self.update_progress(66, "imcoadd-run-convolution-completed")

            # swarp coaddition
            self.coadd_with_swarp()
            self.update_progress(68, "imcoadd-coadd-with-swarp-completed")

            self.plot_coadd_image()
            self.update_progress(69, "imcoadd-plot-coadded-image-completed")

            if self.is_connected and self.process_status_id is not None:
                coadd_image = self.config_node.imcoadd.coadd_image
                if coadd_image and os.path.exists(coadd_image):
                    self.qa_id = self.create_image_qa_data(coadd_image, process_status_id=self.process_status_id)

            # Update QA data from header if database is connected
            if self.is_connected and self.qa_id is not None:
                coadd_image = self.config_node.imcoadd.coadd_image
                if coadd_image and os.path.exists(coadd_image):
                    qa_data = ImageQATable.from_file(
                        coadd_image,
                        process_status_id=self.process_status_id,
                    )
                    self.image_qa.update_data(qa_data.id, **qa_data.to_dict())

            self.update_progress(70, "imcoadd-completed")

            self.config_node.flag.coadd = True
            self.logger.info(f"'ImCoadd' is Completed in {time_diff_in_seconds(self._st)} seconds")
        except Exception as e:
            self.logger.error(f"Error during imcoadd processing: {str(e)}", CoaddError.UnknownError)

            raise
        # self.logger.debug(MemoryMonitor.log_memory_usage)

    def initialize(self):
        self._st = time.time()
        self.logger.info(f"Start 'ImCoadd'")
        # use common input if imcoadd.input_files override is not set
        local_input_images = get_key(self.config_node.imcoadd, "input_images")
        self.input_images = local_input_images or self.config_node.input.calibrated_images
        self.apply_sanity_filter_and_report()
        self.config_node.imcoadd.input_images = self.input_images
        if not self.input_images:
            raise PipelineError("No Input for ImCoadd")

        self.zpkey = self.config_node.imcoadd.zp_key or ZP_KEY
        # self.ic_keys = IC_KEYS
        self.keys_to_propagate = CORE_KEYS

        # self.define_paths(working_dir=self.config.path.path_processed)
        self.path_tmp = self.path.imcoadd.tmp_dir

        # self.set_metadata()
        self.set_metadata_without_ccdproc()

        # Output coadd image file name
        self.config_node.imcoadd.coadd_image = (
            self.path.imcoadd.daily_coadd_image
            if self.config_node.settings.is_pipeline
            else self.path.imcoadd.coadd_image
        )
        self.config_node.input.coadd_image = self.config_node.imcoadd.coadd_image
        self.logger.debug(f"Coadd Image: {self.config_node.imcoadd.coadd_image}")

        self.logger.info(f"Initialization for ImCoadd is completed")

    def set_metadata_without_ccdproc(self):
        """
        Use OBJCTRA and OBJCTDEC of the first image as the deprojection center.
        The code currently does not check existence and uniqueness of the keys.

        The image with the highest ZP value gives the refernce ZP. This
        is a conservative choice for saturation levels across images.
        """
        self.n_coadd = len(self.input_images)

        header_list = [fits.getheader(f) for f in self.input_images]
        self.header_list = header_list

        self.total_exptime = np.sum([hdr["EXPTIME"] for hdr in header_list])  # sec
        self.zpvalues = [hdr[self.zpkey] for hdr in header_list]
        self.skyvalues = [hdr["SKYVAL"] for hdr in header_list]
        self.mjd_coadd = np.mean([hdr["MJD"] for hdr in header_list])
        # self.satur_level = np.min([hdr["SATURATE"] * hdr["FLXSCALE"] for hdr in header_list])  # this is before FLSCALE def
        # self.coadd_egain = np.sum([hdr["EGAIN"] / hdr["FLXSCALE"] for hdr in header_list])
        self.satur_level_list = [hdr["SATURATE"] for hdr in header_list]
        self.coadd_egain_list = [hdr["EGAIN"] for hdr in header_list]

        objs = list(set([hdr["OBJECT"] for hdr in header_list]))
        filters = list(set([hdr["FILTER"] for hdr in header_list]))
        egains = list(set([hdr["EGAIN"] for hdr in header_list]))
        camera_gains = list(set([hdr["GAIN"] for hdr in header_list]))
        self.obj = objs[0]
        self.camera_gain = camera_gains[0]

        if len(objs) != 1:
            self.logger.warning("Multiple OBJECT found. Using the first one.", CoaddError.AssumptionFailedError)

        self.filte = filters[0]
        if len(filters) != 1:
            self.logger.warning("Multiple FILTER found. Using the first one.", CoaddError.AssumptionFailedError)

        if len(egains) != 1:
            self.logger.warning("Multiple EGAIN found. Using the first one.", CoaddError.AssumptionFailedError)

        if len(camera_gains) != 1:
            self.logger.warning("Multiple GAIN found. Using the first one.", CoaddError.AssumptionFailedError)

        #   Hard coding for the UDS field
        # self.gain_default = 0.78

        self.header_ref_img = self.input_images[0]

        # Determine Deprojection Center
        # 	Tile object (e.g. T01026)
        if is_ris_tile(self.obj):
            self.logger.info(f"Using predefined deprojection center for {self.obj}.")

            skygrid_table = Table.read(os.path.join(REF_DIR, "skygrid.fits"))
            idx_tile = skygrid_table["tile"] == find_ris_tile(self.obj)

            ra = Angle(skygrid_table["ra"][idx_tile][0], unit="deg")
            objra = ra.to_string(unit="hourangle", sep=":", pad=True)

            dec = Angle(skygrid_table["dec"][idx_tile][0], unit="deg")
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

        self.config_node.imcoadd.bkgsub_images = [
            f"{self.path_bkgsub}/{add_suffix(get_basename(f), 'bkgsub')}" for f in self.input_images
        ]

        if self.config_node.imcoadd.bkgsub_type.lower() == "dynamic":
            self.logger.info("Start dynamic background subtraction")
            self._dynamic_bkgsub()
            # if self._bkg_qa():
            #     self._const_bkgsub()
        elif self.config_node.imcoadd.bkgsub_type.lower() == "constant":
            self.logger.info("Start constant background subtraction")
            self._const_bkgsub()
        else:
            raise ValueError(f"bkgsub_type: {self.config_node.imcoadd.bkgsub_type} is invalid")

        self.logger.info(
            f"Background subtraction is completed in {time_diff_in_seconds(st)} ({time_diff_in_seconds(st, return_float=True)/len(self.input_images):.1f} s/image)"
        )

        self.images_to_coadd = self.config_node.imcoadd.bkgsub_images

    def _const_bkgsub(self):
        for ii, (inim, _bkg, outim) in enumerate(
            zip(self.input_images, self.skyvalues, self.config_node.imcoadd.bkgsub_images)
        ):
            self.logger.debug(f"[{ii:>6}] {get_basename(inim)}")
            if os.path.exists(outim) and not self.overwrite:
                self.logger.info(f"Background subtraction result exists; skipping: {get_basename(outim)}")
                continue

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

        self.config_node.imcoadd.bkg_images = bkg_images
        self.config_node.imcoadd.bkg_rms_images = bkg_rms_images

        for inim, outim, bkg, bkg_rms in zip(
            self.input_images, self.config_node.imcoadd.bkgsub_images, bkg_images, bkg_rms_images
        ):

            # if result is already in factory, use that
            if (os.path.exists(outim) and os.path.exists(bkg) and os.path.exists(bkg_rms)) and not self.overwrite:
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

    # TODO:
    def _bkg_qa(self, bkgsub_type: str = "dynamic"):
        if bkgsub_type == "dynamic":
            # do assessment below
            for f in self.config_node.imcoadd.bkg_images:
                data = fits.getdata(f)
                H, W = data.shape
                stripe = np.mean(data[H // 2 - 100 : H // 2 + 100, :], axis=0)

            pass
        elif bkgsub_type == "constant":
            # add dummy key
            for f in self.input_images:
                update_padded_header(f, {"BACKARTF": (False, "Dynamic bkgsub will cause artifacts")})
        else:
            raise ValueError(f"_bkg_qa - Invalid bkgsub_type: {bkgsub_type}")

        update_padded_header(f, {"BACKARTF": (False, "Dynamic bkgsub will cause artifacts")})

        recommenced_bkgsub_type = "constant"  # BACKTYPE "Recommended bkgsub type"
        return recommenced_bkgsub_type

    @staticmethod
    def _group_IMCMB(
        input_images: list[str], output_images: list[str] = None
    ) -> dict[tuple[str, str, str], list[list[str]]]:
        """
        Group images by their master frames (IMCMB).
        Same logic as the preprocessing grouping, but relies on header info
        instead of parsing filename as in NameHandler.get_grouped_files()
        """
        # construct zdf bundles for dict keys
        calibs = []
        for image in input_images:
            calibs.append(get_zdf_from_header_IMCMB(image))

        # make a dict of zdf bundles and their corresponding input and output images
        groups = dict()
        if output_images is not None:
            for input_image, output_image, zdf in zip(input_images, output_images, calibs):
                key = tuple(zdf)
                groups.setdefault(key, [[], []])[0].append(input_image)
                groups[key][1].append(output_image)
        else:
            for input_image, zdf in zip(input_images, calibs):
                key = tuple(zdf)
                groups.setdefault(key, []).append(input_image)

        return groups

    def calculate_weight_map(self, device_id=None, use_gpu: bool = True, overwrite: bool = False):
        """Retains cpu support as a code template"""
        st = time.time()
        self._use_gpu = False  # all([use_gpu, self.config.imcoadd.gpu, self._use_gpu])
        device_id = device_id if self._use_gpu else "CPU"

        self.logger.info(f"Start weight-map calculation")

        bkgsub_images = self.config_node.imcoadd.bkgsub_images
        self.config_node.imcoadd.bkgsub_weight_images = []

        groups = self._group_IMCMB(bkgsub_images)
        self.logger.info(f"{len(groups)} groups for weight map calculation.")
        self.logger.debug(f"calculate_weight_map groups: {groups}")

        for i, ((z_m_file, d_m_file, f_m_file), input_images) in enumerate(groups.items()):
            st_loop = time.time()
            calibs = get_zdf_from_header_IMCMB(input_images[0])  # trust the grouping and use the first image for calibs
            self.logger.debug(f"Group {i} calibs: {calibs}")
            d_m_file, f_m_file, sig_z_file, sig_f_file = PathHandler.weight_map_input(calibs)

            self.logger.debug(f"{time_diff_in_seconds(st_loop)} seconds for group {i} preparation")

            uncalculated_images = []

            for img in input_images:
                weight_image_file = add_suffix(img, "weight")
                self.config_node.imcoadd.bkgsub_weight_images.append(weight_image_file)
                if os.path.exists(weight_image_file) and not self.overwrite:
                    self.logger.debug(f"Already exists; skip generating {weight_image_file}")
                    continue
                else:
                    uncalculated_images.append(img)

            if uncalculated_images:
                st_image = time.time()
                with acquire_available_gpu(device_id=device_id) as device_id:
                    if device_id is None:
                        from .weight import calc_weight_with_cpu

                        calc_weight = calc_weight_with_cpu
                        self.logger.info("Calculate weight map with CPU")
                        device_id = "CPU"
                        calc_weight(uncalculated_images, d_m_file, f_m_file, sig_z_file, sig_f_file)
                    else:
                        from .weight import calc_weight_with_gpu

                        calc_weight = calc_weight_with_gpu
                        self.logger.info(f"Calculate weight map with GPU device {device_id}")
                        calc_weight(
                            uncalculated_images,
                            d_m_file,
                            f_m_file,
                            sig_z_file,
                            sig_f_file,
                            device_id=device_id,
                        )

                self.logger.debug(
                    f"Weight-map calculation (device={device_id}) for group {i} is completed in {time_diff_in_seconds(st_image)} seconds"
                )
            else:
                self.logger.info("All weight images already exist. Skipping weight map calculation")

            self.logger.info(
                f"Weight-map calculation is completed in {time_diff_in_seconds(st)} seconds ({time_diff_in_seconds(st, return_float=True)/len(input_images):.1f} s/image)"
            )

    def _get_bpmask(self, image) -> tuple[str, int]:
        mask_file = PathHandler.get_bpmask(image)
        mask_header = fits.getheader(mask_file, ext=1)
        if "BADPIX" in mask_header.keys():
            badpix = mask_header["BADPIX"]
            self.logger.debug(f"BADPIX found in header. Using badpix {badpix}.")
        else:

            self.logger.warning("BADPIX not found in header. Using default value 0.", CoaddError.KeyError)
        return mask_file, badpix

    def apply_bpmask(self, device_id=None, use_gpu: bool = True):
        st = time.time()

        self._use_gpu = all([use_gpu, self.config_node.imcoadd.gpu, self._use_gpu])
        device_id = device_id if self._use_gpu else "CPU"

        self.logger.info("Start the interpolation for bad pixels")

        path_interp = os.path.join(self.path_tmp, "interp")
        os.makedirs(path_interp, exist_ok=True)

        self.config_node.imcoadd.interp_images = [
            os.path.join(path_interp, add_suffix(get_basename(f), "interp")) for f in self.input_images
        ]

        # bpmask_array, header = fits.getdata(self.config.preprocess.bpmask_file, header=True)
        # ad-hoc. Todo: group input_files for different bpmask files

        method = self.config_node.imcoadd.interp_type
        weight = self.config_node.imcoadd.weight_map  # boolean flag for generating weight map

        # find images that need interpolation
        uncalculated_images = []
        calculated_outputs = []
        for i in range(len(self.config_node.imcoadd.bkgsub_images)):
            input_image_file = self.config_node.imcoadd.bkgsub_images[i]
            output_file = self.config_node.imcoadd.interp_images[i]

            if os.path.exists(output_file) and not self.overwrite:
                self.logger.debug(f"Already exists; skip generating {output_file}")
                continue
            else:
                uncalculated_images.append(input_image_file)
                calculated_outputs.append(output_file)

        # interpolate
        if not uncalculated_images:
            self.logger.info("No images to interpolate. Skipping")
        else:
            groups = self._group_IMCMB(uncalculated_images, calculated_outputs)
            self.logger.info(f"{len(groups)} groups for bad pixel interpolation.")
            self.logger.debug(f"apply_bpmask groups: {groups}")

            for group_id, ((z, d, f), [input_images, output_images]) in enumerate(groups.items()):
                mask_file, badpix = self._get_bpmask(input_images[0])

                with acquire_available_gpu(device_id=device_id) as device_id:
                    if device_id is None:
                        from .interpolate import interpolate_masked_pixels_cpu

                        interpolate_masked_pixels = interpolate_masked_pixels_cpu
                        self.logger.info(f"Interpolate masked pixels with CPU")
                    else:
                        from .interpolate import interpolate_masked_pixels_subprocess

                        interpolate_masked_pixels = interpolate_masked_pixels_subprocess
                        self.logger.info(f"Interpolate masked pixels with GPU device {device_id}")

                    interpolate_masked_pixels(
                        input_images,
                        mask_file,
                        output_images,
                        method=method,
                        badpix=badpix,
                        weight=weight,
                        device=device_id,
                    )
                self.logger.debug(f"[group {group_id}] Interpolation for bad pixels is completed")

            self.logger.info(
                f"Interpolation for bad pixels is completed in {time_diff_in_seconds(st)} seconds "
                f"({time_diff_in_seconds(st, return_float=True)/len(self.images_to_coadd):.1f} s/image)"
            )

        # advance the target images of interest
        self.images_to_coadd = self.config_node.imcoadd.interp_images

    def zpscale(self):
        """
        Store the value in header as FLXSCALE, and use it in coadding.
        Keep FSCALE_KEYWORD = FLXSCALE in SWarp config.
        The headers of the last processed images are modified.
        """
        st = time.time()
        self.flxscale_list = []
        for file, zp in zip(self.images_to_coadd, self.zpvalues):
            flxscale = 10 ** (0.4 * (self.zp_base - zp))
            self.flxscale_list.append(flxscale)
            with fits.open(file, mode="update") as hdul:
                hdul[0].header["FLXSCALE"] = (
                    flxscale,
                    "flux scaling factor by 7DT Pipeline (ImCoadd)",
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
        #     zip(self.config.imcoadd.bkgsub_files, self.zpvalues)
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
        Just give the new joint WCS to all images and let individual ImCoadd
        handle the rest of the process.
        """
        pass

    def prepare_convolution(self):
        """
        This is ad-hoc. Change it to convolve after resampling and take
        advantage of uniform pixel scale.
        """

        method = self.config_node.imcoadd.convolve.lower()
        self.conv_method = method
        self.logger.info(f"Prepare the convolution with {method} method")

        self.kernels = []

        if method == "gaussian":
            from ..utils import force_symlink

            # Define output path
            path_conv = os.path.join(self.path_tmp, "conv")
            os.makedirs(path_conv, exist_ok=True)
            self.config_node.imcoadd.conv_files = [
                os.path.join(path_conv, add_suffix(get_basename(f), "conv")) for f in self.input_images
            ]

            # Get peeings for convolution
            peeings = [fits.getval(inim, "PEEING") for inim in self.images_to_coadd]
            # max_peeing = np.max(peeings)
            self._max_peeing = (
                self.config_node.imcoadd.target_seeing / collapse(self.path.pixscale, raise_error=True)
                if hasattr(self.config_node.imcoadd, "target_seeing")
                and isinstance(self.config_node.imcoadd.target_seeing, (int, float))
                else np.max(peeings)
            )
            delta_peeings = [self._calc_delta_peeing(peeing) for peeing in peeings]
            self.delta_peeings = delta_peeings
            self.logger.debug(f"PEEINGs: {peeings}")

            for i, delta_peeing in enumerate(delta_peeings):
                # symlink images to conv output folder that don't need convolution
                if delta_peeing is None:
                    force_symlink(self.images_to_coadd[i], self.config_node.imcoadd.conv_files[i])
                    if self.config_node.imcoadd.weight_map:
                        force_symlink(
                            add_suffix(self.images_to_coadd[i], "weight"),
                            add_suffix(self.config_node.imcoadd.conv_files[i], "weight"),
                        )
                    self.kernels.append(None)
                else:
                    self.kernels.append(delta_peeing)  # 8*sig + 1 sized

        else:
            self.logger.info("Undefined convolution method. Skipping seeing match")

    def run_convolution(self, device_id=None, use_gpu: bool = True, weight=False):
        # from .convolve import convolve_fft, get_edge_mask

        st = time.time()
        method = self.conv_method
        self._use_gpu = all([use_gpu, self.config_node.imcoadd.gpu, self._use_gpu])
        device_id = device_id if self._use_gpu else "CPU"

        # compute
        kernels = [k for k in self.kernels if k is not None]
        image_list = [f for f, k in zip(self.images_to_coadd, self.kernels) if k is not None]
        outim_list = [f for f, k in zip(self.config_node.imcoadd.conv_files, self.kernels) if k is not None]
        delta_peeing_list = [v for v, k in zip(self.delta_peeings, self.kernels) if k is not None]

        with acquire_available_gpu(device_id=device_id) as device_id:

            if device_id is None:
                from .convolve import convolve_fft_cpu

                convolve_fft = convolve_fft_cpu
                self.logger.info(f"Convolution with CPU")
            else:

                from .convolve import convolve_fft_subprocess

                convolve_fft = convolve_fft_subprocess
                self.logger.info(f"Convolution with GPU device {device_id}")

            output = convolve_fft(
                image_list,
                outim_list,
                kernels=kernels,
                device=device_id,
                apply_edge_mask=weight,
                method=method,
                delta_peeing=delta_peeing_list,
            )

            if weight:
                weight_list = [
                    f for f, k in zip(PathHandler(self.images_to_coadd).weight, self.kernels) if k is not None
                ]
                outwim_list = [f for f, k in zip(PathHandler(self.config_node.imcoadd.conv_files).weight, self.kernels) if k is not None]  # fmt:skip
                self.logger.debug(f"weight_list {weight_list}")
                self.logger.debug(f"outwim_list {outwim_list}")

                if not all([os.path.exists(f) for f in atleast_1d(weight_list)]):
                    self.logger.error(f"Weight map not found for all images.", CoaddError.FileNotFoundError)
                    raise CoaddError.FileNotFoundError(f"Weight map not found for all images.")

                # compute
                convolve_fft(
                    weight_list,
                    outwim_list,
                    kernels=kernels,
                    device=device_id,
                    apply_edge_mask=weight,
                    method=method,
                    delta_peeing=delta_peeing_list,
                )

        self.logger.info(
            f"Convolution is completed in {time_diff_in_seconds(st)} seconds ({time_diff_in_seconds(st, return_float=True)/len(self.images_to_coadd):.1f} s/image)"
        )

    def _calc_delta_peeing(self, peeing):
        delta_peeing = np.sqrt(self._max_peeing**2 - peeing**2)
        if delta_peeing == 0:
            self.logger.debug(f"Skipping calculating delta peeing.")
            return None
        else:
            return delta_peeing

    def coadd_with_swarp(self):
        st = time.time()
        self.logger.info("Start to run swarp for coadding images")

        # Write target images to a text file
        self.path_imagelist = os.path.join(self.path_tmp, "images_to_coadd.txt")

        with open(self.path_imagelist, "w") as f:
            for inim in self.images_to_coadd:  # self.zpscaled_images:
                f.write(f"{inim}\n")

        self.logger.debug(f"Total Exptime: {self.total_exptime}")

        if not self.config_node.imcoadd.weight_map:
            # if no weight map, just coadd the images
            self._run_swarp("")

        else:
            # science images
            self._run_swarp("sci", args=["-RESAMPLING_TYPE", "LANCZOS3"])

            # weight images
            self._run_swarp("wht", args=["-RESAMPLING_TYPE", "NEAREST"])

            # Update/Todo: consider uncollapsed bpmask files
            if self.config_node.imcoadd.propagate_mask:
                # bpmask_file = self.config.preprocess.bpmask_file
                bpmask_file = PathHandler.get_bpmask(self.images_to_coadd)
                bpmask_inverted = 1 - fits.getdata(bpmask_file)
                bpmask_inverted_file = os.path.join(self.path_tmp, get_basename(bpmask_file))
                fits.writeto(bpmask_inverted_file, bpmask_inverted, overwrite=True)
                self.logger.debug(f"Inverted bpmask saved as {bpmask_inverted_file}")
                args = ["-WEIGHT_IMAGE", bpmask_file, "-RESAMPLING_TYPE", "LANCZOS3"]
                self._run_swarp("bpm", args=args)

        self._update_header()

        self.logger.info(f"Running swarp is completed in {time_diff_in_seconds(st)} seconds")

    def _run_swarp(self, type="", args=None):
        """Pass type='' for no weight"""
        working_dir = os.path.join(self.path_tmp, type)
        resample_dir = os.path.join(working_dir, "resamp")
        log_file = os.path.join(working_dir, "_".join([self.config_node.name, type, "swarp.log"]))

        if type == "":
            output_file = self.config_node.imcoadd.coadd_image  # output to output_dir directly
        else:
            # output to tmp dir, and then selectively move to output_dir
            output_file = os.path.join(working_dir, get_basename(self.config_node.imcoadd.coadd_image))

        external.swarp(
            input=self.path_imagelist,
            output=output_file,
            overwrite=self.overwrite,
            center=self.center,
            resample_dir=resample_dir,
            log_file=log_file,
            logger=self.logger,
            weight_map=self.config_node.imcoadd.weight_map,
            swarp_args=args,
        )

        # move files to the final directory
        if type == "sci":
            shutil.move(output_file, self.config_node.imcoadd.coadd_image)
        elif type == "wht":
            shutil.move(
                add_suffix(output_file, "weight"),
                add_suffix(self.config_node.imcoadd.coadd_image, "weight"),
            )
        elif type == "bpm":
            shutil.move(
                add_suffix(output_file, "weight"),
                add_suffix(self.config_node.imcoadd.coadd_image, "bpmask"),
            )

        return

    def coadd_with_cupy(self):
        pass

    def plot_coadd_image(self):
        coadd_img = self.config_node.imcoadd.coadd_image
        basename = os.path.basename(coadd_img)
        save_fits_as_figures(fits.getdata(coadd_img), self.path.figure_dir_to_path / swap_ext(basename, "png"))
        return

    def _update_header(self):
        # 	Get Header info
        exptime_coadd = self.total_exptime
        mjd_coadd = self.mjd_coadd
        jd_coadd = Time(mjd_coadd, format="mjd").jd
        dateobs_coadd = Time(mjd_coadd, format="mjd").isot
        # gain = (2 / 3) * self.n_coadd * self.egain
        # airmass_coadd = np.mean(airmasslist)
        # dateloc_coadd = calc_mean_dateloc(dateloclist)
        # alt_coadd = np.mean(altlist)
        # az_coadd = np.mean(azlist)
        coadd_satur_level = np.min(
            [satur * flxscale for satur, flxscale in zip(self.satur_level_list, self.flxscale_list)]
        )
        coadd_egain = np.sum([egain / flxscale for egain, flxscale in zip(self.coadd_egain_list, self.flxscale_list)])

        # datestr, timestr = extract_date_and_time(dateobs_coadd)
        # comim = f"{self.path_save}/calib_{self.config.unit}_{self.obj}_{datestr}_{timestr}_{self.filte}_{exptime_coadd:g}.com.fits"

        # 	Get Select Header Keys from Base Image
        with fits.open(self.header_ref_img) as hdulist:
            header = hdulist[0].header
            select_header_dict = {key: header.get(key, None) for key in self.keys_to_propagate}

        # 	Put them into coadded image
        with fits.open(self.config_node.imcoadd.coadd_image) as hdulist:
            header = hdulist[0].header
            for key in select_header_dict.keys():
                header[key] = select_header_dict[key]

        # 	Additional Header Information
        keywords_to_update = {
            "DATE-OBS": (dateobs_coadd, "Time of observation (UTC) for coadded image"),
            # 'DATE-LOC': (dateloc_coadd, 'Time of observation (local) for coadded image'),
            "EXPTIME": (exptime_coadd, "[s] Total exposure duration for coadded image"),
            "EXPOSURE": (exptime_coadd, "[s] Total exposure duration for coadded image"),
            # 'CENTALT' : (alt_coadd,     '[deg] Average altitude of telescope for coadded image'),
            # 'CENTAZ'  : (az_coadd,      '[deg] Average azimuth of telescope for coadded image'),
            # 'AIRMASS' : (airmass_coadd, 'Average airmass at frame center for coadded image (Gueymard 1993)'),
            "MJD": (mjd_coadd, "Modified Julian Date at start of observations for coadded image"),
            "JD": (jd_coadd, "Julian Date at start of observations for coadded image"),
            "SKYVAL": (0, "SKY MEDIAN VALUE (Subtracted)"),
            "EGAIN": (coadd_egain, "Effective EGAIN for coadded image (e-/ADU)"),  # swarp calculates it as GAIN, but irreproducible.
            "GAIN": (self.camera_gain, "Gain from the camera configuration"),
            "SATURATE": (coadd_satur_level, "Conservative saturation level for coadded image"),  # let swarp handle this
        }  # fmt: skip

        # 	Update Header
        with fits.open(self.config_node.imcoadd.coadd_image, mode="update") as hdul:
            header = hdul[0].header

            for key, (value, comment) in keywords_to_update.items():
                header[key] = (value, comment)

            # 	Names of coadded single images
            for nn, inim in enumerate(self.input_images):
                header[f"IMG{nn:0>5}"] = (get_basename(inim), "single exposures")

            hdul.flush()
