import os
import time
import shutil
import warnings
from typing import Literal

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from ..const import REF_DIR
from ..const.sciproc import COADD_SPEC, SCIPROCESS_REGISTRY
from ..errors import CoaddError
from ..config import SciProcConfiguration
from ..path.path import PathHandler
from ..services.setup import BaseSetup
from ..services.utils import acquire_available_gpu
from ..config.utils import get_key
from ..utils import collapse, add_suffix, time_diff_in_seconds, get_basename, atleast_1d, swap_ext
from ..preprocess.utils import get_zdf_from_header_IMCMB
from ..preprocess.plotting import save_fits_as_figures
from .. import external
from ..utils.header import update_padded_header

from ..services.database.handler import DatabaseHandler
from ..services.database.image_qa import ImageQATable
from ..services.checker import Checker
from ..services.version_check import RuntimeVersionMixin

from .const import ZP_KEY
from .header_set import InputHeaderSet
from .calc import mean_coadd_numpy, median_coadd_numpy


warnings.filterwarnings("ignore")


class ImCoadd(BaseSetup, DatabaseHandler, Checker, RuntimeVersionMixin):
    _process_spec = COADD_SPEC
    zp_base: float = 23.9  # uJy; flux-scaling reference zero point

    def __init__(
        self,
        config=None,
        logger=None,
        queue=None,
        overwrite=False,
        use_gpu: bool = True,
    ) -> None:

        super().__init__(config, logger, queue)
        self.overwrite = self.resolve_overwrite(overwrite)
        self._device_id = None
        self._use_gpu = use_gpu
        self.logger.process_error = CoaddError

        if self.config_node.settings.is_pipeline:
            self.config_node.imcoadd.convolve = False

        self.qa_id = None
        DatabaseHandler.__init__(
            self, use_database=self.config_node.settings.is_pipeline, is_too=self.config_node.settings.is_too
        )

        if self.is_connected:

            self.reset_exceptions("coadd")

            if self.process_status_id is not None:
                from ..services.database.handler import ExceptionHandler

                self.logger.database = ExceptionHandler(self.process_status_id)

            self.process_status_id = self.create_process_data(self.config_node)
            if self.too_id is not None:
                self.logger.debug(f"Initialized DatabaseHandler for ToO data management, ToO ID: {self.too_id}")
            else:
                self.logger.debug(
                    f"Initialized DatabaseHandler for pipeline and QA data management, Pipeline ID: {self.process_status_id}"
                )
            self.update_progress(SCIPROCESS_REGISTRY.configured_progress("coadd"), "imcoadd-configured")

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

    def legacy_coadd_routine(self, use_gpu: bool = False, device_id=None):
        self._use_gpu = all([use_gpu, self.config_node.imcoadd.gpu, self._use_gpu])

        self.initialize()

        # background subtraction
        images = self.bkgsub(self.input_images)
        self.update_progress(SCIPROCESS_REGISTRY.milestone_progress("coadd", "bkgsub"), "imcoadd-bkgsub-completed")
        # zero point scaling
        if get_key(self.config_node.imcoadd, "zpscale", default=True):
            self.zpscale(images)
        self.update_progress(SCIPROCESS_REGISTRY.milestone_progress("coadd", "zpscale"), "imcoadd-zpscale-completed")

        if self.config_node.imcoadd.weight_map:
            self.calculate_weight_map(images, device_id=device_id)
            self.update_progress(
                SCIPROCESS_REGISTRY.milestone_progress("coadd", "calculate_weight_map"),
                "imcoadd-calculate-weight-map-completed",
            )

        # replace hot pixels
        if self.config_node.imcoadd.apply_bpmask:
            images = self.apply_bpmask(images, device_id=device_id)
            self.update_progress(
                SCIPROCESS_REGISTRY.milestone_progress("coadd", "apply_bpmask"), "imcoadd-apply-bpmask-completed"
            )

        # re-registration
        if self.config_node.imcoadd.joint_wcs:
            images = self.joint_registration(images)
            self.update_progress(
                SCIPROCESS_REGISTRY.milestone_progress("coadd", "joint_registration"),
                "imcoadd-joint-registration-completed",
            )

        # seeing convolution
        if self.config_node.imcoadd.convolve:
            self.prepare_convolution(images)
            images = self.run_convolution(images, device_id=device_id)
            self.update_progress(
                SCIPROCESS_REGISTRY.milestone_progress("coadd", "run_convolution"),
                "imcoadd-run-convolution-completed",
            )

        # swarp coaddition
        self.reproject_and_coadd_with_swarp(images, coadd=True)
        self.update_progress(
            SCIPROCESS_REGISTRY.milestone_progress("coadd", "coadd_with_swarp"),
            "imcoadd-coadd-with-swarp-completed",
        )

        self.plot_coadd_image()
        self.update_progress(
            SCIPROCESS_REGISTRY.milestone_progress("coadd", "plot_coadd_image"),
            "imcoadd-plot-coadded-image-completed",
        )

        if self.is_connected and self.process_status_id is not None:
            coadd_image = self.config_node.imcoadd.coadd_image
            if coadd_image and os.path.exists(coadd_image):
                self.qa_id = self.create_image_qa_data(coadd_image, process_status_id=self.process_status_id)
                self.create_image_qa_dependencies(coadd_image, self.qa_id)

        # Update QA data from header if database is connected
        if self.is_connected and self.qa_id is not None:
            coadd_image = self.config_node.imcoadd.coadd_image
            if coadd_image and os.path.exists(coadd_image):
                qa_data = ImageQATable.from_file(
                    coadd_image,
                    process_status_id=self.process_status_id,
                )
                self.image_qa.update_data(qa_data.id, **qa_data.to_dict())

        self.update_progress(SCIPROCESS_REGISTRY.completed_progress("coadd"), "imcoadd-completed")

    def reproject_first_coadd_routine(self, use_gpu: bool = False, device_id=None):
        """SWarp for reprojection only; the combine step runs in memory via
        ``coadd_in_memory``, which picks mean/median from ``imcoadd.coadd_mode``."""
        self._use_gpu = all([use_gpu, self.config_node.imcoadd.gpu, self._use_gpu])

        optional_steps = (
            int(bool(self.config_node.imcoadd.weight_map))
            + int(bool(self.config_node.imcoadd.apply_bpmask))
            + int(bool(self.config_node.imcoadd.joint_wcs))
            + int(bool(self.config_node.imcoadd.convolve))
        )
        TOTAL_STEPS = 4 + optional_steps  # reproject + bkgsub + zpscale + coadd + optionals
        step = 0

        self.initialize()

        images = self.input_images
        if self.config_node.imcoadd.weight_map:
            self.calculate_weight_map(images, device_id=device_id)
            step += 1
            self.update_progress(
                SCIPROCESS_REGISTRY.step_progress("coadd", step, TOTAL_STEPS),
                "imcoadd-calculate-weight-map-completed",
            )

        # Bad pixel interpolation
        if self.config_node.imcoadd.apply_bpmask:
            images = self.apply_bpmask(images, device_id=device_id)
            step += 1
            self.update_progress(
                SCIPROCESS_REGISTRY.step_progress("coadd", step, TOTAL_STEPS), "imcoadd-apply-bpmask-completed"
            )

        # Optional joint registration
        if self.config_node.imcoadd.joint_wcs:
            images = self.joint_registration(images)
            step += 1
            self.update_progress(
                SCIPROCESS_REGISTRY.step_progress("coadd", step, TOTAL_STEPS),
                "imcoadd-joint-registration-completed",
            )

        # Reproject (no combine) onto a common WCS
        images = self.reproject_and_coadd_with_swarp(images, coadd=False)
        step += 1
        self.update_progress(
            SCIPROCESS_REGISTRY.step_progress("coadd", step, TOTAL_STEPS), "imcoadd-reproject-completed"
        )

        # PSF homogenization
        if self.config_node.imcoadd.convolve:
            self.prepare_convolution(images)
            images = self.run_convolution(images, device_id=device_id)
            step += 1
            self.update_progress(
                SCIPROCESS_REGISTRY.step_progress("coadd", step, TOTAL_STEPS),
                "imcoadd-run-convolution-completed",
            )

        # Background subtraction on reprojected (+ optionally convolved) frames
        images = self.bkgsub(images)
        step += 1
        self.update_progress(SCIPROCESS_REGISTRY.step_progress("coadd", step, TOTAL_STEPS), "imcoadd-bkgsub-completed")

        # Flux zero-point scaling (writes FLXSCALE to header in place)
        self.zpscale(images)
        step += 1
        self.update_progress(SCIPROCESS_REGISTRY.step_progress("coadd", step, TOTAL_STEPS), "imcoadd-zpscale-completed")

        # In-memory coaddition (mean/median selected by imcoadd.coadd_mode)
        self.coadd_in_memory(images, device_id=device_id)
        step += 1
        self.update_progress(SCIPROCESS_REGISTRY.step_progress("coadd", step, TOTAL_STEPS), "imcoadd-coadd-completed")

        # Plot coadd image
        self.plot_coadd_image()
        step += 1
        self.update_progress(SCIPROCESS_REGISTRY.step_progress("coadd", step, TOTAL_STEPS), "imcoadd-plot-completed")

        # Update QA data from header if database is connected
        if self.is_connected and self.qa_id is not None:
            coadd_image = self.config_node.imcoadd.coadd_image
            if coadd_image and os.path.exists(coadd_image):
                qa_data = ImageQATable.from_file(
                    coadd_image,
                    process_status_id=self.process_status_id,
                )
                self.image_qa.update_data(qa_data.id, **qa_data.to_dict())

        self.update_progress(SCIPROCESS_REGISTRY.completed_progress("coadd"), "imcoadd-completed")

    def run(self, use_gpu: bool = False, device_id=None):
        try:
            routine = self.config_node.imcoadd.coadd_routine
            if "legacy" in routine.lower():
                self.legacy_coadd_routine(use_gpu=use_gpu, device_id=device_id)
            elif "reproject-first" in routine.lower():
                self.reproject_first_coadd_routine(use_gpu=use_gpu, device_id=device_id)
            else:
                raise ValueError(f"Invalid coadd routine: {routine!r} (expected 'legacy' or 'reproject-first')")

            self.config_node.flag.coadd = True
            self.logger.info(f"'ImCoadd' is Completed in {time_diff_in_seconds(self._st)} seconds")
        except Exception as e:
            self.logger.error(f"Error during imcoadd processing: {str(e)}", e, exc_info=True)

            raise
        # self.logger.debug(MemoryMonitor.log_memory_usage)

    def initialize(self):
        self._st = time.time()
        self.logger.info(f"Start 'ImCoadd'")
        # use common input if imcoadd.input_files override is not set
        local_input_images = get_key(self.config_node.imcoadd, "input_images")
        self.input_images = (
            local_input_images
            if local_input_images is not None  # local_input_images can be an empty list
            else self.config_node.input.calibrated_images
        )
        self.apply_sanity_filter_and_report(current_process=COADD_SPEC, overwrite=self.overwrite)
        if not self.input_images:
            self.logger.error("No Input for ImCoadd", CoaddError.EmptyInputAfterSanityRejection)
            raise CoaddError.EmptyInputAfterSanityRejection("No Input for ImCoadd")
        # if rejected, let the input remain so that a rerun has a change to reevaluate SANITY
        self._recreate_pathhandler_instance()  # resync
        self.config_node.imcoadd.input_images = self.input_images

        self.zpkey = self.config_node.imcoadd.zp_key or ZP_KEY
        # self.ic_keys = IC_KEYS

        # self.define_paths(working_dir=self.config.path.path_processed)

        # Single read of every input header; all aggregates/coadd_header live on this snapshot.
        self.input_headers = InputHeaderSet.from_files(self.input_images)
        self.input_headers.check_uniqueness(["OBJECT", "FILTER", "EGAIN", "GAIN"], self.logger)
        self.center = self.input_headers.deprojection_center
        self.logger.debug(f"Deprojection center: {self.center}")

        # Output coadd image file name
        self.config_node.imcoadd.coadd_image = self.path.imcoadd.coadd_image
        self.config_node.input.coadd_image = self.config_node.imcoadd.coadd_image
        self.logger.debug(f"Coadd Image: {self.config_node.imcoadd.coadd_image}")

        self.logger.info(f"Initialization for ImCoadd is completed")

    def bkgsub(
        self,
        input_images: list[str] | None = None,
        ignore_steppy_flag: bool = False,
        skyval_cut: float = 40,
    ) -> list[str]:
        # ------------------------------------------------------------
        # 	Global Background Subtraction
        # ------------------------------------------------------------
        if input_images is None:
            input_images = self.input_images
        st = time.time()

        self.path_bkgsub = self.path.imcoadd.factory.bkgsub_dir

        bkgsub_images = atleast_1d(self.path.imcoadd.factory.bkgsub_images)
        self.config_node.imcoadd.bkgsub_images = bkgsub_images

        bkg_images = atleast_1d(self.path.imcoadd.factory.bkg_images)
        bkg_rms_images = atleast_1d(self.path.imcoadd.factory.bkg_rms_images)

        skyvalues = self.input_headers.values("SKYVAL")
        # TODO: ad-hoc; later derive is_steppy from actual data check
        if any([skyval < skyval_cut for skyval in skyvalues]):
            self.config_node.imcoadd.bkgsub_type = "constant"
        else:
            self.config_node.imcoadd.bkgsub_type = "dynamic"
        # Stamp BACKTYPE onto the in-memory snapshot so coadd_header picks it up
        backtype_upper = self.config_node.imcoadd.bkgsub_type.upper()
        for hdr in self.input_headers:
            hdr["BACKTYPE"] = (backtype_upper, "Background subtraction type")

        if self.config_node.imcoadd.bkgsub_type.lower() == "dynamic":
            self.logger.info("Start dynamic background subtraction")
            bkg_func = self._dynamic_bkgsub
            self.config_node.imcoadd.bkg_images = bkg_images
            self.config_node.imcoadd.bkg_rms_images = bkg_rms_images

        elif self.config_node.imcoadd.bkgsub_type.lower() == "constant":
            self.logger.info("Start constant background subtraction")
            bkg_func = self._const_bkgsub
            if get_key(self.config_node.imcoadd, "bkg_images"):
                self.config_node.imcoadd.bkg_images = None
            if get_key(self.config_node.imcoadd, "bkg_rms_images"):
                self.config_node.imcoadd.bkg_rms_images = None

        else:
            raise ValueError(f"bkgsub_type: {self.config_node.imcoadd.bkgsub_type} is invalid")

        for i, (inim, outim, bkg, bkg_rms, skyvalue) in enumerate(
            zip(input_images, bkgsub_images, bkg_images, bkg_rms_images, skyvalues)
        ):
            st_loop = time.time()
            is_steppy = bkg_func(
                inim, outim, bkg=bkg, bkg_rms=bkg_rms, skyval=skyvalue, ignore_steppy_flag=ignore_steppy_flag
            )

            # if is_steppy and not ignore_steppy_flag:
            #     self.logger.warning(f"Background subtraction failed for {get_basename(outim)}")
            #     self.logger.warning(f"Re-running background subtraction with constant value")
            #     self._const_bkgsub(inim, outim, skyval=skyvalue)

            self.logger.info(
                f"Background subtraction completed for {get_basename(outim)} [image {i+1}/{len(input_images)}] in {time_diff_in_seconds(st_loop)} seconds"
            )

        self.logger.info(
            f"Background subtraction is completed in {time_diff_in_seconds(st)} ({time_diff_in_seconds(st, return_float=True)/len(input_images):.1f} s/image)"
        )

        self.images_to_coadd = bkgsub_images
        return bkgsub_images

    def _const_bkgsub(self, inim, outim, skyval, skyval_cut=40, **kwargs):

        if os.path.exists(outim):
            if fits.getval(outim, "BACKTYPE", default="").upper() == "CONSTANT":
                if not self.overwrite:
                    self.logger.info(f"Background subtraction result exists; skipping: {get_basename(outim)}")
                    return

        is_steppy = skyval < skyval_cut

        with fits.open(inim, memmap=True) as hdul:
            _data = hdul[0].data
            _hdr = hdul[0].header
            _hdr["BACKTYPE"] = ("CONSTANT", "Background subtraction type")
            # _hdr["BKG_STEP"] = (is_steppy, "SE Background can be step-like")
            _data -= skyval
            self.logger.debug(f"Using SKYVAL: {skyval:.3f}")
            fits.writeto(outim, _data, header=_hdr, overwrite=True)

        return False  # is_steppy is False by definition for constant background subtraction

    def _dynamic_bkgsub(self, inim, outim, bkg, bkg_rms, ignore_steppy_flag=False, **kwargs):
        """
        Later to be refined using iterations
        """
        from ..external import sextractor

        # from .bkg_step import step_background_check

        sex_options = {
            "-CATALOG_TYPE": "NONE",  # save no source catalog
            "-CHECKIMAGE_TYPE": "BACKGROUND,BACKGROUND_RMS",
            "-CHECKIMAGE_NAME": f"{bkg},{bkg_rms}",
        }
        sex_log = os.path.join(
            self.path_bkgsub,
            os.path.splitext(get_basename(outim))[0] + "_sextractor.log",
        )
        sextractor(inim, sex_options=sex_options, log_file=sex_log, logger=self.logger)

        bkg_data = fits.getdata(bkg)

        # if ignore_steppy_flag:
        #     is_steppy = False
        # else:
        #     h, w = bkg_data.shape
        #     stripe = np.mean(bkg_data[h // 2 - 100 : h // 2 + 100, :], axis=0)  # already smooth bkg: mean is okay?
        #     is_steppy, info = step_background_check(stripe)
        #     if is_steppy:
        #         self.logger.warning(f"Background is steppy in {get_basename(outim)}")
        #         self.logger.debug(f"Background is steppy: {info}")
        #         return True
        #     else:
        #         self.logger.debug(f"Background is not steppy in {get_basename(outim)}: {info}")

        with fits.open(inim, memmap=True) as hdul:
            _data = hdul[0].data
            _hdr = hdul[0].header
            _hdr["BACKTYPE"] = ("DYNAMIC", "Background subtraction type")
            # _hdr["BKG_STEP"] = (is_steppy, "Background is step-like; likely quantization artifact")
            _data -= bkg_data
            fits.writeto(outim, _data, header=_hdr, overwrite=True)

        # return is_steppy

    # # TODO:
    # def _bkg_qa(self, bkgsub_type: str = "dynamic"):
    #     if bkgsub_type == "dynamic":
    #         # do assessment below
    #         for f in self.config_node.imcoadd.bkg_images:
    #             data = fits.getdata(f)
    #             H, W = data.shape
    #             stripe = np.mean(data[H // 2 - 100 : H // 2 + 100, :], axis=0)

    #         pass
    #     elif bkgsub_type == "constant":
    #         # add dummy key
    #         for f in self.input_images:
    #             update_padded_header(f, {"BACKARTF": (False, "Dynamic bkgsub will cause artifacts")})
    #     else:
    #         raise ValueError(f"_bkg_qa - Invalid bkgsub_type: {bkgsub_type}")

    #     update_padded_header(f, {"BACKARTF": (False, "Dynamic bkgsub will cause artifacts")})

    #     recommenced_bkgsub_type = "constant"  # BACKTYPE "Recommended bkgsub type"
    #     return recommenced_bkgsub_type

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

    def calculate_weight_map(
        self,
        input_images: list[str] | None = None,
        device_id=None,
        use_gpu: bool = True,
        overwrite: bool = False,
    ) -> list[str]:
        """
        Uses self.input_images. Takes in input_images just for name carrying.
        """
        if input_images is None:
            input_images = get_key(self.config_node.imcoadd, "bkgsub_images") or self.input_images

        value_images = self.input_images  # r_p. input_images for name carrying

        st = time.time()
        self._use_gpu = False  # all([use_gpu, self.config.imcoadd.gpu, self._use_gpu])
        device_id = device_id if self._use_gpu else "CPU"

        self.logger.info(f"Start weight-map calculation")

        out_weights = atleast_1d(add_suffix(input_images, "weight"))
        self.config_node.imcoadd.bkgsub_weight_images = out_weights

        groups = self._group_IMCMB(value_images, out_weights)
        self.logger.info(f"{len(groups)} groups for weight map calculation.")
        self.logger.debug(f"calculate_weight_map groups: {groups}")

        for i, ((z_m_file, d_m_file, f_m_file), (group_values, group_outputs)) in enumerate(groups.items()):
            st_loop = time.time()
            self.logger.debug(f"IMCMB group {i}: {z_m_file}, {d_m_file}, {f_m_file}")
            # calibs = get_zdf_from_header_IMCMB(input_images[0])  # trust the grouping and use the first image for calibs
            calibs = [z_m_file, d_m_file, f_m_file]
            self.logger.debug(f"Group {i} calibs: {calibs}")
            d_m_file, f_m_file, sig_z_file, sig_f_file = PathHandler.resolve_weight_map_input_abspath(calibs)

            self.logger.debug(f"{time_diff_in_seconds(st_loop)} seconds for group {i} preparation")

            uncalculated_images = []
            uncalculated_outputs = []

            for vimg, oname in zip(group_values, group_outputs):
                if os.path.exists(oname) and not self.overwrite:
                    self.logger.debug(f"Already exists; skip generating {oname}")
                    continue
                else:
                    uncalculated_images.append(vimg)
                    uncalculated_outputs.append(oname)

            if uncalculated_images:
                st_image = time.time()
                with acquire_available_gpu(device_id=device_id) as device_id:
                    if device_id is None:
                        from .weight import calc_weight_with_cpu

                        calc_weight = calc_weight_with_cpu
                        self.logger.info("Calculate weight map with CPU")
                        device_id = "CPU"
                        calc_weight(
                            uncalculated_images,
                            d_m_file,
                            f_m_file,
                            sig_z_file,
                            sig_f_file,
                            out_names=uncalculated_outputs,
                        )
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
                            out_names=uncalculated_outputs,
                        )

                self.logger.debug(
                    f"Weight-map calculation (device={device_id}) for group {i} is completed in {time_diff_in_seconds(st_image)} seconds"
                )
            else:
                self.logger.info("All weight images already exist. Skipping weight map calculation")

            self.logger.info(
                f"Weight-map calculation is completed in {time_diff_in_seconds(st)} seconds ({time_diff_in_seconds(st, return_float=True)/len(group_values):.1f} s/image)"
            )

        return self.config_node.imcoadd.bkgsub_weight_images

    def _get_bpmask(self, image) -> tuple[str, int]:
        mask_file = PathHandler.get_bpmask(image)
        mask_header = fits.getheader(mask_file, ext=1)
        if "BADPIX" in mask_header.keys():
            badpix = mask_header["BADPIX"]
            self.logger.debug(f"BADPIX found in header. Using badpix {badpix}.")
        else:

            self.logger.warning("BADPIX not found in header. Using default value 0.", CoaddError.KeyError)
        return mask_file, badpix

    def apply_bpmask(
        self,
        input_images: list[str] | None = None,
        device_id=None,
        use_gpu: bool = True,
    ) -> list[str]:
        if input_images is None:
            input_images = get_key(self.config_node.imcoadd, "bkgsub_images") or self.input_images
        st = time.time()

        self._use_gpu = all([use_gpu, self.config_node.imcoadd.gpu, self._use_gpu])
        device_id = device_id if self._use_gpu else "CPU"

        self.logger.info("Start the interpolation for bad pixels")

        interp_images = atleast_1d(self.path.imcoadd.factory.interp_images)
        self.config_node.imcoadd.interp_images = interp_images

        # bpmask_array, header = fits.getdata(self.config.preprocess.bpmask_file, header=True)
        # ad-hoc. Todo: group input_files for different bpmask files

        method = self.config_node.imcoadd.interp_type
        weight = self.config_node.imcoadd.weight_map  # boolean flag for generating weight map

        # find images that need interpolation
        uncalculated_images = []
        calculated_outputs = []
        for input_image_file, output_file in zip(input_images, interp_images):
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
                f"({time_diff_in_seconds(st, return_float=True)/len(input_images):.1f} s/image)"
            )

        # advance the target images of interest
        self.images_to_coadd = interp_images
        return interp_images

    def zpscale(self, input_images: list[str] | None = None) -> list[str]:
        """
        Store the value in header as FLXSCALE, and use it in coadding.
        Keep FSCALE_KEYWORD = FLXSCALE in SWarp config.
        The headers of the last processed images are modified.
        """
        if input_images is None:
            input_images = self.images_to_coadd
        st = time.time()
        zpvalues = self.input_headers.values(self.zpkey)
        # base zero point for flux scaling
        # base = np.where(zpvalues == np.max(zpvalues))[0][0]
        # self.zp_base = zpvalues[base]
        # if self.zp_base < np.max(zpvalues):
        #     self.logger.warning(
        #         f"Scaline downward: destination ZP: ({self.zp_base}), "
        #         f"max image ZP: ({np.max(zpvalues)})"
        #     )
        self.logger.debug(f"Reference zero point: {self.zp_base}")
        for i, (file, zp) in enumerate(zip(input_images, zpvalues)):
            flxscale = 10 ** (0.4 * (self.zp_base - zp))
            with fits.open(file, mode="update") as hdul:
                hdul[0].header["FLXSCALE"] = (
                    flxscale,
                    "flux scaling factor by 7DT Pipeline (ImCoadd)",
                )
                hdul.flush()
            # Stamp on snapshot so coadd_header (SATURATE/EGAIN) can read it back without fits I/O
            self.input_headers[i]["FLXSCALE"] = (flxscale, "flux scaling factor by 7DT Pipeline (ImCoadd)")
            self.logger.debug(f"{get_basename(file)} FLXSCALE: {flxscale:.3f}")

        self.logger.info(f"ZP scaling is completed in {time_diff_in_seconds(st)} seconds")
        return input_images

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

    def joint_registration(self, input_images: list[str] | None = None) -> list[str] | None:
        """
        It can address cross-filter registration when given just the image paths.
        Just give the new joint WCS to all images and let individual ImCoadd
        handle the rest of the process.
        """
        if input_images is None:
            input_images = self.images_to_coadd
        return input_images

    def prepare_convolution(self, input_images: list[str] | None = None):
        """
        This is ad-hoc. Change it to convolve after resampling and take
        advantage of uniform pixel scale.
        """
        if input_images is None:
            input_images = self.images_to_coadd

        method = self.config_node.imcoadd.convolve.lower()
        self.conv_method = method
        self.logger.info(f"Prepare the convolution with {method} method")

        self._conv_inputs = input_images
        self.kernels = []

        if method == "gaussian":
            from ..utils import force_symlink

            # Define output path; conv name follows the actual stage input
            path_conv = self.path.imcoadd.factory.conv_dir
            self.config_node.imcoadd.conv_files = [
                os.path.join(path_conv, add_suffix(get_basename(f), "conv")) for f in input_images
            ]

            # Get peeings for convolution
            peeings = [fits.getval(inim, "PEEING") for inim in input_images]
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
                    force_symlink(input_images[i], self.config_node.imcoadd.conv_files[i])
                    if self.config_node.imcoadd.weight_map:
                        force_symlink(
                            add_suffix(input_images[i], "weight"),
                            add_suffix(self.config_node.imcoadd.conv_files[i], "weight"),
                        )
                    self.kernels.append(None)
                else:
                    self.kernels.append(delta_peeing)  # 8*sig + 1 sized

        else:
            self.logger.info("Undefined convolution method. Skipping seeing match")

    def run_convolution(
        self,
        input_images: list[str] | None = None,
        device_id=None,
        use_gpu: bool = True,
        weight=False,
    ) -> list[str]:
        # from .convolve import convolve_fft, get_edge_mask

        if input_images is None:
            input_images = getattr(self, "_conv_inputs", None) or self.images_to_coadd
        st = time.time()
        method = self.conv_method
        self._use_gpu = all([use_gpu, self.config_node.imcoadd.gpu, self._use_gpu])
        device_id = device_id if self._use_gpu else "CPU"

        # compute
        kernels = [k for k in self.kernels if k is not None]
        image_list = [f for f, k in zip(input_images, self.kernels) if k is not None]
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
                weight_list = [f for f, k in zip(PathHandler(input_images).weight, self.kernels) if k is not None]
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
            f"Convolution is completed in {time_diff_in_seconds(st)} seconds ({time_diff_in_seconds(st, return_float=True)/len(input_images):.1f} s/image)"
        )

        conv_files = self.config_node.imcoadd.conv_files
        self.images_to_coadd = conv_files
        return conv_files

    def _calc_delta_peeing(self, peeing):
        delta_peeing = np.sqrt(self._max_peeing**2 - peeing**2)
        if delta_peeing == 0:
            self.logger.debug(f"Skipping calculating delta peeing.")
            return None
        else:
            return delta_peeing

    def reproject_and_coadd_with_swarp(
        self,
        input_images: list[str] | None = None,
        coadd: bool = True,
        swarp_options_override: list[str] = [],
    ) -> str | list[str]:
        """Run SWarp. ``coadd=True`` produces a single coadd image;
        ``coadd=False`` only reprojects each input and returns the resampled paths."""
        st = time.time()
        action = "coadding" if coadd else "reprojecting"
        self.logger.info(f"Start to run swarp for {action} images")

        if input_images is None:
            input_images = self.images_to_coadd
        self.logger.debug(f"input_images: {input_images}")

        swarp_options_override_from_config = get_key(self.config_node.imcoadd, "swarp_options_override", default=[])
        swarp_options_override = swarp_options_override_from_config + swarp_options_override
        if swarp_options_override:
            self.logger.warning(f"SWarp options override: {swarp_options_override}")

        # Write target images to a text file
        self.path_imagelist = os.path.join(self.path.imcoadd.tmp_dir, "images_to_coadd.txt")
        with open(self.path_imagelist, "w") as f:
            for inim in input_images:
                f.write(f"{inim}\n")

        self.logger.debug(f"Total Exptime: {self.input_headers.total_exptime}")

        if not self.config_node.imcoadd.weight_map:
            self._run_swarp("", coadd=coadd, swarp_args=swarp_options_override)
        else:
            self._run_swarp(
                "sci",
                coadd=coadd,
                swarp_args=["-RESAMPLING_TYPE", "LANCZOS3"] + swarp_options_override,
                use_weight_map=False,
            )  # Disable weight in the sci pass
            self._run_swarp("wht", coadd=coadd, swarp_args=["-RESAMPLING_TYPE", "NEAREST"] + swarp_options_override)

            # Update/TODO: consider uncollapsed bpmask files
            if self.config_node.imcoadd.propagate_mask:
                # bpmask_file = self.config.preprocess.bpmask_file
                bpmask_file = PathHandler.get_bpmask(input_images)
                bpmask_inverted = 1 - fits.getdata(bpmask_file)
                bpmask_inverted_file = self.path.imcoadd.factory.bpmask_inverted(bpmask_file)
                fits.writeto(bpmask_inverted_file, bpmask_inverted, overwrite=True)
                self.logger.debug(f"Inverted bpmask saved as {bpmask_inverted_file}")
                args = ["-WEIGHT_IMAGE", bpmask_file, "-RESAMPLING_TYPE", "LANCZOS3"]
                self._run_swarp("bpm", coadd=coadd, swarp_args=args + swarp_options_override)

        if coadd:
            self._update_header()
            self.logger.info(f"Running swarp is completed in {time_diff_in_seconds(st)} seconds")
            return self.config_node.imcoadd.coadd_image

        # reproject-only branch: predict resampled output paths (named by SWarp from its inputs)
        pass_type = "sci" if self.config_node.imcoadd.weight_map else ""
        resampled = atleast_1d(self.path.imcoadd.factory.resampled_images(input_images, pass_type=pass_type))
        self.config_node.imcoadd.resampled_images = resampled
        self.images_to_coadd = resampled
        self.logger.info(f"SWarp reprojection completed in {time_diff_in_seconds(st)} seconds")
        return resampled

    def _run_swarp(
        self, type: Literal["sci", "wht", "bpm"] = "", coadd=True, swarp_args=None, use_weight_map: bool = True
    ) -> str:
        """Pass type='' for no weight. Returns the SWarp resample directory."""
        working_dir = os.path.join(self.path.imcoadd.tmp_dir, type)
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
            coadd=coadd,
            log_file=log_file,
            logger=self.logger,
            use_weight_map=self.config_node.imcoadd.weight_map and use_weight_map,
            swarp_args=swarp_args,
        )

        # only the coadd branch produces output_file to move
        if coadd:
            if type == "sci":
                shutil.move(output_file, self.config_node.imcoadd.coadd_image)
            elif type == "wht":
                shutil.move(
                    add_suffix(output_file, "weight"),
                    self.path.imcoadd.factory.coadd_weight_image,
                )
            elif type == "bpm":
                shutil.move(
                    add_suffix(output_file, "weight"),
                    self.path.imcoadd.factory.coadd_bpmask_image,
                )

        return resample_dir

    def coadd_in_memory(self, input_images: list[str] | None = None, device_id=None) -> str:
        """Dispatcher to numpy/cupy and mean/median backends.
        ``imcoadd.coadd_mode`` picks the combine algorithm; for ``mean``, the
        weighted variant kicks in when ``imcoadd.weight_map`` is set and the
        NEAREST-resampled weight maps (``tmp/wht/resamp/<base>.weight.fits``)
        produced by ``reproject_and_coadd_with_swarp`` exist for every input."""
        if input_images is None:
            input_images = self.images_to_coadd

        if device_id is not None:
            self.coadd_with_cupy(input_images, device_id=device_id)
            return self.config_node.imcoadd.coadd_image

        mode = self.config_node.imcoadd.coadd_mode
        if mode == "mean":
            weights = None
            if self.config_node.imcoadd.weight_map:
                # NEAREST-resampled weights live next to the wht pass output; the
                # LANCZOS3 companions next to the sci resamp ring to ~0 almost
                # everywhere (99%+ zeros) and must NOT be used.
                wht_dir = self.path.imcoadd.factory.swarp_resample_dir("wht")
                candidates = atleast_1d(
                    self.path.imcoadd.factory.resampled_weight_images(input_images, pass_type="wht")
                )
                if all(os.path.exists(w) for w in candidates):
                    weights = candidates
                else:
                    missing = [w for w in candidates if not os.path.exists(w)][:3]
                    self.logger.warning(
                        f"weight_map=True but NEAREST-resampled weight maps not found "
                        f"in {wht_dir} (e.g. {missing}); falling back to simple mean."
                    )
            self.coadd_with_numpy(input_images, weights=weights)
        elif mode == "median":
            self.coadd_median_with_numpy(input_images)
        else:
            raise ValueError(f"Invalid coadd mode: {mode!r} (expected 'mean' or 'median')")
        return self.config_node.imcoadd.coadd_image

    def coadd_with_numpy(
        self,
        input_images: list[str],
        weights: list[str] | None = None,
        match_swarp_size: bool = True,
    ) -> str:
        return mean_coadd_numpy(
            input_images,
            output_path=self.config_node.imcoadd.coadd_image,
            coadd_header=self.input_headers.coadd_header,
            weights=weights,
            weight_output=self.path.imcoadd.factory.coadd_weight_image,
            flxscales=self.input_headers.values("FLXSCALE"),
            match_swarp_size=match_swarp_size,
            logger=self.logger,
        )

    # ---- median backend ----

    _MEDIAN_CHUNK_H: int = 128  # row strips; 142 frames × 128 × 10200 × 4 B ≈ 750 MB per strip

    def coadd_median_with_numpy(self, input_images: list[str], match_swarp_size: bool = True) -> str:
        return median_coadd_numpy(
            input_images,
            output_path=self.config_node.imcoadd.coadd_image,
            coadd_header=self.input_headers.coadd_header,
            flxscales=self.input_headers.values("FLXSCALE"),
            match_swarp_size=match_swarp_size,
            chunk_h=self._MEDIAN_CHUNK_H,
            logger=self.logger,
        )

    def coadd_with_cupy(self, input_images: list[str], device_id) -> str:
        raise NotImplementedError("GPU coadd_with_cupy is not implemented yet")

    def plot_coadd_image(self):
        coadd_img = self.config_node.imcoadd.coadd_image
        basename = os.path.basename(coadd_img)
        path_to_plot = self.path.figure_dir_to_path / swap_ext(basename, "jpg")
        save_fits_as_figures(fits.getdata(coadd_img), path_to_plot)
        self.logger.info(f"Coadd image is plotted and saved in {path_to_plot}.")

    def _update_header(self):
        """Legacy routine: overlay ``self.input_headers.coadd_header`` onto the SWarp-written coadd FITS."""
        coadd_header = self.input_headers.coadd_header
        # 	Put them into coadded image / Update Header
        with fits.open(self.config_node.imcoadd.coadd_image, mode="update") as hdul:
            header = hdul[0].header
            for card in coadd_header.cards:
                header[card.keyword] = (card.value, card.comment)
            hdul.flush()
