import json
import os
import threading
import time
import shutil
import warnings
from typing import Literal

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, WCSCOMPARE_ANCILLARY

from ..const import REF_DIR
from ..const.sciproc import COADD_SPEC, SCIPROCESS_REGISTRY
from ..errors import CoaddError
from ..config import SciProcConfiguration
from ..path.path import PathHandler
from ..services.setup import BaseSetup
from ..services.utils import acquire_available_gpu, conservative_worker_count
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
from .calc import clipped_mean_coadd_numpy, mean_coadd_numpy, median_coadd_numpy


warnings.filterwarnings("ignore")


class ImCoadd(BaseSetup, DatabaseHandler, Checker, RuntimeVersionMixin):
    _process_spec = COADD_SPEC
    _process_registry = SCIPROCESS_REGISTRY
    _process_error = CoaddError
    _homogeneous_header_keys = ("OBJECT", "FILTER")
    _input_label = "singles"
    _output_filter = None
    _extra_header_keys = ()
    _max_header_keys = ()
    _proper_requires_interpolation = True
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
        self.logger.process_error = self._process_error

        self.qa_id = None
        DatabaseHandler.__init__(
            self, use_database=self.config_node.settings.is_pipeline, is_too=self.config_node.settings.is_too
        )

        if self.is_connected:

            self.process_status_id = self.create_process_data(self.config_node)
            self.reset_exceptions(self._process_spec.name)

            if self.process_status_id is not None:
                from ..services.database.handler import ExceptionHandler

                self.logger.database = ExceptionHandler(self.process_status_id)

            if self.too_id is not None:
                self.logger.debug(f"Initialized DatabaseHandler for ToO data management, ToO ID: {self.too_id}")
            else:
                self.logger.debug(
                    f"Initialized DatabaseHandler for pipeline and QA data management, Pipeline ID: {self.process_status_id}"
                )
            self.update_progress(
                self._process_registry.configured_progress(self._process_spec),
                self._progress_status("configured"),
            )

    def _progress_status(self, suffix: str) -> str:
        return f"{self._process_spec.name}-{suffix}"

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
        self.update_progress(
            self._process_registry.milestone_progress(self._process_spec, "bkgsub"),
            self._progress_status("bkgsub-completed"),
        )
        # zero point scaling
        self.zpscale(images)
        self.update_progress(
            self._process_registry.milestone_progress(self._process_spec, "zpscale"),
            self._progress_status("zpscale-completed"),
        )

        if self._coadd_plan()["need_weights"]:
            self.calculate_weight_map(images, device_id=device_id)
            self.update_progress(
                self._process_registry.milestone_progress(self._process_spec, "calculate_weight_map"),
                self._progress_status("calculate-weight-map-completed"),
            )

        # replace hot pixels
        if self._coadd_plan()["interpolate"]:
            images = self.apply_bpmask(images, device_id=device_id)
            self.update_progress(
                self._process_registry.milestone_progress(self._process_spec, "apply_bpmask"),
                self._progress_status("apply-bpmask-completed"),
            )

        # re-registration
        if self.config_node.imcoadd.joint_wcs:
            images = self.joint_registration(images)
            self.update_progress(
                self._process_registry.milestone_progress(self._process_spec, "joint_registration"),
                self._progress_status("joint-registration-completed"),
            )

        # seeing convolution
        if self.config_node.imcoadd.convolve:
            self.prepare_convolution(images)
            images = self.run_convolution(images, device_id=device_id)
            self.update_progress(
                self._process_registry.milestone_progress(self._process_spec, "run_convolution"),
                self._progress_status("run-convolution-completed"),
            )

        # swarp coaddition
        self.reproject_and_coadd_with_swarp(images, coadd=True)
        self.update_progress(
            self._process_registry.milestone_progress(self._process_spec, "coadd_with_swarp"),
            self._progress_status("coadd-with-swarp-completed"),
        )

        self.plot_coadd_image()
        self.update_progress(
            self._process_registry.milestone_progress(self._process_spec, "plot_coadd_image"),
            self._progress_status("plot-coadded-image-completed"),
        )

        self.register_coadd_qa()

        self.update_progress(
            self._process_registry.completed_progress(self._process_spec),
            self._progress_status("completed"),
        )

    def reproject_first_coadd_routine(self, use_gpu: bool = False, device_id=None):
        """SWarp for reprojection only; the combine step runs in memory via
        ``coadd_in_memory``, which picks mean/median from ``imcoadd.coadd_mode``."""
        self._use_gpu = all([use_gpu, self.config_node.imcoadd.gpu, self._use_gpu])

        do_zpscale = bool(get_key(self.config_node.imcoadd, "zpscale", default=True))
        optional_steps = (
            int(bool(self._coadd_plan()["need_weights"]))
            + int(bool(self._coadd_plan()["interpolate"]))
            + int(bool(self.config_node.imcoadd.joint_wcs))
            + int(bool(self.config_node.imcoadd.convolve))
            + int(do_zpscale)
        )
        TOTAL_STEPS = 4 + optional_steps  # reproject + bkgsub + coadd + plot + optionals
        step = 0

        self.initialize()

        images = self.input_images
        weight_images = None
        do_weight = bool(self._coadd_plan()["need_weights"])
        do_bpmask = bool(self._coadd_plan()["interpolate"])
        if do_weight and do_bpmask:
            # fused: the weight map is handed to interpolation in memory, one read and one
            # write per image instead of three reads and two writes
            images = self.weight_and_interpolate(images)
            weight_images = [add_suffix(im, "weight") for im in images]
            step += 1
            self.update_progress(
                self._process_registry.step_progress(self._process_spec, step, TOTAL_STEPS),
                self._progress_status("calculate-weight-map-completed"),
            )
            step += 1
            self.update_progress(
                self._process_registry.step_progress(self._process_spec, step, TOTAL_STEPS),
                self._progress_status("apply-bpmask-completed"),
            )
        else:
            if do_weight:
                # weights come from the pristine frames: the Poisson term must see the measured
                # pixel, not an interpolated one (masked pixels are zeroed at interp anyway).
                # Hence the names cannot follow a later stage product nor sit next to the inputs.
                factory = self.path.imcoadd.factory
                weight_images = factory.stage_images(images, "weight", factory.weight_dir)
                self.calculate_weight_map(images, device_id=device_id, out_weights=weight_images)
                step += 1
                self.update_progress(
                    self._process_registry.step_progress(self._process_spec, step, TOTAL_STEPS),
                    self._progress_status("calculate-weight-map-completed"),
                )

            # Bad pixel interpolation
            if do_bpmask:
                images = self.apply_bpmask(images, device_id=device_id, weight_images=weight_images)
                if weight_images is not None:
                    # hand the interp sidecars onward: they carry the zeroed bad-pixel holes
                    weight_images = [add_suffix(im, "weight") for im in images]
                step += 1
                self.update_progress(
                    self._process_registry.step_progress(self._process_spec, step, TOTAL_STEPS),
                    self._progress_status("apply-bpmask-completed"),
                )

        # Optional joint registration
        if self.config_node.imcoadd.joint_wcs:
            images = self.joint_registration(images)
            step += 1
            self.update_progress(
                self._process_registry.step_progress(self._process_spec, step, TOTAL_STEPS),
                self._progress_status("joint-registration-completed"),
            )

        # Reproject (no combine) onto a common WCS
        images = self.reproject_and_coadd_with_swarp(images, coadd=False, weight_images=weight_images)
        if self.config_node.imcoadd.convolve:
            # only convolution runs between here and bkgsub, and it is what destroys the
            # exactly-zero padding -- so only then must the footprint be captured now.
            # Without it bkgsub derives the identical mask from the read it already does,
            # saving a serial 262 MB read + 65 MB write per frame (see build_fov_masks).
            self.build_fov_masks(images)
        self._discard_consumed_interp()
        step += 1
        self.update_progress(
            self._process_registry.step_progress(self._process_spec, step, TOTAL_STEPS),
            self._progress_status("reproject-completed"),
        )

        # PSF homogenization
        if self.config_node.imcoadd.convolve:
            self.prepare_convolution(images)
            images = self.run_convolution(images, device_id=device_id)
            self.shrink_fov_masks(self.delta_peeings)
            step += 1
            self.update_progress(
                self._process_registry.step_progress(self._process_spec, step, TOTAL_STEPS),
                self._progress_status("run-convolution-completed"),
            )

        # Background subtraction on reprojected (+ optionally convolved) frames
        images = self.bkgsub(
            images,
            mask_out_of_fov=True,
            mask_sources=get_key(self.config_node.imcoadd, "source_mask", default=True),
            fov_masks=getattr(self, "_fov_masks", None),
        )
        self._discard_consumed_bkgsub_inputs()
        step += 1
        self.update_progress(
            self._process_registry.step_progress(self._process_spec, step, TOTAL_STEPS),
            self._progress_status("bkgsub-completed"),
        )

        # Flux zero-point scaling (snapshot only; the in-memory combine takes the values directly)
        self.zpscale(images, write_headers=False)
        if do_zpscale:
            step += 1
            self.update_progress(
                self._process_registry.step_progress(self._process_spec, step, TOTAL_STEPS),
                self._progress_status("zpscale-completed"),
            )

        # In-memory coaddition (mean/median selected by imcoadd.coadd_mode)
        self.coadd_in_memory(images, device_id=device_id)
        step += 1
        self.update_progress(
            self._process_registry.step_progress(self._process_spec, step, TOTAL_STEPS),
            self._progress_status("coadd-completed"),
        )

        # Plot coadd image
        self.plot_coadd_image()
        step += 1
        self.update_progress(
            self._process_registry.step_progress(self._process_spec, step, TOTAL_STEPS),
            self._progress_status("plot-completed"),
        )

        self.register_coadd_qa()

        self.update_progress(
            self._process_registry.completed_progress(self._process_spec),
            self._progress_status("completed"),
        )

    def direct_coadd_routine(self, use_gpu: bool = False, device_id=None):
        """Combine inputs already on one pixel grid without a SWarp pass."""
        self._use_gpu = all([use_gpu, self.config_node.imcoadd.gpu, self._use_gpu])
        if self.config_node.imcoadd.joint_wcs or self.config_node.imcoadd.convolve:
            raise self._process_error.ValueError("Direct coaddition requires joint_wcs=False and convolve=False")

        plan = self._coadd_plan()
        do_zpscale = bool(get_key(self.config_node.imcoadd, "zpscale", default=True))
        total_steps = 3 + int(plan["need_weights"]) + int(plan["interpolate"]) + int(do_zpscale)
        step = 0

        self.initialize()
        self._validate_direct_grid()
        images = self.input_images
        weight_images = None

        if plan["need_weights"]:
            factory = self.path.imcoadd.factory
            weight_images = factory.stage_images(images, "weight", factory.weight_dir)
            weight_images = self.calculate_weight_map(images, device_id=device_id, out_weights=weight_images)
            step += 1
            self.update_progress(
                self._process_registry.step_progress(self._process_spec, step, total_steps),
                self._progress_status("calculate-weight-map-completed"),
            )

        if plan["interpolate"]:
            images = self.apply_bpmask(images, device_id=device_id, weight_images=weight_images)
            if weight_images is not None:
                weight_images = [add_suffix(image, "weight") for image in images]
            step += 1
            self.update_progress(
                self._process_registry.step_progress(self._process_spec, step, total_steps),
                self._progress_status("apply-bpmask-completed"),
            )

        images = self.bkgsub(
            images,
            mask_out_of_fov=True,
            mask_sources=get_key(self.config_node.imcoadd, "source_mask", default=True),
        )
        step += 1
        self.update_progress(
            self._process_registry.step_progress(self._process_spec, step, total_steps),
            self._progress_status("bkgsub-completed"),
        )

        self.zpscale(images, write_headers=False)
        if do_zpscale:
            step += 1
            self.update_progress(
                self._process_registry.step_progress(self._process_spec, step, total_steps),
                self._progress_status("zpscale-completed"),
            )

        self.coadd_in_memory(images, device_id=device_id, weight_images=weight_images)
        step += 1
        self.update_progress(
            self._process_registry.step_progress(self._process_spec, step, total_steps),
            self._progress_status("coadd-completed"),
        )

        self.plot_coadd_image()
        step += 1
        self.update_progress(
            self._process_registry.step_progress(self._process_spec, step, total_steps),
            self._progress_status("plot-completed"),
        )
        self.register_coadd_qa()
        self.update_progress(
            self._process_registry.completed_progress(self._process_spec),
            self._progress_status("completed"),
        )

    def run(self, use_gpu: bool = False, device_id=None):
        try:
            routine = self.config_node.imcoadd.coadd_routine
            if "legacy" in routine.lower():
                self.legacy_coadd_routine(use_gpu=use_gpu, device_id=device_id)
            elif "reproject-first" in routine.lower():
                self.reproject_first_coadd_routine(use_gpu=use_gpu, device_id=device_id)
            elif "direct" in routine.lower():
                self.direct_coadd_routine(use_gpu=use_gpu, device_id=device_id)
            else:
                raise ValueError(
                    f"Invalid coadd routine: {routine!r} (expected 'legacy', 'reproject-first', or 'direct')"
                )

            setattr(self.config_node.flag, self._process_spec.name, True)
            self.record_runtime_version()
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
        self.apply_sanity_filter_and_report(current_process=self._process_spec, overwrite=self.overwrite)
        if not self.input_images:
            self.logger.error("No Input for ImCoadd", self._process_error.EmptyInputAfterSanityRejection)
            raise self._process_error.EmptyInputAfterSanityRejection("No Input for ImCoadd")
        # if rejected, let the input remain so that a rerun has a change to reevaluate SANITY

        if str(get_key(self.config_node.imcoadd, "coadd_mode") or "").lower() == "proper":
            self._validate_proper_mode()
        self._prune_factory_scratch()
        self.select_input_images()  # may drop inputs, so it precedes the snapshot and resync
        # Single read of every kept header; all aggregates/coadd_header live on this snapshot.
        self.input_headers = InputHeaderSet.from_files(self.input_images)
        if self._output_filter is not None:
            self.input_headers.output_filter = self._output_filter
        self.input_headers.input_label = self._input_label
        self.input_headers.extra_core_keys = self._extra_header_keys
        self.input_headers.max_core_keys = self._max_header_keys
        self.input_headers.selection_metrics = getattr(self, "_selection_meta", {})
        self.input_headers.coadd_provenance = self._coadd_provenance()
        self.input_headers.multi_epoch = bool(self.config_node.settings.is_multi_epoch)

        self._recreate_pathhandler_instance()  # resync
        self.config_node.imcoadd.input_images = self.input_images

        self.zpkey = self.config_node.imcoadd.zp_key or ZP_KEY
        # self.ic_keys = IC_KEYS

        # self.define_paths(working_dir=self.config.path.path_processed)

        self.input_headers.check_uniqueness(self._homogeneous_header_keys, self.logger)
        self.center = (
            None
            if "direct" in str(self.config_node.imcoadd.coadd_routine or "").lower()
            else self.input_headers.deprojection_center
        )
        self.logger.debug(f"Deprojection center: {self.center}")

        # Output coadd image file name
        if not get_key(self.config_node.imcoadd, "coadd_image"):
            self.config_node.imcoadd.coadd_image = self.path.imcoadd.coadd_image
        self.config_node.input.coadd_image = self.config_node.imcoadd.coadd_image
        self.logger.debug(f"Coadd Image: {self.config_node.imcoadd.coadd_image}")
        if self.config_node.settings.is_multi_epoch:
            self._guard_coadd_identity()

        self.logger.info(f"Initialization for ImCoadd is completed")

    def _validate_direct_grid(self, tolerance: float = 1e-7) -> None:
        reference_header = self.input_headers[0]
        reference_shape = (reference_header.get("NAXIS2"), reference_header.get("NAXIS1"))
        reference_wcs = WCS(reference_header).celestial.wcs
        mismatched = []
        for name, header in zip(self.input_headers.names[1:], self.input_headers.headers[1:]):
            shape = (header.get("NAXIS2"), header.get("NAXIS1"))
            same_wcs = reference_wcs.compare(
                WCS(header).celestial.wcs,
                cmp=WCSCOMPARE_ANCILLARY,
                tolerance=tolerance,
            )
            if shape != reference_shape or not same_wcs:
                mismatched.append(name)
        if mismatched:
            raise self._process_error.ValueError(
                f"Direct coaddition requires one matched pixel grid; mismatched inputs: {mismatched[:3]}"
            )
        self.logger.info(f"Validated one shared pixel grid for {len(self.input_images)} inputs")

    def select_input_images(self, nsigma: float = 1.0, metrics=None, extra=None) -> list[str]:
        """Cull the inputs on SEEING / ELLIP / depth. Multi-epoch coadds only.

        A nightly stack already ran the SANITY filter and takes everything that passed;
        a multi-epoch stack spans very different conditions, so ``imcoadd.image_selection``
        applies quality cuts on top: False = off, 'auto' = apply the suggested cuts without
        asking (the multi-epoch default), 'interactive' = show them and take your edits
        first. 'interactive' needs a notebook and falls back to 'auto' anywhere else, so a
        queued run can never block on stdin.

        ``metrics``/``extra`` override or extend the default SEEING/ELLIP/depth set —
        see pipeline/select/select.py."""
        # default mirrors sciproc_base.yml: a config predating the key must not silently
        # gain a filtering step. New multi-epoch configs get 'auto' from the override yml.
        self._selection_meta = {}
        mode = get_key(self.config_node.imcoadd, "image_selection", default=False)
        if not (mode and self.config_node.settings.is_multi_epoch):
            return self.input_images

        from ..select.select import (CATEGORICAL_METRICS, metrics_for_paths_from_image_qa, ppflag_mask,
                                     ppflag_spec, resolve_fixed_cuts, select_from_table, select_images)  # fmt: skip

        plot_path = os.path.join(
            # multi-epoch inputs span nightdates, so figure_dir is a list of dirs;
            # figure_dir is shared by every coadd config of this target
            collapse(self.path.figure_dir, force=True),
            f"{os.path.splitext(get_basename(self.config_node.info.file))[0]}_imcoadd_selection.jpg",
        )
        # PPFLAG is a bitmask: the "cut" is an allow-list of bits, not a threshold
        fixed_cuts = {
            "ppflag": ppflag_mask(get_key(self.config_node.imcoadd, "ppflag_bitmask", default="110000")),
            **(get_key(self.config_node.imcoadd, "image_selection_cuts") or {}),
        }

        # Metrics from image_qa when possible: header reads then happen only for the kept
        # frames, in the snapshot. Any failure (unknown column, missing rows would just be
        # NaN, DB down) falls back to reading every header, as before.
        table = None
        source = str(get_key(self.config_node.imcoadd, "image_selection_source", default="db")).lower()
        if source == "db" and len(self.input_images) >= 20 and self.is_connected:
            try:
                numeric_cuts, db_extra = resolve_fixed_cuts(fixed_cuts, metrics, extra)
                table, n_found = metrics_for_paths_from_image_qa(self.input_images, metrics=metrics, extra=db_extra)
                self.logger.info(
                    f"Selection metrics from image_qa for {n_found}/{len(self.input_images)} images -- "
                    "may lag the headers if edited out of band; set "
                    "imcoadd.image_selection_source: 'headers' to read headers"
                )
            except Exception as e:
                self.logger.warning(f"image_qa selection metrics unavailable; reading headers instead: {e}")
                table = None

        if table is not None:
            keep, cuts = select_from_table(
                table, mode=str(mode).lower(), nsigma=nsigma, plot_path=plot_path,
                logger=self.logger, fixed_cuts=numeric_cuts,
            )  # fmt: skip
        else:
            keep, cuts, table = select_images(
                [get_basename(f) for f in self.input_images],
                [fits.getheader(f) for f in self.input_images],
                mode=str(mode).lower(),
                nsigma=nsigma,
                plot_path=plot_path,
                logger=self.logger,
                metrics=metrics,
                extra=extra,
                fixed_cuts=fixed_cuts,
            )
        self._selection_meta = {
            name: (key, table.meta["directions"][name]) for name, key in table.meta.get("keys", {}).items()
        }
        if keep.all():
            return self.input_images
        if not keep.any():
            self.logger.error(f"Quality cuts {cuts} reject all {len(keep)} images",
                              self._process_error.EmptyInputAfterSanityRejection)  # fmt: skip
            raise self._process_error.EmptyInputAfterSanityRejection(f"Quality cuts {cuts} reject all {len(keep)} images")

        self.input_images = [f for f, ok in zip(self.input_images, keep) if ok]
        # written back in the operator grammar, so a rerun pins exactly what this run applied
        directions = table.meta.get("directions", {})
        self.config_node.imcoadd.image_selection_cuts = {
            m: (
                ppflag_spec(v)
                if m in CATEGORICAL_METRICS
                else f"{'<=' if directions.get(m) == 'lower' else '>='}{v:.6g}"
            )
            for m, v in cuts.items()
        }
        return self.input_images

    def bkgsub(
        self,
        input_images: list[str] | None = None,
        ignore_steppy_flag: bool = False,
        skyval_cut: float = 40,
        mask_out_of_fov: bool = False,
        mask_sources: bool | str = False,
        fov_masks: list | None = None,
    ) -> list[str]:
        """``imcoadd.bkgsub_type`` names a routine from `bkgsub_methods` for every image,
        or 'individual' to choose per image from that frame's SKYVAL. Left unset it is
        filled in with one routine for the whole group (`_default_bkgsub_type`), which is
        what it has always done.

        ``mask_sources`` mirrors ``imcoadd.source_mask``: False, True, or "iterative"
        (a second round after coaddition, not yet wired). Source masking only means
        anything for a mesh background, so images on the constant routine ignore it."""
        # ------------------------------------------------------------
        # 	Global Background Subtraction
        # ------------------------------------------------------------
        if input_images is None:
            input_images = self.input_images
        st = time.time()

        factory = self.path.imcoadd.factory
        self.path_bkgsub = factory.bkgsub_dir

        bkgsub_images = factory.stage_images(input_images, "bkgsub", self.path_bkgsub)
        self.config_node.imcoadd.bkgsub_images = bkgsub_images

        bkg_images = factory.stage_images(input_images, "bkg", self.path_bkgsub)
        bkg_rms_images = factory.stage_images(input_images, "bkgrms", self.path_bkgsub)

        # Reprojected frames carry zero-padded out-of-FOV corners; mask them out.
        # Masks built earlier from the pristine resamps (`build_fov_masks`) win where they
        # exist: once a stage has run in between, the padding is no longer exactly 0 and
        # cannot be found. Otherwise the mask is derived in memory from the frame this
        # stage reads anyway; the path is then only scratch for a SExtractor weight image.
        if fov_masks is not None:
            fov_mask_images = list(fov_masks)
        elif mask_out_of_fov:
            fov_mask_images = factory.stage_images(input_images, "fovmask", self.path_bkgsub)
        else:
            fov_mask_images = [None] * len(input_images)

        skyvalues = self.input_headers.values("SKYVAL")
        methods = self.bkgsub_methods()
        requested = get_key(self.config_node.imcoadd, "bkgsub_type")
        if requested:
            requested = str(requested).lower()
        else:
            # unset: fill it in with one routine for the whole group, as it always did
            requested = self._default_bkgsub_type(skyvalues, skyval_cut)
            self.logger.debug(f"bkgsub_type unset; filled in as {requested!r} for the group")
        if requested != "individual" and requested not in methods:
            raise ValueError(
                f"bkgsub_type: {requested!r} is invalid (expected 'individual' or one of {sorted(methods)})"
            )
        types = [self._resolve_bkgsub_type(requested, sv, skyval_cut) for sv in skyvalues]
        self.config_node.imcoadd.bkgsub_type = requested

        # Stamp BACKTYPE onto the in-memory snapshot so coadd_header picks it up;
        # a mixed group aggregates to "MIXED" there.
        for hdr, btype in zip(self.input_headers, types):
            hdr["BACKTYPE"] = (btype.upper(), "Background subtraction type")

        # Sources bias the mesh statistics upward, but only a mesh background has meshes
        any_dynamic = "dynamic" in types
        source_mask_images = (
            factory.stage_images(input_images, "srcmask", self.path_bkgsub)
            if (mask_sources and any_dynamic)
            else [None] * len(input_images)
        )
        if mask_sources and not any_dynamic:
            self.logger.info("No image takes a mesh background: source_mask has no effect")

        # catalog reuse needs the inputs to be the singles' derivatives 1:1
        singles = atleast_1d(self.input_images)
        try:
            catalogs = atleast_1d(self.path.photometry.final_catalog)
        except Exception as e:  # an optimization must not become a new failure mode
            self.logger.debug(f"No photometry catalogs resolvable ({e}); detection pass it is")
            catalogs = []
        if not (len(singles) == len(catalogs) == len(input_images)):
            singles = catalogs = [None] * len(input_images)

        counts = {name: types.count(name) for name in sorted(set(types))}
        self.logger.info(f"Start background subtraction (bkgsub_type={requested!r}): {counts}")
        if any_dynamic:
            self.config_node.imcoadd.bkg_images = (
                bkg_images if get_key(self.config_node.imcoadd, "output_bkg_map", default=False) else None
            )
            self.config_node.imcoadd.bkg_rms_images = (
                bkg_rms_images if get_key(self.config_node.imcoadd, "output_sky_rms_map", default=False) else None
            )
        else:
            if get_key(self.config_node.imcoadd, "bkg_images"):
                self.config_node.imcoadd.bkg_images = None
            if get_key(self.config_node.imcoadd, "bkg_rms_images"):
                self.config_node.imcoadd.bkg_rms_images = None

        def _bkgsub_one(i, inim, outim, bkg, bkg_rms, skyvalue, fov_mask, src_mask, btype, single, phot_cat):
            st_loop = time.time()
            # One read of the frame serves the FOV mask, the detection pass and the
            # background model, and the masks stay arrays. Round-tripping them through
            # 65 MB files (plus a second and third read of the frame) was ~1 GB of NFS
            # traffic per image that carried nothing this loop did not already hold.
            # memmap=False on purpose: page-fault reads measure ~2x slower over NFS.
            data, header = fits.getdata(inim, header=True, memmap=False)
            data = np.ascontiguousarray(data, dtype=np.float32)

            # the FOV mask serves every routine (it re-zeros the padding afterwards);
            # the source mask only means something to a mesh background
            if fov_mask is None:
                fov_valid = None
            elif fov_masks is not None and os.path.exists(fov_mask):
                fov_valid = fits.getdata(fov_mask, memmap=False).astype(bool)
            else:
                fov_valid = self._fov_valid(data, get_basename(inim))
            exclude = None if fov_valid is None else ~fov_valid
            if src_mask is not None and btype == "dynamic":
                valid, usable = self._source_mask(
                    inim, header, fov_valid, src_mask, fov_mask,
                    photometry_catalog=phot_cat, source_image=single,
                )  # fmt: skip
                if usable < 20.0:
                    # crowded field: too little unmasked sky for a mesh -- fall back to a
                    # constant background for this frame. Defining a GOOD constant level
                    # under crowding is its own open problem (see scientific-guidelines);
                    # this only avoids a mesh built almost entirely on interpolation.
                    self.logger.warning(
                        f"{get_basename(inim)}: source mask leaves {usable:.0f}% of the FOV; "
                        f"falling back to constant background subtraction"
                    )
                    btype = "constant"
                else:
                    exclude = ~valid
            is_steppy = methods[btype](
                inim,
                outim,
                data=data,
                header=header,
                bkg=bkg,
                bkg_rms=bkg_rms,
                skyval=skyvalue,
                ignore_steppy_flag=ignore_steppy_flag,
                exclude=exclude,
                fov_valid=fov_valid,
            )

            # if is_steppy and not ignore_steppy_flag:
            #     self.logger.warning(f"Background subtraction failed for {get_basename(outim)}")
            #     self.logger.warning(f"Re-running background subtraction with constant value")
            #     self._const_bkgsub(inim, outim, skyval=skyvalue)

            self.logger.info(
                f"Background subtraction ({btype}) completed for {get_basename(outim)} [image {i+1}/{len(input_images)}] in {time_diff_in_seconds(st_loop)} seconds"
            )

        # Per-image work is independent (SExtractor subprocesses + FITS I/O, both
        # GIL-releasing); worker count is load-aware so a busy system queue stays safe.
        jobs = list(enumerate(zip(input_images, bkgsub_images, bkg_images, bkg_rms_images, skyvalues,
                                  fov_mask_images, source_mask_images, types, singles, catalogs)))  # fmt: skip
        if not self.overwrite:
            n_all = len(jobs)
            jobs = [(i, job) for i, job in jobs if not os.path.exists(job[1])]
            if len(jobs) < n_all:
                self.logger.info(f"{n_all - len(jobs)} existing bkgsub products skipped, {len(jobs)} to compute")
        n_workers = conservative_worker_count(len(jobs))
        if n_workers <= 1:
            for i, job in jobs:
                _bkgsub_one(i, *job)
        else:
            self.logger.info(f"Background subtraction with {n_workers} workers")
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = [pool.submit(_bkgsub_one, i, *job) for i, job in jobs]
                for f in futures:
                    f.result()

        self.logger.info(
            f"Background subtraction is completed in {time_diff_in_seconds(st)} ({time_diff_in_seconds(st, return_float=True)/len(input_images):.1f} s/image)"
        )

        self.images_to_coadd = bkgsub_images
        return bkgsub_images

    def _guard_coadd_identity(self):
        """Refuse to silently overwrite a coadd built with different settings.

        The provenance cards are the product's identity; same name + different settings
        must either take a config_suffix (coexist) or overwrite=True (replace)."""
        coadd_image = collapse(self.config_node.imcoadd.coadd_image, force=True)
        if self.overwrite or not (coadd_image and os.path.exists(coadd_image)):
            return
        header = fits.getheader(coadd_image)
        wanted = self._coadd_provenance()
        if not any(k in header for k in wanted):
            self.logger.warning(
                f"Existing coadd {get_basename(coadd_image)} predates provenance cards; "
                "its settings are unknown and it will be replaced"
            )
            return
        # an absent card records no setting: the product predates it, which is not a conflict
        mismatch = {
            k: (header[k], v)
            for k, (v, _) in wanted.items()
            if k in header and str(header[k]).upper() != str(v).upper()
        }
        if mismatch:
            detail = ", ".join(f"{k}: disk={d!r} config={c!r}" for k, (d, c) in mismatch.items())
            self.logger.error(f"Coadd identity mismatch on {get_basename(coadd_image)}: {detail}")
            raise self._process_error.ValueError(
                f"{get_basename(coadd_image)} exists with different settings ({detail}). "
                "Use config_suffix to keep both products, or overwrite=True to replace it."
            )

    def _prune_factory_scratch(self, min_idle_hours: float = 6.0):
        """Rotate old factory trees on the scratch volume down to the configured cap.

        Prune unit is one config-stem tree; a tree whose newest file is younger than
        ``min_idle_hours`` is treated as an active run and never touched, so concurrent
        filters cannot delete each other's intermediates."""
        scratch = get_key(self.config_node.settings, "factory_scratch")
        if not scratch:
            return
        cap = float(get_key(self.config_node.settings, "factory_scratch_cap_gb", default=1200)) * 1e9
        own = os.path.abspath(collapse(self.path.factory_dir, force=True))
        trees = []
        total = 0
        for root, dirs, files in os.walk(scratch):
            # stem-level trees: scratch/coadd/<obj>/<filter>/imcoadd/<stem>
            if os.path.basename(os.path.dirname(root)) == "imcoadd" and root != own:
                size = newest = 0
                for r2, _, fs in os.walk(root):
                    for f in fs:
                        try:
                            st_ = os.stat(os.path.join(r2, f))
                        except FileNotFoundError:
                            continue
                        size += st_.st_size
                        newest = max(newest, st_.st_mtime)
                trees.append((newest, size, root))
                total += size
                dirs[:] = []
        if total <= cap:
            return
        for newest, size, root in sorted(trees):
            if total <= cap:
                break
            if time.time() - newest < min_idle_hours * 3600:
                continue
            self.logger.info(
                f"Scratch rotation: removing {root} ({size/1e9:.0f} GB, idle {(time.time()-newest)/3600:.1f} h)"
            )
            shutil.rmtree(root, ignore_errors=True)
            total -= size
        if total > cap:
            self.logger.warning(
                f"Scratch still over cap after rotation ({total/1e9:.0f} GB > {cap/1e9:.0f} GB); all trees active"
            )

    def _coadd_provenance(self) -> dict[str, tuple]:
        """Config options that change the coadd, as coadd header cards."""
        node = self.config_node.imcoadd
        shown = lambda value: "NONE" if value is None or value is False else value  # noqa: E731
        bp = self._coadd_plan()
        interp = get_key(node, "interp_type") if bp["interpolate"] else None
        cards = {
            "COADDRTN": (shown(get_key(node, "coadd_routine")), "imcoadd.coadd_routine"),
            "COADDMOD": (shown(get_key(node, "coadd_mode")), "imcoadd.coadd_mode"),
            "COADDWGT": (shown(get_key(node, "coadd_weighting", default="global")), "imcoadd.coadd_weighting"),
            "BPMPOL":   (shown(bp["policy"]), "imcoadd.badpix_reprojection_policy"),
            "ZBPWGT":   (bool(bp["zero"]), "imcoadd.zero_badpix_weight"),
            "ZPSCALE":  (bool(get_key(node, "zpscale")), "imcoadd.zpscale"),
            "INTERP":   (shown(interp), "imcoadd.interp_type"),
            "CONVOLVE": (shown(get_key(node, "convolve")), "imcoadd.convolve"),
            "JOINTWCS": (bool(get_key(node, "joint_wcs")), "imcoadd.joint_wcs"),
            "IMGSELEC": (shown(get_key(node, "image_selection")), "imcoadd.image_selection"),
        }  # fmt: skip
        if str(get_key(node, "coadd_mode") or "").lower() == "proper":
            cards["PROPWMP"] = (self._proper_weight_policy().upper(), "imcoadd.proper_coadd_weight_map_policy")
        return cards

    def bkgsub_methods(self) -> dict:
        """Registered background routines: ``bkgsub_type`` value -> per-image callable.

        Adding a background algorithm should be this one entry plus the method itself.
        Every routine is called with the same keywords and must accept ``**kwargs`` so it
        can ignore the ones it has no use for:

            inim, outim, data, header, bkg, bkg_rms, skyval, ignore_steppy_flag,
            exclude, fov_valid

        ``data``/``header`` are the frame the caller already read (native float32);
        ``inim`` remains only as the name to report. ``exclude`` marks the pixels the
        background may NOT be estimated from (out of FOV, and on sources for a mesh
        background) and is only built for routines named ``"dynamic"``; ``fov_valid`` is
        the boolean in-field mask every routine must use to re-zero the out-of-FOV
        padding, because exact 0 is what the in-memory combine treats as "no data".

        ``individual`` is deliberately absent: it is a per-image chooser over these
        entries (`_resolve_bkgsub_type`), not a routine.
        """
        return {"constant": self._const_bkgsub, "dynamic": self._dynamic_bkgsub}

    def _default_bkgsub_type(self, skyvalues, skyval_cut: float) -> str:
        """One routine for the whole group, used when ``bkgsub_type`` is unset.

        The historical rule, unchanged: a single frame below the cut puts everyone on the
        scalar background. A missing SKYVAL no longer participates in the comparison
        rather than raising, and an all-missing group lands on dynamic, which needs none.
        """
        return "constant" if any(sv is not None and sv < skyval_cut for sv in skyvalues) else "dynamic"

    def _resolve_bkgsub_type(self, requested: str, skyval, skyval_cut: float) -> str:
        """Which routine one image gets. Only 'individual' decides per image."""
        if requested != "individual":
            return requested
        if skyval is None:
            # nothing to subtract as a scalar; let the mesh background measure it
            return "dynamic"
        # a nearly-empty sky quantises into a step-like mesh background, so prefer the
        # scalar SKYVAL there
        return "constant" if skyval < skyval_cut else "dynamic"

    def _source_mask(
        self,
        inim: str,
        header,
        fov_valid: np.ndarray | None,
        outmask: str,
        fov_mask: str | None = None,
        star_scale: float = 2.0,
        galaxy_scale: float = 2.5,
        class_star_cut: float = 0.5,
        min_radius: float = 3.0,
        min_usable: float = 20.0,
        photometry_catalog: str | None = None,
        source_image: str | None = None,
    ) -> tuple[np.ndarray, float]:
        """Detection pass, then what background estimation may use: in FOV and off-source.

        Returns the boolean usable mask, not a path: the only consumer is the background
        routine in the same loop. The durable PLIO copy is still written, and a plain
        uint8 FOV weight image is materialized only when SExtractor really runs.

        Kron ellipses are widened by ``galaxy_scale`` below ``class_star_cut`` and by
        ``star_scale`` above it; ``min_usable`` is the % of surviving pixels under which
        the background meshes are mostly interpolated and we say so.

        Takes any image, not just a single frame: the planned ``source_mask: iterative``
        re-runs this on the coadd and feeds the result back into a second bkgsub round."""
        from .utils import build_source_mask, read_mask_plio, source_ellipses_on_frame, write_mask_plio

        param_file = os.path.join(REF_DIR, "srcExt", "bkgdet.param")

        base = os.path.splitext(get_basename(inim))[0]
        catalog = os.path.join(self.path_bkgsub, f"{base}_bkgdet.cat")

        shape = (header["NAXIS2"], header["NAXIS1"])
        persist = os.path.join(collapse(self.path.imcoadd.factory.source_mask_dir, force=True), get_basename(outmask))
        if not self.overwrite:
            # persisted mask (PLIO, output dir): skip detection+painting
            valid = read_mask_plio(persist)
            if valid is not None and valid.shape == shape:
                return valid, float(100 * valid.mean())

        # reuse photometry's catalog: its DETECT_THRESH 3.0 is accepted over a 1.5 pass to save the run
        detection_override = self._detection_override()
        ellipses = None
        if photometry_catalog and os.path.exists(photometry_catalog) and not detection_override:
            ellipses = source_ellipses_on_frame(
                photometry_catalog, fits.getheader(source_image or inim), header, logger=self.logger
            )
            if ellipses is not None:
                self.logger.debug(f"{len(ellipses)} source ellipses from {get_basename(photometry_catalog)}")
        if ellipses is None:
            sex_options = {
                "-CATALOG_TYPE": "ASCII_HEAD",
                "-PARAMETERS_NAME": param_file,
                "-CHECKIMAGE_TYPE": "NONE",
            }
            sex_options.update({f"-{k}": str(v) for k, v in detection_override.items()})
            # SExtractor cannot be handed an array; this is the one caller that still
            # needs the FOV mask on disk, so it is written here and only here
            if fov_mask is not None and fov_valid is not None:
                fits.writeto(fov_mask, fov_valid.astype(np.uint8), overwrite=True)
                sex_options.update({"-WEIGHT_TYPE": "MAP_WEIGHT", "-WEIGHT_IMAGE": f"{fov_mask}", "-WEIGHT_THRESH": "0"})  # fmt: skip
            external.sextractor(
                inim,
                outcat=catalog,
                sex_options=sex_options,
                log_file=os.path.join(self.path_bkgsub, f"{base}_bkgdet_sextractor.log"),
                overwrite=self.overwrite,
                logger=self.logger,
            )
            ellipses = catalog

        sources = build_source_mask(
            ellipses,
            shape,
            star_scale=star_scale,
            galaxy_scale=galaxy_scale,
            class_star_cut=class_star_cut,
            min_radius=min_radius,
            logger=self.logger,
        )

        valid = ~sources if fov_valid is None else (fov_valid & ~sources)
        write_mask_plio(persist, valid)  # durable copy next to the config, survives factory cleanup
        usable = float(100 * valid.mean())
        self.logger.debug(f"Source mask ({usable:.1f}% usable) saved as {get_basename(persist)}")
        if usable < min_usable:
            self.logger.warning(
                f"Only {usable:.1f}% of {get_basename(inim)} is left to estimate the background on; "
                f"SExtractor will interpolate most meshes. Consider lowering the source-mask scales."
            )
        return valid, usable

    def build_fov_masks(self, resampled_images, erode_iter: int = 3) -> list[str | None]:
        """Footprint masks taken from the **pristine SWarp resamps**, stored for bkgsub.

        This has to happen the moment the resamps exist, not later at bkgsub time. The
        padding is identified as ``== 0``, and that marker only survives while nothing has
        touched the frame: convolution (or any other in-between stage) turns the padding
        into small non-zero values, after which the test finds nothing and the padding is
        silently treated as sky — which is exactly how a -6.7 ADU rim reached a coadd.
        """
        factory = self.path.imcoadd.factory
        outputs = factory.stage_images(resampled_images, "fovmask", factory.bkgsub_dir)
        self._fov_masks = []
        for inim, outmask in zip(atleast_1d(resampled_images), outputs):
            if os.path.exists(outmask) and not self.overwrite:
                # a written mask means the frame needed one; frames that needed none wrote
                # nothing and re-derive below (they pay one resamp read, nothing else)
                self._fov_masks.append(outmask)
            else:
                self._fov_masks.append(
                    outmask if self._write_fov_mask(inim, outmask, erode_iter=erode_iter) is not None else None
                )
        return self._fov_masks

    def shrink_fov_masks(self, delta_peeings, kernel_extent: float = 4.0) -> list[str | None]:
        """Shrink the stored masks by the convolution kernel's reach.

        Convolution genuinely mixes the padding inward, so the trustworthy footprint is
        smaller afterwards. ``delta_peeing`` is a FWHM in pixels and the kernel is
        Gaussian with ``sigma = delta / sqrt(8 ln 2)`` truncated at 8*sigma+1
        (imcoadd/convolve.py), so the reach is ``kernel_extent`` sigma.
        """
        from scipy.ndimage import binary_erosion

        for i, (mask, delta) in enumerate(zip(getattr(self, "_fov_masks", []), atleast_1d(delta_peeings))):
            if mask is None or not delta:
                continue
            extra = int(np.ceil(kernel_extent * float(delta) / np.sqrt(8 * np.log(2))))
            if extra < 1:
                continue
            valid = fits.getdata(mask).astype(bool)
            valid = binary_erosion(valid, np.ones((3, 3), dtype=bool), iterations=extra, border_value=0)
            fits.writeto(mask, valid.astype(np.uint8), overwrite=True)
            self.logger.debug(f"Shrank {get_basename(mask)} by {extra} px for a {delta:.2f} px kernel")
        return self._fov_masks

    def _fov_valid(self, data: np.ndarray, name: str, erode_iter: int = 3) -> np.ndarray | None:
        """Valid-pixel mask of a reprojected frame; SWarp pads out-of-FOV with 0.

        ``erode_iter`` 3x3 erosions trim the resampled footprint edge, which rings over
        LANCZOS3's support radius."""
        from scipy.ndimage import binary_erosion

        valid = data != 0
        if valid.all():
            self.logger.debug(f"No out-of-FOV pixels in {name}; skipping FOV mask")
            return None

        # border_value=1: the array bound is not an FOV edge, only the zero padding is
        valid = binary_erosion(valid, np.ones((3, 3), dtype=bool), iterations=erode_iter, border_value=1)
        self.logger.debug(f"FOV mask ({100 * valid.mean():.1f}% valid) for {name}")
        return valid

    def _write_fov_mask(self, inim: str, outmask: str, erode_iter: int = 3) -> np.ndarray | None:
        """`_fov_valid` on a frame read from disk, persisted for a later stage to reuse."""
        valid = self._fov_valid(fits.getdata(inim, memmap=False), get_basename(inim), erode_iter=erode_iter)
        if valid is None:
            return None
        fits.writeto(outmask, valid.astype(np.uint8), overwrite=True)
        self.logger.debug(f"FOV mask saved as {get_basename(outmask)}")
        return valid

    def _sex_vars(self, section: str = "imcoadd") -> dict:
        """main.sex settings under a section's sex_vars override; empty override = inherit."""
        from .utils import parse_sex_config

        keys = ("BACK_SIZE", "BACK_FILTERSIZE", "DETECT_THRESH", "DETECT_MINAREA")
        values = parse_sex_config(os.path.join(REF_DIR, "srcExt", "main.sex"), keys)
        overrides = get_key(getattr(self.config_node, section), "sex_vars") or {}
        for key in keys:
            if overrides.get(key) is not None:
                values[key] = overrides[key]
        return values

    def _background_mesh(self) -> tuple[int, int]:
        """(BACK_SIZE, BACK_FILTERSIZE) for the dynamic background."""
        values = self._sex_vars()
        return int(values["BACK_SIZE"]), int(values["BACK_FILTERSIZE"])

    def _detection_override(self) -> dict:
        """Explicit sex_vars detection keys, or {} to take the photometry catalog as detected."""
        overrides = get_key(self.config_node.imcoadd, "sex_vars") or {}
        wanted = {k: overrides.get(k) for k in ("DETECT_THRESH", "DETECT_MINAREA")}
        wanted = {k: v for k, v in wanted.items() if v is not None}
        if not wanted:
            return {}
        # a catalog already detected at these values needs no second pass
        photometry = self._sex_vars("photometry")
        if all(float(v) == float(photometry[k]) for k, v in wanted.items()):
            return {}
        return wanted

    def _guard_sky_rms_propagation(self):
        """Raise if a coadd sky-noise map is asked for; propagation is unimplemented."""
        if get_key(self.config_node.imcoadd, "output_sky_rms_map", default=False):
            raise NotImplementedError(
                "imcoadd.output_sky_rms_map: the per-frame sky-RMS models are written, but "
                "propagating them into a coadd sky-noise map (the source-free counterpart of "
                "the current weight map) is not implemented yet"
            )

    def _const_bkgsub(self, inim, outim, skyval, data=None, header=None, skyval_cut=40, fov_valid=None, **kwargs):

        if os.path.exists(outim):
            try:
                _backtype = fits.getval(outim, "BACKTYPE")
            except KeyError:
                _backtype = ""
            if _backtype.upper() == "CONSTANT":
                if not self.overwrite:
                    self.logger.info(f"Background subtraction result exists; skipping: {get_basename(outim)}")
                    return

        is_steppy = skyval < skyval_cut

        _data, _hdr = self._read_frame(inim, data, header)
        _hdr["BACKTYPE"] = ("CONSTANT", "Background subtraction type")
        # _hdr["BKG_STEP"] = (is_steppy, "SE Background can be step-like")
        _data -= skyval
        if fov_valid is not None:
            _data[~fov_valid] = 0.0  # keep out-of-FOV at 0: the coadd's validity marker
        self.logger.debug(f"Using SKYVAL: {skyval:.3f}")
        fits.writeto(outim, _data, header=_hdr, overwrite=True)

        return False  # is_steppy is False by definition for constant background subtraction

    def _dynamic_bkgsub(self, inim, outim, bkg, bkg_rms, data=None, header=None, ignore_steppy_flag=False, exclude=None, fov_valid=None, **kwargs):  # fmt: skip
        """
        Later to be refined using iterations
        """
        from .utils import estimate_background

        # from .bkg_step import step_background_check

        back_size, filter_size = self._background_mesh()
        # `exclude` holds the pixels the meshes must not see, as -WEIGHT_THRESH 0 did

        # one read: the models are built in memory, not round-tripped through check images
        _data, _hdr = self._read_frame(inim, data, header)
        bkg_data, bkg_rms_data = estimate_background(_data, mask=exclude, back_size=back_size, filter_size=filter_size)
        if get_key(self.config_node.imcoadd, "output_sky_rms_map", default=False):
            fits.writeto(bkg_rms, bkg_rms_data, overwrite=True)
        del bkg_rms_data  # do not hold a second full frame past its write
        if get_key(self.config_node.imcoadd, "output_bkg_map", default=False):
            # nothing downstream reads it; a full-frame model per input is 262 MB of
            # write and of factory space that only a human inspecting the mesh wants
            fits.writeto(bkg, bkg_data, overwrite=True)

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

        _hdr["BACKTYPE"] = ("DYNAMIC", "Background subtraction type")
        # _hdr["BKG_STEP"] = (is_steppy, "Background is step-like; likely quantization artifact")
        _data -= bkg_data
        if fov_valid is not None:
            _data[~fov_valid] = 0.0  # keep out-of-FOV at 0: the coadd's validity marker
        fits.writeto(outim, _data, header=_hdr, overwrite=True)

        # return is_steppy

    @staticmethod
    def _read_frame(inim, data, header):
        """The frame the caller already read, or read it now for a direct routine call."""
        if data is not None and header is not None:
            return data, header
        data, header = fits.getdata(inim, header=True, memmap=False)
        return np.ascontiguousarray(data, dtype=np.float32), header

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

    def _group_IMCMB(
        self, input_images: list[str], output_images: list[str] = None
    ) -> dict[tuple[str, str, str], list[list[str]]]:
        """
        Group images by their master frames (IMCMB).
        Same logic as the preprocessing grouping, but relies on header info
        instead of parsing filename as in NameHandler.get_grouped_files()
        """
        # construct zdf bundles for dict keys; cached per instance so the header reads
        # (~80 ms each over NFS) are paid once per run, not once per stage
        cache = getattr(self, "_zdf_cache", None)
        if cache is None:
            cache = self._zdf_cache = {}
        calibs = []
        for image in input_images:
            if image not in cache:
                cache[image] = get_zdf_from_header_IMCMB(image)
            calibs.append(cache[image])

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
        out_weights: list[str] | None = None,
    ) -> list[str]:
        """
        Uses self.input_images. Takes in input_images just for name carrying.

        ``out_weights`` defaults to the factory paths keyed on the bkgsub
        products, which is where the legacy routine's names land anyway. Never
        derive them from ``input_images`` directly: those may be the pristine
        inputs, and the weight maps would then be written into the directory the
        inputs were read from.
        """
        if input_images is None:
            input_images = get_key(self.config_node.imcoadd, "bkgsub_images") or self.input_images

        value_images = self.input_images  # r_p. input_images for name carrying

        st = time.time()
        self._use_gpu = False  # all([use_gpu, self.config.imcoadd.gpu, self._use_gpu])
        device_id = device_id if self._use_gpu else "CPU"

        self.logger.info(f"Start weight-map calculation")

        factory = self.path.imcoadd.factory
        # SWarp finds these by WEIGHT_SUFFIX, so they must sit beside the images they weight
        out_weights = atleast_1d(out_weights if out_weights is not None else factory.stage_images(input_images, "weight", factory.bkgsub_dir))  # fmt: skip
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
            if len(uncalculated_images) < len(group_values):
                self.logger.info(
                    f"Group {i + 1}: {len(group_values) - len(uncalculated_images)} existing weight maps "
                    f"skipped, {len(uncalculated_images)} to compute"
                )

            if uncalculated_images:
                st_image = time.time()
                with acquire_available_gpu(device_id=device_id) as acquired:
                    if acquired is None:
                        from .weight import calc_weight_with_cpu

                        calc_weight = calc_weight_with_cpu
                        self.logger.info(f"Calculate weight map with CPU [group {i + 1}/{len(groups)}]")
                        acquired = "CPU"
                        bp = self._coadd_plan()
                        zero_mask = None
                        if bp["zero"] and not bp["interpolate"]:
                            # interpolation off but bad-pixel weights still zeroed
                            mask_file, badpix = self._get_bpmask(uncalculated_images[0])
                            zero_mask = fits.getdata(mask_file) == badpix
                        calc_weight(
                            uncalculated_images,
                            d_m_file,
                            f_m_file,
                            sig_z_file,
                            sig_f_file,
                            out_names=uncalculated_outputs,
                            weight_store=bool(get_key(self.config_node.imcoadd, "persist_weight_maps", default=False)),
                            zero_mask=zero_mask,
                        )
                    else:
                        bp = self._coadd_plan()
                        if bp["zero"] and not bp["interpolate"]:
                            raise NotImplementedError(
                                "zero_badpix_weight without interpolation is CPU-only "
                                "(the GPU weight kernel is untrusted anyway); set imcoadd.gpu: False"
                            )
                        from .weight import calc_weight_with_gpu

                        calc_weight = calc_weight_with_gpu
                        self.logger.info(
                            f"Calculate weight map with GPU device {acquired} [group {i + 1}/{len(groups)}]"
                        )
                        calc_weight(
                            uncalculated_images,
                            d_m_file,
                            f_m_file,
                            sig_z_file,
                            sig_f_file,
                            acquired=acquired,
                            out_names=uncalculated_outputs,
                        )

                self.logger.debug(
                    f"Weight-map calculation (device={device_id}) for group {i} is completed in {time_diff_in_seconds(st_image)} seconds"
                )
            else:
                self.logger.info("All weight images already exist. Skipping weight map calculation")

            self.logger.info(
                f"Weight maps completed for group {i + 1}/{len(groups)} in {time_diff_in_seconds(st_loop)} seconds "
                f"({time_diff_in_seconds(st_loop, return_float=True) / len(group_values):.1f} s/image)"
            )

        return self.config_node.imcoadd.bkgsub_weight_images

    def _get_bpmask(self, image) -> tuple[str, int]:
        mask_file = PathHandler.get_bpmask(image)
        mask_header = fits.getheader(mask_file, ext=1)
        if "BADPIX" in mask_header.keys():
            badpix = mask_header["BADPIX"]
            self.logger.debug(f"BADPIX found in header. Using badpix {badpix}.")
        else:

            self.logger.warning("BADPIX not found in header. Using default value 0.", self._process_error.KeyError)
        return mask_file, badpix

    def _discard_consumed_interp(self):
        """Delete interp pairs whose reprojected products exist (`imcoadd.discard_interp`).

        The look-ahead skip regenerates nothing for these on a rerun, so the ~350 GB per
        1000-frame filter they occupy buys nothing once the resamps are on disk."""
        if not get_key(self.config_node.imcoadd, "discard_interp", default=False):
            return
        interp_images = atleast_1d(get_key(self.config_node.imcoadd, "interp_images") or [])
        method = self.config_node.imcoadd.interp_type
        freed = n = 0
        for outim in interp_images:
            sidecar = add_suffix(outim, "weight")
            if os.path.exists(outim) and self._lookahead_done(outim, method):
                freed += os.path.getsize(outim) + (os.path.getsize(sidecar) if os.path.exists(sidecar) else 0)
                os.remove(outim)
                if os.path.exists(sidecar):
                    os.remove(sidecar)
                n += 1
        if n:
            self.logger.info(f"Discarded {n} consumed interp pairs ({freed/1e9:.0f} GB freed)")

    def _discard_consumed_bkgsub_inputs(self):
        """Delete sci resamps + bkg/bkgrms models once their bkgsub product exists
        (`imcoadd.lean_factory`). Buys scratch space; a resume recomputes them."""
        if not get_key(self.config_node.imcoadd, "lean_factory", default=False):
            return
        bkgsub_images = atleast_1d(get_key(self.config_node.imcoadd, "bkgsub_images") or [])
        resampled = atleast_1d(get_key(self.config_node.imcoadd, "resampled_images") or [])
        freed = n = 0
        for resamp, bkgsub in zip(resampled, bkgsub_images):
            if not os.path.exists(bkgsub):
                continue
            # bkg/bkgrms are staged off the resamp name (stage_images suffix convention)
            doomed = [add_suffix(resamp, "bkg"), add_suffix(resamp, "bkgrms"), resamp]
            for f in doomed:
                if os.path.exists(f):
                    freed += os.path.getsize(f)
                    os.remove(f)
                    n += 1
        if n:
            self.logger.info(f"Lean factory: discarded {n} consumed bkgsub inputs/models ({freed/1e9:.0f} GB freed)")

    def _reproject_single(self, interp_im: str, sidecar: str) -> None:
        """Per-image sci+wht SWarp passes; grid pinned by CENTER MANUAL + IMAGE_SIZE, so the
        outputs are bit-identical to the batch passes (measured), and parallel singles beat
        one batch 6x because batch SWarp resamples sequentially."""
        factory = self.path.imcoadd.factory
        base = os.path.splitext(get_basename(interp_im))[0]
        self._stagger_swarp()
        for pass_type, args, use_w in (
            ("sci", ["-RESAMPLING_TYPE", "LANCZOS3"], False),
            ("wht", ["-RESAMPLING_TYPE", "NEAREST", "-WEIGHT_IMAGE", sidecar], True),
        ):
            rdir = factory.swarp_resample_dir(pass_type)
            external.swarp(
                input=[interp_im],
                output=os.path.join(os.path.dirname(rdir), f"{base}_single_coadd.fits"),
                overwrite=self.overwrite,
                center=self.center,
                resample_dir=rdir,
                coadd=False,
                log_file=os.path.join(os.path.dirname(rdir), f"{base}_swarp.log"),
                use_weight_map=use_w,
                logger=self.logger,
                swarp_args=args,
            )
            self._drop_swarp_byproduct([interp_im], pass_type)  # as it appears, not in a storm at the end
        sci = collapse(factory.resampled_images([interp_im], pass_type="sci"), force=True)
        self._manifest_note(sci, interp=str(self.config_node.imcoadd.interp_type).upper())

    def _drop_swarp_byproduct(self, swarp_inputs, pass_type: str) -> None:
        """Delete the half of a reproject-only SWarp pass that nothing may read.

        Every pass emits both a resampled image and a resampled weight, and each of these
        two passes has exactly one product, 262 MB per frame each:
        - "sci" is LANCZOS3 and is read for its **image**; its weight rings to ~0 almost
          everywhere (99%+ zeros) and `coadd_in_memory` explicitly forbids using it.
        - "wht" is NEAREST and is read for its **weight**; its image is a NEAREST-resampled
          science frame, which is not something to coadd.

        Only these two, and only reproject-only: the single-pass (`need_weights` False) and
        "bpm" rosters are left alone. Resume stays intact because neither skip check looks
        at what is deleted -- `external.swarp` checks `<base>_resamp.fits` and, for a
        weighted pass, its companion; the sci pass here is unweighted, and the wht pass is
        guarded on its weights alone by `reproject_and_coadd_with_swarp`."""
        if pass_type not in ("sci", "wht"):
            return
        images = atleast_1d(self.path.imcoadd.factory.resampled_images(swarp_inputs, pass_type=pass_type))
        doomed = images if pass_type == "wht" else [swap_ext(f, "weight.fits") for f in images]
        for f in doomed:
            if os.path.exists(f):
                os.remove(f)

    # ---- factory manifest: stat-validated option cache for intermediates ----
    # An entry is trusted only while the file's (mtime_ns, size) still match, so a file
    # regenerated behind the manifest's back is treated as unknown and falls back to a
    # header read, which repopulates the entry. Keys are paths relative to tmp_dir.

    def _manifest_load(self) -> dict:
        if getattr(self, "_manifest", None) is None:
            try:
                with open(self.path.imcoadd.factory.manifest_file) as fp:
                    self._manifest = json.load(fp)
            except (OSError, ValueError):
                self._manifest = {}
        return self._manifest

    def _manifest_flush(self) -> None:
        if getattr(self, "_manifest", None) is None:
            return
        manifest_file = self.path.imcoadd.factory.manifest_file
        os.makedirs(os.path.dirname(manifest_file), exist_ok=True)
        tmp = manifest_file + ".tmp"
        with open(tmp, "w") as fp:
            json.dump(self._manifest, fp)
        os.replace(tmp, manifest_file)

    def _manifest_key(self, path: str) -> str:
        return os.path.relpath(path, self.path.imcoadd.tmp_dir)

    def _manifest_note(self, path: str, **options) -> None:
        try:
            st = os.stat(path)
        except OSError:
            return
        self._manifest_load()[self._manifest_key(path)] = {
            "mtime_ns": st.st_mtime_ns, "size": st.st_size, **options
        }  # fmt: skip

    def _manifest_options(self, path: str) -> dict | None:
        """Recorded options for path, or None when absent or stale (stat mismatch)."""
        entry = self._manifest_load().get(self._manifest_key(path))
        if not entry:
            return None
        try:
            st = os.stat(path)
        except OSError:
            return None
        if st.st_mtime_ns != entry.get("mtime_ns") or st.st_size != entry.get("size"):
            return None
        return entry

    _swarp_launch_lock = threading.Lock()
    _swarp_last_launch = 0.0

    def _stagger_swarp(self, gap: float = 0.5) -> None:
        """Space out per-image SWarp launches so parallel singles don't hit the disk at once."""
        cls = type(self)
        with cls._swarp_launch_lock:
            now = time.time()
            wait = max(0.0, cls._swarp_last_launch + gap - now)
            cls._swarp_last_launch = now + wait
        if wait:
            time.sleep(wait)

    def _lookahead_done(self, interp_im: str, method: str) -> bool:
        """A frame is reproducible from its resamps alone: interp may have been discarded.

        True when the sci resamp and wht weight exist AND the recorded interp option
        matches the requested interp_type -- from the manifest when fresh; only a
        missing/stale entry costs a header read (INTERP card), which then heals it."""
        factory = self.path.imcoadd.factory
        sci = collapse(factory.resampled_images([interp_im], pass_type="sci"), force=True)
        wht = collapse(factory.resampled_weight_images([sci], pass_type="wht"), force=True)
        if not (os.path.exists(sci) and os.path.exists(wht)):
            return False
        entry = self._manifest_options(sci)
        if entry is not None:
            return str(entry.get("interp", "")).upper() == str(method).upper()
        try:
            ok = str(fits.getheader(sci).get("INTERP", "")).upper() == str(method).upper()
        except OSError:
            return False
        if ok:
            self._manifest_note(sci, interp=str(method).upper())
        return ok

    def weight_and_interpolate(self, input_images: list[str] | None = None) -> list[str]:
        """Fused per-group weight + interpolation; the weight map never touches disk raw.

        CPU only by design: both kernels are sub-second and the stage is NFS-bound, so
        GPU acquisition would serialize on I/O anyway.
        """
        if input_images is None:
            input_images = self.input_images
        st = time.time()
        self.logger.info("Start fused weight-map calculation + bad-pixel interpolation")

        factory = self.path.imcoadd.factory
        interp_images = factory.stage_images(input_images, "interp", factory.interp_dir)
        self.config_node.imcoadd.interp_images = interp_images

        method = self.config_node.imcoadd.interp_type
        zero_interp = bool(self._coadd_plan()["zero"])

        streamline = bool(get_key(self.config_node.imcoadd, "streamline_reprojection", default=False))
        # The reprojection tail used to run inline in the interp loop's single writer
        # thread, so the two SWarp passes of frame N blocked the kernels of frame N+1:
        # measured 15 s/frame of which read+weight+interp was ~2 s. Its own bounded pool
        # overlaps the passes with each other and with interpolation. The backpressure cap
        # keeps not-yet-discarded interp pairs from piling up when SWarp falls behind.
        tail_pool, tail_futures, n_tail = None, None, 0
        if streamline:
            from collections import deque
            from concurrent.futures import ThreadPoolExecutor

            n_tail = conservative_worker_count(len(input_images))
            tail_pool = ThreadPoolExecutor(max_workers=n_tail)
            tail_futures = deque()
            self.logger.info(f"Reprojection tail on {n_tail} workers")

        def _reproject_frame(sci_out, discard):
            sidecar = add_suffix(sci_out, "weight")
            self._reproject_single(sci_out, sidecar)
            if discard:
                os.remove(sci_out)
                os.remove(sidecar)

        def _drain_tail(keep: int = 0):
            while tail_futures and len(tail_futures) > keep:
                tail_futures.popleft().result()

        # resamp-first: a frame whose reprojected products exist needs nothing here,
        # whatever the state of its interp/weight intermediates
        todo_in, todo_out, n_lookahead, reproject_only = [], [], 0, []
        for inim, outim in zip(input_images, interp_images):
            if not self.overwrite and self._lookahead_done(outim, method):
                n_lookahead += 1
                self.logger.debug(f"Resamps exist with matching options; nothing to do for {outim}")
            elif os.path.exists(outim) and os.path.exists(add_suffix(outim, "weight")) and not self.overwrite:
                if streamline:
                    reproject_only.append(outim)
                else:
                    self.logger.debug(f"Already exists; skip generating {outim}")
            else:
                todo_in.append(inim)
                todo_out.append(outim)
        if n_lookahead:
            self.logger.info(f"{n_lookahead} frames skipped via their reprojected products")
        if reproject_only:
            self.logger.info(f"{len(reproject_only)} frames reproject-only (interp exists, resamps missing)")
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(lambda im: self._reproject_single(im, add_suffix(im, "weight")), reproject_only))

        if len(todo_in) < len(input_images):
            self.logger.info(
                f"{len(input_images) - len(todo_in)} existing fused products skipped, {len(todo_in)} to compute"
            )
        if todo_in:
            from .interpolate import weight_and_interpolate_cpu
            from .weight import _load_calibration_data

            groups = self._group_IMCMB(todo_in, todo_out)
            self.logger.info(f"{len(groups)} groups for fused weight+interpolation.")
            persist = bool(get_key(self.config_node.imcoadd, "persist_weight_maps", default=False))
            for group_id, ((z, d, f), [group_in, group_out]) in enumerate(groups.items()):
                st_group = time.time()
                mask_file, badpix = self._get_bpmask(group_in[0])
                d_m_file, f_m_file, sig_z_file, sig_f_file = PathHandler.resolve_weight_map_input_abspath([z, d, f])
                weight_store = None
                calib = None
                if persist:
                    from .weight_store import check_single_weight

                    masters = {"d": d_m_file, "f": f_m_file, "sz": sig_z_file, "sf": sig_f_file}
                    store_paths = [PathHandler.single_weight_map(im) for im in group_in]
                    weight_store = (store_paths, masters)
                    n_reusable = sum(check_single_weight(p, masters) for p in store_paths)
                    if n_reusable:
                        self.logger.info(
                            f"Group {group_id + 1}: {n_reusable}/{len(group_in)} single weight maps reusable"
                        )
                    if n_reusable == len(group_in):
                        calib = "skip"  # every frame verified: masters never touched
                if calib != "skip":
                    calib = _load_calibration_data(d_m_file, f_m_file, sig_z_file, sig_f_file)
                else:
                    calib = None
                post_frame = None
                if streamline:
                    discard = bool(get_key(self.config_node.imcoadd, "discard_interp", default=False))

                    def post_frame(sci_out, _discard=discard):
                        _drain_tail(keep=2 * n_tail)
                        tail_futures.append(tail_pool.submit(_reproject_frame, sci_out, _discard))

                weight_and_interpolate_cpu(
                    group_in,
                    mask_file,
                    group_out,
                    calib,
                    weight_store=weight_store,
                    method=method,
                    badpix=badpix,
                    zero_interp_weight=zero_interp,
                    logger=self.logger,
                    post_frame=post_frame,
                )
                self.logger.info(
                    f"Weight+interp completed for group {group_id + 1}/{len(groups)} in "
                    f"{time_diff_in_seconds(st_group)} seconds "
                    f"({time_diff_in_seconds(st_group, return_float=True) / len(group_in):.1f} s/image)"
                )
        else:
            self.logger.info("All fused weight+interp products already exist. Skipping")

        if tail_pool is not None:
            try:
                _drain_tail()  # the manifest below must record every reprojection
            finally:
                tail_pool.shutdown()

        self.config_node.imcoadd.bkgsub_weight_images = [add_suffix(im, "weight") for im in interp_images]
        self._manifest_flush()
        self.logger.info(f"Fused weight+interp completed in {time_diff_in_seconds(st)} seconds")
        return interp_images

    def apply_bpmask(
        self,
        input_images: list[str] | None = None,
        device_id=None,
        use_gpu: bool = True,
        weight_images: list[str] | None = None,
    ) -> list[str]:
        if input_images is None:
            input_images = get_key(self.config_node.imcoadd, "bkgsub_images") or self.input_images
        st = time.time()

        self._use_gpu = all([use_gpu, self.config_node.imcoadd.gpu, self._use_gpu])
        device_id = device_id if self._use_gpu else "CPU"

        self.logger.info("Start the interpolation for bad pixels")

        factory = self.path.imcoadd.factory
        interp_images = factory.stage_images(input_images, "interp", factory.interp_dir)
        self.config_node.imcoadd.interp_images = interp_images

        # bpmask_array, header = fits.getdata(self.config.preprocess.bpmask_file, header=True)

        method = self.config_node.imcoadd.interp_type
        weight = self._coadd_plan()["need_weights"]  # derived: outputs or internal consumers
        # Where this run wrote them, not wherever a sibling of the input happens to sit:
        # reproject-first writes weights to the factory, and a stale one next to the input
        # would be read in silence.
        weight_of = dict(zip(input_images, weight_images)) if weight_images is not None else {}
        zero_interp = bool(self._coadd_plan()["zero"])

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

        if 0 < len(uncalculated_images) < len(input_images):
            self.logger.info(
                f"{len(input_images) - len(uncalculated_images)} existing interp products skipped, "
                f"{len(uncalculated_images)} to compute"
            )
        # interpolate
        if not uncalculated_images:
            self.logger.info("No images to interpolate. Skipping")
        else:
            groups = self._group_IMCMB(uncalculated_images, calculated_outputs)
            self.logger.info(f"{len(groups)} groups for bad pixel interpolation.")
            self.logger.debug(f"apply_bpmask groups: {groups}")

            for group_id, ((z, d, f), [input_images, output_images]) in enumerate(groups.items()):
                mask_file, badpix = self._get_bpmask(input_images[0])

                with acquire_available_gpu(device_id=device_id) as acquired:
                    if acquired is None:
                        from .interpolate import interpolate_masked_pixels_cpu

                        interpolate_masked_pixels = interpolate_masked_pixels_cpu
                        self.logger.info(f"Interpolate masked pixels with CPU [group {group_id + 1}/{len(groups)}]")
                    else:
                        from .interpolate import interpolate_masked_pixels_subprocess

                        interpolate_masked_pixels = interpolate_masked_pixels_subprocess
                        self.logger.info(
                            f"Interpolate masked pixels with GPU device {acquired} [group {group_id + 1}/{len(groups)}]"
                        )

                    group_weights = [weight_of[f] for f in input_images] if weight_of else weight
                    st_group = time.time()
                    try:
                        interpolate_masked_pixels(
                            input_images,
                            mask_file,
                            output_images,
                            method=method,
                            badpix=badpix,
                            weight=group_weights,
                            zero_interp_weight=zero_interp,
                            device=acquired,
                            **({"logger": self.logger} if acquired is None else {}),
                        )
                    except Exception as e:
                        # The GPU subprocess fails for reasons that have nothing to do with
                        # the data — a cupy/CUDA toolkit mismatch is the usual one — and the
                        # numba kernel next door gives the same answer. Preprocess already
                        # falls back this way; imcoadd used to let it kill the whole coadd.
                        if acquired is None:
                            raise
                        from .interpolate import interpolate_masked_pixels_cpu

                        self.logger.warning(f"GPU interpolation failed, falling back to CPU: {e}")
                        interpolate_masked_pixels_cpu(
                            input_images,
                            mask_file,
                            output_images,
                            method=method,
                            badpix=badpix,
                            weight=group_weights,
                            zero_interp_weight=zero_interp,
                            device=None,
                            logger=self.logger,
                        )
                self.logger.info(
                    f"Interpolation completed for group {group_id + 1}/{len(groups)} in "
                    f"{time_diff_in_seconds(st_group)} seconds "
                    f"({time_diff_in_seconds(st_group, return_float=True) / len(input_images):.1f} s/image)"
                )

            self.logger.info(
                f"Interpolation for bad pixels is completed in {time_diff_in_seconds(st)} seconds "
                f"({time_diff_in_seconds(st, return_float=True)/len(input_images):.1f} s/image)"
            )

        # advance the target images of interest
        self.images_to_coadd = interp_images
        return interp_images

    def zpscale(self, input_images: list[str] | None = None, write_headers: bool = True) -> list[str]:
        """
        Store the value in header as FLXSCALE, and use it in coadding.
        Keep FSCALE_KEYWORD = FLXSCALE in SWarp config.
        The headers of the last processed images are modified.
        write_headers=False stamps only the in-memory snapshot (legacy needs the
        file cards for SWarp's FSCALE_KEYWORD; the in-memory combine does not).
        With imcoadd.zpscale off it scrubs instead of stamps, so both routines
        reach the same state through one decision.
        """
        if input_images is None:
            input_images = self.images_to_coadd
        if not get_key(self.config_node.imcoadd, "zpscale", default=True):
            # Nothing scales the pixels (combine gets flxscales=False, SWarp gets
            # -FSCALE_KEYWORD NOFSCALE), so the snapshot must not carry a factor either.
            for hdr in self.input_headers:
                hdr.pop("FLXSCALE", None)  # stale photometry-era cards must not aggregate
            self.logger.debug("zpscale off; stale FLXSCALE scrubbed from the snapshot")
            return input_images
        st = time.time()
        zpvalues = self.input_headers.values(self.zpkey)
        for zp in zpvalues:
            if zp is None:
                msg = f"{self.zpkey} is None for {input_images[i]}"
                self.logger.error(msg, self._process_error.PreviousStageError)
                raise self._process_error.PreviousStageError(msg)
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
            if write_headers:
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

    def prepare_convolution(self, input_images: list[str] | None = None, weight: bool = False):
        """
        This is ad-hoc. Change it to convolve after resampling and take
        advantage of uniform pixel scale.

        ``weight`` must match the flag `run_convolution` is called with: the two together
        decide whether a weight companion exists beside every conv file or beside none.
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
            factory = self.path.imcoadd.factory
            self.config_node.imcoadd.conv_files = factory.stage_images(input_images, "conv", factory.conv_dir)

            # Get peeings for convolution. Read them off the snapshot, not the files:
            # under reproject-first these inputs are SWarp resamp products, and PEEING is
            # not in the swarp COPY_KEYWORDS list. The pixel scale is unchanged by the
            # resampling, so the singles' PEEING is still the right value.
            peeings = self.input_headers.values("PEEING")
            if len(peeings) != len(input_images):
                peeings = [None] * len(input_images)
            peeings = [p if p is not None else fits.getheader(f).get("PEEING") for f, p in zip(input_images, peeings)]
            if any(p is None for p in peeings):
                missing = [get_basename(f) for f, p in zip(input_images, peeings) if p is None]
                self.logger.error(f"No PEEING for {missing[:3]}; cannot match seeing", self._process_error.KeyError)
                raise self._process_error.KeyError(f"No PEEING for {len(missing)} input(s); cannot match seeing")

            # max_peeing = np.max(peeings)
            max_peeing = float(np.max(peeings))
            target_seeing = get_key(self.config_node.imcoadd, "target_seeing")
            if isinstance(target_seeing, (int, float)) and not isinstance(target_seeing, bool):
                target_peeing = target_seeing / collapse(self.path.pixscale, raise_error=True)
                if target_peeing < max_peeing:
                    # convolving *down* is not possible; the yml documents this fallback
                    self.logger.warning(
                        f"target_seeing {target_seeing} is below the worst input seeing "
                        f"({max_peeing * collapse(self.path.pixscale, raise_error=True):.3f}); using that instead"
                    )
                    target_peeing = max_peeing
                self._max_peeing = target_peeing
            else:
                self._max_peeing = max_peeing
            delta_peeings = [self._calc_delta_peeing(peeing) for peeing in peeings]
            self.delta_peeings = delta_peeings
            self.logger.debug(f"PEEINGs: {peeings}")

            for i, delta_peeing in enumerate(delta_peeings):
                # symlink images to conv output folder that don't need convolution
                if delta_peeing is None:
                    force_symlink(input_images[i], self.config_node.imcoadd.conv_files[i])
                    if weight and self._coadd_plan()["need_weights"]:
                        # Only when the weights genuinely travel with the conv files, i.e.
                        # when `run_convolution(weight=True)` writes the other half of the
                        # set for the frames that ARE convolved. Unconditionally it built
                        # a half-set -- a companion for the frames needing no convolution
                        # and none for the rest -- which no consumer can use, and in
                        # reproject-first nothing reads these at all (the combine takes
                        # its weights from the wht pass via `resampled_images`).
                        force_symlink(
                            self._resolve_weight_companion(input_images[i]),
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

        if not image_list:
            # Every frame already sits at the target seeing, so prepare_convolution
            # symlinked all of them and there is nothing left to convolve. Always true for
            # a single input (it defines the maximum), and for any group of equal PEEING.
            conv_files = self.config_node.imcoadd.conv_files
            self.logger.info("Every input already matches the target seeing; nothing to convolve")
            self.images_to_coadd = conv_files
            return conv_files

        with acquire_available_gpu(device_id=device_id) as acquired:

            if acquired is None:
                from .convolve import convolve_fft_cpu

                convolve_fft = convolve_fft_cpu
                self.logger.info(f"Convolution with CPU")
            else:

                from .convolve import convolve_fft_subprocess

                convolve_fft = convolve_fft_subprocess
                self.logger.info(f"Convolution with GPU device {acquired}")

            output = convolve_fft(
                image_list,
                outim_list,
                kernels=kernels,
                device=acquired,
                apply_edge_mask=weight,
                method=method,
                delta_peeing=delta_peeing_list,
            )

            if weight:
                # resolve the inputs' companions (naming differs between stage products and
                # SWarp resamp outputs); the outputs are named to match what
                # prepare_convolution symlinks for the frames it skips
                weight_list = [self._resolve_weight_companion(f) for f, k in zip(input_images, self.kernels) if k is not None]  # fmt: skip
                outwim_list = [add_suffix(f, "weight") for f, k in zip(self.config_node.imcoadd.conv_files, self.kernels) if k is not None]  # fmt: skip
                self.logger.debug(f"weight_list {weight_list}")
                self.logger.debug(f"outwim_list {outwim_list}")

                if not all([os.path.exists(f) for f in atleast_1d(weight_list)]):
                    self.logger.error(f"Weight map not found for all images.", self._process_error.FileNotFoundError)
                    raise self._process_error.FileNotFoundError(f"Weight map not found for all images.")

                # compute
                convolve_fft(
                    weight_list,
                    outwim_list,
                    kernels=kernels,
                    device=acquired,
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

    def _resolve_weight_companion(self, image: str) -> str:
        """Weight map belonging to *image*, under either naming convention in play.

        Stage products carry theirs as ``<stem>_weight.fits`` (the SWarp WEIGHT_SUFFIX);
        SWarp's own resampled outputs use ``<stem>.weight.fits`` instead.

        A **sci-pass resamp is the exception**: its weight is the NEAREST one from the wht
        pass, not the LANCZOS3 file sitting beside it under the second naming convention.
        That one rings to ~0 almost everywhere (99%+ zeros) and `coadd_in_memory` forbids
        it outright, so returning it -- which is what the plain fallback did, silently,
        for every convolve-on run -- handed the seeing-match a weight map nothing else in
        the pipeline is allowed to use. `_drop_swarp_byproduct` now deletes it too."""
        factory = self.path.imcoadd.factory
        candidates = []
        if os.path.dirname(image) == factory.swarp_resample_dir("sci"):
            candidates.append(collapse(factory.resampled_weight_images([image], pass_type="wht"), force=True))
        candidates += [add_suffix(image, "weight"), swap_ext(image, "weight.fits")]
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return candidate
        self.logger.error(f"No weight map found for {get_basename(image)}", self._process_error.FileNotFoundError)
        raise self._process_error.FileNotFoundError(f"No weight map found for {image}")

    def _calc_delta_peeing(self, peeing):
        # clamped: a target below the worst input is already corrected upstream, this
        # only absorbs float noise on the frame that defines the maximum
        delta_peeing = np.sqrt(max(self._max_peeing**2 - peeing**2, 0.0))
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
        weight_images: list[str] | None = None,
    ) -> str | list[str]:
        """Run SWarp. ``coadd=True`` produces a single coadd image;
        ``coadd=False`` only reprojects each input and returns the resampled paths.

        ``weight_images`` names the input weight maps explicitly; without it SWarp
        finds them by ``WEIGHT_SUFFIX`` next to each input image."""
        st = time.time()
        action = "coadding" if coadd else "reprojecting"
        self.logger.info(f"Start to run swarp for {action} images")

        if input_images is None:
            input_images = self.images_to_coadd
        self.logger.debug(f"input_images: {input_images}")

        swarp_options_override_from_config = get_key(self.config_node.imcoadd, "swarp_options_override", default=[])
        swarp_options_override = swarp_options_override_from_config + swarp_options_override
        if not get_key(self.config_node.imcoadd, "zpscale", default=True):
            # zpscale off: stale FLXSCALE cards on the files must not flux-scale the combine
            swarp_options_override = swarp_options_override + ["-FSCALE_KEYWORD", "NOFSCALE"]
        if swarp_options_override:
            self.logger.warning(f"SWarp options override: {swarp_options_override}")

        # Write target images to a text file
        self.path_imagelist = os.path.join(self.path.imcoadd.tmp_dir, "images_to_coadd.txt")
        with open(self.path_imagelist, "w") as f:
            for inim in input_images:
                f.write(f"{inim}\n")

        self.logger.debug(f"Total Exptime: {self.input_headers.total_exptime}")

        sci_resampling = ["-RESAMPLING_TYPE", "LANCZOS3"]
        if not self._coadd_plan()["need_weights"]:
            # no weight consumer anywhere: single pass, no wht division
            self._run_swarp("", coadd=coadd, swarp_args=sci_resampling + swarp_options_override)
        else:
            self._run_swarp(
                "sci",
                coadd=coadd,
                swarp_args=sci_resampling + swarp_options_override,
                use_weight_map=False,
            )  # Disable weight in the sci pass
            if not coadd:
                self._drop_swarp_byproduct(input_images, "sci")
            # The wht pass is skipped on its weights alone, the same shape as the bpm guard
            # below: `external.swarp` would also demand the NEAREST-resampled science that
            # `_drop_swarp_byproduct` deletes, and re-resample every frame on every rerun.
            factory = self.path.imcoadd.factory
            wht_predicted = atleast_1d(
                factory.resampled_weight_images(
                    atleast_1d(factory.resampled_images(input_images, pass_type="sci")), pass_type="wht"
                )
            )
            if not coadd and not self.overwrite and all(os.path.exists(w) for w in wht_predicted):
                self.logger.info(f"wht pass outputs already exist ({len(wht_predicted)} weights), skipping")
            else:
                self._run_swarp(
                    "wht",
                    coadd=coadd,
                    swarp_args=["-RESAMPLING_TYPE", "NEAREST"] + swarp_options_override,
                    weight_images=weight_images,
                )
                if not coadd:
                    self._drop_swarp_byproduct(input_images, "wht")

        # conservative policy: LANCZOS3 bpm masks, needed with or without weights
        factory = self.path.imcoadd.factory
        masks_predicted = atleast_1d(
            factory.resampled_weight_images(
                atleast_1d(factory.resampled_images(input_images, pass_type="bpm")), pass_type="bpm"
            )
        )
        bp_policy = self._coadd_plan()["policy"]
        if bp_policy == "conservative" and not self.overwrite and all(os.path.exists(m) for m in masks_predicted):
            # checked before get_bpmask: resolving 1000 bpmasks costs ~20 min
            self.logger.info(f"bpm pass outputs already exist ({len(masks_predicted)} masks), skipping")
        elif bp_policy == "conservative":
            # bpmask_file = self.config.preprocess.bpmask_file
            per_image = atleast_1d(PathHandler.get_bpmask(input_images))
            if len(per_image) != len(input_images):
                per_image = per_image * len(input_images)
            by_mask: dict[str, list[str]] = {}
            for inim, mfile in zip(input_images, per_image):
                by_mask.setdefault(mfile, []).append(inim)
            self.logger.info(f"bpm pass over {len(by_mask)} distinct bpmask(s)")
            for k, (bpmask_file, group_frames) in enumerate(by_mask.items()):
                bpmask_inverted = 1 - fits.getdata(bpmask_file)
                bpmask_inverted_file = self.path.imcoadd.factory.bpmask_inverted(bpmask_file)
                fits.writeto(bpmask_inverted_file, bpmask_inverted, overwrite=True)
                self.logger.debug(f"Inverted bpmask saved as {bpmask_inverted_file}")
                group_list = os.path.join(self.path.imcoadd.tmp_dir, f"images_bpm_{k}.txt")
                with open(group_list, "w") as fp:
                    fp.write("\n".join(group_frames) + "\n")
                # BADPIX=1 means bad, MAP_WEIGHT means >0 is good: SWarp needs the inverse.
                # The resampling follows the sci pass rather than being pinned here: a mask
                # resampled differently from the image it describes would not line up with it.
                args = ["-WEIGHT_IMAGE", bpmask_inverted_file] + sci_resampling
                self._run_swarp("bpm", coadd=coadd, swarp_args=args + swarp_options_override,
                                input_list=group_list)  # fmt: skip

        if coadd:
            self._guard_sky_rms_propagation()
            self._update_header()
            self.logger.info(f"Running swarp is completed in {time_diff_in_seconds(st)} seconds")
            return self.config_node.imcoadd.coadd_image

        # reproject-only branch: predict resampled output paths (named by SWarp from its inputs)
        pass_type = "sci" if self._coadd_plan()["need_weights"] else ""
        resampled = atleast_1d(self.path.imcoadd.factory.resampled_images(input_images, pass_type=pass_type))
        self.config_node.imcoadd.resampled_images = resampled
        self._save_single_weight_products(resampled)
        self.images_to_coadd = resampled
        self.logger.info(f"SWarp reprojection completed in {time_diff_in_seconds(st)} seconds")
        return resampled

    def _save_single_weight_products(self, resampled: list[str]) -> None:
        """Keep each frame's resampled weight beside its single, as a product not a scratch file.

        Off by default (`imcoadd.output_single_weight_map`): the factory copies SWarp wrote
        are the working ones and get discarded with the rest of the factory."""
        if not get_key(self.config_node.imcoadd, "output_single_weight_map", default=False):
            return
        from .interpolate import write_weight_int16

        sources = atleast_1d(self.path.imcoadd.factory.resampled_weight_images(resampled, pass_type="wht"))
        targets = atleast_1d(self.path.weight)
        if not (len(sources) == len(targets) == len(atleast_1d(resampled))):
            self.logger.warning("Resampled weights do not map 1:1 onto the inputs; not saved as products")
            return
        n = 0
        for src, dst in zip(sources, targets):
            if not os.path.exists(src):
                continue
            with fits.open(src, memmap=True) as hdul:
                write_weight_int16(dst, hdul[0].data, hdul[0].header)
            n += 1
        self.logger.info(f"Saved {n} resampled weight maps beside their singles")

    def _propagated_bpmasks(self) -> list[str] | None:
        """Per-frame resampled bad-pixel masks from the bpm pass, or None if it did not run.

        These narrow the per-frame validity the combine counts under
        badpix_reprojection_policy 'conservative' (LANCZOS3-support rejection). The
        footprint product itself always exists; this only decides whether kernel-touched
        pixels count towards it.
        """
        if self._coadd_plan()["policy"] != "conservative":
            return None
        resampled = atleast_1d(get_key(self.config_node.imcoadd, "resampled_images") or [])
        masks = atleast_1d(self.path.imcoadd.factory.resampled_weight_images(resampled, pass_type="bpm"))
        missing = [m for m in masks if not os.path.exists(m)]
        if not masks or missing:
            self.logger.warning(f"conservative policy set but the bpm pass left no masks (e.g. {missing[:2]})")
            return None
        return masks

    def _run_swarp(
        self,
        type: Literal["sci", "wht", "bpm"] = "",
        coadd=True,
        swarp_args=None,
        use_weight_map: bool = True,
        weight_images: list[str] | None = None,
        input_list: str | None = None,
    ) -> str:
        """Pass type='' for no weight. Returns the SWarp resample directory."""
        if weight_images:
            # @file, never a comma-joined argument: ~1000 paths overflow SWarp's option
            # buffer (SIGABRT). Order must match the input imagelist.
            weight_list = os.path.join(os.path.dirname(self.path_imagelist), f"weights_to_coadd_{type or 'all'}.txt")
            with open(weight_list, "w") as fp:
                fp.write("\n".join(atleast_1d(weight_images)) + "\n")
            swarp_args = (swarp_args or []) + ["-WEIGHT_IMAGE", f"@{weight_list}"]

        # the factory is the single authority for the resamp location (config-scoped);
        # a locally built path here is how SWarp once wrote where nothing looked
        factory = self.path.imcoadd.factory
        resample_dir = factory.swarp_resample_dir(type)
        working_dir = os.path.dirname(resample_dir)  # created by external.swarp's makedirs(resample_dir)

        log_file = os.path.join(working_dir, "_".join([self.config_node.name, type, "swarp.log"]))

        if type == "":
            output_file = self.config_node.imcoadd.coadd_image  # output to output_dir directly
        else:
            # output to tmp dir, and then selectively move to output_dir
            output_file = os.path.join(working_dir, get_basename(self.config_node.imcoadd.coadd_image))

        external.swarp(
            input=input_list or self.path_imagelist,
            output=output_file,
            overwrite=self.overwrite,
            center=self.center,
            resample_dir=resample_dir,
            coadd=coadd,
            log_file=log_file,
            logger=self.logger,
            use_weight_map=self._coadd_plan()["need_weights"] and use_weight_map,
            swarp_args=swarp_args,
        )

        # only the coadd branch produces output_file to move
        if coadd:
            if type == "sci":
                shutil.move(output_file, self.config_node.imcoadd.coadd_image)
            elif type == "wht":
                shutil.move(
                    add_suffix(output_file, "weight"),
                    add_suffix(self.config_node.imcoadd.coadd_image, "weight"),
                )
            elif type == "bpm":
                # legacy: SWarp's own combine produced the summed good-pixel coverage
                shutil.move(
                    add_suffix(output_file, "weight"),
                    add_suffix(self.config_node.imcoadd.coadd_image, "footprint"),
                )

        return resample_dir

    # local filesystems worth staging onto; anything else (nfs/cifs/tmpfs/overlay/fuse) is
    # either the problem we are escaping or too small/volatile to hold a combine
    _LOCAL_FSTYPES = {"ext2", "ext3", "ext4", "xfs", "btrfs", "zfs", "f2fs", "reiserfs"}

    @staticmethod
    def _fstype_of(path: str) -> tuple[str, str]:
        """(mount point, fstype) of the filesystem holding *path*, longest prefix wins."""
        best = ("", "")
        real = os.path.realpath(path)
        try:
            with open("/proc/mounts") as fp:
                entries = [ln.split()[:3] for ln in fp if len(ln.split()) >= 3]
        except OSError:
            return best
        for _dev, mnt, fstype in entries:
            if (real == mnt or real.startswith(mnt.rstrip("/") + "/")) and len(mnt) > len(best[0]):
                best = (mnt, fstype)
        return best

    def _pick_combine_scratch(self, files) -> str | None:
        """`combine_scratch: auto` -- a local disk to stage onto, or None to stay put.

        Only fires when the inputs actually sit on a network filesystem and the combine is
        big enough for the strided reads to hurt (`combine_lock_threshold` frames, the
        same line that decides a combine is worth serializing). Nightly-scale stacks fall
        under it and are untouched.

        Why it is worth a full copy: the combine reads every input once per strip, and on
        NFS those reads are latency bound. Measured on UDS 2026-08-14 -- m475 (614 frames,
        NFS, 3 strips) took 14292 s while m850 (649 frames, local, **6** strips) took
        3269 s: twice the passes, 4.4x faster. m400 was worse still, forced to 54 strips by
        a momentary memory shortage, and after 7 h it was reading at 1 MB/s and had to be
        killed. Staging is one sequential pass in; everything after it is local.

        Candidates come from /proc/mounts rather than config, so no path list can go stale;
        `/` is excluded (filling the system disk is its own outage) as is the filesystem
        the inputs are already on."""
        n_frames = len({f for _g, f in files})
        if n_frames < int(self._coadd_plan()["combine_lock_threshold"]):
            return None
        src_mnt, src_fstype = self._fstype_of(os.path.dirname(files[0][1]))
        if src_fstype in self._LOCAL_FSTYPES:
            self.logger.debug(f"combine_scratch auto: inputs already local on {src_mnt} ({src_fstype})")
            return None

        need = sum(os.path.getsize(f) for _g, f in files if os.path.exists(f)) * 1.1
        best = None
        seen_dev = set()
        try:
            with open("/proc/mounts") as fp:
                entries = [ln.split()[:3] for ln in fp if len(ln.split()) >= 3]
        except OSError:
            return None
        for _dev, mnt, fstype in entries:
            # never the system disk, never a home directory: /home is often the same
            # physical disk as a scratch mount anyway, and filling either is an outage
            if fstype not in self._LOCAL_FSTYPES or mnt == "/" or mnt.startswith(("/home", "/root", "/boot")):
                continue
            try:
                dev = os.stat(mnt).st_dev  # one entry per physical filesystem, not per mount
                if dev in seen_dev:
                    continue
                seen_dev.add(dev)
                root = os.path.join(mnt, "pipeline_coadd_scratch")
                os.makedirs(root, exist_ok=True)
                free = shutil.disk_usage(mnt).free
            except OSError as e:
                # worth saying: a big local disk whose ROOT is not writable by this user is
                # invisible here, and that is a provisioning fix, not a code one
                self.logger.debug(f"combine_scratch auto: {mnt} ({fstype}) unusable -- {e}")
                continue
            if free > need and (best is None or free > best[1]):
                best = (root, free)
        if best is None:
            self.logger.warning(
                f"combine_scratch auto: inputs are on {src_fstype} ({src_mnt}) and no local "
                f"filesystem has the {need/1e9:.0f} GB needed; combining over the network"
            )
            return None
        self.logger.info(
            f"combine_scratch auto: inputs on {src_fstype} ({src_mnt}); staging "
            f"{need/1e9:.0f} GB to {best[0]} ({best[1]/1e12:.1f} TB free)"
        )
        return best[0]

    def _stage_for_combine(self, groups: dict[str, list[str] | None]):
        """Copy the combine inputs to local scratch so the strided reads never touch NFS.

        One sequential NFS pass in, all random access local. Subdir per group: the wht and
        bpm weight companions share basenames. Returns (remapped groups, cleanup)."""
        scratch = get_key(self.config_node.imcoadd, "combine_scratch")
        files = [(g, f) for g, lst in groups.items() if lst for f in lst]
        if not files:
            return groups, lambda: None
        if str(scratch).lower() == "auto":
            scratch = self._pick_combine_scratch(files)
        if not scratch:
            return groups, lambda: None

        need = sum(os.path.getsize(f) for _, f in files)
        free = shutil.disk_usage(scratch).free
        if need * 1.05 > free:
            self.logger.warning(
                f"combine_scratch {scratch}: need {need/1e9:.0f} GB, only {free/1e9:.0f} GB free; combining from NFS"
            )
            return groups, lambda: None

        stem = os.path.splitext(get_basename(self.config_node.info.file))[0]
        base = os.path.join(scratch, "imcoadd_staged", stem)
        st = time.time()
        self.logger.info(f"Staging {len(files)} files ({need/1e9:.0f} GB) to {base}")
        from concurrent.futures import ThreadPoolExecutor

        def _copy(item):
            group, src = item
            dst = os.path.join(base, group, os.path.basename(src))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
            return src, dst

        with ThreadPoolExecutor(max_workers=2) as pool:
            mapping = dict(pool.map(_copy, files))
        self.logger.info(
            f"Staged in {time_diff_in_seconds(st)} seconds "
            f"({need/1e9/max(time.time()-st, 1):.2f} GB/s sequential from NFS)"
        )
        remapped = {g: ([mapping[f] for f in lst] if lst else lst) for g, lst in groups.items()}
        return remapped, lambda: shutil.rmtree(base, ignore_errors=True)

    def coadd_in_memory(
        self,
        input_images: list[str] | None = None,
        device_id=None,
        weight_images: list[str] | None = None,
    ) -> str:
        """Dispatcher to numpy/cupy and mean/median backends.
        ``imcoadd.coadd_mode`` picks the combine algorithm; for ``mean``, the
        weighted variant kicks in when ``imcoadd.weight_map`` is set and the
        NEAREST-resampled weight maps (``tmp/wht/resamp/<base>.weight.fits``)
        produced by ``reproject_and_coadd_with_swarp`` exist for every input."""
        if input_images is None:
            input_images = self.images_to_coadd
        self._guard_sky_rms_propagation()

        if device_id is not None:
            self.coadd_with_cupy(input_images, device_id=device_id)
            return self.config_node.imcoadd.coadd_image

        plan = self._coadd_plan()
        weighting = plan["weighting"]
        policy = plan["policy"]
        self.logger.info(f"Coadd weighting: {weighting}; badpix policy: {policy}")
        if weighting == "pixelwise" and policy == "off" and self._coadd_plan()["zero"]:
            self.logger.info(
                "pixel-wise weighting with zeroed bad-pixel weights: bad pixels cannot vote "
                "regardless of policy 'off' (a zero-weight vote is no vote); set "
                "zero_badpix_weight: False to let interpolated pixels vote"
            )

        wht_maps = None
        if self._coadd_plan()["need_weights"]:
            # NEAREST-resampled weights live next to the wht pass output; the
            # LANCZOS3 companions next to the sci resamp ring to ~0 almost
            # everywhere (99%+ zeros) and must NOT be used.
            wht_dir = self.path.imcoadd.factory.swarp_resample_dir("wht")
            if weight_images is not None:
                candidates = atleast_1d(weight_images)
            else:
                # named after what SWarp resampled, not after the later bkgsub products
                resampled = get_key(self.config_node.imcoadd, "resampled_images") or input_images
                candidates = atleast_1d(
                    self.path.imcoadd.factory.resampled_weight_images(resampled, pass_type="wht")
                )
            if all(os.path.exists(w) for w in candidates):
                wht_maps = candidates
            else:
                missing = [w for w in candidates if not os.path.exists(w)][:3]
                self.logger.warning(
                    f"NEAREST-resampled weight maps not found in {wht_dir} (e.g. {missing}); "
                    f"pixel-wise weighting / 1px hole masking unavailable for this run."
                )

        weights = None
        if weighting == "pixelwise":
            weights = wht_maps
        elif weighting == "global":
            skysigs = self.input_headers.values("SKYSIG")
            n_missing = sum(1 for s in skysigs if not s)
            if n_missing:
                self.logger.warning(f"SKYSIG missing on {n_missing}/{len(skysigs)} frames; weighting those 1.0")
            weights = [1.0 / float(s) ** 2 if s else 1.0 for s in skysigs]

        if policy == "conservative":
            masks = self._propagated_bpmasks()
        elif policy == "1px" and weighting != "pixelwise":
            masks = wht_maps  # NEAREST resamp: each hole is a single sub-eps pixel
        else:
            masks = None  # pixel-wise weights already exclude holes via calc.WEIGHT_EPS

        if str(self.config_node.imcoadd.coadd_mode).lower() == "proper":
            # one sequential read per frame and no strips: staging buys nothing here
            return self.coadd_proper_with_numpy(input_images, holes=masks)

        var_maps = wht_maps if weighting != "pixelwise" else None
        stage_wht = weights if weighting == "pixelwise" else None
        staged, cleanup = self._stage_for_combine({"sci": input_images, "wht": stage_wht, "bpm": masks})
        input_images, masks = staged["sci"], staged["bpm"]
        if weighting == "pixelwise":
            weights = staged["wht"]

        mode = self.config_node.imcoadd.coadd_mode
        match_swarp_size = bool(get_key(self.config_node.imcoadd, "match_swarp_size", default=True))
        # combines on one filesystem only divide its bandwidth: serialize per st_dev,
        # and lease planned stack bytes so concurrent (cross-fs) combines cannot each
        # size their strips against the same MemAvailable. Multi-epoch only: nightly
        # coadds are small and must never queue behind a deep combine.
        from ..services.combine_lock import CombineSlot, NullSlot

        anchor = os.path.dirname(collapse(atleast_1d(input_images)[0], force=True))
        # lock only high-demand combines: the threshold (config) replaces the old
        # is_multi_epoch gate -- nightly-scale stacks fall under it naturally
        slot_ctx = (
            CombineSlot(anchor, logger=self.logger)
            if len(atleast_1d(input_images)) >= plan["combine_lock_threshold"]
            else NullSlot()
        )
        try:
            with slot_ctx as slot:
                if mode == "mean":
                    slot.lease(4 * 110_000_000 * 8)  # sum/norm/count/gain accumulators, ~3.5 GB
                    self.coadd_with_numpy(
                        input_images,
                        weights=weights,
                        masks=masks,
                        var_maps=var_maps,
                        match_swarp_size=match_swarp_size,
                        write_weight=plan["output_weight_map"],
                        write_footprint=plan["output_footprint"],
                    )
                elif mode == "clipped":
                    slot.lease(6 * 110_000_000 * 8)  # two-pass accumulators, ~5 GB
                    self.coadd_clipped_with_numpy(
                        input_images,
                        weights=weights,
                        masks=masks,
                        var_maps=var_maps,
                        match_swarp_size=match_swarp_size,
                        write_weight=plan["output_weight_map"],
                        write_footprint=plan["output_footprint"],
                    )
                elif mode == "median":
                    from ..services.combine_lock import memory_headroom_bytes
                    from .calc import plan_median_memory
                    from .utils import _parse_swarp_image_size

                    # lease what the model says will actually be allocated, not a bound
                    reserved = slot.reserved_bytes
                    grid_w, grid_h = _parse_swarp_image_size(os.path.join(REF_DIR, "7dt.swarp"))
                    budget = int(0.3 * memory_headroom_bytes(reserved))
                    _, planned = plan_median_memory(len(atleast_1d(input_images)), grid_w, grid_h, budget)
                    slot.lease(planned)
                    self.coadd_median_with_numpy(
                        input_images,
                        weights=weights,
                        masks=masks,
                        match_swarp_size=match_swarp_size,
                        reserved_bytes=reserved,
                        var_maps=var_maps,
                        write_weight=plan["output_weight_map"],
                        write_footprint=plan["output_footprint"],
                    )
                else:
                    raise ValueError(f"Invalid coadd mode: {mode!r} (expected 'mean', 'median' or 'clipped')")
        finally:
            cleanup()
        return self.config_node.imcoadd.coadd_image

    def coadd_proper_with_numpy(self, input_images: list[str], holes: list[str] | None = None) -> str:
        """Zackay & Ofek proper coaddition; imcoadd.proper_coadd_weight_map_policy picks the weight product."""
        from ..services.combine_lock import CombineSlot, NullSlot
        from .proper import proper_coadd_numpy

        plan = self._coadd_plan()
        policy = self._proper_weight_policy()
        coadd_image = self.config_node.imcoadd.coadd_image
        anchor = os.path.dirname(collapse(atleast_1d(input_images)[0], force=True))
        slot_ctx = (
            CombineSlot(anchor, logger=self.logger)
            if len(atleast_1d(input_images)) >= plan["combine_lock_threshold"]
            else NullSlot()
        )
        with slot_ctx as slot:
            slot.lease(5 * 110_000_000 * 8)  # numerator + share accumulators + final FFT pair, ~4.4 GB
            return proper_coadd_numpy(
                input_images,
                output_path=coadd_image,
                coadd_header=self.input_headers.coadd_header,
                peeings=self._proper_peeings(input_images),
                skysigs=self.input_headers.values("SKYSIG"),
                flxscales=self._combine_flxscales(),
                weight_map_policy=policy,
                weight_output=add_suffix(coadd_image, "weight") if policy != "off" else False,
                footprint_output=add_suffix(coadd_image, "footprint") if plan["output_footprint"] else False,
                psf_output=add_suffix(coadd_image, "psf"),
                holes=holes,
                match_swarp_size=bool(get_key(self.config_node.imcoadd, "match_swarp_size", default=True)),
                logger=self.logger,
            )

    def _proper_weight_policy(self) -> str:
        """Validated imcoadd.proper_coadd_weight_map_policy."""
        from .proper import WEIGHT_MAP_POLICIES

        raw = get_key(self.config_node.imcoadd, "proper_coadd_weight_map_policy", default="white-noise")
        policy = str(raw or "off").lower().replace("_", "-")
        if policy not in WEIGHT_MAP_POLICIES:
            raise self._process_error.ValueError(
                f"Invalid imcoadd.proper_coadd_weight_map_policy: {raw!r} (expected one of {WEIGHT_MAP_POLICIES})"
            )
        return policy

    def _proper_peeings(self, input_images: list[str]) -> list[float]:
        """Per-frame PSF FWHM in pixels; the homogenized target when convolution ran."""
        n = len(atleast_1d(input_images))
        if self.config_node.imcoadd.convolve and getattr(self, "_max_peeing", None):
            return [float(self._max_peeing)] * n
        peeings = self.input_headers.values("PEEING")
        if len(peeings) != n or any(p is None for p in peeings):
            missing = [name for name, p in zip(self.input_headers.names, peeings) if p is None]
            self.logger.error(
                f"No PEEING for {missing[:3]}; proper coadd needs a per-frame PSF",
                self._process_error.KeyError,
            )
            raise self._process_error.KeyError(f"No PEEING for {len(missing)} input(s); proper coadd needs a per-frame PSF")
        return [float(p) for p in peeings]

    def _validate_proper_mode(self):
        """Fail fast on option combinations the Fourier-domain combine cannot honor."""
        routine = str(self.config_node.imcoadd.coadd_routine or "")
        is_direct = "direct" in routine.lower()
        if "reproject-first" not in routine.lower() and not is_direct:
            raise self._process_error.ValueError(
                f"coadd_mode 'proper' requires coadd_routine 'reproject-first' or 'direct', not {routine!r}"
            )
        plan = self._coadd_plan()
        if plan["weighting"] == "pixelwise":
            raise self._process_error.ValueError(
                "coadd_mode 'proper' is incompatible with pixel-wise weighting; use 'global' or False"
            )
        if not plan["interpolate"] and self._proper_requires_interpolation:
            raise self._process_error.ValueError(
                "coadd_mode 'proper' requires interpolate_badpix: True (a Fourier-domain vote cannot skip pixels)"
            )
        self._proper_weight_policy()
        if plan["weighting"] == "off":
            self.logger.info("coadd_weighting has no effect under 'proper': frames are inverse-variance weighted by construction")  # fmt: skip

    def _coadd_plan(self) -> dict:
        """Resolve the run plan: bad-pixel handling, weighting, outputs, derived needs.

        Active dispatcher, not a registry: `need_weights` decides whether intermediate
        weight maps exist at all (outputs OR internal consumers force them), which in
        turn shapes the fused/standalone prep, the SWarp pass roster, and the combine.
        Legacy keys are read as fallback when the new key is absent; permanently
        renamed configs are the expected long-term state (manual conversion).
        """
        node = self.config_node.imcoadd

        def opt(new, old, default):
            v = get_key(node, new)
            return get_key(node, old, default=default) if v is None else v

        interpolate = bool(opt("interpolate_badpix", "apply_bpmask", False))
        zero = bool(opt("zero_badpix_weight", "zero_interp_weight", True))
        policy = str(opt("badpix_reprojection_policy", "bpmask_policy", "1px") or "off").lower()
        policy = {"false": "off", "none": "off", "no": "off"}.get(policy, policy)
        if policy not in ("off", "1px", "conservative"):
            raise ValueError(f"Invalid badpix_reprojection_policy: {policy!r} ('off', '1px' or 'conservative')")

        weighting = (
            str(get_key(node, "coadd_weighting", default="global") or "off").lower().replace("-", "").replace("_", "")
        )
        weighting = {"false": "off", "none": "off", "no": "off"}.get(weighting, weighting)
        if weighting not in ("off", "global", "pixelwise"):
            raise ValueError(f"Invalid imcoadd.coadd_weighting: {weighting!r} (False, 'global' or 'pixel-wise')")

        output_weight_map = bool(opt("output_weight_map", "weight_map", True))
        output_footprint = bool(get_key(node, "output_footprint", default=True))
        # intermediate weight maps exist iff something consumes them; requesting
        # pixel-wise weighting or the 1px channel FORCES generation, so the old
        # "pixel-wise without weight maps" conflict can no longer be expressed
        need_weights = output_weight_map or weighting == "pixelwise" or policy == "1px"

        if policy == "1px" and not zero:
            raise NotImplementedError(
                "badpix_reprojection_policy '1px' with zero_badpix_weight: False has no "
                "information channel (dual weight maps / bpmask-only reprojection are "
                "deliberately unimplemented); zero the weights or use 'conservative'"
            )

        return {
            "interpolate": interpolate,
            "zero": zero,
            "policy": policy,
            "weighting": weighting,
            "output_weight_map": output_weight_map,
            "output_footprint": output_footprint,
            "need_weights": need_weights,
            "combine_lock_threshold": int(get_key(node, "combine_lock_threshold", default=50)),
        }

    def _combine_flxscales(self):
        """zpscale's snapshot values; with zpscale off, the stale photometry-era FLXSCALE
        cards on the singles must not scale anything -- False disables scaling."""
        if get_key(self.config_node.imcoadd, "zpscale", default=True):
            return self.input_headers.values("FLXSCALE")
        return False

    def coadd_with_numpy(
        self,
        input_images: list[str],
        weights: list[str] | None = None,
        masks: list[str] | None = None,
        match_swarp_size: bool = True,
        var_maps: list[str] | None = None,
        write_weight: bool = True,
        write_footprint: bool = True,
    ) -> str:
        return mean_coadd_numpy(
            input_images,
            output_path=self.config_node.imcoadd.coadd_image,
            coadd_header=self.input_headers.coadd_header,
            weights=weights,
            weight_output=add_suffix(self.config_node.imcoadd.coadd_image, "weight") if write_weight else False,
            footprint_output=(
                add_suffix(self.config_node.imcoadd.coadd_image, "footprint") if write_footprint else False
            ),
            masks=masks,
            flxscales=self._combine_flxscales(),
            match_swarp_size=match_swarp_size,
            logger=self.logger,
        )

    def coadd_clipped_with_numpy(
        self,
        input_images: list[str],
        weights: list[str] | None = None,
        masks: list[str] | None = None,
        match_swarp_size: bool = True,
        var_maps: list[str] | None = None,
        write_weight: bool = True,
        write_footprint: bool = True,
    ) -> str:
        return clipped_mean_coadd_numpy(
            input_images,
            output_path=self.config_node.imcoadd.coadd_image,
            coadd_header=self.input_headers.coadd_header,
            weights=weights,
            weight_output=add_suffix(self.config_node.imcoadd.coadd_image, "weight") if write_weight else False,
            footprint_output=(
                add_suffix(self.config_node.imcoadd.coadd_image, "footprint") if write_footprint else False
            ),
            masks=masks,
            flxscales=self._combine_flxscales(),
            match_swarp_size=match_swarp_size,
            var_maps=var_maps,
            logger=self.logger,
        )

    # ---- median backend ----

    def coadd_median_with_numpy(
        self,
        input_images: list[str],
        weights: list[str] | None = None,
        masks: list[str] | None = None,
        match_swarp_size: bool = True,
        chunk_h: int | None = None,  # None: auto-sized from idle memory (see calc._auto_chunk_h)
        reserved_bytes: int = 0,
        var_maps: list[str] | None = None,
        write_weight: bool = True,
        write_footprint: bool = True,
    ) -> str:
        return median_coadd_numpy(
            input_images,
            output_path=self.config_node.imcoadd.coadd_image,
            coadd_header=self.input_headers.coadd_header,
            weights=weights,
            weight_output=add_suffix(self.config_node.imcoadd.coadd_image, "weight") if write_weight else False,
            footprint_output=(
                add_suffix(self.config_node.imcoadd.coadd_image, "footprint") if write_footprint else False
            ),
            masks=masks,
            flxscales=self._combine_flxscales(),
            match_swarp_size=match_swarp_size,
            chunk_h=chunk_h,
            reserved_bytes=reserved_bytes,
            var_maps=var_maps,
            logger=self.logger,
        )

    def coadd_with_cupy(self, input_images: list[str], device_id) -> str:
        raise NotImplementedError("GPU coadd_with_cupy is not implemented yet")

    def register_coadd_qa(self):
        """image_qa row, its dependency rows, and the header-derived QA update for the coadd."""
        if not self.is_connected:
            return

        coadd_image = self.config_node.imcoadd.coadd_image
        if not (coadd_image and os.path.exists(coadd_image)):
            self.logger.warning(f"No coadd image to register in image_qa: {coadd_image}")
            return

        if self.process_status_id is not None:
            self.qa_id = self.create_image_qa_data(coadd_image, process_status_id=self.process_status_id)
            self.create_image_qa_dependencies(coadd_image, self.qa_id)

        # Update QA data from header if database is connected
        if self.qa_id is not None:
            qa_data = ImageQATable.from_file(
                coadd_image,
                process_status_id=self.process_status_id,
            )
            self.image_qa.update_data(self.qa_id, **qa_data.to_dict())

        self.sync_config_dependencies()

    def plot_coadd_image(self):
        coadd_img = self.config_node.imcoadd.coadd_image
        basename = os.path.basename(coadd_img)
        # same as the selection plot: figure_dir is a list for multi-epoch inputs
        path_to_plot = os.path.join(collapse(self.path.figure_dir, force=True), swap_ext(basename, "jpg"))
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
