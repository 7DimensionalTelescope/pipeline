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

        self.register_coadd_qa()

        self.update_progress(SCIPROCESS_REGISTRY.completed_progress("coadd"), "imcoadd-completed")

    def reproject_first_coadd_routine(self, use_gpu: bool = False, device_id=None):
        """SWarp for reprojection only; the combine step runs in memory via
        ``coadd_in_memory``, which picks mean/median from ``imcoadd.coadd_mode``."""
        self._use_gpu = all([use_gpu, self.config_node.imcoadd.gpu, self._use_gpu])

        do_zpscale = bool(get_key(self.config_node.imcoadd, "zpscale", default=True))
        optional_steps = (
            int(bool(self.config_node.imcoadd.weight_map))
            + int(bool(self.config_node.imcoadd.apply_bpmask))
            + int(bool(self.config_node.imcoadd.joint_wcs))
            + int(bool(self.config_node.imcoadd.convolve))
            + int(do_zpscale)
        )
        TOTAL_STEPS = 4 + optional_steps  # reproject + bkgsub + coadd + plot + optionals
        step = 0

        self.initialize()

        images = self.input_images
        weight_images = None
        do_weight = bool(self.config_node.imcoadd.weight_map)
        do_bpmask = bool(self.config_node.imcoadd.apply_bpmask)
        if do_weight and do_bpmask:
            # fused: the weight map is handed to interpolation in memory, one read and one
            # write per image instead of three reads and two writes
            images = self.weight_and_interpolate(images)
            weight_images = [add_suffix(im, "weight") for im in images]
            step += 1
            self.update_progress(
                SCIPROCESS_REGISTRY.step_progress("coadd", step, TOTAL_STEPS),
                "imcoadd-calculate-weight-map-completed",
            )
            step += 1
            self.update_progress(
                SCIPROCESS_REGISTRY.step_progress("coadd", step, TOTAL_STEPS), "imcoadd-apply-bpmask-completed"
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
                    SCIPROCESS_REGISTRY.step_progress("coadd", step, TOTAL_STEPS),
                    "imcoadd-calculate-weight-map-completed",
                )

            # Bad pixel interpolation
            if do_bpmask:
                images = self.apply_bpmask(images, device_id=device_id, weight_images=weight_images)
                if weight_images is not None:
                    # hand the interp sidecars onward: they carry the zeroed bad-pixel holes
                    weight_images = [add_suffix(im, "weight") for im in images]
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
        images = self.reproject_and_coadd_with_swarp(images, coadd=False, weight_images=weight_images)
        # while the padding is still exactly zero -- see build_fov_masks
        self.build_fov_masks(images)
        step += 1
        self.update_progress(
            SCIPROCESS_REGISTRY.step_progress("coadd", step, TOTAL_STEPS), "imcoadd-reproject-completed"
        )

        # PSF homogenization
        if self.config_node.imcoadd.convolve:
            self.prepare_convolution(images)
            images = self.run_convolution(images, device_id=device_id)
            self.shrink_fov_masks(self.delta_peeings)
            step += 1
            self.update_progress(
                SCIPROCESS_REGISTRY.step_progress("coadd", step, TOTAL_STEPS),
                "imcoadd-run-convolution-completed",
            )

        # Background subtraction on reprojected (+ optionally convolved) frames
        images = self.bkgsub(
            images,
            mask_out_of_fov=True,
            mask_sources=get_key(self.config_node.imcoadd, "source_mask", default=True),
            fov_masks=getattr(self, "_fov_masks", None),
        )
        step += 1
        self.update_progress(SCIPROCESS_REGISTRY.step_progress("coadd", step, TOTAL_STEPS), "imcoadd-bkgsub-completed")

        # Flux zero-point scaling (writes FLXSCALE to header in place)
        if do_zpscale:
            self.zpscale(images)
            step += 1
            self.update_progress(
                SCIPROCESS_REGISTRY.step_progress("coadd", step, TOTAL_STEPS), "imcoadd-zpscale-completed"
            )

        # In-memory coaddition (mean/median selected by imcoadd.coadd_mode)
        self.coadd_in_memory(images, device_id=device_id)
        step += 1
        self.update_progress(SCIPROCESS_REGISTRY.step_progress("coadd", step, TOTAL_STEPS), "imcoadd-coadd-completed")

        # Plot coadd image
        self.plot_coadd_image()
        step += 1
        self.update_progress(SCIPROCESS_REGISTRY.step_progress("coadd", step, TOTAL_STEPS), "imcoadd-plot-completed")

        self.register_coadd_qa()

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
        self.apply_sanity_filter_and_report(current_process=COADD_SPEC, overwrite=self.overwrite)
        if not self.input_images:
            self.logger.error("No Input for ImCoadd", CoaddError.EmptyInputAfterSanityRejection)
            raise CoaddError.EmptyInputAfterSanityRejection("No Input for ImCoadd")
        # if rejected, let the input remain so that a rerun has a change to reevaluate SANITY

        self.select_input_images()  # may drop inputs, so it precedes the snapshot and resync
        # Single read of every kept header; all aggregates/coadd_header live on this snapshot.
        self.input_headers = InputHeaderSet.from_files(self.input_images)
        self.input_headers.selection_metrics = getattr(self, "_selection_meta", {})
        self.input_headers.coadd_provenance = self._coadd_provenance()

        self._recreate_pathhandler_instance()  # resync
        self.config_node.imcoadd.input_images = self.input_images

        self.zpkey = self.config_node.imcoadd.zp_key or ZP_KEY
        # self.ic_keys = IC_KEYS

        # self.define_paths(working_dir=self.config.path.path_processed)

        self.input_headers.check_uniqueness(["OBJECT", "FILTER"], self.logger)
        self.center = self.input_headers.deprojection_center
        self.logger.debug(f"Deprojection center: {self.center}")

        # Output coadd image file name
        self.config_node.imcoadd.coadd_image = self.path.imcoadd.coadd_image
        self.config_node.input.coadd_image = self.config_node.imcoadd.coadd_image
        self.logger.debug(f"Coadd Image: {self.config_node.imcoadd.coadd_image}")

        self.logger.info(f"Initialization for ImCoadd is completed")

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
        # Prefer masks built earlier from the pristine resamps (`build_fov_masks`): once a
        # stage has run in between, the padding is no longer exactly 0 and cannot be found.
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

        counts = {name: types.count(name) for name in sorted(set(types))}
        self.logger.info(f"Start background subtraction (bkgsub_type={requested!r}): {counts}")
        if any_dynamic:
            self.config_node.imcoadd.bkg_images = bkg_images
            self.config_node.imcoadd.bkg_rms_images = bkg_rms_images
        else:
            if get_key(self.config_node.imcoadd, "bkg_images"):
                self.config_node.imcoadd.bkg_images = None
            if get_key(self.config_node.imcoadd, "bkg_rms_images"):
                self.config_node.imcoadd.bkg_rms_images = None

        def _bkgsub_one(i, inim, outim, bkg, bkg_rms, skyvalue, fov_mask, src_mask, btype):
            st_loop = time.time()
            # the FOV mask serves every routine (it re-zeros the padding afterwards);
            # the source mask only means something to a mesh background
            if fov_mask is None:
                fov_valid = None
            elif fov_masks is not None:
                fov_valid = fits.getdata(fov_mask).astype(bool)
            else:
                fov_valid = self._write_fov_mask(inim, fov_mask)
                fov_mask = fov_mask if fov_valid is not None else None
            weight_image = (
                self._write_source_mask(inim, fov_mask, fov_valid, src_mask)
                if (src_mask is not None and btype == "dynamic")
                else fov_mask
            )
            is_steppy = methods[btype](
                inim,
                outim,
                bkg=bkg,
                bkg_rms=bkg_rms,
                skyval=skyvalue,
                ignore_steppy_flag=ignore_steppy_flag,
                weight_image=weight_image,
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
        jobs = list(zip(input_images, bkgsub_images, bkg_images, bkg_rms_images, skyvalues,
                        fov_mask_images, source_mask_images, types))  # fmt: skip
        n_workers = conservative_worker_count(len(jobs))
        if n_workers <= 1:
            for i, job in enumerate(jobs):
                _bkgsub_one(i, *job)
        else:
            self.logger.info(f"Background subtraction with {n_workers} workers")
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = [pool.submit(_bkgsub_one, i, *job) for i, job in enumerate(jobs)]
                for f in futures:
                    f.result()

        self.logger.info(
            f"Background subtraction is completed in {time_diff_in_seconds(st)} ({time_diff_in_seconds(st, return_float=True)/len(input_images):.1f} s/image)"
        )

        self.images_to_coadd = bkgsub_images
        return bkgsub_images

    def _coadd_provenance(self) -> dict[str, tuple]:
        """Config options that change the coadd, as coadd header cards."""
        node = self.config_node.imcoadd
        shown = lambda value: "NONE" if value is None or value is False else value  # noqa: E731
        interp = get_key(node, "interp_type") if get_key(node, "apply_bpmask") else None
        return {
            "COADDRTN": (shown(get_key(node, "coadd_routine")), "imcoadd.coadd_routine"),
            "COADDMOD": (shown(get_key(node, "coadd_mode")), "imcoadd.coadd_mode"),
            "ZPSCALE":  (bool(get_key(node, "zpscale")), "imcoadd.zpscale"),
            "INTERP":   (shown(interp), "imcoadd.interp_type"),
            "CONVOLVE": (shown(get_key(node, "convolve")), "imcoadd.convolve"),
            "JOINTWCS": (bool(get_key(node, "joint_wcs")), "imcoadd.joint_wcs"),
            "IMGSELEC": (shown(get_key(node, "image_selection")), "imcoadd.image_selection"),
        }  # fmt: skip

    def bkgsub_methods(self) -> dict:
        """Registered background routines: ``bkgsub_type`` value -> per-image callable.

        Adding a background algorithm should be this one entry plus the method itself.
        Every routine is called with the same keywords and must accept ``**kwargs`` so it
        can ignore the ones it has no use for:

            inim, outim, bkg, bkg_rms, skyval, ignore_steppy_flag, weight_image, fov_valid

        ``weight_image`` is the mask of pixels the background may be estimated from and is
        only built for routines named ``"dynamic"``; ``fov_valid`` is the boolean
        in-field mask every routine must use to re-zero the out-of-FOV padding, because
        exact 0 is what the in-memory combine treats as "no data".

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

    def _write_source_mask(
        self,
        inim: str,
        fov_mask: str | None,
        fov_valid: np.ndarray | None,
        outmask: str,
        star_scale: float = 2.0,
        galaxy_scale: float = 3.0,
        class_star_cut: float = 0.5,
        min_radius: float = 3.0,
        min_usable: float = 20.0,
        detect_params: tuple[str, ...] = ("X_IMAGE", "Y_IMAGE", "A_IMAGE", "B_IMAGE", "THETA_IMAGE", "KRON_RADIUS", "CLASS_STAR"),  # fmt: skip
    ) -> str:
        """Detection pass, then what background estimation may use: in FOV and off-source.

        Kron ellipses are widened by ``galaxy_scale`` below ``class_star_cut`` and by
        ``star_scale`` above it; ``min_usable`` is the % of surviving pixels under which
        the background meshes are mostly interpolated and we say so.

        Takes any image, not just a single frame: the planned ``source_mask: iterative``
        re-runs this on the coadd and feeds the result back into a second bkgsub round."""
        from .utils import build_source_mask

        # own .param file so the hash-locked ref/srcExt presets stay untouched
        param_file = os.path.join(self.path_bkgsub, "bkgdet.param")
        with open(param_file, "w") as f:
            f.write("\n".join(detect_params) + "\n")

        base = os.path.splitext(get_basename(inim))[0]
        catalog = os.path.join(self.path_bkgsub, f"{base}_bkgdet.cat")
        sex_options = {
            "-CATALOG_TYPE": "ASCII_HEAD",
            "-PARAMETERS_NAME": param_file,
            "-CHECKIMAGE_TYPE": "NONE",
        }
        if fov_mask is not None:
            sex_options.update({"-WEIGHT_TYPE": "MAP_WEIGHT", "-WEIGHT_IMAGE": f"{fov_mask}", "-WEIGHT_THRESH": "0"})
        external.sextractor(
            inim,
            outcat=catalog,
            sex_options=sex_options,
            log_file=os.path.join(self.path_bkgsub, f"{base}_bkgdet_sextractor.log"),
            overwrite=self.overwrite,
            logger=self.logger,
        )

        header = fits.getheader(inim)
        shape = (header["NAXIS2"], header["NAXIS1"])
        sources = build_source_mask(
            catalog,
            shape,
            star_scale=star_scale,
            galaxy_scale=galaxy_scale,
            class_star_cut=class_star_cut,
            min_radius=min_radius,
            logger=self.logger,
        )

        valid = ~sources if fov_valid is None else (fov_valid & ~sources)
        fits.writeto(outmask, valid.astype(np.uint8), overwrite=True)
        usable = 100 * valid.mean()
        self.logger.debug(f"Source mask ({usable:.1f}% usable) saved as {get_basename(outmask)}")
        if usable < min_usable:
            self.logger.warning(
                f"Only {usable:.1f}% of {get_basename(inim)} is left to estimate the background on; "
                f"SExtractor will interpolate most meshes. Consider lowering the source-mask scales."
            )
        return outmask

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
        self._fov_masks = [
            outmask if self._write_fov_mask(inim, outmask, erode_iter=erode_iter) is not None else None
            for inim, outmask in zip(atleast_1d(resampled_images), outputs)
        ]
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

    def _write_fov_mask(self, inim: str, outmask: str, erode_iter: int = 3) -> np.ndarray | None:
        """Valid-pixel mask of a reprojected frame; SWarp pads out-of-FOV with 0.

        ``erode_iter`` 3x3 erosions trim the resampled footprint edge, which rings over
        LANCZOS3's support radius."""
        from scipy.ndimage import binary_erosion

        valid = fits.getdata(inim) != 0
        if valid.all():
            self.logger.debug(f"No out-of-FOV pixels in {get_basename(inim)}; skipping FOV mask")
            return None

        # border_value=1: the array bound is not an FOV edge, only the zero padding is
        valid = binary_erosion(valid, np.ones((3, 3), dtype=bool), iterations=erode_iter, border_value=1)
        fits.writeto(outmask, valid.astype(np.uint8), overwrite=True)
        self.logger.debug(f"FOV mask ({100 * valid.mean():.1f}% valid) saved as {get_basename(outmask)}")
        return valid

    def _const_bkgsub(self, inim, outim, skyval, skyval_cut=40, fov_valid=None, **kwargs):

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

        with fits.open(inim, memmap=True) as hdul:
            _data = hdul[0].data
            _hdr = hdul[0].header
            _hdr["BACKTYPE"] = ("CONSTANT", "Background subtraction type")
            # _hdr["BKG_STEP"] = (is_steppy, "SE Background can be step-like")
            _data -= skyval
            if fov_valid is not None:
                _data[~fov_valid] = 0.0  # keep out-of-FOV at 0: the coadd's validity marker
            self.logger.debug(f"Using SKYVAL: {skyval:.3f}")
            fits.writeto(outim, _data, header=_hdr, overwrite=True)

        return False  # is_steppy is False by definition for constant background subtraction

    def _dynamic_bkgsub(self, inim, outim, bkg, bkg_rms, ignore_steppy_flag=False, weight_image=None, fov_valid=None, **kwargs):  # fmt: skip
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
        if weight_image is not None:
            # SExtractor drops zero-weight pixels from the background meshes
            sex_options.update({"-WEIGHT_TYPE": "MAP_WEIGHT", "-WEIGHT_IMAGE": f"{weight_image}", "-WEIGHT_THRESH": "0"})  # fmt: skip
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
            if fov_valid is not None:
                _data[~fov_valid] = 0.0  # keep out-of-FOV at 0: the coadd's validity marker
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

            self.logger.warning("BADPIX not found in header. Using default value 0.", CoaddError.KeyError)
        return mask_file, badpix

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
        zero_interp = bool(get_key(self.config_node.imcoadd, "zero_interp_weight", default=True))

        todo_in, todo_out = [], []
        for inim, outim in zip(input_images, interp_images):
            if os.path.exists(outim) and os.path.exists(add_suffix(outim, "weight")) and not self.overwrite:
                self.logger.debug(f"Already exists; skip generating {outim}")
            else:
                todo_in.append(inim)
                todo_out.append(outim)

        if len(todo_in) < len(input_images):
            self.logger.info(
                f"{len(input_images) - len(todo_in)} existing fused products skipped, {len(todo_in)} to compute"
            )
        if todo_in:
            from .interpolate import weight_and_interpolate_cpu
            from .weight import _load_calibration_data

            groups = self._group_IMCMB(todo_in, todo_out)
            self.logger.info(f"{len(groups)} groups for fused weight+interpolation.")
            for group_id, ((z, d, f), [group_in, group_out]) in enumerate(groups.items()):
                st_group = time.time()
                mask_file, badpix = self._get_bpmask(group_in[0])
                d_m_file, f_m_file, sig_z_file, sig_f_file = PathHandler.resolve_weight_map_input_abspath([z, d, f])
                calib = _load_calibration_data(d_m_file, f_m_file, sig_z_file, sig_f_file)
                weight_and_interpolate_cpu(
                    group_in,
                    mask_file,
                    group_out,
                    calib,
                    method=method,
                    badpix=badpix,
                    zero_interp_weight=zero_interp,
                    logger=self.logger,
                )
                self.logger.info(
                    f"Weight+interp completed for group {group_id + 1}/{len(groups)} in "
                    f"{time_diff_in_seconds(st_group)} seconds "
                    f"({time_diff_in_seconds(st_group, return_float=True) / len(group_in):.1f} s/image)"
                )
        else:
            self.logger.info("All fused weight+interp products already exist. Skipping")

        self.config_node.imcoadd.bkgsub_weight_images = [add_suffix(im, "weight") for im in interp_images]
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
        weight = self.config_node.imcoadd.weight_map  # boolean flag for generating weight map
        # Where this run wrote them, not wherever a sibling of the input happens to sit:
        # reproject-first writes weights to the factory, and a stale one next to the input
        # would be read in silence.
        weight_of = dict(zip(input_images, weight_images)) if weight_images is not None else {}
        zero_interp = bool(get_key(self.config_node.imcoadd, "zero_interp_weight", default=True))

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
                self.logger.error(f"No PEEING for {missing[:3]}; cannot match seeing", CoaddError.KeyError)
                raise CoaddError.KeyError(f"No PEEING for {len(missing)} input(s); cannot match seeing")

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
                    if self.config_node.imcoadd.weight_map:
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
                    self.logger.error(f"Weight map not found for all images.", CoaddError.FileNotFoundError)
                    raise CoaddError.FileNotFoundError(f"Weight map not found for all images.")

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
        SWarp's own resampled outputs use ``<stem>.weight.fits`` instead."""
        for candidate in (add_suffix(image, "weight"), swap_ext(image, "weight.fits")):
            if os.path.exists(candidate):
                return candidate
        self.logger.error(f"No weight map found for {get_basename(image)}", CoaddError.FileNotFoundError)
        raise CoaddError.FileNotFoundError(f"No weight map found for {image}")

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
            sci_resampling = ["-RESAMPLING_TYPE", "LANCZOS3"]
            self._run_swarp(
                "sci",
                coadd=coadd,
                swarp_args=sci_resampling + swarp_options_override,
                use_weight_map=False,
            )  # Disable weight in the sci pass
            self._run_swarp(
                "wht",
                coadd=coadd,
                swarp_args=["-RESAMPLING_TYPE", "NEAREST"] + swarp_options_override,
                weight_images=weight_images,
            )

            if self.config_node.imcoadd.propagate_mask:
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

    def _propagated_bpmasks(self) -> list[str] | None:
        """Per-frame resampled bad-pixel masks from the bpm pass, or None if it did not run.

        These narrow the per-frame validity the combine counts, so `propagate_mask` decides
        whether a pixel a bad column touched still counts towards the footprint. It is a
        separate question from the footprint itself, which always exists.
        """
        if not self.config_node.imcoadd.propagate_mask:
            return None
        resampled = atleast_1d(get_key(self.config_node.imcoadd, "resampled_images") or [])
        masks = atleast_1d(self.path.imcoadd.factory.resampled_weight_images(resampled, pass_type="bpm"))
        missing = [m for m in masks if not os.path.exists(m)]
        if not masks or missing:
            self.logger.warning(f"propagate_mask is on but the bpm pass left no masks (e.g. {missing[:2]})")
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

        working_dir = os.path.join(self.path.imcoadd.tmp_dir, type)
        resample_dir = os.path.join(working_dir, "resamp")
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
                # legacy: SWarp's own combine produced the summed good-pixel coverage
                shutil.move(
                    add_suffix(output_file, "weight"),
                    self.path.imcoadd.factory.coadd_footprint_image,
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

        weights = None
        if self.config_node.imcoadd.weight_map:
            # NEAREST-resampled weights live next to the wht pass output; the
            # LANCZOS3 companions next to the sci resamp ring to ~0 almost
            # everywhere (99%+ zeros) and must NOT be used.
            wht_dir = self.path.imcoadd.factory.swarp_resample_dir("wht")
            # named after what SWarp resampled, not after the later bkgsub products
            resampled = get_key(self.config_node.imcoadd, "resampled_images") or input_images
            candidates = atleast_1d(self.path.imcoadd.factory.resampled_weight_images(resampled, pass_type="wht"))
            if all(os.path.exists(w) for w in candidates):
                weights = candidates
            else:
                missing = [w for w in candidates if not os.path.exists(w)][:3]
                self.logger.warning(
                    f"weight_map=True but NEAREST-resampled weight maps not found "
                    f"in {wht_dir} (e.g. {missing}); coadding unweighted and writing a "
                    f"frame-count weight map instead."
                )

        masks = self._propagated_bpmasks()

        mode = self.config_node.imcoadd.coadd_mode
        if mode == "mean":
            self.coadd_with_numpy(input_images, weights=weights, masks=masks)
        elif mode == "median":
            self.coadd_median_with_numpy(input_images, weights=weights, masks=masks)
        else:
            raise ValueError(f"Invalid coadd mode: {mode!r} (expected 'mean' or 'median')")
        return self.config_node.imcoadd.coadd_image

    def coadd_with_numpy(
        self,
        input_images: list[str],
        weights: list[str] | None = None,
        masks: list[str] | None = None,
        match_swarp_size: bool = True,
    ) -> str:
        return mean_coadd_numpy(
            input_images,
            output_path=self.config_node.imcoadd.coadd_image,
            coadd_header=self.input_headers.coadd_header,
            weights=weights,
            weight_output=self.path.imcoadd.factory.coadd_weight_image,
            footprint_output=self.path.imcoadd.factory.coadd_footprint_image,
            masks=masks,
            flxscales=self.input_headers.values("FLXSCALE"),
            match_swarp_size=match_swarp_size,
            logger=self.logger,
        )

    # ---- median backend ----

    def coadd_median_with_numpy(
        self,
        input_images: list[str],
        weights: list[str] | None = None,
        masks: list[str] | None = None,
        match_swarp_size: bool = True,
        chunk_h: int = 128,  # row strips; 142 frames x 128 x 10200 x 4 B ~ 750 MB per strip
    ) -> str:
        return median_coadd_numpy(
            input_images,
            output_path=self.config_node.imcoadd.coadd_image,
            coadd_header=self.input_headers.coadd_header,
            weights=weights,
            weight_output=self.path.imcoadd.factory.coadd_weight_image,
            footprint_output=self.path.imcoadd.factory.coadd_footprint_image,
            masks=masks,
            flxscales=self.input_headers.values("FLXSCALE"),
            match_swarp_size=match_swarp_size,
            chunk_h=chunk_h,
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
