import os
import time
import shutil
import warnings

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
from ..config.utils import get_key, get_or_set_key
from ..utils import (
    collapse,
    add_suffix,
    time_diff_in_seconds,
    get_basename,
    atleast_1d,
    swap_ext,
)
from ..preprocess.utils import get_zdf_from_header_IMCMB
from ..preprocess.plotting import save_fits_as_figures
from .. import external
from ..services.database.handler import DatabaseHandler
from ..services.database.image_qa import ImageQATable
from ..services.checker import Checker
from ..services.version_check import RuntimeVersionMixin

from .coadd_plan import CoaddPlan
from .const import ZP_KEY
from .header_set import InputHeaderSet
from .background import BackgroundMixin
from .combine import InMemoryCoaddMixin
from .legacy import LegacyCoaddMixin
from .masks import MaskMixin
from .storage import IntermediateStorageMixin
from .swarp import SwarpMixin


warnings.filterwarnings("ignore")


class ImCoadd(
    LegacyCoaddMixin,
    SwarpMixin,
    BackgroundMixin,
    IntermediateStorageMixin,
    MaskMixin,
    InMemoryCoaddMixin,
    BaseSetup,
    DatabaseHandler,
    Checker,
    RuntimeVersionMixin,
):
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
        use_gpu: bool = True,
    ) -> None:

        super().__init__(config, logger, queue)
        self._plan = None  # resolved on first use and by run(): the config is editable until then
        self._device_id = None
        self._use_gpu = use_gpu
        self.logger.process_error = self._process_error

        self.qa_id = None
        DatabaseHandler.__init__(
            self,
            use_database=self.config_node.settings.is_pipeline,
            is_too=self.config_node.settings.is_too,
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

    def direct_coadd_routine(self, use_gpu: bool = False, device_id=None):
        """Same RA-Dec plane, No SWarp reprojection"""
        self._use_gpu = all([use_gpu, self.config_node.imcoadd.gpu, self._use_gpu])
        if self.config_node.imcoadd.joint_wcs or self.config_node.imcoadd.convolve:
            raise self._process_error.ValueError("Direct coaddition requires joint_wcs=False and convolve=False")

        plan = self.plan
        do_zpscale = bool(get_key(self.config_node.imcoadd, "zpscale", default=True))
        total_steps = 3 + int(plan.need_weights) + int(plan.interpolate) + int(do_zpscale)
        step = 0

        self.initialize()
        self._validate_direct_grid()
        images = self.input_images
        self._prepare_intermediate_storage(images)
        if self.plan.output_mask_map or self.plan.dump_reprojected_masks:
            self.prepare_quality_masks(images, detector_images=self.input_images)
        weight_images = None

        if plan.need_weights:
            factory = self.path.imcoadd.factory
            weight_images = factory.stage_images(images, "weight", self._weight_dir)
            weight_images = self.calculate_weight_map(images, device_id=device_id, out_weights=weight_images)
            step += 1
            self.update_progress(
                self._process_registry.step_progress(self._process_spec, step, total_steps),
                self._progress_status("calculate-weight-map-completed"),
            )

        if plan.interpolate:
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
        self._coadd_completed = True
        self.finalize_quality_masks()
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

    def reproject_first_coadd_routine(self, use_gpu: bool = False, device_id=None):
        """Reproject with SWarp, then coadd in memory."""
        self._use_gpu = all([use_gpu, self.config_node.imcoadd.gpu, self._use_gpu])

        do_zpscale = bool(get_key(self.config_node.imcoadd, "zpscale", default=True))
        optional_steps = (
            int(bool(self.plan.need_weights))
            + int(bool(self.plan.interpolate))
            + int(bool(self.config_node.imcoadd.joint_wcs))
            + int(bool(self.config_node.imcoadd.convolve))
            + int(do_zpscale)
        )
        total_steps = 4 + optional_steps
        step = 0

        self.initialize()

        images = self.input_images
        weight_images = None
        do_weight = bool(self.plan.need_weights)
        do_bpmask = bool(self.plan.interpolate)
        joint_wcs = bool(self.config_node.imcoadd.joint_wcs)
        fused_reprojection = do_weight and do_bpmask and not joint_wcs
        if fused_reprojection:
            images = self.weight_and_interpolate(images)
            step += 1
            self.update_progress(
                self._process_registry.step_progress(self._process_spec, step, total_steps),
                self._progress_status("calculate-weight-map-completed"),
            )
            step += 1
            self.update_progress(
                self._process_registry.step_progress(self._process_spec, step, total_steps),
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
                    self._process_registry.step_progress(self._process_spec, step, total_steps),
                    self._progress_status("calculate-weight-map-completed"),
                )

            if do_bpmask:
                images = self.apply_bpmask(images, device_id=device_id, weight_images=weight_images)
                if weight_images is not None:
                    # hand the interp sidecars onward: they carry the zeroed bad-pixel holes
                    weight_images = [add_suffix(im, "weight") for im in images]
                step += 1
                self.update_progress(
                    self._process_registry.step_progress(self._process_spec, step, total_steps),
                    self._progress_status("apply-bpmask-completed"),
                )

        if joint_wcs:
            images = self.joint_registration(images)
            step += 1
            self.update_progress(
                self._process_registry.step_progress(self._process_spec, step, total_steps),
                self._progress_status("joint-registration-completed"),
            )

        if not fused_reprojection:
            images = self.reproject_and_coadd_with_swarp(
                images, coadd=False, weight_images=weight_images
            )
        self._prepare_intermediate_storage(images)
        if (
            self.plan.output_mask_map
            or self.plan.dump_reprojected_masks
            or self.plan.satellite_mask_enabled
        ):
            self.prepare_quality_masks(images, detector_images=self.input_images)
        if self.config_node.imcoadd.convolve:
            self.build_fov_masks(images)
        self._remove_reprojection_intermediates()
        step += 1
        self.update_progress(
            self._process_registry.step_progress(self._process_spec, step, total_steps),
            self._progress_status("reproject-completed"),
        )

        if self.config_node.imcoadd.convolve:
            self.discard_cached_frames()
            self.prepare_convolution(images)
            images = self.run_convolution(images, device_id=device_id)
            self.shrink_fov_masks(self.delta_peeings)
            step += 1
            self.update_progress(
                self._process_registry.step_progress(self._process_spec, step, total_steps),
                self._progress_status("run-convolution-completed"),
            )

        images = self.bkgsub(
            images,
            mask_out_of_fov=True,
            mask_sources=get_key(self.config_node.imcoadd, "source_mask", default=True),
            fov_masks=getattr(self, "_fov_masks", None),
        )
        self._discard_consumed_bkgsub_inputs()
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

        self.coadd_in_memory(images, device_id=device_id)
        self._coadd_completed = True
        self.finalize_quality_masks()
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

    @property
    def plan(self) -> CoaddPlan:
        """Run plan, resolved once and cached; ``run()`` re-resolves so config edits before it land."""
        if self._plan is None:
            self._plan = self._coadd_plan()
        return self._plan

    def run(self, overwrite=False, use_gpu: bool = False, device_id=None):
        self._coadd_completed = False
        self._quality_masks = None
        self._coadd_mask_builder = None
        try:
            self.overwrite = self.resolve_overwrite(overwrite)
            self._plan = self._coadd_plan()  # the config is write-through and editable until here
            if self.plan.routine == "legacy":
                self.legacy_coadd_routine(use_gpu=use_gpu, device_id=device_id)
            elif self.plan.routine == "reproject-first":
                self.reproject_first_coadd_routine(use_gpu=use_gpu, device_id=device_id)
            else:
                self.direct_coadd_routine(use_gpu=use_gpu, device_id=device_id)

            setattr(self.config_node.flag, self._process_spec.name, True)
            self.record_runtime_version()
            self.logger.info(f"'ImCoadd' is Completed in {time_diff_in_seconds(self._st)} seconds")
        except Exception as e:
            self.logger.error(f"Error during imcoadd processing: {str(e)}", e, exc_info=True)

            raise
        finally:
            self._cleanup_imcoadd_intermediates()
        # self.logger.debug(MemoryMonitor.log_memory_usage)

    def initialize(self, overwrite=False):
        self._st = time.time()
        self.logger.info(f"Start 'ImCoadd'")
        local_input_images = get_key(self.config_node.imcoadd, "input_images")
        self.input_images = (
            local_input_images
            if local_input_images is not None  # local_input_images can be an empty list
            else self.config_node.input.calibrated_images
        )
        self.apply_sanity_filter_and_report(current_process=self._process_spec, overwrite=self.overwrite)
        if not self.input_images:
            self.logger.error(
                "No Input for ImCoadd",
                self._process_error.EmptyInputAfterSanityRejection,
            )
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
        self.center = None if self.plan.routine == "direct" else self.input_headers.deprojection_center
        self.logger.debug(f"Deprojection center: {self.center}")

        if not get_key(self.config_node.imcoadd, "coadd_image"):
            self.config_node.imcoadd.coadd_image = self.path.imcoadd.coadd_image
        self.config_node.input.coadd_image = self.config_node.imcoadd.coadd_image
        self.logger.debug(f"Coadd Image: {self.config_node.imcoadd.coadd_image}")
        if self.config_node.settings.is_multi_epoch:
            self._guard_coadd_identity()

        self.logger.info(f"Initialization for ImCoadd is completed")

    def _validate_direct_grid(self, tolerance: float = 1e-7) -> None:
        reference_header = self.input_headers[0]
        reference_shape = (
            reference_header.get("NAXIS2"),
            reference_header.get("NAXIS1"),
        )
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
        """Cull multi-epoch inputs by seeing, ellipticity, and depth."""
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
        default_image_selection_source = "db" if self.config_node.settings.is_pipeline else "headers"
        source = str(
            get_or_set_key(
                self.config_node.imcoadd,
                "image_selection_source",
                default=default_image_selection_source,
            )
        ).lower()
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
            raise self._process_error.EmptyInputAfterSanityRejection(
                f"Quality cuts {cuts} reject all {len(keep)} images"
            )

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

    def _guard_coadd_identity(self):
        """Reject reuse of a coadd built with different settings."""
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
        """Prune inactive factory trees until scratch fits its configured cap."""
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
        shown = lambda value: ("NONE" if value is None or value is False else value)  # noqa: E731
        bp = self.plan
        interp = get_key(node, "interp_type") if bp.interpolate else None
        cards = {
            "COADDRTN": (shown(get_key(node, "coadd_routine")), "imcoadd.coadd_routine"),
            "COADDMOD": (shown(get_key(node, "coadd_mode")), "imcoadd.coadd_mode"),
            "COADDWGT": (shown(get_key(node, "coadd_weighting", default="global")), "imcoadd.coadd_weighting"),
            "BPMPOL":   (shown(bp.policy), "imcoadd.badpix_reprojection_policy"),
            "ZBPWGT":   (bool(bp.zero), "imcoadd.zero_badpix_weight"),
            "ZPSCALE":  (bool(get_key(node, "zpscale")), "imcoadd.zpscale"),
            "INTERP":   (shown(interp), "imcoadd.interp_type"),
            "CONVOLVE": (shown(get_key(node, "convolve")), "imcoadd.convolve"),
            "JOINTWCS": (bool(get_key(node, "joint_wcs")), "imcoadd.joint_wcs"),
            "IMGSELEC": (shown(get_key(node, "image_selection")), "imcoadd.image_selection"),
            "SMTHWGT":  (bool(bp.smooth_weight), "weight map smoothed (not coadd_weighting pixel-wise)"),
            "COVPOL":   (bp.coverage_policy.upper(), "imcoadd.coverage_policy"),
        }  # fmt: skip
        mode = str(get_key(node, "coadd_mode") or "").lower()
        if mode == "clipped":
            cards["CLIPSIG"] = (bp.clip_sigma, "coadd_options.clipped.clip_sigma")
            cards["CLIPAFR"] = (bp.clip_ampfrac, "coadd_options.clipped.clip_ampfrac")
        if mode == "proper":
            cards["PROPWMP"] = (
                self._proper_weight_policy().upper(),
                "coadd_options.proper.weight_map_policy",
            )
        return cards

    def _group_IMCMB(
        self, input_images: list[str], output_images: list[str] = None
    ) -> dict[tuple[str, str, str], list[list[str]]]:
        """Group images by the master frames recorded in IMCMB."""
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
        out_weights: list[str] | None = None,
    ) -> list[str]:
        """Calculate weights from pristine inputs, using input_images only for naming."""
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

        for i, (
            (z_m_file, d_m_file, f_m_file),
            (group_values, group_outputs),
        ) in enumerate(groups.items()):
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
                        bp = self.plan
                        zero_mask = None
                        if bp.zero and not bp.interpolate and not bp.catalog_badpix_zeros:
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
                            weight_store=bool(
                                get_key(
                                    self.config_node.imcoadd,
                                    "persist_weight_maps",
                                    default=False,
                                )
                            ),
                            zero_mask=zero_mask,
                            source_catalogs=self._source_catalogs(uncalculated_images),
                        )
                    else:
                        bp = self.plan
                        if bp.smooth_weight:
                            raise NotImplementedError(
                                "smoothed weight maps are CPU-only (the GPU weight kernel has no smoothing "
                                "pass); set imcoadd.gpu: False, or coadd_weighting: pixel-wise"
                            )
                        if bp.zero and not bp.interpolate:
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
        with fits.open(mask_file, memmap=True) as hdul:
            mask_header = next((hdu.header for hdu in hdul if hdu.data is not None), hdul[0].header)
        if "BADPIX" in mask_header:
            badpix = mask_header["BADPIX"]
            self.logger.debug(f"BADPIX found in header. Using badpix {badpix}.")
        else:
            badpix = 1
            self.logger.warning(
                "BADPIX not found in header. Using default value 1.",
                self._process_error.KeyError,
            )
        return mask_file, badpix

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
        interp_dir = getattr(self, "_interp_dir", factory.interp_dir)
        interp_images = factory.stage_images(input_images, "interp", interp_dir)
        self.config_node.imcoadd.interp_images = interp_images

        # bpmask_array, header = fits.getdata(self.config.preprocess.bpmask_file, header=True)

        method = self.config_node.imcoadd.interp_type
        weight = self.plan.need_weights  # derived: outputs or internal consumers
        # Where this run wrote them, not wherever a sibling of the input happens to sit:
        # reproject-first writes weights to the factory, and a stale one next to the input
        # would be read in silence.
        weight_of = dict(zip(input_images, weight_images)) if weight_images is not None else {}
        zero_interp = bool(self.plan.zero) and not self.plan.catalog_badpix_zeros

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

        self.images_to_coadd = interp_images
        return interp_images

    def zpscale(self, input_images: list[str] | None = None, write_headers: bool = True) -> list[str]:
        """Stamp or scrub FLXSCALE in the snapshot and optional file headers."""
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
        for i, zp in enumerate(zpvalues):
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
            self.input_headers[i]["FLXSCALE"] = (
                flxscale,
                "flux scaling factor by 7DT Pipeline (ImCoadd)",
            )
            self.logger.debug(f"{get_basename(file)} FLXSCALE: {flxscale:.3f}")

        self.logger.info(f"ZP scaling is completed in {time_diff_in_seconds(st)} seconds")
        return input_images

        # ------------------------------------------------------------
        # \tZP Scale
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
        """Return images awaiting a future joint-registration implementation."""
        if input_images is None:
            input_images = self.images_to_coadd
        return input_images

    def prepare_convolution(self, input_images: list[str] | None = None, weight: bool = False):
        """Prepare seeing-match kernels and output paths."""
        if input_images is None:
            input_images = self.images_to_coadd

        method = self.config_node.imcoadd.convolve.lower()
        self.conv_method = method
        self.logger.info(f"Prepare the convolution with {method} method")

        self._conv_inputs = input_images
        self.kernels = []

        if method == "gaussian":
            from ..utils import force_symlink

            factory = self.path.imcoadd.factory
            conv_dir = getattr(self, "_conv_dir", factory.conv_dir)
            self.config_node.imcoadd.conv_files = factory.stage_images(input_images, "conv", conv_dir)

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
                self.logger.error(
                    f"No PEEING for {missing[:3]}; cannot match seeing",
                    self._process_error.KeyError,
                )
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
                if delta_peeing is None:
                    force_symlink(input_images[i], self.config_node.imcoadd.conv_files[i])
                    if weight and self.plan.need_weights:
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
        if input_images is None:
            input_images = getattr(self, "_conv_inputs", None) or self.images_to_coadd
        st = time.time()
        method = self.conv_method
        self._use_gpu = all([use_gpu, self.config_node.imcoadd.gpu, self._use_gpu])
        device_id = device_id if self._use_gpu else "CPU"

        # from .convolve import convolve_fft, get_edge_mask

        kernels = [k for k in self.kernels if k is not None]
        image_list = [f for f, k in zip(input_images, self.kernels) if k is not None]
        outim_list = [f for f, k in zip(self.config_node.imcoadd.conv_files, self.kernels) if k is not None]
        delta_peeing_list = [v for v, k in zip(self.delta_peeings, self.kernels) if k is not None]

        if not image_list:
            # A single frame or an equal-seeing group needs no convolution.
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
                    self.logger.error(
                        f"Weight map not found for all images.",
                        self._process_error.FileNotFoundError,
                    )
                    raise self._process_error.FileNotFoundError(f"Weight map not found for all images.")

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
        """Resolve an image's valid weight companion under either naming convention."""
        factory = self.path.imcoadd.factory
        candidates = []
        if os.path.dirname(image) == factory.swarp_resample_dir("sci") and not self.plan.weight_on_sci_pass:
            candidates.append(
                collapse(
                    factory.resampled_weight_images([image], pass_type="wht"),
                    force=True,
                )
            )
        candidates += [add_suffix(image, "weight"), swap_ext(image, "weight.fits")]
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return candidate
        self.logger.error(
            f"No weight map found for {get_basename(image)}",
            self._process_error.FileNotFoundError,
        )
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
        path_to_plot = os.path.join(collapse(self.path.figure_dir, force=True), swap_ext(basename, "jpg"))
        save_fits_as_figures(fits.getdata(coadd_img), path_to_plot)
        self.logger.info(f"Coadd image is plotted and saved in {path_to_plot}.")

    def _update_header(self):
        """Legacy routine: overlay ``self.input_headers.coadd_header`` onto the SWarp-written coadd FITS."""
        coadd_header = self.input_headers.coadd_header
        with fits.open(self.config_node.imcoadd.coadd_image, mode="update") as hdul:
            header = hdul[0].header
            for card in coadd_header.cards:
                header[card.keyword] = (card.value, card.comment)
            hdul.flush()
