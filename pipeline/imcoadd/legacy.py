from typing import TYPE_CHECKING

from ..path.path import PathHandler
from ..services.logger import Logger
from .coadd_plan import CoaddPlan

if TYPE_CHECKING:
    from ..config._crossfilter_stubs import CrossFilterNode
    from ..config._sciproc_stubs import SciProcNode

    ConfigNodeT = SciProcNode | CrossFilterNode  # ImCoadd runs on the first, WhiteImage on the second


class LegacyCoaddMixin:
    config_node: "ConfigNodeT"
    logger: Logger
    path: PathHandler
    plan: CoaddPlan
    input_images: list[str]
    images_to_coadd: list[str] | None
    _use_gpu: bool
    _coadd_completed: bool

    def legacy_coadd_routine(self, use_gpu: bool = False, device_id=None):
        """
        Uses sci/wht double pass for LANCZOS3 sci reprojection & NEAREST weight reprojection (~1px)
        But the interpolated pixel values contribute in SWarp median coadd.
        """
        self._use_gpu = all([use_gpu, self.config_node.imcoadd.gpu, self._use_gpu])

        self.initialize()
        if not self.plan.reject_saturated_pixels:
            self.logger.info("Legacy coadds in SWarp: saturated pixels are not rejected (saturation_reprojection_policy ignored)")
        self._prepare_intermediate_storage(self.input_images)

        images = self.bkgsub(self.input_images)
        self.update_progress(
            self._process_registry.milestone_progress(self._process_spec, "bkgsub"),
            self._progress_status("bkgsub-completed"),
        )
        self.zpscale(images)
        self.update_progress(
            self._process_registry.milestone_progress(self._process_spec, "zpscale"),
            self._progress_status("zpscale-completed"),
        )

        if self.plan.compute_single_weight_maps:
            self.calculate_weight_map(images, device_id=device_id)
            self.update_progress(
                self._process_registry.milestone_progress(self._process_spec, "calculate_weight_map"),
                self._progress_status("calculate-weight-map-completed"),
            )

        if self.plan.interpolate_badpix:
            images = self.apply_bpmask(images, device_id=device_id)
            self.update_progress(
                self._process_registry.milestone_progress(self._process_spec, "apply_bpmask"),
                self._progress_status("apply-bpmask-completed"),
            )

        if self.plan.joint_wcs:
            images = self.joint_registration(images)
            self.update_progress(
                self._process_registry.milestone_progress(self._process_spec, "joint_registration"),
                self._progress_status("joint-registration-completed"),
            )

        if self.plan.convolve:
            self.prepare_convolution(images)
            images = self.run_convolution(images, device_id=device_id)
            self.update_progress(
                self._process_registry.milestone_progress(self._process_spec, "run_convolution"),
                self._progress_status("run-convolution-completed"),
            )

        self.coadd_with_swarp(images)
        self.apply_legacy_coverage_policy(images)
        if self._need_quality_masks:
            resampled = self.path.imcoadd.factory.resampled_images(images, pass_type=self.plan.sci_pass_type)
            self.prepare_quality_masks(resampled, detector_images=self.input_images)
        self._coadd_completed = True
        self.finalize_quality_masks()
        self.fill_coadd_nan()
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
