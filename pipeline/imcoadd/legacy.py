from ..path.path import PathHandler
from .coadd_plan import CoaddPlan


class LegacyCoaddMixin:
    path: PathHandler
    plan: CoaddPlan

    def legacy_coadd_routine(self, use_gpu: bool = False, device_id=None):
        self._use_gpu = all([use_gpu, self.config_node.imcoadd.gpu, self._use_gpu])
        self._coadd_completed = False
        self._quality_masks = None
        self._coadd_mask_builder = None
        self._zdf_cache = {}

        self.initialize()
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

        if self.plan.need_weights:
            self.calculate_weight_map(images, device_id=device_id)
            self.update_progress(
                self._process_registry.milestone_progress(
                    self._process_spec, "calculate_weight_map"
                ),
                self._progress_status("calculate-weight-map-completed"),
            )

        if self.plan.interpolate:
            images = self.apply_bpmask(images, device_id=device_id)
            self.update_progress(
                self._process_registry.milestone_progress(
                    self._process_spec, "apply_bpmask"
                ),
                self._progress_status("apply-bpmask-completed"),
            )

        if self.plan.joint_wcs:
            images = self.joint_registration(images)
            self.update_progress(
                self._process_registry.milestone_progress(
                    self._process_spec, "joint_registration"
                ),
                self._progress_status("joint-registration-completed"),
            )

        if self.plan.convolve:
            self.prepare_convolution(images)
            images = self.run_convolution(images, device_id=device_id)
            self.update_progress(
                self._process_registry.milestone_progress(
                    self._process_spec, "run_convolution"
                ),
                self._progress_status("run-convolution-completed"),
            )

        self.reproject_and_coadd_with_swarp(images, coadd=True)
        self.apply_legacy_coverage_policy(images)
        if self.plan.output_mask_map:
            pass_type = "sci" if self.plan.need_weights else ""
            resampled = self.path.imcoadd.factory.resampled_images(
                images, pass_type=pass_type
            )
            self.prepare_quality_masks(resampled, detector_images=self.input_images)
        self._coadd_completed = True
        self.finalize_quality_masks()
        self.update_progress(
            self._process_registry.milestone_progress(
                self._process_spec, "coadd_with_swarp"
            ),
            self._progress_status("coadd-with-swarp-completed"),
        )

        self.plot_coadd_image()
        self.update_progress(
            self._process_registry.milestone_progress(
                self._process_spec, "plot_coadd_image"
            ),
            self._progress_status("plot-coadded-image-completed"),
        )

        self.register_coadd_qa()

        self.update_progress(
            self._process_registry.completed_progress(self._process_spec),
            self._progress_status("completed"),
        )
