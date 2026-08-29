from ..const.crossfilter import CROSSFILTERPROCESS_REGISTRY, WHITE_PHOTOMETRY_SPEC
from ..errors import WhiteCatalogError
from .photometry import Photometry, PhotometrySingle


class WhiteCatalog(Photometry):
    _process_registry = CROSSFILTERPROCESS_REGISTRY
    _process_errors = {WHITE_PHOTOMETRY_SPEC: WhiteCatalogError}

    def __init__(self, config, logger=None, queue=False, overwrite=False):
        super().__init__(
            config=config,
            logger=logger,
            queue=queue,
            photometry_mode=WHITE_PHOTOMETRY_SPEC.photometry_mode,
            overwrite=overwrite,
        )
        if not self.input_images:
            raise self._process_error.PrerequisiteNotMetError("input.white_image is not set on this config")

    def _run_sequential(self, overwrite=True) -> None:
        single_config = self.config_node.extract_single_image_config(0)
        PhotometrySingle(
            single_config,
            logger=self.logger,
            ref_cat_type=self.ref_cat_type,
            total_image=1,
            check_filter=False,
            reset_count=True,
            current_process=self._process_spec,
            parent_path=self.path,
        ).run_source_catalog(overwrite=overwrite)

    def _run_parallel(self, overwrite=True) -> None:
        self._run_sequential(overwrite=overwrite)
