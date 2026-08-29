from ..const import SEPP_CONFIG
from ..const.crossfilter import CROSSFILTERPROCESS_REGISTRY, PHOT7DS_SPEC
from ..errors import Phot7DSError
from ..services.database.handler import DatabaseHandler
from ..services.setup import BaseSetup
from ..services.version_check import RuntimeVersionMixin
from ..utils import atleast_1d, collapse, time_diff_in_seconds


def run_phot7ds(config_node, path, overwrite=False, thread_count=None):
    """Run phot7ds forced photometry on the white image; every output path comes from PathHandler."""
    from phot7ds import ensure_sepp_config, run_photometry

    detection_image = collapse(atleast_1d(config_node.imcoadd.coadd_image), raise_error=True)
    science_images = list(config_node.input.coadd_images or config_node.imcoadd.input_images or [])
    obj = collapse(atleast_1d(path.name.obj), raise_error=True)

    return run_photometry(
        science_images=science_images,
        detection_image=detection_image,
        reference_catalog=path.photometry.get_ref_cat(obj, ref_cat_type="GaiaXP"),
        catalog_path=path.phot7ds.catalog,
        coverage_mask=None,
        sepp_config_file=str(ensure_sepp_config(SEPP_CONFIG)),
        detection_label="7DT",
        overwrite=overwrite,
        thread_count=thread_count,
    )


class Phot7DS(BaseSetup, DatabaseHandler, RuntimeVersionMixin):
    """The phot7ds stage: forced photometry via run_phot7ds, with normal flag/progress/DB bookkeeping."""

    _process_spec = PHOT7DS_SPEC
    _process_registry = CROSSFILTERPROCESS_REGISTRY
    _process_error = Phot7DSError

    def __init__(self, config=None, logger=None, queue=False, overwrite=False, thread_count=None):
        super().__init__(config, logger, queue)
        self.overwrite = self.resolve_overwrite(overwrite)
        self.thread_count = thread_count
        self.logger.process_error = self._process_error

        DatabaseHandler.__init__(
            self, use_database=self.config_node.settings.is_pipeline, is_too=self.config_node.settings.is_too
        )
        if self.is_connected:
            self.process_status_id = self.create_process_data(self.config_node)
            self.reset_exceptions(self._process_spec.name)

            if self.process_status_id is not None:
                from ..services.database.handler import ExceptionHandler

                self.logger.database = ExceptionHandler(self.process_status_id)
            self.update_progress(
                self._process_registry.configured_progress(self._process_spec),
                f"{self._process_spec.name}-configured",
            )

    def run(self, overwrite: bool = False):
        import time

        st = time.time()
        try:
            result = run_phot7ds(
                self.config_node,
                self.path,
                overwrite=self.overwrite or overwrite,
                thread_count=self.thread_count,
            )
            self.config_node.phot7ds.catalog = str(result.catalog_path)
            setattr(self.config_node.flag, self._process_spec.name, True)
            self.record_runtime_version()
            self.update_progress(
                self._process_registry.completed_progress(self._process_spec),
                f"{self._process_spec.name}-completed",
            )
            self.logger.info(
                f"'Phot7DS' is Completed in {time_diff_in_seconds(st)} seconds ({result.n_sources} sources)"
            )
            return result
        except Exception as e:
            self.logger.error(f"Error during phot7ds processing: {str(e)}", e, exc_info=True)
            raise
