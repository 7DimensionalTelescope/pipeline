from __future__ import annotations
import json
import logging
from typing import TYPE_CHECKING

from .too import TooDB
from .process_status import ProcessStatus, dispatch_host
from .image_qa import ImageQA
from .image_qa_dependency import ImageQADependency
from .process_status_dependency import ProcessStatusDependency
from ...errors import registry
from ...const import (
    AUTO_RECORD_PROCESS_STATUS_DEPENDENCIES,
    CONFIG_TYPE_CROSSFILTER,
    CONFIG_TYPE_SCIENCE,
    CALIB_TYPE_BIAS,
    CALIB_TYPE_DARK,
    CALIB_TYPE_FLAT,
    IMAGE_TYPE_SINGLE,
)

if TYPE_CHECKING:
    from ...config import ConfigNode


class DatabaseHandler:

    def __init__(
        self,
        db_params=None,
        use_database: bool = True,
        is_too: bool = False,
        logger=None,
    ):
        self.is_too = is_too
        self.use_database = use_database

        if not hasattr(self, "logger"):
            self.logger = logger or logging.getLogger(__name__)

        if use_database:
            self.too_db = TooDB() if is_too else None
            self.too_id = None
            self.process_status = None if is_too else ProcessStatus(db_params)
            self.process_status_id = None if is_too else None
            self.image_qa = None if is_too else ImageQA(db_params, logger=self.logger)
            self.image_qa_id = None if is_too else None
            self.image_qa_dependency = None if is_too else ImageQADependency(db_params)
            self.process_status_dependency = None if is_too else ProcessStatusDependency(db_params)
        else:
            self.too_db = None
            self.too_id = None
            self.process_status = None
            self.process_status_id = None
            self.image_qa = None
            self.image_qa_id = None
            self.image_qa_dependency = None
            self.process_status_dependency = None

    @property
    def is_connected(self) -> bool:

        if not self.use_database:
            return False
        elif self.is_too and self.too_db is not None:
            return True
        else:
            return self.process_status is not None and self.image_qa is not None

    def create_process_data(self, config_node: ConfigNode, overwrite: bool = False):
        if self.is_too and self.too_db is not None:
            self.too_db.read_data(config_node.name)
            self.too_id = self.too_db.too_id
            return None
        elif not self.is_connected:
            return None

        table = self.process_status.pyTable.from_file(config_node.info.file)

        # Identity is name (unique index); these two vary per run and must not narrow the match.
        identity = {k: v for k, v in table.to_dict().items() if k not in ("input_type", "dispatch")}
        existing_process_id = self.process_status.read_data_by_params(**identity)
        if existing_process_id:
            self.logger.info(f"Found existing process db record (PID: {existing_process_id})")
            existing_row = self.process_status.read_data_by_id(existing_process_id)
            prev_errors = existing_row.errors
            prev_warnings = existing_row.warnings
            self.process_status_id = existing_process_id
            self.reset_exceptions()
            self.logger.info(
                "Reset warnings/errors for existing process db record "
                f"(PID: {existing_process_id}), previous warnings: {prev_warnings}, previous errors: {prev_errors}"
            )
            if overwrite:
                # Use cascade delete to properly handle foreign key constraints
                try:
                    self.process_status.delete_data(existing_process_id)
                    self.logger.info(f"Deleted existing process db record (PID: {existing_process_id}) for overwrite")
                except Exception as e:
                    self.logger.error(f"Error deleting existing process db record (PID: {existing_process_id}): {e}")
                    raise
            else:
                self.logger.info(f"Using existing process db record (PID: {existing_process_id})")
                self.process_status_id = existing_process_id
                self._stamp_dispatch(existing_process_id)
                return existing_process_id

        self.process_status_id = self.process_status.create_data(table)
        self._stamp_dispatch(self.process_status_id)
        return self.process_status_id

    def _stamp_dispatch(self, process_status_id):
        """Which host ran it, always rewritten: update_data drops None, so a rerun elsewhere would keep the old name."""
        self.process_status.update_data(process_status_id, dispatch=dispatch_host() or "None")

    def update_progress(self, progress: int, status: str = None) -> bool:
        """
        Update pipeline progress in database.

        Args:
            progress: Progress percentage (0-100)
            status: Optional status string

        Returns:
            True if successful, False otherwise
        """
        if self.is_too and self.too_db is not None:
            return self.too_db.update_too_progress(self.too_id, progress, status)
        elif not self.is_connected:
            return False
        else:
            from ...version import __version__

            self.process_status.update_data(
                self.process_status_id, progress=progress, status=status, pipeline_version=__version__
            )

            return True

    def create_image_qa_data(self, file: str, process_status_id: int, overwrite: bool = False):
        """Reads image header to generate image QA data"""

        if not self.is_connected:
            self.logger.warning(f"Skipping QA data creation: process_status_id is not set")
            return None

        try:
            table = self.image_qa.pyTable.from_file(file, process_status_id)
        except ValueError as e:
            self.logger.error(f"image_qa registration refused: {e}")
            raise

        if overwrite:
            self.image_qa.delete_data(table.id)

        qa_id = self.image_qa.create_data(table)

        return qa_id

    def create_image_qa_dependencies(self, file: str, qa_id: int) -> int:
        """Sync image_qa_dependency rows for the given file and image_qa id."""
        if not self.is_connected or self.image_qa_dependency is None or qa_id is None:
            return 0
        try:
            n = self.image_qa_dependency.sync(file, qa_id)
            if n:
                self.logger.debug(f"Synced {n} image_qa_dependency rows for qa_id={qa_id}")
            return n
        except Exception as e:
            self.logger.warning(f"Failed to sync image_qa_dependency for qa_id={qa_id}: {e}")
            return 0

    def mirror_config_dependencies(self, edges) -> int:
        """Best-effort mirror of config-level dependency edges into postgres.

        ``edges`` is an iterable of ``(derived_config_name, source_config_name,
        dependency_role)``.  Never raises: the scheduler's own store remains
        authoritative, so a database outage must not break callers.
        """
        if (
            not AUTO_RECORD_PROCESS_STATUS_DEPENDENCIES
            or not self.is_connected
            or self.process_status_dependency is None
        ):
            return 0
        try:
            n = self.process_status_dependency.replace_dependencies(edges)
            if n:
                self.logger.debug(f"Mirrored {n} process_status_dependency rows")
            return n
        except Exception as e:
            self.logger.warning(f"Failed to mirror process_status_dependency: {e}")
            return 0

    def sync_config_dependencies(self, process_status_id: int = None) -> int:
        """Refresh this config's dependency rows from the products it actually wrote.

        Called at the end of every stage that registers images, so the edges follow
        reprocessing without depending on how the config was launched.  Never raises:
        dependency bookkeeping must not fail a run that produced good data.
        """
        if (
            not AUTO_RECORD_PROCESS_STATUS_DEPENDENCIES
            or not self.is_connected
            or self.process_status_dependency is None
        ):
            return 0
        pid = process_status_id if process_status_id is not None else getattr(self, "process_status_id", None)
        if pid is None:
            return 0
        try:
            n = self.process_status_dependency.sync_from_products(pid)
            if n:
                self.logger.debug(f"Synced {n} process_status_dependency rows from products")
            return n
        except Exception as e:
            self.logger.warning(f"Failed to sync process_status_dependency: {e}")
            return 0

    def update_image_qa_data(self, image_qa_id: int, data):
        data.pop("id", None)
        self.image_qa.update_data(image_qa_id, **data)

    def get_process_status(self, nightdate, config_type=CONFIG_TYPE_SCIENCE):

        rows = self.process_status.read_data_by_params(
            return_pyTable=True, nightdate=nightdate, config_type=config_type
        )
        if rows is None:
            return None

        if config_type in (CONFIG_TYPE_SCIENCE, CONFIG_TYPE_CROSSFILTER):
            dicts = [row.to_dict() for row in rows]
        else:
            dicts = []
            for row in rows:
                classify_images = self.image_qa.classify_images(self.image_qa.get_by_process_status_id(row.id))
                temp_dict = row.to_dict()
                temp_dict[CALIB_TYPE_BIAS] = classify_images[CALIB_TYPE_BIAS]
                temp_dict[CALIB_TYPE_DARK] = classify_images[CALIB_TYPE_DARK]
                temp_dict[CALIB_TYPE_FLAT] = classify_images[CALIB_TYPE_FLAT]
                dicts.append(temp_dict)

        return dicts

    def get_image_qa(self, params, image_type=IMAGE_TYPE_SINGLE, date_min=None, date_max=None):
        import numpy as np

        params = np.atleast_1d(params)
        default_params = ["date_obs", "nightdate", "unit", "filter", "object", "exptime", "image_name"]
        params = list(params) + default_params
        rows = self.image_qa.read_data_by_params_with_date_range(
            columns=params,
            date_min=date_min,
            date_max=date_max,
            image_type=image_type,
        )

        rows = [dict(zip(params, row)) for row in rows if row[0] is not None]
        return rows

    def add_exception_code(self, code_type: str, code_value: int):

        row = self.process_status.read_data_by_id(self.process_status_id)

        if row is None:
            raise ValueError(f"Process ID {self.process_status_id} not found")

        if code_type == "warning":
            if row.warnings is None:
                row.warnings = []
            row.warnings.append(code_value)

            warnings = list(set(row.warnings))

            self.process_status.update_data(self.process_status_id, warnings=json.dumps(warnings))
        elif code_type == "error":

            if row.errors is None:
                row.errors = code_value
                self.process_status.update_data(self.process_status_id, errors=code_value)
            else:
                return False
        else:
            raise ValueError(f"Invalid code type: {code_type}")

    def reset_exceptions(self, procsss_name=None):

        if self.process_status_id is None:
            return

        # Empty lists need to be converted to JSON strings for jsonb columns
        if procsss_name is None:
            self.process_status.update_data(self.process_status_id, warnings=[], errors="None")
            return True
        else:

            base_code = registry.process(procsss_name).code

            warnings = self.process_status.read_data_by_id(self.process_status_id).warnings

            if warnings is not None:

                for warning in warnings:
                    if warning // 100 == base_code:
                        warnings.remove(warning)

                if 999 in warnings:
                    warnings.remove(999)

            else:
                warnings = []

            self.process_status.update_data(self.process_status_id, warnings=json.dumps(warnings), errors="None")

            return True

        return False


class ExceptionHandler:
    def __init__(self, process_status_id: int):
        self.process_status = ProcessStatus()
        self.process_status_id = process_status_id

    def add_exception_code(self, code_type: str, code_value: int):

        row = self.process_status.read_data_by_id(self.process_status_id)
        if row is None:
            raise ValueError(f"Process ID {self.process_status_id} not found")

        if code_type == "warning":
            if row.warnings is None:
                row.warnings = []
            row.warnings.append(code_value)

            warnings = list(set(row.warnings))

            self.process_status.update_data(self.process_status_id, warnings=json.dumps(warnings))
            return True
        elif code_type == "error":
            # overwrite if this is the first error or existing is UnknownError
            # first non-unknown error takes precedence
            if row.errors is None or self.check_unknown_code(row.errors):
                row.errors = code_value
                self.process_status.update_data(self.process_status_id, errors=code_value)
                return True
            else:
                return False
        else:
            raise ValueError(f"Invalid code type: {code_type}")

    def check_unknown_code(self, code_value: int):
        return (int(code_value) - 99) % 100 == 0
