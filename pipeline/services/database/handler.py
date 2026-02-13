import json
import logging

from .too import TooDB
from .process_status import ProcessStatus
from .image_qa import ImageQA
from ...errors import registry


class DatabaseHandler:

    def __init__(
        self,
        db_params=None,
        add_database: bool = True,
        is_too: bool = False,
        logger=None,
    ):
        self.is_too = is_too
        self.add_database = add_database

        if not hasattr(self, "logger"):
            self.logger = logger or logging.getLogger(__name__)

        if add_database:
            self.too_db = TooDB() if is_too else None
            self.too_id = None
            self.process_status = None if is_too else ProcessStatus(db_params)
            self.process_status_id = None if is_too else None
            self.image_qa = None if is_too else ImageQA(db_params)
            self.image_qa_id = None if is_too else None
        else:
            self.too_db = None
            self.too_id = None
            self.process_status = None
            self.process_status_id = None
            self.image_qa = None
            self.image_qa_id = None

    @property
    def is_connected(self) -> bool:

        if not self.add_database:
            return False
        elif self.is_too and self.too_db is not None:
            return True
        else:
            return self.process_status is not None and self.image_qa is not None

    def create_process_data(self, config, overwrite: bool = False):
        if self.is_too and self.too_db is not None:
            self.too_db.read_data(config.name)
            self.too_id = self.too_db.too_id
            return None
        elif not self.is_connected:
            return None

        table = self.process_status.pyTable.from_file(config.name)

        existing_process_id = self.process_status.read_data_by_params(**table.to_dict())
        if existing_process_id:
            self.logger.info(f"Found existing process db record (PID: {existing_process_id})")
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
                return existing_process_id

        self.process_status_id = self.process_status.create_data(table)
        return self.process_status_id

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

        if not self.is_connected:
            self.logger.warning(f"Skipping QA data creation: process_status_id is not set")
            return None

        table = self.image_qa.pyTable.from_file(file, process_status_id)

        if overwrite:
            self.image_qa.delete_data(table.id)

        qa_id = self.image_qa.create_data(table)

        return qa_id

    def update_image_qa_data(self, image_qa_id: int, data):
        data.pop("id", None)
        self.image_qa.update_data(image_qa_id, **data)

    def get_process_status(self, nightdate, config_type="science"):

        rows = self.process_status.read_data_by_params(
            return_pyTable=True, nightdate=nightdate, config_type=config_type
        )
        if rows is None:
            return None

        if config_type == "science":
            dicts = [row.to_dict() for row in rows]
        else:
            dicts = []
            for row in rows:
                classify_images = self.image_qa.classify_images(self.image_qa.get_by_process_status_id(row.id))
                temp_dict = row.to_dict()
                temp_dict["bias"] = classify_images["bias"]
                temp_dict["dark"] = classify_images["dark"]
                temp_dict["flat"] = classify_images["flat"]
                dicts.append(temp_dict)

        return dicts

    def get_image_qa(self, params, image_type="single", date_min=None, date_max=None):
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
        elif code_type == "error":
            if row.errors is None or self.check_unknown_code(row.errors):
                row.errors = code_value
                self.process_status.update_data(self.process_status_id, errors=code_value)
            else:
                return False
        else:
            raise ValueError(f"Invalid code type: {code_type}")

    def check_unknown_code(self, code_value: int):
        return (int(code_value) - 99) % 100 == 0
