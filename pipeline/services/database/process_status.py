from datetime import date, datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import json

from .base import BaseDatabase, DatabaseError

from .query_string import *
import os


@dataclass
class ProcessStatusTable:
    """Data class for process_status table records"""

    # Required fields
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    name: Optional[str] = None
    config_type: Optional[str] = None
    input_type: Optional[str] = None
    pipeline_version: Optional[str] = None
    nightdate: Optional[date] = None
    unit: Optional[str] = None
    filter: Optional[str] = None
    object: Optional[str] = None
    progress: Optional[int] = 0
    status: Optional[str] = "pending"
    warnings: Optional[List[int]] = None
    errors: Optional[List[int]] = None
    config_file: Optional[str] = None
    log_file: Optional[str] = None
    debug_log_file: Optional[str] = None
    comments_file: Optional[str] = None

    @classmethod
    def from_row(cls, row: tuple, columns: List[str] = None):

        if columns is None:
            columns = cls.__annotations__.keys()

        if len(row) != len(columns):
            raise ValueError(f"Row length ({len(row)}) doesn't match columns length ({len(columns)})")

        def parse_json_field(value):
            if value and isinstance(value, str):
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return None
            elif value and not isinstance(value, (dict, list)):
                return None
            return value

        row_dict = dict(zip(columns, row))

        # Parse JSON fields
        if "warnings" in row_dict:
            row_dict["warnings"] = parse_json_field(row_dict["warnings"])
        if "errors" in row_dict:
            row_dict["errors"] = parse_json_field(row_dict["errors"])

        # Create instance using column names (which match field names)
        return cls(**row_dict)

    def to_dict(self) -> Dict[str, Any]:

        data = asdict(self)

        data = {k: v for k, v in data.items() if v is not None and k != "id"}

        # Convert JSON fields
        if "warnings" in data and isinstance(data["warnings"], (dict, list)):
            data["warnings"] = json.dumps(data["warnings"])
        if "errors" in data and isinstance(data["errors"], (dict, list)):
            data["errors"] = json.dumps(data["errors"])

        return data

    @classmethod
    def from_file(cls, config: str):
        from ...config.utils import find_config
        from ...version import __version__

        config_file, config_properties = find_config(config, return_properties=True)

        name = os.path.basename(config_file).replace(".yml", "")

        return cls(
            name=name,
            input_type="daily",
            pipeline_version=__version__,
            config_file=config_file,
            log_file=config_file.replace(".yml", ".log"),
            debug_log_file=config_file.replace(".yml", "_debug.log"),
            comments_file=config_file.replace(".yml", ".txt"),
            **config_properties,
        )


class ProcessStatus(BaseDatabase):
    """Database class for managing process_status records"""

    def __init__(self, db_params: Optional[Dict[str, Any]] = None):
        """Initialize with database parameters"""
        self._table_name = "process_status"
        self._pyTable = ProcessStatusTable
        super().__init__(db_params)

    @property
    def table_name(self):
        return self._table_name

    @property
    def pyTable(self):
        return self._pyTable

    def add_exception_code(self, process_id: int, code_type: str, code_value: int):
        row = self.read_data_by_id(process_id)
        if row is None:
            raise ValueError(f"Process ID {process_id} not found")
        if code_type == "warning":
            if row.warnings is None:
                row.warnings = []
            row.warnings.append(code_value)

            warnings = list(set(row.warnings))

            self.update_data(process_id, warnings=json.dumps(warnings))
        elif code_type == "error":
            if row.errors is None:
                row.errors = []
            row.errors.append(code_value)
            errors = list(set(row.errors))
            self.update_data(process_id, errors=json.dumps(errors))
        else:
            raise ValueError(f"Invalid code type: {code_type}")

    def reset_exceptions(self, process_id: int):
        # Empty lists need to be converted to JSON strings for jsonb columns
        self.update_data(process_id, warnings=[], errors=[])
