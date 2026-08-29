from datetime import date, datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import json

from .base import BaseDatabase, DatabaseError
from ...const.environ import MAIN_HOST

from .query_string import *
import os
import socket


_INPUT_TYPE = None


def set_input_type(value):
    """The scheduler row's freeform input_type, handed to this run by the reduction CLI's -input_type."""
    global _INPUT_TYPE
    _INPUT_TYPE = (value or "").strip() or None


def dispatch_host():
    """Worker host that ran it; None on MAIN_HOST, whose runs are the unremarkable case."""
    host = socket.gethostname()
    return None if host.startswith(MAIN_HOST) else host


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
    errors: Optional[int] = None
    config_file: Optional[str] = None
    log_file: Optional[str] = None
    debug_log_file: Optional[str] = None
    comments_file: Optional[str] = None
    # human inspection axis. None everywhere in from_file, so reruns never overwrite it.
    sanity: Optional[bool] = None  # None: not inspected; False: human-rejected; True: human-approved
    inspcomm: Optional[str] = None  # mirrors the FITS INSPCOMM card
    inspected_by: Optional[str] = None
    inspected_at: Optional[datetime] = None
    # Last, to stay aligned with ALTER TABLE ADD COLUMN, which appends at the end of the row.
    dispatch: Optional[str] = None

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

        # Parse JSON fields (only warnings, errors is now integer)
        if "warnings" in row_dict:
            row_dict["warnings"] = parse_json_field(row_dict["warnings"])

        # Create instance using column names (which match field names)
        return cls(**row_dict)

    def to_dict(self) -> Dict[str, Any]:

        data = asdict(self)

        data = {k: v for k, v in data.items()}

        # Convert JSON fields (only warnings, errors is now integer)
        if "warnings" in data and isinstance(data["warnings"], (dict, list)):
            data["warnings"] = json.dumps(data["warnings"])

        return data

    @classmethod
    def from_file(cls, config_path: str):
        # from ...config.utils import find_config
        from ...version import __version__
        from ...path.name import NameHandler

        # config_file, config_properties = find_config(config, return_properties=True)
        nh = NameHandler(config_path)
        config_file = nh.abspath
        config_properties = nh.config_properties
        name = nh.stem

        return cls(
            name=name,
            input_type=_INPUT_TYPE,
            dispatch=dispatch_host(),
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

    def set_config_sanity(self, name: str, sanity) -> Optional[int]:
        """Machine-set config sanity (the return-code-2 path). Never touches a human verdict. None clears it."""
        row = self.read_data(name)
        if row is None or row.inspected_at is not None or row.sanity is sanity:
            return None

        self.update_data(row.id, sanity="None" if sanity is None else sanity)
        return row.id

    def set_inspection(
        self, name: str, sanity: bool, inspcomm: str = None, by: str = None, overwrite: bool = False
    ) -> Optional[int]:
        """Record a human verdict on one config. Returns the row id, or None if the config has no row."""
        from ..inspection import resolve_inspcomm

        row = self.read_data(name)
        if row is None:
            return None

        self.update_data(
            row.id,
            sanity=sanity,
            inspcomm=resolve_inspcomm(row.inspcomm, inspcomm, overwrite),
            inspected_by=by or os.environ.get("USER"),
            inspected_at=datetime.now(),
        )
        return row.id
