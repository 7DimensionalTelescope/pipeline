from datetime import date, datetime
from typing import List, Dict, Optional, Any
from typing_extensions import override
from dataclasses import dataclass, asdict
import json
import logging

from .base import BaseDatabase, DatabaseError

from .query_string import *
import os

from astropy.io import fits
import re


@dataclass
class ImageQATable:
    """Data class for process_status table records"""

    # Required fields
    id: Optional[int] = None
    process_status_id: Optional[int] = None

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    image_name: Optional[str] = None
    image_type: Optional[str] = None
    image_group: Optional[str] = None
    image_path: Optional[str] = None
    imageid: Optional[str] = None
    m_epoch: Optional[bool] = None

    nightdate: Optional[date] = None
    unit: Optional[str] = None
    filter: Optional[str] = None
    object: Optional[str] = None
    exptime: Optional[float] = None

    date_obs: Optional[datetime] = None
    altitude: Optional[float] = None
    azimuth: Optional[float] = None
    ra: Optional[float] = None
    dec: Optional[float] = None

    sanity: Optional[bool] = None
    inspectd: Optional[bool] = None
    err_msgs: Optional[List[str]] = None
    ppflag: Optional[int] = None

    clipped: Optional[float] = None
    clipmed: Optional[float] = None
    clipstd: Optional[float] = None
    clipmin: Optional[float] = None
    clipmax: Optional[float] = None
    sigmean: Optional[float] = None
    shifted: Optional[bool] = None
    shftscr: Optional[float] = None
    edgevar: Optional[float] = None
    uniform: Optional[float] = None
    nhotpix: Optional[int] = None
    ntotpix: Optional[int] = None

    seeingmn: Optional[float] = None
    seeingsd: Optional[float] = None
    pa_quad: Optional[float] = None
    pa_align: Optional[float] = None
    isep_q2: Optional[float] = None
    isep_p95: Optional[float] = None
    i_recall: Optional[float] = None
    bin0fwhm: Optional[float] = None
    bin1fwhm: Optional[float] = None
    bin2fwhm: Optional[float] = None
    bin0mad: Optional[float] = None
    bin1mad: Optional[float] = None
    bin2mad: Optional[float] = None
    unmatch: Optional[float] = None
    rsep_rms: Optional[float] = None
    rsep_q2: Optional[float] = None
    rsep_p95: Optional[float] = None
    awincrmn: Optional[float] = None
    ellipmn: Optional[float] = None

    zp_5: Optional[float] = None
    ezp_5: Optional[float] = None
    seeing: Optional[float] = None
    peeing: Optional[float] = None
    rotang: Optional[float] = None
    skyval: Optional[float] = None
    skysig: Optional[float] = None
    zp_auto: Optional[float] = None
    ezp_auto: Optional[float] = None
    ul5_5: Optional[float] = None
    stdnumb: Optional[float] = None

    inf_filt: Optional[str] = None
    bkg_step: Optional[bool] = None

    @classmethod
    def from_row(cls, row: tuple, columns: List[str] = None):

        if columns is None:
            columns = list(cls.__annotations__.keys())
            # If row length doesn't match, we can't proceed without column names
            if len(row) != len(columns):
                raise ValueError(
                    f"Row length ({len(row)}) doesn't match columns length ({len(columns)}). Please provide column names."
                )

        # Get dataclass field names
        dataclass_fields = set(cls.__annotations__.keys())

        # Create a mapping of column names to row values
        row_dict = {}
        for i, col_name in enumerate(columns):
            if i < len(row) and col_name in dataclass_fields:
                row_dict[col_name] = row[i]

        def parse_json_field(value):
            if value and isinstance(value, str):
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return None
            elif value and not isinstance(value, (dict, list)):
                return None
            return value

        # Parse JSON fields
        if "warnings" in row_dict:
            row_dict["warnings"] = parse_json_field(row_dict["warnings"])
        if "errors" in row_dict:
            row_dict["errors"] = parse_json_field(row_dict["errors"])
        if "err_msgs" in row_dict:
            row_dict["err_msgs"] = parse_json_field(row_dict["err_msgs"])

        # Create instance using column names (which match field names)
        return cls(**row_dict)

    def to_dict(self) -> Dict[str, Any]:

        data = asdict(self)

        data = {k: v for k, v in data.items()}

        # Convert JSON fields
        if "warnings" in data and isinstance(data["warnings"], (dict, list)):
            data["warnings"] = json.dumps(data["warnings"])
        if "errors" in data and isinstance(data["errors"], (dict, list)):
            data["errors"] = json.dumps(data["errors"])
        if "err_msgs" in data and isinstance(data["err_msgs"], (dict, list)):
            data["err_msgs"] = json.dumps(data["err_msgs"])

        return data

    @classmethod
    def from_file(cls, file: str, process_status_id: int = None):
        cls = cls(image_name=os.path.basename(file.replace(".fits", "")), process_status_id=process_status_id)

        cls.image_path = file

        header = fits.getheader(file)

        keys = list(cls.__annotations__.keys())

        for key in keys:
            possible_keys = [key.replace("_", "-").upper(), key.upper()]
            for header_key in possible_keys:
                if key == "unit":
                    header_key = "TELESCOP"
                if header_key in header:
                    setattr(cls, key, header[header_key])

        if header["IMAGETYP"] == "LIGHT":
            if "_coadd" in file:
                cls.image_type = "coadd"
            elif "_diff" in file:
                cls.image_type = "diff"
            else:
                cls.image_type = "single"

            cls.image_group = "science"
        else:
            cls.image_type = header["IMAGETYP"].lower()
            cls.image_group = "masterframe"

        # multi-epoch coadds live in COADD_DIR/{obj}/{filter}/ -- dateless by design
        m = re.search(r"/\d{4}-\d{2}-\d{2}/", file)
        cls.nightdate = m.group(0).replace("/", "") if m else None

        # Every image the pipeline writes is stamped by add_image_id, so a missing card means
        # the writer skipped it. Refuse the record rather than let update_data keep the id of
        # the version this file replaced: a dependency resolved through that id would look
        # healthy while pointing at pixels that no longer exist.
        if not str(cls.imageid or "").strip():
            raise ValueError(f"No IMAGEID in the header of {file}; refusing to register an image with no identity")

        return cls


class ImageQA(BaseDatabase):
    """Database class for managing image_qa records"""

    def __init__(self, db_params: Optional[Dict[str, Any]] = None, logger=None):
        """Initialize with database parameters"""
        self._table_name = "image_qa"
        self._pyTable = ImageQATable
        self.logger = logger or logging.getLogger(__name__)
        super().__init__(db_params)

    @property
    def table_name(self):
        return self._table_name

    @property
    def pyTable(self):
        return self._pyTable

    @override
    def _find_existing_id(self, data: Dict[str, Any]) -> Optional[int]:
        """One row per name and path: a process reusing the image updates that row, not adds one.

        Keyed on the name rather than IMAGEID, so aliases of one file -- a synth linked under
        several filter names -- keep a row each, while a regenerated file, carrying a fresh
        IMAGEID under the same name, takes over the row of the version it overwrote. The path
        separates those two: same name elsewhere on disk is a different file, so it gets a row.
        """
        name = data.get("image_name")
        if not name:
            return None
        rows, _ = self.execute_query(
            f"SELECT id, image_path FROM {self.table_name} WHERE image_name = %(name)s ORDER BY id",
            {"name": name},
            return_columns=True,
        )
        if not rows:
            return None

        path = data.get("image_path")
        if not path:
            return rows[0][0]

        same_path = next((i for i, p in rows if p == path), None)
        if same_path is not None:
            return same_path
        # a row registered before its file location was known
        pathless = next((i for i, p in rows if p is None), None)
        if pathless is not None:
            return pathless

        self.logger.warning(
            f"{name} is already registered elsewhere on disk, so it is being added as a"
            f" separate row rather than overwriting: incoming {path},"
            f" existing {', '.join(p for _, p in rows)}"
        )
        return None

    def get_by_process_status_id(self, process_status_id: int) -> List[ImageQATable]:
        """Get all image_qa rows that have the given process_status_id"""
        try:
            query = query_by_params.format(
                table_name=self.table_name, params="process_status_id = %(process_status_id)s"
            )
            params = {"process_status_id": process_status_id}

            rows, columns = self.execute_query(query, params, return_columns=True)

            if not rows or len(rows) == 0:
                return []

            return [self.pyTable.from_row(row, columns=columns) for row in rows]

        except Exception as e:
            raise DatabaseError(f"Failed to read {self.table_name} by process_status_id: {e}")

    def classify_images(self, images, group="masterframe") -> List[str]:

        classified = {}

        if group == "masterframe":
            classified["bias"] = False
            classified["dark"] = []
            classified["flat"] = []
            for image in images:
                if image.image_type == "bias":
                    classified["bias"] = True
                elif image.image_type == "dark":
                    classified["dark"].append(image.exptime)
                elif image.image_type == "flat":
                    classified["flat"].append(image.filter)

        elif group == "science":
            for image in images:
                if image.image_type == "science":
                    classified.setdefault(image.object, [])
                    classified[image.object].append(image.filter)

        return classified
