from datetime import date, datetime
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass, asdict
import json


@dataclass
class PipelineData:
    """Data class for pipeline data records"""

    # Required fields
    tag_id: str
    run_date: Union[str, date]
    data_type: str
    status: str
    progress: int
    bias: bool
    dark: List[str]
    flat: List[str]
    warnings: int
    errors: int
    comments: int

    # Optional fields
    obj: Optional[str] = None
    filt: Optional[str] = None
    unit: Optional[str] = None

    config_file: Optional[str] = None
    log_file: Optional[str] = None
    debug_file: Optional[str] = None
    comments_file: Optional[str] = None

    output_combined_frame_id: Optional[int] = None
    last_modified: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # Parameter fields
    param1: Optional[str] = None
    param2: Optional[str] = None
    param3: Optional[str] = None
    param4: Optional[str] = None
    param5: Optional[str] = None
    param6: Optional[str] = None
    param7: Optional[str] = None
    param8: Optional[str] = None
    param9: Optional[str] = None
    param10: Optional[str] = None

    # Internal field (not stored in DB)
    id: Optional[int] = None

    def __post_init__(self):
        """Convert date string to date object if needed"""
        if isinstance(self.run_date, str):
            self.run_date = date.fromisoformat(self.run_date)

        # Ensure filter lists are lists
        if not isinstance(self.dark, list):
            self.dark = []
        if not isinstance(self.flat, list):
            self.flat = []

    @classmethod
    def from_row(cls, row: tuple) -> "PipelineData":
        """Create PipelineData from database row"""
        # Parse JSON fields
        dark_filters = json.loads(row[10]) if row[10] and isinstance(row[10], str) else []
        flat_filters = json.loads(row[11]) if row[11] and isinstance(row[11], str) else []

        return cls(
            id=row[0],
            tag_id=row[1],
            run_date=row[2],
            data_type=row[3],
            obj=row[4],
            filt=row[5],
            unit=row[6],
            status=row[7],
            progress=row[8],
            bias=row[9],
            dark=dark_filters,
            flat=flat_filters,
            warnings=row[12],
            errors=row[13],
            comments=row[14],
            config_file=row[15],
            log_file=row[16],
            debug_file=row[17],
            comments_file=row[18],
            output_combined_frame_id=row[19],
            last_modified=row[20],
            created_at=row[21],
            updated_at=row[22],
            param1=row[23],
            param2=row[24],
            param3=row[25],
            param4=row[26],
            param5=row[27],
            param6=row[28],
            param7=row[29],
            param8=row[30],
            param9=row[31],
            param10=row[32],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database operations"""
        data = asdict(self)
        # Remove None values and internal fields
        data = {k: v for k, v in data.items() if v is not None and k != "id"}
        return data


@dataclass
class QAData:
    """Data class for QA data records"""

    # Required fields
    qa_id: str
    qa_type: str
    pipeline_id_id: int

    # Optional fields
    imagetyp: Optional[str] = None
    filter_name: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # QA for masterframe
    clipmed: Optional[float] = None
    clipstd: Optional[float] = None
    clipmin: Optional[float] = None
    clipmax: Optional[float] = None
    nhotpix: Optional[int] = None
    ntotpix: Optional[int] = None
    qa1: Optional[str] = None  # to be uniform
    qa2: Optional[str] = None  # to be sigmean
    qa3: Optional[str] = None  # to be edgevar
    qa4: Optional[str] = None  # to be trimmed

    # QA for science
    seeing: Optional[float] = None
    ellipticity: Optional[float] = None
    rotang1: Optional[float] = None
    astrometric_offset: Optional[float] = None
    skyval: Optional[float] = None
    skysig: Optional[float] = None
    zp_auto: Optional[float] = None
    ezp_auto: Optional[float] = None
    ul5_5: Optional[float] = None
    stdnumb: Optional[int] = None

    # QA parameter fields
    qa5: Optional[str] = None
    qa6: Optional[str] = None
    qa7: Optional[str] = None
    qa8: Optional[str] = None
    qa9: Optional[str] = None

    # Passed QA
    qa10: Optional[str] = None  # to be sanity

    # Internal field (not stored in DB)
    id: Optional[int] = None

    @classmethod
    def from_row(cls, row: tuple) -> "QAData":
        """Create QAData from database row"""
        return cls(
            id=row[0],
            qa_id=row[1],
            qa_type=row[2],
            pipeline_id_id=row[3],
            imagetyp=row[4],
            filter_name=row[5],
            clipmed=row[6],
            clipstd=row[7],
            clipmin=row[8],
            clipmax=row[9],
            nhotpix=row[10],
            ntotpix=row[11],
            seeing=row[12],
            ellipticity=row[13],
            rotang1=row[14],
            astrometric_offset=row[15],
            skyval=row[16],
            skysig=row[17],
            zp_auto=row[18],
            ezp_auto=row[19],
            ul5_5=row[20],
            stdnumb=row[21],
            created_at=row[22],
            updated_at=row[23],
            qa1=row[24],
            qa2=row[25],
            qa3=row[26],
            qa4=row[27],
            qa5=row[28],
            qa6=row[29],
            qa7=row[30],
            qa8=row[31],
            qa9=row[32],
            qa10=row[33],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database operations"""
        data = asdict(self)
        # Remove None values and internal fields
        data = {k: v for k, v in data.items() if v is not None and k != "id"}
        return data
