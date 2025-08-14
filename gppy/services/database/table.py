from datetime import date, datetime
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass, asdict
import json
from .utils import generate_id


@dataclass
class PipelineData:
    """Data class for pipeline data records"""

    # Required fields
    tag_id: str
    run_date: Union[str, date]
    data_type: str
    status: str
    progress: int
    warnings: int
    errors: int
    comments: int

    # Optional fields
    bias: Optional[bool] = None
    dark: Optional[List[str]] = None
    flat: Optional[List[str]] = None

    obj: Optional[str] = None
    filt: Optional[str] = None
    unit: Optional[str] = None

    config_file: Optional[str] = None
    log_file: Optional[str] = None
    debug_file: Optional[str] = None
    comments_file: Optional[str] = None

    output_combined_frame_id: Optional[int] = None
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

    @classmethod
    def from_config(cls, config, data_type):

        # Generate process ID
        process_id = generate_id()

        if data_type == "masterframe":
            run_date, unit_name = config.name.split("_")

            pipeline_data = cls(
                tag_id=process_id,
                run_date=run_date,
                data_type=data_type,
                unit=unit_name,
                status="pending",
                progress=0,
                warnings=0,
                errors=0,
                comments=0,
                config_file=config.name,
                log_file=config.logging.file,
                debug_file=config.logging.file.replace(".log", "_debug.log"),
                comments_file=config.logging.file.replace(".log", "_comments.txt"),
                output_combined_frame_id=None,
            )

        return pipeline_data


@dataclass
class QAData:
    """Data class for QA data records"""

    # Required fields
    qa_id: str
    qa_type: str  # "masterframe" or "science"
    pipeline_id_id: int

    # Optional fields
    imagetyp: Optional[str] = None  # "bias", "dark", "flat" for masterframe; "science" for science
    filter_name: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # QA for masterframe (when qa_type="masterframe")
    clipmed: Optional[float] = None
    clipstd: Optional[float] = None
    clipmin: Optional[float] = None
    clipmax: Optional[float] = None
    nhotpix: Optional[int] = None
    ntotpix: Optional[int] = None
    uniform: Optional[str] = None  # qa1
    sigmean: Optional[str] = None  # qa2
    edgevar: Optional[str] = None  # qa3
    trimmed: Optional[str] = None  # qa4

    # QA for science (when qa_type="science")
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

    # Additional QA parameters
    qa5: Optional[str] = None  # bpmask_filename (for dark frames)
    qa6: Optional[str] = None  # n_frames_combined
    qa7: Optional[str] = None  # quality_flag
    qa8: Optional[str] = None  # processing_time
    qa9: Optional[str] = None  # device_used (CPU/GPU)
    qa10: Optional[str] = None  # device_id

    # Internal field (not stored in DB)
    id: Optional[int] = None

    def __post_init__(self):
        """Validate qa_type and imagetyp combinations"""
        valid_qa_types = ["masterframe", "science"]
        if self.qa_type not in valid_qa_types:
            raise ValueError(f"qa_type must be one of {valid_qa_types}")

        if self.qa_type == "masterframe":
            valid_imagetyps = ["bias", "dark", "flat"]
            if self.imagetyp and self.imagetyp not in valid_imagetyps:
                raise ValueError(f"For masterframe qa_type, imagetyp must be one of {valid_imagetyps}")

        elif self.qa_type == "science":
            if self.imagetyp and self.imagetyp != "science":
                self.imagetyp = "science"  # Force science for science QA data

    def is_masterframe(self) -> bool:
        """Check if this is masterframe QA data"""
        return self.qa_type == "masterframe"

    def is_science(self) -> bool:
        """Check if this is science QA data"""
        return self.qa_type == "science"

    def get_frame_type(self) -> Optional[str]:
        """Get the frame type for masterframe QA data"""
        if self.is_masterframe():
            return self.imagetyp
        return None

    def get_masterframe_info(self) -> Dict[str, Any]:
        """Get masterframe-specific information"""
        if not self.is_masterframe():
            return {}

        return {
            "frame_type": self.imagetyp,
            "filter_name": self.filter_name,
            "uniform": self.uniform,  # UNIFORM
            "sigmean": self.sigmean,  # SIGMEAN
            "edgevar": self.edgevar,  # EDGEVAR
            "trimmed": self.trimmed,  # TRIMMED
            "bpmask_filename": self.qa5,
            "n_frames_combined": self.qa6,
            "quality_flag": self.qa7,
            "processing_time": self.qa8,
            "device_used": self.qa9,
            "device_id": self.qa10,
            "clipmed": self.clipmed,
            "clipstd": self.clipstd,
            "clipmin": self.clipmin,
            "clipmax": self.clipmax,
            "nhotpix": self.nhotpix,
            "ntotpix": self.ntotpix,
        }

    def get_science_info(self) -> Dict[str, Any]:
        """Get science-specific information"""
        if not self.is_science():
            return {}

        return {
            "filter_name": self.filter_name,
            "seeing": self.seeing,
            "ellipticity": self.ellipticity,
            "rotang1": self.rotang1,
            "astrometric_offset": self.astrometric_offset,
            "skyval": self.skyval,
            "skysig": self.skysig,
            "zp_auto": self.zp_auto,
            "ezp_auto": self.ezp_auto,
            "ul5_5": self.ul5_5,
            "stdnumb": self.stdnumb,
        }

    @classmethod
    def from_row(cls, row: tuple) -> "QAData":
        """Create QAData from database row"""
        return cls(
            id=row[0],
            qa_id=row[1],
            qa_type=row[2],
            imagetyp=row[3],
            filter_name=row[4],
            clipmed=row[5],
            clipstd=row[6],
            clipmin=row[7],
            clipmax=row[8],
            nhotpix=row[9],
            ntotpix=row[10],
            seeing=row[11],
            ellipticity=row[12],
            rotang1=row[13],
            astrometric_offset=row[14],
            skyval=row[15],
            skysig=row[16],
            zp_auto=row[17],
            ezp_auto=row[18],
            ul5_5=row[19],
            stdnumb=row[20],
            created_at=row[21],
            updated_at=row[22],
            pipeline_id_id=row[23],
            uniform=row[24],
            sigmean=row[25],
            edgevar=row[26],
            trimmed=row[27],
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

        # Map the new field names back to database column names
        if "uniform" in data:
            data["qa1"] = data.pop("uniform")
        if "sigmean" in data:
            data["qa2"] = data.pop("sigmean")
        if "edgevar" in data:
            data["qa3"] = data.pop("edgevar")
        if "trimmed" in data:
            data["qa4"] = data.pop("trimmed")

        return data

    @classmethod
    def from_header(cls, header, imagetyp, pipeline_id):
        qa_id = generate_id()

        qa_data = cls(
            qa_id=qa_id,
            qa_type="masterframe" if imagetyp in ["bias", "dark", "flat"] else "science",
            imagetyp=imagetyp,
            pipeline_id_id=pipeline_id,
        )
        if qa_data.qa_type == "masterframe":
            qa_data.clipmed = header["CLIPMED"]
            qa_data.clipstd = header["CLIPSTD"]
            qa_data.clipmin = header["CLIPMIN"]
            qa_data.clipmax = header["CLIPMAX"]
            qa_data.qa10 = header["SANITY"]
            if imagetyp == "dark":
                qa_data.nhotpix = header["NHOTPIX"]
                qa_data.uniform = header["UNIFORM"]
            elif imagetyp == "flat":
                qa_data.sigmean = header["SIGMEAN"]
                qa_data.edgevar = header["EDGEVAR"]
                qa_data.trimmed = header["TRIMMED"]

        return qa_data
