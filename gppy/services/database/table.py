from datetime import date, datetime
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass, asdict
import json
from .utils import generate_id
from astropy.io.fits.header import Header


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
    bias: Optional[bool] = False
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
    filename: Optional[str] = None
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
            try:
                from datetime import datetime

                # Handle YYYY-MM-DD format (most common from config names)
                if len(self.run_date) == 10 and self.run_date.count("-") == 2:
                    self.run_date = datetime.strptime(self.run_date, "%Y-%m-%d").date()
                else:
                    # Try ISO format for other cases
                    self.run_date = date.fromisoformat(self.run_date)
            except ValueError:
                # If date parsing fails, keep as string
                pass

        # Ensure filter lists are lists
        if not isinstance(self.dark, list):
            self.dark = []
        if not isinstance(self.flat, list):
            self.flat = []

    @classmethod
    def from_row(cls, row: tuple) -> "PipelineData":
        """Create PipelineData from database row"""
        # Add bounds checking to prevent index errors
        if len(row) < 32:
            raise ValueError(f"Expected at least 32 columns, got {len(row)}. Row: {row}")

        # Parse JSON fields with safe access
        dark = json.loads(row[10]) if row[10] and isinstance(row[10], str) else []
        flat = json.loads(row[11]) if row[11] and isinstance(row[11], str) else []

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
            dark=dark,
            flat=flat,
            warnings=row[12],
            errors=row[13],
            comments=row[14],
            config_file=row[15],
            log_file=row[16],
            debug_file=row[17],
            comments_file=row[18],
            output_combined_frame_id=row[19],
            created_at=row[20],
            updated_at=row[21],
            filename=row[22],
            param2=row[23],
            param3=row[24],
            param4=row[25],
            param5=row[26],
            param6=row[27],
            param7=row[28],
            param8=row[29],
            param9=row[30],
            param10=row[31],
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
        elif data_type == "science":
            obj, filt, run_date = config.name.split("_")
            pipeline_data = cls(
                tag_id=process_id,
                run_date=run_date,
                data_type=data_type,
                obj=obj,
                filt=filt,
                unit=None,
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
    imagetyp: str  # "masterframe" or "science"
    pipeline_id_id: int

    # Optional fields
    qa_type: Optional[str] = None  # "bias", "dark", "flat" for masterframe; "science" for science
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # QA for masterframe (when qa_type="masterframe")
    clipmed: Optional[float] = None
    clipstd: Optional[float] = None
    clipmin: Optional[float] = None
    clipmax: Optional[float] = None
    nhotpix: Optional[int] = None
    ntotpix: Optional[int] = None
    uniform: Optional[str] = None  
    sigmean: Optional[str] = None  
    edgevar: Optional[str] = None  
    trimmed: Optional[str] = None  

    # QA for science (when qa_type="science")
    filter: Optional[str] = None
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
    exptime: Optional[str] = None
    unmatch: Optional[str] = None
    rsep_rms: Optional[str] = None
    rsep_q2: Optional[str] = None
    rsep_p95: Optional[str] = None
    awincrmn: Optional[str] = None
    ellipmn: Optional[str] = None
    pa_align: Optional[str] = None

    # filename
    filename: Optional[str] = None

    # sanity flag
    sanity: Optional[str] = None

    # Internal field (not stored in DB)
    id: Optional[int] = None

    @classmethod
    def from_row(cls, row: tuple) -> "QAData":
        """Create QAData from database row"""
        # Add bounds checking to prevent index errors
        if len(row) < 38:
            raise ValueError(f"Expected at least 38 columns, got {len(row)}. Row: {row}")

        return cls(
            id=row[0],
            qa_id=row[1],
            qa_type=row[2],
            imagetyp=row[3],
            filter=row[4],
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
            edgevar=row[24],
            exptime=row[25],
            filename=row[26],
            sanity=row[27],
            sigmean=row[28],
            trimmed=row[29],
            unmatch=row[30],
            rsep_rms=row[31],
            rsep_q2=row[32],
            uniform=row[33],
            awincrmn=row[34],
            ellipmn=row[35],
            rsep_p95=row[36],
            pa_align=row[37],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database operations"""
        data = asdict(self)
        # Remove None values and internal fields
        data = {k: v for k, v in data.items() if v is not None and k != "id"}

        return data

    @classmethod
    def from_header(cls, header, imagetyp, qa_type, pipeline_id, filename):

        if not isinstance(header, Header):
            from astropy.io import fits

            header = fits.getheader(header)

        qa_id = generate_id()

        qa_data = cls(
            qa_id=qa_id,
            imagetyp=imagetyp,
            qa_type=qa_type.lower(),
            pipeline_id_id=pipeline_id,
        )

        if qa_data.imagetyp == "masterframe":
            keywords = ["EXPTIME", "FILTER", "CLIPMED", "CLIPSTD", "CLIPMIN", "CLIPMAX", "SANITY"]

            for keyword in keywords:
                if keyword in header:
                    setattr(qa_data, keyword.lower(), header[keyword])

            qa_data.filename = filename
            if qa_data.qa_type == "dark":
                qa_data.nhotpix = header["NHOTPIX"]
                qa_data.ntotpix = header["NTOTPIX"]
                qa_data.uniform = header["UNIFORM"]
            elif qa_data.qa_type == "flat":
                qa_data.sigmean = header["SIGMEAN"]
                qa_data.edgevar = header["EDGEVAR"]
                qa_data.trimmed = header["TRIMMED"]

        elif qa_data.imagetyp == "science":
            keywords = [
                "FILTER",
                "SEEING",
                "ELLIPTICITY",
                "ROTANG1",
                "ASTROMETRIC_OFFSET",
                "SKYVAL",
                "SKYSIG",
                "ZP_AUTO",
                "EZP_AUTO",
                "UL5_5",
                "STDNUMB",
                "EXPTIME",
                "SANITY",
            ]

            for keyword in keywords:
                if keyword in header:
                    setattr(qa_data, keyword.lower(), header[keyword])

            qa_data.filename = filename

        return qa_data

    @classmethod
    def check_header(cls, header, qa_type):
        if qa_type.startswith("bias") or qa_type.startswith("dark") or qa_type.startswith("flat"):
            keywords = ["EXPTIME", "FILTER", "CLIPMED", "CLIPSTD", "CLIPMIN", "CLIPMAX", "SANITY"]
            if qa_type == "dark":
                keywords.extend(["NHOTPIX", "NTOTPIX", "UNIFORM"])
            elif qa_type == "flat":
                keywords.extend(["SIGMEAN", "EDGEVAR", "TRIMMED"])
        elif qa_type == "science":
            keywords = ["FILTER"]

        for keyword in keywords:
            if keyword not in header:
                return False

        return True
