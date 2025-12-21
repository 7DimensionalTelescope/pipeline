from __future__ import annotations
from datetime import date, datetime
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass, asdict
import json
import re
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
    def from_row(cls, row: tuple) -> PipelineData:
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

            # Get log file path - handle case where logging might not be initialized
            if hasattr(config, "logging") and hasattr(config.logging, "file"):
                log_file = config.logging.file
            elif (
                hasattr(config, "config")
                and hasattr(config.config, "logging")
                and hasattr(config.config.logging, "file")
            ):
                log_file = config.config.logging.file
            else:
                # Default log file path if logging is not available
                log_file = f"/tmp/{config.name}.log"

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
                log_file=log_file,
                debug_file=log_file.replace(".log", "_debug.log"),
                comments_file=log_file.replace(".log", "_comments.txt"),
                output_combined_frame_id=None,
            )
        elif data_type == "science":
            obj, filt, run_date = config.name.split("_")
            # Get log file path - handle case where logging might not be initialized
            if hasattr(config, "logging") and hasattr(config.logging, "file"):
                log_file = config.logging.file
            elif (
                hasattr(config, "config")
                and hasattr(config.config, "logging")
                and hasattr(config.config.logging, "file")
            ):
                log_file = config.config.logging.file
            else:
                # Default log file path if logging is not available
                log_file = f"/tmp/{config.name}.log"

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
                log_file=log_file,
                debug_file=log_file.replace(".log", "_debug.log"),
                comments_file=log_file.replace(".log", "_comments.txt"),
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
    date_obs: Optional[str] = None
    # Optional fields
    qa_type: Optional[str] = None  # "bias", "dark", "flat" for masterframe; "science" for science
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    filter: Optional[str] = None
    exptime: Optional[str] = None

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

    visible: Optional[float] = None

    # Astrometry QA
    rotang: Optional[float] = None
    unmatch: Optional[str] = None
    rsep_rms: Optional[str] = None
    rsep_q2: Optional[str] = None
    rsep_p95: Optional[str] = None
    awincrmn: Optional[str] = None
    ellipmn: Optional[str] = None
    pa_align: Optional[str] = None
    ptnoff: Optional[float] = None

    # Photomtry QA
    seeing: Optional[float] = None
    peeing: Optional[float] = None
    skyval: Optional[float] = None
    skysig: Optional[float] = None
    zp_auto: Optional[float] = None
    ezp_auto: Optional[float] = None
    ul5_5: Optional[float] = None
    stdnumb: Optional[int] = None

    # filename
    filename: Optional[str] = None

    # unit
    unit: Optional[str] = None

    # sanity flag
    sanity: Optional[str] = None
    q_desc: Optional[str] = None
    eye_insp: Optional[str] = None

    # Telescope pointing
    alt: Optional[float] = None
    az: Optional[float] = None
    seeingmn: Optional[float] = None

    # Internal field (not stored in DB)
    id: Optional[int] = None

    @classmethod
    def from_row(cls, row: tuple) -> QAData:
        """Create QAData from database row"""
        # Add bounds checking to prevent index errors
        if len(row) < 24:
            raise ValueError(f"Expected at least 24 columns, got {len(row)}. Row: {row}")

        return cls(
            id=row[0],
            qa_id=row[1],
            qa_type=row[2],
            imagetyp=row[3],
            filter=row[4] if len(row) > 4 else None,
            clipmed=row[5] if len(row) > 5 else None,
            clipstd=row[6] if len(row) > 6 else None,
            clipmin=row[7] if len(row) > 7 else None,
            clipmax=row[8] if len(row) > 8 else None,
            nhotpix=row[9] if len(row) > 9 else None,
            ntotpix=row[10] if len(row) > 10 else None,
            seeing=row[11] if len(row) > 11 else None,
            rotang=row[12] if len(row) > 12 else None,
            ptnoff=row[13] if len(row) > 13 else None,
            skyval=row[14] if len(row) > 14 else None,
            skysig=row[15] if len(row) > 15 else None,
            zp_auto=row[16] if len(row) > 16 else None,
            ezp_auto=row[17] if len(row) > 17 else None,
            ul5_5=row[18] if len(row) > 18 else None,
            stdnumb=row[19] if len(row) > 19 else None,
            created_at=row[20] if len(row) > 20 else None,
            updated_at=row[21] if len(row) > 21 else None,
            pipeline_id_id=row[22] if len(row) > 22 else None,
            edgevar=row[23] if len(row) > 23 else None,
            exptime=row[24] if len(row) > 24 else None,
            filename=row[25] if len(row) > 25 else None,
            sanity=row[26] if len(row) > 26 else None,
            sigmean=row[27] if len(row) > 27 else None,
            trimmed=row[28] if len(row) > 28 else None,
            unmatch=row[29] if len(row) > 29 else None,
            rsep_rms=row[30] if len(row) > 30 else None,
            rsep_q2=row[31] if len(row) > 31 else None,
            uniform=row[32] if len(row) > 32 else None,
            awincrmn=row[33] if len(row) > 33 else None,
            ellipmn=row[34] if len(row) > 34 else None,
            rsep_p95=row[35] if len(row) > 35 else None,
            pa_align=row[36] if len(row) > 36 else None,
            eye_insp=row[37] if len(row) > 37 else None,
            peeing=row[38] if len(row) > 38 else None,
            q_desc=row[39] if len(row) > 39 else None,
            visible=row[40] if len(row) > 40 else None,
            date_obs=row[41] if len(row) > 41 else None,
            unit=row[42] if len(row) > 42 else None,
            alt=row[43] if len(row) > 43 else None,
            az=row[44] if len(row) > 44 else None,
            seeingmn=row[45] if len(row) > 45 else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database operations"""
        data = asdict(self)
        # Remove None values and internal fields
        data = {k: v for k, v in data.items() if v is not None and k != "id"}

        return data

    @classmethod
    def from_header(cls, header: str | Header, imagetyp, qa_type, pipeline_id, filename):

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

        if "DATE-OBS" in header:
            # DATE-OBS in FITS is UTC (as per FITS standard and header comments)
            # Parse as UTC, then convert to KST (Korea Standard Time, UTC+9) for storage
            from datetime import datetime
            import pytz

            date_obs_str = str(header["DATE-OBS"])
            try:
                # Parse the date string (format: "2025-11-17T01:50:24.000" or "2025-11-17 01:50:24.000")
                if "T" in date_obs_str:
                    date_obs_dt = datetime.fromisoformat(date_obs_str.replace("T", " ").split(".")[0])
                else:
                    date_obs_dt = datetime.strptime(date_obs_str.split(".")[0], "%Y-%m-%d %H:%M:%S")

                # Localize as UTC (DATE-OBS is always UTC in FITS), then convert to KST
                date_obs_utc = pytz.UTC.localize(date_obs_dt)
                kst = pytz.timezone("Asia/Seoul")  # KST is same as Asia/Seoul (UTC+9)
                date_obs_kst = date_obs_utc.astimezone(kst)
                qa_data.date_obs = date_obs_kst
            except Exception:
                # Fallback to original string if parsing fails
                qa_data.date_obs = date_obs_str

        if qa_data.imagetyp == "masterframe":
            keywords = ["EXPTIME", "FILTER", "CLIPMED", "CLIPSTD", "CLIPMIN", "CLIPMAX", "SANITY"]

            for keyword in keywords:
                if keyword in header:
                    setattr(qa_data, keyword.lower(), header[keyword])

            qa_data.filename = filename
            # Extract unit from TELESCOP header keyword
            if "TELESCOP" in header:
                telescope = str(header["TELESCOP"])
                # Extract unit pattern (e.g., "7DT01", "7DT16") from TELESCOP value
                match = re.search(r"7DT\d{2}", telescope)
                if match:
                    qa_data.unit = match.group(0)
            elif "UNIT" in header:
                # Fallback to UNIT keyword if TELESCOP doesn't contain unit pattern
                qa_data.unit = header["UNIT"]

            # Extract telescope pointing information (available for all image types)
            if "ALTITUDE" in header:
                qa_data.alt = float(header["ALTITUDE"])
            if "AZIMUTH" in header:
                qa_data.az = float(header["AZIMUTH"])

            if qa_data.qa_type == "dark":
                qa_data.ntotpix = header["NTOTPIX"]
                qa_data.uniform = header["UNIFORM"]
                if "NHOTPIX" in header:
                    qa_data.nhotpix = header["NHOTPIX"]
            elif qa_data.qa_type == "flat":
                qa_data.sigmean = header["SIGMEAN"]
                qa_data.edgevar = header["EDGEVAR"]
                qa_data.trimmed = header["TRIMMED"]

        elif qa_data.imagetyp == "science":
            keywords = [
                "FILTER",
                "EXPTIME",
                "VISIBLE",
                # Astrometry QA
                "UNMATCH",
                "RSEP_RMS",
                "RSEP_Q2",
                "RSEP_P95",
                "AWINCRMN",
                "ELLIPMN",
                "PA_ALIGN",
                # Photomtry QA
                "SEEING",
                "PEEING",
                "ROTANG",
                "PTNOFF",
                "SKYVAL",
                "SKYSIG",
                "ZP_AUTO",
                "EZP_AUTO",
                "UL5_5",
                "STDNUMB",
                # Quality Control QA
                "SANITY",
                "EYE_INSP",
                "Q_DESC",
                "SEEINGMN",
            ]

            for keyword in keywords:
                if keyword in header:
                    setattr(qa_data, keyword.lower(), header[keyword])

            qa_data.filename = filename
            # Extract unit from TELESCOP header keyword
            if "TELESCOP" in header:
                telescope = str(header["TELESCOP"])
                # Extract unit pattern (e.g., "7DT01", "7DT16") from TELESCOP value
                match = re.search(r"7DT\d{2}", telescope)
                if match:
                    qa_data.unit = match.group(0)
            elif "UNIT" in header:
                # Fallback to UNIT keyword if TELESCOP doesn't contain unit pattern
                qa_data.unit = header["UNIT"]

            # Extract telescope pointing information
            if "ALTITUDE" in header:
                qa_data.alt = float(header["ALTITUDE"])
            if "AZIMUTH" in header:
                qa_data.az = float(header["AZIMUTH"])

            # Extract seeingmn if available (might be SEEINGMN or calculated from SEEING)
            if "SEEINGMN" in header:
                qa_data.seeingmn = float(header["SEEINGMN"])
            elif "SEEING" in header and qa_data.seeingmn is None:
                # If SEEINGMN not available, use SEEING as seeingmn (mean seeing)
                qa_data.seeingmn = float(header["SEEING"])

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
            keywords = [
                "FILTER",
                "SEEING",
                "PEEING",
                "ROTANG",
                "PTNOFF",
                "VISIBLE",
                "SKYVAL",
                "SKYSIG",
                "ZP_AUTO",
                "EZP_AUTO",
                "UL5_5",
                "STDNUMB",
                "EXPTIME",
                "SANITY",
                "EYE_INSP",
                "Q_DESC",
                # Astrometry QA
                "UNMATCH",
                "RSEP_RMS",
                "RSEP_Q2",
                "RSEP_P95",
                "AWINCRMN",
                "ELLIPMN",
                "PA_ALIGN",
                "SEEINGMN",
            ]

        for keyword in keywords:
            if keyword not in header:
                return False

        return True
