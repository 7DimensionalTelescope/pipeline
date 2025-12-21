import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)


# Bashrc-defined system paths
RAWDATA_DIR = os.environ.get("RAWDATA_DIR") or "/lyman/data1/obsdata"  # , "/lyman/data1/obsdata/")
FACTORY_DIR = os.environ.get("FACTORY_DIR") or "/lyman/data2/factory"
MASTER_FRAME_DIR = os.environ.get("MASTER_FRAME_DIR") or "/lyman/data2/master_frame"
PROCESSED_DIR = os.environ.get("PROCESSED_DIR") or "/lyman/data2/processed"
TOO_DIR = os.environ.get("TOO_DIR") or "/lyman/data2/too"
TOO_FACTORY_DIR = os.environ.get("TOO_FACTORY_DIR") or "/lyman/data2/too_factory"

STACKED_DIR = os.environ.get("STACKED_DIR") or "/lyman/data2/stacked"
SLACK_TOKEN = os.environ.get("SLACK_TOKEN", None)
INSTRUM_STATUS_DICT = os.environ.get("INSTRUM_STATUS_DICT") or (
    x if os.path.exists(x := "/home/7dt/7dt_too/backend/data/7dt/multitelescopes.dict") else None
)
SEXTRACTOR_COMMAND = os.environ.get("SEXTRACTOR_COMMAND") or "source-extractor"

# Paths to pre-generated data
ASTRM_TILE_REF_DIR = "/lyman/data2/py7dt_requisites/ref_scamp/gaia_dr3_7DS"  # fmt: skip # "/lyman/data1/factory/catalog/gaia_dr3_7DT"
ASTRM_CUSTOM_REF_DIR = "/lyman/data2/py7dt_requisites/ref_scamp/gaia_dr3_custom"
GAIA_ROOT_DIR = "/lyman/data1/factory/catalog/gaia_source_dr3/healpix_nside64"  # for dynamic refcat generation.
SCAMP_QUERY_DIR = "/lyman/data2/py7dt_requisites/ref_scamp/queried"  # "/lyman/data1/factory/ref_scamp"
PHOT_REF_DIR = "/lyman/data1/factory/ref_cat"  # divided by RIS tiles
GAIA_REF_DIR = "/lyman/data1/Calibration/7DT-Calibration/output/Calibration_Tile"
REF_IMAGE_DIR = "/lyman/data1/factory/ref_frame"


# Internal paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(SCRIPT_DIR, "..", "ref")
PIPELINE_DIRS = {RAWDATA_DIR, FACTORY_DIR, MASTER_FRAME_DIR, PROCESSED_DIR, STACKED_DIR, TOO_DIR, TOO_FACTORY_DIR}


# Image grouping structure
INSTRUM_GROUP_KEYS = ["unit", "n_binning", "gain", "camera"]
ALL_GROUP_KEYS = ["obj", "filter", "nightdate", "exptime"] + INSTRUM_GROUP_KEYS
BIAS_GROUP_KEYS = ["nightdate"] + INSTRUM_GROUP_KEYS  # no exp: account for potential ms exp difference
DARK_GROUP_KEYS = BIAS_GROUP_KEYS + ["exptime"]  # darks have arbitrary filters
FLAT_GROUP_KEYS = BIAS_GROUP_KEYS + ["filter"]  # flats have different exptimes
SURVEY_SCIENCE_GROUP_KEYS = ["obj", "filter"]  # , "n_binning", "unit"]
TRANSIENT_SCIENCE_GROUP_KEYS = ["nightdate"] + SURVEY_SCIENCE_GROUP_KEYS  # used for processed image directory structure


# OBS-related
CalibType = ["BIAS", "DARK", "FLAT"]
available_7dt_units = [f"7DT0{unit}" if unit < 10 else f"7DT{unit}" for unit in range(1, 20)]
WIDE_FILTERS = ["m375w", "m425w"]
MEDIUM_FILTERS = [f"m{s}" for s in range(400, 900, 25)]
BROAD_FILTERS = ["u", "g", "r", "i", "z"]
ALL_FILTERS = WIDE_FILTERS + MEDIUM_FILTERS + BROAD_FILTERS
PIXSCALE = 0.505  # arcsec/pixel. Default plate scale assumed prior to astrometric solving
NUM_MIN_CALIB = 5  # 2


FILTER_WAVELENGTHS = {
    "m375w": 3750,
    "m425w": 4250,
    "u": 3500,
    "g": 4750,
    "r": 6250,
    "i": 7700,
    "z": 9000,
}
for w in range(400, 900, 25):
    FILTER_WAVELENGTHS[f"m{w}"] = w * 10

FILTER_WIDTHS = {
    "m375w": 250,
    "m425w": 250,
    "u": 600,
    "g": 1150,
    "r": 1150,
    "i": 1000,
    "z": 1000,
}
for w in range(400, 900, 25):
    FILTER_WIDTHS[f"m{w}"] = 250


HEADER_KEY_MAP = {
    "exptime": "EXPOSURE",
    "gain": "GAIN",
    "filter": "FILTER",
    # "nightdate": "DATE-LOC",
    # "date_loc": "DATE-LOC",
    "obstime": "DATE-OBS",
    "obj": "OBJECT",
    "unit": "TELESCOP",
    "n_binning": "XBINNING",
    "ra": "OBJCTRA",  # intended pointing, not the actual mount position
    "dec": "OBJCTDEC",
}


# Computing Hardware
NUM_GPUS = 2  # maximum number of GPUs to use
MAX_WORKERS = 128

# Storage Configuration
DISK_CHANGE_DATE = "20260101"

EMAIL_USER = os.environ.get("EMAIL_USER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")

DEFAULT_RECIPIENT = os.environ.get("DEFAULT_RECIPIENT")

SCHEDULER_DB_PATH = os.environ.get("SCHEDULER_DB_PATH")

QUEUE_SOCKET_PATH = os.environ.get("QUEUE_SOCKET_PATH")

TOO_DB_PATH = os.environ.get("TOO_DB_PATH")


class PipelineError(Exception):
    pass
