import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


# Bashrc-defined system paths
RAWDATA_DIR = os.environ.get("RAWDATA_DIR")  # , "/lyman/data1/obsdata/")
FACTORY_DIR = os.environ.get("FACTORY_DIR")
MASTER_FRAME_DIR = os.environ.get("MASTER_FRAME_DIR")
PROCESSED_DIR = os.environ.get("PROCESSED_DIR")
DAILY_STACKED_DIR = os.environ.get("DAILY_STACKED_DIR")
STACKED_DIR = os.environ.get("STACKED_DIR")
SLACK_TOKEN = os.environ.get("SLACK_TOKEN", None)
INSTRUM_STATUS_DICT = os.environ.get("INSTRUM_STATUS_DICT")

# Paths to pre-generated data
ASTRM_REF_DIR = "/lyman/data2/factory/ref_scamp/gaia_dr3_7DT"  #  "/lyman/data1/factory/catalog/gaia_dr3_7DT"
SCAMP_QUERY_DIR = "/lyman/data1/factory/ref_scamp"  # "/lyman/data1/factory/catalog/gaia_dr3_7DT"
PHOT_REF_DIR = "/lyman/data1/factory/ref_cat"  # divided by RIS tiles
GAIA_REF_DIR = "/lyman/data1/Calibration/7DT-Calibration/output/Calibration_Tile"
REF_IMAGE_DIR = "/lyman/data1/factory/ref_frame"

# Internal paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(SCRIPT_DIR, "ref")
PIPELINE_DIRS = {RAWDATA_DIR, FACTORY_DIR, MASTER_FRAME_DIR, PROCESSED_DIR, STACKED_DIR}

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


class PipelineError(Exception):
    pass
