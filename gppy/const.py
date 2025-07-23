import os

# System paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAWDATA_DIR = os.getenv("RAWDATA_DIR", "/lyman/data1/obsdata/")
FACTORY_DIR = os.getenv("FACTORY_DIR", "/lyman/data2/factory/")
MASTER_FRAME_DIR = os.getenv("MASTER_FRAME_DIR", "/lyman/data2/master_frame/")
PROCESSED_DIR = os.getenv("PROCESSED_DIR", "/lyman/data2/processed/")
DAILY_STACKED_DIR = os.getenv("DAILY_STACKED_DIR", "/lyman/data2/daily_stacked/")
STACKED_DIR = os.getenv("STACKED_DIR", "/lyman/data2/stacked/")

REF_DIR = os.path.join(SCRIPT_DIR, "ref")

PIPELINE_DIRS = {RAWDATA_DIR, FACTORY_DIR, MASTER_FRAME_DIR, PROCESSED_DIR, STACKED_DIR}
SLACK_TOKEN = os.getenv("SLACK_TOKEN", None)
INSTRUM_STATUS_DICT = "/home/7dt/7dt_too/backend/data/7dt/multitelescopes.dict"


# Image grouping structure
INSTRUM_GROUP_KEYS = ["unit", "n_binning", "gain", "camera"]
ALL_GROUP_KEYS = ["obj", "filter", "nightdate", "exptime"] + INSTRUM_GROUP_KEYS
BIAS_GROUP_KEYS = ["nighdate"] + INSTRUM_GROUP_KEYS  # no exp: account for potential ms exp difference
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
