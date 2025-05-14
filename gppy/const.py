import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# REF_DIR = os.environ["REF_DIR"]
# REF_DIR = os.path.join(SCRIPT_DIR, "gppy", "ref")
REF_DIR = os.path.join(SCRIPT_DIR, "ref")

RAWDATA_DIR = "/lyman/data1/obsdata/"  # os.environ["RAWDATA_DIR"]
FACTORY_DIR = os.environ["FACTORY_DIR"]
MASTER_FRAME_DIR = os.environ["MASTER_FRAME_DIR"]

PROCESSED_DIR = os.environ["PROCESSED_DIR"]
DAILY_STACKED_DIR = os.environ["DAILY_STACKED_DIR"]
STACKED_DIR = os.environ["STACKED_DIR"]

SLACK_TOKEN = os.environ["SLACK_TOKEN"]

CalibType = ["BIAS", "DARK", "FLAT"]

available_7dt_units = [f"7DT0{unit}" if unit < 10 else f"7DT{unit}" for unit in range(1, 20)]

STRICT_KEYS = {"nightdate", "obj", "filter", "unit", "exptime", "n_binning", "gain", "camera"}
ANCILLARY_KEYS = {"ra", "dec", "obstime"}  # hms

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
