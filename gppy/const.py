import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# REF_DIR = os.environ["REF_DIR"]
# REF_DIR = os.path.join(SCRIPT_DIR, "gppy", "ref")
REF_DIR = os.path.join(SCRIPT_DIR, "ref")

RAWDATA_DIR = os.environ["RAWDATA_DIR"]
FACTORY_DIR = os.environ["FACTORY_DIR"]
MASTER_FRAME_DIR = os.environ["MASTER_FRAME_DIR"]

PROCESSED_DIR = os.environ["PROCESSED_DIR"]
DAILY_STACKED_DIR = os.environ["DAILY_STACKED_DIR"]
STACKED_DIR = os.environ["STACKED_DIR"]

SLACK_TOKEN = os.environ["SLACK_TOKEN"]

CalibType = ["BIAS", "DARK", "FLAT"]

available_7dt_units = [f"7DT0{unit}" if unit < 10 else f"7DT{unit}" for unit in range(1, 20)]

IMAGE_IDENTIFIERS = {"nightdate", "obj", "filter", "unit", "exposure", "n_binning", "gain", "camera"}

HEADER_KEY_MAP = {
    "exposure": "EXPOSURE",
    "gain": "GAIN",
    "filter": "FILTER",
    # "date_loc": "DATE-LOC",
    "obstime": "DATE-OBS",
    "obj": "OBJECT",
    "unit": "TELESCOP",
    "n_binning": "XBINNING",
    "ra": "OBJCTRA",
    "dec": "OBJCTDEC",
}


class PipelineError(Exception):
    pass
