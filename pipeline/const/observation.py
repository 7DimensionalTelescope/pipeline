import os

import yaml

from .environ import REF_DIR

# Image grouping structure
INSTRUM_GROUP_KEYS = ["unit", "n_binning", "gain", "camera"]
ALL_GROUP_KEYS = ["obj", "filter", "nightdate", "exptime"] + INSTRUM_GROUP_KEYS
BIAS_GROUP_KEYS = ["nightdate"] + INSTRUM_GROUP_KEYS  # no exp: account for potential ms exp difference
DARK_GROUP_KEYS = BIAS_GROUP_KEYS + ["exptime"]  # darks have arbitrary filters
FLAT_GROUP_KEYS = BIAS_GROUP_KEYS + ["filter"]  # flats have different exptimes
SURVEY_SCIENCE_GROUP_KEYS = ["obj", "filter"]  # , "n_binning", "unit"]
TRANSIENT_SCIENCE_GROUP_KEYS = ["nightdate"] + SURVEY_SCIENCE_GROUP_KEYS  # used for processed image directory structure
CROSS_FILTER_GROUP_KEYS = ["nightdate", "obj"]
WHITE_FILTER = "white"

# nightdate is the most lenient; separate logic for it
BIAS_GROUP_LENIENT_KEYS = ["unit"]
DARK_GROUP_LENIENT_KEYS = ["unit"]  # exptime scaling is NYI
FLAT_GROUP_LENIENT_KEYS = ["gain", "camera"]

# OBS-related
CalibType = ["BIAS", "DARK", "FLAT"]
available_7dt_units = [f"7DT0{unit}" if unit < 10 else f"7DT{unit}" for unit in range(1, 21)]
PIXSCALE = 0.505  # arcsec/pixel. Default plate scale assumed prior to astrometric solving
TARGET_PA = 0.0  # deg; commanded field rotation, the reference for PADEV
BIAS_LEVEL = 513  # ADU
EGAIN_2750 = 0.2562  # e-/ADU at camera GAIN 2750
DEFAULT_EXPTIME = 100.0  # s
NUM_MIN_CALIB = 5  # 2


# lazy load variables
_FILTER_CONSTANTS = (
    "FILTERS",
    "WIDE_FILTERS",
    "FIRST_MEDIUM_FILTERS",
    "SECOND_MEDIUM_FILTERS",
    "SECOND_FILTERS",
    "MEDIUM_FILTERS",
    "BROAD_FILTERS",
    "ALL_FILTERS",
    "FILTER_WAVELENGTHS",
    "FILTER_WIDTHS",
    "ZP_MAX_AT",
)


def _load_filters() -> None:
    global FILTERS, WIDE_FILTERS, FIRST_MEDIUM_FILTERS, SECOND_MEDIUM_FILTERS, SECOND_FILTERS, MEDIUM_FILTERS
    global BROAD_FILTERS, ALL_FILTERS, FILTER_WAVELENGTHS, FILTER_WIDTHS, ZP_MAX_AT
    with open(os.path.join(REF_DIR, "filters.yml"), encoding="utf-8") as stream:
        doc = yaml.safe_load(stream)
    FILTERS, ZP_MAX_AT = doc["filters"], doc["zp_max_at"]
    WIDE_FILTERS = [f for f, s in FILTERS.items() if s["family"] == "wide"]
    FIRST_MEDIUM_FILTERS = [f for f, s in FILTERS.items() if s["family"] == "medium" and s["set"] == "first"]
    SECOND_MEDIUM_FILTERS = [f for f, s in FILTERS.items() if s["family"] == "medium" and s["set"] == "second"]
    SECOND_FILTERS = sorted(f for f, s in FILTERS.items() if s["set"] == "second")
    MEDIUM_FILTERS = sorted(FIRST_MEDIUM_FILTERS + SECOND_MEDIUM_FILTERS)
    BROAD_FILTERS = [f for f, s in FILTERS.items() if s["family"] == "broad"]
    ALL_FILTERS = sorted(MEDIUM_FILTERS + WIDE_FILTERS) + BROAD_FILTERS
    FILTER_WAVELENGTHS = {f: s["center_nm"] * 10 for f, s in FILTERS.items()}
    FILTER_WIDTHS = {f: s["fwhm_nm"] * 10 for f, s in FILTERS.items()}


def __getattr__(name: str):
    if name not in _FILTER_CONSTANTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    _load_filters()
    return globals()[name]


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

IMAGE_SIZE_X = 9576
IMAGE_SIZE_Y = 6388
