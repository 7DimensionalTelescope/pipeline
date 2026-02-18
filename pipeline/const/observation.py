# Image grouping structure
INSTRUM_GROUP_KEYS = ["unit", "n_binning", "gain", "camera"]
ALL_GROUP_KEYS = ["obj", "filter", "nightdate", "exptime"] + INSTRUM_GROUP_KEYS
BIAS_GROUP_KEYS = ["nightdate"] + INSTRUM_GROUP_KEYS  # no exp: account for potential ms exp difference
DARK_GROUP_KEYS = BIAS_GROUP_KEYS + ["exptime"]  # darks have arbitrary filters
FLAT_GROUP_KEYS = BIAS_GROUP_KEYS + ["filter"]  # flats have different exptimes
SURVEY_SCIENCE_GROUP_KEYS = ["obj", "filter"]  # , "n_binning", "unit"]
TRANSIENT_SCIENCE_GROUP_KEYS = ["nightdate"] + SURVEY_SCIENCE_GROUP_KEYS  # used for processed image directory structure

# nightdate is the most lenient; separate logic for it
BIAS_GROUP_LENIENT_KEYS = ["unit"]
DARK_GROUP_LENIENT_KEYS = ["unit"]  # exptime scaling is NYI
FLAT_GROUP_LENIENT_KEYS = ["gain", "camera"]

# OBS-related
CalibType = ["BIAS", "DARK", "FLAT"]
available_7dt_units = [f"7DT0{unit}" if unit < 10 else f"7DT{unit}" for unit in range(1, 20)]
WIDE_FILTERS = ["m375w", "m425w", "m466w", "m692w", "m710w"]
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
    "m375w": 450,  # TODO: check accurate number
    "m425w": 450,  # TODO: check accurate number
    "m386": 270,
    "m438": 280,
    "m466w": 450,
    "m483": 360,
    "m512": 300,
    "m534": 250,
    "m561": 210,
    "m586": 260,
    "m612": 260,
    "m640": 200,
    "m661": 260,
    "m692w": 470,
    "m710w": 470,
    "m769w": 480,
    "m832w": 450,
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
