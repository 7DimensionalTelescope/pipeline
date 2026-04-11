import os

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

# Internal paths
SOURCE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.abspath(os.path.join(SOURCE_DIR, ".."))
REF_DIR = os.path.abspath(os.path.join(ROOT_DIR, "ref"))
SCRIPTS_DIR = os.path.join(SOURCE_DIR, "cli")  # "scripts"
STORAGE_CONFIG_FILE = os.path.join(REF_DIR, "storage.yml")

# load environment variables from .env file when python-dotenv is available
if load_dotenv is not None:
    load_dotenv(os.path.join(ROOT_DIR, ".env"), override=True)

if yaml is not None:
    with open(STORAGE_CONFIG_FILE, encoding="utf-8") as stream:
        _storage_config = yaml.safe_load(stream) or {}

    if not isinstance(_storage_config, dict):
        raise ValueError(f"Storage config must contain a mapping: {STORAGE_CONFIG_FILE}")
else:
    _storage_config = {}

_storage_paths = _storage_config.get("storage") or {}
_reference_paths = _storage_config.get("references") or {}
_external_paths = _storage_config.get("external") or {}


# Storage Configuration
RAWDATA_DIR = _storage_paths.get("RAWDATA_DIR") or "/lyman/data1/obsdata"
FACTORY_DIR = _storage_paths.get("FACTORY_DIR") or "/lyman/data2/factory"
MASTER_FRAME_DIR = _storage_paths.get("MASTER_FRAME_DIR") or "/lyman/data2/master_frame"
PROCESSED_DIR = _storage_paths.get("PROCESSED_DIR") or "/lyman/data2/processed"
TOO_PROCESSED_DIR = _storage_paths.get("TOO_PROCESSED_DIR") or "/lyman/data2/too"
TOO_FACTORY_DIR = _storage_paths.get("TOO_FACTORY_DIR") or "/lyman/data2/too_factory"
COADD_DIR = _storage_paths.get("COADD_DIR") or "/lyman/data2/coadd"

# Next disk
DISK_CHANGE_NIGHTDATE = _storage_paths.get("DISK_CHANGE_NIGHTDATE") or "2026-04-08"
MASTER_FRAME_DIR_2 = _storage_paths.get("MASTER_FRAME_DIR_2")
FACTORY_DIR_2 = _storage_paths.get("FACTORY_DIR_2")
PROCESSED_DIR_2 = _storage_paths.get("PROCESSED_DIR_2")
TOO_PROCESSED_DIR_2 = _storage_paths.get("TOO_PROCESSED_DIR_2")
TOO_FACTORY_DIR_2 = _storage_paths.get("TOO_FACTORY_DIR_2")

# Next disk
DISK_CHANGE_NIGHTDATE_2 = _storage_paths.get("DISK_CHANGE_NIGHTDATE_2") or "2027-01-10"

# collection of pipeline directories
PIPELINE_DIRS = {
    path
    for path in {
        RAWDATA_DIR,
        FACTORY_DIR,
        FACTORY_DIR_2,
        MASTER_FRAME_DIR,
        MASTER_FRAME_DIR_2,
        PROCESSED_DIR,
        PROCESSED_DIR_2,
        COADD_DIR,
        TOO_PROCESSED_DIR,
        TOO_FACTORY_DIR,
    }
    if path is not None
}

# Paths to pre-generated data
ASTRM_TILE_REF_DIR = _reference_paths.get("ASTRM_TILE_REF_DIR", "/lyman/data2/py7dt_requisites/ref_scamp/gaia_dr3_7DS")
ASTRM_CUSTOM_REF_DIR = _reference_paths.get(
    "ASTRM_CUSTOM_REF_DIR", "/lyman/data2/py7dt_requisites/ref_scamp/gaia_dr3_custom"
)
GAIA_ROOT_DIR = _reference_paths.get("GAIA_ROOT_DIR", "/lyman/data1/factory/catalog/gaia_source_dr3/healpix_nside64")
SCAMP_QUERY_DIR = _reference_paths.get("SCAMP_QUERY_DIR", "/lyman/data2/py7dt_requisites/ref_scamp/queried")
PHOT_REF_DIR = _reference_paths.get("PHOT_REF_DIR", "/lyman/data1/factory/ref_cat")
GAIA_REF_DIR = _reference_paths.get("GAIA_REF_DIR", "/lyman/data1/Calibration/7DT-Calibration/output/Calibration_Tile")
REF_IMAGE_DIR = _reference_paths.get("REF_IMAGE_DIR", "/lyman/data1/factory/ref_frame")

# define a collection
REQUISITE_DIRS = {
    ASTRM_TILE_REF_DIR,
    ASTRM_CUSTOM_REF_DIR,
    GAIA_ROOT_DIR,
    SCAMP_QUERY_DIR,
    PHOT_REF_DIR,
    GAIA_REF_DIR,
    REF_IMAGE_DIR,
}

# Miscellaneous
SERVICES_TMP_DIR = _storage_paths.get("SERVICES_TMP_DIR") or "/tmp/pipeline"
SLACK_TOKEN = os.environ.get("SLACK_TOKEN", None)
INSTRUM_STATUS_DICT = _external_paths.get("INSTRUM_STATUS_DICT")
SEXTRACTOR_COMMAND = os.environ.get("SEXTRACTOR_COMMAND") or "source-extractor"
RECENT_RAWDATA_TRANSFER_HISTORY = _external_paths.get("RECENT_RAWDATA_TRANSFER_HISTORY")


# database access
EMAIL_USER = os.environ.get("EMAIL_USER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
DEFAULT_RECIPIENT = os.environ.get("DEFAULT_RECIPIENT")
SCHEDULER_DB_PATH = _external_paths.get("SCHEDULER_DB_PATH")
QUEUE_SOCKET_PATH = _external_paths.get("QUEUE_SOCKET_PATH")
TOO_DB_PATH = _external_paths.get("TOO_DB_PATH")
