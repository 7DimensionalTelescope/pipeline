import os
from dotenv import load_dotenv

# Internal paths
SOURCE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.abspath(os.path.join(SOURCE_DIR, ".."))
REF_DIR = os.path.abspath(os.path.join(ROOT_DIR, "ref"))
SCRIPTS_DIR = os.path.join(SOURCE_DIR, "cli")  # "scripts"

# load environment variables from .env file
load_dotenv(os.path.join(ROOT_DIR, ".env"), override=True)


# Bashrc-defined system paths
RAWDATA_DIR = os.environ.get("RAWDATA_DIR") or "/lyman/data1/obsdata"  # , "/lyman/data1/obsdata/")
FACTORY_DIR = os.environ.get("FACTORY_DIR") or "/lyman/data2/factory"
MASTER_FRAME_DIR = os.environ.get("MASTER_FRAME_DIR") or "/lyman/data2/master_frame"
PROCESSED_DIR = os.environ.get("PROCESSED_DIR") or "/lyman/data2/processed"
TOO_PROCESSED_DIR = os.environ.get("TOO_DIR") or "/lyman/data2/too"
TOO_FACTORY_DIR = os.environ.get("TOO_FACTORY_DIR") or "/lyman/data2/too_factory"
STACKED_DIR = os.environ.get("STACKED_DIR") or "/lyman/data2/stacked"

SERVICES_TMP_DIR = os.environ.get("SERVICES_TMP_DIR") or "/tmp/pipeline"

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


# define pipeline directories
PIPELINE_DIRS = {
    RAWDATA_DIR,
    FACTORY_DIR,
    MASTER_FRAME_DIR,
    PROCESSED_DIR,
    STACKED_DIR,
    TOO_PROCESSED_DIR,
    TOO_FACTORY_DIR,
}

REQUISITE_DIRS = {
    ASTRM_TILE_REF_DIR,
    ASTRM_CUSTOM_REF_DIR,
    GAIA_ROOT_DIR,
    SCAMP_QUERY_DIR,
    PHOT_REF_DIR,
    GAIA_REF_DIR,
    REF_IMAGE_DIR,
}


# Storage Configuration
DISK_CHANGE_NIGHTDATE = "2026-02-01"


# database access
EMAIL_USER = os.environ.get("EMAIL_USER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")

DEFAULT_RECIPIENT = os.environ.get("DEFAULT_RECIPIENT")

SCHEDULER_DB_PATH = os.environ.get("SCHEDULER_DB_PATH")

QUEUE_SOCKET_PATH = os.environ.get("QUEUE_SOCKET_PATH")

TOO_DB_PATH = os.environ.get("TOO_DB_PATH")
