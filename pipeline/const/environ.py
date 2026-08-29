import os
import warnings

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None


def _get_bool_env(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

# Internal paths
SOURCE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.abspath(os.path.join(SOURCE_DIR, ".."))
REF_DIR = os.path.abspath(os.path.join(ROOT_DIR, "ref"))
SCRIPTS_DIR = os.path.join(SOURCE_DIR, "cli")  # "scripts"
DEPLOYMENT_CONFIG_FILE = os.path.join(REF_DIR, "deployment.yml")
# Transitional: drop once every deployment has migrated off ref/storage.yml.
_LEGACY_CONFIG_FILE = os.path.join(REF_DIR, "storage.yml")
if not os.path.exists(DEPLOYMENT_CONFIG_FILE) and os.path.exists(_LEGACY_CONFIG_FILE):
    warnings.warn(
        f"{_LEGACY_CONFIG_FILE} is deprecated; rename it to {DEPLOYMENT_CONFIG_FILE}",
        DeprecationWarning,
        stacklevel=2,
    )
    DEPLOYMENT_CONFIG_FILE = _LEGACY_CONFIG_FILE
STORAGE_CONFIG_FILE = DEPLOYMENT_CONFIG_FILE  # backward-compatible alias

# load environment variables from .env file when python-dotenv is available
if load_dotenv is not None:
    load_dotenv(os.path.join(ROOT_DIR, ".env"), override=True)

if yaml is not None:
    with open(DEPLOYMENT_CONFIG_FILE, encoding="utf-8") as stream:
        _deployment_config = yaml.safe_load(stream) or {}

    if not isinstance(_deployment_config, dict):
        raise ValueError(f"Deployment config must contain a mapping: {DEPLOYMENT_CONFIG_FILE}")
else:
    _deployment_config = {}

_storage_paths = _deployment_config.get("storage") or {}
_reference_paths = _deployment_config.get("references") or {}
_external_paths = _deployment_config.get("external") or {}
_commands = _deployment_config.get("commands") or {}
_database_settings = _deployment_config.get("database") or {}


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
SEPP_CONFIG = _reference_paths.get("SEPP_CONFIG", "/lyman/data1/7DS/RIS/config/7ds_sepp.config")
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
IS_PIPELINE_LOCK = _get_bool_env("IS_PIPELINE", False)
PIPELINE_LOCK_WAIT_SECONDS = 60
# One deployment-wide policy controls every automatic writer of the config graph.
# Explicit maintenance tools remain available when this is disabled.
AUTO_RECORD_PROCESS_STATUS_DEPENDENCIES = _get_bool_env(
    "AUTO_RECORD_PROCESS_STATUS_DEPENDENCIES",
    bool(_database_settings.get("AUTO_RECORD_PROCESS_STATUS_DEPENDENCIES", True)),
)
SLACK_TOKEN = os.environ.get("SLACK_TOKEN", None)
INSTRUM_STATUS_DICT = _external_paths.get("INSTRUM_STATUS_DICT")
SEXTRACTOR_COMMAND = os.environ.get("SEXTRACTOR_COMMAND") or _commands.get("SEXTRACTOR_COMMAND") or "source-extractor"
SWARP_COMMAND = os.environ.get("SWARP_COMMAND") or _commands.get("SWARP_COMMAND") or "swarp"
RECENT_RAWDATA_TRANSFER_HISTORY = _external_paths.get("RECENT_RAWDATA_TRANSFER_HISTORY")
PIPELINE_LOG_DIR = _external_paths.get("PIPELINE_LOG_DIR") or "/var/log/pipeline"
PIPELINE_TRIGGER_LOG_FILE = (
    _external_paths.get("PIPELINE_TRIGGER_LOG_FILE") or "/var/log/pipeline-trigger.log"
)
HIGH_LEVEL_TASK_LOG_FILE = (
    _external_paths.get("HIGH_LEVEL_TASK_LOG_FILE") or os.path.join(PIPELINE_LOG_DIR, "high_level_tasks.log")
)


# Origin host: it owns the scheduler sqlite and the queue daemon. Any other host is a dispatch
# worker, and that is what distinguishes them in process_status.dispatch.
MAIN_HOST = "proton"
# Where a worker reaches it: sshd binds the public interface only, not 10.1.1.51.
MAIN_HOST_ADDRESS = f"{MAIN_HOST}.snu.ac.kr"

# host-local coordination (ephemeral: /var/lock is tmpfs, cleared on boot)
HOST_LOCK_DIR = _external_paths.get("HOST_LOCK_DIR") or "/var/lock/py7dt"

# database access
EMAIL_USER = os.environ.get("EMAIL_USER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
DEFAULT_RECIPIENT = os.environ.get("DEFAULT_RECIPIENT")
SCHEDULER_DB_PATH = _external_paths.get("SCHEDULER_DB_PATH")
QUEUE_SOCKET_PATH = _external_paths.get("QUEUE_SOCKET_PATH")
TOO_DB_PATH = _external_paths.get("TOO_DB_PATH")
