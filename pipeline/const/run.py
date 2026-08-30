# this exists separate from sciproc.py for leaner imports
from .sciproc import SCIPROCESS_REGISTRY
from .crossfilter import PHOT7DS_SPEC, WHITE_COADD_SPEC

# process vocabulary is the ProcessSpec name (also the `flag:` key), in run order
DEFAULT_SCIDATA_PROCESSES = [spec.name for spec in SCIPROCESS_REGISTRY.specs]
# white_photometry (WhiteCatalog) is user-input only, not part of the daily chain
DEFAULT_CROSSFILTER_PROCESSES = [WHITE_COADD_SPEC.name, PHOT7DS_SPEC.name]

# return code policy: 0 = success, 1 = failure
SUCCESS_RETURN_CODE = 0
FAILURE_RETURN_CODE = 1
EMPTY_INPUT_AFTER_SANITY_REJECTION_RETURN_CODE = 2

# scheduler table vocabularies; NameHandler.config_properties["config_type"] shares the config_type values
CONFIG_TYPE_PREPROCESS = "preprocess"
CONFIG_TYPE_SCIENCE = "science"
CONFIG_TYPE_CROSSFILTER = "crossfilter"
CONFIG_TYPE_DEBUG = "debug"

INPUT_TYPE_DAILY = "Daily"
INPUT_TYPE_TOO = "ToO"
INPUT_TYPE_REPROCESS = "Reprocess"
INPUT_TYPE_USER = "User-input"

# scheduler task status, unrelated to the free-form process_status.status progress string
TASK_STATUS_READY = "Ready"
TASK_STATUS_PENDING = "Pending"
TASK_STATUS_PROCESSING = "Processing"
TASK_STATUS_COMPLETED = "Completed"
TASK_STATUS_FAILED = "Failed"
TASK_STATUS_REJECTED = "Rejected"
TASK_STATUS_PAUSED = "Paused"
TASK_STATUS_STASHED = "Stashed"
