# this exists separate from sciproc.py for leaner imports
from .sciproc import SCIPROCESS_REGISTRY

# process vocabulary is the ProcessSpec name (also the `flag:` key), in run order
DEFAULT_SCIDATA_PROCESSES = [spec.name for spec in SCIPROCESS_REGISTRY.specs]

# return code policy: 0 = success, 1 = failure
SUCCESS_RETURN_CODE = 0
FAILURE_RETURN_CODE = 1
EMPTY_INPUT_AFTER_SANITY_REJECTION_RETURN_CODE = 2
