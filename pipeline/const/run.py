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
