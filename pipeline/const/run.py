# this exists separate from sciproc.py for leaner imports
DEFAULT_SCIDATA_PROCESSES = ["astrometry", "single_photometry", "coadd", "coadd_photometry", "subtract", "difference_photometry"]

# return code policy: 0 = success, 1 = failure
SUCCESS_RETURN_CODE = 0
FAILURE_RETURN_CODE = 1
EMPTY_INPUT_AFTER_SANITY_REJECTION_RETURN_CODE = 2
