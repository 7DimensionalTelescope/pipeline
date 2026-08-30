"""Vocabularies of pipeline products: what a file is, and what role it played in making another one.

Every value here is fixed by something outside the package — a Postgres column, a CHECK constraint,
a FITS card or the filename grammar — so the names may be renamed but the strings may not.
"""

# calibration frame kind: the FITS IMAGETYP card lowercased, the masterframe filename token,
# the Preprocess dtype, and the image_qa.image_type of a master frame
CALIB_TYPE_BIAS = "bias"
CALIB_TYPE_DARK = "dark"
CALIB_TYPE_FLAT = "flat"
CALIB_TYPES = (CALIB_TYPE_BIAS, CALIB_TYPE_DARK, CALIB_TYPE_FLAT)

# image_qa.image_type; a master frame carries its CALIB_TYPE, so MASTER_IMAGE_TYPES completes the column
IMAGE_TYPE_SINGLE = "single"
IMAGE_TYPE_COADD = "coadd"
IMAGE_TYPE_DIFF = "diff"
IMAGE_TYPE_WHITE = "white"
# the two families partition the column: a science frame carries one of these, a master frame its CALIB_TYPE
SCIENCE_IMAGE_TYPES = (IMAGE_TYPE_SINGLE, IMAGE_TYPE_COADD, IMAGE_TYPE_DIFF, IMAGE_TYPE_WHITE)
MASTER_IMAGE_TYPES = CALIB_TYPES

# image_qa.image_group
IMAGE_GROUP_SCIENCE = "science"
IMAGE_GROUP_MASTERFRAME = "masterframe"

# image_qa_dependency.dependency_role, pinned by the table's CHECK constraint
BIAS_DEPENDENCY_ROLE = CALIB_TYPE_BIAS
DARK_DEPENDENCY_ROLE = CALIB_TYPE_DARK
FLAT_DEPENDENCY_ROLE = CALIB_TYPE_FLAT
SINGLE_DEPENDENCY_ROLE = IMAGE_TYPE_SINGLE
COADD_DEPENDENCY_ROLE = IMAGE_TYPE_COADD
SCIENCE_DEPENDENCY_ROLE = "science"
REFERENCE_DEPENDENCY_ROLE = "reference"
DIFF_DEPENDENCY_ROLE = IMAGE_TYPE_DIFF
WHITE_DEPENDENCY_ROLE = IMAGE_TYPE_WHITE
IMAGE_DEPENDENCY_ROLES = (
    BIAS_DEPENDENCY_ROLE,
    DARK_DEPENDENCY_ROLE,
    FLAT_DEPENDENCY_ROLE,
    SINGLE_DEPENDENCY_ROLE,
    COADD_DEPENDENCY_ROLE,
    SCIENCE_DEPENDENCY_ROLE,
    REFERENCE_DEPENDENCY_ROLE,
    DIFF_DEPENDENCY_ROLE,
    WHITE_DEPENDENCY_ROLE,
)

# NameType.kind (pipeline/path/name.py)
NAME_TYPE_RAW = "raw"
NAME_TYPE_MASTER = "master"
NAME_TYPE_CALIBRATED = "calibrated"

# NameType.exposure_type: what the exposure holds; the sigma masters carry the base kind plus the suffix
NAME_TYPE_BIAS = CALIB_TYPE_BIAS
NAME_TYPE_DARK = CALIB_TYPE_DARK
NAME_TYPE_FLAT = CALIB_TYPE_FLAT
NAME_TYPE_SCIENCE = "science"
NAME_TYPE_SIGMA_SUFFIX = "sig"

# NameType.image_type is image_qa.image_type itself: IMAGE_TYPE_* above, or the exposure_type of a master

# NameType.product_type
NAME_TYPE_IMAGE = "image"
NAME_TYPE_WEIGHT = "weight"
NAME_TYPE_CATALOG = "catalog"
NAME_TYPE_MASK = "mask"
NAME_TYPE_CONFIG = "config"
NAME_TYPE_TOO = "ToO"  # the ToO marker inside a config filename, not the scheduler input_type
