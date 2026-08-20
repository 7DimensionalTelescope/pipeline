from __future__ import annotations
from packaging.version import Version

# increase version with ANY change in scientific config. e.g., prep.sex
__version__ = "1.10.42"

MIN_PREPROC_RUNTIME_VERSION = "1.10.6"
MIN_SCIPROC_RUNTIME_VERSION = "1.9.6"  # for sciprocess overall
MIN_SCIPROC_RUNTIME_VERSION_MAP = {  # for individual modules
    "astrometry": "1.10.14",
    "photometry": "1.8.14",
    "imcoadd": "1.10.26",
    "imsubtract": "1.10.17",
}
# guard: the overwrite version of overall sciproc >= the min of individual modules'
MIN_SCIPROC_RUNTIME_VERSION = str(
    max(
        min([Version(v) for v in MIN_SCIPROC_RUNTIME_VERSION_MAP.values()]),
        Version(MIN_SCIPROC_RUNTIME_VERSION),
    )
)


def is_below_min(recorded, minimum: str) -> bool:
    """True if `recorded` is missing or strictly older than `minimum`."""
    if not recorded:
        return True
    try:
        return tuple(int(p) for p in str(recorded).split(".")) < tuple(int(p) for p in minimum.split("."))
    except Exception:
        return True
