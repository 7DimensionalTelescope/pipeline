from __future__ import annotations

# increase version with ANY change in scientific config. e.g., prep.sex
__version__ = "1.10.3"

MIN_PREPROC_RUNTIME_VERSION = "1.10.3"
MIN_SCIPROC_RUNTIME_VERSION = "1.9.6"
MIN_SCIPROC_RUNTIME_VERSION_MAP = {
    "astrometry": "1.9.6",
    "photometry": "1.8.14",
    "imcoadd": "1.8.12",
    "imsubtract": "1.8.12",
}


def is_below_min(recorded, minimum: str) -> bool:
    """True if `recorded` is missing or strictly older than `minimum`."""
    if not recorded:
        return True
    try:
        return tuple(int(p) for p in str(recorded).split(".")) < tuple(int(p) for p in minimum.split("."))
    except Exception:
        return True
