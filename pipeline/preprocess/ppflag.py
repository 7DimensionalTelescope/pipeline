"""
PPFLAG: Preprocessing quality flag for master frames and science images.

Bit definitions (combined with bitwise OR):
    0: On-the-date raw calib frames (grouped raw calib input directly leads to mframe output)
    1: Different date, but still okay if close enough
    2: Manually generated master frames (not set by pipeline; input may have this. e.g., exptime-scaled dark, superflat)
    4: SANITY was F, but used
    8: Strict search failed; match found by ignoring lenient keys (unit for bias/dark, gain/camera for flat)
    16: Hard keys had to be ignored (not set by pipeline)

Range: 0-31 when all bits set.
"""

import os
from astropy.io import fits

from ..path.utils import get_nightdate
from ..errors import PpflagNotFoundError

# Bit masks for PPFLAG
PPFLAG_RAW_ON_DATE = 0  # implicit when no other bits
PPFLAG_DIFFERENT_DATE = 1
PPFLAG_MANUAL = 2
PPFLAG_SANITY_F_USED = 4
PPFLAG_LENIENT_KEYS_IGNORED = 8
PPFLAG_HARD_KEYS_IGNORED = 16

PPFLAG_KEY = "PPFLAG"
PPFLAG_MAX = 31


def get_ppflag_from_header(header_or_path, raise_if_missing: bool = False):
    """
    Read PPFLAG from header or FITS file.

    Returns 0 if PPFLAG is missing (backward compat). If raise_if_missing=True,
    raises PpflagNotFoundError when PPFLAG is absent in an ingredient frame.
    """
    missing = False
    source = ""
    try:
        if isinstance(header_or_path, (str, os.PathLike)):
            source = str(header_or_path)
            try:
                val = fits.getval(header_or_path, PPFLAG_KEY)
                return int(val) & PPFLAG_MAX
            except (KeyError, OSError):
                missing = True
        else:
            source = "header"
            if PPFLAG_KEY not in header_or_path:
                missing = True
            else:
                return int(header_or_path[PPFLAG_KEY]) & PPFLAG_MAX
    except Exception:
        if raise_if_missing:
            raise
        return 0

    if missing and raise_if_missing:
        raise PpflagNotFoundError(f"PPFLAG not found in ingredient frame: {source}")
    return 0


def is_same_nightdate(path1: str, path2: str) -> bool:
    """Return True if both paths have the same nightdate (YYYY-MM-DD) in directory."""
    n1 = get_nightdate(path1)
    n2 = get_nightdate(path2)
    if n1 is None or n2 is None:
        return False
    return n1 == n2


def compute_fetch_ppflag(
    found_path: str,
    template: str,
    sanity_value: bool,
    *,
    flatdark_same_nightdate: bool = False,
    ignored_lenient_keys: bool = False,
) -> int:
    """
    Compute PPFLAG for a fetched master frame.

    Args:
        found_path: Path to the found/existing master frame.
        template: Template path used for search (contains target dates).
        sanity_value: SANITY header value of the fetched frame.
        flatdark_same_nightdate: If True, treat as PPFLAG 0 (flatdark with same nightdate).
        ignored_lenient_keys: If True, match was found by relaxing lenient keys (bit 8).

    Returns:
        PPFLAG value (0-31).
    """
    if flatdark_same_nightdate:
        return 0

    result = 0
    # Bit 1: different date
    if not is_same_nightdate(found_path, template):
        result |= PPFLAG_DIFFERENT_DATE

    # Bit 4: SANITY was F but used
    if sanity_value is False:
        result |= PPFLAG_SANITY_F_USED

    # Bit 8: lenient keys were ignored to find match
    if ignored_lenient_keys:
        result |= PPFLAG_LENIENT_KEYS_IGNORED

    return result


def propagate_ppflag(*ppflags: int) -> int:
    """Combine PPFLAGs from multiple dependencies via bitwise OR."""
    result = 0
    for p in ppflags:
        result |= int(p) & PPFLAG_MAX
    return result


def set_ppflag_in_header(header, ppflag: int, comment: str = "Preprocessing quality flag"):
    """Set PPFLAG in header. Mutates header in place."""
    header[PPFLAG_KEY] = (ppflag, comment)
    return header


def compute_ppflag_for_science_image(bias_path: str, dark_path: str, flat_path: str) -> int:
    """
    Compute PPFLAG for a science image from its master frame paths.
    Use get_ppflag_from_header on each master; returns 0 for missing/invalid paths.
    """
    ppflags = []
    for p in (bias_path, dark_path, flat_path):
        if p and os.path.exists(p):
            ppflags.append(get_ppflag_from_header(p))
        else:
            ppflags.append(0)
    return propagate_ppflag(*ppflags)
