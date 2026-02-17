"""
PPFLAG: Preprocessing quality flag for master frames and science images.

Bit definitions (combined with bitwise OR):
    0: On-the-date raw calib frames (input directly leads to output)
    1: Different date, but still okay in every regard
    2: Manually generated master frames (not set by pipeline; input may have this)
    4: SANITY was F, but used
    8: Different instrument used (not set by pipeline)
    16: Different settings used (not set by pipeline)

Range: 0-31 when all bits set.
"""

import os
from astropy.io import fits

from ..path.utils import get_nightdate

# Bit masks for PPFLAG
PPFLAG_RAW_ON_DATE = 0  # implicit when no other bits
PPFLAG_DIFFERENT_DATE = 1
PPFLAG_MANUAL = 2
PPFLAG_SANITY_F_USED = 4
PPFLAG_DIFFERENT_INSTRUMENT = 8
PPFLAG_DIFFERENT_SETTINGS = 16

PPFLAG_KEY = "PPFLAG"


def get_ppflag_from_header(header_or_path):
    """Read PPFLAG from header or FITS file. Returns 0 if missing (backward compat)."""
    if isinstance(header_or_path, (str, os.PathLike)):
        try:
            val = fits.getval(header_or_path, PPFLAG_KEY)
            return int(val) & 31
        except (KeyError, OSError):
            return 0
    else:
        return int(header_or_path.get(PPFLAG_KEY, 0)) & 31


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
) -> int:
    """
    Compute PPFLAG for a fetched master frame.

    Args:
        found_path: Path to the found/existing master frame.
        template: Template path used for search (contains target dates).
        sanity_value: SANITY header value of the fetched frame.
        flatdark_same_nightdate: If True, treat as PPFLAG 0 (flatdark with same nightdate).

    Returns:
        PPFLAG value (0-31).
    """
    ppflag = 0

    if flatdark_same_nightdate:
        return 0

    result = 0
    # Bit 1: different date
    if not is_same_nightdate(found_path, template):
        result |= PPFLAG_DIFFERENT_DATE

    # Bit 4: SANITY was F but used
    if sanity_value is False:
        result |= PPFLAG_SANITY_F_USED

    return result


def propagate_ppflag(*ppflags: int) -> int:
    """Combine PPFLAGs from multiple dependencies via bitwise OR."""
    result = 0
    for p in ppflags:
        result |= int(p) & 31
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
