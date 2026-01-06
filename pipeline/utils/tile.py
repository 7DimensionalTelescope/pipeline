import os
import re
from typing import Optional, List, Union
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u

from ..const import REF_DIR


_RIS_TILE_RE = re.compile(r"T\d{5}")


AngleLike = Union[str, float, int, u.Quantity, Angle]


def _looks_like_hms(s: str) -> bool:
    s = s.strip().lower()
    # Heuristic: sexagesimal or explicit hour units.
    return (":" in s) or ("h" in s) or ("m" in s) or ("s" in s) or (len(s.split()) > 1)


def _parse_ra_dec(ra: AngleLike, dec: AngleLike) -> SkyCoord:
    """
    Parse RA/Dec inputs into a SkyCoord.

    - floats/ints: assumed degrees
    - Quantities: must be angle-like (deg, hourangle, rad, ...)
    - strings: interpreted as either degrees or HMS/DMS (heuristic for RA)
    """

    def parse_ra(x: AngleLike) -> Angle:
        if isinstance(x, Angle):
            return x
        if isinstance(x, u.Quantity):
            return Angle(x)
        if isinstance(x, (int, float, np.integer, np.floating)):
            return Angle(x, unit=u.deg)
        if isinstance(x, str):
            s = x.strip()
            unit = u.hourangle if _looks_like_hms(s) else u.deg
            try:
                return Angle(s, unit=unit)
            except Exception:
                return Angle(s, unit=(u.deg if unit == u.hourangle else u.hourangle))
        raise TypeError(f"Unsupported RA type: {type(x)}")

    def parse_dec(x: AngleLike) -> Angle:
        if isinstance(x, Angle):
            return x
        if isinstance(x, u.Quantity):
            return Angle(x)
        if isinstance(x, (int, float, np.integer, np.floating)):
            return Angle(x, unit=u.deg)
        if isinstance(x, str):
            return Angle(x.strip(), unit=u.deg)
        raise TypeError(f"Unsupported Dec type: {type(x)}")

    return SkyCoord(parse_ra(ra), parse_dec(dec), frame="icrs")


def is_ris_tile(obj: str, *, loose: bool = True) -> bool:
    """
    Return True if `obj` matches T + 5 digits.

    loose=False (default): must be exactly 'T01234'  -> fullmatch
    loose=True:            must start with 'T01234'  -> match
    """
    m = _RIS_TILE_RE.match(obj) if loose else _RIS_TILE_RE.fullmatch(obj)
    return m is not None


def find_ris_tile(obj: str, *, loose: bool = True) -> Optional[str]:
    """
    Return the RIS tile like 'T01234' if found, else None.

    loose=True (default): find tile anywhere in obj  -> search
    loose=False:          obj must be exactly tile   -> fullmatch
    """
    m = _RIS_TILE_RE.search(obj) if loose else _RIS_TILE_RE.fullmatch(obj)
    return m.group(0) if m else None


def get_ris_tiles_near(ra: AngleLike, dec: AngleLike, radius: Union[float, u.Quantity] = 1.62) -> List[str]:
    """
    All units in deg, accepts astropy Angle or Quantity

    Default radius is the upper limit: 0.505 * np.sqrt(9576**2 + 6388**2) / 3600
    """

    skygrid_table = Table.read(os.path.join(REF_DIR, "skygrid.fits"))
    c_grid = SkyCoord(skygrid_table["ra"], skygrid_table["dec"], unit="deg")
    c_target = _parse_ra_dec(ra, dec)

    radius_deg = radius.to_value(u.deg) if isinstance(radius, u.Quantity) else float(radius)
    sep_arr = c_target.separation(c_grid).deg
    mask = sep_arr < radius_deg

    tiles = skygrid_table["tile"][mask]
    out: List[str] = []
    for t in tiles:
        if isinstance(t, (bytes, np.bytes_)):
            out.append(t.decode())
        else:
            out.append(str(t))
    return out
