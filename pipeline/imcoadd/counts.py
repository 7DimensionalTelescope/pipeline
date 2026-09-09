"""The coadd's integer count planes: one declaration, and an accumulator addressed by attribute.

`COUNT_PLANES` is the only place a plane is named. Everything else — what the mask builder allocates and
accumulates, what a coadd backend fills in, the order and the legend of the extensions in
`<coadd>_counts.fits`, and which planes a stage cannot produce — is derived from it.
"""

from dataclasses import dataclass

import numpy as np

from .const import MaskBit


@dataclass(frozen=True, slots=True)
class CountPlane:
    """One integer plane: what it counts, where its numbers come from, and where it cannot be produced."""

    name: str  # the FITS EXTNAME, and the key of the written product
    attr: str  # the CoaddCounts attribute a producer assigns or accumulates into
    legend: str  # the card comment; keep it inside the 68-character FITS value budget
    bit: MaskBit | None = None  # set: accumulated per frame from the quality bit masks
    accumulated: bool = False  # needs a zeroed array up front, rather than one array assigned whole
    needs_detector: bool = False  # omitted on a stage whose inputs carry no detector bad-pixel mask


COUNT_PLANES = (
    CountPlane("NGEOM", "geometric", "inputs with geometric support before quality rejection"),
    CountPlane("NUSED", "used", "inputs that entered the estimator"),
    CountPlane(
        "NBAD",
        "bad",
        "inputs whose detector bad-pixel mask projects here",
        bit=MaskBit.BADPIX,
        accumulated=True,
        needs_detector=True,
    ),
    CountPlane(
        "NSAT",
        "saturated",
        "inputs saturated here",
        bit=MaskBit.SATURATED,
        accumulated=True,
    ),
    CountPlane(
        "NTRAIL",
        "trail",
        "inputs carrying an enabled satellite-trail mask",
        bit=MaskBit.SATELLITE,
        accumulated=True,
    ),
    CountPlane("NOUTLIER", "outlier", "samples rejected by clipped coaddition (0 in other modes)", accumulated=True),
)

# Reserved names are deliberately not part of COUNT_PLANES until a producer sets their mask bits.
# Writing an all-zero extension would incorrectly mean the category was evaluated and absent.
RESERVED_COUNT_PLANES = (
    CountPlane(
        "NHOT",
        "hot",
        "inputs carrying a classified hot detector pixel",
        bit=MaskBit.HOT,
        accumulated=True,
        needs_detector=True,
    ),
    CountPlane(
        "NDEAD",
        "dead",
        "inputs carrying a classified dead detector pixel",
        bit=MaskBit.DEAD,
        accumulated=True,
        needs_detector=True,
    ),
    CountPlane(
        "NSTRAY",
        "stray",
        "inputs affected by a diffuse stray-light artifact",
        bit=MaskBit.STRAY,
        accumulated=True,
    ),
)

LEGEND = {plane.name: plane.legend for plane in COUNT_PLANES}


class CoaddCounts:
    """Count planes addressed by attribute, so no producer spells a plane name."""

    __slots__ = tuple(plane.attr for plane in COUNT_PLANES)

    def __init__(self):
        for plane in COUNT_PLANES:
            setattr(self, plane.attr, None)

    def allocate(self, shape, dtype) -> None:
        """Zero the planes accumulated frame by frame; the rest are assigned whole by a coadd backend."""
        for plane in COUNT_PLANES:
            if plane.accumulated:
                setattr(self, plane.attr, np.zeros(shape, dtype=dtype))

    def bit_planes(self):
        """(array, bit) of every allocated plane driven by a per-frame quality bit."""
        return [
            (getattr(self, plane.attr), plane.bit)
            for plane in COUNT_PLANES
            if plane.bit is not None and getattr(self, plane.attr) is not None
        ]

    def produced(self, has_detector: bool = True) -> dict:
        """Ordered {EXTNAME: array} of the planes this run actually produced."""
        return {
            plane.name: getattr(self, plane.attr)
            for plane in COUNT_PLANES
            if getattr(self, plane.attr) is not None and not (plane.needs_detector and not has_detector)
        }

    @staticmethod
    def missing(produced) -> list[str]:
        """Declared planes absent from *produced*, in declaration order."""
        return [plane.name for plane in COUNT_PLANES if plane.name not in produced]
