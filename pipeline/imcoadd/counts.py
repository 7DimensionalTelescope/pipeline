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
    short: str = ""  # check-plot legend caption
    bit: MaskBit | None = None  # the OR-bitmask bit this plane counts; also its check-plot draw order
    color: str = ""  # check-plot colour of this reason
    from_frame_mask: bool = False  # accumulated in set_frame from each frame's own bit mask
    accumulated: bool = False  # needs a zeroed array up front, rather than one array assigned whole
    needs_detector: bool = False  # omitted on a stage whose inputs carry no detector bad-pixel mask


COUNT_PLANES = (
    CountPlane(
        "NGEOM",
        "geometric",
        "inputs with geometric support before quality rejection",
        short="geometric support",
    ),
    CountPlane("NUSED", "used", "inputs that entered the estimator", short="used by the estimator"),
    CountPlane(
        "NBAD",
        "bad",
        "inputs whose detector bad-pixel mask projects here",
        short="detector bad pixel",
        bit=MaskBit.BADPIX,
        color="#3fb950",
        from_frame_mask=True,
        accumulated=True,
        needs_detector=True,
    ),
    CountPlane(
        "NSAT",
        "saturated",
        "inputs saturated here",
        short="saturated",
        bit=MaskBit.SATURATED,
        color="#e5484d",
        from_frame_mask=True,
        accumulated=True,
    ),
    CountPlane(
        "NTRAIL",
        "trail",
        "inputs carrying an enabled satellite-trail mask",
        short="satellite trail",
        bit=MaskBit.SATELLITE,
        color="#f0a202",
        from_frame_mask=True,
        accumulated=True,
    ),
    CountPlane(
        "NOUTLIER",
        "outlier",
        "samples rejected by clipped coaddition (0 in other modes)",
        short="clipped as an outlier",
        bit=MaskBit.OUTLIER,  # written by mark_outliers, not by set_frame
        color="#4c9be8",
        accumulated=True,
    ),
)

# Reserved names are deliberately not part of COUNT_PLANES until a producer sets their mask bits.
# Writing an all-zero extension would incorrectly mean the category was evaluated and absent.
RESERVED_COUNT_PLANES = (
    CountPlane(
        "NHOT",
        "hot",
        "inputs carrying a classified hot detector pixel",
        short="hot detector pixel",
        bit=MaskBit.HOT,
        color="#00c2c7",
        from_frame_mask=True,
        accumulated=True,
        needs_detector=True,
    ),
    CountPlane(
        "NDEAD",
        "dead",
        "inputs carrying a classified dead detector pixel",
        short="dead detector pixel",
        bit=MaskBit.DEAD,
        color="#8b949e",
        from_frame_mask=True,
        accumulated=True,
        needs_detector=True,
    ),
    CountPlane(
        "NSTRAY",
        "stray",
        "inputs affected by a diffuse stray-light artifact",
        short="stray light",
        bit=MaskBit.STRAY,
        color="#a371f7",
        from_frame_mask=True,
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
            if plane.from_frame_mask and getattr(self, plane.attr) is not None
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
