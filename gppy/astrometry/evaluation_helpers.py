from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import Optional, TypedDict, List, Tuple
import numpy as np
import astropy.units as u
from astropy.table import Table
from collections import Counter

from ..tools.angle import pa_alignment, azimuth_deg_from_center, pa_quadrupole_alignment
from .utils import find_id_rows


@dataclass(frozen=True)
class SeparationStats:
    """
    separation statistics. reminiscent of scamp ASTRRMS
    No generating classmethod; flexibly choose which keys to include
    """

    # fits keys and comments
    N: Optional[int] = field(default=None, metadata={"COMMENT": "Number of sources used" + " " * 47})  # max 47 chars
    RMS: Optional[float] = field(default=None, metadata={"COMMENT": f"RMS separation"})
    MIN: Optional[float] = field(default=None, metadata={"COMMENT": f"Min separation"})
    MAX: Optional[float] = field(default=None, metadata={"COMMENT": f"Max separation"})
    Q1: Optional[float] = field(default=None, metadata={"COMMENT": f"1st quartile sep"})
    Q2: Optional[float] = field(default=None, metadata={"COMMENT": f"Median separation"})
    Q3: Optional[float] = field(default=None, metadata={"COMMENT": f"3rd quartile sep"})
    P95: Optional[float] = field(default=None, metadata={"COMMENT": f"95 percentile sep"})
    P99: Optional[float] = field(default=None, metadata={"COMMENT": f"99 percentile sep"})
    MAD: Optional[float] = field(default=None, metadata={"COMMENT": f"0-centered MAD sep"})

    # decomposed
    RMSX: Optional[float] = field(default=None, metadata={"COMMENT": "RMS in x"})  # actually RA
    RMSY: Optional[float] = field(default=None, metadata={"COMMENT": "RMS in y"})  # actually Dec
    MADX: Optional[float] = field(default=None, metadata={"COMMENT": "0-centered MAD in x"})
    MADY: Optional[float] = field(default=None, metadata={"COMMENT": "0-centered MAD in y"})
    MADR: Optional[float] = field(default=None, metadata={"COMMENT": "0-centered MAD in radial direction [arcsec]" + " " * 47})  # fmt: skip
    MADT: Optional[float] = field(default=None, metadata={"COMMENT": "0-centered MAD in tangential direction [arcsec]" + " " * 47})  # fmt: skip

    # Keyword differentiator
    prefix: str = "RSEP_"  # or "ISEP_" TODO make isep use this too, not dict
    description: str = "from reference catalog [arcsec]"

    @property
    def fits_header_cards(self) -> List[Tuple[str, float, str]]:
        """list of (KEY, VALUE, COMMENT) tuples for astropy.io.fits.Header"""
        # len(KEY) max is 8; len(COMMENT) max is 47
        return [
            (self.prefix[: (8 - len(f.name))] + f.name, v, f"{f.metadata.get('COMMENT', '')} {self.description}"[:47])
            for f in fields(self)
            if (f.name != "prefix" and f.name != "description" and (v := getattr(self, f.name)) is not None)
        ]


@dataclass(frozen=True)
class PSFStats:
    """
    PSF statistics like FWHM and ellipticity
    No generating classmethod; flexibly choose which keys to include
    """

    FWHM: float = field(metadata={"COMMENT": "Median FWHM [pix]"})
    ELLIP: float = field(metadata={"COMMENT": "Median ellipticity"})

    # Keyword differentiator
    prefix: str = "RSEP_"  # or "ISEP_" TODO
    description: str = "from reference catalog [arcsec]"

    @property
    def fits_header_cards(self) -> List[Tuple[str, float, str]]:
        return [
            ((self.prefix + f.name)[:8], v, f"{f.metadata.get('COMMENT', '')} {self.description}"[:47])
            for f in fields(self)
            if (f.name != "prefix" and f.name != "description" and (v := getattr(self, f.name)) is not None)
        ]


@dataclass(frozen=True)
class CornerStats:
    FWHMCRMN: float = field(metadata={"COMMENT": "Mean corner/center FWHM ratio (4 corner PSFs)"})
    FWHMCRMX: float = field(metadata={"COMMENT": "MAX corner/center FWHM ratio (4 corner PSFs)"})
    # FWHMCRSD: float # (f"FWHMCRSD", self.corner_stats.FWHMCRSD, "STD of corner/center FWHM ratio (4 corner PSFs)"),
    AWINCRMN: float = field(metadata={"COMMENT": "Mean corner/center AWIN ratio (4 corner PSFs)"})
    AWINCRMX: float = field(metadata={"COMMENT": "MAX corner/center AWIN ratio (4 corner PSFs)"})
    # AWINCRSD: float # (f"AWINCRSD", self.corner_stats.AWINCRSD, "STD of corner/center AWIN ratio (4 corner PSFs)"),
    PA_ALIGN: float = field(metadata={"COMMENT": "PA alignment score of 4 corner PSFs"})
    PA_QUAD: float = field(metadata={"COMMENT": "PA quadrupole alignment score of 4 corner PSFs"})

    @property
    def fits_header_cards(self) -> List[Tuple[str, float, str]]:
        return [
            (f.name, v, f"{f.metadata.get('COMMENT', '')}"[:47])
            for f in fields(self)
            if (v := getattr(self, f.name)) is not None
        ]

    @classmethod
    def from_matched_catalog(cls, matched_catalog: Table, matched_ids: list[int]) -> CornerStats:
        """Use only corner 4 stars and the center star given 3x3 stars"""
        selected_stars = find_id_rows(matched_catalog, matched_ids)  # 3x3 = 9 stars
        if len(selected_stars) == 0:
            return cls(FWHMCRMN=None, FWHMCRMX=None, AWINCRMN=None, AWINCRMX=None, PA_ALIGN=None, PA_QUAD=None)

        # assume 3x3 grid of stars
        assert len(matched_ids) == 9  # can contain None

        # indices: 0, 2, 6, 8 = corners; 4 = center
        corner_idx = np.array([0, 2, 6, 8])
        center_idx = 4
        corner_stars = selected_stars[corner_idx]
        center_star = selected_stars[center_idx]
        # filter out potential None rows
        corner_stars = corner_stars[[True if v is not None else False for v in corner_stars["X_IMAGE"]]]

        # --- Tracking Issue & Astigmatism ---
        pa = corner_stars["THETA_IMAGE"]  # [-90, 90]
        elon = corner_stars["ELONGATION"]  # or 1/(1 - ellip)
        pa_align, _, _, _, _ = pa_alignment(pa, weights=elon, normalize=False)  # no normalization to get the magnitude
        phi_deg = azimuth_deg_from_center(
            corner_stars["X_IMAGE"], corner_stars["Y_IMAGE"], center_star["X_IMAGE"], center_star["Y_IMAGE"]
        )
        pa_quadrupole, _, _, _, _ = pa_quadrupole_alignment(pa, phi_deg, weights=elon, normalize=False)

        # --- 2D PSF Variation: corner / center ---
        fwhm_ratio = corner_stars["FWHM_IMAGE"] / center_star["FWHM_IMAGE"]
        awin_ratio = corner_stars["AWIN_IMAGE"] / center_star["AWIN_IMAGE"]

        return cls(
            FWHMCRMN=np.mean(fwhm_ratio),
            FWHMCRMX=np.max(fwhm_ratio),
            # FWHMCRSD=np.std(fwhm_ratio),
            AWINCRMN=np.mean(awin_ratio),
            # AWINCRSD=np.std(awin_ratio),
            AWINCRMX=np.max(awin_ratio),
            PA_ALIGN=pa_align,
            PA_QUAD=pa_quadrupole,
            # ELLIPMN=np.mean(ellip),  # we have it in image_stats
            # ELLIPSTD=np.std(ellip),
        )


@dataclass(frozen=True)
class BinStats:
    sep_stats: SeparationStats
    psf_stats: PSFStats

    @property
    def fits_header_cards(self) -> List[Tuple[str, float, str]]:
        return self.sep_stats.fits_header_cards + self.psf_stats.fits_header_cards

    @classmethod
    def from_binned_catalog(cls, binned_catalog: Table) -> BinStats:

        bin_idx = binned_catalog["RADIAL_BIN"][0]  # 0, 1, 2
        binned_catalog = binned_catalog[~binned_catalog["separation"].mask]  # filter out ref-only rows

        if len(binned_catalog) == 0:
            return cls(
                sep_stats=SeparationStats(prefix=f"BIN{bin_idx}"),
                psf_stats=PSFStats(prefix=f"BIN{bin_idx}"),
            )

        sep = binned_catalog["separation"]
        sep_stats = SeparationStats(
            prefix=f"BIN{bin_idx}",
            description=f"in BIN{bin_idx} from reference [arcsec]",
            MAD=np.median(np.abs(sep - 0)),  # it's 0-centered MAD, not median-centered
        )

        fwhm = binned_catalog["FWHM_IMAGE"].compressed()  # unmask
        ellip = binned_catalog["ELLIPTICITY"].compressed()
        psf_stats = PSFStats(
            prefix=f"BIN{bin_idx}",
            description=f"in BIN{bin_idx} from reference [pix]",
            FWHM=np.median(fwhm),
            ELLIP=np.median(ellip),
        )

        return cls(
            sep_stats=sep_stats,
            psf_stats=psf_stats,
        )


@dataclass(frozen=True)
class RadialStats:
    """Radially-varying statistics"""

    BIN0: BinStats
    BIN1: BinStats
    BIN2: BinStats

    @property
    def fits_header_cards(self) -> List[Tuple[str, float, str]]:
        return self.BIN0.fits_header_cards + self.BIN1.fits_header_cards + self.BIN2.fits_header_cards

    @classmethod
    def from_matched_catalog(cls, matched_catalog: Table, naxis1: int, naxis2: int) -> RadialStats:
        """Use only corner 4 stars and the center star given 3x3 stars"""
        # if len(matched_catalog) == 0 or not all(c in matched_catalog.colnames for c in ["dra_cosdec", "ddec"]):
        #     return RadialStats(
        #         RSEP0P95=np.nan,
        #     )

        x = matched_catalog["X_IMAGE"]
        y = matched_catalog["Y_IMAGE"]
        x0 = (naxis1 + 1) / 2.0  # match sextractor convention (1-starting index)
        y0 = (naxis2 + 1) / 2.0
        r = np.hypot(x - x0, y - y0)  # radial distance in pixels

        r_max = max(naxis1, naxis2) / 2
        r_edges = r_max * np.sqrt(np.linspace(0, 1, 4))  # square equal spacing for 2nd order aberrations
        radial_bin = np.digitize(r, r_edges[1:-1])  # use inner edges for splitting
        matched_catalog["RADIAL_BIN"] = radial_bin  # 0, 1, 2

        return RadialStats(
            BIN0=BinStats.from_binned_catalog(matched_catalog[radial_bin == 0]),
            BIN1=BinStats.from_binned_catalog(matched_catalog[radial_bin == 1]),
            BIN2=BinStats.from_binned_catalog(matched_catalog[radial_bin == 2]),
        )


# ---------------------------- Internal RMS stats ----------------------------


def compute_rms_stats(out: Table, cat_names, unit: u.Quantity | str = u.arcsec):
    """
    Astrometric precision statistics per catalog, including SCAMP-style internal rms.

    Uses dra_cosdec_arcsec_<name>, ddec_arcsec_<name> for component RMS,
    and sep_arcsec_<name> for radial RMS and distribution statistics.

    Returns dict:
        {
            name: {
                'n': N,
                'rms_x': Quantity,
                'rms_y': Quantity,
                'rms': Quantity (radial RMS),
                'min': Quantity,
                'q1': Quantity,
                'q2': Quantity (median),
                'q3': Quantity,
                'max': Quantity
            }
        }
    """
    unit = u.Unit(unit)
    res = {}

    for name in cat_names:
        cx = f"dra_cosdec_arcsec_{name}"
        cy = f"ddec_arcsec_{name}"
        cr = f"sep_arcsec_{name}"

        # Skip if none of the relevant columns exist
        if all(c not in out.colnames for c in [cx, cy, cr]):
            continue

        # collect data
        x = out[cx] if cx in out.colnames else None
        y = out[cy] if cy in out.colnames else None
        r = out[cr] if cr in out.colnames else None

        # validity mask
        valid = np.ones(len(out), dtype=bool)
        if x is not None and hasattr(x, "mask"):
            valid &= ~x.mask
        if y is not None and hasattr(y, "mask"):
            valid &= ~y.mask
        if r is not None and hasattr(r, "mask"):
            valid &= ~r.mask

        if valid.sum() == 0:
            res[name] = {
                "n": 0,
                "rms_x": np.nan * unit,
                "rms_y": np.nan * unit,
                "rms": np.nan * unit,
                "min": np.nan * unit,
                "q1": np.nan * unit,
                "q2": np.nan * unit,
                "q3": np.nan * unit,
                "max": np.nan * unit,
            }
            continue

        # components
        if x is not None:
            xv = np.asarray(x)[valid] * u.arcsec
            rms_x = np.sqrt(np.mean(xv**2)).to(unit)
        else:
            rms_x = np.nan * unit

        if y is not None:
            yv = np.asarray(y)[valid] * u.arcsec
            rms_y = np.sqrt(np.mean(yv**2)).to(unit)
        else:
            rms_y = np.nan * unit

        # radial
        if r is not None:
            rv = np.asarray(r)[valid] * u.arcsec
        elif x is not None and y is not None:
            rv = np.sqrt(xv**2 + yv**2)
        else:
            rv = np.array([]) * u.arcsec

        if len(rv) > 0:
            rms = np.sqrt(np.mean(rv**2)).to(unit)
            rv_q = np.percentile(rv.value, [0, 25, 50, 75, 95, 99, 100]) * rv.unit
            min_, q1, q2, q3, p95, p99, max_ = rv_q.to(unit)
        else:
            rms = min_ = q1 = q2 = q3 = p95 = p99 = max_ = np.nan * unit

        res[name] = {
            "n": int(valid.sum()),
            "rms_x": rms_x,
            "rms_y": rms_y,
            "rms": rms,
            "min": min_,
            "q1": q1,
            "q2": q2,
            "q3": q3,
            "p95": p95,
            "p99": p99,
            "max": max_,
        }

    return res


def well_matchedness_stats(merged, n_cats=3):
    """
    Compute well-matchedness stats from an outer-merged catalog
    produced by match_multi_catalogs with pivot='centroid' and sep_components=True.

    Assumes:
      - sep_arcsec_cat{i} columns (masked if absent)
      - dra_cosdec_arcsec_cat{i}, ddec_arcsec_cat{i}
    """
    cat_names = [f"cat{i}" for i in range(n_cats)]

    # Presence matrix
    P = np.zeros((len(merged), n_cats), dtype=bool)
    for j, c in enumerate(cat_names):
        col = merged[f"sep_arcsec_{c}"]
        m = getattr(col, "mask", None)
        if m is not None and len(m):
            P[:, j] = ~m
        else:
            P[:, j] = np.isfinite(col)

    sizes = P.sum(axis=1)

    # Counts by group size
    counts_by_group_size = dict(Counter(sizes))

    # Recall per catalog
    recall = {}
    for j, c in enumerate(cat_names):
        others_present = (sizes - P[:, j]) >= 1
        denom = others_present.sum()
        num = (P[:, j] & others_present).sum()
        recall[c] = (num / denom) if denom else np.nan

    return {
        "counts_by_group_size": counts_by_group_size,
        "recall": recall,
    }
