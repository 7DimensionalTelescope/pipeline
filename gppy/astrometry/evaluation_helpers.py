from dataclasses import dataclass
import numpy as np
import astropy.units as u
from astropy.table import Table
from collections import Counter

from ..tools.angle import pa_alignment, azimuth_deg_from_center, pa_quadrupole_alignment
from .utils import find_id_rows


@dataclass(frozen=True)
class PsfStats:
    FWHMCRMN: float
    FWHMCRMX: float
    AWINCRMN: float
    AWINCRMX: float
    # AWINCRSD: float
    PA_ALIGN: float
    PA_QUAD: float


def compute_psf_stats(matched_catalog: Table, matched_ids: list[int]) -> PsfStats:
    """Use only corner 4 stars and the center star given 3x3 stars"""
    selected_stars = find_id_rows(matched_catalog, matched_ids)  # 3x3 = 9 stars
    if len(selected_stars) == 0:
        stats = PsfStats(
            FWHMCRMN=np.nan,
            FWHMCRMX=np.nan,
            AWINCRMN=np.nan,
            AWINCRMX=np.nan,
            PA_ALIGN=np.nan,
        )
        return stats

    # assume 3x3 grid of stars
    assert len(matched_ids) == 9

    # indices: 0, 2, 6, 8 = corners; 4 = center
    corner_idx = np.array([0, 2, 6, 8])
    center_idx = 4
    corner_stars = selected_stars[corner_idx]
    center_star = selected_stars[center_idx]
    # filter out potential None rows
    corner_stars = corner_stars[[True if v is not None else False for v in corner_stars["X_IMAGE"]]]

    # --- Tracking Issue & Astigmatism ---
    pa = corner_stars["THETA_IMAGE"]  # [-90, 90]
    pa_align, _, _, _, _ = pa_alignment(pa)
    phi_deg = azimuth_deg_from_center(
        corner_stars["X_IMAGE"], corner_stars["Y_IMAGE"], center_star["X_IMAGE"], center_star["Y_IMAGE"]
    )
    pa_quadrupole, _, _, _, _ = pa_quadrupole_alignment(pa, phi_deg)

    # --- 2D PSF Variation: corner / center ---
    fwhm_ratio = corner_stars["FWHM_IMAGE"] / center_star["FWHM_IMAGE"]
    awin_ratio = corner_stars["AWIN_IMAGE"] / center_star["AWIN_IMAGE"]

    stats = PsfStats(
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
    return stats

    # matched_ids = [i for i in matched_ids if i is not None]  # clean potential Nones

    # # 2D PSF Variation
    # fwhm_in_pix = selected_stars["FWHM_IMAGE"]
    # fwhm_ratio = fwhm_in_pix / fwhm_in_pix[4]
    # awin_in_pix = selected_stars["AWIN_IMAGE"]
    # awin_ratio = awin_in_pix / awin_in_pix[4]
    # # rms_in_pix = selected_stars["A_IMAGE"]
    # # rms_ratio = rms_in_pix / rms_in_pix[4]

    # # Tracking Issue
    # pa = selected_stars["THETA_IMAGE"]  # [-90, 90]

    # pa_align, _, _, _, _ = pa_alignment(pa)

    # # Both
    # ellip = selected_stars["ELLIPTICITY"]

    # stats = PsfStats(
    #     FWHMCRMN=np.mean(fwhm_ratio),
    #     FWHMCRSD=np.std(fwhm_ratio),
    #     AWINCRMN=np.mean(awin_ratio),
    #     AWINCRSD=np.std(awin_ratio),
    #     AWINCRMX=np.max(awin_ratio),
    #     # "PA_MEAN": np.mean(pa),
    #     # "PA_STD": np.std(pa),
    #     PA_ALIGN=pa_align,
    #     ELLIPMN=np.mean(ellip),
    #     ELLIPSTD=np.std(ellip),
    # )
    # return stats


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
