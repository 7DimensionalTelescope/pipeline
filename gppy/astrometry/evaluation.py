from __future__ import annotations
import os
import numpy as np
from dataclasses import dataclass, field, fields
from typing import Optional, TypedDict, List, Tuple
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS
import astropy.io.fits as fits

from ..const import PIXSCALE
from ..tools.table import match_two_catalogs, add_id_column, match_multi_catalogs
from ..tools.angle import azimuth_deg_from_center
from ..utils import add_suffix, swap_ext
from .plotting import wcs_check_plot
from ..subtract.utils import create_ds9_region_file
from .utils import inside_quad_spherical, get_3x3_stars, get_fov_quad
from .evaluation_helpers import (
    compute_rms_stats,
    well_matchedness_stats,
    CornerStats,
    SeparationStats,
    RadialStats,
)


def table_has_nan(tbl: Table) -> bool:
    for col in tbl.itercols():
        if col.dtype.kind == "f":  # floating types
            if np.isnan(col).any():
                return True
    return False


@dataclass(frozen=True)
class RSEPStats:
    """SeparationStats + metadata"""

    ref_max_mag: float
    sci_max_mag: float
    num_ref_sources: int
    unmatched_fraction: float
    subpixel_fraction: float
    subsecond_fraction: float
    separation_stats: SeparationStats

    @property
    def fits_header_cards_for_metadata(self) -> List[Tuple[str, float, str]]:
        """not the entire header cards, just the metadata"""
        return [
            (f"REFMXMAG", self.ref_max_mag, "Highest g mag of selected reference sources"),
            (f"SCIMXMAG", self.sci_max_mag, "Highest inst mag of selected science sources"),
            (f"NUM_REF", self.num_ref_sources, "Number of reference sources selected"),
            (f"UNMATCH", self.unmatched_fraction, "Fraction of unmatched reference sources"),
            (f"SUBPIXEL", self.subpixel_fraction, "Fraction of matched with sep < PIXSCALE"),
            (f"SUBSEC", self.subsecond_fraction, "Fraction of matched with sep < 1"),
        ]


@dataclass(frozen=True)
class ImageStats:
    """Overall PSF characteristics"""

    PEEINGMN: float = field(metadata={"COMMENT": "Mean FWHM of all ref matched sources [pix]"})
    PEEINGSD: float = field(metadata={"COMMENT": "STD of FWHM of all ref matched sources [pix]"})
    SEEINGMN: float = field(metadata={"COMMENT": "Mean seeing of all ref matched srcs [arcsec]"})
    SEEINGSD: float = field(metadata={"COMMENT": "STD of seeing of all ref matched srcs [arcsec]"})
    PAWINMN: float = field(metadata={"COMMENT": "Mean AWIN of ref matched sources [pix]"})
    PAWINSD: float = field(metadata={"COMMENT": "STD of AWIN of ref matched sources [pix]"})
    ELLIPMN: float = field(metadata={"COMMENT": "Mean ellipticity of ref matched sources"})
    ELLIPSD: float = field(metadata={"COMMENT": "STD of ellipticity of ref matched sources"})

    @classmethod
    def from_matched_catalog(cls, matched: Table) -> ImageStats:
        PEEINGMN = matched["FWHM_IMAGE"].mean()
        PEEINGSD = matched["FWHM_IMAGE"].std()
        return cls(
            PEEINGMN=PEEINGMN,
            PEEINGSD=PEEINGSD,
            SEEINGMN=PEEINGMN * PIXSCALE,
            SEEINGSD=PEEINGSD * PIXSCALE,
            PAWINMN=matched["AWIN_IMAGE"].mean(),
            PAWINSD=matched["AWIN_IMAGE"].std(),
            ELLIPMN=matched["ELLIPTICITY"].mean(),
            ELLIPSD=matched["ELLIPTICITY"].std(),
        )

    @property
    def fits_header_cards(self) -> List[Tuple[str, float, str]]:
        return [(f.name, getattr(self, f.name), f.metadata.get("COMMENT", "")) for f in fields(self)]


@dataclass(frozen=True)
class EvaluationResult:
    matched: Table
    rsep_stats: RSEPStats
    corner_stats: CornerStats
    image_stats: ImageStats
    radial_stats: RadialStats


def evaluate_single_wcs(
    image: str,
    ref_cat: Table,
    source_cat: str | Table,
    date_obs: str,
    wcs: WCS = None,
    H: int = None,
    W: int = None,
    match_radius=10,
    fov_ra=None,
    fov_dec=None,
    write_matched_catalog=True,
    plot_save_path=None,
    num_sci=100,
    num_ref=100,
    num_plot=100,
    ds9_region=True,
    cutout_size=30,
    logger=None,
    overwrite=True,
) -> EvaluationResult:
    """Ensure num_plot <= num_sci, num_ref"""

    def chatter(msg: str, level: str = "debug"):
        if logger is not None:
            return getattr(logger, level)(msg)
        else:
            print(f"[evaluate_single_wcs:{level.upper()}] {msg}")

    matched_catalog_path = add_suffix(source_cat, "matched")

    # Failed attempt to skip if matched catalog already exists
    # if os.path.exists(matched_catalog_path) and not overwrite:
    #     chatter(f"Matched catalog already exists: {matched_catalog_path}, skipping...", "info")
    #     matched = Table.read(matched_catalog_path)
    #     return EvaluationResult(
    #         matched=Table.read(matched_catalog_path),
    #         rsep_stats=None,
    #         psf_stats=None,
    #         image_stats=None,
    #     )

    matched, tbl, ref_cat, (REF_MAX_MAG, SCI_MAX_MAG, NUM_REF) = prepare_matched_catalog(
        chatter,
        image=image,
        ref_cat=ref_cat,
        source_cat=source_cat,
        date_obs=date_obs,
        wcs=wcs,
        H=H,
        W=W,
        match_radius=match_radius,
        fov_ra=fov_ra,
        fov_dec=fov_dec,
        write_matched_catalog=write_matched_catalog,
        num_sci=num_sci,
        num_ref=num_ref,
        ds9_region=ds9_region,
        overwrite=overwrite,
        matched_catalog_path=matched_catalog_path,
    )

    # compute separation statistics
    sep = matched["separation"]
    unmatched_fraction = sep.mask.sum() / sep.size
    n_valid = sep.count()
    matched_cleaned = matched[~sep.mask]

    if n_valid > 0:
        # Fractions
        subpixel_fraction = float(np.ma.mean(sep < PIXSCALE))
        subsecond_fraction = float(np.ma.mean(sep < 1.0))

        # compute
        sep_sci = matched_cleaned["separation"]
        sep_x_sci = matched_cleaned["dra_cosdec"]
        sep_y_sci = matched_cleaned["ddec"]
        phi_deg = azimuth_deg_from_center(
            matched_cleaned["X_IMAGE"], matched_cleaned["Y_IMAGE"], (W + 1) / 2, (H + 1) / 2
        )
        sep_dr = np.cos(np.deg2rad(phi_deg)) * sep_x_sci + np.sin(np.deg2rad(phi_deg)) * sep_y_sci
        sep_r_dphi = np.sin(np.deg2rad(phi_deg)) * sep_x_sci - np.cos(np.deg2rad(phi_deg)) * sep_y_sci

        separation_stats = SeparationStats(
            RMS=np.sqrt(np.mean(sep_sci**2)),
            MADX=np.median(np.abs(sep_x_sci)),
            MADY=np.median(np.abs(sep_y_sci)),
            MADR=np.median(np.abs(sep_dr)),
            MADT=np.median(np.abs(sep_r_dphi)),
            MIN=np.min(sep_sci),
            MAX=np.max(sep_sci),
            Q1=np.percentile(sep_sci, 25),
            Q2=np.percentile(sep_sci, 50),
            Q3=np.percentile(sep_sci, 75),
            P95=np.percentile(sep_sci, 95),
            P99=np.percentile(sep_sci, 99),
        )
    else:
        unmatched_fraction = 1.0  # no NaN. fits requires values
        subpixel_fraction = 0
        subsecond_fraction = 0
        separation_stats = SeparationStats()  # empty

    rsep_stats = RSEPStats(
        ref_max_mag=REF_MAX_MAG,
        sci_max_mag=SCI_MAX_MAG,
        num_ref_sources=NUM_REF,
        unmatched_fraction=unmatched_fraction,
        subpixel_fraction=subpixel_fraction,
        subsecond_fraction=subsecond_fraction,
        separation_stats=separation_stats,
    )

    # 2D PSF stats
    matched_ids = get_3x3_stars(matched_cleaned, H, W, cutout_size)
    try:
        corner_stats = CornerStats.from_matched_catalog(matched_cleaned, matched_ids)
    except Exception as e:
        chatter(f"evaluate_single_wcs: failed to compute corner_stats: {e}")
        corner_stats = None

    chatter(f"evaluate_single_wcs: rsep_stats {rsep_stats}")
    chatter(f"evaluate_single_wcs: matched_ids {matched_ids}")
    chatter(f"evaluate_single_wcs: corner_stats {corner_stats}")

    # overall image stats
    image_stats = ImageStats.from_matched_catalog(matched_cleaned)

    # radial stats
    radial_stats = RadialStats.from_matched_catalog(matched_cleaned, W, H)

    if plot_save_path is not None and unmatched_fraction < 1.0:
        chatter(f"evaluate_single_wcs: plotting to {plot_save_path}")
        wcs_check_plot(
            ref_cat,
            tbl,
            matched_cleaned,
            wcs,
            image,
            plot_save_path,
            fov_ra=fov_ra,
            fov_dec=fov_dec,
            num_plot=num_plot,
            sep_stats=separation_stats,
            subpixel_fraction=subpixel_fraction,
            subsecond_fraction=subsecond_fraction,
            matched_ids=matched_ids,
            cutout_size=cutout_size,
        )
        chatter(f"evaluate_single_wcs: plot saved to {plot_save_path}")

    return EvaluationResult(
        matched=matched_cleaned,  # this has to be cleaned of ref-only rows for joint evaluation
        rsep_stats=rsep_stats,
        corner_stats=corner_stats,
        image_stats=image_stats,
        radial_stats=radial_stats,
    )


def prepare_matched_catalog(
    chatter: callable,
    image: str,
    ref_cat: Table,
    source_cat: str | Table,
    date_obs: str,
    wcs: WCS = None,
    H: int = None,
    W: int = None,
    match_radius=10,
    fov_ra=None,
    fov_dec=None,
    write_matched_catalog=True,
    num_sci=100,
    num_ref=100,
    ds9_region=True,
    overwrite=True,
    matched_catalog_path=None,
):

    # load the source catalog
    if isinstance(source_cat, str):
        tbl = Table(fits.getdata(source_cat, ext=2))
    elif isinstance(source_cat, Table):
        tbl = source_cat
        source_cat = "table_evaluated.fits"
    else:
        raise ValueError(f"Invalid input type: {type(source_cat)}")

    if wcs is None:
        chatter(f"WCS is not provided. Loading from {image}", "info")
        wcs = WCS(fits.getheader(image))

    if H is None or W is None:
        chatter(f"H and W are not provided. Loading from {image}", "info")
        H, W = fits.getdata(image).shape

    if fov_ra is None or fov_dec is None:
        try:
            fov_ra, fov_dec = get_fov_quad(wcs, W, H)
            chatter(f"Updated FOV polygon. RA: {fov_ra}, Dec: {fov_dec}")
        except Exception as e:
            chatter(f"Failed to update FOV polygon for {image}: {e}")

    # # update the source catalog with the WCS
    # if head is not None:
    #     wcs = head or WCS(head)
    #     ra, dec = wcs.all_pix2world(tbl["X_IMAGE"], tbl["Y_IMAGE"], 1)
    #     tbl["ALPHA_J2000"] = ra
    #     tbl["DELTA_J2000"] = dec

    # # sort tables by SNR (higher first)
    # ref_cat["snr"] = 1 / (0.4 * np.log(10) * ref_cat["phot_g_mean_mag_error"])
    # ref_cat.sort("snr", reverse=True)
    # tbl["snr"] = 1 / (0.4 * np.log(10) * tbl["MAGERR_AUTO"])
    # tbl.sort("snr", reverse=True)

    # sort tables by magnitude (brighter first)
    # ref_cat.sort("phot_g_mean_mag")  # threading unsafe
    order = np.argsort(ref_cat["phot_g_mean_mag"])
    ref_cat = ref_cat[order]
    tbl.sort("MAG_AUTO")
    tbl = tbl[tbl["FLAGS"] == 0]  # exclude saturated sources
    # tbl = tbl[tbl["MAG_AUTO"] > sci_inst_mag_llim]  # exclude too bright sci sources

    # filter sources in REFCAT by FOV
    if fov_ra is not None and fov_dec is not None:
        assert len(fov_ra) == len(fov_dec)
        sel = inside_quad_spherical(ref_cat["ra"], ref_cat["dec"], fov_ra, fov_dec)  # inside_quad_radec fails at poles
        ref_cat = ref_cat[sel]

    if ds9_region:
        x, y = wcs.all_world2pix(ref_cat["ra"], ref_cat["dec"], 1)
        create_ds9_region_file(x=x, y=y, filename=add_suffix(swap_ext(source_cat, "reg"), "ref_all"))
        create_ds9_region_file(
            x=tbl["X_IMAGE"], y=tbl["Y_IMAGE"], filename=add_suffix(swap_ext(source_cat, "reg"), "sci_all")
        )

    # extract brightest num_ref/num_sci sources to calculate separation statistics
    ref_cat = ref_cat[:num_ref]
    tbl = tbl[:num_sci]
    if ds9_region:
        x, y = wcs.all_world2pix(ref_cat["ra"], ref_cat["dec"], 1)
        create_ds9_region_file(x=x, y=y, filename=add_suffix(swap_ext(source_cat, "reg"), f"ref_{num_ref}"))
        create_ds9_region_file(
            x=tbl["X_IMAGE"], y=tbl["Y_IMAGE"], filename=add_suffix(swap_ext(source_cat, "reg"), f"sci_{num_sci}")
        )

    # REF_MIN_SNR = ref_cat["snr"][-1]
    # SCI_MIN_SNR = tbl["snr"][-1]
    REF_MAX_MAG = ref_cat["phot_g_mean_mag"][-1]
    SCI_MAX_MAG = tbl["MAG_AUTO"][-1]
    NUM_REF = len(ref_cat)

    # match source tbl with refcat
    if os.path.exists(matched_catalog_path) and not overwrite:
        matched = Table.read(matched_catalog_path)
        write_matched_catalog = False
        chatter(f"Matched catalog already exists: {matched_catalog_path}, skipping...", "info")
    else:

        pm_keys = dict(pmra="pmra", pmdec="pmdec", parallax=None, ref_epoch=2016.0)
        matched = match_two_catalogs(
            tbl,
            ref_cat,
            x1="ra",
            y1="dec",
            join="right",
            sep_components=True,
            radius=match_radius,
            correct_pm=True,
            obs_time=Time(date_obs),
            pm_keys=pm_keys,
        )  # right join: preserve all ref sources
        matched = add_id_column(matched)  # id needed to plot selected 9 stars

    if write_matched_catalog:
        matched.write(matched_catalog_path, overwrite=overwrite)

    return matched, tbl, ref_cat, (REF_MAX_MAG, SCI_MAX_MAG, NUM_REF)


###############################################################################


def evaluate_joint_wcs(matched_cats: List[Table]):
    """
    Uses the matched cats from the previous step as it helps point source selection,
    but it hurts completeness as the gaia reference is trimmed to num_ref sources.
    You may want to refine the logic to use the original sextractor catalogs instead.
    """
    matched_all = match_multi_catalogs(matched_cats, radius=3, join="outer", sep_components=True, suffix_first=True)

    rms_stats = compute_rms_stats(matched_all, [f"cat{i}" for i in range(len(matched_cats))])
    match_stats = well_matchedness_stats(matched_all, n_cats=len(matched_cats))

    # reformat
    rms_stats_list = [d for d in rms_stats.values()]
    # Refactor recall into a list of dicts
    match_stats_list = [
        {
            "catalog": cat,  # keep cat0, cat1, etc.
            "recall": val,  # the recall value
            **{k: v for k, v in match_stats.items() if k != "recall"},  # all other keys
        }
        for cat, val in match_stats["recall"].items()
    ]

    return rms_stats_list, match_stats_list
