from typing import List
import numpy as np
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS
import astropy.io.fits as fits

from ..const import PIXSCALE, PipelineError
from ..tools.table import match_two_catalogs, add_id_column, match_multi_catalogs
from ..utils import add_suffix, swap_ext
from .plotting import wcs_check_plot
from ..subtract.utils import create_ds9_region_file
from .utils import inside_quad_spherical, compute_rms_stats, well_matchedness_stats


def evaluate_single_wcs(
    image: str,
    ref_cat: Table,
    source_cat: str | Table,
    date_obs: str,
    wcs: WCS,
    match_radius=10,
    fov_ra=None,
    fov_dec=None,
    write_matched_catalog=True,
    plot_save_path=None,
    num_sci=100,
    num_ref=100,
    num_plot=50,
    ds9_region=True,
):
    """Ensure num_plot <= num_sci, num_ref"""

    # load the source catalog
    if isinstance(source_cat, str):
        tbl = Table(fits.getdata(source_cat, ext=2))
    elif isinstance(source_cat, Table):
        tbl = source_cat
        source_cat = "table_evaluated.fits"
    else:
        raise ValueError(f"Invalid input type: {type(source_cat)}")

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
        matched.write(add_suffix(source_cat, "matched"), overwrite=True)

    sep = matched["separation"]
    n_valid = sep.count()
    sep_sci = sep.compressed()  # unmasked
    if n_valid > 0:
        # Fractions
        unmatched_fraction = sep.mask.sum() / sep.size
        subpixel_fraction = float(np.ma.mean(sep < PIXSCALE))
        subsecond_fraction = float(np.ma.mean(sep < 1.0))

        separation_stats = {
            "rms": np.sqrt(np.mean(sep_sci**2)),
            "min": np.min(sep_sci),
            "max": np.max(sep_sci),
            "q1": np.percentile(sep_sci, 25),
            "q2": np.percentile(sep_sci, 50),  # same as median
            "q3": np.percentile(sep_sci, 75),
            "p95": np.percentile(sep_sci, 95),
            "p99": np.percentile(sep_sci, 99),
        }
    else:
        unmatched_fraction = 1.0  # no NaN. fits requires values
        subpixel_fraction = 0
        subsecond_fraction = 0
        separation_stats = {
            "rms": 0.0,
            "min": 0.0,
            "max": 0.0,
            "q1": 0.0,
            "q2": 0.0,
            "q3": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }

    # if unmatched_fraction == 1.0:
    #     raise PipelineError(f"Unmatched fraction is 1.0 for {image}")

    if plot_save_path is not None and unmatched_fraction < 1.0:
        wcs_check_plot(
            ref_cat,
            tbl,
            matched,
            wcs,
            image,
            plot_save_path,
            fov_ra=fov_ra,
            fov_dec=fov_dec,
            num_plot=num_plot,
            sep_stats=separation_stats,
            subpixel_fraction=subpixel_fraction,
            subsecond_fraction=subsecond_fraction,
        )

    return (
        matched,
        REF_MAX_MAG,
        SCI_MAX_MAG,
        NUM_REF,
        unmatched_fraction,
        subpixel_fraction,
        subsecond_fraction,
        separation_stats,
    )


###############################################################################


def evaluate_joint_wcs(images_info: List["ImageInfo"]):
    """
    Uses the matched cats from the previous step as it helps point source selection,
    but it hurts completeness as the gaia reference is trimmed to num_ref sources.
    You may want to refine the logic to use the original sextractor catalogs instead.
    """

    matched_cats = [image_info.matched_catalog for image_info in images_info]
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
