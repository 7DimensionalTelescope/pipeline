from reprlib import recursive_repr
from typing import Any, List, Dict, Tuple, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass

# astropy
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astropy.table import Table, hstack, MaskedColumn
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip


# separated from the photometry class. sort out self later.


def plot_filter_check(
    self,
    alleged_filter,
    inferred_filter,
    narrowed_filters,
    filters_checked,
    zps,
    zperrs,
    inactive_filters=(),
    zp_limits=None,
):
    import os
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.legend_handler import HandlerTuple

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # 1) Plot EZP on ax1
    c_ezp = "tab:blue"
    ax1.plot(
        filters_checked,
        zperrs,
        marker="o",
        linestyle="-",
        label="EZP",
        color=c_ezp,
    )
    ax1.set_ylabel("Zero Point Error (EZP)", color=c_ezp)
    ax1.tick_params(axis="y", labelcolor=c_ezp)

    # 2) Shade backgrounds for ruled-out filters: gray = not in the night's filter budget
    for i, flt in enumerate(filters_checked):
        if flt in inactive_filters:
            ax1.axvspan(i - 0.5, i + 0.5, color="gray", alpha=0.3)
    gray_patch = mpatches.Patch(color="gray", alpha=0.3)  # patch for legend

    inf_idx = filters_checked.index(inferred_filter)
    ax1.axvspan(inf_idx - 0.5, inf_idx + 0.5, color="dodgerblue", alpha=0.3)
    blue_patch = mpatches.Patch(color="dodgerblue", alpha=0.3, label=f"Inferred Filter")

    if alleged_filter in filters_checked:  # red box edge around the filter recorded in the header
        rec_idx = filters_checked.index(alleged_filter)
        ax1.axvspan(rec_idx - 0.5, rec_idx + 0.5, fill=False, edgecolor="red", linewidth=2)
    red_patch = mpatches.Patch(fill=False, edgecolor="red", linewidth=2)

    # x-ticks and tilted labels
    ax1.set_xticks(range(len(filters_checked)))
    ax1.set_xticklabels(filters_checked, rotation=45, ha="right")

    # 3) Create twin axis for ZP
    c_zp = "orange"
    ax2 = ax1.twinx()
    deep = [
        i
        for i, flt in enumerate(filters_checked)
        if zp_limits and zp_limits.get(flt) is not None and zps[i] is not None and zps[i] > zp_limits[flt]
    ]
    ax2.plot(
        filters_checked,
        zps,
        marker="o",
        ls="--",
        label="ZP",
        color=c_zp,
        markevery=[i for i in range(len(zps)) if i not in deep],
    )
    ax2.plot(
        [filters_checked[i] for i in deep],
        [zps[i] for i in deep],
        ls="none",
        marker="x",
        markersize=8,
        markeredgewidth=1.6,
        color=c_zp,
    )
    x_marker = plt.Line2D([], [], ls="none", marker="x", markersize=8, markeredgewidth=1.6, color=c_zp)
    limits = [(zp_limits or {}).get(flt) for flt in filters_checked]
    known = [z for z in zps if z is not None] + [limit for limit in limits if limit is not None]
    shallow, deep = min(known) - 0.3, max(known) + 0.3
    if any(limit is not None for limit in limits):  # the ZP region deeper than each filter's zp_max, one stepped band
        steps = np.array([np.nan if limit is None else limit for limit in limits] + [np.nan], dtype=float)
        ax2.fill_between(
            np.arange(len(filters_checked) + 1) - 0.5,
            steps,
            deep,
            step="post",
            facecolor="gray",
            hatch="xx",
            edgecolor="gray",
            alpha=0.2,
            linewidth=0,
        )
    light_patch = mpatches.Patch(facecolor="gray", hatch="xx", edgecolor="gray", alpha=0.2)
    ax2.set_ylabel("Zero Point (ZP)", color=c_zp)
    ax2.tick_params(axis="y", labelcolor=c_zp)
    ax2.set_ylim(deep, shallow)
    ax1.set_xlim(-0.5, len(filters_checked) - 0.5)

    # Combine legends
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(
        h1 + h2 + [gray_patch, (light_patch, x_marker), blue_patch, red_patch],
        l1
        + l2
        + [
            "Ruled out: not in filter budget",
            "Ruled out: ZP deeper than zp_max",
            f"Inferred Filter ({inferred_filter})",
            f"Recorded Filter ({alleged_filter})",
        ],
        loc="upper left",
        handler_map={tuple: HandlerTuple(ndivide=None)},
    )

    stem = os.path.splitext(os.path.basename(self.input_image))[0]
    plt.title(f"{stem}: Filter Sanity Check for {alleged_filter}")
    plt.tight_layout()

    f = self.path.photometry.figures.filter_check
    plt.savefig(f, dpi=100)
    plt.close()
    return


def plot_zp(
    self,
    mag_key: str,
    src_table: Table,
    zp_arr: np.ndarray,
    zperr_arr: np.ndarray,
    zp: float,
    zperr: float,
    mask: np.ndarray,
    filt: str = None,
) -> None:
    """Generates and saves a zero-point calibration plot.
    The plot shows the zero-point values for each source and the final calibrated zero-point.
    Sources inside and outside the magnitude limits are plotted with different markers.
    Parameters
    ----------
    mag_key : str
        Key for the magnitude column in the source table. e.g., MAG_AUTO
    src_table : astropy.table.Table
        Table containing source measurements and reference magnitudes
    zp_arr : array-like
        Array of individual zero-point values for each source
    zperr_arr : array-like
        Array of zero-point uncertainties for each source
    zp : float
        Final calibrated zero-point value
    zperr : float
        Uncertainty in the final zero-point
    mask : numpy.ndarray
        Boolean mask indicating which sources are within magnitude limits
    filt :
        Filter for zp calculation. e.g., m625
    Returns
    -------
    None
        Saves plot as PNG file in the processed/images directory
    """

    ref_mag_key = f"mag_{filt}" if filt else self.image_info.ref_mag_key
    ref_mag = src_table[ref_mag_key]

    obs_mag = src_table[mag_key]

    plt.errorbar(ref_mag, zp_arr, xerr=0, yerr=zperr_arr, ls="none", c="grey", alpha=0.5)

    plt.plot(
        ref_mag[~mask],
        ref_mag[~mask] - obs_mag[~mask],
        ".",
        c="dodgerblue",
        alpha=0.75,
        zorder=999,
        label=f"{len(ref_mag[~mask])}",
    )

    plt.plot(ref_mag[mask], ref_mag[mask] - obs_mag[mask], "x", c="tomato", alpha=0.75, label=f"{len(ref_mag[mask])}")

    plt.axhline(y=zp, ls="-", lw=1, c="grey", zorder=1, label=f"ZP: {zp:.3f}+/-{zperr:.3f}")
    plt.axhspan(ymin=zp - zperr, ymax=zp + zperr, color="silver", alpha=0.5, zorder=0)
    plt.axvspan(xmin=0, xmax=self.phot_conf.ref_mag_lower, color="silver", alpha=0.25, zorder=0)
    plt.axvspan(xmin=self.phot_conf.ref_mag_upper, xmax=25, color="silver", alpha=0.25, zorder=0)

    plt.xlim([10, 20])
    plt.ylim([zp - 0.25, zp + 0.25])

    plt.xlabel(self.image_info.ref_mag_key)
    plt.ylabel(f"ZP_{mag_key}")

    plt.legend(loc="upper center", ncol=3)
    plt.tight_layout()

    fpath = self.path.photometry.figures.zp(mag_key, filt)
    plt.savefig(fpath, dpi=100)
    plt.close()
