import os
from typing import List, Tuple
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import ZScaleInterval
from astropy import units as u
from astropy.wcs import WCS

from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Ellipse
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas  # thread-safe, but savefig only.
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.patheffects as pe
from PIL import Image, ImageEnhance


from ..utils import lupton_asinh
from .utils import get_3x3_stars, find_id_rows, resolve_rows_by_id
from .plotting_helpers import cutout, adaptive_ra_spacing, draw_ellipse, HandlerEllipse


def wcs_check_plot(
    refcat,
    tbl,
    matched,
    wcs,
    image,
    plot_save_path=None,
    fov_ra=None,
    fov_dec=None,
    num_plot=50,
    sep_stats=None,
    subpixel_fraction=None,
    subsecond_fraction=None,
    matched_ids=None,  # row index of matched catalog
    cutout_size=30,
    stretch_type="Lupton Asinh",  # "Linear" "Log"
):
    fig = Figure(figsize=(7, 12))
    fig.set_constrained_layout(True)
    canvas = FigureCanvas(fig)

    # --- Master grid: 4 rows (scatter on top, PSF grid below, spacers between) ---
    gs_master = fig.add_gridspec(4, 1, height_ratios=[1.4, 0.18, 3.1, 0.06], hspace=0.00)

    ax_spacer = fig.add_subplot(gs_master[1])
    ax_spacer.axis("off")
    ax_leg = fig.add_subplot(gs_master[3])
    ax_leg.axis("off")  # for psf plot legend

    # Scatterplot at the top
    ax_scatter = fig.add_subplot(gs_master[0], projection=wcs)

    # PSF grid (rows 1–3)
    gs_psf = gs_master[2].subgridspec(3, 3, hspace=0.02, wspace=-0.04)
    psf_axes = [fig.add_subplot(gs_psf[i, j]) for i in range(3) for j in range(3)]

    # Plot PSF grid
    wcs_check_psf_plot(
        psf_axes,
        image,
        matched,
        wcs,
        matched_ids=matched_ids,
        cutout_size=cutout_size,
        stretch_type=stretch_type,
        title_ax=ax_spacer,
        legend_ax=ax_leg,
    )

    # Highlight psf inspected sources, preserving order
    if matched_ids is not None:
        rows = resolve_rows_by_id(matched, matched_ids)
        hi_ra = [float(r["ALPHA_J2000"]) for r in rows if r is not None]
        hi_dec = [float(r["DELTA_J2000"]) for r in rows if r is not None]
    else:
        hi_ra, hi_dec = None, None

    # Plot FOV scatter plot
    wcs_check_scatter_plot(
        ax_scatter,
        refcat[:num_plot],
        tbl[:num_plot],
        fov_ra=fov_ra,
        fov_dec=fov_dec,
        highlight_ra=hi_ra,
        highlight_dec=hi_dec,
    )

    # file path title
    # fig.subplots_adjust(top=0.90)  # add space at the top
    fig.text(0.5, 0.998, os.path.basename(plot_save_path), ha="center", va="top", fontsize=10)

    # # Separation statistics (centered between blocks)
    if sep_stats is not None:
        text = f"Sep Stats: RMS={sep_stats['rms']:.2f}\", Q1={sep_stats['q1']:.2f}\", Q3={sep_stats['q3']:.2f}\""
        if subsecond_fraction is not None:
            text = f"Superarcsec Frac: {1 - subsecond_fraction:.3f}, " + text
        if subpixel_fraction is not None:
            text = f"Subpixel Frac: {subpixel_fraction:.2f}, " + text
        text = (
            f"[{np.sum(~matched['separation'].mask)} Matched Sources {np.sum(~matched['separation'].mask)/len(matched)*100:.1f}%]  "
            + text
        )

        # fig.text(0.5, 0.62, text, ha="center", va="center", fontsize=8)
        ax_spacer.text(0.47, 0.9, text, ha="center", va="center", fontsize=8, transform=ax_spacer.transAxes)

    if plot_save_path:
        # canvas.print_figure(plot_save_path, bbox_inches="tight", dpi=150)
        canvas.print_figure(plot_save_path, dpi=150)
    else:
        fig.show()


def wcs_check_scatter_plot(
    ax,
    refcat: Table,
    tbl: Table,
    # wcs: WCS,
    fov_ra=None,
    fov_dec=None,
    highlight_ra=None,
    highlight_dec=None,
    tick_spacing_arcmin=15,  # warning occurs if not a multiple of 15
):
    # scatter in world coords
    ax.scatter(
        tbl["ALPHA_J2000"],
        tbl["DELTA_J2000"],
        marker="x",
        alpha=0.8,
        s=15,
        zorder=10,
        label="Image Sources (Scamp WCS)",
        transform=ax.get_transform("world"),
    )
    ax.scatter(
        refcat["ra"],
        refcat["dec"],
        alpha=0.5,
        s=20,
        label="Reference Catalog (Gaia DR3)",
        transform=ax.get_transform("world"),
    )

    if highlight_ra is not None and highlight_dec is not None:
        ax.scatter(
            highlight_ra,
            highlight_dec,
            marker="o",
            alpha=0.8,
            s=50,
            facecolor="none",
            edgecolor="red",
            zorder=10,
            label="Cutout Inspected Sources",
            transform=ax.get_transform("world"),
        )
        # Annotate each highlighted source with its index (1-based)
        for i, (ra, dec) in enumerate(zip(highlight_ra, highlight_dec), start=1):
            ax.annotate(
                str(i),
                (ra, dec),
                xycoords=ax.get_transform("world"),  # <-- key line
                xytext=(3, 3),
                textcoords="offset points",
                color="red",
                fontsize=7,
                ha="left",
                va="bottom",
                zorder=11,
            )

    # optional FOV outline
    if fov_ra is not None and fov_dec is not None:
        ax.plot(
            np.r_[fov_ra, fov_ra[:1]],
            np.r_[fov_dec, fov_dec[:1]],
            ls="--",
            color="k",
            label="FOV",
            transform=ax.get_transform("world"),
        )

    # coordinate helpers
    lon = ax.coords[0]
    lat = ax.coords[1]

    # axis labels (WCSAxes wants this, not ax.set_xlabel/ylable)
    lon.set_axislabel("RA")
    lat.set_axislabel("Dec")

    # --- Major ticks at your chosen spacing ---
    spacing = tick_spacing_arcmin * u.arcmin
    dec_center = np.mean(fov_dec) if fov_dec is not None else np.mean(tbl["DELTA_J2000"])
    lon_spacing = adaptive_ra_spacing(base_arcmin=tick_spacing_arcmin, dec=dec_center)
    lon.set_ticks(spacing=lon_spacing)
    lat.set_ticks(spacing=spacing)
    lon.set_ticklabel(size=8)
    lat.set_ticklabel(size=8)

    # nice sexagesimal formatting for dense ticks
    # (works across Astropy WCSAxes versions)
    try:
        lon.set_major_formatter("hh:mm")
        # lon.set_major_formatter("d.ddd")
        lat.set_major_formatter("dd:mm:ss")
    except Exception:
        # older versions fall back automatically; safe to ignore
        pass

    # Add a twin RA axis overlay on top, in degrees
    overlay = ax.get_coords_overlay("icrs")  # same frame
    lon_top = overlay[0]
    lon_top.set_format_unit(u.deg)
    lon_top.set_major_formatter("d.dd")
    lon_top.set_ticks(spacing=2 * lon_spacing)  # reduce text overlap
    lon_top.set_ticklabel(size=8)

    # Move labels/ticks to the top
    lon_top.set_axislabel("")
    lon_top.set_axislabel_position("t")
    lon_top.set_ticklabel_position("t")
    lon_top.set_ticks_position("t")

    # right overlay axis in degrees
    lat_right = overlay[1]
    lat_right.set_axislabel("")
    # lat_right.set_ticklabel_visible(False)
    lat_right.set_major_formatter("d.dd")
    lat_right.set_ticks(spacing=spacing)
    lat_right.set_ticklabel(size=8)

    # Grid at the (major) tick positions
    ax.coords.grid(True, color="gray", ls=":", alpha=0.6)

    # legend – anchor in axes coords to be explicit
    leg = ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        borderaxespad=0.0,
        fontsize=8,
        facecolor="white",
        framealpha=0.8,
    )
    leg.set_zorder(30)

    ax.set_aspect("equal", adjustable="box")
    num_plot = len(tbl)
    ax.set_title(f"\n{num_plot} Brightest Sources in Each Catalog", ha="center", va="top", fontsize=12, pad=45)
    # fig.suptitle(f"{num_plot} brightest sources in each catalog", y=0.97)


def wcs_check_psf_plot(
    axes,
    image: str,
    matched_catalog: Table,
    wcs: WCS,
    matched_ids: list[int | None] = None,
    cutout_size: int = 30,
    stretch_type: str = "Lupton Asinh",
    centroid_legend_idx: int = 2,
    ellipse_legend_idx: int = 2,
    title_ax=None,
    legend_ax=None,
):
    """
    Select 9 sources nearest to a 3x3 grid over the image and plot PSF cutouts.
    matched_catalog must be cleaned first to contain no masked rows.
    """
    data = fits.getdata(image)
    H, W = data.shape  # numpy: (y, x)

    if matched_ids is None:
        print(f"matched_ids not provided. Selecting sources now.")
        # # remove ref-only rows
        matched_catalog = matched_catalog[~matched_catalog["separation"].mask]
        if len(matched_catalog) == 0:
            return
        selected_idx = get_3x3_stars(matched_catalog, H, W, cutout_size, return_id=False)
        selected_stars = matched_catalog[selected_idx]
        rows = [selected_stars[i] if i is not None else None for i in selected_idx]
    else:
        # print(f"matched_ids in psf_plot {matched_ids}")
        # selected_stars = find_id_rows(matched_catalog, matched_ids)
        rows = resolve_rows_by_id(matched_catalog, matched_ids)  # list of Table row or None

    if all(row is None for row in rows):
        print("[ERROR] No sources found in the matched catalog")
        return

    # centers in 0-based pixel coords (SExtractor is 1-based)
    # x_img = selected_stars["X_IMAGE"].astype(float) - 1.0
    # y_img = selected_stars["Y_IMAGE"].astype(float) - 1.0

    # get a row for colname check
    for row in rows:
        if row is not None:
            break

    if all(col in row.colnames for col in ("A_IMAGE", "B_IMAGE", "THETA_IMAGE")):
        has_a_image = True
    else:
        has_a_image = False
        print("A_IMAGE, B_IMAGE, THETA_IMAGE not found")

    if all(col in row.colnames for col in ("AWIN_IMAGE", "BWIN_IMAGE", "THETA_IMAGE")):
        has_awin_image = True
    else:
        has_awin_image = False
        print("AWIN_IMAGE, BWIN_IMAGE, THETA_IMAGE not found")

    if all(col in row.colnames for col in ("FWHM_IMAGE", "ELLIPTICITY", "THETA_IMAGE")):
        has_fwhm_image = True
    else:
        has_fwhm_image = False
        print("FWHM_IMAGE, ELLIPTICITY, THETA_IMAGE not found")

    # for idx, (id, ax) in enumerate(zip(matched_ids, axes)):
    for idx, (id_, ax) in enumerate(zip(matched_ids, axes)):
        row = rows[idx] if idx < len(rows) else None

        ax.set_xticks([])
        ax.set_yticks([])

        # if id is None:
        if (id_ is None) or (row is None):
            ax.text(0.5, 0.5, "No source", ha="center", va="center", transform=ax.transAxes)
            continue

        # add source number (not matched id)
        ax.text(0, 27, str(idx + 1), fontsize=18, color="white")

        # add source stats
        ax.text(30, 28.5, f"{row['X_IMAGE']:.0f}, {row['Y_IMAGE']:.0f}", fontsize=9, ha="right", color="w")
        if "ELLIPTICITY" in row.colnames:
            ax.text(30, 3, f"Ellipticity: {row['ELLIPTICITY']:.2f}", fontsize=9, ha="right", color="w")
        text = f"FWHM: "
        if "FWHM_WORLD" in row.colnames:
            text = text + f"{row['FWHM_WORLD']*3600:.1f}\""
        if "FWHM_IMAGE" in row.colnames:
            text = text + f", {row['FWHM_IMAGE']:.1f}pix"
        if len(text) > 10:
            ax.text(30, 0.5, text, fontsize=9, ha="right", color="white")

        # science centroid
        xc = float(row["X_IMAGE"]) - 1.0
        yc = float(row["Y_IMAGE"]) - 1.0

        # reference coordinates in pixel coords
        xr, yr = wcs.all_world2pix(
            [row["ra"]],
            [row["dec"]],
            0,  # origin=0 because we converted to 0-based
        )
        xr = float(xr[0])
        yr = float(yr[0])

        # cutout around (xc, yc)
        data_cut, coords_shifted = cutout(data, [(xc, yc), (xr, yr)], xc, yc, size=cutout_size)
        sci_shifted, ref_shifted = coords_shifted

        # np.save(f"data_cut_{idx}.npy", data_cut)

        # Cutout display
        if stretch_type == "Lupton Asinh":
            NOISE_SCALE = 10  # hard-coded
            SKYVAL = 7
            ax.imshow(lupton_asinh(data_cut, sky=SKYVAL, noise=NOISE_SCALE), cmap="gray", origin="lower")
        elif stretch_type == "Log":
            ax.imshow(np.log10(np.maximum(data_cut, 1e-3)), cmap="gray", origin="lower")
        elif stretch_type == "Linear":
            ax.imshow(data_cut, cmap="gray", origin="lower")
        else:
            raise ValueError(f"Invalid stretch type: {stretch_type}")

        # overlay centroids
        sci_rect = Rectangle(
            (sci_shifted[0] - 0.5, sci_shifted[1] - 0.5),  # rectangle is centered at the lower-left corner
            1,  # width
            1,  # height
            edgecolor="dodgerblue",
            facecolor="none",
            linewidth=2.0,
            zorder=10,
            label="Source Centroid" if idx == centroid_legend_idx else None,
        )
        ref_rect = Rectangle(
            (ref_shifted[0] - 0.5, ref_shifted[1] - 0.5),  # rectangle is centered at the lower-left corner
            1,
            1,
            edgecolor="red",
            facecolor="none",
            linewidth=1.0,
            zorder=11,
            label="Reference Coordinates" if idx == centroid_legend_idx else None,
        )
        ax.add_patch(sci_rect)
        ax.add_patch(ref_rect)

        # add RMS (flux second moment) ellipses
        if has_a_image:
            draw_ellipse(
                ax,
                x_cen=float(sci_shifted[0]),
                y_cen=float(sci_shifted[1]),
                a_image=float(row["A_IMAGE"]),
                b_image=float(row["B_IMAGE"]),
                theta_image=float(row["THETA_IMAGE"]),
                edgecolor="yellow",
                linewidth=1.5,
                label="RMS Ellipse (A_IMAGE)" if idx == ellipse_legend_idx else None,  # legend once
            )

        # add windowed (2" gaussian) RMS ellipses
        if has_awin_image:
            draw_ellipse(
                ax,
                x_cen=float(sci_shifted[0]),
                y_cen=float(sci_shifted[1]),
                a_image=float(row["AWIN_IMAGE"]),
                b_image=float(row["BWIN_IMAGE"]),
                theta_image=float(row["THETA_IMAGE"]),
                edgecolor="orange",
                linewidth=1.5,
                label="Windowed Ellipse (AWIN_IMAGE)" if idx == ellipse_legend_idx else None,  # legend once
            )

        # add FWHM ellipses
        if has_fwhm_image:
            draw_ellipse(
                ax,
                x_cen=float(sci_shifted[0]),
                y_cen=float(sci_shifted[1]),
                a_image=float(row["FWHM_IMAGE"]) / 2,
                b_image=float(row["FWHM_IMAGE"] * (1 - row["ELLIPTICITY"])) / 2,
                theta_image=float(row["THETA_IMAGE"]),
                edgecolor="green",
                linewidth=1.5,
                label="FWHM Ellipse" if idx == ellipse_legend_idx else None,  # legend once
            )

        # add matched id as title
        ax.set_title(f"matched id: {id_}", fontsize=8)

        # overlay reference star magnitude
        if "phot_g_mean_mag" in row.colnames:
            ax.annotate(
                rf"mag$_g$: {row['phot_g_mean_mag']:.2f}",
                (ref_shifted[0] - 1, ref_shifted[1] - 4),
                fontsize=9,
                color="r",
                path_effects=[pe.withStroke(linewidth=2, foreground="w")],
            )

        # overlay separation labels optionally
        if "separation" in row.colnames:
            ax.annotate(
                f"Sep: {row['separation']:.2f}\"",
                (sci_shifted[0], sci_shifted[1] + 3),
                fontsize=9,
                color="magenta",
                path_effects=[pe.withStroke(linewidth=2, foreground="w")],
            )

        # legend placement
        if legend_ax is None:  # a compromise: add divided legends to only two subplots
            if idx == centroid_legend_idx or idx == ellipse_legend_idx:
                ax.legend(loc="upper right", fontsize=8)
        else:  # a joint legend outside the plot, if the space is given as legend_ax
            handles, labels = [], []
            for ax in axes:
                h, l = ax.get_legend_handles_labels()
                handles.extend(h)
                labels.extend(l)

            # replace ellipse entries with elliptical proxies
            proxy_handles = []
            for h, l in zip(handles, labels):
                if "ellipse" in l.lower():
                    color = h.get_edgecolor() if hasattr(h, "get_edgecolor") else h.get_color()
                    proxy = Ellipse((0, 0), width=1, height=0.6, facecolor="none", edgecolor=color, lw=1.5)
                    proxy_handles.append(proxy)
                else:
                    proxy_handles.append(h)

            if proxy_handles:
                legend_ax.legend(
                    proxy_handles,
                    labels,
                    loc="center",
                    bbox_to_anchor=(0.5, 0.5),
                    bbox_transform=legend_ax.transAxes,
                    ncol=3,
                    frameon=True,
                    fontsize=8,
                    borderaxespad=0.2,
                    columnspacing=1.2,
                    handlelength=2.0,
                    handler_map={Ellipse: HandlerEllipse()},
                )

            # if handles:
            #     legend_ax.legend(
            #         handles,
            #         labels,
            #         loc="center",
            #         ncol=4,
            #         frameon=True,
            #         fontsize=8,
            #         borderaxespad=0.2,
            #         columnspacing=1.2,
            #         handlelength=2.0,
            #     )

        # PSF block title (centered between blocks)
        if title_ax is not None:
            title_ax.text(
                0.5,
                0.18,
                f"3x3 PSF Cutouts Across Image ({stretch_type})",
                ha="center",
                va="center",
                fontsize=12,
                transform=title_ax.transAxes,
            )
        else:
            fig = ax.get_figure()
            fig.suptitle(
                f"3x3 PSF Cutouts Across Image ({stretch_type})",
                fontsize=12,
            )
    return
    # return [matched_catalog[i]["id"] for i in selected_idx if i is not None]
