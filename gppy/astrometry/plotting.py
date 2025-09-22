import os
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
from .utils import get_3x3_stars, find_id_rows


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

    # --- Master grid: 2 rows (scatter on top, PSF grid below) ---
    # gs_master = fig.add_gridspec(
    #     2,
    #     1,
    #     height_ratios=[1.4, 3.0],
    #     hspace=0.20,
    # )
    gs_master = fig.add_gridspec(4, 1, height_ratios=[1.4, 0.18, 3.0, 0.06], hspace=0.00)
    # gs_master = fig.add_gridspec(3, 1, height_ratios=[1.4, 3.0, 0.28])

    ax_spacer = fig.add_subplot(gs_master[1])
    ax_spacer.axis("off")
    ax_leg = fig.add_subplot(gs_master[3])
    ax_leg.axis("off")  # for psf plot legend

    # Scatterplot at the top
    ax_scatter = fig.add_subplot(gs_master[0], projection=wcs)
    # PSF grid (rows 1–3)
    gs_psf = gs_master[2].subgridspec(
        3,
        3,
        hspace=0.02,
        wspace=-0.04,
    )
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
        legend_ax=ax_leg,
    )

    # Highlight psf inspected sources, preserving order
    inspected = find_id_rows(matched, matched_ids)

    # Plot FOV scatter plot
    wcs_check_scatter_plot(
        ax_scatter,
        refcat[:num_plot],
        tbl[:num_plot],
        fov_ra=fov_ra,
        fov_dec=fov_dec,
        highlight_ra=inspected["ALPHA_J2000"],
        highlight_dec=inspected["DELTA_J2000"],
    )
    ax_scatter.set_aspect("equal", adjustable="box")

    # file path title
    # fig.subplots_adjust(top=0.90)  # add space at the top
    fig.text(0.5, 0.998, os.path.basename(plot_save_path), ha="center", va="top", fontsize=10)

    # fig.suptitle(f"{num_plot} brightest sources in each catalog", y=0.97)
    ax_scatter.set_title(f"\n{num_plot} Brightest Sources in Each Catalog", ha="center", va="top", fontsize=12, pad=45)

    if sep_stats is not None:
        text = f"Sep Stats: RMS={sep_stats['rms']:.2f}\", Q1={sep_stats['q1']:.2f}\", Q3={sep_stats['q3']:.2f}\""
        if subsecond_fraction is not None:
            text = f"Superarcsec Frac: {1 - subsecond_fraction:.3f}, " + text
        if subpixel_fraction is not None:
            text = f"Subpixel Frac: {subpixel_fraction:.2f}, " + text
        text = f"[{np.sum(~matched['separation'].mask)} Matched Sources]  " + text

        # fig.text(0.5, 0.62, text, ha="center", va="center", fontsize=8)
        ax_spacer.text(0.5, 0.9, text, ha="center", va="center", fontsize=8, transform=ax_spacer.transAxes)

    # PSF block title (centered between blocks)
    # fig.text(0.5, 0.60, f"3x3 PSF Cutouts Across Image ({stretch_type})", ha="center", va="center", fontsize=12)
    ax_spacer.text(
        0.5,
        0.18,
        f"3x3 PSF Cutouts Across Image ({stretch_type})",
        ha="center",
        va="center",
        fontsize=12,
        transform=ax_spacer.transAxes,
    )

    if plot_save_path:
        canvas.print_figure(plot_save_path, bbox_inches="tight", dpi=150)
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
    lon_spacing = _adaptive_ra_spacing(base_arcmin=tick_spacing_arcmin, dec=np.mean(fov_dec))
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
    lon_top.set_ticks(spacing=lon_spacing)  # reduce text overlap
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

    # ax.set_title(f"{len(tbl)} brightest sources in each catalog", pad=30)
    ax.set_title(f"", pad=30)  # give the combined plot some room


def _adaptive_ra_spacing(base_arcmin=15, dec=None, min_cos=0.15):
    """Return an Angle for RA tick spacing that keeps roughly-constant on-sky separation.

    base_arcmin: desired *on-sky* spacing (arcmin) at equator.
    max_ticks:   cap the number of RA ticks to avoid label collisions.
    min_cos:     floor for cos(|dec|) so spacing doesn't blow up at the pole.
    """
    spacing_on_sky = base_arcmin * u.arcmin

    cosd = max(min_cos, np.cos(np.deg2rad(dec)))

    ra_spacing = (spacing_on_sky / cosd).to(u.deg)

    return _round_to_nice_angle(ra_spacing)


def _round_to_nice_angle(angle_deg, base_arcmin=15):
    """
    Round an angle (Quantity in deg) to a 'nice' value acceptable by WCSAxes
    tick locator (multiples of 1, 2, 3, 5 x 10^n).
    """
    import numpy as np
    from astropy import units as u

    # convert to degrees
    val = angle_deg.to(u.deg).value

    # find power of 10
    exp = np.floor(np.log10(val))
    frac = val / 10**exp

    # snap fraction to 1, 2, 3, or 5
    if frac < 1.5:
        nice = 1
    elif frac < 3.5:
        nice = 2
    elif frac < 7.5:
        nice = 5
    else:
        nice = 10

    rounded = nice * 10**exp
    return rounded * u.deg


def wcs_check_psf_plot(
    axes,
    image: str,
    matched_catalog: Table,
    wcs: WCS,
    matched_ids: list[int] = None,
    cutout_size: int = 30,
    stretch_type: str = "Lupton Asinh",
    centroid_legend_idx: int = 2,
    ellipse_legend_idx: int = 2,
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
    else:
        print(f"matched_ids in psf_plot {matched_ids}")
        selected_stars = find_id_rows(matched_catalog, matched_ids)

    # centers in 0-based pixel coords (SExtractor is 1-based)
    x_img = selected_stars["X_IMAGE"].astype(float) - 1.0
    y_img = selected_stars["Y_IMAGE"].astype(float) - 1.0

    for idx, (id, ax) in enumerate(zip(matched_ids, axes)):
        # ax = fig.add_subplot(3, 3, k + 1)
        ax.set_xticks([])
        ax.set_yticks([])

        if id is None:
            ax.text(0.5, 0.5, "No source", ha="center", va="center", transform=ax.transAxes)
            continue

        # add source number (not matched id)
        ax.text(0, 27, str(idx + 1), fontsize=18, color="white")

        # add source stats
        if "ELLIPTICITY" in selected_stars.colnames:
            ax.text(30, 3, f"Ellipticity: {selected_stars[idx]['ELLIPTICITY']:.2f}", fontsize=9, ha="right", color="w")
        text = f"FWHM: "
        if "FWHM_WORLD" in selected_stars.colnames:
            text = text + f"{selected_stars[idx]['FWHM_WORLD']*3600:.1f}\""
        if "FWHM_IMAGE" in selected_stars.colnames:
            text = text + f", {selected_stars[idx]['FWHM_IMAGE']:.1f}pix"
        if len(text) > 10:
            ax.text(30, 0.5, text, fontsize=9, ha="right", color="white")

        # science centroid
        xc = x_img[idx]
        yc = y_img[idx]

        # reference coordinates in pixel coords
        xr, yr = wcs.all_world2pix(
            selected_stars["ra"][idx : idx + 1],
            selected_stars["dec"][idx : idx + 1],
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
        draw_ellipse(
            ax,
            x_cen=float(sci_shifted[0]),
            y_cen=float(sci_shifted[1]),
            a_image=float(selected_stars[idx]["A_IMAGE"]),
            b_image=float(selected_stars[idx]["B_IMAGE"]),
            theta_image=float(selected_stars[idx]["THETA_IMAGE"]),
            edgecolor="yellow",
            linewidth=1.5,
            label="RMS Ellipse (A_IMAGE)" if idx == ellipse_legend_idx else None,  # legend once
        )

        # add windowed (2" gaussian) RMS ellipses
        draw_ellipse(
            ax,
            x_cen=float(sci_shifted[0]),
            y_cen=float(sci_shifted[1]),
            a_image=float(selected_stars[idx]["AWIN_IMAGE"]),
            b_image=float(selected_stars[idx]["BWIN_IMAGE"]),
            theta_image=float(selected_stars[idx]["THETA_IMAGE"]),
            edgecolor="orange",
            linewidth=1.5,
            label="Windowed Ellipse (AWIN_IMAGE)" if idx == ellipse_legend_idx else None,  # legend once
        )

        # add FWHM ellipses
        draw_ellipse(
            ax,
            x_cen=float(sci_shifted[0]),
            y_cen=float(sci_shifted[1]),
            a_image=float(selected_stars[idx]["FWHM_IMAGE"]) / 2,
            b_image=float(selected_stars[idx]["FWHM_IMAGE"] * (1 - selected_stars[idx]["ELLIPTICITY"])) / 2,
            theta_image=float(selected_stars[idx]["THETA_IMAGE"]),
            edgecolor="green",
            linewidth=1.5,
            label="FWHM Ellipse" if idx == ellipse_legend_idx else None,  # legend once
        )

        # add matched id as title
        ax.set_title(f"matched id: {id}", fontsize=8)

        # overlay reference star magnitude
        if "phot_g_mean_mag" in selected_stars.colnames:
            ax.annotate(
                rf"mag$_g$: {selected_stars['phot_g_mean_mag'][idx]:.2f}",
                (ref_shifted[0] - 1, ref_shifted[1] - 4),
                fontsize=9,
                color="r",
                path_effects=[pe.withStroke(linewidth=2, foreground="w")],
            )

        # overlay separation labels optionally
        if "separation" in selected_stars.colnames:
            ax.annotate(
                f"Sep: {selected_stars['separation'][idx]:.2f}\"",
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
                    ncol=4,
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

    return
    # return [matched_catalog[i]["id"] for i in selected_idx if i is not None]


class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        # scale the ellipse into the legend box
        center = width / 2 - xdescent, height / 2 - ydescent
        p = Ellipse(
            xy=center,
            width=width,
            height=height,
            facecolor=orig_handle.get_facecolor(),
            edgecolor=orig_handle.get_edgecolor(),
            lw=orig_handle.get_linewidth(),
        )
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


def cutout(data, coords, x, y, size=30):
    h = size / 2
    x_min = int(x - h)
    x_max = int(x + h) + 1
    y_min = int(y - h)
    y_max = int(y + h) + 1
    data_cut = data[y_min:y_max, x_min:x_max]
    coords_shifted = []
    for coord in coords:
        coord_shifted = (coord[0] - x_min, coord[1] - y_min)
        coords_shifted.append(coord_shifted)
    return data_cut, coords_shifted


def draw_ellipse(
    ax,
    x_cen: float,
    y_cen: float,
    a_image: float,  # SExtractor: RMS along major axis (pixels)
    b_image: float,  # SExtractor: RMS along minor axis (pixels)
    theta_image: float,  # degrees, CCW from +X (SExtractor convention)
    *,
    edgecolor="yellow",
    linewidth=1.2,
    alpha=1.0,
    label=None,
):
    """
    Draw the FWHM ellipse implied by SExtractor's A_IMAGE/B_IMAGE/THETA_IMAGE
    on a Matplotlib Axes (assumes the image is shown with origin='lower').

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    x_cen, y_cen : float
        Ellipse center in pixel coords (0-based if you've converted).
    a_image, b_image : float
        SExtractor RMS lengths along major/minor axes, in pixels.
    theta_image : float
        Position angle in degrees, CCW from +X (SExtractor).
    """
    FWHM_FACTOR = 2.354820045  # 2.0 * np.sqrt(2.0 * np.log(2.0))  why bother calculating?

    # convert sigma to FWHM
    width = FWHM_FACTOR * a_image  # along X' (major), in pixels
    height = FWHM_FACTOR * b_image  # along Y' (minor), in pixels

    e = Ellipse(
        (x_cen, y_cen),
        width=width,
        height=height,
        angle=theta_image,  # Matplotlib uses degrees CCW from +X, matches SExtractor
        fill=False,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        label=label,
    )
    ax.add_patch(e)
    return e
