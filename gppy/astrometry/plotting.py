import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import ZScaleInterval
from astropy import units as u
from astropy.wcs import WCS

from matplotlib.patches import Rectangle
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas  # thread-safe, but savefig only.
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from PIL import Image, ImageEnhance


def wcs_check_scatter_plot(
    refcat: Table,
    tbl: Table,
    wcs: WCS,
    plot_save_path=None,
    fov_ra=None,
    fov_dec=None,
    tick_spacing_arcmin=15,  # warning occurs if not a multiple of 15
    legend_loc="upper right",
):

    fig = Figure(figsize=(6, 4))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1, projection=wcs)

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
    lon.set_ticks(spacing=spacing)
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

    # Add a twin RA axis on top, in degrees

    # Top overlay axis in degrees
    overlay = ax.get_coords_overlay("icrs")  # same frame
    lon_top = overlay[0]  # <— NOTE: version dependent api.
    lon_top.set_format_unit(u.deg)
    lon_top.set_major_formatter("d.dd")
    lon_top.set_ticks(spacing=spacing)
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
    lat_right.set_ticks(spacing=2 * spacing)
    lat_right.set_ticklabel(size=8)

    # Grid at the (major) tick positions
    ax.coords.grid(True, color="gray", ls=":", alpha=0.6)

    # legend – anchor in axes coords to be explicit
    ax.legend(loc=legend_loc, bbox_to_anchor=(0.98, 0.98), frameon=True, borderaxespad=0.0, fontsize=8)

    ax.set_title(f"{len(tbl)} brightest sources in each catalog", pad=20)

    if plot_save_path is not None:
        canvas.print_figure(plot_save_path, bbox_inches="tight", dpi=150)
    else:
        fig.tight_layout()
        fig.show()


def wcs_check_psf_plot(
    image: str,
    matched_catalg: Table,
    wcs: WCS,
    plot_save_path=None,
):
    fig = Figure(figsize=(6, 4))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)

    data = fits.getdata(image)

    x_center = matched_catalg[0]["X_IMAGE"] - 1
    y_center = matched_catalg[0]["Y_IMAGE"] - 1
    x_ref, y_ref = wcs.all_world2pix(matched_catalg["ra"][:1], matched_catalg["dec"][:1], 0)

    coords = [(x_center, y_center), (x_ref[0], y_ref[0])]
    data_cut, coords_shifted = cutout(data, coords, x_center, y_center)
    sci_coord_shifted = coords_shifted[0]
    ref_coord_shifted = coords_shifted[1]

    ax.imshow(
        np.log10(data_cut),
        cmap="gray",
    )

    # Rectangle overlay: edge only
    rect = Rectangle(
        sci_coord_shifted,
        1,  # width
        1,  # height
        edgecolor="C0",
        facecolor="none",
        linewidth=1.5,
    )
    ax.add_patch(rect)

    rect = Rectangle(
        ref_coord_shifted,
        1,  # width
        1,  # height
        edgecolor="red",
        facecolor="none",
        linewidth=1.5,
    )
    ax.add_patch(rect)

    ax.annotate(
        f"sep: {matched_catalg['separation'][0]:.2f} arcsec",
        ref_coord_shifted,
        fontsize=8,
    )

    if plot_save_path is not None:
        canvas.print_figure(plot_save_path, bbox_inches="tight", dpi=150)
    else:
        fig.tight_layout()
        fig.show()


try:
    from scipy.spatial import cKDTree as KDTree

    _HAVE_KDTREE = True
except Exception:
    _HAVE_KDTREE = False


def wcs_check_psf_grid_plot(
    image: str,
    matched_catalog: Table,  # cleaned: required cols present; no masks
    wcs: WCS,
    plot_save_path=None,
    cutout_size: int = 30,
):
    """Select 9 sources nearest to a 3x3 grid over the image and plot PSF cutouts."""
    data = fits.getdata(image)
    H, W = data.shape  # numpy: (y, x)

    # remove ref-only rows
    matched_catalog = matched_catalog[~matched_catalog["separation"].mask]

    # centers in 0-based pixel coords (SExtractor is 1-based)
    x_img = matched_catalog["X_IMAGE"].astype(float) - 1.0
    y_img = matched_catalog["Y_IMAGE"].astype(float) - 1.0

    # 3x3 target points (corners, edge centers, center), clamped so cutouts fit
    margin = int(np.ceil(cutout_size / 2)) + 1
    xs = np.array([0, W / 2, W - 1], dtype=float)
    ys = np.array([0, H / 2, H - 1], dtype=float)
    targets = np.array([(x, y) for y in ys for x in xs], dtype=float)
    targets[:, 0] = np.clip(targets[:, 0], margin, W - 1 - margin)
    targets[:, 1] = np.clip(targets[:, 1], margin, H - 1 - margin)

    # Precompute candidate matrix
    cand_xy = np.c_[x_img, y_img]

    # For each target, pick nearest candidate; if already taken, use next nearest
    selected_idx = _select_3x3_by_nearest(cand_xy, targets)

    # Build figure
    fig = Figure(figsize=(8, 8))
    canvas = FigureCanvas(fig)

    for k in range(9):
        ax = fig.add_subplot(3, 3, k + 1)
        ax.set_xticks([])
        ax.set_yticks([])

        idx = selected_idx[k]
        if idx is None:
            ax.text(0.5, 0.5, "No source", ha="center", va="center", transform=ax.transAxes)
            continue

        # science centroid
        xc = x_img[idx]
        yc = y_img[idx]

        # reference pixel from RA/Dec
        xr, yr = wcs.all_world2pix(
            matched_catalog["ra"][idx : idx + 1],
            matched_catalog["dec"][idx : idx + 1],
            0,  # origin=0 because we converted to 0-based
        )
        xr = float(xr[0])
        yr = float(yr[0])

        # cutout around (xc, yc)
        data_cut, coords_shifted = cutout(data, [(xc, yc), (xr, yr)], xc, yc, size=cutout_size)
        sci_shifted, ref_shifted = coords_shifted

        # display (log stretch with floor to avoid log(0))
        ax.imshow(np.log10(np.maximum(data_cut, 1e-3)), cmap="gray", origin="lower")

        # overlay centroids
        sci_rect = Rectangle(
            sci_shifted,
            1,
            1,
            edgecolor="C0",
            facecolor="none",
            linewidth=1.0,
            label="Source Centroid",
        )
        ref_rect = Rectangle(
            ref_shifted,
            1,
            1,
            edgecolor="red",
            facecolor="none",
            linewidth=1.0,
            label="Reference Coordinates",
        )
        ax.add_patch(sci_rect)
        ax.add_patch(ref_rect)

        # optional separation label if present
        if "separation" in matched_catalog.colnames:
            ax.annotate(
                f"sep: {matched_catalog['separation'][idx]:.2f}\"", (ref_shifted[0], ref_shifted[1]), fontsize=7
            )

        ax.set_title(f"matched id: {matched_catalog['id'][idx]}", fontsize=8)
        if k == 0:
            ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    if plot_save_path:
        canvas.print_figure(plot_save_path, bbox_inches="tight", dpi=150)
    else:
        fig.show()

    return matched_catalog[selected_idx]["id"]


def _select_3x3_by_nearest(cand_xy, targets):
    """
    For each target, pick the nearest candidate (by Euclidean distance).
    Deduplicate: if the same source would be used twice, assign to the target
    where it is closer; for the other target, pick next-nearest.

    Returns a list of length 9 with catalog indices (into the original table)
    or None if no candidate available (should be rare given the logic).
    """
    selected = [None] * len(targets)
    taken = set()

    if _HAVE_KDTREE and len(cand_xy) > 0:
        tree = KDTree(cand_xy)
        # Query more than 1 in case of duplicates; cap at len(cand_xy)
        k_step = 5
        max_k = min(200, len(cand_xy))
        for ti, t in enumerate(targets):
            k = k_step
            while selected[ti] is None and k <= max_k:
                dists, nn = tree.query(t, k=k)
                if np.isscalar(nn):
                    nn = np.array([nn])
                    dists = np.array([dists])
                # Sort by distance explicitly (tree.query already returns ordered, but be safe)
                order = np.argsort(dists)
                for oi in order:
                    ci = int(nn[oi])
                    gi = int(ci)
                    if gi not in taken:
                        selected[ti] = gi
                        taken.add(gi)
                        break
                k += k_step

    else:
        # NumPy fallback: brute-force distances
        for ti, t in enumerate(targets):
            if len(cand_xy) == 0:
                break
            d = np.hypot(cand_xy[:, 0] - t[0], cand_xy[:, 1] - t[1])
            order = np.argsort(d)
            for ci in order:
                gi = int(ci)
                if gi not in taken:
                    selected[ti] = gi
                    taken.add(gi)
                    break

    return selected


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
