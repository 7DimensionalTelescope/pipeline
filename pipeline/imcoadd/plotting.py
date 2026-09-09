"""Coadd check plots: the coverage/rejection count planes, and a frame's background mask over its own sky.

Both fill the canvas edge to edge, like the coadd's own JPEG, and carry their title, legend and coordinates
inside the image. Arrays are block-reduced before anything is drawn, and a block's tint is its share of the
input samples lost there, not a flag. Tinting a block because ANY pixel in it was flagged floods the figure:
a detector defect scattered over 1.6% of pixels reaches 93% of blocks at a reduction factor of 13.

Every figure here is oriented RA increasing to the right and Dec increasing upwards, which mirrors the FITS
array whenever its WCS runs the other way. `ImCoadd.plot_coadd_image` applies the same flips so that
`<coadd>.jpg` and `<coadd>_counts.jpg` can be blinked against each other.
"""

import os

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import astropy.units as u
from astropy.coordinates import Angle
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from .counts import COUNT_PLANES

COUNTS_MAX_WIDTH = 5200  # one figure per coadd, so it can afford a factor of 2 on the SWarp grid
MASK_MAX_WIDTH = 1600  # one figure per dynamic single
DPI = 200
JPEG = {"quality": 85, "optimize": True}
DEPTH_CMAP = mcolors.LinearSegmentedColormap.from_list("depth", ["black", "#e8e8e8"])
# (ink, stroke, legend backing): light lettering suits the counts map's black padding, dark the pale sky
LIGHT = ("#f0f0f0", "#000000", "black")
DARK = ("#101010", "#ffffff", "white")


def display_flips(header) -> tuple[bool, bool]:
    """Which array axes to reverse so RA runs rightwards and Dec upwards; (False, False) without a sky WCS."""
    try:
        wcs = WCS(header).celestial
        if not wcs.has_celestial:
            return False, False
        (ra0, dec0), (ra1, _), (_, dec1) = wcs.all_pix2world([[0, 0], [1, 0], [0, 1]], 0)
    except Exception:
        return False, False
    return bool((ra1 - ra0 + 180.0) % 360.0 - 180.0 < 0), bool(dec1 - dec0 < 0)


def orient_for_raster(data, flips):
    """The array as a top-down raster (PIL order) in the figures' RA-right, Dec-up orientation."""
    flip_x, flip_y = flips
    data = data if flip_y else data[::-1]  # PIL puts row 0 at the top; imshow's origin="lower" does not
    return data[:, ::-1] if flip_x else data


def _orient(ax, shape, flips):
    """Reverse the axis limits the sky orientation asks for. The EXTENT must stay in array order, or a tick
    would be labelled with one pixel index while showing another; only the limits may be flipped, and only
    after every imshow, each of which resets them."""
    height, width = shape
    flip_x, flip_y = flips
    ax.set_xlim(*((width, 0) if flip_x else (0, width)))
    ax.set_ylim(*((height, 0) if flip_y else (0, height)))


def _factor(width, max_width):
    return max(1, int(np.ceil(width / max_width)))


def _reduce(array, factor):
    """Block mean, accumulated straight into float32 so no full-size copy of the grid is made."""
    if factor <= 1:
        return np.asarray(array, dtype=np.float32)
    h, w = array.shape[0] // factor * factor, array.shape[1] // factor * factor
    view = array[:h, :w].reshape(h // factor, factor, w // factor, factor)
    return view.sum(axis=(1, 3), dtype=np.float32) / (factor * factor)


def _overlay(ax, counted, factor, total, color, extent, gamma=0.4, peak=1.0):
    """Tint each block by its share of lost samples, gamma-lifted so a faint reason is still visible."""
    # reduce first, then divide: scaling the full grid would allocate a float copy of it
    share = np.clip(_reduce(counted, factor) / total, 0.0, 1.0)
    shade = np.ma.masked_where(share <= 0, peak * share**gamma)
    red, green, blue = mcolors.to_rgb(color)
    cmap = mcolors.LinearSegmentedColormap.from_list("tint", [(red, green, blue, 0.0), (red, green, blue, 1.0)])
    ax.imshow(shade, cmap=cmap, vmin=0.0, vmax=1.0, origin="lower", extent=extent, interpolation="nearest")


def _percent(value) -> str:
    """Enough digits that a rare reason does not read as zero."""
    if value <= 0:
        return "0%"
    if value >= 10:
        return f"{value:.0f}%"
    if value >= 1:
        return f"{value:.1f}%"
    if value >= 0.01:
        return f"{value:.2f}%"
    return f"{value:.2g}%"


def _canvas(shape):
    """A figure the exact pixel size of the reduced array, with one axes filling it."""
    height, width = shape
    fig = Figure(figsize=(width / DPI, height / DPI), dpi=DPI)
    canvas = FigureCanvas(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    return fig, canvas, ax, 0.72 * height / DPI  # points per 1% of the figure height


def _sky_labels(header, width, height, xs, ys):
    """Sexagesimal RA along the x ticks and Dec along the y ticks, both read at mid-frame."""
    try:
        wcs = WCS(header).celestial
        if not wcs.has_celestial:
            return None, None
        ra = wcs.all_pix2world(np.column_stack([xs, np.full_like(xs, height / 2)]), 0)[:, 0]
        dec = wcs.all_pix2world(np.column_stack([np.full_like(ys, width / 2), ys]), 0)[:, 1]
    except Exception:
        return None, None
    hms = Angle(ra, u.deg).wrap_at(360 * u.deg).to_string(unit=u.hour, sep=":", precision=0, pad=True)
    dms = Angle(dec, u.deg).to_string(unit=u.deg, sep=":", precision=0, alwayssign=True, pad=True)
    return list(hms), list(dms)


def _coordinates(ax, width, height, points, palette, header=None):
    """Short thick ticks inside the frame, labelled with the pixel index and the sky coordinate."""
    ink, stroke, _ = palette
    effects = [pe.withStroke(linewidth=2.2, foreground=stroke, alpha=0.85)]
    step = 10 ** np.floor(np.log10(width / 6))
    step *= next(m for m in (1, 2, 5, 10) if width / (step * m) <= 8)
    xs, ys = np.arange(step, width, step), np.arange(step, height, step)
    hms, dms = _sky_labels(header, width, height, xs, ys) if header is not None else (None, None)
    size, lw = 1.05 * points, 0.22 * points
    long, short = 0.013, 0.013 * height / width

    for i, x in enumerate(xs):
        ax.plot([x, x], [0, long], transform=ax.get_xaxis_transform(), color=ink, lw=lw,
                solid_capstyle="butt", path_effects=effects, clip_on=False)  # fmt: skip
        text = f"{x:,.0f}\n{hms[i]}" if hms else f"{x:,.0f}"
        ax.text(x, long + 0.006, text, transform=ax.get_xaxis_transform(), color=ink, fontsize=size,
                ha="center", va="bottom", linespacing=1.35, path_effects=effects)  # fmt: skip
    for i, y in enumerate(ys):
        ax.plot([0, short], [y, y], transform=ax.get_yaxis_transform(), color=ink, lw=lw,
                solid_capstyle="butt", path_effects=effects, clip_on=False)  # fmt: skip
        text = f"{y:,.0f}\n{dms[i]}" if dms else f"{y:,.0f}"
        ax.text(short + 0.004, y, text, transform=ax.get_yaxis_transform(), color=ink, fontsize=size,
                ha="center", va="top", rotation=90, rotation_mode="anchor", linespacing=1.35,
                path_effects=effects)  # fmt: skip


def _label(ax, name, subtitle, handles, points, palette):
    ink, stroke, backing = palette
    ax.text(0.5, 0.994, f"{name}\n{subtitle}" if subtitle else name, transform=ax.transAxes, color=ink,
            fontsize=1.5 * points, ha="center", va="top", linespacing=1.6, zorder=5,
            path_effects=[pe.withStroke(linewidth=2.2, foreground=stroke, alpha=0.8)],
            bbox=dict(facecolor=backing, alpha=0.4, edgecolor="none", pad=0.5 * points))  # fmt: skip
    if handles:
        legend = ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.994, 0.045),
                           fontsize=1.25 * points, labelcolor=ink, facecolor=backing, edgecolor="none",
                           framealpha=0.45, borderpad=0.7, labelspacing=0.55, handlelength=1.6)  # fmt: skip
        legend.set_zorder(5)


def _write(fig, canvas, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.print_figure(output_path, dpi=DPI, pil_kwargs=JPEG)
    return output_path


def plot_coadd_counts(planes, output_path, name, subtitle="", n_inputs=None, header=None,
                      max_width=COUNTS_MAX_WIDTH):  # fmt: skip
    """Geometric coverage in grey, one colour per rejection reason, drawn in MaskBit order."""
    grey = "NGEOM" if planes.get("NGEOM") is not None else "NUSED"
    depth = planes.get(grey)
    if depth is None:
        return None
    height, width = depth.shape
    extent = (0, width, 0, height)
    factor = _factor(width, max_width)
    total = float(n_inputs or max(int(depth.max()), 1))

    reduced = _reduce(depth, factor)
    fig, canvas, ax, points = _canvas(reduced.shape)
    ax.imshow(reduced, cmap=DEPTH_CMAP, vmin=0, vmax=total, origin="lower", extent=extent, interpolation="nearest")

    handles = [Patch(facecolor="#9a9a9a", edgecolor="none",
                     label=f"grey  {grey}: 0 to {int(depth.max())} input frames per pixel")]  # fmt: skip
    drawn = [p for p in COUNT_PLANES if p.bit is not None and planes.get(p.name) is not None]
    for plane in sorted(drawn, key=lambda p: int(p.bit)):
        counted = planes[plane.name]
        _overlay(ax, counted, factor, total, plane.color, extent)
        flagged = int(np.count_nonzero(counted))
        handles.append(
            Patch(
                facecolor=plane.color,
                edgecolor="none",
                label=f"{plane.name}  {plane.short}: {flagged:,} px "
                f"({_percent(100 * flagged / counted.size)}), {_percent(100 * counted.mean() / total)} of samples",
            )
        )
    _orient(ax, depth.shape, display_flips(header))
    _coordinates(ax, width, height, points, LIGHT, header)
    _label(ax, name, subtitle, handles, points, LIGHT)
    return _write(fig, canvas, output_path)


def plot_source_mask(data, excluded, output_path, name, subtitle="", n_inputs=None, header=None,
                     max_width=MASK_MAX_WIDTH):  # fmt: skip
    """The image the background meshes were fitted on, with the excluded pixels washed red."""
    height, width = data.shape
    extent = (0, width, 0, height)
    factor = _factor(width, max_width)
    image = _reduce(np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0), factor)
    # stretched on the image's own pixel noise, not on the block average, so sources stand out of a flat sky;
    # exact zeros are SWarp's uncovered exterior and would drag the limits
    sample = data[::4, ::4].ravel()
    finite = sample[np.isfinite(sample) & (sample != 0.0)]
    sample = finite if finite.size >= 100 else sample
    high = ZScaleInterval().get_limits(sample)[1]
    sky = float(np.median(sample))
    vmin = sky - 0.20 * max(high - sky, 1e-6)  # sky just off white, so the red wash reads against it

    fig, canvas, ax, points = _canvas(image.shape)
    ax.imshow(image, cmap="gray_r", vmin=vmin, vmax=high, origin="lower", extent=extent, interpolation="nearest")
    _overlay(ax, excluded, factor, float(n_inputs or 1), "#e5484d", extent, gamma=0.5, peak=0.75)
    _orient(ax, data.shape, display_flips(header))
    _coordinates(ax, width, height, points, DARK, header)
    _label(ax, name, subtitle,
           [Patch(facecolor="#e5484d", edgecolor="none", label="excluded from the background mesh")],
           points, DARK)  # fmt: skip
    return _write(fig, canvas, output_path)
