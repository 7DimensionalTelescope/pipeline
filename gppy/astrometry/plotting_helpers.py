from typing import List, Tuple
import numpy as np
from astropy import units as u
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Ellipse


def adaptive_ra_spacing(base_arcmin=15, dec=None, min_cos=0.2):
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
    Round an angle (Quantity in deg) to a 'nice' value that is a multiple of
    base_arcmin arcminutes (default: 15 arcmin = 0.25 deg). This avoids warning
    in the plotting.
    """
    import numpy as np
    from astropy import units as u

    # Convert base arcmin to degrees
    base_deg = (base_arcmin * u.arcmin).to(u.deg).value

    # Convert input to degrees
    val = angle_deg.to(u.deg).value

    # Round to nearest multiple of base_deg
    rounded_val = np.round(val / base_deg) * base_deg

    return rounded_val * u.deg


def cutout(data: np.array, coords: List[Tuple[float]], x: float, y: float, size: int = 30):
    """Derive the shifted coordinates in the cutout frame too"""

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


def show_agg(fig, canvas, **savefig_kwargs):
    """
    Not used in the pipeline, but useful when you want to see the plot
    in the jupyter notebook.

    example:
    wcs_check_psf_plot(...)
    show_agg(fig, canvas)  # instead of plt.show()
    """
    import io
    from IPython.display import display, Image

    canvas.draw()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", **savefig_kwargs)
    buf.seek(0)
    display(Image(data=buf.read()))
