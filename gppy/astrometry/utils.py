import unicodedata
from typing import List
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import SkyCoord, SkyOffsetFrame

from ..tools.table import match_two_catalogs, add_id_column
from ..utils import add_suffix, swap_ext
from .plotting import wcs_check_plot
from ..subtract.utils import create_ds9_region_file


def read_scamp_header(file, return_wcs=False):
    """
    Read a SCAMP output HEAD file, normalizing unicode and correcting WCS types.

    Args:
        file (str): Path to the header file

    Returns:
        fits.Header: Processed and cleaned FITS header with corrected WCS types

    Note:
        - Removes non-ASCII characters
        - Converts WCS projection type from TAN to TPV
    """

    with open(file, "r", encoding="utf-8") as f:
        content = f.read()

    # Clean non-ASCII characters
    cleaned_string = unicodedata.normalize("NFKD", content).encode("ascii", "ignore").decode("ascii")

    # Correct CTYPE (TAN --> TPV)
    hdr = fits.Header.fromstring(cleaned_string, sep="\n")
    hdr["CTYPE1"] = ("RA---TPV", "WCS projection type for this axis")
    hdr["CTYPE2"] = ("DEC--TPV", "WCS projection type for this axis")

    if return_wcs:
        return WCS(hdr)

    return hdr


def build_wcs(ra_deg, dec_deg, crpix1, crpix2, pixscale_arcsec, pa_deg, flip=True):
    """
    Build an Astropy WCS with TAN projection and a CD matrix from pixel scale + PA.
    FITS convention assumed:
    - +X increases to the right; +Y increases up
    - For PA = 0 deg (north up, east left), CD becomes diag([-s, +s]) in deg/pix
    - PA measured from +Y (north) toward +X (east)
    - If flip is True, the image is flipped along the RA-axis. 7DT images are flipped.
    """
    s = pixscale_arcsec / 3600.0  # deg/pix
    pa = np.deg2rad(pa_deg)

    # CD matrix per FITS standard for a PA about the reference point:
    #   [CD1_1  CD1_2] = s * [ -cos(PA)   sin(PA) ]
    #   [CD2_1  CD2_2]       [  sin(PA)   cos(PA) ]
    cd11 = -s * np.cos(pa)
    cd12 = s * np.sin(pa)
    cd21 = s * np.sin(pa)
    cd22 = s * np.cos(pa)
    if flip:
        cd11, cd12 = -cd11, -cd12

    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cunit = ["deg", "deg"]
    w.wcs.crval = [ra_deg, dec_deg]  # sky at reference pixel
    w.wcs.crpix = [crpix1, crpix2]  # reference pixel (in SEXtractor pixel convention; origin handled later)
    w.wcs.cd = np.array([[cd11, cd12], [cd21, cd22]], dtype=float)
    return w


def read_TPV_wcs(image: str | fits.Header) -> WCS:
    """Read a FITS header with TPV WCS and return a WCS object. If malformed, inject null PV values."""
    if isinstance(image, str):
        hdr = fits.getheader(image)
    elif isinstance(image, fits.Header):
        hdr = image
    else:
        raise ValueError(f"Invalid input type: {type(image)}")

    formatted_pv = hdr.get("CTYPE1", "").endswith("TPV") or hdr.get("CTYPE2", "").endswith("TPV")
    has_pv = any(k.startswith("PV") for k in hdr.keys())

    # print(formatted_pv, has_pv)

    if formatted_pv and not has_pv:
        # inject null PV values
        for ax in (1, 2):
            hdr[f"PV{ax}_0"] = 0.0
            hdr[f"PV{ax}_1"] = 1.0
            hdr[f"PV{ax}_2"] = 0.0
    #     hdr['CTYPE1'] = 'RA---TAN'
    #     hdr['CTYPE2'] = 'DEC--TAN'
    wcs = WCS(hdr)
    return wcs


def evaluate_single_wcs(
    image: str,
    ref_cat: Table,
    source_cat: str | Table,
    date_obs: str,
    wcs: WCS,
    head: fits.Header = None,
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

    # update the source catalog with the WCS
    if head is not None:
        wcs = head or WCS(head)
        ra, dec = wcs.all_pix2world(tbl["X_IMAGE"], tbl["Y_IMAGE"], 1)
        tbl["ALPHA_J2000"] = ra
        tbl["DELTA_J2000"] = dec

    # # sort tables by SNR (higher first)
    # ref_cat["snr"] = 1 / (0.4 * np.log(10) * ref_cat["phot_g_mean_mag_error"])
    # ref_cat.sort("snr", reverse=True)
    # tbl["snr"] = 1 / (0.4 * np.log(10) * tbl["MAGERR_AUTO"])
    # tbl.sort("snr", reverse=True)

    # sort tables by magnitude (brighter first)
    ref_cat.sort("phot_g_mean_mag")
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
        create_ds9_region_file(x=x, y=y, filename=swap_ext(source_cat, "reg"))
        # create_ds9_region_file(ra=ref_cat["ra"], dec=ref_cat["dec"], filename=swap_ext(source_cat, "reg"))

    # extract highest SNR sources to calculate separation statistics
    ref_cat = ref_cat[:num_ref]
    tbl = tbl[:num_sci]
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
        radius=match_radius,
        correct_pm=True,
        obs_time=Time(date_obs),
        pm_keys=pm_keys,
    )  # right join: preserve all ref sources
    matched = add_id_column(matched)  # id needed to plot selected 9 stars
    if write_matched_catalog:
        matched.write(add_suffix(source_cat, "matched"), overwrite=True)

    unmatched_fraction = matched["separation"].mask.sum() / len(matched)

    x = matched["separation"]
    # print(f"x: {x}")
    separation_stats = {
        "min": np.ma.min(x),
        "max": np.ma.max(x),
        "rms": np.sqrt(np.ma.mean(x**2)),
        "median": np.ma.median(x),
        "std": np.ma.std(x),
    }

    if plot_save_path is not None:
        wcs_check_plot(ref_cat, tbl, matched, wcs, image, plot_save_path, fov_ra=fov_ra, fov_dec=fov_dec)
        # matched_ids = wcs_check_psf_plot(image, matched, wcs, add_suffix(plot_save_path, "psf"))

        # mask = np.isin(matched["id"], matched_ids)
        # inspected_sources = matched[mask]
        # wcs_check_scatter_plot(
        #     ref_cat[:num_plot],
        #     tbl[:num_plot],
        #     wcs,
        #     plot_save_path=add_suffix(plot_save_path, "scatter"),
        #     fov_ra=fov_ra,
        #     fov_dec=fov_dec,
        #     highlight_ra=inspected_sources["ALPHA_J2000"],
        #     highlight_dec=inspected_sources["DELTA_J2000"],
        # )

    return (REF_MAX_MAG, SCI_MAX_MAG, NUM_REF, unmatched_fraction, separation_stats)


def evaluate_joint_wcs(images_info: List["ImageInfo"]):
    separation_stats = {
        "min": np.ma.min(x),
        "max": np.ma.max(x),
        "rms": np.sqrt(np.ma.mean(x**2)),
        "median": np.ma.median(x),
        "std": np.ma.std(x),
    }
    return separation_stats


def get_fov_center(wcs: WCS, naxis1: int, naxis2: int):
    center_pix = (naxis1 - 1) / 2, (naxis2 - 1) / 2  # 0-indexed
    center_world = wcs.pixel_to_world(*center_pix)
    return center_world.ra.deg, center_world.dec.deg


def get_fov_quad(wcs, x, y, edge=True):
    if is_flipped(wcs):
        vertices = np.array([[x, y], [x, 1], [0, 1], [0, y]], dtype="float")
        if edge:
            vertices[0] += 0.5
            vertices[1, 0] += 0.5
            vertices[1, 1] -= 0.5
            vertices[2] -= 0.5
            vertices[3, 0] -= 0.5
            vertices[3, 1] += 0.5
    else:
        vertices = np.array([[0, y], [0, 0], [x, 0], [x, y]], dtype="float")
        if edge:
            vertices[0, 0] -= 0.5
            vertices[0, 1] += 0.5
            vertices[1] -= 0.5
            vertices[2, 0] += 0.5
            vertices[2, 1] -= 0.5
            vertices[3] += 0.5
    ra, dec = wcs.all_pix2world(vertices[:, 0], vertices[:, 1], 1)
    return ra, dec


def polygon_info_header(image_info: "ImageInfo") -> fits.Header:
    fov_ra = image_info.get("fov_ra", None)
    fov_dec = image_info.get("fov_dec", None)
    return make_polygon_header(image_info.wcs, image_info.naxis1, image_info.naxis2, fov_ra=fov_ra, fov_dec=fov_dec)


def make_polygon_header(
    wcs: WCS, naxis1: int, naxis2: int, fov_ra: np.array = None, fov_dec: np.array = None
) -> fits.Header:
    """Returns header with polygon info & field rotation"""

    ra, dec = get_fov_center(wcs, naxis1, naxis2)
    # FOV Center
    cards = [
        ("RACENT", round(ra, 3), "RA CENTER [deg]"),
        ("DECCENT", round(dec, 3), "DEC CENTER [deg]"),
    ]

    # FOV Polygon
    if fov_ra is None or fov_dec is None:
        fov_ra, fov_dec = get_fov_quad(wcs, naxis1, naxis2)

    for i in range(len(fov_ra)):
        cards.append((f"RAPOLY{i}", round(fov_ra[i], 3), f"RA POLYGON {i} [deg]"))
        cards.append((f"DEPOLY{i}", round(fov_dec[i], 3), f"DEC POLYGON {i} [deg]"))

    # Field Rotation
    rotation_angle_1 = field_rotation(wcs)
    cards.append(("ROTANG", rotation_angle_1, "Counterclockwise from North [deg]"))
    # updates.append(("ROTANG2", rotation_angle_2, "Rotation angle from East [deg]"))

    return fits.Header(cards)


def field_rotation(wcs: WCS) -> float:
    """
    From the WCS standard, the field rotation is measured counterclockwise from the North.
    Formally, arctan of CD2_1 / CD2_2.

    Note:
    - cd1_1: CD1_1 value from the FITS header. delta RA / delta X_IMAGE
    - cd1_2: CD1_2 value from the FITS header. delta Dec / delta X_IMAGE
    - cd2_1: CD2_1 value from the FITS header. delta RA / delta Y_IMAGE
    - cd2_2: CD2_2 value from the FITS header. delta Dec / delta Y_IMAGE


    Returns:
        - counterclockwise rotation angle from the North in degrees
    """
    ((cd1_1, cd1_2), (cd2_1, cd2_2)) = wcs.wcs.cd
    return np.degrees(np.arctan2(cd2_1, cd2_2))


def is_flipped(wcs: WCS) -> bool:
    """
    Check if the image is flipped.
    7DT raw single images are mirrored along the RA-axis (RA increases with x),
    but the stacked images are not.
    """

    ((cd1_1, cd1_2), (cd2_1, cd2_2)) = wcs.wcs.cd
    determinant = cd1_1 * cd2_2 - cd1_2 * cd2_1
    return np.sign(determinant) > 0


def _order_ccw(xy):
    """Return points ordered CCW around their centroid."""
    c = xy.mean(axis=0)
    ang = np.arctan2(xy[:, 1] - c[1], xy[:, 0] - c[0])
    return xy[np.argsort(ang)]


def inside_quad_radec(points_ra, points_dec, quad_ra, quad_dec, eps=1e-12):
    """
    Vectorized inclusion test of points in a convex quad on the sky.
    points_*: arrays of candidate points (deg)
    quad_*:   length-4 arrays/lists of quad vertices (deg), any order
    eps:      numerical tolerance in projected plane units (deg)
    Returns:  boolean array of length len(points_ra)
    """
    quad_ra = np.asarray(quad_ra, dtype=float)
    quad_dec = np.asarray(quad_dec, dtype=float)

    # 1) Choose a projection center near the quad, handling RA wrap
    ra0 = np.mean(quad_ra)
    dra = ((quad_ra - ra0 + 180) % 360) - 180
    ra0 = ra0 + np.mean(dra)
    dec0 = np.mean(quad_dec)
    center = SkyCoord(ra0 * u.deg, dec0 * u.deg, frame="icrs")
    off = SkyOffsetFrame(origin=center)

    # 2) Project vertices and points to the tangent plane
    verts = SkyCoord(quad_ra * u.deg, quad_dec * u.deg, frame="icrs").transform_to(off)
    V = np.column_stack([verts.lon.to_value(u.deg), verts.lat.to_value(u.deg)])
    V = _order_ccw(V)  # enforce CCW

    pts = SkyCoord(points_ra * u.deg, points_dec * u.deg, frame="icrs").transform_to(off)
    P = np.column_stack([pts.lon.to_value(u.deg), pts.lat.to_value(u.deg)])

    # 3) Half-space (cross-product) test
    # For CCW vertices, (E_i x (P - V_i)) >= 0 for all i means inside.
    E = np.roll(V, -1, axis=0) - V  # edges (4,2)
    # 2D "cross" z-component for all points vs each edge
    # cross((dx,dy), (ux,uy)) = dx*uy - dy*ux
    U = P[:, None, :] - V[None, :, :]  # (N,4,2)
    cross = E[None, :, 0] * U[:, :, 1] - E[None, :, 1] * U[:, :, 0]  # (N,4)

    # 4) Accept points on or inside (tolerance eps)
    return np.all(cross >= -eps, axis=1)


# precise but slower
def inside_quad_spherical(points_ra, points_dec, quad_ra, quad_dec):
    """
    points_*: array-like (deg) — can be scalars or numpy arrays
    quad_*:   4 vertices (deg), ordered around the boundary (CW or CCW)
    returns:  boolean array of shape broadcast(points_ra, points_dec)
    """
    from spherical_geometry.polygon import SphericalPolygon

    # build polygon in degrees
    poly = SphericalPolygon.from_radec(quad_ra, quad_dec, degrees=True)

    # ensure arrays and broadcast to a common shape
    ra = np.atleast_1d(points_ra)
    dec = np.atleast_1d(points_dec)
    ra_b, dec_b = np.broadcast_arrays(ra, dec)

    # evaluate point-by-point (contains_lonlat expects scalars)
    out = np.empty(ra_b.shape, dtype=bool)
    it = np.nditer(
        [ra_b, dec_b, out], flags=["multi_index", "refs_ok"], op_flags=[["readonly"], ["readonly"], ["writeonly"]]
    )
    for lon, lat, o in it:
        o[...] = bool(poly.contains_lonlat(float(lon), float(lat), degrees=True))

    # return a scalar bool if inputs were scalars
    return out if out.shape else bool(out)
