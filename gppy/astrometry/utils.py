import unicodedata
from typing import List
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord, SkyOffsetFrame

from ..tools.table import match_two_catalogs
from ..utils import add_suffix


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


def evaluate_wcs(
    ref_cat: Table,
    source_cat: str,
    date_obs: str,
    head: fits.Header = None,
    match_radius=10,
    write_matched_catalog=True,
    plot=True,
    plot_save_path=None,
):
    """tbl: catalog of detected sources"""

    # load the source catalog
    tbl = Table(fits.getdata(source_cat, ext=2))

    # update the source catalog with the WCS
    if head is not None:
        wcs = head or WCS(head)
        ra, dec = wcs.all_pix2world(tbl["X_IMAGE"], tbl["Y_IMAGE"], 1)
        tbl["ALPHA_J2000"] = ra
        tbl["DELTA_J2000"] = dec

    if plot:
        wcs_check_plot(ref_cat, tbl, plot_save_path)

    # iteratively try decreasing SNR thresholds to find NUM_MIN_REFCAT sources
    NUM_MIN_REFCAT = 20
    refcat_snr_thresh_list = [5000, 3000, 1000, 300, 100, 50]
    for refcat_snr_thresh in refcat_snr_thresh_list:
        snr = 1 / (0.4 * np.log(10) * ref_cat["phot_g_mean_mag_error"])
        sel = snr > refcat_snr_thresh
        # print(np.sum(sel))
        if np.sum(sel) >= NUM_MIN_REFCAT:
            break
    ref_cat = ref_cat[sel]
    num_ref_sources = len(ref_cat)

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
    )
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

    return (refcat_snr_thresh, num_ref_sources, unmatched_fraction, separation_stats)


def wcs_check_plot(refcat: Table, tbl: Table, plot_save_path=None):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, dpi=100, figsize=(13, 7))

    for ax, sci_mag_cut, ref_mag_cut in zip(axes, [14, 16], [14, 16]):
        refcat_sel = refcat[refcat["phot_g_mean_mag"] < ref_mag_cut]
        # print("ref stars:", len(refcat))

        # tbl = tbl[tbl["FLUX_AUTO"] / tbl["FLUXERR_AUTO"] > 100]
        tbl = tbl[tbl["MAG_AUTO"] + 26 < sci_mag_cut]
        # print("image stars:", len(tbl))

        ra, dec = tbl["ALPHA_J2000"], tbl["DELTA_J2000"]

        ax.scatter(ra, dec, alpha=0.5, label="scamp wcs", s=10)
        ax.scatter(refcat_sel["ra"], refcat_sel["dec"], alpha=0.5, label="refcat", s=20)
        ax.legend()
        ax.set_title(f"ref mag < {ref_mag_cut}, sci mag < {sci_mag_cut}")
        ax.set_aspect("equal")

    if plot_save_path is not None:
        plt.savefig(plot_save_path)
    else:
        plt.show()


def polygon_info(input_header: fits.Header, naxis1: int = None, naxis2: int = None) -> fits.Header:
    """
    Returns header with polygon info & field rotation
    If not given NAXIS1, NAXIS2, it assumes CRPIX1, CRPIX2 is the center
    """

    # FOV Center
    x = naxis1 or (input_header.get("NAXIS1", None) or input_header["CRPIX1"])
    y = naxis2 or (input_header.get("NAXIS2", None) or input_header["CRPIX2"])
    center_pix = (x - 1) / 2, (y - 1) / 2  # 0-indexed
    wcs = WCS(input_header)
    center_world = wcs.pixel_to_world(*center_pix)
    cards = [
        ("RACENT", round(center_world.ra.deg, 3), "RA CENTER [deg]"),
        ("DECCENT", round(center_world.dec.deg, 3), "DEC CENTER [deg]"),
    ]

    # FOV Polygon
    if is_flipped(wcs):
        vertices = np.array([[x - 1, y - 1], [x - 1, 0], [0, 0], [0, y - 1]])
    else:
        vertices = np.array([[0, y - 1], [0, 0], [x - 1, 0], [x - 1, y - 1]])
    for i in range(4):
        coord = wcs.pixel_to_world(vertices[i, 0], vertices[i, 1])
        ra, dec = coord.ra.deg, coord.dec.deg
        cards.append((f"RAPOLY{i}", round(ra, 3), f"RA POLYGON {i} [deg]"))
        cards.append((f"DEPOLY{i}", round(dec, 3), f"DEC POLYGON {i} [deg]"))

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


def inside_quad_radec(points_ra, points_dec, quad_ra, quad_dec):
    """
    points_*: arrays of candidate points (deg)
    quad_*:   length-4 arrays of the quad vertices (deg), ordered either CW or CCW
    returns:  boolean array (len = len(points_ra)) of inside/outside
    """
    # 1) Tangent-plane projection using an offset frame about the quad centroid
    ra0 = np.mean(quad_ra)
    # handle RA wraparound for the centroid:
    d_ra = ((quad_ra - ra0 + 180) % 360) - 180
    ra0 += np.mean(d_ra)
    dec0 = np.mean(quad_dec)
    center = SkyCoord(ra0 * u.deg, dec0 * u.deg, frame="icrs")

    off = SkyOffsetFrame(origin=center)
    # Project vertices
    verts_icrs = SkyCoord(quad_ra * u.deg, quad_dec * u.deg, frame="icrs").transform_to(off)
    vx = verts_icrs.lon.to_value(u.deg)
    vy = verts_icrs.lat.to_value(u.deg)

    # Project points
    pts_icrs = SkyCoord(points_ra * u.deg, points_dec * u.deg, frame="icrs").transform_to(off)
    x = pts_icrs.lon.to_value(u.deg)
    y = pts_icrs.lat.to_value(u.deg)

    # 2) Half-space test for a convex polygon (works for CW or CCW consistently)
    # Build edge vectors and their outward normals
    V = np.stack([vx, vy], axis=1)  # (4,2)
    E = np.roll(V, -1, axis=0) - V  # edges (4,2)
    # normals that keep "inside" on one side: rotate edges by +90°
    N = np.stack([-E[:, 1], E[:, 0]], axis=1)  # (4,2)

    # For consistent inequality direction, determine orientation once using the polygon center
    poly_cent = V.mean(axis=0)
    orient = np.sign(np.cross(E, (np.roll(V, -1, axis=0) + V) / 2 - poly_cent).sum()) or 1.0
    # Evaluate (p - Vi)·Ni ; inside if all have same sign (≥0) after fixing orientation
    PX = np.stack([x, y], axis=1)  # (N,2)
    diff = PX[:, None, :] - V[None, :, :]  # (N,4,2)
    s = (diff * N[None, :, :]).sum(axis=2)  # (N,4)
    if orient < 0:
        s = -s
    return np.all(s >= 0, axis=1)
