import unicodedata
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


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


def polygon_info(input_header: fits.Header) -> fits.Header:
    """Add polygon info to header - field rotation"""

    # FOV Center
    x, y = input_header["NAXIS1"], input_header["NAXIS2"]
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
