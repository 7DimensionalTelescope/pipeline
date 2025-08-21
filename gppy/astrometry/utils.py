import unicodedata
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def read_scamp_header(file):
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
    return hdr


def build_wcs(ra_deg, dec_deg, crpix1, crpix2, pixscale_arcsec, pa_deg, flip=False):
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
