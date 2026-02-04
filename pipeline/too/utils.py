from astropy.io import fits
from photutils.aperture import CircularAperture, aperture_photometry, CircularAnnulus
from astropy.wcs import WCS
from astropy.table import Table
import numpy as np
import os
from ..const import FILTER_WAVELENGTHS, FILTER_WIDTHS, BROAD_FILTERS


def get_diff_image_set(image_path):

    primary_header = fits.getheader(image_path)
    target_path = primary_header.get("TARGET", None)
    template_path = primary_header.get("TEMPLATE", None)
    diffim_path = primary_header.get("DIFFIM", image_path)  # Fallback to current file

    output = {}
    with fits.open(target_path) as hdul:
        output["target"] = hdul[0].data
        output["target_header"] = hdul[0].header.copy()
    with fits.open(template_path) as hdul:
        output["template"] = hdul[0].data
        output["template_header"] = hdul[0].header.copy()
    with fits.open(diffim_path) as hdul:
        output["diffim_header"] = hdul[0].header.copy()
        output["diffim"] = hdul[0].data

    return output


def get_coord_in_pixel(header, sky_position, return_wcs=False):
    wcs = WCS(header)
    x, y = wcs.world_to_pixel(sky_position)
    if return_wcs:
        return x, y, wcs
    else:
        return x, y


def extract_mag_from_catalog(image_path, sky_position):
    header = fits.getheader(image_path)
    filter_name = header.get("FILTER", "").lower()
    x, y = get_coord_in_pixel(header, sky_position)
    cat_file = image_path.replace(".fits", "_cat.fits")
    if os.path.exists(cat_file):
        tbl = Table.read(cat_file)
        distances = np.sqrt((tbl["X_IMAGE"] - x) ** 2 + (tbl["Y_IMAGE"] - y) ** 2)
        nearest_idx = np.argmin(distances)
        nearest_dist = distances[nearest_idx]
        if nearest_dist > 4.0:
            return None, None
    else:
        return None, None

    mag_auto = tbl[f"MAG_AUTO_{filter_name}"][nearest_idx]
    mag_auto_err = tbl[f"MAGERR_AUTO_{filter_name}"][nearest_idx]
    return mag_auto, mag_auto_err


def extract_flux_from_aperture(image_path, sky_position, aperture_key="4"):
    with fits.open(image_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    # 1. Get Metadata
    ZP = float(header.get(f"ZP_{aperture_key}"))
    EZP = float(header.get(f"EZP_{aperture_key}"))
    UL5 = float(header.get(f"UL5_{aperture_key}"))

    aper_header_key = "APER" if aperture_key == "0" else f"APER_{aperture_key}"
    aperture_diameter = header.get(aper_header_key)

    if aperture_diameter is None:
        raise ValueError(f"Aperture key {aper_header_key} not found in header.")

    # 2. Define Apertures
    x, y = get_coord_in_pixel(header, sky_position)
    r = aperture_diameter / 2.0
    aper = CircularAperture((x, y), r=r)

    # Define annulus for background (standard practice is ~5-7x radius)
    annulus_aperture = CircularAnnulus((x, y), r_in=r * 5.0, r_out=r * 6.0)

    # 3. Extract Flux
    # Use photutils built-in method for speed and precision (handles sub-pixel overlaps)
    total_flux_in_aperture, _ = aper.do_photometry(data)
    total_flux_in_aperture = total_flux_in_aperture[0]

    # 4. Accurate Background Subtraction
    annulus_mask = annulus_aperture.to_mask(method="center")
    annulus_data = annulus_mask.get_values(data)  # Extracts only pixels within annulus
    sky_median = np.median(annulus_data)

    # Subtract sky contribution (sky per pixel * number of pixels in aperture)
    actual_flux = total_flux_in_aperture - (sky_median * aper.area)

    # 5. Calculate Magnitude
    if actual_flux <= 0:
        return None, UL5, 0
    else:
        mag = -2.5 * np.log10(actual_flux) + ZP
        # If EZP is a global instrument error, we return it;
        # otherwise, you might want to calculate Poisson error here.
        return mag, EZP, r


def get_image_info(image_path):
    with fits.open(image_path) as hdul:
        header = hdul[0].header
        filter_name = header.get("FILTER", "").lower()
        units = header.get("TELESCOP")
        exposure = header.get("EXPOSURE")
        date_obs = header.get("DATE-OBS")
        is_broadband = filter_name in BROAD_FILTERS
        if filter_name in FILTER_WAVELENGTHS:
            wavelength = FILTER_WAVELENGTHS[filter_name]
            filter_width = FILTER_WIDTHS.get(filter_name, 250)
            return wavelength, filter_width, filter_name, is_broadband, units, exposure, date_obs
        else:
            raise ValueError(
                f"Filter {filter_name} not found in FILTER_WAVELENGTHS. Update the FILTER_WAVELENGTHS dictionary."
            )
