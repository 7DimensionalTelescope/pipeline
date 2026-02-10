from astropy.io import fits
from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats
from astropy.wcs import WCS
from astropy.table import Table
import numpy as np
import os
from glob import glob
from ..const import FILTER_WAVELENGTHS, FILTER_WIDTHS, BROAD_FILTERS


def file_list(base_dir):
    file_list = glob(os.path.join(base_dir, "**", "coadd", "*_coadd.fits"))
    catalog_file_list = glob(os.path.join(base_dir, "**", "coadd", "*_coadd_cat.fits"))

    return file_list + catalog_file_list


def get_filter_sort_key(image_path):
    try:
        with fits.open(image_path) as hdul:
            filter_name = hdul[0].header.get("FILTER", "").lower()
            is_broadband = filter_name in BROAD_FILTERS
            return (is_broadband, filter_name)
    except:
        return (True, "")


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
    return (x, y, wcs) if return_wcs else (x, y)


def extract_mag_from_catalog(image_path, sky_position, aperture_key="auto"):
    header = fits.getheader(image_path)
    filter_name = header.get("FILTER", "").lower()
    x, y = get_coord_in_pixel(header, sky_position)
    cat_file = image_path.replace(".fits", "_cat.fits")
    if not os.path.exists(cat_file):
        return None, None
    tbl = Table.read(cat_file)
    distances = np.sqrt((tbl["X_IMAGE"] - x) ** 2 + (tbl["Y_IMAGE"] - y) ** 2)
    nearest_idx = np.argmin(distances)
    if distances[nearest_idx] > 4.0:
        return None, None

    if aperture_key == "auto":
        mag, mag_err = tbl[f"MAG_AUTO_{filter_name}"][nearest_idx], tbl[f"MAGERR_AUTO_{filter_name}"][nearest_idx]
    elif aperture_key == "0":
        mag, mag_err = (
            tbl[f"MAG_APER_{filter_name}"][nearest_idx],
            tbl[f"MAGERR_APER_{filter_name}"][nearest_idx],
        )
    else:
        mag, mag_err = (
            tbl[f"MAG_APER_{aperture_key}_{filter_name}"][nearest_idx],
            tbl[f"MAGERR_APER_{aperture_key}_{filter_name}"][nearest_idx],
        )

    return mag, mag_err


def extract_flux_from_aperture(
    image_path, sky_position, aperture_key="0", annulus_factor_in=5.0, annulus_factor_out=6.0
):
    with fits.open(image_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    if aperture_key == "auto":
        print("AUTO aperture is not defined in the header. Using APER_2 (3*seeing) instead.")
        aperture_key = "2"

    if aperture_key == "0":
        aper_header_key = "APER"
    else:
        aper_header_key = f"APER_{aperture_key}"

    # 1. Get Metadata
    ZP = float(header.get(f"ZP_{aperture_key}", 0.0))
    EZP = float(header.get(f"EZP_{aperture_key}", 0.0))
    UL5 = float(header.get(f"UL5_{aperture_key}", 0.0))
    EGAIN = float(header.get("EGAIN", 1.0))  # Default to 1.0 if missing

    aperture_diameter = header.get(aper_header_key)

    if aperture_diameter is None:
        raise ValueError(f"Aperture key {aper_header_key} not found in header.")

    # 2. Define Apertures
    x, y = get_coord_in_pixel(header, sky_position)
    r = float(aperture_diameter) / 2.0
    aper = CircularAperture((x, y), r=r)
    annulus_aperture = CircularAnnulus((x, y), r_in=r * annulus_factor_in, r_out=r * annulus_factor_out)

    # 3. Extract Flux & Background Stats (Modern Way)
    # ApertureStats calculates sum, median, std, etc. efficiently
    aper_stats = ApertureStats(data, aper)
    annulus_stats = ApertureStats(data, annulus_aperture)

    total_flux_in_aperture = aper_stats.sum
    sky_median = annulus_stats.median
    sky_std = annulus_stats.std
    n_sky_pixels = annulus_stats.sum_aper_area.value  # Precise area of annulus

    # 4. Subtract Background
    actual_flux = total_flux_in_aperture - (sky_median * aper.area)

    # 5. Calculate Error (CCD Equation)
    # Term 1: Source + Sky Poisson Noise (Use abs to avoid sqrt of negative)
    # Note: total_flux_in_aperture is the raw sum (Source+Sky), so it dominates Poisson noise.
    term1 = abs(total_flux_in_aperture) * EGAIN

    # Term 2: Background Noise inside the aperture
    # (sky_std in ADU * EGAIN) gives electrons. Squared gives variance.
    term2 = aper.area * (sky_std * EGAIN) ** 2

    # Term 3: Uncertainty in the Mean Background determination
    term3 = (aper.area**2 * (sky_std * EGAIN) ** 2) / n_sky_pixels

    variance_electrons = term1 + term2 + term3
    sigma_flux_electrons = np.sqrt(variance_electrons)

    # Convert noise back to ADU
    sigma_flux_adu = sigma_flux_electrons / EGAIN  # FIXED: Changed GAIN to EGAIN

    # 6. Calculate Magnitude
    # Calculate SNR
    SNR = actual_flux / sigma_flux_adu if sigma_flux_adu > 0 else 0
    SNR_THRESHOLD = 3.0  # Standard threshold for detection significance

    if actual_flux <= 0 or SNR < SNR_THRESHOLD:
        return None, UL5, 0
    else:
        mag = -2.5 * np.log10(actual_flux) + ZP

        # Inst Magnitude Error
        mag_err_inst = 2.5 / np.log(10) * (sigma_flux_adu / actual_flux)

        # Quadrature Sum with ZP Error
        total_mag_err = np.sqrt(mag_err_inst**2 + EZP**2)

        return mag, total_mag_err, r


def get_image_info(image_path):
    with fits.open(image_path) as hdul:
        header = hdul[0].header
        filter_name = header.get("FILTER", "").lower()
        if filter_name not in FILTER_WAVELENGTHS:
            raise ValueError(
                f"Filter {filter_name} not found in FILTER_WAVELENGTHS. Update the FILTER_WAVELENGTHS dictionary."
            )
        wavelength = FILTER_WAVELENGTHS[filter_name]
        filter_width = FILTER_WIDTHS.get(filter_name, 250)
        return (
            wavelength,
            filter_width,
            filter_name,
            filter_name in BROAD_FILTERS,
            header.get("TELESCOP"),
            header.get("EXPOSURE"),
            header.get("DATE-OBS"),
        )


def get_sed_data(image_path, sky_position, aperture_key="auto"):
    """
    Build a single SED data dict for one image: magnitude, errors, filter info.
    Uses catalog magnitude if available, otherwise aperture flux.
    """
    wavelength, filter_width, filter_name, is_broadband, units, exposure, date_obs = get_image_info(image_path)
    cat_mag, cat_mag_err = extract_mag_from_catalog(image_path, sky_position, aperture_key=aperture_key)

    aper_mag, aper_mag_err, aperture_size = extract_flux_from_aperture(
        image_path, sky_position, aperture_key=aperture_key
    )

    # if cat_mag is not None:
    #     print(
    #         os.path.basename(image_path),
    #         f"{cat_mag:.2f}+/-{cat_mag_err:.2f} versus {aper_mag:.2f}+/-{aper_mag_err:.2f}",
    #     )

    return {
        "magnitude": cat_mag if cat_mag is not None else aper_mag,
        "mag_error": cat_mag_err if cat_mag is not None else aper_mag_err,
        "wavelength": wavelength,
        "filter_width": filter_width,
        "is_upper_limit": aper_mag is None,
        "filter_name": filter_name,
        "units": units,
        "aperture_size": aperture_size if cat_mag is None else 0,
        "image_path": image_path,
        "exposure": exposure,
        "date_obs": date_obs,
        "cat_mag": cat_mag,
        "cat_mag_err": cat_mag_err,
        "aper_mag": aper_mag,
        "aper_mag_err": aper_mag_err,
    }
