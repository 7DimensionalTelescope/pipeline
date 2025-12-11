from glob import glob
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs import FITSFixedWarning
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy import units as u
from astropy.visualization import ZScaleInterval, ImageNormalize, LinearStretch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import warnings
import seaborn as sns
from ..const import PIXSCALE
from .catalog import SkyCatalog

warnings.simplefilter("ignore", category=FITSFixedWarning)


def make_spec_colors(n: int = 40) -> list:
    """Create spectral color palette."""
    cmap = sns.color_palette("Spectral_r", as_cmap=True)
    if n == 1:
        return [cmap(0.5)]  # Return middle color when n=1
    return [cmap(i / (n - 1)) for i in range(n)]


def find_rec(N):
    """
    Find optimal grid dimensions (rows, cols) for N subplots.
    Similar to plot.py find_rec function.
    """
    num_found = False
    while not num_found:
        for k in range(int(N**0.5), 0, -1):
            if N % k == 0:
                l = N // k
                if k <= 2 * l and l <= 2 * k:
                    num_found = True
                    return k, l
        N = N + 1
    return None, None


def find_nearest_aperture(aperture_input, seeing, pixscale, verbose=True):
    """
    Find the nearest predefined aperture from aperture_dict.

    Parameters
    ----------
    aperture_input : str or float
        Aperture specification: string like "APER_1" or float (radius in pixels)
    seeing : float
        Seeing in pixels
    pixscale : float
        Pixel scale in arcsec/pixel
    verbose : bool, optional
        If True (default), print warning messages

    Returns
    -------
    tuple
        (aperture_key, aperture_radius, aperture_diameter)
    """
    from ..photometry.utils import get_aperture_dict

    aperture_dict = get_aperture_dict(seeing, pixscale)

    if isinstance(aperture_input, str):
        # String input: use directly if valid
        if aperture_input in aperture_dict:
            diameter = aperture_dict[aperture_input][0]
            return aperture_input, diameter / 2.0, diameter
        else:
            raise ValueError(f"Invalid aperture key: {aperture_input}. Valid keys: {list(aperture_dict.keys())}")

    elif isinstance(aperture_input, (int, float)):
        # Float input: find nearest aperture
        target_radius = float(aperture_input)
        target_diameter = target_radius * 2.0

        # Find nearest aperture by diameter
        best_key = None
        best_diff = float("inf")
        for key, (diameter, _) in aperture_dict.items():
            diff = abs(diameter - target_diameter)
            if diff < best_diff:
                best_diff = diff
                best_key = key

        if best_key is None:
            raise ValueError("Could not find nearest aperture")

        diameter = aperture_dict[best_key][0]
        if verbose:
            print(
                f"[Warning] The input aperture radius {aperture_input} pix is not in the aperture dictionary. Using the nearest aperture: {best_key}, Radius: {diameter / 2.0:.2f} pix"
            )
        return best_key, diameter / 2.0, diameter

    else:
        raise TypeError(f"aperture_input must be str or float, got {type(aperture_input)}")


def plot_cutouts_and_sed(
    base_dir,
    position,
    image_type="difference",
    size=30,
    position_type="sky",
    unit=u.arcsec,
    output_path=None,
    aperture_radius=5,
    use_catalog=True,
    cmap="gray",
    scale="zscale",
    figsize_per_subplot=(1.5, 1.5),
    dpi=200,
    verbose=True,
    mark_catalog_sources=False,
    catalog_type="GAIAXP",
    catalog_mag_range=(10, 20),
    query_all_catalogs=False,
):
    """
    Create combined plot with SED (magnitude only) on top and cutouts on bottom.

    Parameters
    ----------
    base_dir : str
        Base directory containing stacked images
    position : tuple or SkyCoord
        Source position
    size : float or tuple
        Size of the cutout
    position_type : str
        'sky' for RA/Dec coordinates, 'pixel' for pixel coordinates
    unit : astropy.units.Unit
        Unit for size when using sky coordinates
    output_path : str or Path, optional
        Path to save the output PNG
    aperture_radius : float or str, optional
        Aperture radius in pixels or aperture key (e.g., "APER_1") for flux extraction and visualization.
        Only used if use_catalog=False.
    use_catalog : bool, optional
        If True (default), use MAG_AUTO from catalog. If False, use the specified aperture_radius.
    cmap : str, optional
        Colormap for cutouts
    scale : str, optional
        Scaling method for cutouts
    figsize_per_subplot : tuple, optional
        Figure size per cutout subplot
    dpi : int, optional
        Resolution for saved figure
    mark_catalog_sources : bool, optional
        If True, mark catalog sources (GAIA, APASS, etc.) on cutout images
    catalog_type : str, optional
        Catalog type to use for marking sources (GAIAXP, GAIA, APASS, PS1, SDSS, SMSS)
    catalog_mag_range : tuple, optional
        Magnitude range for catalog sources to mark (min, max)
    query_all_catalogs : bool, optional
        If True, query all available catalogs and mark sources from all of them

    Returns
    -------
    None
    """
    try:
        from photutils.aperture import CircularAperture
    except ImportError:
        raise ImportError("photutils is required. Install with: pip install photutils")

    if image_type == "stacked":
        image_paths = glob(os.path.join(base_dir, "**", image_type, "*_coadd.fits"))
    elif image_type == "difference":
        image_paths = glob(os.path.join(base_dir, "**", image_type, "*_diff.fits"))
    else:
        raise ValueError(f"Invalid image type: {image_type}")

    if not image_paths:
        raise ValueError("No images found")

    # Sort images by filter
    BROADBAND_FILTERS = ["u", "g", "r", "i", "z"]
    FILTER_WAVELENGTHS = {
        "m375w": 3750,
        "m425w": 4250,
        "u": 3500,
        "g": 4750,
        "r": 6250,
        "i": 7700,
        "z": 9000,
    }
    for w in range(400, 900, 25):
        FILTER_WAVELENGTHS[f"m{w}"] = w * 10

    FILTER_WIDTHS = {
        "m375w": 250,
        "m425w": 250,
        "u": 600,
        "g": 1150,
        "r": 1150,
        "i": 1000,
        "z": 1000,
    }
    for w in range(400, 900, 25):
        FILTER_WIDTHS[f"m{w}"] = 250

    def get_filter_sort_key(image_path):
        try:
            with fits.open(image_path) as hdul:
                filter_name = hdul[0].header.get("FILTER", "").lower()
                is_broadband = filter_name in BROADBAND_FILTERS
                return (is_broadband, filter_name)
        except:
            return (True, "")

    image_paths = sorted(image_paths, key=get_filter_sort_key)
    n_images = len(image_paths)
    k, l = find_rec(n_images)

    # Extract SED data - store with image paths
    sed_data = []

    for image_path in image_paths:
        with fits.open(image_path) as hdul:
            data = hdul[0].data
            header = hdul[0].header.copy()

        wcs = None
        try:
            wcs = WCS(header)
        except Exception:
            if position_type == "sky":
                continue

        if position_type == "sky":
            if isinstance(position, SkyCoord):
                coord = position
            else:
                ra, dec = position
                coord = SkyCoord(ra=ra, dec=dec, unit="deg")
            x, y = wcs.world_to_pixel(coord)
            x, y = float(x), float(y)
        else:
            x, y = position
            x, y = float(x), float(y)

        filter_name = header.get("FILTER", "").lower()
        if filter_name in FILTER_WAVELENGTHS:
            wavelength = FILTER_WAVELENGTHS[filter_name]
            filter_width = FILTER_WIDTHS.get(filter_name, 250)
        else:
            continue

        # Get seeing and pixscale for aperture calculation
        seeing_arcsec = header.get("SEEING", None)
        if seeing_arcsec is None:
            seeing_arcsec = 3.0  # Default seeing in arcsec
        pixscale = header.get("PIXSCALE", PIXSCALE)
        seeing_pix = seeing_arcsec / pixscale

        # Store original input position for distance calculation
        x_input = x
        y_input = y

        # Check if position is in bounds
        position_in_bounds = 0 <= x < data.shape[1] and 0 <= y < data.shape[0]

        # Try to get magnitude from catalog first (even if position is out of bounds)
        cat_mag = None
        cat_mag_err = None
        cat_flux_auto = None  # For use_catalog=True, use FLUX_AUTO from catalog
        use_cat_position = False  # Flag to use catalog position instead of input position

        cat_file = image_path.replace(".fits", "_cat.fits")
        if os.path.exists(cat_file):
            try:
                from astropy.table import Table

                tbl = Table.read(cat_file)
                if "X_IMAGE" in tbl.colnames and "Y_IMAGE" in tbl.colnames:
                    # Find nearest source (even if input position is out of bounds)
                    distances = np.sqrt((tbl["X_IMAGE"] - x) ** 2 + (tbl["Y_IMAGE"] - y) ** 2)
                    nearest_idx = np.argmin(distances)
                    nearest_dist = distances[nearest_idx]

                    # If position is out of bounds, use nearest catalog source regardless of distance
                    # If position is in bounds, only use catalog source if within 5 pixels
                    if not position_in_bounds or nearest_dist < 5.0:
                        # If position is out of bounds, use catalog position for flux extraction
                        if not position_in_bounds:
                            use_cat_position = True
                            x, y = float(tbl["X_IMAGE"][nearest_idx]), float(tbl["Y_IMAGE"][nearest_idx])
                            position_in_bounds = 0 <= x < data.shape[1] and 0 <= y < data.shape[0]

                        # Determine aperture to use for catalog reading
                        if use_catalog:
                            aperture_key = "AUTO"
                            # Use calibrated MAG_AUTO_{filter} only (do not fall back to instrumental MAG_AUTO)
                            mag_col = f"MAG_AUTO_{filter_name}"
                            magerr_col = f"MAGERR_AUTO_{filter_name}"

                            if mag_col in tbl.colnames:
                                cat_mag = float(tbl[mag_col][nearest_idx])
                            if magerr_col in tbl.colnames:
                                cat_mag_err = float(tbl[magerr_col][nearest_idx])

                            # Also get FLUX_AUTO for proper magnitude calculation comparison
                            if "FLUX_AUTO" in tbl.colnames:
                                cat_flux_auto = float(tbl["FLUX_AUTO"][nearest_idx])
            except Exception as e:
                pass  # If catalog read fails, continue with our calculation

        # If no catalog source found and position is out of bounds, skip this image
        if not position_in_bounds:
            continue

        # Determine aperture to use
        if use_catalog:
            # Use MAG_AUTO from catalog
            aperture_key = "AUTO"
            actual_radius = None  # Don't show aperture for AUTO
            ZP = header.get("ZP_AUTO", None)
            EZP = header.get("EZP_AUTO", None)
            x_int, y_int = int(round(x)), int(round(y))
            flux = data[y_int, x_int]
            aperture_area = 1.0
        elif aperture_radius is not None:
            # Use specified aperture
            aperture_key, actual_radius, actual_diameter = find_nearest_aperture(
                aperture_radius, seeing_pix, pixscale, verbose=verbose
            )
            # Get corresponding ZP and EZP
            if aperture_key == "AUTO":
                zp_key = "ZP_AUTO"
                ezp_key = "EZP_AUTO"
            else:
                # APER -> 0, APER_1 -> 1, etc.
                suffix = aperture_key.replace("APER", "0").replace("0_", "")
                zp_key = f"ZP_{suffix}"
                ezp_key = f"EZP_{suffix}"

            ZP = header.get(zp_key, None)
            EZP = header.get(ezp_key, None)

            # Extract flux using the aperture
            aperture = CircularAperture((x, y), r=actual_radius)
            mask = aperture.to_mask().to_image(data.shape)
            flux = np.sum(data * mask)
            aperture_area = np.sum(mask)
        else:
            # Fallback: use AUTO if no aperture specified
            aperture_key = "AUTO"
            actual_radius = None
            ZP = header.get("ZP_AUTO", None)
            EZP = header.get("EZP_AUTO", None)
            x_int, y_int = int(round(x)), int(round(y))
            flux = data[y_int, x_int]
            aperture_area = 1.0

        UL3 = header.get("UL3_5", None)

        if ZP is None:
            continue

        # If we didn't get catalog data above, try again with the (possibly updated) position
        if cat_mag is None and os.path.exists(cat_file):
            try:
                from astropy.table import Table

                tbl = Table.read(cat_file)
                if "X_IMAGE" in tbl.colnames and "Y_IMAGE" in tbl.colnames:
                    # Find nearest source to current position
                    distances = np.sqrt((tbl["X_IMAGE"] - x) ** 2 + (tbl["Y_IMAGE"] - y) ** 2)
                    nearest_idx = np.argmin(distances)
                    if distances[nearest_idx] < 5.0:  # Within 5 pixels
                        if not use_cat_position:
                            use_cat_position = True

                        if not use_catalog:
                            # Use specified aperture magnitude from catalog
                            if aperture_key == "AUTO":
                                mag_col = (
                                    f"MAG_AUTO_{filter_name}"
                                    if f"MAG_AUTO_{filter_name}" in tbl.colnames
                                    else "MAG_AUTO"
                                )
                                magerr_col = (
                                    f"MAGERR_AUTO_{filter_name}"
                                    if f"MAGERR_AUTO_{filter_name}" in tbl.colnames
                                    else "MAGERR_AUTO"
                                )
                            else:
                                # APER -> 0, APER_1 -> 1, etc.
                                mag_col = (
                                    f"MAG_{aperture_key}_{filter_name}"
                                    if f"MAG_{aperture_key}_{filter_name}" in tbl.colnames
                                    else f"MAG_{aperture_key}"
                                )
                                magerr_col = (
                                    f"MAGERR_{aperture_key}_{filter_name}"
                                    if f"MAGERR_{aperture_key}_{filter_name}" in tbl.colnames
                                    else f"MAGERR_{aperture_key}"
                                )

                            if mag_col in tbl.colnames:
                                cat_mag = float(tbl[mag_col][nearest_idx])
                            if magerr_col in tbl.colnames:
                                cat_mag_err = float(tbl[magerr_col][nearest_idx])
            except Exception as e:
                pass  # If catalog read fails, continue with our calculation

        # Calculate magnitude
        if use_catalog and cat_flux_auto is not None and cat_flux_auto > 0:
            # When using AUTO, use FLUX_AUTO from catalog for proper comparison
            mag_calc = -2.5 * np.log10(cat_flux_auto) + ZP
        elif flux > 0:
            # Use our extracted flux
            mag_calc = -2.5 * np.log10(flux) + ZP
        else:
            mag_calc = np.nan

        if not np.isnan(mag_calc):
            # Use catalog MAGERR_AUTO if available, otherwise use EZP_AUTO
            if cat_mag_err is not None and not np.isnan(cat_mag_err):
                mag_error = cat_mag_err
            else:
                mag_error = EZP if EZP is not None else 0.0

            # Use catalog MAG_AUTO if available, otherwise use our calculation
            if cat_mag is not None and not np.isnan(cat_mag):
                mag = cat_mag
            else:
                mag = mag_calc

            is_ul = UL3 is not None and UL3 > 0 and mag > UL3
            if is_ul:
                mag = UL3
        else:
            if UL3 is not None and UL3 > 0:
                mag = UL3
            else:
                sky_sig = header.get("SKYSIG", np.std(data))
                noise = sky_sig * np.sqrt(aperture_area) if aperture_radius is not None else sky_sig
                flux_ul = 3.0 * noise
                if flux_ul > 0:
                    mag = -2.5 * np.log10(flux_ul) + ZP
                else:
                    mag = np.nan
            mag_error = EZP if EZP is not None else 0.0
            is_ul = True

        # Extract exposure and date/time information
        exptime = header.get("EXPOSURE", header.get("EXPTIME", None))
        date_obs = header.get("DATE-OBS", header.get("DATE_OBS", None))

        # Format exposure time (e.g., "300s" - just show total exposure)
        exposure_str = "N/A"
        if exptime is not None:
            try:
                exptime_float = float(exptime)
                exposure_str = f"{exptime_float:.0f}s"
            except:
                exposure_str = str(exptime)

        sed_data.append(
            {
                "image_path": image_path,
                "wavelength": wavelength,
                "magnitude": mag,
                "mag_error": mag_error,
                "filter_width": filter_width,
                "is_upper_limit": is_ul,
                "aperture_key": aperture_key,
                "aperture_radius": actual_radius,
                "filter_name": filter_name,
                "exposure": exposure_str,
                "date_obs": date_obs,
                "x": x,  # Store actual position used (may be from catalog if original was out of bounds)
                "y": y,  # Store actual position used (may be from catalog if original was out of bounds)
            }
        )

    if not sed_data:
        error_msg = (
            f"No valid flux measurements extracted from {n_images} image(s). "
            "Possible reasons: position out of bounds, missing WCS, invalid filter, or missing ZP. "
            "Check that the input position matches the image coordinate system."
        )
        raise ValueError(error_msg)

    # Sort by wavelength
    sed_data = sorted(sed_data, key=lambda x: x["wavelength"])

    wavelengths = np.array([d["wavelength"] for d in sed_data])
    magnitudes = np.array([d["magnitude"] for d in sed_data])
    mag_errors = np.array([d["mag_error"] for d in sed_data])
    filter_widths = np.array([d["filter_width"] for d in sed_data])
    is_upper_limit = np.array([d["is_upper_limit"] for d in sed_data])
    image_paths_sorted = [d["image_path"] for d in sed_data]

    # Query catalog sources if requested
    # Note: FOV is calculated from the size parameter. Sources are queried for this FOV,
    # then filtered to only show those within each cutout's actual bounds (which may vary
    # slightly due to pixel scale differences between images).
    catalog_sources_list = []

    if mark_catalog_sources and position_type == "sky":
        if isinstance(position, SkyCoord):
            coord = position
        else:
            ra, dec = position
            coord = SkyCoord(ra=ra, dec=dec, unit="deg")

        # Calculate FOV from size (will be refined with actual cutout size if available)
        if isinstance(size, (int, float)):
            fov_ra = (size * unit).to(u.deg).value
            fov_dec = (size * unit).to(u.deg).value
        else:
            fov_ra = (size[0] * unit).to(u.deg).value
            fov_dec = (size[1] * unit).to(u.deg).value

        # Determine which catalogs to query
        if query_all_catalogs:
            catalogs_to_query = ["GAIAXP", "GAIA", "APASS", "PS1", "SDSS", "SMSS"]
        else:
            catalogs_to_query = [catalog_type]

        for cat_type in catalogs_to_query:
            try:
                sky_catalog = SkyCatalog(
                    ra=coord.ra.deg,
                    dec=coord.dec.deg,
                    fov_ra=fov_ra,
                    fov_dec=fov_dec,
                    catalog_type=cat_type,
                    verbose=verbose,
                )

                ref_sources, _ = sky_catalog.get_reference_sources(
                    mag_lower=catalog_mag_range[0], mag_upper=catalog_mag_range[1]
                )

                if len(ref_sources) > 0:
                    catalog_sources_list.append({"catalog": cat_type, "sources": ref_sources})
            except Exception as e:
                pass

        # Combine and deduplicate sources from all catalogs
        # Match sources by position (within 1 arcsec tolerance)
        if len(catalog_sources_list) > 0:
            from astropy.table import vstack

            # Collect all sources with their RA/Dec
            all_sources_coords = []
            all_sources_tables = []

            for cat_info in catalog_sources_list:
                catalog_sources = cat_info["sources"]
                # Get RA/Dec column names
                ra_col = None
                dec_col = None
                for col in catalog_sources.colnames:
                    col_lower = col.lower()
                    if ra_col is None and ("ra" in col_lower and "err" not in col_lower and "e_" not in col_lower):
                        ra_col = col
                    if (
                        dec_col is None
                        and ("dec" in col_lower or "de" in col_lower)
                        and "err" not in col_lower
                        and "e_" not in col_lower
                    ):
                        dec_col = col

                if ra_col is not None and dec_col is not None:
                    coords = SkyCoord(ra=catalog_sources[ra_col], dec=catalog_sources[dec_col], unit="deg")
                    all_sources_coords.append(coords)
                    all_sources_tables.append(catalog_sources)

            if len(all_sources_coords) > 0:
                # Combine all coordinates
                # Convert to arrays and combine
                all_ra = []
                all_dec = []
                for coords in all_sources_coords:
                    all_ra.extend(coords.ra.deg)
                    all_dec.extend(coords.dec.deg)
                combined_coords = SkyCoord(ra=all_ra, dec=all_dec, unit="deg")
                combined_table = vstack(all_sources_tables)

                # Deduplicate sources within 1 arcsec tolerance
                tolerance = 1.0 * u.arcsec
                unique_mask = np.ones(len(combined_coords), dtype=bool)

                for i in range(len(combined_coords)):
                    if unique_mask[i]:
                        # Find all sources within tolerance of this source
                        sep = combined_coords[i].separation(combined_coords)
                        matches = sep < tolerance
                        matches[i] = False  # Exclude self
                        if np.any(matches):
                            # Mark duplicates as False (keep first occurrence)
                            unique_mask[matches] = False

                unique_sources = combined_table[unique_mask]

                if verbose:
                    total_sources = len(combined_table)
                    unique_count = len(unique_sources)
                    duplicate_count = total_sources - unique_count
                    print(f"Combined {total_sources} sources from {len(catalog_sources_list)} catalogs")
                    print(f"Found {unique_count} unique sources ({duplicate_count} duplicates removed)")

                # Replace catalog_sources_list with a single unified list
                catalog_sources_list = [{"catalog": "UNIFIED", "sources": unique_sources}]

    # Create spectral colors
    n_valid = len(wavelengths)
    mcolors = np.array(make_spec_colors(n_valid))

    # Recalculate grid for valid images
    # For difference images, show 6 columns (2 filters per row: target, template, difference x2)
    if image_type == "difference":
        l = 6  # Six columns: target, template, difference (repeated twice per row)
        k = (n_valid + 1) // 2  # Two filters per row (plus one for headers)
        # Add extra row for column headers
        k_with_headers = k + 1
    else:
        k, l = find_rec(n_valid)
        k_with_headers = k

    # Create figure: SED on top, cutouts on bottom
    # For difference images, account for gap column (l + 1 columns total, but gap is small)
    if image_type == "difference":
        fig_width = (l + 0.1) * figsize_per_subplot[0]  # Add small amount for gap column
    else:
        fig_width = l * figsize_per_subplot[0]
    fig = plt.figure(figsize=(fig_width, 5 + k_with_headers * figsize_per_subplot[1]), dpi=dpi)
    # Reduce spacing between SED and cutouts
    gs = fig.add_gridspec(2, 1, height_ratios=[4, k_with_headers * figsize_per_subplot[1]], hspace=0.05)

    # Top: SED plot
    ax_sed = fig.add_subplot(gs[0])
    det_mask = ~is_upper_limit
    ul_mask = is_upper_limit
    det_colors = mcolors[np.where(det_mask)[0]]
    ul_colors = mcolors[np.where(ul_mask)[0]]
    det_widths = filter_widths[det_mask]
    ul_widths = filter_widths[ul_mask]
    det_mag_errors = mag_errors[det_mask]

    for i, (wav, mag, mag_err, width, c) in enumerate(
        zip(wavelengths[det_mask], magnitudes[det_mask], det_mag_errors, det_widths, det_colors)
    ):
        if not np.isnan(mag):
            ax_sed.errorbar(
                [wav],
                [mag],
                yerr=[mag_err] if mag_err > 0 else [0.0],
                xerr=[width / 2],
                marker="o",
                markersize=8,
                capsize=2,
                color=c,
                mec="#333333",
                mew=0.3,
                label="7DT" if i == 0 else None,
                zorder=2,
            )

    for i, (wav, mag, width, c) in enumerate(zip(wavelengths[ul_mask], magnitudes[ul_mask], ul_widths, ul_colors)):
        if not np.isnan(mag):
            # For inverted y-axis: magnitude increases downward
            # Upper limits mean fainter (larger magnitude), so arrow should point downward
            # Draw arrow manually starting from mag and pointing downward
            arrow_length = 0.1  # Arrow length in magnitude units
            ax_sed.arrow(
                wav,
                mag,  # Start from the magnitude value
                0,
                arrow_length,  # Point downward (positive in data space = downward on inverted axis)
                head_width=50,
                head_length=0.05,
                fc=c,
                ec=c,
                linewidth=1.5,
                zorder=1,
            )
            # Plot the marker at the magnitude value
            ax_sed.scatter(
                [wav],
                [mag],
                marker="v",
                s=100,
                color=c,
                label="Upper Limit (3σ)" if i == 0 else None,
                zorder=2,
                edgecolors="#333333",
                linewidths=1,
            )
            # Add horizontal error bar for filter width
            ax_sed.errorbar(
                [wav],
                [mag],
                yerr=[0.0],  # No vertical error
                xerr=[width / 2],
                color=c,
                capsize=2,
                zorder=1,
                fmt="none",  # No marker, just error bars
            )

    ax_sed.set_xlabel(r"Wavelength [$\mathrm{\AA}$]", fontsize=12)
    ax_sed.set_ylabel("Magnitude", fontsize=12)
    ax_sed.set_title("SED - Magnitude", fontsize=14, fontweight="bold")
    ax_sed.grid(True, which="both", color="#666666", linestyle="--", alpha=0.5)
    ax_sed.legend(loc="best", fontsize=10)
    ax_sed.invert_yaxis()
    ax_sed.set_xlim(3600, 9150)

    # Add warning text if using coadd (stacked) images
    if image_type == "stacked":
        ax_sed.text(
            0.98,
            0.02,
            "Note: Results derived from combined images, not difference images",
            transform=ax_sed.transAxes,
            fontsize=10,
            color="red",
            fontweight="bold",
            horizontalalignment="right",
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor="red", linewidth=1.5),
        )

    # Bottom: Cutouts
    if image_type == "difference":
        # Make header row smaller - use height_ratios to reduce header height
        header_height_ratio = 0.3  # Header row is 30% of normal row height
        height_ratios = [header_height_ratio] + [1.0] * k
        # Add spacing between filters: columns 0-2 (first filter), gap, columns 3-5 (second filter)
        # Use width_ratios to add a small gap between columns 2 and 3
        width_ratios = [1.0, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0]  # Small gap (0.1) between filters
        gs_cutouts = gs[1].subgridspec(
            k_with_headers, l + 1, wspace=0.0, hspace=0.0, height_ratios=height_ratios, width_ratios=width_ratios
        )
        # Add column headers at the top (row 0) - repeat for 6 columns, skip gap column (col 3)
        header_labels = ["Target", "Template", "Difference", "Target", "Template", "Difference"]
        header_cols = [0, 1, 2, 4, 5, 6]  # Skip column 3 (gap)
        for header_idx, header_label in enumerate(header_labels):
            col_idx = header_cols[header_idx]
            header_ax = fig.add_subplot(gs_cutouts[0, col_idx])
            header_ax.axis("off")
            header_ax.text(
                0.5,
                0.2,
                header_label,
                transform=header_ax.transAxes,
                fontsize=10,
                horizontalalignment="center",
                verticalalignment="center",
            )

        # For difference images: 6 columns + 1 gap (2 filters per row, 3 images per filter)
        # Start from row 1 (row 0 is for headers)
        axes = []
        for i in range(n_valid):
            row = i // 2 + 1  # Two filters per row, start from row 1
            # First filter: cols 0-2, gap at col 3, second filter: cols 4-6
            col_offset = 0 if (i % 2 == 0) else 4
            axes.append(fig.add_subplot(gs_cutouts[row, col_offset + 0]))  # Target
            axes.append(fig.add_subplot(gs_cutouts[row, col_offset + 1]))  # Template
            axes.append(fig.add_subplot(gs_cutouts[row, col_offset + 2]))  # Difference
    else:
        gs_cutouts = gs[1].subgridspec(k, l, wspace=0.0, hspace=0.0)
        if n_valid == 1:
            axes = [fig.add_subplot(gs_cutouts[0, 0])]
        else:
            axes = [fig.add_subplot(gs_cutouts[i // l, i % l]) for i in range(n_valid)]

    # Set interval and stretch
    if scale.lower() == "zscale":
        interval = ZScaleInterval()
        stretch = LinearStretch()
    elif scale.lower() == "log":
        from astropy.visualization import LogStretch

        interval = None
        stretch = LogStretch()
    else:
        interval = ZScaleInterval()
        stretch = LinearStretch()

    def mark_catalog_sources_on_cutout(ax, cutout, cutout_data, catalog_sources_list, wcs, verbose=False):
        """Helper function to mark catalog sources on a cutout."""
        if not catalog_sources_list or len(catalog_sources_list) == 0 or wcs is None:
            return

        cutout_bbox = cutout.bbox_original
        y_start = cutout_bbox[0][0]
        x_start = cutout_bbox[1][0]

        for cat_info in catalog_sources_list:
            catalog_sources = cat_info["sources"]
            try:
                ra_col = None
                dec_col = None
                for col in catalog_sources.colnames:
                    col_lower = col.lower()
                    if ra_col is None and ("ra" in col_lower and "err" not in col_lower and "e_" not in col_lower):
                        ra_col = col
                    if (
                        dec_col is None
                        and ("dec" in col_lower or "de" in col_lower)
                        and "err" not in col_lower
                        and "e_" not in col_lower
                    ):
                        dec_col = col

                if ra_col is not None and dec_col is not None:
                    catalog_coords = SkyCoord(ra=catalog_sources[ra_col], dec=catalog_sources[dec_col], unit="deg")
                    catalog_x, catalog_y = wcs.world_to_pixel(catalog_coords)
                    catalog_x_cutout = catalog_x - x_start
                    catalog_y_cutout = catalog_y - y_start
                    mask = (
                        (catalog_x_cutout >= 0)
                        & (catalog_x_cutout < cutout_data.shape[1])
                        & (catalog_y_cutout >= 0)
                        & (catalog_y_cutout < cutout_data.shape[0])
                    )
                    if np.any(mask):
                        ax.scatter(
                            catalog_x_cutout[mask],
                            catalog_y_cutout[mask],
                            marker="+",
                            color="black",
                            s=50,
                            linewidths=1.5,
                            zorder=20,
                        )
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not mark catalog sources on cutout: {e}")

    def plot_single_cutout(
        ax,
        data,
        header,
        wcs,
        x,
        y,
        size_pixels,
        position_type,
        position,
        spec_color=None,
        catalog_sources_list=None,
        title=None,
        filter_name=None,
        FILTER_WAVELENGTHS=None,
        mark_catalog_sources=False,
    ):
        """Helper function to plot a single cutout."""
        try:
            cutout = Cutout2D(data, position=(x, y), size=size_pixels, wcs=wcs, mode="partial")
            cutout_data = cutout.data
        except Exception as e:
            ax.axis("off")
            return

        # For normalization, use only finite (non-NaN, non-inf) pixels
        valid_mask = np.isfinite(cutout_data)
        if valid_mask.sum() > 0:
            valid_data = cutout_data[valid_mask]
            if valid_data.max() > valid_data.min():
                if scale.lower() == "zscale":
                    interval_obj = ZScaleInterval()
                    vmin, vmax = interval_obj.get_limits(valid_data)
                    if vmax <= vmin:
                        vmin, vmax = valid_data.min(), valid_data.max()
                else:
                    vmin, vmax = valid_data.min(), valid_data.max()
                norm = ImageNormalize(cutout_data, vmin=vmin, vmax=vmax, stretch=stretch)
            else:
                norm = ImageNormalize(cutout_data, interval=interval, stretch=stretch)
        else:
            norm = ImageNormalize(cutout_data, interval=interval, stretch=stretch)

        ax.imshow(cutout_data, origin="lower", cmap=cmap, norm=norm)

        pos_in_cutout = cutout.position_cutout

        # Draw yellow vertical and horizontal lines at input RA/Dec position
        ax.axvline(pos_in_cutout[0], color="yellow", linewidth=0.5, linestyle="-", alpha=0.8, zorder=10)
        ax.axhline(pos_in_cutout[1], color="yellow", linewidth=0.5, linestyle="-", alpha=0.8, zorder=10)

        # Mark catalog sources (known sources) if requested
        if mark_catalog_sources:
            mark_catalog_sources_on_cutout(ax, cutout, cutout_data, catalog_sources_list, wcs, verbose)

        # Add title
        if title:
            if filter_name and filter_name in FILTER_WAVELENGTHS:
                wavelength = FILTER_WAVELENGTHS[filter_name]
                text_color = "black" if 5000 <= wavelength <= 7000 else "white"
            else:
                text_color = "white"
            ax.text(
                0.02,
                0.98,
                title,
                transform=ax.transAxes,
                fontsize=7,
                fontweight="bold",
                verticalalignment="top",
                horizontalalignment="left",
                color=text_color,
                bbox=dict(boxstyle="round,pad=0.4", facecolor=spec_color, alpha=0.9, edgecolor="white", linewidth=1.5),
            )

    # Process each image for cutouts (use sorted order to match SED)
    if image_type == "difference":
        # For difference images: plot target, template, and difference for each filter
        for i, sed_item in enumerate(sed_data):
            image_path = sed_item["image_path"]
            spec_color = mcolors[i]
            filter_name = sed_item.get("filter_name", "").lower()

            # Get position
            if "x" in sed_item and "y" in sed_item:
                x, y = sed_item["x"], sed_item["y"]
            elif position_type == "sky":
                # Will be handled when we open the file and get WCS
                x, y = None, None
            else:
                x, y = position
                x, y = float(x), float(y)

            # Calculate size_pixels
            if isinstance(size, (int, float)):
                size_pixels = (size, size)
            else:
                size_pixels = size

            with fits.open(image_path) as hdul:
                # Get WCS from primary header
                primary_header = hdul[0].header.copy()
                wcs = None
                try:
                    wcs = WCS(primary_header)
                except Exception:
                    if position_type == "sky":
                        continue

                # Get position if needed
                if x is None or y is None:
                    if position_type == "sky":
                        if isinstance(position, SkyCoord):
                            coord = position
                        else:
                            ra, dec = position
                            coord = SkyCoord(ra=ra, dec=dec, unit="deg")
                        x, y = wcs.world_to_pixel(coord)
                        x, y = float(x), float(y)

                if position_type == "sky":
                    if isinstance(size, (int, float)):
                        size_arcsec = (size * unit).to(u.arcsec).value
                    else:
                        size_arcsec = max((s * unit).to(u.arcsec).value for s in size)
                    if wcs is not None:
                        try:
                            pixscale = wcs.proj_plane_pixel_scales()[0].to(u.arcsec / u.pixel).value
                        except:
                            try:
                                pixscale = np.abs(wcs.pixel_scale_matrix[0, 0]) * 3600
                            except:
                                pixscale = 0.505
                    else:
                        pixscale = 0.505
                    if isinstance(size, (int, float)):
                        size_pixels = (size_arcsec / pixscale, size_arcsec / pixscale)
                    else:
                        size_pixels = (
                            (size[0] * unit).to(u.arcsec).value / pixscale,
                            (size[1] * unit).to(u.arcsec).value / pixscale,
                        )

                # Read TARGET, TEMPLATE, and DIFFIM from header keywords (they point to file paths)
                target_data = None
                template_data = None
                diffim_data = None
                target_header = None
                template_header = None
                diffim_header = None

                # Get file paths from header keywords
                target_path = primary_header.get("TARGET", None)
                template_path = primary_header.get("TEMPLATE", None)
                diffim_path = primary_header.get("DIFFIM", image_path)  # Fallback to current file

                # Read target image
                if target_path and os.path.exists(target_path):
                    try:
                        with fits.open(target_path) as target_hdul:
                            target_data = target_hdul[0].data
                            target_header = target_hdul[0].header.copy()
                    except Exception as e:
                        if verbose:
                            print(f"Warning: Could not read TARGET image {target_path}: {e}")

                # Read template image
                if template_path and os.path.exists(template_path):
                    try:
                        with fits.open(template_path) as template_hdul:
                            template_data = template_hdul[0].data
                            template_header = template_hdul[0].header.copy()
                    except Exception as e:
                        if verbose:
                            print(f"Warning: Could not read TEMPLATE image {template_path}: {e}")

                # Read difference image (use current file)
                try:
                    diffim_data = hdul[0].data
                    diffim_header = primary_header.copy()
                except Exception as e:
                    if verbose:
                        print(f"Warning: Could not read DIFFIM image: {e}")

                # Plot each image
                # Note: row starts from 1 (row 0 is for headers)
                # With 2 filters per row, calculate row as i // 2 + 1
                row = i // 2 + 1
                image_types = [
                    ("TARGET", target_data, target_header, "Target"),
                    ("TEMPLATE", template_data, template_header, "Template"),
                    ("DIFFIM", diffim_data, diffim_header, "Difference"),
                ]

                for col_idx, (ext_name, img_data, img_header, label) in enumerate(image_types):
                    if img_data is None:
                        continue
                    # Use i (filter index) for axes indexing, not row (which includes header offset)
                    ax = axes[i * 3 + col_idx]

                    # Use image's own header
                    header_to_use = img_header if img_header else primary_header
                    # Get WCS from this image's header
                    wcs_to_use = None
                    try:
                        wcs_to_use = WCS(header_to_use)
                    except Exception:
                        wcs_to_use = wcs  # Fallback to original WCS

                    # Show filter and unit in title
                    title_parts = []
                    if primary_header.get("FILTER"):
                        title_parts.append(primary_header.get("FILTER"))
                    if primary_header.get("TELESCOP"):
                        title_parts.append(primary_header.get("TELESCOP"))
                    title = " | ".join(title_parts) if title_parts else ""
                    plot_single_cutout(
                        ax,
                        img_data,
                        header_to_use,
                        wcs_to_use,
                        x,
                        y,
                        size_pixels,
                        position_type,
                        position,
                        spec_color=spec_color,
                        catalog_sources_list=catalog_sources_list,
                        title=title,
                        filter_name=filter_name,
                        FILTER_WAVELENGTHS=FILTER_WAVELENGTHS,
                        mark_catalog_sources=mark_catalog_sources,
                    )

                    # Set axis labels
                    # Note: row starts from 1 (row 0 is for headers), so last row is k
                    if row == k:
                        ax.set_xlabel("X [pixels]", size=5, labelpad=2)
                        ax.tick_params(axis="x", which="both", bottom=True, top=False, labelsize=5, pad=2)
                    else:
                        ax.set_xlabel("")
                        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

                    # Show y-axis label for first column of each filter (Target column, col_idx == 0)
                    if col_idx == 0:
                        ax.set_ylabel("Y [pixels]", size=5, labelpad=2)
                        ax.tick_params(axis="y", which="both", left=True, right=False, labelsize=5, pad=2)
                    else:
                        ax.set_ylabel("")
                        ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

                    ax.tick_params(
                        axis="both",
                        which="both",
                        direction="in",
                        length=2,
                        width=0.5,
                        grid_color="#BBBBBB",
                        grid_alpha=0.5,
                        grid_linewidth=0.5,
                        grid_linestyle="--",
                    )
                    ax.grid(True)
    else:
        # For stacked images: original behavior
        for i, sed_item in enumerate(sed_data):
            if i >= len(axes):
                break
            ax = axes[i]
            image_path = sed_item["image_path"]

            with fits.open(image_path) as hdul:
                data = hdul[0].data
                header = hdul[0].header.copy()

            wcs = None
            try:
                wcs = WCS(header)
            except Exception:
                if position_type == "sky":
                    ax.axis("off")
                    continue

            # Use stored position from sed_data (may be from catalog if original was out of bounds)
            if "x" in sed_item and "y" in sed_item:
                x, y = sed_item["x"], sed_item["y"]
            elif position_type == "sky":
                if isinstance(position, SkyCoord):
                    coord = position
                else:
                    ra, dec = position
                    coord = SkyCoord(ra=ra, dec=dec, unit="deg")
                x, y = wcs.world_to_pixel(coord)
                x, y = float(x), float(y)
            else:
                x, y = position
                x, y = float(x), float(y)

            if isinstance(size, (int, float)):
                size_pixels = (size, size)
            else:
                size_pixels = size

            if position_type == "sky":
                if isinstance(size, (int, float)):
                    size_arcsec = (size * unit).to(u.arcsec).value
                else:
                    size_arcsec = max((s * unit).to(u.arcsec).value for s in size)

                if wcs is not None:
                    try:
                        pixscale = wcs.proj_plane_pixel_scales()[0].to(u.arcsec / u.pixel).value
                    except:
                        try:
                            pixscale = np.abs(wcs.pixel_scale_matrix[0, 0]) * 3600
                        except:
                            pixscale = 0.505
                else:
                    pixscale = 0.505

                if isinstance(size, (int, float)):
                    size_pixels = (size_arcsec / pixscale, size_arcsec / pixscale)
                else:
                    size_pixels = (
                        (size[0] * unit).to(u.arcsec).value / pixscale,
                        (size[1] * unit).to(u.arcsec).value / pixscale,
                    )

            try:
                cutout = Cutout2D(data, position=(x, y), size=size_pixels, wcs=wcs, mode="partial")
                cutout_data = cutout.data
            except Exception as e:
                ax.axis("off")
                continue

            # For normalization, use only finite (non-NaN, non-inf) pixels
            # This helps when source is near edge and cutout has NaN padding
            valid_mask = np.isfinite(cutout_data)
            if valid_mask.sum() > 0:
                valid_data = cutout_data[valid_mask]
                if valid_data.max() > valid_data.min():
                    # Use only valid pixels for normalization
                    if scale.lower() == "zscale":
                        interval = ZScaleInterval()
                        # Apply interval to valid data only
                        vmin, vmax = interval.get_limits(valid_data)
                        # Ensure we have a reasonable range
                        if vmax <= vmin:
                            vmin, vmax = valid_data.min(), valid_data.max()
                    else:
                        vmin, vmax = valid_data.min(), valid_data.max()

                    norm = ImageNormalize(cutout_data, vmin=vmin, vmax=vmax, stretch=stretch)
                else:
                    # All valid pixels have same value, use default
                    norm = ImageNormalize(cutout_data, interval=interval, stretch=stretch)
            else:
                # No valid pixels, use default normalization
                norm = ImageNormalize(cutout_data, interval=interval, stretch=stretch)

            im = ax.imshow(cutout_data, origin="lower", cmap=cmap, norm=norm)

            # Get spectral color (matches SED plot)
            spec_color = mcolors[i]

            pos_in_cutout = cutout.position_cutout

            # Draw yellow vertical and horizontal lines at input RA/Dec position
            ax.axvline(pos_in_cutout[0], color="yellow", linewidth=0.5, linestyle="-", alpha=0.8, zorder=10)
            ax.axhline(pos_in_cutout[1], color="yellow", linewidth=0.5, linestyle="-", alpha=0.8, zorder=10)

            # Mark catalog sources (known sources) if requested
            if mark_catalog_sources:
                mark_catalog_sources_on_cutout(ax, cutout, cutout_data, catalog_sources_list, wcs, verbose)

            title_parts = []
            if header.get("FILTER"):
                title_parts.append(header["FILTER"])
            if header.get("TELESCOP"):
                title_parts.append(header["TELESCOP"])
            title = " | ".join(title_parts) if title_parts else f"Cutout {i+1}"

            # Determine text color based on wavelength (black for 5250-7000 Å range)
            filter_name = header.get("FILTER", "").lower()
            if filter_name in FILTER_WAVELENGTHS:
                wavelength = FILTER_WAVELENGTHS[filter_name]
                text_color = "black" if 5000 <= wavelength <= 7000 else "white"
            else:
                text_color = "white"

            ax.text(
                0.02,
                0.98,
                title,
                transform=ax.transAxes,
                fontsize=7,
                fontweight="bold",
                verticalalignment="top",
                horizontalalignment="left",
                color=text_color,
                bbox=dict(boxstyle="round,pad=0.4", facecolor=spec_color, alpha=0.9, edgecolor="white", linewidth=1.5),
            )

            row = i // l
            col = i % l

            if row == k - 1 and i < n_valid:
                ax.set_xlabel("X [pixels]", size=5, labelpad=2)
                ax.tick_params(axis="x", which="both", bottom=True, top=False, labelsize=5, pad=2)
            else:
                ax.set_xlabel("")
                ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

            if col == 0:
                ax.set_ylabel("Y [pixels]", size=5, labelpad=2)
                ax.tick_params(axis="y", which="both", left=True, right=False, labelsize=5, pad=2)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

            ax.tick_params(
                axis="both",
                which="both",
                direction="in",
                length=2,
                width=0.5,
                grid_color="#BBBBBB",
                grid_alpha=0.5,
                grid_linewidth=0.5,
                grid_linestyle="--",
            )
            ax.grid(True)

    if output_path is None:
        output_path = os.path.join(base_dir, "combined_sed_cutouts.png")
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    if verbose:
        print(f"Combined SED and cutouts plot saved to: {output_path}")

    # Save magnitude data to text file
    output_path = Path(output_path)
    txt_output_path = output_path.parent / (output_path.stem + "_output.txt")

    from .table import write_sed_table_file, sort_sed_data

    write_sed_table_file(sed_data, txt_output_path, verbose=verbose)

    return sort_sed_data(sed_data)


def file_list(base_dir):
    file_list = glob(os.path.join(base_dir, "**", "stacked", "*_coadd.fits"))
    catalog_file_list = glob(os.path.join(base_dir, "**", "stacked", "*_coadd_cat.fits"))

    return file_list + catalog_file_list


def make_too_output(too_id, sky_position=None, image_type="difference", verbose=False, **kwargs):
    from ..services.database.too import TooDB

    too_db = TooDB()
    too_data = too_db.read_too_data_by_id(too_id)
    if too_data is None:
        raise ValueError(f"Too data not found for ID: {too_id}")

    if sky_position is None:
        sky_position = SkyCoord(too_data["ra"], too_data["dec"], unit=("hourangle", "deg"))

    sed_data = plot_cutouts_and_sed(
        too_data["base_path"],
        position=sky_position,
        image_type=image_type,
        mark_catalog_sources=True,
        query_all_catalogs=True,
        catalog_type="GAIAXP",
        catalog_mag_range=(10, 20),
        verbose=verbose,
        **kwargs,
    )

    files = file_list(too_data["base_path"])
    too_db.update_too_data(too_id, file_list=files)

    return sed_data
