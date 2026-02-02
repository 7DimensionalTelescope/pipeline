from glob import glob

from astropy.wcs import FITSFixedWarning
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D

from astropy.visualization import ImageNormalize, LinearStretch, LogStretch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import warnings
import seaborn as sns
from .catalog import query_catalogs, _get_ra_dec_columns
from .utils import (
    get_diff_image_set,
    extract_mag_from_catalog,
    extract_flux_from_aperture,
    get_coord_in_pixel,
    get_image_info,
)

from matplotlib.patches import Circle

warnings.simplefilter("ignore", category=FITSFixedWarning)


def make_spec_colors(n: int = 40) -> list:
    """Create spectral color palette."""
    cmap = sns.color_palette("Spectral_r", as_cmap=True)
    if n == 1:
        return [cmap(0.5)]  # Return middle color when n=1
    return [cmap(i / (n - 1)) for i in range(n)]


def find_rec(N, image_type):
    """
    Find optimal grid dimensions (rows, cols) for N subplots.
    Similar to plot.py find_rec function.
    """
    if image_type == "difference":
        l = 6  # Six columns: target, template, difference (repeated twice per row)
        k = (N + 1) // 2  # Two filters per row (plus one for headers)
        # Add extra row for column headers
        k_with_headers = k + 1
        return k, l, k_with_headers
    else:
        num_found = False
        while not num_found:
            for k in range(int(N**0.5), 0, -1):
                if N % k == 0:
                    l = N // k
                    if k <= 2 * l and l <= 2 * k:
                        num_found = True
                        return k, l, k
            N = N + 1
    return None, None


def plot_single_cutout(
    ax,
    data,
    header,
    sky_position,
    aperture_size,
    spec_color=None,
    title=None,
):
    """Helper function to plot a single cutout."""
    try:
        x, y, wcs = get_coord_in_pixel(header, sky_position, return_wcs=True)
        cutout = Cutout2D(data, position=(x, y), size=30, wcs=wcs, mode="partial")
        cutout_data = cutout.data
    except Exception as e:
        ax.axis("off")
        return

    # For normalization, use only finite (non-NaN, non-inf) pixels
    valid_mask = np.isfinite(cutout_data)
    if valid_mask.sum() > 0:
        valid_data = cutout_data[valid_mask]
        if valid_data.max() > valid_data.min():
            vmin, vmax = valid_data.min(), valid_data.max()
            norm = ImageNormalize(cutout_data, vmin=vmin, vmax=vmax, stretch=LinearStretch())

    ax.imshow(cutout_data, origin="lower", cmap="gray", norm=norm)

    pos_in_cutout = cutout.position_cutout

    # Draw yellow vertical and horizontal lines at input RA/Dec position
    ax.axvline(pos_in_cutout[0], color="yellow", linewidth=0.5, linestyle="-", alpha=0.8, zorder=10)
    ax.axhline(pos_in_cutout[1], color="yellow", linewidth=0.5, linestyle="-", alpha=0.8, zorder=10)

    if aperture_size > 0:
        ax.add_patch(
            Circle(
                (pos_in_cutout[0], pos_in_cutout[1]),
                aperture_size,
                color="yellow",
                fill=False,
                linewidth=0.5,
                alpha=0.8,
                zorder=10,
            )
        )
    # Mark catalog sources (known sources) if requested
    # if mark_catalog_sources:
    #     mark_catalog_sources_on_cutout(ax, cutout, cutout_data, catalog_sources_list, wcs)

    # # Add title
    if title:
        text_color = "black"
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


def plot_cutouts_and_sed(
    base_dir,
    image_type="difference",
    sky_position=None,
    output_path=None,
    aperture_key="1",
    cmap="gray",
    scale="zscale",
    figsize_per_subplot=(1.5, 1.5),
    dpi=200,
    mark_catalog_sources=False,
    **kwargs,
):
    """
    Create combined plot with SED (magnitude only) on top and cutouts on bottom.
    -------
    None
    """

    if image_type == "coadd":
        path = f"{base_dir}/**/{image_type}/*_coadd.fits"
    elif image_type == "difference":
        path = f"{base_dir}/**/{image_type}/*_diff.fits"
    else:
        raise ValueError(f"Invalid image type: {image_type}")

    if verbose:
        print("finding coadd images in", path)
    image_paths = glob(path)

    if not image_paths:
        raise ValueError("No images found")
    else:
        if verbose:
            print(f"{len(image_paths)} images are found.")

    def get_filter_sort_key(image_path):
        try:
            with fits.open(image_path) as hdul:
                filter_name = hdul[0].header.get("FILTER", "").lower()
                is_broadband = filter_name in BROAD_FILTERS
                return (is_broadband, filter_name)
        except:
            return (True, "")

    image_paths = sorted(image_paths, key=get_filter_sort_key)
    n_images = len(image_paths)
    k, l = find_rec(n_images)

    # Extract SED data - store with image paths
    sed_data = []

    for image_path in image_paths:
        print(image_path)
        wavelength, filter_width, filter_name, is_broadband, units, exposure, date_obs = get_image_info(image_path)

        mag, mag_err = extract_mag_from_catalog(image_path, sky_position)
        aperture_size = 0

        if mag is None:
            mag, mag_err, aperture_size = extract_flux_from_aperture(
                image_path, sky_position, aperture_key=aperture_key
            )
        else:
            print(extract_flux_from_aperture(image_path, sky_position, aperture_key=aperture_key))
        print(mag, mag_err)

        sed_data.append(
            {
                "magnitude": mag,
                "mag_error": mag_err,
                "wavelength": wavelength,
                "filter_width": filter_width,
                "is_upper_limit": mag is None,
                "filter_name": filter_name,
                "units": units,
                "aperture_size": aperture_size,
                "image_path": image_path,
                "exposure": exposure,
                "date_obs": date_obs,
            }
        )

    if not sed_data:
        error_msg = (
            f"No valid flux measurements extracted from {len(image_paths)} image(s). "
            "Possible reasons: position out of bounds, missing WCS, invalid filter, or missing ZP. "
            "Check that the input position matches the image coordinate system."
        )
        raise ValueError(error_msg)

    if mark_catalog_sources:
        catalog_sources_list = query_catalogs(sky_position, **kwargs)

    # Sort by wavelength
    sed_data = sorted(sed_data, key=lambda x: x["wavelength"])
    wavelengths = np.array([d["wavelength"] for d in sed_data])
    magnitudes = np.array([d["magnitude"] for d in sed_data])
    mag_errors = np.array([d["mag_error"] for d in sed_data])
    filter_widths = np.array([d["filter_width"] for d in sed_data])
    is_upper_limit = np.array([d["is_upper_limit"] for d in sed_data])

    # Create spectral colors
    n_valid = len(wavelengths)
    mcolors = np.array(make_spec_colors(n_valid))

    k, l, k_with_headers = find_rec(n_valid, image_type)

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
        if mag is not None:
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

    for i, (wav, mag, width, c) in enumerate(zip(wavelengths[ul_mask], mag_errors[ul_mask], ul_widths, ul_colors)):
        if mag is not None:
            ax_sed.arrow(
                wav,
                mag,  # Start from the magnitude value
                0,
                0.1,  # Point downward (positive in data space = downward on inverted axis)
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
                label="Upper Limit (5σ)" if i == 0 else None,
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

    # Add warning text if using coadd images
    if image_type == "coadd":
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
                ra_col, dec_col = _get_ra_dec_columns(catalog_sources)
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

    # # Process each image for cutouts (use sorted order to match SED)
    if image_type == "difference":
        # For difference images: plot target, template, and difference for each filter
        for i, sed_item in enumerate(sed_data):
            image_path = sed_item["image_path"]
            spec_color = mcolors[i]
            filter_name = sed_item.get("filter_name").lower()
            units = sed_item.get("units")
            images = get_diff_image_set(image_path)

            # Note: row starts from 1 (row 0 is for headers)
            # With 2 filters per row, calculate row as i // 2 + 1
            row = i // 2 + 1

            for col_idx, imtype in enumerate(["target", "template", "diffim"]):
                # Use i (filter index) for axes indexing, not row (which includes header offset)
                ax = axes[i * 3 + col_idx]
                # Get WCS from this image's header
                title = f"{filter_name} | {units}"

                plot_single_cutout(
                    ax,
                    images[imtype],
                    images[imtype + "_header"],
                    sky_position,
                    aperture_size=sed_item.get("aperture_size"),
                    spec_color=spec_color,
                    title=title,
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
    # else:
    #     # For stacked images: original behavior
    #     for i, sed_item in enumerate(sed_data):
    #         if i >= len(axes):
    #             break
    #         ax = axes[i]
    #         image_path = sed_item["image_path"]

    #         with fits.open(image_path) as hdul:
    #             data = hdul[0].data
    #             header = hdul[0].header.copy()

    #         wcs = None
    #         try:
    #             wcs = WCS(header)
    #         except Exception:
    #             if position_type == "sky":
    #                 ax.axis("off")
    #                 continue

    #         # Use stored position from sed_data (may be from catalog if original was out of bounds)
    #         # This ensures we use the same position that was used for flux extraction
    #         if "x" in sed_item and "y" in sed_item:
    #             x, y = sed_item["x"], sed_item["y"]
    #         elif position_type == "sky":
    #             # Fallback: recalculate from original position (should not happen if image was processed)
    #             if isinstance(position, SkyCoord):
    #                 coord = position
    #             else:
    #                 ra, dec = position
    #                 coord = SkyCoord(ra=ra, dec=dec, unit="deg")
    #             x, y = wcs.world_to_pixel(coord)
    #             x, y = float(x), float(y)
    #         else:
    #             # Fallback: use original position (should not happen if image was processed)
    #             x, y = position
    #             x, y = float(x), float(y)

    #         if isinstance(size, (int, float)):
    #             size_pixels = (size, size)
    #         else:
    #             size_pixels = size

    #         if position_type == "sky":
    #             if isinstance(size, (int, float)):
    #                 size_arcsec = (size * unit).to(u.arcsec).value
    #             else:
    #                 size_arcsec = max((s * unit).to(u.arcsec).value for s in size)

    #             if wcs is not None:
    #                 try:
    #                     pixscale = wcs.proj_plane_pixel_scales()[0].to(u.arcsec / u.pixel).value
    #                 except:
    #                     try:
    #                         pixscale = np.abs(wcs.pixel_scale_matrix[0, 0]) * 3600
    #                     except:
    #                         pixscale = 0.505
    #             else:
    #                 pixscale = 0.505

    #             if isinstance(size, (int, float)):
    #                 size_pixels = (size_arcsec / pixscale, size_arcsec / pixscale)
    #             else:
    #                 size_pixels = (
    #                     (size[0] * unit).to(u.arcsec).value / pixscale,
    #                     (size[1] * unit).to(u.arcsec).value / pixscale,
    #                 )

    #         try:
    #             cutout = Cutout2D(data, position=(x, y), size=size_pixels, wcs=wcs, mode="partial")
    #             cutout_data = cutout.data
    #         except Exception as e:
    #             ax.axis("off")
    #             continue

    #         # For normalization, use only finite (non-NaN, non-inf) pixels
    #         # This helps when source is near edge and cutout has NaN padding
    #         valid_mask = np.isfinite(cutout_data)
    #         if valid_mask.sum() > 0:
    #             valid_data = cutout_data[valid_mask]
    #             if valid_data.max() > valid_data.min():
    #                 # Use only valid pixels for normalization
    #                 if scale.lower() == "zscale":
    #                     interval = ZScaleInterval()
    #                     # Apply interval to valid data only
    #                     vmin, vmax = interval.get_limits(valid_data)
    #                     # Ensure we have a reasonable range
    #                     if vmax <= vmin:
    #                         vmin, vmax = valid_data.min(), valid_data.max()
    #                 else:
    #                     vmin, vmax = valid_data.min(), valid_data.max()

    #                 norm = ImageNormalize(cutout_data, vmin=vmin, vmax=vmax, stretch=stretch)
    #             else:
    #                 # All valid pixels have same value, use default
    #                 norm = ImageNormalize(cutout_data, interval=interval, stretch=stretch)
    #         else:
    #             # No valid pixels, use default normalization
    #             norm = ImageNormalize(cutout_data, interval=interval, stretch=stretch)

    #         im = ax.imshow(cutout_data, origin="lower", cmap=cmap, norm=norm)

    #         # Get spectral color (matches SED plot)
    #         spec_color = mcolors[i]

    #         pos_in_cutout = cutout.position_cutout

    #         # Draw yellow vertical and horizontal lines at input RA/Dec position
    #         ax.axvline(pos_in_cutout[0], color="yellow", linewidth=0.5, linestyle="-", alpha=0.8, zorder=10)
    #         ax.axhline(pos_in_cutout[1], color="yellow", linewidth=0.5, linestyle="-", alpha=0.8, zorder=10)

    #         # Mark catalog sources (known sources) if requested
    #         if mark_catalog_sources:
    #             mark_catalog_sources_on_cutout(ax, cutout, cutout_data, catalog_sources_list, wcs, verbose)

    #         title_parts = []
    #         if header.get("FILTER"):
    #             title_parts.append(header["FILTER"])
    #         if header.get("TELESCOP"):
    #             title_parts.append(header["TELESCOP"])
    #         title = " | ".join(title_parts) if title_parts else f"Cutout {i+1}"

    #         # Determine text color based on wavelength (black for 5250-7000 Å range)
    #         filter_name = header.get("FILTER", "").lower()
    #         if filter_name in FILTER_WAVELENGTHS:
    #             wavelength = FILTER_WAVELENGTHS[filter_name]
    #             text_color = "black" if 5000 <= wavelength <= 7000 else "white"
    #         else:
    #             text_color = "white"

    #         ax.text(
    #             0.02,
    #             0.98,
    #             title,
    #             transform=ax.transAxes,
    #             fontsize=7,
    #             fontweight="bold",
    #             verticalalignment="top",
    #             horizontalalignment="left",
    #             color=text_color,
    #             bbox=dict(boxstyle="round,pad=0.4", facecolor=spec_color, alpha=0.9, edgecolor="white", linewidth=1.5),
    #         )

    #         row = i // l
    #         col = i % l

    #         if row == k - 1 and i < n_valid:
    #             ax.set_xlabel("X [pixels]", size=5, labelpad=2)
    #             ax.tick_params(axis="x", which="both", bottom=True, top=False, labelsize=5, pad=2)
    #         else:
    #             ax.set_xlabel("")
    #             ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

    #         if col == 0:
    #             ax.set_ylabel("Y [pixels]", size=5, labelpad=2)
    #             ax.tick_params(axis="y", which="both", left=True, right=False, labelsize=5, pad=2)
    #         else:
    #             ax.set_ylabel("")
    #             ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

    #         ax.tick_params(
    #             axis="both",
    #             which="both",
    #             direction="in",
    #             length=2,
    #             width=0.5,
    #             grid_color="#BBBBBB",
    #             grid_alpha=0.5,
    #             grid_linewidth=0.5,
    #             grid_linestyle="--",
    #         )
    #         ax.grid(True)

    if output_path is None:
        output_path = os.path.join(base_dir, "combined_sed_cutouts.png")
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    # Save magnitude data to text file
    output_path = Path(output_path)
    txt_output_path = output_path.parent / (output_path.stem + "_output.txt")

    from .table import write_sed_table_file, sort_sed_data

    write_sed_table_file(sed_data, txt_output_path)

    return sort_sed_data(sed_data)


def file_list(base_dir):
    file_list = glob(os.path.join(base_dir, "**", "coadd", "*_coadd.fits"))
    catalog_file_list = glob(os.path.join(base_dir, "**", "coadd", "*_coadd_cat.fits"))

    return file_list + catalog_file_list


def make_too_output(too_id, sky_position=None, image_type="difference", verbose=False, **kwargs):
    from ..services.database.too import TooDB

    too_db = TooDB()
    too_data = too_db.read_data_by_id(too_id)

    files = file_list(too_data["base_path"])
    too_db.update_too_data(too_id, file_list=files)

    if too_data is None:
        raise ValueError(f"Too data not found for ID: {too_id}")

    if sky_position is None:
        sky_position = SkyCoord(too_data["ra_deg"], too_data["dec_deg"], unit="deg")
    else:
        sky_position = SkyCoord(sky_position[0], sky_position[1], unit="deg")

    if image_type == "difference":

        sed_data = plot_cutouts_and_sed(
            too_data["base_path"],
            sky_position=sky_position,
            image_type=image_type,
            mark_catalog_sources=True,
            **kwargs,
        )

        if sed_data is not None:
            return sed_data

    sed_data = plot_cutouts_and_sed(
        too_data["base_path"],
        sky_position=sky_position,
        image_type=image_type,
        mark_catalog_sources=True,
        **kwargs,
    )
    return sed_data
