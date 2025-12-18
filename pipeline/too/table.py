"""
SED table formatting utilities for ToO output.

This module provides functions to format SED (Spectral Energy Distribution) data
as tables for both text files and email content.
"""
import numpy as np

BROADBAND_FILTERS = ["u", "g", "r", "i", "z"]


def sort_sed_data(sed_data):
    """
    Sort SED data: broadband filters first, then medium filters by wavelength.

    Parameters
    ----------
    sed_data : list of dict
        List of SED data dictionaries, each containing filter_name, wavelength, etc.

    Returns
    -------
    list of dict
        Sorted SED data
    """
    def get_txt_sort_key(item):
        filter_name = item.get("filter_name", "")
        is_broadband = filter_name in BROADBAND_FILTERS
        wavelength = item.get("wavelength", 0)
        return (not is_broadband, wavelength)  # False (broadband) sorts before True (medium)

    return sorted(sed_data, key=get_txt_sort_key)


def format_magnitude(mag, mag_err, is_upper_limit):
    """
    Format magnitude and error as strings.

    Parameters
    ----------
    mag : float or None
        Magnitude value
    mag_err : float or None
        Magnitude error
    is_upper_limit : bool
        Whether this is an upper limit

    Returns
    -------
    tuple
        (mag_str, mag_err_str) formatted strings
    """
    # Format magnitude
    if mag is not None and not np.isnan(mag):
        if is_upper_limit:
            mag_str = f">{mag:.2f}"
        else:
            mag_str = f"{mag:.2f}"
    else:
        mag_str = "N/A"

    # Format error
    if is_upper_limit:
        mag_err_str = "N/A"
    elif mag_err is not None and not np.isnan(mag_err) and mag_err > 0:
        mag_err_str = f"{mag_err:.2f}"
    else:
        mag_err_str = "0.00"

    return mag_str, mag_err_str


def format_sed_table_string(sed_data, title="Magnitude Measurements"):
    """
    Format SED data as a table string for email or display.

    Parameters
    ----------
    sed_data : list of dict
        List of SED data dictionaries
    title : str, optional
        Title for the table (default: "Magnitude Measurements")

    Returns
    -------
    str
        Formatted table as string
    """
    if not sed_data:
        return ""

    sed_data_sorted = sort_sed_data(sed_data)

    # Build table
    table = f"\n{title}:\n"
    table += "-" * 100 + "\n"
    table += f"{'Filter':<12} {'Mag':<12} {'Mag_err':<12} {'Exposure':<15} {'Date Time':<30}\n"
    table += "-" * 100 + "\n"

    ul_exists = False
    for item in sed_data_sorted:
        filter_name = item.get("filter_name", "N/A")
        mag = item.get("magnitude")
        mag_err = item.get("mag_error")
        exposure = item.get("exposure", "N/A")
        date_obs = item.get("date_obs", "N/A")
        if date_obs is None:
            date_obs = "N/A"
        is_ul = item.get("is_upper_limit", False)

        mag_str, mag_err_str = format_magnitude(mag, mag_err, is_ul)
        if is_ul:
            ul_exists = True

        table += f"{filter_name:<12} {mag_str:<12} {mag_err_str:<12} {exposure:<15} {date_obs:<30}\n"

    table += "-" * 100 + "\n"
    if ul_exists:
        table += "* Upper limits (marked with '>') are 3-sigma limits\n"

    return table


def write_sed_table_file(sed_data, output_path, verbose=True):
    """
    Write SED data to a text file.

    Parameters
    ----------
    sed_data : list of dict
        List of SED data dictionaries
    output_path : str or Path
        Path to output text file
    verbose : bool, optional
        If True, print confirmation message (default: True)

    Returns
    -------
    None
    """
    from pathlib import Path

    if not sed_data:
        return

    sed_data_sorted = sort_sed_data(sed_data)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        # Write header
        f.write(f"{'Filter':<12} {'Mag':<12} {'Mag_err':<12} {'Exposure':<15} {'Date Time':<30}\n")
        f.write("-" * 100 + "\n")
        # Write data rows
        ul_exists = False
        for item in sed_data_sorted:
            filter_name = item["filter_name"]
            mag = item["magnitude"]
            mag_err = item["mag_error"]
            exposure = item["exposure"]
            date_obs = item["date_obs"] if item["date_obs"] else "N/A"
            is_ul = item.get("is_upper_limit", False)

            mag_str, mag_err_str = format_magnitude(mag, mag_err, is_ul)
            if is_ul:
                ul_exists = True

            f.write(f"{filter_name:<12} {mag_str:<12} {mag_err_str:<12} {exposure:<15} {date_obs:<30}\n")
        f.write("-" * 100 + "\n")
        if ul_exists:
            f.write("* Upper limits (marked with '>') are 3-sigma limits\n")

    if verbose:
        print(f"Magnitude data saved to: {output_path}")

