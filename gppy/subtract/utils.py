import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from ..tools.table import filter_table


def create_ds9_region_file(ra_array, dec_array, radius=10, filename="ds9_regions.reg"):
    """
    Create a DS9 region file containing circular regions centered at given RA and Dec coordinates.

    Parameters
    ----------
    ra_array : array-like
        Array of right ascension (RA) values in degrees.
    dec_array : array-like
        Array of declination (Dec) values in degrees.
    radius : float
        Radius of each circular region in arcseconds.
    filename : str, optional
        Name of the DS9 region file to be created (default is 'ds9_regions.reg').

    Returns
    -------
    None
        Writes a DS9 region file to disk
    """

    # header for DS9 region file
    header = 'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\nfk5'

    with open(filename, "w") as file:
        file.write(header + "\n")

        # Add circles for each RA, Dec pair
        for ra, dec in zip(ra_array, dec_array):
            region_line = f'circle({ra},{dec},{radius}")\n'
            file.write(region_line)
    # print(f"DS9 region file '{filename}' has been created.")


def select_sources(
    table: Table,
    aperture_suffix: str = "AUTO",
    snr_min: float = 10,
    class_star_min: float = 0.2,
    flags_max: int = 0,
) -> Table:
    """
    Select high-quality sources from a photometric table based on SNR, CLASS_STAR, and FLAGS.

    Parameters
    ----------
    table : Table
        Astropy Table containing photometric measurements and metadata.
    aperture_suffix : str, optional
        Suffix used to identify SNR column (e.g., "AUTO", "APER_1"). Default is "AUTO".
    snr_min : float, optional
        Minimum signal-to-noise ratio. Default is 10.
    class_star_min : float, optional
        Minimum stellarity index (CLASS_STAR). Default is 0.2.
    flags_max : int, optional
        Maximum allowed FLAGS value. Default is 0.

    Returns
    -------
    Table
        Filtered table containing only sources meeting the criteria.
    """

    # get the name of the SNR column
    if hasattr(table, "meta") and "FILTER" in table.meta:
        filt = table.meta["FILTER"]
        snr_key = f"SNR_{aperture_suffix}_{filt}"
    else:
        snr_key = [s for s in table.columns if "SNR" in s and aperture_suffix in s][0]

    # selected_indices = np.where(
    #     (table[snr_key] > snr_min)
    #     & (table[f"CLASS_STAR"] > class_star_min)
    #     & (table["FLAGS"] <= flags_max)
    # )
    # return table[selected_indices]

    # conditions = [
    #     (snr_key, ">", snr_min),
    #     ("CLASS_STAR", ">", class_star_min),
    #     ("FLAGS", "<=", flags_max),
    # ]
    conditions = [
        snr_key, ">", snr_min,
        "CLASS_STAR", ">", class_star_min,
        "FLAGS", "<=", flags_max,
    ]  # fmt: skip
    return filter_table(table, conditions)
