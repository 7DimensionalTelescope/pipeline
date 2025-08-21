import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from ..tools.table import filter_table


# def create_ds9_region_file(ra_array, dec_array, radius=10, filename="ds9_regions.reg"):
#     """
#     Create a DS9 region file containing circular regions centered at given RA and Dec coordinates.

#     Parameters
#     ----------
#     ra_array : array-like
#         Array of right ascension (RA) values in degrees.
#     dec_array : array-like
#         Array of declination (Dec) values in degrees.
#     radius : float
#         Radius of each circular region in arcseconds.
#     filename : str, optional
#         Name of the DS9 region file to be created (default is 'ds9_regions.reg').

#     Returns
#     -------
#     None
#         Writes a DS9 region file to disk
#     """

#     # header for DS9 region file
#     header = 'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\nfk5'

#     with open(filename, "w") as file:
#         file.write(header + "\n")

#         # Add circles for each RA, Dec pair
#         for ra, dec in zip(ra_array, dec_array):
#             region_line = f'circle({ra},{dec},{radius}")\n'
#             file.write(region_line)
#     # print(f"DS9 region file '{filename}' has been created.")


def create_ds9_region_file(
    ra=None,
    dec=None,
    x=None,
    y=None,
    radius=10,
    filename="ds9_regions.reg",
    color="green",
    shape="circle",
):
    """
    Create a DS9 region file containing regions centered at either RA/Dec (FK5) or image X/Y coordinates.

    Parameters
    ----------
    ra : array-like, optional
        Right ascension values in degrees (FK5). Must be paired with `dec`.
    dec : array-like, optional
        Declination values in degrees (FK5). Must be paired with `ra`.
    x : array-like, optional
        X image coordinates in pixels. Must be paired with `y`.
    y : array-like, optional
        Y image coordinates in pixels. Must be paired with `x`.
    radius : float, default 10
        Region radius. If using RA/Dec (FK5), interpreted as **arcseconds**.
        If using image X/Y, interpreted as **pixels**.
    filename : str, default "ds9_regions.reg"
        Output DS9 region file name.
    color : str, default "green"
        DS9 color for regions (e.g., 'green', 'red', 'yellow').
    shape : str, default "circle"
        Region shape. Currently supports only 'circle' (others could be added).

    Returns
    -------
    None
        Writes a DS9 region file to disk.

    Notes
    -----
    - Exactly one coordinate mode must be provided: either (ra_array & dec_array) or (x_array & y_array).
    - Output uses:
        * 'fk5' coordinate system with radius in arcseconds (e.g., 10")
        * 'image' coordinate system with radius in pixels (e.g., 10p)
    """
    # Determine which coordinate set is provided
    using_fk5 = (ra is not None) or (dec is not None)
    using_image = (x is not None) or (y is not None)

    if using_fk5 and using_image:
        raise ValueError("Provide either RA/Dec (FK5) OR X/Y (image) coordinates, not both.")

    if using_fk5:
        if ra is None or dec is None:
            raise ValueError("Both ra and dec must be provided for FK5 mode.")
        if len(ra) != len(dec):
            raise ValueError("ra and dec must have the same length.")
        coord_system_line = "fk5"
        # DS9 expects arcseconds with a double-quote suffix for fk5
        radius_str = f'{radius}"'
        coords_iter = zip(ra, dec)
    elif using_image:
        if x is None or y is None:
            raise ValueError("Both x and y must be provided for image mode.")
        if len(x) != len(y):
            raise ValueError("x and y must have the same length.")
        coord_system_line = "image"
        # DS9 pixels use 'p' suffix (explicit & unambiguous)
        radius_str = f"{radius}p"
        coords_iter = zip(x, y)
    else:
        raise ValueError("You must provide either RA/Dec or X/Y arrays.")

    if shape.lower() != "circle":
        raise NotImplementedError("Only 'circle' shape is currently supported.")

    # DS9 global header (you can tweak defaults via parameters if desired)
    header = (
        f'global color={color} dashlist=8 3 width=1 font="helvetica 10 normal roman" '
        "select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1"
    )

    with open(filename, "w") as f:
        f.write(header + "\n")
        f.write(coord_system_line + "\n")

        for x, y in coords_iter:
            # DS9 wants decimals; no extra spaces
            f.write(f"{shape}({x},{y},{radius_str})\n")


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
