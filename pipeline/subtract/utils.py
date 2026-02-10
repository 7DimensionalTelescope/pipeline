import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from ..tools.table import filter_table


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
