from __future__ import annotations
import os
import tempfile
import numpy as np
from contextlib import contextmanager
from functools import lru_cache
from typing import TYPE_CHECKING
from numba import njit
from astropy.table import Table, hstack, vstack, unique
from astropy.coordinates import SkyCoord

from ..const import GAIA_REF_DIR, REF_DIR
from ..path.path import PathHandler

if TYPE_CHECKING:
    from ..config._sciproc_stubs import PhotometryNode


@njit
def rss(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Calculate Root Sum Square of two arrays.

    Args:
        a: First input array
        b: Second input array

    Returns:
        Root sum square of inputs: sqrt(a^2 + b^2)
    """
    return np.sqrt(np.nan_to_num(a) ** 2.0 + np.nan_to_num(b) ** 2.0)


@njit
def is_within_ellipse(x: np.ndarray, y: np.ndarray, center_x: float, center_y: float, a: float, b: float) -> np.ndarray:
    """
    Check if points lie within an ellipse.

    Args:
        x, y: Arrays of point coordinates
        center_x, center_y: Ellipse center coordinates
        a, b: Semi-major and semi-minor axes

    Returns:
        Boolean array indicating points inside ellipse
    """
    term1 = ((x - center_x) ** 2) / (a**2)
    term2 = ((y - center_y) ** 2) / (b**2)
    return term1 + term2 <= 1


@njit
def compute_median_nmad(values: np.ndarray, normalize: bool = True) -> tuple:
    """
    Computes the median and Median Absolute Deviation (MAD) of a given array.

    The MAD is a robust measure of the variability of a univariate sample of quantitative data.
    If `normalize` is set to True, the MAD is scaled by a constant (1.4826) to make it
    consistent with the standard deviation under the assumption of normality.

    Args:
        values (np.ndarray): Input array of numerical values.
        normalize (bool, optional): If True, normalize the MAD to be consistent with the
            standard deviation under a normal distribution. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - median (float): The median of the input array.
            - mad (float): The (optionally normalized) Median Absolute Deviation.
    """
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if normalize:
        return median, 1.4826 * mad
    return median, mad


@njit
def compute_median_rms(values: np.ndarray) -> tuple:
    """
    Compute median and Root Mean Square Error (RMS).

    Args:
        values: Input array

    Returns:
        Tuple of (median, RMS)
    """
    median = np.median(values)
    rms = np.sqrt(np.sum((values - median) ** 2) / (len(values) - 1))
    # rms /= np.sqrt(len(values))
    return median, rms


@njit
def limitmag(n_sigma: np.ndarray, zp: float, aper: float, skysigma: float, noise_factor: float = 1.0) -> np.ndarray:
    """
    Calculate limiting magnitude.

    Args:
        N: Signal-to-noise ratio array
        zp: Zero point
        aper: Aperture diameter
        skysigma: Sky background sigma
        noise_factor: aperture noise / (skysigma * sqrt(pi R^2)); 1.0 assumes independent pixels

    Returns:
        Array of limiting magnitudes
    """
    R = aper / 2.0
    braket = n_sigma * skysigma * np.sqrt(np.pi * R**2) * noise_factor
    upperlimit = zp - 2.5 * np.log10(braket)
    return np.round(upperlimit, 3)


def aperture_weight_squared(aperture: float, phases: int = 16) -> float:
    """Sum of the squared exact-aperture pixel weights, averaged over sub-pixel phase.

    An exact aperture sums w_i x_i with fractional weights on the rim, so for INDEPENDENT pixels its
    variance is sigma^2 sum w_i^2, not sigma^2 sum w_i = sigma^2 pi R^2. The two differ by 3% at a
    20-pixel diameter and 11% at 6."""
    from photutils.aperture import CircularAperture

    r = 0.5 * float(aperture)
    span = int(np.ceil(r)) + 2
    shape = (2 * span + 1, 2 * span + 1)
    offsets = np.linspace(0.0, 1.0, phases, endpoint=False)
    total = 0.0
    for dx in offsets:
        for dy in offsets:
            weights = CircularAperture([(span + dx, span + dy)], r=r).to_mask(method="exact")[0].to_image(shape)
            total += float(np.sum(weights**2))
    return total / phases**2


def aperture_noise_factor(acf, aperture: float) -> float:
    """Aperture noise over the independent-pixel value sigma*sqrt(pi R^2), from a measured autocorrelation.

    Var(sum w_i x_i) = sigma^2 sum_h rho(h) O(h), with O the area two copies of the aperture share at
    lag h -- analytic for a circle, and sum w_i^2 at zero lag. Returns 1.0 for a missing autocorrelation
    so the caller keeps the historical white-noise definition."""
    if acf is None:
        return 1.0
    acf = np.asarray(acf, dtype=float)
    half = acf.shape[0] // 2
    radius = 0.5 * float(aperture)
    ly, lx = np.mgrid[-half : half + 1, -half : half + 1]
    separation = np.hypot(lx, ly)
    overlap = np.zeros_like(separation)
    inside = separation < 2 * radius
    t = np.clip(separation[inside] / (2 * radius), 0.0, 1.0)
    overlap[inside] = 2 * radius**2 * np.arccos(t) - 0.5 * separation[inside] * np.sqrt(
        np.maximum(4 * radius**2 - separation[inside] ** 2, 0.0)
    )
    overlap[half, half] = aperture_weight_squared(aperture)
    variance = float(np.sum(acf * overlap))
    if not np.isfinite(variance) or variance <= 0:
        return 1.0
    return float(np.sqrt(variance / (np.pi * radius**2)))


def bin_noise_factor(acf, box: int) -> float:
    """Noise of a box x box binned pixel over the independent-pixel value sqrt(box^2)*sigma.

    For a top-hat block O(h) = (box-|hx|)(box-|hy|), so at box = 2 this is exactly
    sqrt(1 + rho(1,0) + rho(0,1) + rho(1,1)) -- the number an IFU-style 2x2 rebin needs."""
    if acf is None:
        return 1.0
    acf = np.asarray(acf, dtype=float)
    half = acf.shape[0] // 2
    box = int(box)
    lags = np.arange(-(box - 1), box)
    if lags.max() > half:
        return float("nan")
    hy, hx = np.meshgrid(lags, lags, indexing="ij")
    overlap = (box - np.abs(hx)) * (box - np.abs(hy))
    window = acf[np.ix_(lags + half, lags + half)]
    return float(np.sqrt(np.sum(window * overlap) / box**2))


@njit
def apply_zp(mag: np.ndarray, mag_err: np.ndarray, zp: float, zperr: float) -> tuple[np.ndarray]:
    """
    Apply zero point correction to magnitudes.

    Args:
        mag: Magnitude array
        mag_err: Magnitude error array
        zp: Zero point value
        zperr: Zero point error

    Returns:
        Tuple of (corrected_mag, corrected_err, flux, flux_err, SNR)
    """
    mag = mag + zp
    # Provide zp error separately. Don't add zperr
    # mag_err = np.sqrt(mag_err**2 + zperr**2)
    flux = 10 ** ((23.9 - mag) / 2.5)  # uJy
    flux_err = 0.4 * np.log(10) * flux * mag_err
    snr = flux / flux_err
    return mag, mag_err, flux, flux_err, snr


def keyset(mag_key: str, filter: str) -> tuple[str]:
    """
    bundle mag, magerr, flux, fluxerr, snr keys for a given filter.

    Args:
        mag_key: Base magnitude key
        filter: Filter name

    Returns:
        Tuple of keys for magnitude, error, flux, flux error, and SNR
    """
    _magkey = f"{mag_key}_{filter}"
    _magerrkey = _magkey.replace("MAG", "MAGERR")
    _fluxkey = _magkey.replace("MAG", "FLUX")
    _fluxerrkey = _magerrkey.replace("MAG", "FLUX")
    _snrkey = _magkey.replace("MAG", "SNR")
    return _magkey, _magerrkey, _fluxkey, _fluxerrkey, _snrkey


def aggregate_gaia_catalogs(target_coord, path_calibration_field: str = None, query_radius=1.0):
    """
    Return a merged Gaia DR3 source catalog near the specified coordinates.

    Parameters:
        target_coord (astropy.coordinates.SkyCoord): Target coordinates
        path_calibration_field (str): Directory path containing catalog files
        matching_radius (float): Matching radius in degrees
        path_save (str): Path to save the results. If None, saves in current directory

    Returns:
        astropy.table.Table: Combined reference catalog table
    """

    path_calibration_field = path_calibration_field or GAIA_REF_DIR

    grid_table = Table.read(os.path.join(path_calibration_field, "grid.csv"))
    c_grid = SkyCoord(grid_table["center_ra"], grid_table["center_dec"], unit="deg")

    sep_arr = target_coord.separation(c_grid).deg
    idx_match = np.where(sep_arr < query_radius)
    matched_grid_table = grid_table[idx_match]

    all_filters = [
        "u", "g", "r", "i", "z",
        "m375w", "m400", "m412", "m425", "m425w", "m437", "m450",
        "m462", "m475", "m487", "m500", "m512", "m525", "m537",
        "m550", "m562", "m575", "m587", "m600", "m612", "m625",
        "m637", "m650", "m662", "m675", "m687", "m700", "m712",
        "m725", "m737", "m750", "m762", "m775", "m787", "m800",
        "m812", "m825", "m837", "m850", "m862", "m875", "m887",
    ]  # fmt:skip

    gaia_column_keys = [
        "source_id",
        "ra",
        "dec",
        "parallax",
        # 'parallax_over_error', # TBD
        "pmra",
        "pmdec",
        "phot_g_mean_mag",
        # 'phot_bp_mean_mag', # TBD
        # 'phot_rp_mean_mag', # TBD
        "bp_rp",
    ]

    all_tablelist = []
    for prefix in matched_grid_table["prefix"]:
        tablelist = []
        for i, filt in enumerate(all_filters):
            f = os.path.join(path_calibration_field, prefix, f"{filt}.fits")
            reftbl = Table.read(f)
            if i == 0:
                tbl = Table()
                for colname in gaia_column_keys:
                    tbl[colname] = reftbl[colname]
            # 	Mag & SNR Keys
            mag_key = f"mag_{filt}"
            snr_key = f"snr_{filt}"
            tbl[mag_key] = reftbl[mag_key]
        tablelist.append(tbl)
        all_tablelist.append(hstack(tablelist))

    all_reftbl = vstack(all_tablelist)
    all_reftbl = unique(all_reftbl, keys="source_id")

    # if not os.path.exists(path_save):
    # 	all_reftbl.write(path_save, overwrite=True)

    return all_reftbl


def filter_table(table: Table, key: str, value: float | int | str, method: str = "equal") -> Table:
    """
    DEPRECATED: use build_condition_mask in tool.utils

    Filter table based on column values.

    Args:
        table: Input table
        key: Column name to filter on
        value: Value to compare against
        method: Comparison method ('equal', 'lower', or 'upper')

    Returns:
        Filtered table
    """
    if method == "equal":
        return table[table[key] == value]
    elif method == "lower":
        return table[table[key] > value]
    elif method == "upper":
        return table[table[key] < value]
    else:
        raise ValueError("method must be 'equal', 'lower', or 'upper'")


def get_aperture_dict(peeing: float | None, pixscale: float) -> dict:
    """
    Generate dictionary of aperture configurations.

    Args:
        peeing: Seeing in pixels
        pixscale: Pixel scale in arcsec/pixel

    Returns:
        Dictionary of aperture configurations
    """
    if peeing is None:
        return {"AUTO": (0.0, "SExtractor AUTO DIAMETER [pix]")}

    aperture_dict = {
        "AUTO": (0.0, "SExtractor AUTO DIAMETER [pix]"),
        "APER": (2 * 0.6731 * peeing, "BEST GAUSSIAN APERTURE DIAMETER [pix]"),
        "APER_1": (2 * peeing, "2*SEEING APERTURE DIAMETER [pix]"),
        "APER_2": (3 * peeing, "3*SEEING APERTURE DIAMETER [pix]"),
        "APER_3": (3 / pixscale, """FIXED 3" APERTURE DIAMETER [pix]"""),
        "APER_4": (5 / pixscale, """FIXED 5" APERTURE DIAMETER [pix]"""),
        "APER_5": (10 / pixscale, """FIXED 10" APERTURE DIAMETER [pix]"""),
    }
    return aperture_dict


@lru_cache(maxsize=1)
def get_flux_fractions() -> tuple:
    """PHOT_FLUXFRAC of main.sex, the flux fractions of FLUX_RADIUS."""
    from ..imcoadd.utils import parse_sex_config

    key = "PHOT_FLUXFRAC"
    value = parse_sex_config(os.path.join(REF_DIR, "srcExt", "main.sex"), [key])[key]
    return tuple(float(v) for v in value.split(","))


def rename_flux_radius_columns(table: Table) -> None:
    """FLUX_RADIUS, FLUX_RADIUS_1, ... -> FLUX_RADIUS_20, FLUX_RADIUS_50, ... in place."""
    fractions = get_flux_fractions()
    for i in reversed(range(len(fractions))):  # descending, so a new name never hits a pending old one
        old_key = "FLUX_RADIUS" if i == 0 else f"FLUX_RADIUS_{i}"
        if old_key in table.colnames:
            table.rename_column(old_key, f"FLUX_RADIUS_{round(fractions[i] * 100)}")


def get_aperture_suffix(aperture_key: str) -> str:
    """
    Such that
    APER -> 0, APER_1 -> 1
    """
    return aperture_key.replace("APER", "0").replace("0_", "")


def get_mag_key(aperture_key: str) -> tuple:
    mag_key = f"MAG_{aperture_key}"
    magerr_key = f"MAGERR_{aperture_key}"
    return (mag_key, magerr_key)


def get_sex_options(
    image: str,
    phot_conf: PhotometryNode,
    egain: float,
    peeing: float,
    pixscale: float,
    satur_level: float = 65000.0,
) -> dict:
    """
    Generate SExtractor configuration arguments.

    Args:
        image: Path to image file
        phot_conf: Photometry configuration object
        gain: CCD gain value
        peeing: Seeing in pixels
        pixscale: Pixel scale in arcsec/pixel

    Returns:
        Dict of SExtractor options (key/value pairs)
    """
    aperture_dict = get_aperture_dict(peeing, pixscale)

    magkeys = list(aperture_dict.keys())
    aperlist = [aperture_dict[key][0] for key in magkeys[1:]]

    PHOT_APERTURES = ",".join(map(str, aperlist))

    sex_options = {}
    if PHOT_APERTURES:
        sex_options["-PHOT_APERTURES"] = PHOT_APERTURES
    sex_options["-SATUR_LEVEL"] = str(satur_level)
    sex_options["-GAIN"] = str(egain)
    sex_options["-PIXEL_SCALE"] = str(pixscale)
    # sex_config["SEEING_FWHM"] = "2.0"  # only adds to confusion. defined in main.sex

    for key in phot_conf.sex_vars.keys():
        if phot_conf.sex_vars[key] is not None:
            key_name = key if key.startswith("-") else f"-{key}"
            sex_options[key_name] = phot_conf.sex_vars[key]

    # 	Add Weight Map (opt-in: weightless coadd photometry was deliberate)
    if getattr(phot_conf, "use_weight_map", False):
        weightim = PathHandler.weight_map(image)
        if os.path.exists(weightim):
            sex_options["-WEIGHT_TYPE"] = "MAP_WEIGHT"
            sex_options["-WEIGHT_IMAGE"] = weightim

    # 	Check Image
    head = image.replace(".fits", "")
    if phot_conf.check:
        sex_options["-CHECKIMAGE_TYPE"] = "SEGMENTATION,APERTURES,BACKGROUND,-BACKGROUND"
        sex_options["-CHECKIMAGE_NAME"] = f"{head}.seg.fits,{head}.aper.fits,{head}.bkg.fits,{head}.sub.fits"
    else:
        pass

    return sex_options


def parse_sex_background(sexout: str) -> tuple[float, float] | None:
    """SExtractor's global Background and RMS from its run summary; None when the line is absent."""
    for line in sexout.splitlines():
        if "Background:" in line and "RMS:" in line:
            return float(line.split("Background:")[1].split("RMS:")[0]), float(line.split("RMS:")[1].split("/")[0])
    return None


def detector_bad_pixels(image) -> np.ndarray:
    """Detector bad pixels of a calibrated single frame, from the mask PathHandler.get_bpmask resolves."""
    from astropy.io import fits
    from ..preprocess.utils import bpmask_id_hdu

    with fits.open(PathHandler.get_bpmask(image), memmap=False) as hdul:
        hdu = hdul[bpmask_id_hdu(hdul)]
        return hdu.data == hdu.header.get("BADPIX", 1)


@contextmanager
def sextractor_zero_weight(
    image, satellite_options=None, bad_pixels=None, skysig=None, psf_fwhm=None, weight_image=None, logger=None
):
    """Trail pixel count and SExtractor options over a /dev/shm weight map zeroing satellite trails and bad pixels."""
    from astropy.io import fits
    from ..imcoadd.masks import detect_satellite_trails

    zero, npix = bad_pixels, 0
    if satellite_options is not None:
        data = fits.getdata(image, memmap=False)
        trail, lines = detect_satellite_trails(data, skysig=skysig, psf_fwhm=psf_fwhm, **satellite_options)
        del data
        npix = int(trail.sum())
        if logger is not None:
            widths = ", half-width " + "/".join(f"{w:.0f}" for w in lines[:, 4]) + " px" if len(lines) else ""
            logger.info(f"Satellite mask: {len(lines)} line(s){widths}, {npix} pixels in {os.path.basename(image)}")
        zero = trail if zero is None else zero | trail
    if zero is None or not zero.any():
        yield npix, {}
        return
    if weight_image:
        weight = fits.getdata(weight_image, memmap=False).astype(np.float32)
        weight[zero] = 0
    else:
        weight = (~zero).astype(np.uint8)
    fd, path = tempfile.mkstemp(prefix="zero_weight_", suffix=".fits", dir="/dev/shm")
    os.close(fd)
    try:
        fits.writeto(path, weight, overwrite=True)
        del weight
        # INTERP_TYPE NONE: the default ALL refills zero-weight runs up to 16 px from their neighbours
        options = {"-WEIGHT_TYPE": "MAP_WEIGHT", "-WEIGHT_IMAGE": path, "-INTERP_TYPE": "NONE"}
        if bad_pixels is not None:
            # ALL refills a bad pixel from its neighbours; a 2 px lag leaves wider zero-weight runs such as trails empty
            options.update({"-INTERP_TYPE": "ALL", "-INTERP_MAXXLAG": "2", "-INTERP_MAXYLAG": "2"})
        yield npix, options
    finally:
        os.remove(path)


def dicts_to_lists(dicts):
    # zps = [v[0] for _, dict_pair in dicts.items() for k, v in dict_pair[0].items() if k.startswith("ZP")]
    filters = []
    zps = []
    zperrs = []
    for filt, dict_pair in dicts.items():
        filters.append(filt)
        for k, t in dict_pair[0].items():
            if k == "ZP_AUTO":
                zps.append(t[0])
            if k == "EZP_AUTO":
                zperrs.append(t[0])

    return filters, zps, zperrs


def get_zp_from_dict(dicts: dict, filter: str) -> dict:
    """
    Get the zero point dictionary.

    Args:
        zp_dict: Zero point dictionary

    Returns:
        Zero point dictionary
    """
    return dicts[filter][0]["ZP_AUTO"][0], dicts[filter][0]["EZP_AUTO"][0]
