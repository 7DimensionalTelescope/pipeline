from __future__ import annotations
import os
import tempfile
import numpy as np
from contextlib import contextmanager
from functools import lru_cache
from typing import TYPE_CHECKING, Callable
from numba import njit
from astropy.table import Table, hstack, vstack, unique
from astropy.coordinates import SkyCoord
from astropy.stats import SigmaClip

from ..const import EGAIN_2750, FILTERS, GAIA_REF_DIR, REF_DIR, ZP_MAX_AT
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
def limitmag(n_sigma: np.ndarray, zp: float, aper: float, skysigma: float, noise_factor: float) -> np.ndarray:
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
    return np.round(upperlimit, 6)


def aperture_noise_factor(acf, aperture: float) -> float:
    """sqrt(sum_h acf(h) O(h) / (pi r^2)), O the exact-weight aperture overlap averaged over sub-pixel phases."""
    from scipy.signal import fftconvolve
    from photutils.aperture import CircularAperture

    if acf is None:
        raise ValueError("A measured sky autocorrelation is required for aperture depth")
    acf = np.asarray(acf, dtype=float)
    half = acf.shape[0] // 2
    if half < np.ceil(float(aperture)):  # exact weights overlap out to ceil(diameter) px on each axis
        raise ValueError(f"Lag window +-{half} px is narrower than the {aperture:.2f} px aperture's overlap")
    radius = 0.5 * float(aperture)
    span = int(np.ceil(radius)) + 2
    shape = (2 * span + 1, 2 * span + 1)
    reach = min(half, shape[0] - 1)
    overlap = np.zeros_like(acf)
    offsets = np.linspace(0.0, 1.0, 16, endpoint=False)
    for dx in offsets:
        for dy in offsets:
            weights = CircularAperture([(span + dx, span + dy)], r=radius).to_mask(method="exact")[0].to_image(shape)
            correlation = fftconvolve(weights, weights[::-1, ::-1], mode="full")
            overlap[half - reach : half + reach + 1, half - reach : half + reach + 1] += correlation[
                shape[0] - 1 - reach : shape[0] + reach, shape[1] - 1 - reach : shape[1] + reach
            ]
    variance = float(np.sum(acf * overlap)) / offsets.size**2
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("Invalid aperture variance from sky autocorrelation")
    return float(np.sqrt(variance / (np.pi * radius**2)))


def bin_noise_factor(acf, box: int) -> float:
    """sqrt(sum_h acf(h) O(h)) / box for a box x box pixel sum, O(h) = (box - |hx|)(box - |hy|)."""
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
    DEPRECATED 2026-10-31: reads the first-generation 1x1 deg lattice (47 bands, no second-medium or wide
    filters); the photometry uses aggregate_tile_refcats. Kept for reference.

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


def aggregate_tile_refcats(target_coord: SkyCoord, ref_cat_for_tile: Callable[[str], str], query_radius: float = 1.0) -> Table:
    """Stack the RIS tile refcats within the radius, unique on source_id; empty Table without a tile file."""
    from ..utils.tile import get_ris_tiles_near

    radius = max(query_radius, 1.2)  # 1.2 deg reaches a field corner at the widest tile spacing, 1.29 deg
    tables = []
    for tile in get_ris_tiles_near(target_coord.ra.deg, target_coord.dec.deg, radius=radius):
        f = ref_cat_for_tile(tile)
        if os.path.exists(f):
            tables.append(Table.read(f))
    if not tables:
        return Table()
    return unique(vstack(tables), keys="source_id")


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
        data, header = fits.getdata(image, memmap=False, header=True)
        trail, lines = detect_satellite_trails(
            data, skysig=skysig, psf_fwhm=psf_fwhm, saturate=header.get("SATURATE"), **satellite_options
        )
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
        try:
            os.remove(path)
        except FileNotFoundError:  # a /dev/shm sweep on the worker took it; the run itself is unaffected
            pass


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


def _weighted_linfit(x, y, sigma, nsig=3.0, maxiter=5):
    """bprp_colorterm.weighted_linfit: (a, a_err, b, n_used)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    sigma = np.asarray(sigma, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
    a = b = a_err = np.nan
    for _ in range(maxiter):
        if m.sum() < 5 or np.ptp(x[m]) == 0:
            return np.nan, np.nan, np.nan, int(m.sum())
        (a, b), cov = np.polyfit(x[m], y[m], 1, w=1.0 / sigma[m], cov=True)
        a_err = float(np.sqrt(cov[0, 0]))
        res = y - (a * x + b)
        s = max(np.std(res[m]), 1e-4)
        new = m & (np.abs(res) < nsig * s)
        if new.sum() == m.sum():
            break
        m = new
    return float(a), a_err, float(b), int(m.sum())


def color_term(ref_mag, inst_mag, inst_magerr, bp_rp, min_stars: int = 15) -> tuple[float | None, float | None, int]:
    """bprp_colorterm.colorterm_bprp: (a, a_err, n_fit); (None, None, 0) under min_stars."""
    ref_mag, inst_mag, bp_rp, inst_magerr = (np.asarray(v, float) for v in (ref_mag, inst_mag, bp_rp, inst_magerr))
    zp = ref_mag - inst_mag
    sig = np.sqrt(inst_magerr**2 + 0.01**2)
    fin = np.isfinite(zp) & np.isfinite(bp_rp) & np.isfinite(sig)
    if fin.sum() < min_stars:
        return None, None, 0
    clipped = SigmaClip(sigma=3.0, maxiters=5)(zp[fin])
    keep = np.where(fin)[0][~np.ma.getmaskarray(clipped)]
    if len(keep) < min_stars:
        return None, None, 0
    zk = zp[keep]
    a, a_err, _, n_fit = _weighted_linfit(bp_rp[keep], zk - np.nanmedian(zk), sig[keep])
    return a, a_err, n_fit


def color_term_of_table(zp_src_table, ref_mag_key: str, min_stars: int) -> tuple[float | None, float | None, int]:
    """color_term on the zero-point star table: reference magnitude, MAG_APER_5, MAGERR_APER_5, bp_rp."""
    col = lambda k: np.ma.filled(np.ma.asarray(zp_src_table[k], dtype=float), np.nan)
    return color_term(col(ref_mag_key), col("MAG_APER_5"), col("MAGERR_APER_5"), col("bp_rp"), min_stars)


def zp_max(filt: str, exptime, gain, egain) -> float | None:
    """ZP_AUTO upper limit of `filt` at this exposure and camera gain; None without a limit, an exposure or a usable gain."""
    if filt not in FILTERS or not exptime or gain is None:
        return None
    limit = FILTERS[filt]["zp_max"] + 2.5 * np.log10(exptime / ZP_MAX_AT["exptime"])
    if float(gain) == float(ZP_MAX_AT["gain"]):
        return limit
    return limit - 2.5 * np.log10(egain / EGAIN_2750) if egain else None


def _zp_cards(header) -> dict:
    return header.zp_dict if header.aperture_info else {}


def zero_point_error_test(header, alleged: str, min_separable_wavelength, sigma_factor) -> tuple | None:
    """(key, value, reason) when INF_FILT beats FILTER beyond min_separable_wavelength by more than sigma_factor times its EZP."""
    ezp = _zp_cards(header).get("EZP_AUTO", (None,))[0]
    if not (alleged in FILTERS and header.INF_FILT in FILTERS and header.INF_DEZF and header.INF_DEZF > 0 and ezp):
        return None
    separation = abs(FILTERS[alleged]["center_nm"] - FILTERS[header.INF_FILT]["center_nm"])
    if separation <= min_separable_wavelength:
        return None
    ezp_inf = ezp - header.INF_DEZF
    ratio = header.INF_DEZF / ezp_inf if ezp_inf > 0 else np.inf
    if ratio > sigma_factor:
        why = (
            f"INF_FILT {header.INF_FILT} is {separation:.1f} nm from FILTER {alleged}; INF_DEZF = {ratio:.1f} x EZP_INF"
        )
        return "INF_DEZF", header.INF_DEZF, why
    return None


def zp_excursion_test(header, alleged: str, limit, exptime) -> tuple | None:
    """(key, value, reason) when ZP_AUTO exceeds the zp_max of FILTER at this exposure."""
    zp = _zp_cards(header).get("ZP_AUTO", (None,))[0]
    if zp is None or limit is None or zp <= limit:
        return None
    return "ZP_AUTO", zp, f"deeper than the zp_max of {alleged}, {limit:.2f} at {exptime:g} s"


def color_term_test(header, alleged: str) -> tuple | None:
    """(key, value, reason) when CTERM lies outside the cterm window of FILTER in ref/filters.yml."""
    window = FILTERS.get(alleged, {}).get("cterm")
    if not window or header.CTERM is None or not np.isfinite(header.CTERM) or window[0] <= header.CTERM <= window[1]:
        return None
    return "CTERM", header.CTERM, f"outside the {alleged} window [{window[0]:+.3f}, {window[1]:+.3f}]"


def filter_check_failures(header, alleged: str, limit, exptime, filter_check: dict) -> list:
    """The (key, value, reason) of every test switched on in the config's photometry.filter_check that fails."""
    tests = {
        "zero_point_error": lambda: zero_point_error_test(
            header, alleged, filter_check["min_separable_wavelength"], filter_check["sigma_factor"]
        ),
        "zp_excursion": lambda: zp_excursion_test(header, alleged, limit, exptime),
        "color_term": lambda: color_term_test(header, alleged),
    }
    return [hit for name, test in tests.items() if filter_check[name] and (hit := test())]
