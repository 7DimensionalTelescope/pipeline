"""Zackay & Ofek (2015) proper coaddition of SWarp-resampled, background-subtracted frames.

Same grid contract as calc.py: frames share pixel scale and projection, differ only in
CRPIX, and exact 0.0 marks "no data". Derivations, policy semantics and measured
verification numbers live in the codebase memory (scientific-guidelines.md).
"""

import os
import time

import numpy as np
from astropy.io import fits

from ..services.logger import Logger
from ..utils import add_suffix, get_basename, time_diff_in_seconds
from .calc import WEIGHT_EPS
from .utils import build_coadd_wcs_header, determine_size

WEIGHT_MAP_POLICIES = ("off", "weighted-mean", "white-noise", "colored-noise")
LANCZOS_PSD_FLOOR = 1e-3  # caps the colored-noise whitening gain where the kernel response dies
FFT_WORKERS = 8
_LANCZOS_PSD_CACHE: dict = {}


def _gauss_radius(sigma: float) -> int:
    # 6 sigma, not convolve.py's 4: the truncation step's spectral lobes must sit below
    # the Gaussian OTF at Nyquist or P_hat/|P_hat| flips sign there (measured 3e-3 of peak)
    return max(int(np.ceil(6.0 * sigma)), 2)


def gaussian_psf_1d(fwhm_px: float) -> np.ndarray:
    """Unit-sum 1-D Gaussian PSF model on a 6-sigma support."""
    sigma = float(fwhm_px) / np.sqrt(8 * np.log(2))
    x = np.arange(-_gauss_radius(sigma), _gauss_radius(sigma) + 1, dtype=np.float64)
    g = np.exp(-0.5 * (x / sigma) ** 2)
    return g / g.sum()


def psf_autocorrelation(fwhm_px: float) -> np.ndarray:
    """Centred |PSF|^2 carrier: full autocorrelation stamp of the separable Gaussian."""
    g = gaussian_psf_1d(fwhm_px)
    a1 = np.correlate(g, g, mode="full")
    return np.outer(a1, a1)


def lanczos_psd_1d(freqs: np.ndarray, a: int = 3) -> np.ndarray:
    """|FT(Lanczos-a kernel)|^2 at cycles/pixel: the stationary resampling noise transfer."""
    if a not in _LANCZOS_PSD_CACHE:
        dx = 1.0 / 64
        x = np.arange(-a, a, dx)
        kernel = np.sinc(x) * np.sinc(x / a)
        spec = np.fft.rfft(kernel, 1 << 14) * dx
        spec /= spec[0].real  # unit response at DC: the kernel conserves flux
        _LANCZOS_PSD_CACHE[a] = (np.fft.rfftfreq(1 << 14, d=dx), np.abs(spec) ** 2)
    nu, t2 = _LANCZOS_PSD_CACHE[a]
    return np.interp(np.abs(np.asarray(freqs, dtype=np.float64)), nu, t2)


def _corner_embedded(stamp: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    big = np.zeros(shape, dtype=np.float64)
    sh, sw = stamp.shape
    big[:sh, :sw] = stamp
    return np.roll(big, (-(sh // 2), -(sw // 2)), axis=(0, 1))


def _power_spectrum(stamp: np.ndarray, q: int, cache: dict, key) -> np.ndarray:
    """Full-plane real spectrum of a centred autocorrelation stamp on a q x q grid."""
    import scipy.fft as sfft

    if key not in cache:
        cache[key] = np.maximum(sfft.fft2(_corner_embedded(stamp, (q, q))).real, 0.0)
    return cache[key]


def _coadd_psf_stamp(acfs: list[np.ndarray], w: np.ndarray, q: int) -> np.ndarray:
    """P_R = IFT(sqrt(sum w_j |P_hat_j|^2)) / F_R, cropped to the widest input stamp."""
    import scipy.fft as sfft

    cache: dict = {}
    den = np.zeros((q, q), dtype=np.float64)
    for i, (acf, wi) in enumerate(zip(acfs, w)):
        den += wi * _power_spectrum(acf, q, cache, i)
    p_r = np.fft.fftshift(sfft.ifft2(np.sqrt(den)).real) / np.sqrt(w.sum())
    size = max(a.shape[0] for a in acfs)
    c = q // 2
    stamp = p_r[c - size // 2 : c + size // 2 + 1, c - size // 2 : c + size // 2 + 1]
    return stamp / stamp.sum()


def _psf_fwhm_px(stamp: np.ndarray) -> float:
    return float(2.0 * np.sqrt(np.sum(stamp > stamp.max() / 2) / np.pi))


def robust_sky_sigma(path: str, step: int = 8) -> float:
    """1.4826 x MAD of the nonzero finite pixels, from every ``step``-th row."""
    with fits.open(path, memmap=True) as hdul:
        sample = np.asarray(hdul[0].data[::step], dtype=np.float64)
    vals = sample[np.isfinite(sample) & (sample != 0.0)]
    if vals.size < 1000:
        return float("nan")
    med = np.median(vals)
    return float(1.4826 * np.median(np.abs(vals - med)))


def proper_coadd_numpy(
    input_images: list[str],
    output_path: str,
    coadd_header: fits.Header,
    peeings: list[float],
    skysigs: list[float | None],
    flxscales: list[float] | bool | None = None,
    weight_map_policy: str = "white-noise",
    weight_output: str | bool | None = None,
    footprint_output: str | bool | None = None,
    psf_output: str | bool | None = None,
    holes: list[str] | None = None,
    match_swarp_size: bool = True,
    logger: Logger | None = None,
) -> str:
    """Streaming proper coadd: flux-normalized, coverage-renormalized, O(1) memory in N."""
    import scipy.fft as sfft
    from scipy.ndimage import gaussian_filter

    st = time.time()
    n = len(input_images)
    policy = str(weight_map_policy or "off").lower().replace("_", "-")
    if policy not in WEIGHT_MAP_POLICIES:
        raise ValueError(f"Invalid weight_map_policy: {weight_map_policy!r} (expected one of {WEIGHT_MAP_POLICIES})")
    for name, seq in (("peeings", peeings), ("skysigs", skysigs)):
        if len(seq) != n:
            raise ValueError(f"{name} ({len(seq)}) and input_images ({n}) length mismatch")
    if holes is not None and len(holes) != n:
        raise ValueError(f"holes ({len(holes)}) and input_images ({n}) length mismatch")
    colored = policy == "colored-noise"
    if logger is not None:
        logger.info(f"Start proper coaddition (Zackay & Ofek 2015): weight-map policy {policy!r}")

    if flxscales is False or flxscales is None:
        f = np.ones(n, dtype=np.float64)
    else:
        if len(flxscales) != n:
            raise ValueError(f"flxscales ({len(flxscales)}) and input_images ({n}) length mismatch")
        f = np.array([1.0 if v is None else float(v) for v in flxscales], dtype=np.float64)

    s = np.empty(n, dtype=np.float64)
    for i, (path, skysig) in enumerate(zip(input_images, skysigs)):
        if skysig:
            s[i] = float(skysig)
        else:
            s[i] = robust_sky_sigma(path)
            if logger is not None:
                logger.warning(f"{get_basename(path)}: no SKYSIG; measured robust sigma {s[i]:.3f}")
    if not np.all(np.isfinite(s) & (s > 0)):
        bad = [get_basename(p) for p, ok in zip(input_images, np.isfinite(s) & (s > 0)) if not ok]
        raise ValueError(f"No usable sky sigma for {bad[:3]} ({len(bad)} total)")

    # sigma of the flux-normalized frame; w_j absorbs F_j^2 of the Z&O formulation
    sigma_scaled = f * s
    w = 1.0 / sigma_scaled**2
    f_r = float(np.sqrt(w.sum()))

    acfs = [psf_autocorrelation(p) for p in peeings]
    canvas = max(a.shape[0] for a in acfs)
    q = 1 << int(np.ceil(np.log2(max(128, 2 * canvas))))
    # k=0 shares: they renormalize aperture (total) flux and scale the aperture-scale
    # weight; per-pixel variance shares were measured misleading at coverage edges
    # (matched-filter correlation) and a diagonal-only map cannot represent that
    r_share = w / w.sum()
    d_acf = np.zeros((canvas, canvas), dtype=np.float64)
    for acf, wi in zip(acfs, w):
        k = acf.shape[0]
        lo = (canvas - k) // 2
        d_acf[lo : lo + k, lo : lo + k] += wi * acf

    target_w, target_h, target_cx, target_cy, x0, y0, shapes = determine_size(input_images, match_swarp_size)

    num_arr = np.zeros((target_h, target_w), dtype=np.float64)
    resp_arr = np.zeros((target_h, target_w), dtype=np.float32)
    count_arr = np.zeros((target_h, target_w), dtype=np.int16)
    # 'weighted-mean' is the conventional product (holes not marked); the formulation
    # policies mark bad-pixel holes share-wise through a second coverage accumulator
    track_holes = holes is not None and policy in ("white-noise", "colored-noise")
    respw_arr = np.zeros((target_h, target_w), dtype=np.float32) if track_holes else None

    for i, path in enumerate(input_images):
        st_img = time.time()
        data = np.ascontiguousarray(fits.getdata(path, memmap=False), dtype=np.float32)
        finite = np.isfinite(data)
        if not finite.all():
            data[~finite] = 0.0
        sigma_px = float(peeings[i]) / np.sqrt(8 * np.log(2))
        # data on the array boundary (a frame clipped by the coadd grid) turns the linear
        # correlation's truncation into 1/sqrt(den)-amplified ringing (measured 6e5 ADU)
        edge = _gauss_radius(sigma_px)
        data[:edge, :] = data[-edge:, :] = data[:, :edge] = data[:, -edge:] = 0.0
        # float64 and constant-0 boundary like convolve_gaussian_separable: float32 roundoff
        # at PSF-dead frequencies is amplified by 1/sqrt(den) (measured 3e-3 of peak)
        matched = gaussian_filter(
            np.asarray(data, dtype=np.float64), sigma_px,
            mode="constant", cval=0.0, radius=_gauss_radius(sigma_px), output=np.float64,
        )  # fmt: skip

        h, wd = data.shape
        tx0 = max(0, x0[i]); tx1 = min(target_w, x0[i] + wd)  # fmt: skip
        ty0 = max(0, y0[i]); ty1 = min(target_h, y0[i] + h)  # fmt: skip
        if tx1 <= tx0 or ty1 <= ty0:
            if logger is not None:
                logger.debug(f"{get_basename(path)}: no overlap with target grid; skipped")
            continue
        sx0 = tx0 - x0[i]; sx1 = tx1 - x0[i]  # fmt: skip
        sy0 = ty0 - y0[i]; sy1 = ty1 - y0[i]  # fmt: skip
        sl = (slice(ty0, ty1), slice(tx0, tx1))

        num_arr[sl] += (w[i] * f[i]) * matched[sy0:sy1, sx0:sx1]
        m = data[sy0:sy1, sx0:sx1] != 0.0
        resp_arr[sl] += np.float32(r_share[i]) * m
        count_arr[sl] += m
        if respw_arr is not None:
            with fits.open(holes[i], memmap=True) as mh:
                respw_arr[sl] += np.float32(r_share[i]) * (m & (mh[0].data[sy0:sy1, sx0:sx1] > WEIGHT_EPS))
        del data, matched
        if logger is not None:
            logger.debug(
                f"{get_basename(path)} [image {i + 1}/{n}] w={w[i]:.4g} sigma'={sigma_scaled[i]:.3f} "
                f"fwhm={peeings[i]:.2f}px in {time_diff_in_seconds(st_img)} seconds"
            )

    den = sfft.rfft2(_corner_embedded(d_acf, (target_h, target_w)), workers=FFT_WORKERS).real
    divisor = np.sqrt(np.maximum(den, den.max() * 1e-12))
    del den
    if colored:
        # colored noise rescales num and den by the same |L_hat|^2, leaving sqrt(|L_hat|^2)
        t2y = lanczos_psd_1d(np.fft.fftfreq(target_h))
        t2x = lanczos_psd_1d(np.fft.rfftfreq(target_w))
        divisor *= np.sqrt(np.maximum(np.outer(t2y, t2x), LANCZOS_PSD_FLOOR))
    num_hat = sfft.rfft2(num_arr, workers=FFT_WORKERS)
    num_hat /= divisor
    del divisor
    proper = sfft.irfft2(num_hat, s=(target_h, target_w), workers=FFT_WORKERS)
    del num_hat, num_arr

    with np.errstate(divide="ignore", invalid="ignore"):
        proper /= f_r * resp_arr.astype(np.float64)
    coadd = proper.astype(np.float32)
    del proper
    coadd[count_arr == 0] = np.nan

    out_header = build_coadd_wcs_header(input_images[0], target_cx, target_cy, coadd_header)
    out_header["PROPFR"] = (f_r, "flux scale F_R of the proper coadd")
    out_header["PROPPSF"] = ("GAUSSIAN-PEEING", "per-frame PSF model of the proper coadd")
    fits.writeto(output_path, coadd, header=out_header, overwrite=True)
    if logger is not None:
        logger.info(f"Proper coadd written (F_R {f_r:.4f}, expected sky sigma {1.0 / f_r:.4f})")

    if weight_output is not False and policy != "off":
        weight_out = weight_output or add_suffix(output_path, "weight")
        # aperture-scale inverse variance: F_R^2 c(x) = sum of covering w_j, the note's
        # F_R^2 at full coverage; the exact per-pixel diagonal is deliberately not shipped
        resp_w = respw_arr if respw_arr is not None else resp_arr
        weight_map = (f_r * f_r) * resp_w.astype(np.float64)
        fits.writeto(weight_out, weight_map.astype(np.float32), header=out_header, overwrite=True)
        if logger is not None:
            logger.debug(f"Wrote proper coadd weight map ({policy}): {weight_out}")

    if footprint_output is not False:
        footprint_out = footprint_output or add_suffix(output_path, "footprint")
        fits.writeto(footprint_out, count_arr, header=out_header, overwrite=True)
        if logger is not None:
            logger.debug(f"Wrote coadd footprint (max {int(count_arr.max())} frames): {footprint_out}")

    if psf_output is not False:
        psf_out = psf_output or add_suffix(output_path, "psf")
        stamp = _coadd_psf_stamp(acfs, w, q)
        psf_header = fits.Header()
        psf_header["PROPFR"] = (f_r, "flux scale F_R of the proper coadd")
        psf_header["PSFFWHM"] = (_psf_fwhm_px(stamp), "[px] FWHM of the coadd PSF P_R")
        fits.writeto(psf_out, stamp.astype(np.float32), header=psf_header, overwrite=True)
        if logger is not None:
            logger.debug(f"Wrote coadd PSF stamp (FWHM {psf_header['PSFFWHM']:.2f} px): {psf_out}")

    if logger is not None:
        logger.info(f"Proper coaddition completed in {time_diff_in_seconds(st)} seconds")
    return output_path
