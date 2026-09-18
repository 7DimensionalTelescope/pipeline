from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve


RESIDUAL_BOX_SIZE = 20
NOISE_LAG = 8
MIN_PATCHES = 16
MIN_FRACTION = 0.75
MAX_PATCHES = 1024
SIGMA_QUANTILES = 9  # NOISQ01..NOISQ09: the master dark writes them, the BACKOFF reference mixes over them


@dataclass(frozen=True)
class BackgroundResiduals:
    backoff: float | None = None
    backsys: float | None = None
    backscl: int = RESIDUAL_BOX_SIZE
    backn: int = 0
    backref: str = "SUBTRACT"
    backnref: str | None = None

    def cards(self, include_missing: bool = False) -> dict:
        descriptions = (
            ("backoff", "[ADU] Sky true mode minus model; positive = under-subtracted"),
            ("backsys", "[ADU] Excess spatial RMS of patch mean residuals"),
            ("backscl", "[pixel] Residual measurement square width"),
            ("backn", "Number of valid residual sky patches measured"),
            ("backref", "Residual reference: MODEL, SUBTRACT, or COADD"),
            ("backnref", "BACKOFF noise reference: SIGMAQ cards or WIDTH"),
        )
        return {key.upper(): (getattr(self, key), comment) for key, comment in descriptions
                if include_missing or getattr(self, key) is not None}


RESIDUAL_KEYS = tuple(BackgroundResiduals.__dataclass_fields__)


def clear_residual_cards(header) -> None:
    for key in RESIDUAL_KEYS:
        header.pop(key.upper(), None)


def sigma_quantile_cards(sigma, n: int = SIGMA_QUANTILES) -> dict:
    """NOISQ01..NOISQnn: the additive noise sigma map's quantiles at (k - 1/2) / n, the BACKOFF reference's noise mixture."""
    values = np.asarray(sigma, dtype=np.float32).ravel()
    values = values[np.isfinite(values) & (values > 0)]
    quantiles = np.quantile(values, (np.arange(n) + 0.5) / n)
    return {f"NOISQ{k + 1:02d}": (round(float(q), 4), f"[ADU] Additive noise sigma, quantile {2 * k + 1}/{2 * n}")
            for k, q in enumerate(quantiles)}


def sigma_quantiles(header, n: int = SIGMA_QUANTILES):
    """The NOISQ cards as an array; None when any is missing."""
    values = [header.get(f"NOISQ{k + 1:02d}") for k in range(n)]
    if any(v is None for v in values):
        return None
    return np.array(values, dtype=np.float64)


def coadd_taps(n_inputs: int, scale: float, kernel=None):
    """The weights one coadd pixel puts on input pixels: n_inputs frames at scale / n_inputs each through the resampling kernel
    (default SWarp LANCZOS3's phase-averaged noise-equivalent kernel: sum k^2 = 0.803, sum k^3 / sum k^2 = 0.867 in 2-D)."""
    if kernel is None:
        taps = np.array([-0.003497, 0.015380, -0.039026, 0.056460, 0.941366, 0.056460, -0.039026, 0.015380, -0.003497])
        kernel = np.outer(taps, taps)
    return np.tile(np.asarray(kernel, dtype=np.float64).ravel() * scale / n_inputs, n_inputs)


def mixture_density(taps, gain: float, sigmas, levels, dx: float = 0.01, span: float = 40.0):
    """Density, about its mean, of sum_j taps_j x_j with x_j = Poisson(level gain e-) / gain + N(0, sigma) in ADU, each x_j drawn
    from the equal-weight mixture over every (sigma, level) pair: the noise-only distribution of a sky pixel (single: taps [1])."""
    taps = np.asarray(taps, dtype=np.float64).ravel()
    sig, lev = (a.ravel() for a in np.meshgrid(np.asarray(sigmas, dtype=np.float64), np.asarray(levels, dtype=np.float64)))
    lam = lev * gain
    width = np.sqrt(np.sum(taps**2) * (np.mean(lev) / gain + np.mean(sig**2)))
    n = int(2 ** np.ceil(np.log2(span * width / dx)))
    x = (np.arange(n) - n // 2) * dx
    t = 2 * np.pi * np.fft.fftfreq(n, dx)
    cf = np.ones(n, dtype=np.complex128)
    values, counts = np.unique(np.round(taps[taps != 0], 12), return_counts=True)
    for tap, m in zip(values, counts):
        u = tap * t / gain
        component = np.zeros(n, dtype=np.complex128)
        for i in range(0, lam.size, 16):  # chunked: n_pairs x n grid points of complex exponentials
            component += np.exp(lam[i:i + 16, None] * (np.expm1(1j * u)[None, :] - 1j * u[None, :])
                                - 0.5 * (sig[i:i + 16, None] * gain * u[None, :]) ** 2).sum(axis=0)
        cf *= (component / lam.size) ** m
    p = np.real(np.fft.fft(cf * np.exp(-1j * t * x[0]))) / (n * dx)
    return x, np.clip(p, 0, None)


def density_gap(x, p) -> float:
    """Median minus mode of a density on a grid; the mode from the parabola through the peak and its neighbours."""
    w = p / p.sum()
    median = float(np.interp(0.5, np.cumsum(w) - 0.5 * w, x))
    i = int(np.argmax(p))
    if 0 < i < p.size - 1 and p[i - 1] > 0 and p[i + 1] > 0:
        y0, y1, y2 = np.log(p[i - 1:i + 2])
        mode = float(x[i] + 0.5 * (x[1] - x[0]) * (y0 - y2) / (y0 - 2 * y1 + y2))
    else:
        mode = float(x[i])
    return median - mode


def modal_offset(residual, exclude, gain: float, levels, sigmas=None, taps=None) -> tuple[float, str]:
    """BACKOFF: the median of the residual's sky pixels minus the reference noise mixture's median - mode, i.e. the sky's true mode
    minus the model; positive = under-subtracted. ``levels`` are the sky levels the pixels sit at (the model's quantiles for a
    single, the inputs' sky for a coadd) and ``sigmas`` the NOISQ additive-noise quantiles; without them the reference is one
    Gaussian whose width is the residual's own robust width less the Poisson part. Returns (backoff, noise reference kind)."""
    sky = np.asarray(residual)[~np.asarray(exclude, bool) & np.isfinite(residual)].astype(np.float64)
    taps = np.array([1.0]) if taps is None else np.asarray(taps, dtype=np.float64)
    levels = np.atleast_1d(np.asarray(levels, dtype=np.float64))
    median = float(np.median(sky))
    if sigmas is None:
        width = 1.4826 * float(np.median(np.abs(sky - median)))
        sigmas = [np.sqrt(max(width**2 / np.sum(taps**2) - np.mean(levels) / gain, 1e-3))]
        kind = "WIDTH"
    else:
        kind = "SIGMAQ"
    return median - density_gap(*mixture_density(taps, gain, sigmas, levels)), kind


def _autocorrelation(values):
    correlation = fftconvolve(values, values[::-1, ::-1], mode="full")
    y, x = np.array(values.shape) - 1
    return correlation[y - NOISE_LAG:y + NOISE_LAG + 1, x - NOISE_LAG:x + NOISE_LAG + 1]


def _noise_covariance(pixels, valid):
    yy, xx = np.mgrid[:pixels.shape[0], :pixels.shape[1]] / max(pixels.shape) - 0.5
    design = np.column_stack([a[valid] for a in (np.ones(pixels.shape), xx, yy, xx**2, xx*yy, yy**2)])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return None
    basis, _ = np.linalg.qr(design, mode="reduced")
    values = pixels[valid].astype(np.float64)
    residual = np.zeros(pixels.shape, dtype=np.float64)
    residual[valid] = values - basis @ (basis.T @ values)
    pairs = np.rint(_autocorrelation(valid.astype(float)))
    if np.any(pairs <= 0):
        return None
    projection = np.zeros(pairs.shape)
    plane = np.zeros(pixels.shape)
    for column in basis.T:
        plane[valid] = column
        projection += _autocorrelation(plane)
    return _autocorrelation(residual) / pairs, projection / pairs


def _patch_noise(valid, covariance, projection):
    n = int(valid.sum())
    pairs = np.rint(_autocorrelation(valid.astype(float)))
    response = float(np.sum(pairs * projection)) / n
    if response >= 0.5:
        return None
    return float(np.sum(pairs * covariance)) / (n**2 * (1 - response))


def measure_background_residuals(
    data, exclude=None, coverage=None, box_size: int = RESIDUAL_BOX_SIZE, mesh_box: int | None = None
) -> BackgroundResiduals:
    """Residual offset and excess patch scatter, with covariance estimated on larger spatial tiles."""
    data = np.asarray(data)
    size = int(box_size)
    if data.ndim != 2 or size < 16:
        raise ValueError("Residual sky requires a 2D image and box_size >= 16")
    # the noise tile stays inside one mesh box, so its quadratic fit does not absorb the model's own error
    group = 4 if mesh_box is None else max(2, min(4, int(mesh_box) // size))
    if exclude is not None and np.shape(exclude) != data.shape:
        raise ValueError("Source mask shape differs from the residual image")
    if coverage is not None and np.shape(coverage) != data.shape:
        raise ValueError("Coverage shape differs from the residual image")
    ny, nx = (length // size for length in data.shape)
    if ny * nx < MIN_PATCHES:
        return BackgroundResiduals(backscl=size)
    means, noises = [], []
    group_nx = (nx + group - 1) // group
    group_ids = np.arange(((ny + group - 1) // group) * group_nx)
    np.random.default_rng(73519).shuffle(group_ids)
    for group_id in group_ids:
        gy, gx = divmod(int(group_id), group_nx)
        sy = slice(gy * group * size, min((gy * group + group), ny) * size)
        sx = slice(gx * group * size, min((gx * group + group), nx) * size)
        pixels = data[sy, sx]
        valid = np.isfinite(pixels)
        if exclude is not None:
            valid &= ~exclude[sy, sx]
        if coverage is not None:
            valid &= coverage[sy, sx]
        if valid.sum() < MIN_FRACTION * pixels.size:
            continue
        noise_model = _noise_covariance(pixels, valid)
        if noise_model is None:
            continue
        for iy in range(pixels.shape[0] // size):
            for ix in range(pixels.shape[1] // size):
                patch = np.s_[iy * size:(iy + 1) * size, ix * size:(ix + 1) * size]
                usable = valid[patch]
                if usable.sum() < MIN_FRACTION * size**2:
                    continue
                noise = _patch_noise(usable, *noise_model)
                if noise is None or not np.isfinite(noise):
                    continue
                means.append(float(np.mean(pixels[patch][usable], dtype=np.float64)))
                noises.append(noise)
        if len(means) >= MAX_PATCHES:
            break
    n = len(means)
    if n < MIN_PATCHES:
        return BackgroundResiduals(backscl=size, backn=n)
    variance = float(np.var(means, ddof=1))
    noise = max(0.0, float(np.mean(noises)))
    # backoff is the caller's: modal_offset against the frame's noise reference, not the patch means' mean
    return BackgroundResiduals(backsys=float(np.sqrt(max(0.0, variance - noise))), backscl=size, backn=n)
