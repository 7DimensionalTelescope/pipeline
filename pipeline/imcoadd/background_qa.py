from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve


RESIDUAL_BOX_SIZE = 20
NOISE_LAG = 8
MIN_PATCHES = 16
MIN_FRACTION = 0.75
MAX_PATCHES = 1024


@dataclass(frozen=True)
class BackgroundResiduals:
    backoff: float | None = None
    backsys: float | None = None
    backscl: int = RESIDUAL_BOX_SIZE
    backn: int = 0
    backref: str = "SUBTRACT"

    def cards(self, include_missing: bool = False) -> dict:
        descriptions = (
            ("backoff", "[ADU] Mean residual sky; positive = under-subtracted"),
            ("backsys", "[ADU] Excess spatial RMS of patch mean residuals"),
            ("backscl", "[pixel] Residual measurement square width"),
            ("backn", "Number of valid residual sky patches measured"),
            ("backref", "Residual reference: MODEL, SUBTRACT, or COADD"),
        )
        return {key.upper(): (getattr(self, key), comment) for key, comment in descriptions
                if include_missing or getattr(self, key) is not None}


RESIDUAL_KEYS = tuple(BackgroundResiduals.__dataclass_fields__)


def clear_residual_cards(header) -> None:
    for key in RESIDUAL_KEYS:
        header.pop(key.upper(), None)


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
    return BackgroundResiduals(
        backoff=float(np.mean(means)), backsys=float(np.sqrt(max(0.0, variance - noise))), backscl=size, backn=n,
    )
