from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter


MIN_PATCHES = 16
MIN_FRACTION = 0.75
MAX_PATCHES = 1024
BOOTSTRAPS = 256


@dataclass(frozen=True)
class BackgroundResiduals:
    backoff: float | None = None
    backsys: float | None = None
    backrms: float | None = None
    backerr: float | None = None
    bkserr: float | None = None
    backnoi: float | None = None
    backscl: int = 64
    backn: int = 0
    backlag: int = 8
    backmeth: str = "PATCHCOV"
    backref: str = "SUBTRACT"

    def cards(self, include_missing: bool = False) -> dict:
        descriptions = (
            ("backoff", "[ADU] Mean residual sky; positive = under-subtracted"),
            ("backsys", "[ADU] Excess spatial RMS of patch mean residuals"),
            ("backrms", "[ADU] Spatial stddev of patch mean residuals"),
            ("backerr", "[ADU] Spatial block-bootstrap error on BACKOFF"),
            ("bkserr", "[ADU] Spatial block-bootstrap error on BACKSYS"),
            ("backnoi", "[ADU] Estimated stochastic RMS of patch means"),
            ("backscl", "[pixel] Residual measurement square width"),
            ("backn", "Number of valid residual sky patches measured"),
            ("backlag", "[pixel] Maximum per-axis noise covariance lag"),
            ("backmeth", "Residual QA method; conditional noise approximation"),
            ("backref", "Residual reference: MODEL, SUBTRACT, or COADD"),
        )
        return {key.upper(): (getattr(self, key), comment) for key, comment in descriptions
                if include_missing or getattr(self, key) is not None}


RESIDUAL_KEYS = tuple(BackgroundResiduals.__dataclass_fields__)


def clear_residual_cards(header) -> None:
    for key in RESIDUAL_KEYS:
        header.pop(key.upper(), None)


def _patch_noise(pixels, valid, design, lag):
    x = design[valid.ravel()]
    values = pixels[valid].astype(np.float64)
    gram = x.T @ x
    if np.linalg.matrix_rank(gram) < 3:
        return None
    inverse = np.linalg.inv(gram)
    residual = np.zeros(pixels.size, dtype=np.float64)
    residual[valid.ravel()] = values - x @ (inverse @ (x.T @ values))
    residual = residual.reshape(pixels.shape)
    width = 2 * lag + 1
    summed = uniform_filter(residual, size=width, mode="constant") * width**2
    quadratic = float(np.sum(residual * summed))
    basis = np.where(valid.ravel()[:, None], design, 0).reshape(*pixels.shape, 3)
    local = uniform_filter(basis, size=(width, width, 1), mode="constant") * width**2
    projection = float(np.sum((basis.reshape(-1, 3) @ inverse) * local.reshape(-1, 3)))
    n = int(valid.sum())
    return quadratic / (n * (n - projection)) if n > 2 * projection else None


def measure_background_residuals(data, exclude=None, coverage=None, box_size: int = 64) -> BackgroundResiduals:
    """Residual offset, excess patch variance, finite-lag noise covariance, and spatial bootstrap errors."""
    data = np.asarray(data)
    size = int(box_size)
    if data.ndim != 2 or size < 16:
        raise ValueError("Residual sky requires a 2D image and box_size >= 16")
    if exclude is not None and np.shape(exclude) != data.shape:
        raise ValueError("Source mask shape differs from the residual image")
    if coverage is not None and np.shape(coverage) != data.shape:
        raise ValueError("Coverage shape differs from the residual image")
    lag = min(8, size // 8)
    offset = 0
    ny, nx = (max(0, (length - offset) // size) for length in data.shape)
    empty = BackgroundResiduals(backscl=size, backlag=lag)
    if ny * nx < MIN_PATCHES:
        return empty
    yy, xx = np.mgrid[:size, :size] / size - 0.5
    design = np.column_stack((np.ones(size**2), xx.ravel(), yy.ravel()))
    means, noises, groups = [], [], []
    group_nx = (nx + 3) // 4
    group_ids = np.arange(((ny + 3) // 4) * group_nx)
    rng = np.random.default_rng(73519)
    rng.shuffle(group_ids)
    for group in group_ids:
        gy, gx = divmod(int(group), group_nx)
        for iy in range(gy * 4, min(gy * 4 + 4, ny)):
            for ix in range(gx * 4, min(gx * 4 + 4, nx)):
                sy = slice(offset + iy * size, offset + (iy + 1) * size)
                sx = slice(offset + ix * size, offset + (ix + 1) * size)
                pixels = data[sy, sx]
                valid = np.isfinite(pixels)
                if exclude is not None:
                    valid &= ~exclude[sy, sx]
                if coverage is not None:
                    valid &= coverage[sy, sx]
                if valid.sum() < MIN_FRACTION * size**2:
                    continue
                noise = _patch_noise(pixels, valid, design, lag)
                if noise is None or not np.isfinite(noise):
                    continue
                means.append(float(np.mean(pixels[valid], dtype=np.float64)))
                noises.append(noise)
                groups.append(group)
        if len(means) >= MAX_PATCHES:
            break
    n = len(means)
    if n < MIN_PATCHES:
        return BackgroundResiduals(backscl=size, backn=n, backlag=lag)
    means, noises, groups = np.asarray(means), np.asarray(noises), np.asarray(groups)
    variance = float(np.var(means, ddof=1))
    noise = max(0.0, float(np.mean(noises)))
    excess = max(0.0, variance - noise)
    unique, inverse = np.unique(groups, return_inverse=True)
    errors = (None, None)
    if len(unique) >= 8:
        count = np.bincount(inverse)
        sums = np.bincount(inverse, weights=means)
        squares = np.bincount(inverse, weights=means**2)
        noise_sums = np.bincount(inverse, weights=noises)
        draws = rng.integers(len(unique), size=(BOOTSTRAPS, len(unique)))
        total = count[draws].sum(axis=1)
        avg = sums[draws].sum(axis=1) / total
        var = (squares[draws].sum(axis=1) - total * avg**2) / (total - 1)
        sys = np.sqrt(np.maximum(0, var - np.maximum(0, noise_sums[draws].sum(axis=1) / total)))
        errors = float(np.std(avg, ddof=1)), float(np.std(sys, ddof=1))
    return BackgroundResiduals(
        backoff=float(np.mean(means)), backsys=float(np.sqrt(excess)), backrms=float(np.sqrt(variance)),
        backerr=errors[0], bkserr=errors[1], backnoi=float(np.sqrt(noise)), backscl=size, backn=n, backlag=lag,
    )
