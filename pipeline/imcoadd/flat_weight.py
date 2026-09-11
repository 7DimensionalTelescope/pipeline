import warnings

import numpy as np
from scipy.interpolate import LSQBivariateSpline
from scipy.optimize import nnls

WEIGHT_MODEL = "FLAT2_V1"
FLAT_KNOT_SPACING = 512
WEIGHT_QA_COMMENTS = {
    "WGTNPTS": "Detector fit: measured cells before clipping",
    "WGTNUSE": "Detector fit: cells retained after clipping",
    "WGTRMAD": "Detector fit: 1.4826*MAD(V/Vfit-1), dimensionless",
    "WGTERR": "Detector fit: formal B/C uncertainties available",
    "WGTBERR": "Detector fit: formal 1-sigma B error, independent cells",
    "WGTCERR": "Detector fit: formal 1-sigma C error, independent cells",
}


def smooth_flat_surface(flat: np.ndarray, exclude=None, block: int = 64) -> np.ndarray:
    """Positive illumination template from a coarse spline fitted to master-flat block medians."""
    from .weight import block_median

    flat = np.asarray(flat, dtype=np.float32)
    if flat.ndim != 2 or not flat.size or block < 1:
        raise ValueError("flat must be a nonempty 2D array and block must be positive")
    valid = np.isfinite(flat) & (flat > 0)
    if exclude is not None:
        valid &= ~exclude
    grid = block_median(flat, valid, block, max(1, block * block // 4))
    usable = np.isfinite(grid) & (grid > 0)
    if not usable.any():
        raise ValueError("no usable master-flat cells")
    h, w = flat.shape
    ny, nx = grid.shape
    if min(ny, nx) < 2:
        return np.full(flat.shape, np.median(grid[usable]), dtype=np.float32)
    y = (np.arange(ny) * block + np.minimum((np.arange(ny) + 1) * block, h) - 1) / 2
    x = (np.arange(nx) * block + np.minimum((np.arange(nx) + 1) * block, w) - 1) / 2
    yy, xx = np.meshgrid(y, x, indexing="ij")
    ky, kx = min(3, ny - 1), min(3, nx - 1)
    spacing = max(FLAT_KNOT_SPACING, 4 * block)
    ty = np.arange(spacing, y[-1] - block, spacing)
    tx = np.arange(spacing, x[-1] - block, spacing)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        spline = LSQBivariateSpline(
            yy[usable], xx[usable], np.log(grid[usable]), ty, tx,
            bbox=[0, h - 1, 0, w - 1], kx=ky, ky=kx,
        )
    surface = np.empty(flat.shape, dtype=np.float32)
    for y0 in range(0, h, 256):
        surface[y0:y0 + 256] = np.exp(spline(np.arange(y0, min(y0 + 256, h)), np.arange(w)))
    if not np.all(np.isfinite(surface) & (surface > 0)):
        raise ValueError("master-flat illumination fit is not finite and positive")
    return surface


def smooth_weight_surface(weight, flat, exclude=None, block: int = 64, logger=None, qa=None, image_name=None):
    """Robust nonnegative B/F + C/F² variance fit on measured cells; return weight and (B, C)."""
    from .weight import block_median

    if weight.shape != flat.shape or weight.ndim != 2 or block < 1:
        raise ValueError("weight and illumination template must have the same 2D shape; block must be positive")
    if not np.all(np.isfinite(flat) & (flat > 0)):
        raise ValueError("illumination template must be finite and positive")
    valid = np.isfinite(weight) & (weight > 0)
    if exclude is not None:
        valid &= ~exclude
    minimum = max(1, block * block // 4)
    wg = block_median(weight, valid, block, minimum)
    fg = block_median(flat, valid, block, minimum)
    usable = np.isfinite(wg) & (wg > 0) & np.isfinite(fg) & (fg > 0)
    if not usable.any():
        raise ValueError("no usable cells for the flat-based weight fit")
    variance = 1.0 / wg[usable]
    inverse_flat = 1.0 / fg[usable]
    design = np.column_stack((inverse_flat, inverse_flat * inverse_flat))
    relative_design = design / variance[:, None]
    scale = np.linalg.norm(relative_design, axis=0)
    keep = np.ones(variance.size, dtype=bool)
    for _ in range(8):
        coefficients = nnls(relative_design[keep] / scale, np.ones(int(keep.sum())))[0] / scale
        residual = variance / (design @ coefficients) - 1.0
        center = np.median(residual[keep])
        scatter = 1.4826 * np.median(np.abs(residual[keep] - center))
        selected = np.abs(residual - center) <= max(3 * scatter, 1e-6)
        if np.array_equal(selected, keep) or not selected.any():
            break
        keep = selected
    coefficients = nnls(relative_design[keep] / scale, np.ones(int(keep.sum())))[0] / scale
    residual = variance[keep] / (design[keep] @ coefficients) - 1.0
    scatter = float(1.4826 * np.median(np.abs(residual - np.median(residual))))
    b, c = coefficients
    surface = np.empty_like(flat, dtype=np.float32)
    for y0 in range(0, flat.shape[0], 256):
        f = flat[y0:y0 + 256].astype(np.float64)
        surface[y0:y0 + 256] = f * f / (b * f + c)
    if not np.all(np.isfinite(surface) & (surface > 0)):
        raise ValueError("flat-based weight fit is not finite and positive")
    errors, error_status = weight_fit_uncertainty(relative_design[keep], coefficients)
    if qa is not None:
        qa.update(WGTNPTS=int(variance.size), WGTNUSE=int(keep.sum()), WGTRMAD=scatter)
        qa.update(WGTERR=errors is not None)
        for key in ("WGTBERR", "WGTCERR"):
            qa.pop(key, None)
        if errors is not None:
            qa.update(WGTBERR=errors[0], WGTCERR=errors[1])
    if logger is not None:
        label = f" for {image_name}" if image_name else ""
        uncertainty = (
            f"formal 1-sigma errors B={errors[0]:.6g}, C={errors[1]:.6g}, corr(B,C)={errors[2]:.6f}"
            if errors is not None else f"B/C uncertainties unavailable: {error_status}"
        )
        logger.debug(
            f"Weight fit succeeded{label}: B/F+C/F^2, B={b:.8g}, C={c:.8g}; "
            f"{int(keep.sum())}/{variance.size} measured cells retained, "
            f"{int((~usable).sum())} unsupported cells excluded; relative scatter {scatter:.3%}; {uncertainty}"
        )
    return surface, (float(b), float(c))


def copy_weight_fit_header(source: str, target: str) -> None:
    """Carry detector-fit coefficients and QA into a resampled single-weight header."""
    from astropy.io import fits

    header = fits.getheader(source)
    if header.get("WGTMODEL") != WEIGHT_MODEL:
        return
    with fits.open(target, mode="update") as hdul:
        for key in ("WGTMODEL", "WGTB", "WGTC", *WEIGHT_QA_COMMENTS):
            if key in header:
                hdul[0].header[key] = (header[key], header.comments[key])
            else:
                hdul[0].header.pop(key, None)


def weight_fit_uncertainty(design: np.ndarray, coefficients: np.ndarray):
    """Formal WLS covariance conditional on retained independent cells and a fixed flat template."""
    if len(design) <= 2:
        return None, "insufficient residual degrees of freedom"
    scale = np.linalg.norm(design, axis=0)
    _, singular, vt = np.linalg.svd(design / scale, full_matrices=False)
    if singular[-1] <= np.finfo(float).eps * max(design.shape) * singular[0]:
        return None, "B and C are not separately identifiable"
    if np.any(coefficients <= 0):
        return None, "nonnegative coefficient bound active; symmetric errors invalid"
    inverse = (vt.T / singular) / scale[:, None]
    geometry = inverse @ inverse.T
    residual = 1.0 - design @ coefficients
    variance = float(residual @ residual / (len(design) - 2))
    errors = np.sqrt(np.diag(geometry) * variance)
    correlation = float(geometry[0, 1] / np.sqrt(geometry[0, 0] * geometry[1, 1]))
    if not np.all(np.isfinite(errors)) or not np.isfinite(correlation):
        return None, "nonfinite covariance"
    return (float(errors[0]), float(errors[1]), float(np.clip(correlation, -1.0, 1.0))), "OK"
