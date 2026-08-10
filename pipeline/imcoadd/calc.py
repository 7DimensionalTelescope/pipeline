"""Pixel-level arithmetic on SWarp-resampled FITS images.

All functions here assume the input frames share the same pixel scale and
WCS projection (same CRVAL / CD matrix), differing only in CRPIX.  Alignment
is therefore a pure integer pixel shift derived from the CRPIX difference —
no reprojection is performed.
"""

import os
import time
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from ..utils import time_diff_in_seconds, add_suffix
from ..services.logger import Logger
from .utils import determine_size, build_coadd_wcs_header

# SWarp resampling turns exact-zero weights into float dust (~1e-15 measured; physical
# weights are >=1e-4), so hole exclusion must test against this instead of 0.
WEIGHT_EPS = 1e-9


# Var(median)/Var(mean) for n Gaussian samples, from the order-statistic density
# n!/(m!m!) F^m (1-F)^m f  with m=(n-1)/2. Approaches pi/2 (Kendall & Stuart Vol.1) but is
# well below it for the frame counts a coadd actually has, so the limit is not usable.
_MEDIAN_VAR_RATIO = {
    1: 1.0, 2: 1.0, 3: 1.3460, 5: 1.4342, 7: 1.4731, 9: 1.4949, 11: 1.5088,
    15: 1.5254, 21: 1.5385, 31: 1.5489, 51: 1.5575, 101: 1.5641, 201: 1.5674,
}  # fmt: skip


def median_variance_ratio(n: float) -> float:
    """How much noisier a median of ``n`` frames is than their mean."""
    knots = np.array(sorted(_MEDIAN_VAR_RATIO), dtype=float)
    values = np.array([_MEDIAN_VAR_RATIO[int(k)] for k in knots])
    if n <= knots[0]:
        return float(values[0])
    if n >= knots[-1]:
        return float(np.pi / 2)
    return float(np.interp(1.0 / n, (1.0 / knots)[::-1], values[::-1]))


def _median_penalty(counts) -> "np.ndarray":
    """Vectorized median_variance_ratio over a per-pixel contributor-count array."""
    knots = np.array(sorted(_MEDIAN_VAR_RATIO), dtype=float)
    values = np.array([_MEDIAN_VAR_RATIO[int(k)] for k in knots])
    n = np.maximum(counts.astype(np.float64), 1.0)
    return np.interp(1.0 / n, (1.0 / knots)[::-1], values[::-1])


def coadd_effective_egain(gain_terms, mode: str = "mean", n_eff: float | None = None) -> float | None:
    """Effective gain of the coadd: ``(sum w)^2 / sum(w^2 / g)``, divided by the median
    penalty when ``mode`` is median.

    ``gain_terms`` is ``(weight, EGAIN/FLXSCALE)`` per contributing frame; ``n_eff`` is the
    typical number of frames behind an output pixel (footprint mean, or Kish
    ``(sum w)^2/sum w^2`` for weighted stacks). The naive ``sum(EGAIN/FLXSCALE)`` equals
    this only when every term is equal, which FLXSCALE alone breaks.
    """
    terms = [(w, g) for w, g in gain_terms if w > 0 and g and np.isfinite(g)]
    if not terms:
        return None
    w = np.array([t[0] for t in terms], dtype=float)
    g = np.array([t[1] for t in terms], dtype=float)
    gain = w.sum() ** 2 / np.sum(w**2 / g)
    if mode == "median":
        gain /= median_variance_ratio(n_eff if n_eff else len(terms))
    return float(gain)


def mean_coadd_numpy(
    input_images: list[str],
    output_path: str,
    coadd_header: fits.Header,
    weights: list[str] | list[float] | None = None,
    weight_output: str | None = None,
    footprint_output: str | None = None,
    masks: list[str] | None = None,
    flxscales: list[float] | bool | None = None,
    match_swarp_size: bool = True,
    var_maps: list[str] | None = None,
    logger: Logger | None = None,
) -> str:
    """Per-pixel flux-scaled mean coadd (simple or inverse-variance weighted).

    Works on SWarp-resampled images that are centered differently.

    ``footprint_output`` receives the count of frames that actually contributed to each
    output pixel -- taken from the same per-frame validity the combine uses, so it cannot
    disagree with the coadd. ``masks`` optionally narrows that validity (bad pixels).
    """
    st = time.time()
    backend = "weighted" if weights is not None else "simple"
    grid = "swarp grid" if match_swarp_size else "tight bbox"
    if logger is not None:
        logger.info(f"Start in-memory numpy coaddition ({backend} mean, {grid})")
    if weights is not None and len(weights) != len(input_images):
        raise ValueError(f"weights ({len(weights)}) and input_images ({len(input_images)}) length mismatch")
    if isinstance(flxscales, list) and len(flxscales) != len(input_images):
        raise ValueError(f"flxscales ({len(flxscales)}) and input_images ({len(input_images)}) length mismatch")

    # Flux-scaling source (logged once): False disables; an explicit list is the
    # snapshot source of truth; None falls back to each file's FLXSCALE header.
    scale_mode = (
        "disabled" if flxscales is False else ("from in-memory values" if flxscales is not None else "from FLXSCALE headers")
    )
    if logger is not None:
        logger.info(f"Flux scaling during coadd: {scale_mode}")

    target_w, target_h, target_cx, target_cy, x0, y0, shapes = determine_size(input_images, match_swarp_size)

    sum_arr = np.zeros((target_h, target_w), dtype=np.float64)
    norm_arr = np.zeros((target_h, target_w), dtype=np.float64 if weights is not None else np.int32)
    count_arr = np.zeros((target_h, target_w), dtype=np.int32)
    gain_denom = np.zeros((target_h, target_w), dtype=np.float64)  # sum w^2/g for the gain map
    gain_terms = []  # (typical weight, EGAIN/FLXSCALE) per contributing image
    all_egain = True
    # scalar/no weighting + per-frame variance maps: propagate them through the same
    # weighting the sci pixels get, so the output weight is the coadd's true 1/sigma^2
    propagate = var_maps is not None and (weights is None or not isinstance(weights[0], str))
    var_den = np.zeros((target_h, target_w), dtype=np.float64) if propagate else None
    for i, f in enumerate(input_images):
        with fits.open(f) as hdul:
            a = hdul[0].data.astype(np.float64)
            egain = hdul[0].header.get("EGAIN")
            # False disables; explicit list = snapshot source of truth; None = file FLXSCALE.
            if flxscales is False:
                flxscale = 1.0
            elif flxscales is not None:
                flxscale = flxscales[i]
            else:
                flxscale = hdul[0].header.get("FLXSCALE", 1.0)
        flxscale = 1.0 if flxscale is None else flxscale
        h, w = a.shape

        tx0 = max(0, x0[i]); tx1 = min(target_w, x0[i] + w)  # fmt: skip
        ty0 = max(0, y0[i]); ty1 = min(target_h, y0[i] + h)  # fmt: skip
        if tx1 <= tx0 or ty1 <= ty0:
            if logger is not None:
                logger.debug(f"{os.path.basename(f)}: no overlap with target grid; skipped")
            continue
        sx0 = tx0 - x0[i]; sx1 = tx1 - x0[i]  # fmt: skip
        sy0 = ty0 - y0[i]; sy1 = ty1 - y0[i]  # fmt: skip
        src = a[sy0:sy1, sx0:sx1]
        valid = np.isfinite(src) & (src != 0.0)
        if masks is not None:
            with fits.open(masks[i], memmap=True) as mh:
                valid &= mh[0].data[sy0:sy1, sx0:sx1] > WEIGHT_EPS

        if weights is None:
            sum_arr[ty0:ty1, tx0:tx1] += np.where(valid, src * flxscale, 0.0)
            norm_arr[ty0:ty1, tx0:tx1] += valid
            if egain is not None:
                gain_terms.append((1.0, float(egain) / flxscale))
                gain_denom[ty0:ty1, tx0:tx1] += np.where(valid, flxscale / float(egain), 0.0)
            else:
                all_egain = False
        else:
            # SWarp's MAP_WEIGHT = 1/variance of the raw resampled data
            # (RESCALE_WEIGHTS = N). After we multiply data by FLXSCALE its
            # variance scales by FLXSCALE^2, so use w/FLXSCALE^2 as the
            # inverse-variance weight of the flux-normalised image.
            if isinstance(weights[i], str):
                with fits.open(weights[i]) as hdul:
                    w_full = hdul[0].data.astype(np.float64)
                w_full[w_full < WEIGHT_EPS] = 0.0
                w_eff = w_full[sy0:sy1, sx0:sx1] / (flxscale * flxscale)
                valid &= w_eff > 0
            else:
                # scalar per-image weight (e.g. 1/SKYSIG^2); same FLXSCALE^2 rule
                w_eff = float(weights[i]) / (flxscale * flxscale)
            sum_arr[ty0:ty1, tx0:tx1] += np.where(valid, w_eff * src * flxscale, 0.0)
            norm_arr[ty0:ty1, tx0:tx1] += np.where(valid, w_eff, 0.0)
            if egain is not None and valid.any():
                w_typ = float(w_eff[valid].mean()) if isinstance(weights[i], str) else w_eff
                gain_terms.append((w_typ, float(egain) / flxscale))
                gain_denom[ty0:ty1, tx0:tx1] += np.where(valid, w_eff * w_eff * flxscale / float(egain), 0.0)
            elif egain is None:
                all_egain = False
        # counted after the weight test, so the footprint is the frames the coadd used
        count_arr[ty0:ty1, tx0:tx1] += valid
        if propagate:
            with fits.open(var_maps[i], memmap=True) as vh:
                vm = vh[0].data[sy0:sy1, sx0:sx1].astype(np.float64)
            vm[vm < WEIGHT_EPS] = 0.0
            # combine weight on the flux-normalized scale (matches w_eff/norm_arr);
            # sigma_norm^2 = flxscale^2 / w_map
            se = 1.0 if weights is None else float(weights[i]) / (flxscale * flxscale)
            ok = valid & (vm > 0)
            var_den[ty0:ty1, tx0:tx1] += np.where(ok, se * se * flxscale * flxscale / np.where(ok, vm, 1.0), 0.0)

    coadd = np.where(norm_arr > 0, sum_arr / np.where(norm_arr > 0, norm_arr, 1), np.nan).astype(np.float32)
    if propagate:
        weight_map_out = np.where(var_den > 0, norm_arr.astype(np.float64) ** 2 / np.where(var_den > 0, var_den, 1), 0.0)
    else:
        weight_map_out = norm_arr

    out_header = build_coadd_wcs_header(input_images[0], target_cx, target_cy, coadd_header)
    covered = count_arr > 0
    n_eff = float(count_arr[covered].mean()) if covered.any() else None
    if all_egain and covered.any() and (gain_denom[covered] > 0).all():
        # footprint-exact: median of the per-pixel effective gain map norm^2 / sum(w^2/g),
        # well defined under any coverage; the frame-level formula needs uniform coverage
        effective = float(np.median(norm_arr[covered] ** 2 / gain_denom[covered]))
    else:
        effective = coadd_effective_egain(gain_terms, mode="mean", n_eff=n_eff)
    if effective is not None:
        # value-only: InputHeaderSet.coadd_header owns the card's comment
        out_header["EGAIN"] = effective
        if not out_header.comments["EGAIN"]:
            out_header.comments["EGAIN"] = "Effective EGAIN for coadded image (e-/ADU)"
        if logger is not None and n_eff is not None:
            logger.debug(f"Mean coadd EGAIN {effective:.4f} ({backend}, n_eff {n_eff:.1f} frames/pixel)")
    fits.writeto(output_path, coadd, header=out_header, overwrite=True)

    # propagated: (sum s)^2 / sum(s^2 sigma_p^2) -- the coadd's own per-pixel inverse
    # variance; otherwise summed inverse variance (pixel-wise) or frame count (simple).
    weight_out = weight_output or add_suffix(output_path, "weight")
    fits.writeto(weight_out, weight_map_out.astype(np.float32), header=out_header, overwrite=True)
    footprint_out = footprint_output or add_suffix(output_path, "footprint")
    fits.writeto(footprint_out, count_arr.astype(np.int16), header=out_header, overwrite=True)
    if logger is not None:
        logger.debug(f"Wrote coadd weight map ({backend}): {weight_out}")
        logger.debug(f"Wrote coadd footprint (max {int(count_arr.max())} frames): {footprint_out}")

    if logger is not None:
        logger.info(f"Numpy coaddition completed in {time_diff_in_seconds(st)} seconds")
    return output_path


CLIP_KAPPA = 3.0  # clip threshold in units of each sample's own expected noise
CLIP_FRAC = 0.1  # fractional tolerance: bright cores scatter with seeing, not with sky noise


def clipped_mean_coadd_numpy(
    input_images: list[str],
    output_path: str,
    coadd_header: fits.Header,
    weights: list[str] | list[float] | None = None,
    weight_output: str | None = None,
    footprint_output: str | None = None,
    masks: list[str] | None = None,
    flxscales: list[float] | bool | None = None,
    match_swarp_size: bool = True,
    kappa: float = CLIP_KAPPA,
    reserved_bytes: int = 0,
    var_maps: list[str] | None = None,
    logger: Logger | None = None,
) -> str:
    """Median-centered kappa-sigma clipped weighted mean (Gruen+2014-style).

    Pass 1 builds a median coadd (the existing strip machinery) as the robust center;
    pass 2 streams the frames once more, keeping samples with
    ``|x - c| <= kappa/sqrt(w) + CLIP_FRAC*|c|`` -- each sample judged against its OWN
    expected noise, with the fractional term protecting bright cores whose frame-to-frame
    scatter is seeing-driven, not sky-noise-driven -- then weighted-means the survivors.
    An empirical-scatter criterion cannot do this job: a single outlier among n frames
    caps at z = sqrt(n-1), inside kappa=3 for n <= 9. Requires weights.
    """
    if weights is None:
        raise ValueError("clipped mean needs weights (set coadd_weighting to 'global' or 'pixel-wise')")
    st = time.time()
    if logger is not None:
        logger.info(f"Start in-memory numpy coaddition (clipped weighted mean, kappa={kappa:g})")
    if len(weights) != len(input_images):
        raise ValueError(f"weights ({len(weights)}) and input_images ({len(input_images)}) length mismatch")
    if isinstance(flxscales, list) and len(flxscales) != len(input_images):
        raise ValueError(f"flxscales ({len(flxscales)}) and input_images ({len(input_images)}) length mismatch")

    # pass 1: robust center from the proven median machinery, into temporaries
    tmp_center = add_suffix(output_path, "clipcenter")
    median_coadd_numpy(
        input_images, tmp_center, coadd_header, weights=weights,
        weight_output=add_suffix(tmp_center, "weight"), footprint_output=add_suffix(tmp_center, "footprint"),
        masks=masks, flxscales=flxscales, match_swarp_size=match_swarp_size,
        chunk_h=None, reserved_bytes=reserved_bytes, logger=logger,
    )
    center = fits.getdata(tmp_center).astype(np.float64)
    for t in (tmp_center, add_suffix(tmp_center, "weight"), add_suffix(tmp_center, "footprint")):
        try:
            os.remove(t)
        except OSError:
            pass
    center = np.where(np.isfinite(center), center, 0.0)  # no coverage: nothing survives anyway

    target_w, target_h, target_cx, target_cy, x0, y0, shapes = determine_size(input_images, match_swarp_size)

    sum_arr = np.zeros((target_h, target_w), dtype=np.float64)
    norm_arr = np.zeros((target_h, target_w), dtype=np.float64)
    count_arr = np.zeros((target_h, target_w), dtype=np.int32)
    gain_denom = np.zeros((target_h, target_w), dtype=np.float64)
    gain_terms = []
    all_egain = True
    propagate = var_maps is not None and not isinstance(weights[0], str)
    var_den = np.zeros((target_h, target_w), dtype=np.float64) if propagate else None
    n_clipped = n_total = 0
    for i, f in enumerate(input_images):
        with fits.open(f) as hdul:
            a = hdul[0].data.astype(np.float64)
            egain = hdul[0].header.get("EGAIN")
            if flxscales is False:
                flxscale = 1.0
            elif flxscales is not None:
                flxscale = flxscales[i]
            else:
                flxscale = hdul[0].header.get("FLXSCALE", 1.0)
        flxscale = 1.0 if flxscale is None else flxscale
        h, w = a.shape
        tx0 = max(0, x0[i]); tx1 = min(target_w, x0[i] + w)  # fmt: skip
        ty0 = max(0, y0[i]); ty1 = min(target_h, y0[i] + h)  # fmt: skip
        if tx1 <= tx0 or ty1 <= ty0:
            continue
        sx0 = tx0 - x0[i]; sx1 = tx1 - x0[i]  # fmt: skip
        sy0 = ty0 - y0[i]; sy1 = ty1 - y0[i]  # fmt: skip
        sl = (slice(ty0, ty1), slice(tx0, tx1))
        raw = a[sy0:sy1, sx0:sx1]
        valid = np.isfinite(raw) & (raw != 0.0)
        if masks is not None:
            with fits.open(masks[i], memmap=True) as mh:
                valid &= mh[0].data[sy0:sy1, sx0:sx1] > WEIGHT_EPS
        if isinstance(weights[i], str):
            with fits.open(weights[i]) as hdul:
                w_full = hdul[0].data.astype(np.float64)
            w_full[w_full < WEIGHT_EPS] = 0.0
            w_eff = w_full[sy0:sy1, sx0:sx1] / (flxscale * flxscale)
            valid &= w_eff > 0
        else:
            w_eff = float(weights[i]) / (flxscale * flxscale)
        src = raw * flxscale

        c = center[sl]
        sigma_i = 1.0 / np.sqrt(np.where(valid, w_eff, 1.0) if not np.isscalar(w_eff) else w_eff)
        keep = valid & (np.abs(src - c) <= kappa * sigma_i + CLIP_FRAC * np.abs(c))
        n_total += int(valid.sum())
        n_clipped += int(valid.sum() - keep.sum())

        wv = np.where(keep, w_eff, 0.0)
        sum_arr[sl] += wv * np.where(keep, src, 0.0)
        norm_arr[sl] += wv
        count_arr[sl] += keep
        if egain is not None and keep.any():
            w_typ = float(w_eff) if np.isscalar(w_eff) else float(w_eff[keep].mean())
            gain_terms.append((w_typ, float(egain) / flxscale))
            gain_denom[sl] += np.where(keep, wv * wv * flxscale / float(egain), 0.0)
        elif egain is None:
            all_egain = False
        if propagate:
            with fits.open(var_maps[i], memmap=True) as vh:
                vm = vh[0].data[sy0:sy1, sx0:sx1].astype(np.float64)
            vm[vm < WEIGHT_EPS] = 0.0
            se = float(weights[i]) / (flxscale * flxscale)
            ok = keep & (vm > 0)
            var_den[sl] += np.where(ok, se * se * flxscale * flxscale / np.where(ok, vm, 1.0), 0.0)

    coadd = np.where(norm_arr > 0, sum_arr / np.where(norm_arr > 0, norm_arr, 1), np.nan).astype(np.float32)
    if logger is not None:
        logger.info(f"Clipped {n_clipped} of {n_total} samples ({100 * n_clipped / max(n_total, 1):.3f}%)")

    out_header = build_coadd_wcs_header(input_images[0], target_cx, target_cy, coadd_header)
    covered = count_arr > 0
    n_eff = float(count_arr[covered].mean()) if covered.any() else None
    if all_egain and covered.any() and (gain_denom[covered] > 0).all():
        effective = float(np.median(norm_arr[covered] ** 2 / gain_denom[covered]))
    else:
        effective = coadd_effective_egain(gain_terms, mode="mean", n_eff=n_eff)
    if effective is not None:
        # value-only: InputHeaderSet.coadd_header owns the card's comment
        out_header["EGAIN"] = effective
        if not out_header.comments["EGAIN"]:
            out_header.comments["EGAIN"] = "Effective EGAIN for coadded image (e-/ADU)"
    fits.writeto(output_path, coadd, header=out_header, overwrite=True)
    # survivors form a weighted mean: propagated (sum s)^2/sum(s^2 sigma^2), no penalty
    if propagate:
        weight_map_out = np.where(var_den > 0, norm_arr.astype(np.float64) ** 2 / np.where(var_den > 0, var_den, 1), 0.0)
    else:
        weight_map_out = norm_arr.astype(np.float64)
    weight_out = weight_output or add_suffix(output_path, "weight")
    fits.writeto(weight_out, weight_map_out.astype(np.float32), header=out_header, overwrite=True)
    footprint_out = footprint_output or add_suffix(output_path, "footprint")
    fits.writeto(footprint_out, count_arr.astype(np.int16), header=out_header, overwrite=True)
    if logger is not None:
        logger.info(f"Numpy clipped-mean coaddition completed in {time_diff_in_seconds(st)} seconds")
    return output_path


def _auto_chunk_h(n_images: int, width: int, height: int, budget_fraction: float = 0.3,
                  floor: int = 128, reserved_bytes: int = 0, logger: Logger | None = None) -> int:
    """Strip height from idle memory: strip count scales the NFS slice round-trips, so
    RAM buys taller strips and directly cuts the latency-bound I/O. ``reserved_bytes``
    subtracts other combines' leased stacks (services.combine_lock) so concurrent
    combines cannot each size against the same free memory."""
    try:
        avail = int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable")).split()[1]) * 1024
    except Exception:
        return floor
    avail = max(0, avail - int(reserved_bytes))
    chunk = int(budget_fraction * avail / (n_images * width * 4))
    chunk = max(floor, min(chunk, height))
    if logger is not None:
        n_strips = -(-height // chunk)
        logger.info(f"Median combine: chunk_h={chunk} ({n_strips} strips, "
                    f"~{n_images * chunk * width * 4 / 1e9:.0f} GB strip stack)")
    return chunk


def median_coadd_numpy(
    input_images: list[str],
    output_path: str,
    coadd_header: fits.Header,
    weights: list[str] | list[float] | None = None,
    weight_output: str | None = None,
    footprint_output: str | None = None,
    masks: list[str] | None = None,
    flxscales: list[float] | bool | None = None,
    match_swarp_size: bool = True,
    chunk_h: int = 128,
    reserved_bytes: int = 0,
    var_maps: list[str] | None = None,
    logger: Logger | None = None,
) -> str:
    """Per-pixel flux-scaled median coadd.

    Works on SWarp-resampled images that are centered differently.

    ``weights`` never enters the median itself -- it only builds the companion
    weight map, summed over the pixels that did contribute, so a median coadd
    ships the same companion a mean coadd (and the legacy SWarp pass) does.

    Peak memory bounded by chunk_h.
    """
    st = time.time()
    size = "swarp FOV size" if match_swarp_size else "tight bbox"
    if logger is not None:
        logger.info(f"Start in-memory numpy coaddition (median, {size})")
    if weights is not None and len(weights) != len(input_images):
        raise ValueError(f"weights ({len(weights)}) and input_images ({len(input_images)}) length mismatch")
    if isinstance(flxscales, list) and len(flxscales) != len(input_images):
        raise ValueError(f"flxscales ({len(flxscales)}) and input_images ({len(input_images)}) length mismatch")

    target_w, target_h, target_cx, target_cy, x0, y0, shapes = determine_size(input_images, match_swarp_size)

    if chunk_h is None:
        chunk_h = _auto_chunk_h(len(input_images), target_w, target_h, reserved_bytes=reserved_bytes, logger=logger)

    handles = [fits.open(f, memmap=True) for f in input_images]
    # Flux-scaling source (logged once): False disables; explicit list = snapshot
    # source of truth; None falls back to each file's FLXSCALE header.
    if flxscales is False:
        scale_mode = "disabled"
        flxscales = np.ones(len(input_images), dtype=np.float32)
    else:
        if flxscales is None:
            scale_mode = "from FLXSCALE headers"
            flxscales = [h[0].header.get("FLXSCALE", 1.0) for h in handles]
        else:
            scale_mode = "from in-memory values"
        flxscales = np.array([1.0 if f is None else f for f in flxscales], dtype=np.float32)
    if logger is not None:
        logger.info(f"Flux scaling during coadd: {scale_mode}")

    coadd = np.full((target_h, target_w), np.nan, dtype=np.float32)
    count_arr = np.zeros((target_h, target_w), dtype=np.int32)
    egains = [fits.getheader(f).get("EGAIN") for f in input_images]
    gain_terms = [(1.0, float(e) / flxscales[i]) for i, e in enumerate(egains) if e is not None]
    all_egain = all(e is not None for e in egains)
    ginv_arr = np.zeros((target_h, target_w), dtype=np.float64)  # sum(1/g) over contributing frames
    pixel_weights = weights is not None and len(weights) > 0 and isinstance(weights[0], str)
    propagate = var_maps is not None and not pixel_weights
    have_sigma = propagate or pixel_weights or weights is not None
    var_den = np.zeros((target_h, target_w), dtype=np.float64) if have_sigma else None
    whandles = [fits.open(f, memmap=True) for f in weights] if pixel_weights else None
    mhandles = [fits.open(f, memmap=True) for f in masks] if masks is not None else None
    try:
        for ys in range(0, target_h, chunk_h):
            ye = min(target_h, ys + chunk_h)
            # (N, strip_h, target_w) stack; NaN-init so np.nanmedian ignores unfilled cells.
            stack = np.full((len(input_images), ye - ys, target_w), np.nan, dtype=np.float32)
            for i, hdul in enumerate(handles):
                h, w = shapes[i]
                tx0 = max(0, x0[i]); tx1 = min(target_w, x0[i] + w)  # fmt: skip
                ty0 = max(ys, max(0, y0[i])); ty1 = min(ye, min(target_h, y0[i] + h))  # fmt: skip
                if tx1 <= tx0 or ty1 <= ty0:
                    continue
                sx0 = tx0 - x0[i]; sx1 = tx1 - x0[i]  # fmt: skip
                sy0 = ty0 - y0[i]; sy1 = ty1 - y0[i]  # fmt: skip
                src = hdul[0].data[sy0:sy1, sx0:sx1].astype(np.float32) * flxscales[i]
                src[(src == 0.0) | ~np.isfinite(src)] = np.nan
                if mhandles is not None:
                    src[mhandles[i][0].data[sy0:sy1, sx0:sx1] <= WEIGHT_EPS] = np.nan
                if whandles is not None:
                    # w is the inverse variance of the raw resampled data; the median is
                    # taken on flux-normalised pixels, whose variance scales by FLXSCALE^2
                    w = whandles[i][0].data[sy0:sy1, sx0:sx1].astype(np.float32)
                    w[w < WEIGHT_EPS] = 0.0
                    w /= flxscales[i] * flxscales[i]
                    # zero weight (interpolated/bad pixel) never enters the stack,
                    # matching the mean path's w_eff > 0 test; footprint follows
                    src[w <= 0] = np.nan
                stack[i, ty0 - ys : ty1 - ys, tx0:tx1] = src
                contributed = np.isfinite(src)
                count_arr[ty0:ty1, tx0:tx1] += contributed
                if egains[i] is not None:
                    ginv_arr[ty0:ty1, tx0:tx1] += np.where(contributed, flxscales[i] / float(egains[i]), 0.0)
                # the median vote is UNWEIGHTED, whatever coadd_weighting says: its weight
                # product is the equal-vote mean's variance, sum sigma_i^2 of contributors
                # (user decision 2026-08-10) -- sigma from the best available source
                if propagate:
                    with fits.open(var_maps[i], memmap=True) as vh:
                        vm = vh[0].data[sy0:sy1, sx0:sx1].astype(np.float64)
                    vm[vm < WEIGHT_EPS] = 0.0
                    fx = flxscales[i]
                    ok = contributed & (vm > 0)
                    var_den[ty0:ty1, tx0:tx1] += np.where(ok, fx * fx / np.where(ok, vm, 1.0), 0.0)
                elif whandles is not None:
                    ok = contributed & (w > 0)
                    var_den[ty0:ty1, tx0:tx1] += np.where(ok, 1.0 / np.where(ok, w, 1.0), 0.0)
                elif weights is not None:
                    var_den[ty0:ty1, tx0:tx1] += np.where(
                        contributed, (flxscales[i] * flxscales[i]) / float(weights[i]), 0.0
                    )
            coadd[ys:ye, :] = np.nanmedian(stack, axis=0)
    finally:
        for hdul in handles:
            hdul.close()
        for hdul in (whandles or []) + (mhandles or []):
            hdul.close()

    out_header = build_coadd_wcs_header(input_images[0], target_cx, target_cy, coadd_header)
    covered = count_arr > 0
    n_eff = float(count_arr[covered].mean()) if covered.any() else None
    if all_egain and covered.any():
        # per-pixel: n^2/sum(1/g) with the median penalty at that pixel's own n, then the
        # covered median -- exact under any coverage, no uniform-rejection assumption
        ratios = np.array([median_variance_ratio(n) for n in range(int(count_arr.max()) + 1)])
        n_pix = count_arr[covered]
        gmap = (n_pix.astype(np.float64) ** 2 / ginv_arr[covered]) / ratios[n_pix]
        effective = float(np.median(gmap))
    else:
        effective = coadd_effective_egain(gain_terms, mode="median", n_eff=n_eff)
    if effective is not None:
        # value-only: InputHeaderSet.coadd_header owns the card's comment
        out_header["EGAIN"] = effective
        if not out_header.comments["EGAIN"]:
            out_header.comments["EGAIN"] = "Effective EGAIN for coadded image (e-/ADU)"
        if logger is not None:
            logger.debug(f"Median coadd EGAIN {effective:.4f} (n_eff {n_eff:.1f} frames/pixel)")
    fits.writeto(output_path, coadd, header=out_header, overwrite=True)

    # equal-vote mean variance (n^2 / sum sigma_i^2) with the order-statistics penalty
    # folded in, so 1/sqrt(weight) is THE per-pixel sigma of this median image. The
    # penalty is exact for homogeneous stacks and conservative for heterogeneous ones.
    if have_sigma:
        base_w = np.where(
            var_den > 0, count_arr.astype(np.float64) ** 2 / np.where(var_den > 0, var_den, 1), 0.0
        )
    else:
        base_w = count_arr.astype(np.float64)  # no sigma source: frame count
    weight_map_out = base_w / _median_penalty(count_arr)
    weight_out = weight_output or add_suffix(output_path, "weight")
    fits.writeto(weight_out, weight_map_out.astype(np.float32), header=out_header, overwrite=True)
    footprint_out = footprint_output or add_suffix(output_path, "footprint")
    fits.writeto(footprint_out, count_arr.astype(np.int16), header=out_header, overwrite=True)
    if logger is not None:
        logger.debug(f"Wrote coadd footprint (max {int(count_arr.max())} frames): {footprint_out}")
        backend = "summed inverse variance" if weights is not None else "frame count"
        logger.debug(f"Wrote coadd weight map ({backend}): {weight_out}")
        logger.info(f"Numpy median coaddition completed in {time_diff_in_seconds(st)} seconds")
    return output_path


def accumulate_weight_maps(
    weight_images: list[str],
    output_path: str,
    coadd_header: fits.Header,
    match_swarp_size: bool = True,
    logger: Logger | None = None,
) -> str:
    """Sum SWarp-resampled weight/mask planes onto the coadd grid, one frame at a time."""
    st = time.time()
    if logger is not None:
        logger.info(f"Accumulating {len(weight_images)} resampled maps into {os.path.basename(output_path)}")

    target_w, target_h, target_cx, target_cy, x0, y0, shapes = determine_size(weight_images, match_swarp_size)

    total = np.zeros((target_h, target_w), dtype=np.float32)
    for i, f in enumerate(weight_images):
        with fits.open(f, memmap=True) as hdul:
            a = hdul[0].data
            h, w = a.shape
            tx0 = max(0, x0[i]); tx1 = min(target_w, x0[i] + w)  # fmt: skip
            ty0 = max(0, y0[i]); ty1 = min(target_h, y0[i] + h)  # fmt: skip
            if tx1 <= tx0 or ty1 <= ty0:
                continue
            total[ty0:ty1, tx0:tx1] += a[ty0 - y0[i] : ty1 - y0[i], tx0 - x0[i] : tx1 - x0[i]]

    out_header = build_coadd_wcs_header(weight_images[0], target_cx, target_cy, coadd_header)
    fits.writeto(output_path, total, header=out_header, overwrite=True)
    if logger is not None:
        logger.info(f"Accumulation completed in {time_diff_in_seconds(st)} seconds")
    return output_path


def subtract_images(
    image_a: str,
    image_b: str,
    output_path: str,
    flxscale: bool = False,
    zp_key: str = "ZP_AUTO",
    logger: Logger | None = None,
    overwrite=True,
) -> str:
    """Subtract *image_b* from *image_a*, aligned by their CRPIX offsets.

    image_a is the reference for flux scaling and registration.
    """

    def chatter(msg: str, level: str = "debug"):
        if logger is not None:
            return getattr(logger, level)(msg)
        else:
            print(f"[subtract_images:{level.upper()}] {msg}")

    chatter("Start image subtraction")

    with fits.open(image_a) as ha:
        data_a = ha[0].data.astype(np.float32)
        hdr_a = ha[0].header.copy()
    with fits.open(image_b) as hb:
        data_b = hb[0].data.astype(np.float32)
        hdr_b = hb[0].header.copy()

    # Match image_b to image_a's level (image_a is the reference, fa=1).
    # Precedence: disabled -> ZP (zp_key) -> FLXSCALE -> none.
    if flxscale is False:
        flxscale_factor, scale_mode = 1.0, "disabled"
    elif hdr_a.get(zp_key) is not None and hdr_b.get(zp_key) is not None:
        flxscale_factor = 10 ** (0.4 * (float(hdr_a[zp_key]) - float(hdr_b[zp_key])))
        scale_mode = f"ZP ({zp_key})"
    elif (flxscale_a := hdr_a.get("FLXSCALE")) is not None:
        flxscale_factor = float(hdr_b.get("FLXSCALE", 1.0)) / float(flxscale_a)
        scale_mode = "FLXSCALE"
    else:
        flxscale_factor, scale_mode = 1.0, "none"
        chatter(f"Cannot determine flux scaling: neither {zp_key} nor FLXSCALE", level="warning")
    chatter(f"Flux scaling ({scale_mode}): image_b x {flxscale_factor:.4f} -> image_a level")

    crpix_a = np.array([hdr_a["CRPIX1"], hdr_a["CRPIX2"]], dtype=float)
    crpix_b = np.array([hdr_b["CRPIX1"], hdr_b["CRPIX2"]], dtype=float)

    h_a, w_a = data_a.shape
    h_b, w_b = data_b.shape

    # Integer pixel offset: image_b's pixel (0,0) [0-indexed] sits at column
    # dx, row dy in image_a's coordinate system.
    dx = int(np.rint(crpix_a[0] - crpix_b[0]))
    dy = int(np.rint(crpix_a[1] - crpix_b[1]))

    # image_a is the reference grid; no a-pixel is cropped. image_b is shifted
    # onto image_a's grid over their overlap; non-overlap stays NaN.
    diff = np.full((h_a, w_a), np.nan, dtype=np.float32)

    ax0 = max(0, dx);      ax1 = min(w_a, dx + w_b)  # fmt: skip
    ay0 = max(0, dy);      ay1 = min(h_a, dy + h_b)  # fmt: skip
    if ax1 > ax0 and ay1 > ay0:
        bx0 = ax0 - dx;  bx1 = ax1 - dx  # fmt: skip
        by0 = ay0 - dy;  by1 = ay1 - dy  # fmt: skip
        patch_a = data_a[ay0:ay1, ax0:ax1]
        patch_b = data_b[by0:by1, bx0:bx1]
        # NaN propagates through the subtraction; only SWarp 0.0 zero-padding
        # needs explicit masking.
        out = patch_a - flxscale_factor * patch_b
        out[(patch_a == 0.0) | (patch_b == 0.0)] = np.nan
        diff[ay0:ay1, ax0:ax1] = out

    out_header = WCS(hdr_a).to_header(relax=True)
    fits.writeto(output_path, diff, header=out_header, overwrite=overwrite)
    return output_path
