"""CuPy clipped weighted mean of frames in host memory; the device form of imcoadd.calc.clipped_mean_coadd_numpy."""

from __future__ import annotations

import numpy as np


def _put(cp, target, positions, sx0, sx1, sy0, value):
    """Write value at the (y, x) frame positions that fall inside the block, as ProjectedBadPixels.apply does."""
    if positions is None:
        return
    yy, xx = positions
    if not yy.size:
        return
    inside = (xx >= sx0) & (xx < sx1)
    target[cp.asarray(yy[inside] - sy0), cp.asarray(xx[inside] - sx0)] = value


def _median_center(cp, stack):
    """nanmedian over axis 0: for an even count (lower + upper middle) * 0.5 in float32; NaN where no finite sample."""
    srt = cp.sort(stack, axis=0)
    m = cp.count_nonzero(~cp.isnan(stack), axis=0).astype(cp.int32)
    lo = cp.take_along_axis(srt, cp.maximum((m - 1) // 2, 0)[None], axis=0)[0]
    hi = cp.take_along_axis(srt, (m // 2)[None], axis=0)[0]
    return cp.where(m == 0, cp.float32(cp.nan), cp.where(m % 2 == 1, lo, (lo + hi) * cp.float32(0.5))), m


def clipped_mean(
    cp,
    frames,
    masks,
    var_maps,
    weights,
    flxscales,
    egains,
    geoms,
    grid,
    badpix,
    saturated,
    clip_sigma,
    clip_ampfrac,
):
    """Median-centred clipped weighted mean, every dtype step as the numpy backend's; host arrays in and out."""
    n = len(frames)
    target_h, target_w = grid
    d_sci = [cp.asarray(f) for f in frames]
    d_msk = [cp.asarray(m) for m in masks]
    d_var = (
        None if var_maps is None else [d_msk[i] if var_maps[i] is None else cp.asarray(var_maps[i]) for i in range(n)]
    )
    f32flx = [np.float32(f) for f in flxscales]

    # the centre: the numpy median pass, flux scaled in float32, masks/bad pixels/saturation as NaN
    stack = cp.full((n, target_h, target_w), cp.nan, dtype=cp.float32)
    geometric = cp.zeros((target_h, target_w), cp.uint16)
    valid_count = cp.zeros((target_h, target_w), cp.int32)
    for i in range(n):
        tx0, tx1, ty0, ty1, sx0, sx1, sy0, sy1 = geoms[i]
        src = d_sci[i][sy0:sy1, sx0:sx1] * f32flx[i]
        support = src != 0.0
        src[(src == 0.0) | ~cp.isfinite(src)] = cp.nan
        geometric[ty0:ty1, tx0:tx1] += support
        src[d_msk[i][sy0:sy1, sx0:sx1] <= 0] = cp.nan
        _put(cp, src, badpix[i], sx0, sx1, sy0, cp.nan)
        _put(cp, src, saturated[i], sx0, sx1, sy0, cp.nan)
        stack[i, ty0:ty1, tx0:tx1] = src
        valid_count[ty0:ty1, tx0:tx1] += cp.isfinite(src)
    center, _ = _median_center(cp, stack)
    del stack, geometric
    center = cp.where(cp.isfinite(center), center, 0.0)
    two = valid_count == 2
    del valid_count

    sum_arr = cp.zeros((target_h, target_w), cp.float64)
    norm_arr = cp.zeros_like(sum_arr)
    gain_denom = cp.zeros_like(sum_arr)
    var_den = None if d_var is None else cp.zeros_like(sum_arr)
    count_arr = cp.zeros((target_h, target_w), cp.int32)
    geometric_count = cp.zeros((target_h, target_w), cp.uint16)
    two_rejected = cp.zeros((target_h, target_w), cp.bool_)
    n_clipped = n_total = 0
    rejected, gain_terms, all_egain = [], [], True
    for i in range(n):
        tx0, tx1, ty0, ty1, sx0, sx1, sy0, sy1 = geoms[i]
        sl = (slice(ty0, ty1), slice(tx0, tx1))
        flxscale = float(flxscales[i])
        raw = d_sci[i][sy0:sy1, sx0:sx1]
        support = raw != 0.0
        valid = support & cp.isfinite(raw)
        geometric_count[sl] += support
        mask_strip = d_msk[i][sy0:sy1, sx0:sx1]
        valid &= mask_strip > 0
        _put(cp, valid, badpix[i], sx0, sx1, sy0, False)
        _put(cp, valid, saturated[i], sx0, sx1, sy0, False)
        w_eff = float(weights[i]) / (flxscale * flxscale)
        src = raw * flxscale
        c = center[sl]
        sigma_i = 1.0 / np.sqrt(w_eff)
        # numpy: float64 scalar + (float * float32 array) is float64; the float32 difference is compared against it
        threshold = cp.float64(clip_sigma * sigma_i) + (cp.abs(c) * clip_ampfrac).astype(cp.float64)
        clip_ok = cp.abs(src - c).astype(cp.float64) <= threshold
        two_here = two[sl]
        keep = valid & (clip_ok | two_here)
        two_rejected[sl] |= valid & two_here & ~clip_ok
        n_valid, n_keep = int(cp.count_nonzero(valid)), int(cp.count_nonzero(keep))
        n_total += n_valid
        n_clipped += n_valid - n_keep
        rejected.append((valid & ~keep).get() if n_valid > n_keep else None)
        wv = cp.where(keep, w_eff, 0.0)
        sum_arr[sl] += wv * cp.where(keep, src, 0.0)
        norm_arr[sl] += wv
        count_arr[sl] += keep
        egain = egains[i]
        if egain is not None and n_keep:
            gain_terms.append((w_eff, float(egain) / flxscale))
            gain_denom[sl] += cp.where(keep, wv * wv * flxscale / float(egain), 0.0)
        elif egain is None:
            all_egain = False
        if d_var is not None:
            vm = d_var[i][sy0:sy1, sx0:sx1]
            ok = keep & (vm > 0)
            var_den[sl] += cp.where(ok, w_eff * w_eff * flxscale * flxscale / cp.where(ok, vm, cp.float32(1.0)), 0.0)
    coadd = cp.where(norm_arr > 0, sum_arr / cp.where(norm_arr > 0, norm_arr, 1), cp.nan).astype(cp.float32)
    if d_var is not None:
        weight_map = cp.where(var_den > 0, norm_arr * norm_arr / cp.where(var_den > 0, var_den, 1), 0.0)
    else:
        weight_map = norm_arr.copy()
    return dict(
        coadd=coadd.get(), weight_map=weight_map.get(), count=count_arr.get(), geometric=geometric_count.get(),
        norm=norm_arr.get(), gain_denom=gain_denom.get(), n_clipped=n_clipped, n_total=n_total,
        n_two=int(cp.count_nonzero(two_rejected)), rejected=rejected, gain_terms=gain_terms, all_egain=all_egain,
    )  # fmt: skip
