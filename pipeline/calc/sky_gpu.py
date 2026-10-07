"""CuPy kernels of the per-frame sky statistics; each takes the cupy module and mirrors numpy/scipy/photutils."""

from __future__ import annotations

import numpy as np
from photutils.background import Background2D, MMMBackground, StdBackgroundRMS
from photutils.background.interpolators import BkgZoomInterpolator

MESH_BYTES = 4 << 30  # peak cupy pool use of the box statistics on one frame
ZOOM_BYTES = 1 << 30
FRAME_BYTES = 1 << 30
SKY_BYTES = 3 << 30
MESH_BUDGET_S = 5.0  # slower than this, the device is in use by another process

_kernels = {}


def _pairwise_rowsum(cp):
    """numpy's float32 pairwise summation order over the last axis of a C-contiguous (rows, n) array."""
    if "rowsum" not in _kernels:
        _kernels["rowsum"] = cp.RawKernel(
            r"""
extern "C" {
__device__ float pairwise(const float* a, long n) {
    if (n < 8) { float res = 0.0f; for (long i = 0; i < n; i++) res += a[i]; return res; }
    else if (n <= 128) {
        float r[8]; for (int j = 0; j < 8; j++) r[j] = a[j];
        long i = 8;
        for (; i < n - (n % 8); i += 8) { for (int j = 0; j < 8; j++) r[j] += a[i + j]; }
        float res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) res += a[i];
        return res;
    } else { long n2 = n / 2; n2 -= n2 % 8; return pairwise(a, n2) + pairwise(a + n2, n - n2); }
}
__global__ void rowsum(const float* a, long nb, long n, float* out) {
    long b = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (b < nb) out[b] = pairwise(a + b * n, n);
}
}""",
            "rowsum",
            options=("--fmad=false",),
        )
    return _kernels["rowsum"]


def _rowsum_f32(cp, z):
    z = cp.ascontiguousarray(z, dtype=cp.float32)
    rows, n = z.shape
    out = cp.empty(rows, dtype=cp.float32)
    _pairwise_rowsum(cp)(((rows + 127) // 128,), (128,), (z, np.int64(rows), np.int64(n), out))
    return out


def _group_statistics(cp, block, sigma, maxiters, npix_threshold):
    """astropy's compiled sigma clip (float64 bounds), then numpy's float32 MMM and std per NaN-masked row."""
    rows, n = block.shape
    xs = cp.sort(block, axis=-1)
    xs64 = xs.astype(cp.float64)
    idx = cp.arange(n, dtype=cp.int64)[None, :]
    lo_i = cp.zeros(rows, dtype=cp.int64)
    hi_i = cp.isfinite(xs).sum(axis=-1).astype(cp.int64)
    lo = cp.full(rows, -cp.inf)
    hi = cp.full(rows, cp.inf)
    for _ in range(maxiters):
        count = hi_i - lo_i
        k = lo_i + count // 2
        a = cp.take_along_axis(xs64, cp.maximum(k - 1, 0)[:, None], axis=-1)[:, 0]
        b = cp.take_along_axis(xs64, cp.minimum(k, n - 1)[:, None], axis=-1)[:, 0]
        median = cp.where(count % 2 == 1, b, 0.5 * (a + b))
        kept = (idx >= lo_i[:, None]) & (idx < hi_i[:, None])
        mean = cp.where(kept, xs64, 0.0).sum(axis=-1) / count
        std = cp.sqrt(cp.where(kept, (xs64 - mean[:, None]) ** 2, 0.0).sum(axis=-1) / count)
        lo, hi = median - sigma * std, median + sigma * std
        new_lo = cp.maximum(lo_i, (xs64 < lo[:, None]).sum(axis=-1))
        new_hi = cp.minimum(hi_i, (xs64 <= hi[:, None]).sum(axis=-1))
        changed = bool(((new_lo != lo_i) | (new_hi != hi_i)).any())
        lo_i, hi_i = new_lo, new_hi
        if not changed:
            break
    clipped = cp.isfinite(block) & (block >= lo[:, None]) & (block <= hi[:, None])
    ngood = clipped.sum(axis=-1)
    first = (xs64 < lo[:, None]).sum(axis=-1)
    k = first + ngood // 2
    a32 = cp.take_along_axis(xs, cp.maximum(k - 1, 0)[:, None], axis=-1)[:, 0]
    b32 = cp.take_along_axis(xs, cp.minimum(k, n - 1)[:, None], axis=-1)[:, 0]
    median32 = cp.where(ngood % 2 == 1, b32, (a32 + b32) / cp.float32(2))
    zeroed = cp.where(clipped, block, cp.float32(0))
    mean32 = (_rowsum_f32(cp, zeroed).astype(cp.float64) / ngood).astype(cp.float32)
    deviation = cp.where(clipped, block - mean32[:, None], cp.float32(0))
    std32 = cp.sqrt((_rowsum_f32(cp, deviation * deviation).astype(cp.float64) / ngood).astype(cp.float32))
    mmm = cp.float32(3.0) * median32 - cp.float32(2.0) * mean32
    drop = ngood <= npix_threshold
    nan = cp.float32(cp.nan)
    return cp.where(drop, nan, mmm), cp.where(drop, nan, std32), ngood


def _corner_statistics(corner, sigma, maxiters, npix_threshold):
    """photutils' corner box: the axis=None clip compresses the survivors, so numpy sums them in that order."""
    from astropy.stats import SigmaClip

    clipped = SigmaClip(sigma=sigma, maxiters=maxiters)(corner, axis=None, masked=False, copy=True)
    bkg = MMMBackground(sigma_clip=None)(clipped, axis=None)
    rms = StdBackgroundRMS(sigma_clip=None)(clipped, axis=None)
    ngood = int(np.count_nonzero(~np.isnan(clipped)))
    if ngood <= npix_threshold:
        bkg = rms = np.float32(np.nan)
    return np.float32(bkg), np.float32(rms), ngood


def box_statistics(cp, data, mask, box_size, npix_threshold, sigma, maxiters):
    """Background2D._calculate_stats with edge_method pad: core boxes, extra row, extra column, corner."""
    height, width = data.shape
    by, bx = int(box_size[0]), int(box_size[1])
    ny, nx = height // by, width // bx
    y1, x1 = ny * by, nx * bx
    d = cp.where(cp.asarray(mask), cp.float32(cp.nan), cp.asarray(data, dtype=cp.float32))
    core = d[:y1, :x1].reshape(ny, by, nx, bx).transpose(0, 2, 1, 3).reshape(ny * nx, by * bx)
    bkg, rms, ngood = (v.reshape(ny, nx) for v in _group_statistics(cp, core, sigma, maxiters, npix_threshold))
    if y1 < height:
        row = d[y1:, :x1].reshape(height - y1, 1, nx, bx).transpose(0, 2, 1, 3)
        row = cp.ascontiguousarray(cp.moveaxis(row, 0, -1).reshape(nx, -1))
        rb, rr, rn = _group_statistics(cp, row, sigma, maxiters, npix_threshold)
        bkg, rms, ngood = cp.vstack([bkg, rb[None]]), cp.vstack([rms, rr[None]]), cp.vstack([ngood, rn[None]])
    if x1 < width:
        col = d[:y1, x1:].reshape(ny, by, width - x1, 1).transpose(0, 2, 1, 3).transpose(0, 3, 1, 2)
        col = cp.ascontiguousarray(col.reshape(ny, -1))
        cb, cr, cn = _group_statistics(cp, col, sigma, maxiters, npix_threshold)
        if y1 < height:
            corner = np.where(
                np.asarray(mask)[y1:, x1:], np.float32(np.nan), np.asarray(data, dtype=np.float32)[y1:, x1:]
            )
            kb, kr, kn = (cp.asarray([v]) for v in _corner_statistics(corner, sigma, maxiters, npix_threshold))
            cb, cr, cn = cp.concatenate([cb, kb]), cp.concatenate([cr, kr]), cp.concatenate([cn, kn])
        bkg, rms, ngood = cp.hstack([bkg, cb[:, None]]), cp.hstack([rms, cr[:, None]]), cp.hstack([ngood, cn[:, None]])
    return cp.asnumpy(bkg), cp.asnumpy(rms), cp.asnumpy(ngood)


def zoom_mesh(cp, mesh, box_size, shape, order, mode, cval):
    """scipy.ndimage.zoom of the mesh by the box size: float64 spline, one rounding to the mesh dtype, cropped."""
    from cupyx.scipy.ndimage import zoom

    full = zoom(
        cp.asarray(mesh, dtype=cp.float64),
        [int(b) for b in box_size],
        order=order,
        mode=mode,
        cval=cval,
        grid_mode=True,
    )
    return cp.asnumpy(full[: shape[0], : shape[1]].astype(mesh.dtype, copy=False))


def correlate_nearest(cp, data, kernel):
    """scipy.ndimage.correlate(float32, mode='nearest') with scipy's double accumulation, rounded once to float32."""
    from cupyx.scipy.ndimage import correlate

    out = correlate(cp.asarray(data, dtype=cp.float64), cp.asarray(kernel, dtype=cp.float64), mode="nearest")
    return cp.asnumpy(out.astype(cp.float32))


def median_mad(cp, residual, exclude):
    """Median and median absolute deviation of the finite, unexcluded residual in float64."""
    r = cp.asarray(residual)
    sky = r[~cp.asarray(exclude, dtype=bool) & cp.isfinite(r)].astype(cp.float64)
    if sky.size == 0:
        return float("nan"), float("nan")
    median = cp.median(sky)
    return float(median), float(cp.median(cp.abs(sky - median)))


def autocorrelation(cp, residual, mask, maxlag, size, clip, stats, coverage):
    """imcoadd.utils.noise_autocorrelation with its statistics and FFTs on the device; False where it returns None."""
    from cupyx.scipy import fft as cfft
    from scipy.fft import next_fast_len

    residual = cp.asarray(residual)
    mask = None if mask is None else cp.asarray(mask, dtype=bool)
    coverage = None if coverage is None else cp.asarray(coverage, dtype=bool)
    height, width = residual.shape
    ny, nx = (max(1, int(np.ceil(n / size))) for n in residual.shape)
    rows = np.array_split(np.arange(height), ny)
    cols = np.array_split(np.arange(width), nx)
    lags = cp.asarray(np.arange(-maxlag, maxlag + 1))
    total_pairs = cp.zeros((lags.size, lags.size), dtype=cp.float64)
    total_product = cp.zeros_like(total_pairs)
    kept, total = [], 0.0
    pixels = covered_used = covered_total = 0
    for ys in rows:
        for xs in cols:
            cut = (slice(ys[0], ys[-1] + 1), slice(xs[0], xs[-1] + 1))
            patch = residual[cut].astype(cp.float64)
            covered = int(cp.count_nonzero(cp.isfinite(patch) if coverage is None else coverage[cut]))
            covered_total += covered
            bad = ~cp.isfinite(patch) if mask is None else (mask[cut] | ~cp.isfinite(patch))
            n_good = int(cp.count_nonzero(~bad))
            if n_good == 0 or n_good < 16 * covered // 100:
                continue
            if clip:
                good = patch[~bad]
                center = cp.median(good)
                scale = 1.482602218505602 * float(cp.median(cp.abs(good - center)))
                if not np.isfinite(scale) or scale <= 0:
                    continue
                bad |= cp.abs(patch - center) > clip * scale
                n_good = int(cp.count_nonzero(~bad))
            if n_good < 16 * covered // 100:
                continue
            kept.append((cut, bad))
            total += float(patch[~bad].sum())
            pixels += n_good
            covered_used += covered
    if pixels == 0:
        return False
    mean = total / pixels
    for cut, bad in kept:
        patch = residual[cut].astype(cp.float64)
        good = (~bad).astype(cp.float64)
        field = cp.where(bad, 0.0, patch - mean)
        shape = [next_fast_len(n + maxlag) for n in patch.shape]
        ft, gt = cfft.rfft2(field, s=shape), cfft.rfft2(good, s=shape)
        pairs = cfft.irfft2(gt * cp.conj(gt), s=shape)
        product = cfft.irfft2(ft * cp.conj(ft), s=shape)
        total_pairs += pairs[cp.ix_(lags, lags)]
        total_product += product[cp.ix_(lags, lags)]
    total_pairs, total_product = cp.asnumpy(total_pairs), cp.asnumpy(total_product)
    if np.any(total_pairs <= 0):
        return False
    window = total_product / total_pairs
    if not np.isfinite(window[maxlag, maxlag]) or window[maxlag, maxlag] <= 0:
        return False
    if stats is not None:
        stats["variance"] = float(window[maxlag, maxlag])
        stats["pixels"] = pixels
        stats["sub_areas"] = len(kept)
        stats["represented"] = covered_used / covered_total if covered_total else 0.0
    return window / window[maxlag, maxlag]


class GpuZoomInterpolator(BkgZoomInterpolator):
    """BkgZoomInterpolator whose spline zoom runs on the device through the injected `on_gpu` policy call."""

    def __init__(self, on_gpu, **kwargs):
        super().__init__(**kwargs)
        self.on_gpu = on_gpu

    def __call__(self, data, **kwargs):
        data = np.asanyarray(data)
        if kwargs["edge_method"] == "pad" and np.ptp(data) != 0:
            result = self.on_gpu(
                zoom_mesh, data, kwargs["box_size"], kwargs["shape"], self.order, self.mode, self.cval,
                need_bytes=ZOOM_BYTES, name="mesh zoom",
            )  # fmt: skip
            if result is not None:
                if self.clip:
                    np.clip(result, np.min(data), np.max(data), out=result)
                return result
        return super().__call__(data, **kwargs)


class GpuBackground2D(Background2D):
    """Background2D with the box statistics on the device through the injected on_gpu; mesh steps are the parent's."""

    def __init__(self, data, box_size, *, on_gpu, logger=None, **kwargs):
        self.on_gpu = on_gpu
        self.logger = logger
        super().__init__(data, box_size, **kwargs)

    def _calculate_stats(self):
        clip = self.sigma_clip
        exact = (
            clip is not None
            and clip.cenfunc == "median"
            and clip.stdfunc == "std"
            and clip.sigma_lower == clip.sigma_upper == clip.sigma
            and np.isfinite(clip.maxiters)
            and not clip.grow
            and self.edge_method == "pad"
            and self._data.dtype == np.float32
            and type(self.bkg_estimator) is MMMBackground
            and type(self.bkgrms_estimator) is StdBackgroundRMS
        )
        if not exact:
            return super()._calculate_stats()
        input_mask = self._mask
        mask = self._combine_all_masks(~np.isfinite(self._data))
        self._box_npixels = np.prod(self.box_size)
        stats = self.on_gpu(
            box_statistics, self._data, mask, self.box_size, self._good_npixels_threshold, float(clip.sigma),
            int(clip.maxiters), need_bytes=MESH_BYTES, budget_s=MESH_BUDGET_S, logger=self.logger, name="mesh",
        )  # fmt: skip
        if stats is None:
            self._mask = input_mask
            return super()._calculate_stats()
        bkg, rms, ngood = stats
        if np.all(np.isnan(bkg)):
            raise ValueError(
                f"All boxes contain <= {self._good_npixels_threshold} unmasked or finite pixels ({self.box_size=}, "
                f'{self.exclude_percentile=}). Please check your data or increase "exclude_percentile" to allow more '
                "boxes to be included."
            )
        del self._data
        return bkg, rms, ngood
