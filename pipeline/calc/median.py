"""Per-pixel median kernels shared by Preprocess and ImCoadd, with the median noise penalty."""

import numpy as np
from numba import njit, prange


@njit
def quickselect(buf, m, k):
    """In-place k-th smallest of buf[:m] (Devillard median-of-three quickselect); leaves buf[:k] <= buf[k]."""
    low = 0
    high = m - 1
    while True:
        if high <= low:
            return buf[k]
        if high == low + 1:
            if buf[low] > buf[high]:
                buf[low], buf[high] = buf[high], buf[low]
            return buf[k]
        middle = (low + high) >> 1
        if buf[middle] > buf[high]:
            buf[middle], buf[high] = buf[high], buf[middle]
        if buf[low] > buf[high]:
            buf[low], buf[high] = buf[high], buf[low]
        if buf[middle] > buf[low]:
            buf[middle], buf[low] = buf[low], buf[middle]
        buf[middle], buf[low + 1] = buf[low + 1], buf[middle]
        ll = low + 1
        hh = high
        while True:
            ll += 1
            while buf[low] > buf[ll]:
                ll += 1
            hh -= 1
            while buf[hh] > buf[low]:
                hh -= 1
            if hh < ll:
                break
            buf[ll], buf[hh] = buf[hh], buf[ll]
        buf[low], buf[hh] = buf[hh], buf[low]
        if hh <= k:
            low = ll
        if hh >= k:
            high = hh - 1


@njit(parallel=True)
def nanmedian_axis0(stack, out):
    """Row-parallel NaN-ignoring median over axis 0; matches np.nanmedian bitwise."""
    n, h, w = stack.shape
    for y in prange(h):
        buf = np.empty(n, dtype=np.float32)
        for x in range(w):
            m = 0
            for i in range(n):
                v = stack[i, y, x]
                if not np.isnan(v):
                    buf[m] = v
                    m += 1
            if m == 0:
                out[y, x] = np.nan
            elif m % 2:
                out[y, x] = quickselect(buf, m, (m - 1) // 2)
            else:
                hi = quickselect(buf, m, m // 2)
                lo = buf[0]
                for j in range(1, m // 2):
                    if buf[j] > lo:
                        lo = buf[j]
                out[y, x] = 0.5 * (lo + hi)


@njit(parallel=True)
def nanmedian_std_axis0(stack, med, sig):
    """Row-parallel NaN-ignoring median and ddof=1 standard deviation over axis 0, in one gather per pixel."""
    n, h, w = stack.shape
    for y in prange(h):
        buf = np.empty(n, dtype=np.float32)
        for x in range(w):
            m = 0
            s = 0.0
            for i in range(n):
                v = stack[i, y, x]
                if not np.isnan(v):
                    buf[m] = v
                    m += 1
                    s += v
            if m == 0:
                med[y, x] = np.nan
                sig[y, x] = np.nan
                continue
            mean = s / m
            var = 0.0
            for j in range(m):
                d = buf[j] - mean
                var += d * d
            sig[y, x] = np.sqrt(var / (m - 1)) if m > 1 else np.float32(0.0)
            if m % 2:
                med[y, x] = quickselect(buf, m, (m - 1) // 2)
            else:
                hi = quickselect(buf, m, m // 2)
                lo = buf[0]
                for j in range(1, m // 2):
                    if buf[j] > lo:
                        lo = buf[j]
                med[y, x] = 0.5 * (lo + hi)


# Var(median)/Var(mean) for n Gaussian samples, from the order-statistic density
# n!/(m!m!) F^m (1-F)^m f. Odd and even n are SEPARATE branches and must never be interpolated
# together: an even-n median averages the two central order statistics and is markedly more
# efficient (n=10 is 1.383, between its odd neighbours' 1.495 and 1.509). Each branch is anchored
# at 1/n = 0 to the pi/2 limit (Kendall & Stuart Vol.1), which is only reached asymptotically.
_MEDIAN_VAR_RATIO = {
    # odd n
    1: 1.0, 3: 1.3460133137, 5: 1.4341683080, 7: 1.4731280307, 9: 1.4949115322, 11: 1.5087867690,
    13: 1.5183869350, 15: 1.5254197812, 17: 1.5307918842, 19: 1.5350285264, 21: 1.5384548547,
    31: 1.5489329407, 51: 1.5575323379, 101: 1.5641094238, 201: 1.5674391273,
    # even n
    2: 1.0, 4: 1.1927984737, 6: 1.2884559995, 8: 1.3454468413, 10: 1.3832643584, 12: 1.4101942454,
    14: 1.4303497151, 16: 1.4460033796, 18: 1.4585131809, 20: 1.4687405541, 32: 1.5047903033,
    50: 1.5276417434, 64: 1.5367949276, 100: 1.5487936039, 200: 1.5596846119,
}  # fmt: skip


def _ratio_branch(parity: int) -> tuple["np.ndarray", "np.ndarray"]:
    """Ascending (1/n, ratio) interpolation coordinates for one n-parity, with 1/n=0 -> pi/2."""
    knots = np.array(sorted(k for k in _MEDIAN_VAR_RATIO if k % 2 == parity), dtype=np.float64)
    values = np.array([_MEDIAN_VAR_RATIO[int(k)] for k in knots], dtype=np.float64)
    return np.concatenate(([0.0], (1.0 / knots)[::-1])), np.concatenate(([np.pi / 2], values[::-1]))


_ODD_RATIO_X, _ODD_RATIO_Y = _ratio_branch(1)
_EVEN_RATIO_X, _EVEN_RATIO_Y = _ratio_branch(0)


def _ratio_at_integer(n) -> "np.ndarray":
    """Parity-aware variance ratio at integer sample sizes, shape-preserving."""
    flat = np.atleast_1d(n).ravel()
    out = np.empty(flat.shape, dtype=np.float64)
    odd = (flat & 1) != 0
    out[odd] = np.interp(1.0 / flat[odd], _ODD_RATIO_X, _ODD_RATIO_Y)
    out[~odd] = np.interp(1.0 / flat[~odd], _EVEN_RATIO_X, _EVEN_RATIO_Y)
    return out.reshape(np.shape(n))


def _median_penalty(counts) -> "np.ndarray":
    """Vectorized median_variance_ratio over a per-pixel contributor-count array."""
    n = np.maximum(np.asarray(counts, dtype=np.float64), 1.0)
    lo = np.floor(n).astype(np.int64)
    hi = np.ceil(n).astype(np.int64)
    r_lo = _ratio_at_integer(lo)
    if np.array_equal(lo, hi):
        return r_lo
    return r_lo + (n - lo) * (_ratio_at_integer(hi) - r_lo)


def median_variance_ratio(n: float) -> float:
    """How much noisier a median of ``n`` frames is than their mean; a fractional ``n`` interpolates between the
    neighbouring integer sample sizes, which is exact when it arises as a mixture of those two counts."""
    if n <= 1:
        return 1.0
    if np.isposinf(n):
        return float(np.pi / 2)
    return float(_median_penalty(np.float64(n)))
