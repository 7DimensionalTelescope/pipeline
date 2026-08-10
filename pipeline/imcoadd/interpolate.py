from ..utils import add_suffix
from numba import njit, prange
import numpy as np
from astropy.io import fits
import subprocess
from ..const import SOURCE_DIR


def interpolate_masked_pixels_subprocess(
    images: list[str],
    mask: str,
    output: list[str],
    window: int = 1,
    method: str = "median",
    badpix: int = 1,
    device: int = 0,
    weight: bool = True,
    zero_interp_weight: bool = True,
):
    # base command
    cmd = [
        "python",
        f"{SOURCE_DIR}/cuda/interpolate_masked_pixels.py",
        "-input",
        *images,
        "-output",
        *output,
        "-mask",
        mask,
        "-window",
        str(window),
        "-method",
        method,
        "-badpix",
        str(badpix),
        "-device",
        str(device),
    ]

    if isinstance(weight, (list, tuple)):
        cmd += ["-weight-input", *weight]
    elif not weight:
        cmd += ["-no-weight"]
    if not zero_interp_weight:
        cmd += ["-keep-interp-weight"]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Error combining images: {result.stderr}")

    return None


def interpolate_masked_pixels_cpu(
    images, mask_path, output_paths, window=1, method="median", badpix=1, weight=True, device=None,
    zero_interp_weight=True, logger=None,
):
    """
    High-level function: reads FITS images, applies numba interpolation, and writes output.

    Params:
        images       : list of sci_path
        mask_path    : path to mask FITS file
        window, method, badpix, weight : as above
        output_paths : list of sci_out paths
    """
    # Load mask
    mask = fits.getdata(mask_path).astype(np.int32)

    weight_paths = weight if isinstance(weight, (list, tuple)) else None
    weight = bool(weight)

    def _wgt_in(idx):
        return weight_paths[idx] if weight_paths is not None else add_suffix(images[idx], "weight")

    def _load(idx):
        sci = fits.getdata(images[idx]).astype(np.float32)
        wgt = fits.getdata(_wgt_in(idx)).astype(np.float32) if weight else None
        return sci, wgt

    def _write(idx, interp_img, interp_wt):
        sci_out = output_paths[idx]
        fits.writeto(sci_out, interp_img, header=add_bpx_method(fits.getheader(images[idx]), method), overwrite=True)
        if weight and interp_wt is not None:
            if zero_interp_weight:
                # an interpolated value is a copy of its neighbours: no independent information
                interp_wt[mask == badpix] = 0.0
            fits.writeto(
                add_suffix(sci_out, "weight"),
                interp_wt,
                header=add_bpx_method(fits.getheader(_wgt_in(idx)), method),
                overwrite=True,
            )

    # Two threads overlap NFS I/O with the kernel: prefetch the next frame and flush the
    # previous write while this one computes. Bounded, so no oversubscription with the
    # system queue; FITS reads/writes release the GIL.
    from concurrent.futures import ThreadPoolExecutor

    import os as _os
    import time as _time

    with ThreadPoolExecutor(max_workers=2) as pool:
        pending_write = None
        nxt = pool.submit(_load, 0)
        for idx in range(len(images)):
            st_img = _time.time()
            sci, wgt = nxt.result()
            t_read = _time.time() - st_img
            if idx + 1 < len(images):
                nxt = pool.submit(_load, idx + 1)

            if weight:
                interp_img, interp_wt = interpolate_masked_pixels_cpu_numba(
                    sci, mask, window=window, weight=wgt, use_median=(method == "median")
                )
            else:
                interp_img, interp_wt = interpolate_masked_pixels_cpu_numba_no_weight(sci, mask, window=window)

            t_kernel = _time.time() - st_img - t_read
            if pending_write is not None:
                pending_write.result()
            pending_write = pool.submit(_write, idx, interp_img, interp_wt)
            if logger is not None:
                logger.debug(
                    f"Interpolated {_os.path.basename(images[idx])} [image {idx + 1}/{len(images)}] "
                    f"in {_time.time() - st_img:.1f} seconds (read-wait {t_read:.1f}, kernel {t_kernel:.1f})"
                )
                if (idx + 1) % 25 == 0 or idx + 1 == len(images):
                    logger.info(f"Interpolation progress {idx + 1}/{len(images)}")
        if pending_write is not None:
            pending_write.result()


@njit(parallel=True)
def interpolate_masked_pixels_cpu_numba(
    image: np.ndarray, mask: np.ndarray, weight: np.ndarray, window: int, use_median: bool  # must always be an array!
):
    """
    image:      2D float array.
    mask:       2D int8 or bool array; 1 = pixel to interpolate.
    weight:     2D float array of same shape (e.g. 1/variance).
                For unweighted median, just pass np.ones_like(image).
    window:     radius in pixels to search neighbors.
    inverse_variance: True => weighted mean; False => median.
    """
    H, W = image.shape
    result = image.copy()
    weight_result = weight.copy()  # GPU kernel does the same; zeros here starved SWarp

    # Collect masked coords
    count = np.sum(mask)

    rows = np.empty(count, dtype=np.int32)
    cols = np.empty(count, dtype=np.int32)
    idx = 0
    for i in range(H):
        for j in range(W):
            if mask[i, j]:
                rows[idx] = i
                cols[idx] = j
                idx += 1

    max_patch = (2 * window + 1) ** 2

    # Parallel loop
    for k in prange(count):
        # scratch buffers per iteration: shared ones raced across threads (wrong medians)
        # and false-shared cache lines (orders of magnitude slower under load)
        vals = np.empty(max_patch, dtype=image.dtype)
        wts = np.empty(max_patch, dtype=image.dtype)
        tmp = np.empty(max_patch, dtype=image.dtype)
        r = rows[k]
        c = cols[k]

        # window bounds
        r0 = r - window if r - window >= 0 else 0
        r1 = r + window + 1 if r + window + 1 <= H else H
        c0 = c - window if c - window >= 0 else 0
        c1 = c + window + 1 if c + window + 1 <= W else W

        # gather neighbors
        n = 0
        for yy in range(r0, r1):
            for xx in range(c0, c1):
                if mask[yy, xx] == 0:
                    vals[n] = image[yy, xx]
                    wts[n] = weight[yy, xx]
                    n += 1

        if n == 0:
            # no unmasked neighbors
            result[r, c] = 0.0
            weight_result[r, c] = 0.0
            continue

        if use_median:
            # median (unweighted or for picking weight)
            # insertion‐sort first n elements of vals → tmp
            for i in range(n):
                tmp[i] = vals[i]
            for i in range(1, n):
                key = tmp[i]
                j = i - 1
                while j >= 0 and tmp[j] > key:
                    tmp[j + 1] = tmp[j]
                    j -= 1
                tmp[j + 1] = key

            # pick median value
            if (n & 1) == 1:
                med = tmp[n // 2]
                # Find weight of the median value
                sel = 0.0
                for i in range(n):
                    if vals[i] == med:
                        sel = wts[i]
                        break
            else:
                med = 0.5 * (tmp[n // 2 - 1] + tmp[n // 2])
                # Average weights of the two middle values
                sel = 0.5 * (wts[n // 2 - 1] + wts[n // 2])

            result[r, c] = med
            weight_result[r, c] = sel

        else:
            # weighted mean
            vsum = 0.0
            wsum = 0.0
            for i in range(n):
                vsum += vals[i] * wts[i]
                wsum += wts[i]
            if wsum > 0.0:
                result[r, c] = vsum / wsum
            else:
                result[r, c] = 0.0
            weight_result[r, c] = wsum

    return result, weight_result


@njit(parallel=True)
def interpolate_masked_pixels_cpu_numba_no_weight(image, mask, window=1):
    assert image.shape == mask.shape
    assert image.ndim == 2

    H, W = image.shape
    result = image.copy()

    # Flatten index lookup of masked pixels
    count = np.sum(mask)

    rows = np.empty(count, dtype=np.int32)
    cols = np.empty(count, dtype=np.int32)
    idx = 0
    for i in range(H):
        for j in range(W):
            if mask[i, j] == 1:
                rows[idx] = i
                cols[idx] = j
                idx += 1

    # Process each masked pixel in parallel
    for k in prange(count):
        r = rows[k]
        c = cols[k]

        r0 = max(0, r - window)
        r1 = min(H, r + window + 1)
        c0 = max(0, c - window)
        c1 = min(W, c + window + 1)

        # Collect valid neighbor values
        vals = []
        for y in range(r0, r1):
            for x in range(c0, c1):
                if mask[y, x] == 0:
                    vals.append(image[y, x])

        if len(vals) > 0:
            vals_np = np.array(vals)
            result[r, c] = np.median(vals_np)

    return result, None


def add_bpx_method(header, method):
    header["INTERP"] = (method.upper(), "Method for bad pixel interpolation")
    # swarp can't propage HIERARCH keywords
    # header["BPX_INTERP"] = (method.upper(), "Method for bad pixel interpolation")
    return header


# @njit(parallel=True)
# def interpolate_masked_pixels_cpu_numba(image, mask, window=1):
#     assert image.shape == mask.shape
#     assert image.ndim == 2

#     H, W = image.shape
#     result = image.copy()

#     # Flatten index lookup of masked pixels
#     count = 0
#     for i in range(H):
#         for j in range(W):
#             if mask[i, j] == 1:
#                 count += 1

#     rows = np.empty(count, dtype=np.int32)
#     cols = np.empty(count, dtype=np.int32)
#     idx = 0
#     for i in range(H):
#         for j in range(W):
#             if mask[i, j] == 1:
#                 rows[idx] = i
#                 cols[idx] = j
#                 idx += 1

#     # Process each masked pixel in parallel
#     for k in prange(count):
#         r = rows[k]
#         c = cols[k]

#         r0 = max(0, r - window)
#         r1 = min(H, r + window + 1)
#         c0 = max(0, c - window)
#         c1 = min(W, c + window + 1)

#         # Collect valid neighbor values
#         vals = []
#         for y in range(r0, r1):
#             for x in range(c0, c1):
#                 if mask[y, x] == 0:
#                     vals.append(image[y, x])

#         if len(vals) > 0:
#             vals_np = np.array(vals)
#             result[r, c] = np.median(vals_np)

#     return result


# def interpolate_masked_pixels_cpu(image, mask, window=1):
#     assert image.shape == mask.shape
#     assert image.ndim == 2

#     result = image.copy()
#     rows, cols = np.where(mask == 1)

#     for i in range(len(rows)):
#         r, c = rows[i], cols[i]

#         r0 = max(0, r - window)
#         r1 = min(image.shape[0], r + window + 1)
#         c0 = max(0, c - window)
#         c1 = min(image.shape[1], c + window + 1)

#         patch = image[r0:r1, c0:c1]
#         patch_mask = mask[r0:r1, c0:c1]

#         valid_values = patch[patch_mask == 0]
#         if valid_values.size > 0:
#             result[r, c] = np.median(valid_values)

#     return result


# def interpolate_masked_pixels_gpu(image, mask, window=1):
#     # image: 2D cupy array
#     # mask: 2D cupy array with 0 (valid) and 1 (masked)
#     import cupy as cp

#     assert image.shape == mask.shape
#     assert image.ndim == 2

#     result = image.copy()
#     rows, cols = cp.where(mask == 1)

#     for i in range(len(rows)):
#         r, c = rows[i], cols[i]

#         r0 = max(0, r - window)
#         r1 = min(image.shape[0], r + window + 1)
#         c0 = max(0, c - window)
#         c1 = min(image.shape[1], c + window + 1)

#         patch = image[r0:r1, c0:c1]
#         patch_mask = mask[r0:r1, c0:c1]

#         valid_values = patch[patch_mask == 0]
#         if valid_values.size > 0:
#             result[r, c] = cp.median(valid_values)

#     return result.get()


# def interpolate_masked_pixels_gpu_vectorized(image, mask, window=1):
#     """avoids python loop and faster"""
#     import cupy as cp

#     H, W = image.shape
#     assert image.shape == mask.shape

#     result = image.copy()

#     # Get coordinates of masked pixels
#     ys, xs = cp.where(mask == 1)
#     N = ys.shape[0]

#     if N == 0:
#         return result  # no masked pixels

#     # Generate patch index offsets
#     # Kernel Size is 2 * window + 1
#     dy, dx = cp.meshgrid(cp.arange(-window, window + 1), cp.arange(-window, window + 1), indexing="ij")
#     dy = dy.ravel()  # (K,)
#     dx = dx.ravel()  # (K,)
#     # K = dy.size

#     # Broadcast and clamp patch indices
#     patch_ys = cp.clip(ys[:, None] + dy[None, :], 0, H - 1)  # (N, K)
#     patch_xs = cp.clip(xs[:, None] + dx[None, :], 0, W - 1)  # (N, K)

#     # Flatten indices for fancy indexing
#     flat_indices = patch_ys * W + patch_xs
#     flat_image = image.ravel()
#     flat_mask = mask.ravel()

#     # Gather patch values and masks
#     patch_vals = flat_image[flat_indices]  # (N, K)
#     patch_mask = flat_mask[flat_indices]  # (N, K)

#     # Mask out invalid pixels
#     valid_vals = cp.where(patch_mask == 0, patch_vals, cp.nan)

#     # Compute nanmean along axis=1 (for each patch)
#     interp_vals = cp.nanmedian(valid_vals, axis=1)

#     # Replace masked pixels in result
#     result[ys, xs] = interp_vals

#     return result


# # def interpolate_masked_pixels_gpu_vectorized_weight(image, mask, weight=None, window=1):
# #     """All inputs have to be cupy arrays."""
# #     import cupy as cp

# #     H, W = image.shape
# #     assert image.shape == mask.shape
# #     if weight is not None:
# #         assert weight.shape == image.shape

# #     result = image.copy()
# #     ys, xs = cp.where(mask == 1)
# #     N = ys.shape[0]
# #     if N == 0:
# #         return result

# #     # Create patch offsets
# #     dy, dx = cp.meshgrid(
# #         cp.arange(-window, window + 1), cp.arange(-window, window + 1), indexing="ij"
# #     )
# #     dy = dy.ravel()  # shape (K,)
# #     dx = dx.ravel()  # shape (K,)
# #     # K = dy.size

# #     # Absolute patch indices
# #     patch_ys = ys[:, None] + dy[None, :]  # shape (N, K)
# #     patch_xs = xs[:, None] + dx[None, :]  # shape (N, K)

# #     # Mask out-of-bound locations
# #     in_bounds = (patch_ys >= 0) & (patch_ys < H) & (patch_xs >= 0) & (patch_xs < W)

# #     # Clip for safe indexing
# #     patch_ys_safe = cp.clip(patch_ys, 0, H - 1)
# #     patch_xs_safe = cp.clip(patch_xs, 0, W - 1)
# #     flat_indices = patch_ys_safe * W + patch_xs_safe

# #     # Fetch data
# #     flat_image = image.ravel()
# #     flat_mask = mask.ravel()
# #     patch_vals = flat_image[flat_indices]
# #     patch_mask = flat_mask[flat_indices]

# #     # Mark valid values: unmasked AND in-bounds
# #     valid = (patch_mask == 0) & in_bounds

# #     if weight is not None:
# #         flat_weight = weight.ravel()
# #         patch_weights = flat_weight[flat_indices]

# #         patch_weights = cp.where(valid, patch_weights, 0)
# #         patch_vals = cp.where(valid, patch_vals, 0)

# #         weighted_sum = cp.sum(patch_weights * patch_vals, axis=1)
# #         weight_total = cp.sum(patch_weights, axis=1)
# #         interp_vals = cp.where(weight_total > 0, weighted_sum / weight_total, 0)
# #     else:
# #         patch_vals = cp.where(valid, patch_vals, cp.nan)
# #         interp_vals = cp.nanmedian(patch_vals, axis=1)

# #     # Fill in interpolated values
# #     result[ys, xs] = interp_vals
# #     return result


from .weight_store import load_single_weight, persist_single_weight


def write_weight_int16(path, weight, header, n_holes=None):
    """Weight sidecar as BITPIX 16 + BSCALE: FITS has no float16, and SWarp reads scaled
    ints exactly (verified). Half the bytes of float32; quantization <= wmax/64000."""
    weight = np.where(np.isfinite(weight) & (weight >= 0), weight, 0.0).astype(np.float32)
    hdu = fits.PrimaryHDU(weight, header=header)
    if n_holes is not None:
        hdu.header["WGTHOLES"] = (bool(n_holes), "zero-weight holes at interpolated pixels")
        hdu.header["NHOLEPIX"] = (int(n_holes), "number of zero-weight (interpolated) pixels")
    wmax = float(np.nanmax(weight)) if weight.size else 0.0
    if wmax > 0:
        # unsigned-int16 convention (BZERO = 32768*BSCALE): weights are nonnegative, so
        # the full 65535 levels map [0, wmax] and physical 0 stays exactly representable
        bscale = wmax / 65535.0
        hdu.scale("int16", bscale=bscale, bzero=32768.0 * bscale)
    hdu.writeto(path, overwrite=True)


def weight_and_interpolate_cpu(
    images, mask_path, output_paths, calib, window=1, method="median", badpix=1,
    zero_interp_weight=True, logger=None, post_frame=None, weight_store=None,
):
    """Fused weight calculation + bad-pixel interpolation, one read and one write per image.

    The weight map lives only in memory between the two kernels: the separate stage wrote
    it (245 MB), read it back, and read the science frame a second time -- ~735 MB of NFS
    traffic per image that carried no information. Read-ahead and write-behind threads
    see-saw I/O against the kernels.
    """
    import os as _os
    import time as _time

    st_stage = _time.time()
    from concurrent.futures import ThreadPoolExecutor

    from .weight import optimized_parallel

    mask = fits.getdata(mask_path).astype(np.int32)
    hole = mask == badpix
    n_holes = int(hole.sum())

    def _load(idx):
        return fits.getdata(images[idx]).astype(np.float32)

    def _write(idx, interp_img, interp_wt):
        sci_out = output_paths[idx]
        hdr = add_bpx_method(fits.getheader(images[idx]), method)
        fits.writeto(sci_out, interp_img, header=hdr, overwrite=True)
        write_weight_int16(add_suffix(sci_out, "weight"), interp_wt, hdr,
                           n_holes=n_holes if zero_interp_weight else 0)
        if post_frame is not None:
            post_frame(sci_out)  # e.g. per-image reprojection (+ optional interp discard)

    with ThreadPoolExecutor(max_workers=3) as pool:
        pending_write = None
        ahead = [pool.submit(_load, i) for i in range(min(2, len(images)))]
        for idx in range(len(images)):
            st_img = _time.time()
            sci = ahead.pop(0).result()
            if idx + 2 < len(images):
                ahead.append(pool.submit(_load, idx + 2))
            t_read = _time.time() - st_img

            # durable store: reuse a provenance-verified map, else compute and persist
            wgt = n_nonfinite = None
            if weight_store is not None:
                store_paths, store_masters = weight_store
                wgt = load_single_weight(store_paths[idx], store_masters)
            if wgt is None:
                if calib is None:
                    raise RuntimeError(f"single weight map vanished mid-run for {images[idx]}")
                wgt = optimized_parallel(sci, *calib)
                # a degenerate noise model (sig_z = dark = pixel = 0) divides to inf/nan;
                # zero certainty about a pixel is weight 0, not weight infinity
                nonfinite = ~np.isfinite(wgt)
                n_nonfinite = int(nonfinite.sum())
                if n_nonfinite:
                    wgt[nonfinite] = 0.0
                if weight_store is not None:
                    pool.submit(persist_single_weight, store_paths[idx], wgt.copy(), store_masters)
            t_weight = _time.time() - st_img - t_read
            interp_img, interp_wt = interpolate_masked_pixels_cpu_numba(
                sci, mask, window=window, weight=wgt, use_median=(method == "median")
            )
            if zero_interp_weight:
                interp_wt[hole] = 0.0
            t_interp = _time.time() - st_img - t_read - t_weight

            if pending_write is not None:
                pending_write.result()
            pending_write = pool.submit(_write, idx, interp_img, interp_wt)
            if logger is not None:
                # per-image detail at DEBUG; INFO gets one summary line per 25 frames
                logger.debug(
                    f"Weight+interp {_os.path.basename(images[idx])} [image {idx + 1}/{len(images)}] "
                    f"in {_time.time() - st_img:.1f} seconds "
                    f"(read-wait {t_read:.1f}, weight {t_weight:.1f}, interp {t_interp:.1f})"
                    + (f" [{n_nonfinite} degenerate weight px zeroed]" if (n_nonfinite or 0) else "")
                )
                if (idx + 1) % 25 == 0 or idx + 1 == len(images):
                    logger.info(
                        f"Weight+interp progress {idx + 1}/{len(images)} "
                        f"({(_time.time() - st_stage) / (idx + 1):.1f} s/image avg)"
                    )
        if pending_write is not None:
            pending_write.result()
