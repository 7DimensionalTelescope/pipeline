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

    if not (weight):
        cmd += ["-no-weight"]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Error combining images: {result.stderr}")

    return None


def interpolate_masked_pixels_cpu(
    images, mask_path, output_paths, window=1, method="median", badpix=1, weight=True, device=None
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

    for idx, sci_in in enumerate(images):
        sci = fits.getdata(sci_in).astype(np.float32)
        wgt = fits.getdata(add_suffix(sci_in, "weight")).astype(np.float32) if weight else None

        if weight:
            interp_img, interp_wt = interpolate_masked_pixels_cpu_numba(
                sci, mask, window=window, weight=wgt, use_median=(method == "median")
            )
        else:
            interp_img, interp_wt = interpolate_masked_pixels_cpu_numba_no_weight(sci, mask, window=window)

        sci_out = output_paths[idx]
        fits.writeto(sci_out, interp_img, header=add_bpx_method(fits.getheader(sci_in), method), overwrite=True)
        if weight and interp_wt is not None:
            fits.writeto(
                add_suffix(sci_out, "weight"),
                interp_wt,
                header=add_bpx_method(fits.getheader(add_suffix(sci_in, "weight")), method),
                overwrite=True,
            )


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
    weight_result = np.zeros_like(image)

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
    vals = np.empty(max_patch, dtype=image.dtype)
    wts = np.empty(max_patch, dtype=image.dtype)
    tmp = np.empty(max_patch, dtype=image.dtype)

    # Parallel loop
    for k in prange(count):
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
