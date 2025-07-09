from numba import njit, prange
import numpy as np
import cupy as cp
from astropy.io import fits
from typing import List, Tuple


def interpolate_masked_pixels(
    images: List[Tuple[str]], mask, window=1, method=None, badpix=None, weight: bool = True, device=None
):

    cpu_buffer = np.empty(mask.shape)
    if weight:
        cpu_buffer_weight = np.empty(mask.shape)
    else:
        cpu_buffer_weight = None

    output = []
    output_weight = []

    is_gpu = device is not None and device != "CPU"
    if is_gpu:
        with cp.cuda.Device(device):
            mask = cp.asarray(mask)

    for o, w in images:
        cpu_buffer[:] = fits.getdata(o)

        if weight:
            cpu_buffer_weight[:] = fits.getdata(w)

        # gpu
        if is_gpu:
            with cp.cuda.Device(device):
                gpu_buffer = cp.asarray(cpu_buffer)
                gpu_buffer[:] = gpu_buffer.astype(cp.float32)

                if weight:
                    gpu_buffer_weight = cp.asarray(cpu_buffer_weight)
                    gpu_buffer_weight = gpu_buffer_weight.astype(cp.float32)
                    gpu_buffer[:], gpu_buffer_weight[:] = interpolate_masked_pixels_gpu_vectorized_weight(
                        gpu_buffer, mask, window=window, method=method, badpix=badpix, weight=gpu_buffer_weight
                    )
                else:
                    gpu_buffer[:] = interpolate_masked_pixels_gpu_vectorized(
                        gpu_buffer, mask, window=window, method=method, badpix=badpix
                    )
                    gpu_buffer_weight = None

                cpu_buffer[:] = cp.asnumpy(gpu_buffer)
                output.append(cpu_buffer.copy())

                if weight:
                    cpu_buffer_weight[:] = cp.asnumpy(gpu_buffer_weight)
                    output_weight.append(cpu_buffer_weight.copy())
        # cpu
        else:
            cpu_buffer[:] = interpolate_masked_pixels_cpu_numba(cpu_buffer, mask, window=window)
            output.append(cpu_buffer.copy())

    if device and device != "CPU":
        with cp.cuda.Device(device):
            del gpu_buffer, gpu_buffer_weight, mask
            cp.get_default_memory_pool().free_all_blocks()

    del cpu_buffer

    return output, output_weight


@njit(parallel=True)
def interpolate_masked_pixels_cpu_numba(image, mask, window=1):
    assert image.shape == mask.shape
    assert image.ndim == 2

    H, W = image.shape
    result = image.copy()

    # Flatten index lookup of masked pixels
    count = 0
    for i in range(H):
        for j in range(W):
            if mask[i, j] == 1:
                count += 1

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

    return result


def interpolate_masked_pixels_cpu(image, mask, window=1):
    assert image.shape == mask.shape
    assert image.ndim == 2

    result = image.copy()
    rows, cols = np.where(mask == 1)

    for i in range(len(rows)):
        r, c = rows[i], cols[i]

        r0 = max(0, r - window)
        r1 = min(image.shape[0], r + window + 1)
        c0 = max(0, c - window)
        c1 = min(image.shape[1], c + window + 1)

        patch = image[r0:r1, c0:c1]
        patch_mask = mask[r0:r1, c0:c1]

        valid_values = patch[patch_mask == 0]
        if valid_values.size > 0:
            result[r, c] = np.median(valid_values)

    return result


def interpolate_masked_pixels_gpu(image, mask, window=1):
    # image: 2D cupy array
    # mask: 2D cupy array with 0 (valid) and 1 (masked)
    import cupy as cp

    assert image.shape == mask.shape
    assert image.ndim == 2

    result = image.copy()
    rows, cols = cp.where(mask == 1)

    for i in range(len(rows)):
        r, c = rows[i], cols[i]

        r0 = max(0, r - window)
        r1 = min(image.shape[0], r + window + 1)
        c0 = max(0, c - window)
        c1 = min(image.shape[1], c + window + 1)

        patch = image[r0:r1, c0:c1]
        patch_mask = mask[r0:r1, c0:c1]

        valid_values = patch[patch_mask == 0]
        if valid_values.size > 0:
            result[r, c] = cp.median(valid_values)

    return result.get()


def interpolate_masked_pixels_gpu_vectorized(image, mask, window=1):
    """avoids python loop and faster"""
    import cupy as cp

    H, W = image.shape
    assert image.shape == mask.shape

    result = image.copy()

    # Get coordinates of masked pixels
    ys, xs = cp.where(mask == 1)
    N = ys.shape[0]

    if N == 0:
        return result  # no masked pixels

    # Generate patch index offsets
    # Kernel Size is 2 * window + 1
    dy, dx = cp.meshgrid(cp.arange(-window, window + 1), cp.arange(-window, window + 1), indexing="ij")
    dy = dy.ravel()  # (K,)
    dx = dx.ravel()  # (K,)
    # K = dy.size

    # Broadcast and clamp patch indices
    patch_ys = cp.clip(ys[:, None] + dy[None, :], 0, H - 1)  # (N, K)
    patch_xs = cp.clip(xs[:, None] + dx[None, :], 0, W - 1)  # (N, K)

    # Flatten indices for fancy indexing
    flat_indices = patch_ys * W + patch_xs
    flat_image = image.ravel()
    flat_mask = mask.ravel()

    # Gather patch values and masks
    patch_vals = flat_image[flat_indices]  # (N, K)
    patch_mask = flat_mask[flat_indices]  # (N, K)

    # Mask out invalid pixels
    valid_vals = cp.where(patch_mask == 0, patch_vals, cp.nan)

    # Compute nanmean along axis=1 (for each patch)
    interp_vals = cp.nanmedian(valid_vals, axis=1)

    # Replace masked pixels in result
    result[ys, xs] = interp_vals

    return result


# def interpolate_masked_pixels_gpu_vectorized_weight(image, mask, weight=None, window=1):
#     """All inputs have to be cupy arrays."""
#     import cupy as cp

#     H, W = image.shape
#     assert image.shape == mask.shape
#     if weight is not None:
#         assert weight.shape == image.shape

#     result = image.copy()
#     ys, xs = cp.where(mask == 1)
#     N = ys.shape[0]
#     if N == 0:
#         return result

#     # Create patch offsets
#     dy, dx = cp.meshgrid(
#         cp.arange(-window, window + 1), cp.arange(-window, window + 1), indexing="ij"
#     )
#     dy = dy.ravel()  # shape (K,)
#     dx = dx.ravel()  # shape (K,)
#     # K = dy.size

#     # Absolute patch indices
#     patch_ys = ys[:, None] + dy[None, :]  # shape (N, K)
#     patch_xs = xs[:, None] + dx[None, :]  # shape (N, K)

#     # Mask out-of-bound locations
#     in_bounds = (patch_ys >= 0) & (patch_ys < H) & (patch_xs >= 0) & (patch_xs < W)

#     # Clip for safe indexing
#     patch_ys_safe = cp.clip(patch_ys, 0, H - 1)
#     patch_xs_safe = cp.clip(patch_xs, 0, W - 1)
#     flat_indices = patch_ys_safe * W + patch_xs_safe

#     # Fetch data
#     flat_image = image.ravel()
#     flat_mask = mask.ravel()
#     patch_vals = flat_image[flat_indices]
#     patch_mask = flat_mask[flat_indices]

#     # Mark valid values: unmasked AND in-bounds
#     valid = (patch_mask == 0) & in_bounds

#     if weight is not None:
#         flat_weight = weight.ravel()
#         patch_weights = flat_weight[flat_indices]

#         patch_weights = cp.where(valid, patch_weights, 0)
#         patch_vals = cp.where(valid, patch_vals, 0)

#         weighted_sum = cp.sum(patch_weights * patch_vals, axis=1)
#         weight_total = cp.sum(patch_weights, axis=1)
#         interp_vals = cp.where(weight_total > 0, weighted_sum / weight_total, 0)
#     else:
#         patch_vals = cp.where(valid, patch_vals, cp.nan)
#         interp_vals = cp.nanmedian(patch_vals, axis=1)

#     # Fill in interpolated values
#     result[ys, xs] = interp_vals
#     return result


def interpolate_masked_pixels_gpu_vectorized_weight(
    image,
    mask,
    weight=None,
    window=1,
    method="median",
    badpix=1,
):
    """Interpolate masked pixels in image (and optional weight map) using a mean filter.

    Args:
        image (cp.ndarray): 2D input image.
        mask (cp.ndarray): Binary mask (1 = masked, 0 = valid) when badpix=1.
        weight (cp.ndarray, optional): Pixel weight map. If None, unweighted mean is used. It must be 1/VARIANCE.
        window (int): Radius of square window.
        method (str): Interpolation method. Options are "inverse_variance" or "median".

    Returns:
        interpolated_image (cp.ndarray): Image with interpolated masked pixels.
        interpolated_weight (cp.ndarray): Weight map with interpolated weights.
    """
    import cupy as cp

    H, W = image.shape
    assert image.shape == mask.shape
    if weight is not None:
        assert weight.shape == image.shape

    result = image.copy()
    weight_result = weight.copy() if weight is not None else None  # cp.ones_like(image)

    ys, xs = cp.where(mask == badpix)
    N = ys.shape[0]
    if N == 0:
        return result, weight_result

    # Create patch offsets
    dy, dx = cp.meshgrid(cp.arange(-window, window + 1), cp.arange(-window, window + 1), indexing="ij")
    dy = dy.ravel()
    dx = dx.ravel()

    patch_ys = ys[:, None] + dy[None, :]
    patch_xs = xs[:, None] + dx[None, :]

    # Mask out-of-bound locations
    in_bounds = (patch_ys >= 0) & (patch_ys < H) & (patch_xs >= 0) & (patch_xs < W)

    # Clip for safe indexing
    patch_ys_safe = cp.clip(patch_ys, 0, H - 1)
    patch_xs_safe = cp.clip(patch_xs, 0, W - 1)
    flat_indices = patch_ys_safe * W + patch_xs_safe

    flat_image = image.ravel()
    flat_mask = mask.ravel()
    patch_vals = flat_image[flat_indices]
    patch_mask = flat_mask[flat_indices]

    valid = (patch_mask == 1 - badpix) & in_bounds

    if weight is not None:
        flat_weight = weight.ravel()
        patch_weights = flat_weight[flat_indices]

        patch_weights = cp.where(valid, patch_weights, 0)
        patch_vals = cp.where(valid, patch_vals, cp.nan)

        if method == "inverse_variance":
            weighted_sum = cp.sum(patch_weights * patch_vals, axis=1)
            weight_total = cp.sum(patch_weights, axis=1)

            # if weight_total == 0, val is 0
            interp_vals = cp.where(weight_total > 0, weighted_sum / weight_total, 0)
            interp_weights = weight_total  # assuming weight = 1/var

        elif method == "median":
            # Compute median and MAD
            interp_vals = cp.nanmedian(patch_vals, axis=1)

            # Handle NaNs safely by using cp.isclose (for float stability)
            mask = cp.isclose(patch_vals, interp_vals[:, None], equal_nan=True)

            # For multiple matches per row, take first match
            def first_match_per_row(data, mask):
                idx = cp.argmax(mask, axis=1)  # index of first True in each row
                row_idx = cp.arange(mask.shape[0])
                return data[row_idx, idx]

            interp_weights = first_match_per_row(patch_weights, mask)

        # Fill interpolated values and weights
        result[ys, xs] = interp_vals
        weight_result[ys, xs] = interp_weights

        return result, weight_result

    else:
        patch_vals = cp.where(valid, patch_vals, cp.nan)
        interp_vals = cp.nanmedian(patch_vals, axis=1)

        result[ys, xs] = interp_vals
        return result, None


def add_bpx_method(header, method):
    header["INTERP"] = (method.upper(), "Method for bad pixel interpolation")
    # swarp can't propage HIERARCH keywords
    # header["BPX_INTERP"] = (method.upper(), "Method for bad pixel interpolation")
    return header
