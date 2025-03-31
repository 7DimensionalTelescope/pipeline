def interpolate_masked_pixels_cpu(image, mask, window=1):
    # image: 2D cupy array
    # mask: 2D cupy array with 0 (valid) and 1 (masked)
    import numpy as np

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
            result[r, c] = np.mean(valid_values)

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
            result[r, c] = cp.mean(valid_values)

    return result


def interpolate_masked_pixels_gpu_batched(image, mask, window=1):
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
    dy, dx = cp.meshgrid(
        cp.arange(-window, window + 1), cp.arange(-window, window + 1), indexing="ij"
    )
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


def interpolate_masked_pixels_gpu_batched_weight(image, mask, window=1, weight=None):
    import cupy as cp

    H, W = image.shape
    assert image.shape == mask.shape
    if weight is not None:
        assert weight.shape == image.shape

    result = image.copy()
    ys, xs = cp.where(mask == 1)
    N = ys.shape[0]
    if N == 0:
        return result

    # Create patch offsets
    dy, dx = cp.meshgrid(
        cp.arange(-window, window + 1), cp.arange(-window, window + 1), indexing="ij"
    )
    dy = dy.ravel()  # shape (K,)
    dx = dx.ravel()  # shape (K,)
    K = dy.size

    # Absolute patch indices
    patch_ys = ys[:, None] + dy[None, :]  # shape (N, K)
    patch_xs = xs[:, None] + dx[None, :]  # shape (N, K)

    # Mask out-of-bound locations
    in_bounds = (patch_ys >= 0) & (patch_ys < H) & (patch_xs >= 0) & (patch_xs < W)

    # Clip for safe indexing
    patch_ys_safe = cp.clip(patch_ys, 0, H - 1)
    patch_xs_safe = cp.clip(patch_xs, 0, W - 1)
    flat_indices = patch_ys_safe * W + patch_xs_safe

    # Fetch data
    flat_image = image.ravel()
    flat_mask = mask.ravel()
    patch_vals = flat_image[flat_indices]
    patch_mask = flat_mask[flat_indices]

    # Mark valid values: unmasked AND in-bounds
    valid = (patch_mask == 0) & in_bounds

    if weight is not None:
        flat_weight = weight.ravel()
        patch_weights = flat_weight[flat_indices]

        patch_weights = cp.where(valid, patch_weights, 0)
        patch_vals = cp.where(valid, patch_vals, 0)

        weighted_sum = cp.sum(patch_weights * patch_vals, axis=1)
        weight_total = cp.sum(patch_weights, axis=1)
        interp_vals = cp.where(weight_total > 0, weighted_sum / weight_total, 0)
    else:
        patch_vals = cp.where(valid, patch_vals, cp.nan)
        interp_vals = cp.nanmean(patch_vals, axis=1)

    # Fill in interpolated values
    result[ys, xs] = interp_vals
    return result
