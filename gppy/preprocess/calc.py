import gc
import cupy as cp
import numpy as np
from astropy.io import fits
from cupy.cuda import Stream
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from numba import njit, prange

# Reduction kernel
reduction_kernel = cp.ElementwiseKernel(
    in_params="T x, T b, T d, T f", out_params="T z", operation="z = (x - b - d) / f", name="reduction"
)

@njit(parallel=True)
def reduction_kernel_cpu(image, bias, dark, flat, subtract, normalize):
    h, w = image.shape
    corrected = np.empty_like(image)
    
    for i in prange(h):
        for j in range(w):
            val = image[i, j] - bias[i, j] - dark[i, j]
            if subtract is not None:
                val -= subtract[i, j]
            val /= flat[i, j]
            corrected[i, j] = val

    if normalize:
        median = np.median(corrected.ravel())
        if median != 0:
            corrected /= median

    return corrected

@contextmanager
def load_data_gpu(fpath, ext=None):
    """Load data into GPU memory with automatic cleanup."""
    data = cp.asarray(fits.getdata(fpath, ext=ext), dtype="float32")
    try:
        yield data  # Provide the loaded data to the block
    finally:
        del data  # Free GPU memory when the block is exited
        gc.collect()  # Force garbage collection
        cp.get_default_memory_pool().free_all_blocks()


def calc_batch_dist(image_list, num_devices=None, use_multi_device=False, device=0):
    if use_multi_device:
        num_devices = cp.cuda.runtime.getDeviceCount()
    else:
        num_devices = 1

    # Step 1: Estimate how many images each GPU can handle
    max_batch_per_device = []
    if num_devices > 1:
        for i in range(num_devices):
            with cp.cuda.Device(i):
                max_batch = estimate_posssible_batch_size(image_list[0])[0]
                max_batch_per_device.append(max_batch)
    else:
        with cp.cuda.Device(device):
            max_batch = estimate_posssible_batch_size(image_list[0])[0]
            max_batch_per_device.append(max_batch)

    # Step 2: Compute initial proportional distribution
    total_capacity = sum(max_batch_per_device)
    ratio = [b / total_capacity for b in max_batch_per_device]
    initial_dist = np.floor(np.array(ratio) * len(image_list)).astype(int)

    # Step 3: Adjust for rounding errors
    remaining = len(image_list) - sum(initial_dist)
    for i in range(remaining):
        initial_dist[i % num_devices] += 1

    # Step 4: First-round distribution (capped at each GPU's capacity)
    first_round = np.minimum(initial_dist, max_batch_per_device)
    overflow = initial_dist - first_round

    # Step 5: Redistribute any remaining overflow (that still hasn't been assigned)
    extra_needed = len(image_list) - sum(first_round)
    second_round = np.zeros(num_devices, dtype=int)

    i = 0
    while extra_needed > 0:
        capacity = max_batch_per_device[i]
        available = capacity - second_round[i]
        assign = min(available, overflow[i], extra_needed)
        second_round[i] += assign
        extra_needed -= assign
        i = (i + 1) % num_devices

    return np.vstack([first_round, second_round])


def estimate_posssible_batch_size(filename):
    H, W = fits.getdata(filename).astype(np.float32).shape
    image_size = cp.float32().nbytes * H * W / 1024**2
    available_mem = cp.cuda.runtime.memGetInfo()[0] / 1024**2
    safe_mem = int(available_mem * 0.7)
    batch_size = max(1, safe_mem // image_size)
    return int(batch_size), image_size, available_mem


def process_batch_on_device_with_cupy(image_paths, bias, dark, flat, results, device_id=0):

    with cp.cuda.Device(device_id):
        load_stream = Stream(non_blocking=True)
        compute_stream = Stream()

        local_results = []
        prev_image = None
        for img_path in image_paths:
            with load_stream:
                curr_image = cp.asarray(fits.getdata(img_path).astype('float32'))
            
            if prev_image is not None:
                with compute_stream:
                    reduced = reduction_kernel(prev_image, bias, dark, flat)
                    local_results.append(cp.asnumpy(reduced))
                del prev_image

            load_stream.synchronize()
            prev_image = curr_image

        if prev_image is not None:
            with compute_stream:
                reduced = reduction_kernel(prev_image, bias, dark, flat)
                local_results.append(cp.asnumpy(reduced))
        compute_stream.synchronize()
    
        del bias, dark, flat, prev_image, curr_image, reduced

        cp.get_default_memory_pool().free_all_blocks()
        if np.shape(results) == (1,):
            results[0] = local_results
        else:
            results[device_id] = local_results

def process_batch_on_device_with_cpu(image_paths, bias, dark, flat, results,
                             device_id=0, max_workers=4, subtract=None, normalize=False):

    def read_fits_image(path):
        return fits.getdata(path).astype(np.float32)

    bias = bias.astype(np.float32)
    dark = dark.astype(np.float32)
    flat = flat.astype(np.float32)
    subtract = subtract.astype(np.float32) if subtract is not None else None

    local_results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        image_iter = executor.map(read_fits_image, image_paths)

        for curr_image in image_iter:
            reduced = reduction_kernel_cpu(curr_image, bias, dark, flat, subtract, normalize)
            local_results.append(reduced.copy())
            del reduced

    if np.shape(results) == (1,):
        results[0] = local_results
    else:
        results[device_id] = local_results

# Combine images
def combine_images_with_cupy(images: str, device_id=None, subtract=None, norm=False):
    """median is gpu, std is cpu"""
    with cp.cuda.Device(device_id):
        cp_stack = cp.stack([cp.asarray(fits.getdata(img).astype(np.float32)) for img in images])
        if subtract is not None:
            cp_stack -= subtract
        if norm:
            cp_stack /= cp.median(cp_stack, axis=(1, 2), keepdims=True)
        cp_median = cp.median(cp_stack, axis=0)
        cp_std = cp.std(cp_stack, axis=0, ddof=1)
        np_std = cp.asnumpy(cp_std)

        del cp_stack, cp_std, subtract
    cp.get_default_memory_pool().free_all_blocks()
    return cp_median, np_std

def combine_images_with_cpu(images: list, subtract=None, norm=False, **kwargs):
    np_stack = np.stack([fits.getdata(img) for img in images])
    if subtract is not None:
        np_stack = np_stack - subtract  # avoid in-place for numba safety

    if norm:
        np_stack = _normalize_stack(np_stack)

    np_median, np_std = _calc_median_and_std(np_stack)
    return np_median, np_std

@njit(parallel=True)
def _normalize_stack(np_stack):
    n, h, w = np_stack.shape
    output = np.empty_like(np_stack)

    for i in prange(n):
        flattened = np_stack[i].ravel()
        med = np.median(flattened)
        if med == 0:
            med = 1e-8  # avoid division by zero
        output[i] = np_stack[i] / med

    return output

@njit(parallel=True)
def _calc_median_and_std(np_stack):
    H, W = np_stack.shape[1], np_stack.shape[2]
    n = np_stack.shape[0]

    median_img = np.empty((H, W), dtype=np.float32)
    std_img = np.empty((H, W), dtype=np.float32)

    for i in prange(H):
        for j in prange(W):
            pixel_series = np_stack[:, i, j]
            med = np.median(pixel_series)

            mean = np.mean(pixel_series)
            var = 0.0
            for k in range(n):
                diff = pixel_series[k] - mean
                var += diff * diff
            std = np.sqrt(var / (n - 1))

            median_img[i, j] = med
            std_img[i, j] = std

    return median_img, std_img

# Sigma Clipped Statistics
def sigma_clipped_stats(np_data, device_id=0, **kwargs):
    if device_id == "CPU":
        return sigma_clipped_stats_cpu(np_data, **kwargs)
    else:
        return sigma_clipped_stats_cupy(np_data, device_id=device_id, **kwargs)

def sigma_clipped_stats_cpu(data, sigma=3.0, maxiters=5,
                               minmax=False, hot_mask=False, hot_mask_sigma=5.0):
    flat = data.ravel()
    mask = _sigma_clip_1d(flat, sigma, maxiters)
    clipped = flat[mask]

    if clipped.size == 0:
        mean_val = 0.0
        median_val = 0.0
        std_val = 0.0
    else:
        mean_val = np.mean(clipped)
        median_val = np.median(clipped)
        std_val = np.std(clipped)

    if hot_mask:
        return _compute_hot_mask_2d(data, median_val, std_val, hot_mask_sigma)

    if minmax:
        return mean_val, median_val, std_val, np.min(flat), np.max(flat)

    return mean_val, median_val, std_val

@njit
def _sigma_clip_1d(data_flat, sigma=3.0, maxiters=5):
    mask = np.ones(data_flat.shape, dtype=np.bool_)
    for _ in range(maxiters):
        clipped = data_flat[mask]
        if clipped.size == 0:
            break
        median = np.median(clipped)
        std = np.std(clipped)
        if std == 0.0:
            break
        for i in range(data_flat.size):
            mask[i] = abs(data_flat[i] - median) < sigma * std
    return mask

@njit(parallel=True)
def _compute_hot_mask_2d(data, median, std, hot_sigma):
    H, W = data.shape
    mask = np.empty((H, W), dtype=np.uint8)
    threshold = hot_sigma * std
    for i in prange(H):
        for j in range(W):
            mask[i, j] = 1 if abs(data[i, j] - median) > threshold else 0
    return mask

def sigma_clipped_stats_cupy(cp_data, device_id=0, sigma=3, maxiters=5, minmax=False, 
                            hot_mask=False, hot_mask_sigma=5):
    """
    Approximate sigma-clipping using CuPy.
    Computes mean, median, and std after iteratively removing outliers
    beyond 'sigma' standard deviations from the median.

    Parameters
    ----------
    cp_data : cupy.ndarray
        Flattened CuPy array of image pixel values.
    sigma : float
        Clipping threshold in terms of standard deviations.
    maxiters : int
        Maximum number of clipping iterations.

    Returns
    -------
    mean_val : float
        Mean of the clipped data (as a GPU float).
    median_val : float
        Median of the clipped data (as a GPU float).
    std_val : float
        Standard deviation of the clipped data (as a GPU float).
    """
    with cp.cuda.Device(device_id):
        # Flatten to 1D for global clipping
        cp_data = cp_data.ravel()

        for _ in range(maxiters):
            median_val = cp.median(cp_data)
            std_val = cp.std(cp_data)
            # Keep only pixels within +/- sigma * std of the median
            mask = cp.abs(cp_data - median_val) < (sigma * std_val)
            cp_data = cp_data[mask]

        # Final statistics on the clipped data
        mean_val = float(cp.mean(cp_data))
        median_val = float(cp.median(cp_data))
        std_val = float(cp.std(cp_data))
        min_val = float(cp.min(cp_data))
        max_val = float(cp.max(cp_data))

        if hot_mask:
            hot_mask_arr = cp.abs(cp_data - median_val) > hot_mask_sigma * std_val  # 1 for bad, 0 for okay
            hot_mask_arr = cp.asnumpy(hot_mask_arr).astype("uint8")
    
    del cp_data
    cp.get_default_memory_pool().free_all_blocks()

    if hot_mask:
        return hot_mask_arr
    if minmax:
        return mean_val, median_val, std_val, min_val, max_val
    return mean_val, median_val, std_val


def record_statistics(data, header, device_id=0, cropsize=500):
    mean, median, std, min, max = sigma_clipped_stats(
        data, device_id=device_id, sigma=3, maxiters=5, minmax=True
    )  # gpu vars
    header["CLIPMEAN"] = (float(mean), "3-sig clipped mean of the pixel values")
    header["CLIPMED"] = (float(median), "3-sig clipped median of the pixel values")
    header["CLIPSTD"] = (float(std), "3-sig clipped standard deviation of the pixels")
    header["CLIPMIN"] = (float(min), "3-sig clipped minimum of the pixel values")
    header["CLIPMAX"] = (float(max), "3-sig clipped maximum of the pixel values")

    height, width = data.shape
    start_row = (height - cropsize) // 2
    start_col = (width - cropsize) // 2

    # Slice the central 500x500 area
    cropped_data = data[start_row : start_row + cropsize, start_col : start_col + cropsize]
    mean, median, std = sigma_clipped_stats(cropped_data, device_id=device_id, sigma=3, maxiters=5)
    header["CENCLPMN"] = (float(mean), f"3-sig clipped mean of center {cropsize}x{cropsize}")  # fmt: skip
    header["CENCLPMD"] = (float(median), f"3-sig clipped median of center {cropsize}x{cropsize}")  # fmt: skip
    header["CENCLPSD"] = (float(std), f"3-sig clipped std of center {cropsize}x{cropsize}")  # fmt: skip

    # header["CENCMIN"] = float(min)
    # header["CENCMAX"] = float(max)

    del cropped_data, mean, median, std, min, max, data
    cp.get_default_memory_pool().free_all_blocks()
    return header
