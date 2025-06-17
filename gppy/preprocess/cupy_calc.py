import gc
import cupy as cp
import numpy as np
from astropy.io import fits
from cupy.cuda import Stream
from contextlib import contextmanager

# Reduction kernel
reduction_kernel = cp.ElementwiseKernel(
    in_params="T x, T b, T d, T f", out_params="T z", operation="z = (x - b - d) / f", name="reduction"
)


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


def process_batch_on_device(image_paths, bias, dark, flat, results, device_id=0):

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


def sigma_clipped_stats_cupy(cp_data, device_id=0, sigma=3, maxiters=5, minmax=False):
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
    
    del cp_data
    cp.get_default_memory_pool().free_all_blocks()

    # Convert results back to Python floats on the CPU
    # return float(mean_val), float(median_val), float(std_val)
    if minmax:
        return mean_val, median_val, std_val, min_val, max_val
    return mean_val, median_val, std_val


def record_statistics(data, header, device_id=0, cropsize=500):
    mean, median, std, min, max = sigma_clipped_stats_cupy(
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
    mean, median, std = sigma_clipped_stats_cupy(cropped_data, device_id=device_id, sigma=3, maxiters=5)
    header["CENCLPMN"] = (float(mean), f"3-sig clipped mean of center {cropsize}x{cropsize}")  # fmt: skip
    header["CENCLPMD"] = (float(median), f"3-sig clipped median of center {cropsize}x{cropsize}")  # fmt: skip
    header["CENCLPSD"] = (float(std), f"3-sig clipped std of center {cropsize}x{cropsize}")  # fmt: skip

    # header["CENCMIN"] = float(min)
    # header["CENCMAX"] = float(max)

    del cropped_data, mean, median, std, min, max, data
    cp.get_default_memory_pool().free_all_blocks()
    return header
