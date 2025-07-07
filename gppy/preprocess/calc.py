import gc
import cupy as cp
import numpy as np
import os
import copy
from astropy.io import fits
from cupy.cuda import Stream
from contextlib import contextmanager

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


def process_image_with_cupy(image_paths, bias, dark, flat, device_id=0, output_paths=None, header=None, **kwargs):
    output = []
    h, w = fits.getdata(image_paths[0]).shape

    with cp.cuda.Device(device_id):
        before = cp.get_default_memory_pool().used_bytes()
        pinned_mem = cp.cuda.alloc_pinned_memory(h * w * 4)  # float32: 4 bytes
        cpu_buffer = np.frombuffer(pinned_mem, dtype=np.float32).reshape(h, w)
        
        gpu_bias = cp.asarray(bias, dtype=cp.float32)
        gpu_dark = cp.asarray(dark, dtype=cp.float32) 
        gpu_flat = cp.asarray(flat, dtype=cp.float32)

        gpu_buffer = cp.empty((h, w), dtype=cp.float32)

        for i, o in enumerate(image_paths):
            cpu_data = fits.getdata(o).astype(np.float32)
            cpu_buffer[:] = cpu_data  # copy into pinned buffer

            # Copy to GPU
            cp.cuda.runtime.memcpyAsync(
                gpu_buffer.data.ptr,
                pinned_mem.ptr,
                gpu_buffer.nbytes,
                cp.cuda.runtime.memcpyHostToDevice,
                cp.cuda.Stream.null.ptr
            )

            cp.cuda.Stream.null.synchronize()

            gpu_buffer[:] = reduction_kernel(gpu_buffer, gpu_bias, gpu_dark, gpu_flat)

            # GPU → CPU (to pinned buffer)
            cp.cuda.runtime.memcpyAsync(
                pinned_mem.ptr,
                gpu_buffer.data.ptr,
                gpu_buffer.nbytes,
                cp.cuda.runtime.memcpyDeviceToHost,
                cp.cuda.Stream.null.ptr
            )

            cp.cuda.Stream.null.synchronize()

            if output_paths is not None:         
                os.makedirs(os.path.dirname(output_paths[i]), exist_ok=True)
                fits.writeto(
                    output_paths[i],
                    data=cpu_buffer,
                    header=header[i],
                    overwrite=True,
                )
            else:
                output.append(cpu_buffer.copy())  # still on CPU

        # Final cleanup
        del gpu_buffer, gpu_bias, gpu_dark, gpu_flat
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        after = cp.get_default_memory_pool().used_bytes()
        memory_leakage = (after - before) / 1024 / 1024

    gc.collect()
    return output, memory_leakage

def process_image_with_cpu(
    image_paths: str,
    bias: np.array,
    dark: np.array,
    flat: np.array,
    subtract=None,
    normalize=False,
    output_paths: list = None,
    header: list = None,
    **kwargs,
):

    def read_fits_image(path):
        return fits.getdata(path).astype(np.float32)

    bias = bias.astype(np.float32)
    dark = dark.astype(np.float32)
    flat = flat.astype(np.float32)
    subtract = subtract.astype(np.float32) if subtract is not None else None

    output = []
    h, w = fits.getdata(image_paths[0]).shape
    cpu_buffer = np.empty((h, w))

    for i, image in enumerate(image_paths):
        cpu_buffer[:] = read_fits_image(image)
        cpu_buffer[:] = reduction_kernel_cpu(cpu_buffer, bias, dark, flat, subtract, normalize)
        if output_paths is not None:         
            os.makedirs(os.path.dirname(output_paths[i]), exist_ok=True)
            fits.writeto(
                output_paths[i],
                data=cpu_buffer,
                header=header[i],
                overwrite=True,
            )
        else:
            output.append(cpu_buffer.copy())
            
    del cpu_buffer, bias, dark, flat, subtract
    gc.collect()
    return output, None

# Combine images
def combine_images_with_cupy(images: str, device_id=None, subtract=None, norm=False):
    """median is gpu, std is cpu"""
    
    with cp.cuda.Device(device_id):
        before = cp.get_default_memory_pool().used_bytes()

        cp_stack = cp.stack([cp.asarray(fits.getdata(img).astype(np.float32)) for img in images])
        if subtract is not None:
            cp_subtract = cp.asarray(subtract)
            cp_stack -= cp_subtract
            del cp_subtract
        if norm:
            cp_stack /= cp.median(cp_stack, axis=(1, 2), keepdims=True)
        cp_median = cp.median(cp_stack, axis=0)
        cp_std = cp.std(cp_stack, axis=0, ddof=1)
        np_std = cp_std.get()
        np_median = cp_median.get()

        del cp_stack, cp_median, cp_std
        cp.get_default_memory_pool().free_all_blocks()
        after = cp.get_default_memory_pool().used_bytes()
        memory_leakage = (after - before)/1024/1024
    gc.collect()
    
    return np_median, np_std, memory_leakage


def combine_images_with_cpu(images: list, subtract=None, norm=False, **kwargs):
    np_stack = np.stack([fits.getdata(img) for img in images])
    if subtract is not None:
        np_stack = np_stack - subtract  # avoid in-place for numba safety

    if norm:
        np_stack = _normalize_stack(np_stack)

    np_median, np_std = _calc_median_and_std(np_stack)
    return np_median, np_std, None


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


def sigma_clipped_stats_cpu(data, sigma=3.0, maxiters=5, minmax=False, hot_mask=False, hot_mask_sigma=5.0):
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


def sigma_clipped_stats_cupy(cp_data, device_id=0, sigma=3, maxiters=5, minmax=False, hot_mask=False, hot_mask_sigma=5):
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
        cp_data = cp.asarray(cp_data)
        cp_data_flat = cp_data.ravel()
        for _ in range(maxiters):
            median_val = cp.median(cp_data_flat)
            std_val = cp.std(cp_data_flat)
            # Keep only pixels within +/- sigma * std of the median
            mask = cp.abs(cp_data_flat - median_val) < (sigma * std_val)
            cp_data_flat = cp_data_flat[mask]
            del mask
        
        # Final statistics on the clipped data
        mean_val = cp.mean(cp_data_flat)
        median_val = cp.median(cp_data_flat)
        std_val = cp.std(cp_data_flat)
        min_val = cp.min(cp_data_flat)
        max_val = cp.max(cp_data_flat)

        if hot_mask:
            hot_mask_arr = cp.abs(cp_data - median_val) > hot_mask_sigma * std_val  # 1 for bad, 0 for okay
            hot_mask_arr[:] = hot_mask_arr.astype(cp.uint8)  # Convert to uint8
            hot_mask_arr_numpy = cp.asnumpy(hot_mask_arr).astype(np.uint8)  # Convert to numpy array
            del hot_mask_arr

        mean_val_cpu = mean_val.get()
        median_val_cpu = median_val.get()
        std_val_cpu = std_val.get()
        min_val_cpu = min_val.get()
        max_val_cpu = max_val.get()
        del cp_data_flat, cp_data
        del mean_val, median_val, std_val, min_val, max_val
        cp.get_default_memory_pool().free_all_blocks()

    gc.collect()
    
    if hot_mask:
        return hot_mask_arr_numpy
    if minmax:
        return mean_val_cpu, median_val_cpu, std_val_cpu, min_val_cpu, max_val_cpu
    else:
        return mean_val_cpu, median_val_cpu, std_val_cpu


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

    return header
