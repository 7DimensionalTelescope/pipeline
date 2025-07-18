import os
import gc
import subprocess
import numpy as np
from astropy.io import fits
from numba import njit, prange
from ..const import SCRIPT_DIR


def read_fits_image(path):
    return fits.getdata(path).astype(np.float32)


def combine_images_with_subprocess_gpu(
    images,
    output,
    sig_output,
    device_id=0,
    subtract=None,
    norm=False,
    scale=None,
    make_bpmask=None,
    bpmask_sigma=5,
    **kwargs,
):
    """
    Combine images using a subprocess call to a CUDA-accelerated script.
    """
    cmd = [
        "python",
        f"{SCRIPT_DIR}/cuda/combine_images.py",
        # f"{SCRIPT_DIR}/cuda/combine_images",
        "-input",
        *images,
        "-device",
        str(device_id),
    ]

    if subtract is not None:
        cmd.extend(["-subtract", *subtract])
        cmd.extend(["-scales", *map(str, scale)])

    if norm:
        cmd.append("-norm")

    cmd.extend(["-median_out", output])
    cmd.extend(["-std_out", sig_output])

    if make_bpmask is not None:
        cmd.extend(["-bpmask", make_bpmask])
        cmd.extend(["-bpmask_sigma", str(bpmask_sigma)])

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Error combining images: {result.stderr}")

    return None


def combine_images_with_cpu(
    images,
    output,
    sig_output,
    subtract=None,
    scale=None,
    norm=False,
    make_bpmask: str = None,
    bpmask_sigma=5,
    **kwargs,  # prevent crash if extra args are passed. e.g., device_id
):
    gc.collect()
    np_stack = np.stack([read_fits_image(img) for img in images])
    if subtract is not None:
        sub_arr = np.zeros_like(np_stack[0], dtype=np.float32)
        for i, sub in enumerate(subtract):
            if isinstance(sub, str):
                sub = read_fits_image(sub)
                sub = sub * scale[i]
            elif isinstance(sub, np.ndarray):
                sub = sub.astype(np.float32) * scale[i]
            else:
                raise ValueError("Subtract must be a FITS file path or a numpy array.")
            sub_arr += sub
        np_stack = np_stack - sub_arr  # avoid in-place for numba safety

    if norm:
        np_stack = _normalize_stack(np_stack)

    np_median, np_std = _calc_median_and_std(np_stack)  # coadded image and std
    fits.writeto(output, data=np_median, overwrite=True)
    fits.writeto(sig_output, data=np_std, overwrite=True)

    if make_bpmask is not None:
        hot_mask = sigma_clipped_stats_cpu(np_median, bpmask_sigma, return_mask=True)
        fits.writeto(make_bpmask, data=hot_mask.astype(np.uint8), overwrite=True)

    return np_median, np_std, None


def process_image_with_subprocess_gpu(image_paths, bias, dark, flat, device_id=0, output_paths=None, **kwargs):

    gc.collect()

    # if len(image_paths) > 20:
    #     module = "process_image_batch"
    # else:
    #     module = "process_image"
    cmd = [
        # f"{SCRIPT_DIR}/cuda/{module}",
        "python",
        f"{SCRIPT_DIR}/cuda/process_image.py",
        "-bias",
        bias,
        "-dark",
        dark,
        "-flat",
        flat,
        "-input",
        *image_paths,
        "-output",
        *output_paths,
        "-device",
        str(device_id),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Error processing images: {result.stderr}")
    return None


def process_image_with_cpu(
    image_paths: str,
    bias: str,
    dark: str,
    flat: str,
    output_paths: list = None,
    **kwargs,
):

    bias = read_fits_image(bias)
    dark = read_fits_image(dark)
    flat = read_fits_image(flat)

    h, w = fits.getdata(image_paths[0]).shape
    data = None

    for i, image in enumerate(image_paths):
        if data is None:
            data = read_fits_image(image)
        else:
            data[:] = read_fits_image(image)
        data = reduction_kernel_cpu(data, bias, dark, flat)
        os.makedirs(os.path.dirname(output_paths[i]), exist_ok=True)

        fits.writeto(
            output_paths[i],
            data=data,
            overwrite=True,
        )

    del bias, dark, flat
    gc.collect()
    return None


@njit(parallel=True)
def reduction_kernel_cpu(image, bias, dark, flat):
    h, w = image.shape
    corrected = np.empty_like(image)

    for i in prange(h):
        for j in range(w):
            val = image[i, j] - bias[i, j] - dark[i, j]
            val /= flat[i, j]
            corrected[i, j] = val

    return corrected


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
    return sigma_clipped_stats_cpu(np_data, **kwargs)
    # else:
    #     return sigma_clipped_stats_cupy(np_data, device_id=device_id, **kwargs)


def sigma_clipped_stats_cpu(data, sigma=3.0, maxiters=5, minmax=False, return_mask=False, bpmask_sigma=5.0):
    fdata = data.ravel()

    for _ in range(int(5)):
        median_val = np.mean(fdata)
        std_val = np.std(fdata, ddof=1)
        mask = np.abs(fdata - median_val) < (3 * std_val)
        fdata = fdata[mask]

    clipped = fdata  # [mask]

    if clipped.size == 0:
        mean_val = 0.0
        median_val = 0.0
        std_val = 0.0
    else:
        mean_val = np.mean(clipped)
        median_val = np.median(clipped)
        std_val = np.std(clipped)

    if return_mask:
        return _compute_outlier_mask_2d(data, median_val, std_val, bpmask_sigma)

    if minmax:
        return mean_val, median_val, std_val, np.min(fdata), np.max(fdata)

    return mean_val, median_val, std_val


@njit(parallel=True)
def _compute_outlier_mask_2d(data, median, std, hot_sigma):
    H, W = data.shape
    mask = np.empty((H, W), dtype=np.uint8)
    threshold = hot_sigma * std
    for i in prange(H):
        for j in range(W):
            mask[i, j] = 1 if abs(data[i, j] - median) > threshold else 0
    return mask


def record_statistics(data, header, device_id=0, cropsize=500):
    data = fits.getdata(data).astype(np.float32)  # Ensure data is float32
    mean, median, std, min, max = sigma_clipped_stats(data, device_id=device_id, sigma=3, maxiters=5, minmax=True)
    header["CLIPMEAN"] = (float(mean), "3-sig clipped mean of the pixel values")
    header["CLIPMED"] = (float(median), "3-sig clipped median of the pixel values")
    header["CLIPSTD"] = (float(std), "3-sig clipped standard deviation of the pixels")
    header["CLIPMIN"] = (float(min), "3-sig clipped minimum of the pixel values")
    header["CLIPMAX"] = (float(max), "3-sig clipped maximum of the pixel values")

    # minmax again... inefficient but insignificant
    header["UNCLPMIN"] = (float(np.min(data)), "unclipped minimum of the pixel values")
    header["UNCLPMAX"] = (float(np.max(data)), "unclipped maximum of the pixel values")

    # Slice the central 500x500 area
    height, width = data.shape
    start_row = (height - cropsize) // 2
    start_col = (width - cropsize) // 2
    cropped_data = data[start_row : start_row + cropsize, start_col : start_col + cropsize]
    mean, median, std = sigma_clipped_stats(cropped_data, device_id=device_id, sigma=3, maxiters=5)
    header["CENCLPMN"] = (float(mean), f"3-sig clipped mean of center {cropsize}x{cropsize}")  # fmt: skip
    header["CENCLPMD"] = (float(median), f"3-sig clipped median of center {cropsize}x{cropsize}")  # fmt: skip
    header["CENCLPSD"] = (float(std), f"3-sig clipped std of center {cropsize}x{cropsize}")  # fmt: skip

    # header["CENCMIN"] = float(min)
    # header["CENCMAX"] = float(max)

    return header
