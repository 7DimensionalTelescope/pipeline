import os
import gc
import subprocess
import numpy as np
from astropy.io import fits
from numba import njit, prange
from ..const import SCRIPT_DIR
from scipy.stats import variation


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

        header_file = output_paths[i].replace(".fits", ".header")
        if os.path.exists(header_file):
            with open(header_file, "r") as f:
                header = fits.Header.fromstring(f.read(), sep="\n")

        fits.writeto(
            output_paths[i],
            data=data,
            header=header,
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


def compute_rms(new_data, ref_data):
    if new_data.shape != ref_data.shape:
        h = min(new_data.shape[0], ref_data.shape[0])
        w = min(new_data.shape[1], ref_data.shape[1])
        new_data = new_data[:h, :w]
        ref_data = ref_data[:h, :w]
    ref_safe = np.where(ref_data == 0, np.nan, ref_data)
    resid = (new_data / ref_safe) - 1.0
    resid = resid[np.isfinite(resid)]
    if resid.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(resid**2)))


def calculate_edge_variation(d):
    """
    Calculate edge variation by comparing median values at four edges.
    Returns (max-min)/min ratio for edge values.
    """
    try:
        h, w = d.shape

        # Define edge regions (100x100 pixels at each corner)
        edge_size = 100

        # Top-left edge
        top_left = d[:edge_size, :edge_size]
        top_left_median = np.median(top_left)

        # Top-right edge
        top_right = d[:edge_size, -edge_size:]
        top_right_median = np.median(top_right)

        # Bottom-left edge
        bottom_left = d[-edge_size:, :edge_size]
        bottom_left_median = np.median(bottom_left)

        # Bottom-right edge
        bottom_right = d[-edge_size:, -edge_size:]
        bottom_right_median = np.median(bottom_right)

        # Calculate edge values
        edge_values = [top_left_median, top_right_median, bottom_left_median, bottom_right_median]

        # Calculate (max-min)/min ratio
        min_val = min(edge_values)
        max_val = max(edge_values)

        has_negative_middle_row = np.any(d[int(h / 2.0)] <= 0)
        has_negative_middle_col = np.any(d[:, int(w / 2.0)] <= 0)

        trimmed = has_negative_middle_row or has_negative_middle_col

        if min_val > 0:  # Avoid division by zero
            edge_var = (max_val - min_val) / min_val
        else:
            edge_var = 0.0

        return edge_var, edge_values, trimmed

    except Exception as e:
        print(f"Error calculating edge variation: {e}")
        return 0.0, [0, 0, 0, 0], False


def uniformity_statistical(fits_path, bpmask_path=None, grid_size=32):
    """
    Fast uniformity checking using statistical grid analysis.
    This divides the image into a grid and analyzes statistics of each cell.
    Fast and gives spatial uniformity information.
    Returns log10(uniformity_score) for easier interpretation.
    """

    # Load data
    d = fits.getdata(fits_path).astype(np.float32)

    if bpmask_path is None:
        bpmask_path = fits_path.replace("dark", "bpmask")
    bpm = fits.getdata(bpmask_path).astype(np.float32)

    if d.ndim != 2 or bpm.ndim != 2 or d.shape != bpm.shape:
        raise ValueError(f"Shape mismatch: data {d.shape}, bpmask {bpm.shape}")

    # BPM mask
    hot_mask = bpm != 0

    # Auto-shift to positive
    dmin = float(d.min())
    if dmin <= 0:
        d = d - dmin + 1e-8

    # Apply mask
    d_masked = d.copy()
    d_masked[hot_mask] = np.nan

    h, w = d_masked.shape
    cell_h, cell_w = h // grid_size, w // grid_size

    # Analyze each grid cell
    cell_means = []
    cell_stds = []

    for i in range(grid_size):
        for j in range(grid_size):
            start_h = i * cell_h
            end_h = (i + 1) * cell_h if i < grid_size - 1 else h
            start_w = j * cell_w
            end_w = (j + 1) * cell_w if j < grid_size - 1 else w

            cell_data = d_masked[start_h:end_h, start_w:end_w]
            valid_cell = cell_data[~np.isnan(cell_data)]

            if len(valid_cell) > 0:
                cell_means.append(np.mean(valid_cell))
                cell_stds.append(np.std(valid_cell))

    # Calculate uniformity metrics
    cell_means = np.array(cell_means)
    cell_stds = np.array(cell_stds)

    # Spatial uniformity (variation between cells)
    spatial_cv = variation(cell_means)

    # Local uniformity (average variation within cells)
    local_cv = np.mean(cell_stds / (cell_means + 1e-10))

    # Overall uniformity score
    uniformity_score = spatial_cv + 0.5 * local_cv

    # Apply log10 for easier interpretation
    log_uniformity_score = np.log10(uniformity_score + 1e-10)

    return float(log_uniformity_score)


def record_statistics(filename, header, device_id=0, cropsize=500, dtype=None):
    data = fits.getdata(filename).astype(np.float32)  # Ensure data is float32
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

    if dtype.upper() == "FLAT":
        datasig = fits.getdata(filename.replace("flat_", "flatsig_")).astype(np.float32)  # Ensure data is float32
        s_mean, s_median, s_std = sigma_clipped_stats(datasig, device_id=device_id, sigma=3, maxiters=5)

        header["SIGMEAN"] = (s_mean, "3-sig clipped mean of the errormap")
        header["SIGMED"] = (s_median, "3-sig clipped median of the errormap")
        header["SIGSTD"] = (s_std, "3-sig clipped std of the errormap")

        edge_var, _, trimmed = calculate_edge_variation(data)
        header["EDGEVAR"] = (edge_var, "Edge variation of the image")
        header["TRIMMED"] = (trimmed, "Non-positive values in the middle of the image")

    elif dtype.upper() == "DARK":
        uniformity_score = uniformity_statistical(filename)
        header["UNIFORM"] = (uniformity_score, "Uniformity score")

    return header


# not used
def delta_edge_center(data, check_size=100):
    h, w = data.shape
    s = check_size  # Focus on size 100
    s_eff = int(min(s, h // 4, w // 4))

    # Square regions: top-left, top-right, bottom-left, bottom-right
    top_left = data[:s_eff, :s_eff]
    top_right = data[:s_eff, -s_eff:]
    bottom_left = data[-s_eff:, :s_eff]
    bottom_right = data[-s_eff:, -s_eff:]
    # Center square
    cy, cx = h // 2, w // 2
    hs = s_eff // 2
    cy0, cy1 = max(0, cy - hs), min(h, cy + hs)
    cx0, cx1 = max(0, cx - hs), min(w, cx + hs)
    center = data[cy0:cy1, cx0:cx1]

    edge_means = np.array([np.median(top_left), np.median(top_right), np.median(bottom_left), np.median(bottom_right)])
    edge_avg = max(edge_means)
    center_mean = center.mean()
    delta = edge_avg - center_mean
    return delta
