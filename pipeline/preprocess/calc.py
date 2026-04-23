import os
import gc
import subprocess
from typing import Literal
import numpy as np
import tempfile
import time
import uuid
import fitsio
from astropy.io import fits
from numba import njit, prange
from scipy.stats import variation

from .utils import (
    SHIFTED_SCORE_THRESHOLD,
    combined_shifted_score,
    load_header_file,
    prepare_raw_qa_header,
    read_fits_image,
    read_fits_images,
    update_header_file,
    write_fits_images,
)
from ..const import SOURCE_DIR, SERVICES_TMP_DIR


def shifted_overscan_score(
    image: np.ndarray,
    row_frac: float = 0.6,
    profile_max_rows: int = 384,
    win_baseline: int = 801,
    win_smooth: int = 8,
    plateau_win: int = 16,
    plateau_range_max: float = 5.0,
    plateau_qlo: float = 0.1,
    plateau_qhi: float = 0.9,
    plateau_min_run: int = 5,
    pair_dmin: int = 5,
    pair_dmax: int = 50,
    pair_sigma_min: float = 8.0,
    dip_abs_min: float = 10.0,
    noise_floor: float = 1.0,
) -> float:
    from numpy.lib.stride_tricks import sliding_window_view

    # Use the middle row_frac of rows to reduce influence of row-edge artifacts
    # and source concentrations at detector edges on the column-mean profile.
    h = image.shape[0]
    r0 = int(h * (1.0 - row_frac) / 2.0)
    r1 = h - r0
    band = image[r0:r1]
    nr = band.shape[0]
    step = max(1, (nr + profile_max_rows - 1) // profile_max_rows)
    sample = band[::step]
    # Per-column mean is biased high in crowded fields; median tracks the sky
    # floor. SExtractor's mode proxy (2.5*median - 1.5*mean) uses both, with no
    # extra passes beyond mean+median on the decimated band. Clip to [lo, hi]
    # with lo=min(mean,median), hi=max(mean,median) so heavily skewed columns
    # do not produce negative or extrapolated values.
    col_mean = sample.mean(axis=0).astype(np.float32, copy=False)
    col_med = np.median(sample, axis=0).astype(np.float32, copy=False)
    mode = 2.5 * col_med - 1.5 * col_mean
    lo = np.minimum(col_mean, col_med)
    hi = np.maximum(col_mean, col_med)
    col = np.clip(mode, lo, hi).astype(np.float32, copy=False)
    bulk_med = float(np.median(col))

    # Reflect padding so an edge-anchored defect is compared to interior columns
    # rather than to a replication of itself.
    pb = win_baseline // 2
    baseline = np.median(sliding_window_view(np.pad(col, pb, mode="reflect"), win_baseline), axis=-1)
    resid = col - baseline
    mad = float(np.median(np.abs(resid - np.median(resid))))
    # Quantized SExtractor-style profiles can have MAD(resid)==0; median|Δcol|
    # matches the usual ~1 ADU column-to-column jitter on bias-like data.
    col_step = float(np.median(np.abs(np.diff(col))))
    # 1 ADU floor is physically motivated: col is a mean over O(100-400) rows,
    # so real column-to-column scatter is already <=1 ADU even on noisy frames,
    # while genuine shifted-overscan defects are tens of ADU. Without a floor,
    # quantization-dominated short exposures (low sky, near-integer col values)
    # collapse sigma to the old +1e-6 safety term and inflate the score by ~1e6.
    sigma = max(1.4826 * mad, col_step, noise_floor)

    # --- Plateau-dip path: catches edge-flush defects and wide interior dips
    # where the col-mean stays flat to within plateau_range_max ADU. A smooth
    # flat-field rolloff fails the plateau test; short coincidental flat spots
    # in bulk are discarded by the plateau_min_run filter.
    pp = plateau_win // 2
    win = sliding_window_view(np.pad(col, (pp, plateau_win - 1 - pp), mode="edge"), plateau_win)
    local_range = np.quantile(win, plateau_qhi, axis=-1) - np.quantile(win, plateau_qlo, axis=-1)
    is_plateau = local_range < plateau_range_max
    if is_plateau.any():
        p_int = is_plateau.astype(np.int8)
        d_run = np.diff(np.concatenate([[0], p_int, [0]]))
        starts = np.where(d_run == 1)[0]
        ends = np.where(d_run == -1)[0]
        for s, e in zip(starts, ends):
            if (e - s) < plateau_min_run:
                is_plateau[s:e] = False
    if is_plateau.sum() >= plateau_min_run:
        dip = np.where(is_plateau, -resid, 0.0).astype(np.float32, copy=False)
        ps = win_smooth // 2
        smooth = sliding_window_view(np.pad(dip, (ps, win_smooth - 1 - ps), mode="edge"), win_smooth).mean(axis=-1)
        plateau_score = -float(smooth.max() / sigma)
    else:
        plateau_score = 0.0

    # --- Paired-cliff path: catches narrow interior dips (e.g. in flats with
    # strong vignetting) by requiring a sharp drop followed by a sharp rise
    # within pair_dmax cols, both steps above pair_sigma_min * median|diff|,
    # and the plateau between both cliffs sitting at least dip_abs_min ADU
    # below the global col-mean median (rejects hot-column pair artifacts).
    dcol = np.diff(col)
    med_abs = max(float(np.median(np.abs(dcol))), noise_floor)
    cliff_thr = pair_sigma_min * med_abs
    bulk_limit = bulk_med - dip_abs_min
    col_right = col[1:]
    best_pair = 0.0
    for dd in range(pair_dmin, pair_dmax + 1):
        fd = dcol[:-dd]
        tr = dcol[dd:]
        right_after_drop = col_right[:-dd]
        right_before_rise = col[dd:-1]
        msk = (fd < -cliff_thr) & (tr > cliff_thr) & (right_after_drop < bulk_limit) & (right_before_rise < bulk_limit)
        if msk.any():
            mag = (-fd) * tr
            v = float(mag[msk].max())
            if v > best_pair:
                best_pair = v
    cliff_score = -float(np.sqrt(best_pair) / med_abs) if best_pair > 0 else 0.0

    return min(plateau_score, cliff_score)


def check_shifted_overscan(
    image: np.ndarray,
    threshold: float = SHIFTED_SCORE_THRESHOLD,
    **kwargs,
) -> tuple[bool, float]:
    score = shifted_overscan_score(image, **kwargs)
    return (score < threshold), score


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
    dtype: str | None = None,
    **kwargs,
):
    """
    Combine images using a subprocess call to a CUDA-accelerated script.
    If the process times out (10 seconds per image), falls back to CPU processing.

    The subprocess writes both raw QA (SHIFTED/SHFTSCR) and the masterframe pixel
    statistics (CLIPMEAN/CLIPMED/..., UNIFORM for dark, SIGMEAN/EDGEVAR for flat)
    into the sibling ``.header`` text file next to ``output``, so the parent never
    re-reads the master FITS.
    """

    cmd = [
        "python",
        f"{SOURCE_DIR}/cuda/combine_images.py",
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

    if dtype is not None:
        cmd.extend(["-dtype", dtype])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Error combining images: {result.stderr}")

    except:
        combine_images_with_cpu(
            images=images,
            output=output,
            sig_output=sig_output,
            subtract=subtract,
            scale=scale,
            norm=norm,
            make_bpmask=make_bpmask,
            bpmask_sigma=bpmask_sigma,
            dtype=dtype,
            **kwargs,
        )


def combine_images_with_cpu(
    images,
    output,
    sig_output,
    subtract=None,
    scale=None,
    norm=False,
    combine_method: Literal["median", "mean"] = "median",
    make_bpmask: str = None,
    bpmask_sigma=5,
    dtype: str | None = None,
    **kwargs,  # prevent crash if extra args are passed. e.g., device_id
):

    raw_scores = []
    raw_data = []
    for img in images:
        d = read_fits_image(img)
        raw_scores.append(shifted_overscan_score(d))
        raw_data.append(d)
    np_stack = np.stack(raw_data)
    del raw_data
    joint_score = combined_shifted_score(raw_scores)

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

    if combine_method == "mean":
        np_combined = np.mean(np_stack, axis=0).astype(np.float32)
        np_std = np.std(np_stack, axis=0, ddof=1).astype(np.float32)
    else:
        np_combined, np_std = _calc_median_and_std(np_stack)
    fits.writeto(output, data=np_combined, overwrite=True)
    fits.writeto(sig_output, data=np_std, overwrite=True)

    if make_bpmask is not None:
        hot_mask = sigma_clipped_stats_cpu(np_combined, bpmask_sigma, return_mask=True)
        fits.writeto(make_bpmask, data=hot_mask.astype(np.uint8), overwrite=True)

    prepare_raw_qa_header(output, joint_score)
    if dtype is not None:
        prepare_masterframe_header(
            output,
            np_combined,
            dtype=dtype,
            sig_data=np_std,
            bpmask_path=make_bpmask,
        )


def process_image_with_subprocess_gpu(
    image_paths, bias, dark, flat, device_id=0, output_paths=None, n_head_blocks=None, **kwargs
):
    """
    Process images using a subprocess call to a CUDA-accelerated script.
    If the process times out (10 seconds per image), falls back to CPU processing.

    The subprocess stamps SHIFTED/SHFTSCR into each output's sibling ``.header`` file.
    """
    gc.collect()

    # Create SERVICES_TMP_DIR/imlist directory if it doesn't exist
    imlist_dir = os.path.join(SERVICES_TMP_DIR, "imlist")
    os.makedirs(imlist_dir, exist_ok=True)

    # Generate a common identifier to associate input, output, and log files
    common_id = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"

    # Create temporary text files for image_paths and output_paths with common identifier
    input_list_path = os.path.join(imlist_dir, f"input_{common_id}.txt")
    output_list_path = os.path.join(imlist_dir, f"output_{common_id}.txt")
    log_file_path = os.path.join(imlist_dir, f"log_{common_id}.txt")

    with open(input_list_path, "w") as input_file:
        input_file.write("\n".join(image_paths))

    with open(output_list_path, "w") as output_file:
        output_file.write("\n".join(output_paths))

    try:
        cmd = [
            "python",
            f"{SOURCE_DIR}/cuda/process_image.py",
            "-bias",
            bias,
            "-dark",
            dark,
            "-flat",
            flat,
            "-input-list",
            input_list_path,
            "-output-list",
            output_list_path,
            "-device",
            str(device_id),
        ]
        if n_head_blocks is not None:
            cmd.extend(["-n-head-blocks", str(n_head_blocks)])

        try:
            # Run subprocess and save output to log file
            with open(log_file_path, "w") as log_file:
                result = subprocess.run(
                    cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True, timeout=5 * len(image_paths)
                )
                log_file.flush()

            if result.returncode != 0:
                raise RuntimeError(f"Error processing images. Check log file: {log_file_path}")
            success = True
            return None

        except Exception as e:
            print(f"GPU processing failed, falling back to CPU processing. See log file: {log_file_path}")
            print(f"Error: {e}")
            # Fall back to CPU processing
            success = False
            return process_image_with_cpu(
                image_paths=image_paths,
                bias=bias,
                dark=dark,
                flat=flat,
                output_paths=output_paths,
                n_head_blocks=n_head_blocks,
                **kwargs,
            )
    finally:
        # Clean up temporary files
        try:
            os.unlink(input_list_path)
            os.unlink(output_list_path)
            if success:
                os.unlink(log_file_path)
        except OSError:
            pass


def process_image_with_cpu(
    image_paths: list,
    bias: str,
    dark: str,
    flat: str,
    output_paths: list,
    n_head_blocks=None,
    **kwargs,
):
    """
    Process images using CPU with batch processing and parallel loading.

    Args:
        image_paths: List of input image paths
        bias: Bias frame path
        dark: Dark frame path
        flat: Flat frame path
        output_paths: List of output image paths
    """

    # Load calibration frames once
    bias_data = read_fits_image(bias)
    dark_data = read_fits_image(dark)
    flat_data = read_fits_image(flat)

    h, w = bias_data.shape
    multiplicative, subtractive = preparation_kernel(bias_data, dark_data, flat_data, h, w)

    batch_size = 30
    max_workers = 10  # 3
    total_batches = (len(image_paths) + batch_size - 1) // batch_size

    for batch_idx, batch_start in enumerate(range(0, len(image_paths), batch_size), 1):
        batch_end = min(batch_start + batch_size, len(image_paths))
        batch_image_paths = image_paths[batch_start:batch_end]
        batch_output_paths = output_paths[batch_start:batch_end]
        batch_num_images = len(batch_image_paths)

        # Load images in parallel using threading

        batch_data, in_paths, out_paths = read_fits_images(
            batch_image_paths, batch_output_paths, max_workers=max_workers
        )

        # Process images in batch
        processed_batch = []
        for image_data, out_path in zip(batch_data, out_paths):
            prepare_raw_qa_header(out_path, shifted_overscan_score(image_data))
            processed_data = reduction_kernel_cpu(image_data, subtractive, multiplicative, h, w)
            processed_batch.append(processed_data)

        # Write outputs in parallel
        write_fits_images(out_paths, processed_batch, n_head_blocks=n_head_blocks)
        gc.collect()

    # Clean up
    del bias_data, dark_data, flat_data
    gc.collect()
    return None


# def _process_single_image(
#     image_path: str,
#     output_path: str,
#     subtractive: np.ndarray,
#     multiplicative: np.ndarray,
#     h: int,
#     w: int,
# ):
#     """
#     Process a single image with given calibration data.
#     This is the core processing function used by both sequential and parallel modes.
#     """
#     try:
#         # Read image
#         image_data = read_fits_image(image_path)

#         # Apply reduction
#         processed_data = reduction_kernel_cpu(image_data, subtractive, multiplicative, h, w)

#         # Create output directory
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)

#         # Handle header
#         header_file = output_path.replace(".fits", ".header")
#         header = None
#         if os.path.exists(header_file):
#             with open(header_file, "r") as f:
#                 header = fits.Header.fromstring(f.read(), sep="\n")

#         # Write output
#         fits.writeto(
#             output_path,
#             data=processed_data,
#             header=header,
#             overwrite=True,
#         )

#     except Exception as e:
#         print(f"Error processing {image_path}: {e}")
#         raise


@njit(parallel=True)
def reduction_kernel_cpu(image, subtractive, multiplicative, h, w):
    h, w = image.shape
    corrected = np.empty_like(image)
    for i in prange(h):
        for j in range(w):
            corrected[i, j] = (image[i, j] - subtractive[i, j]) * multiplicative[i, j]

    return corrected


@njit(parallel=True)
def preparation_kernel(bias, dark, flat, h, w):
    multiplicative = np.empty_like(flat)
    subtractive = np.empty_like(flat)
    for i in prange(h):
        for j in range(w):
            multiplicative[i, j] = 1.0 / flat[i, j]
            subtractive[i, j] = bias[i, j] + dark[i, j]
    return multiplicative, subtractive


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
def _compute_outlier_mask_2d(data: np.ndarray, median: float, std: float, hot_sigma: float):
    H, W = data.shape
    mask = np.empty((H, W), dtype=np.uint8)
    threshold = hot_sigma * std
    for i in prange(H):
        for j in range(W):
            mask[i, j] = 1 if abs(data[i, j] - median) > threshold else 0
    return mask


def compute_rms(new_data: np.ndarray, ref_data: np.ndarray):
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
        edge_size = 100

        top_left_median = np.median(d[:edge_size, :edge_size])
        top_right_median = np.median(d[:edge_size, -edge_size:])
        bottom_left_median = np.median(d[-edge_size:, :edge_size])
        bottom_right_median = np.median(d[-edge_size:, -edge_size:])

        # Calculate edge values
        edge_values = [top_left_median, top_right_median, bottom_left_median, bottom_right_median]

        # Calculate (max-min)/min ratio
        min_val = min(edge_values)
        max_val = max(edge_values)

        if min_val > 0:  # Avoid division by zero
            edge_var = (max_val - min_val) / min_val
        else:
            edge_var = 0.0

        return edge_var, edge_values

    except Exception as e:
        print(f"Error calculating edge variation: {e}")
        return 0.0, [0, 0, 0, 0]


# def uniformity_statistical(fits_path: str, bpmask_path: str = None, grid_size: int = 32):
#     """
#     Fast uniformity checking using statistical grid analysis.
#     This divides the image into a grid and analyzes statistics of each cell.
#     Fast and gives spatial uniformity information.
#     Returns log10(uniformity_score) for easier interpretation.
#     """

#     # Load data
#     d = fits.getdata(fits_path).astype(np.float32)

#     if bpmask_path is None:
#         bpmask_path = fits_path.replace("dark", "bpmask")
#     bpm = fits.getdata(bpmask_path).astype(np.float32)

#     if d.ndim != 2 or bpm.ndim != 2 or d.shape != bpm.shape:
#         raise ValueError(f"Shape mismatch: data {d.shape}, bpmask {bpm.shape}")

#     # BPM mask
#     hot_mask = bpm != 0

#     # Auto-shift to positive
#     dmin = float(d.min())
#     if dmin <= 0:
#         d = d - dmin + 1e-8

#     # Apply mask
#     d_masked = d.copy()
#     d_masked[hot_mask] = np.nan

#     h, w = d_masked.shape
#     cell_h, cell_w = h // grid_size, w // grid_size

#     # Analyze each grid cell
#     cell_means = []
#     cell_stds = []

#     for i in range(grid_size):
#         for j in range(grid_size):
#             start_h = i * cell_h
#             end_h = (i + 1) * cell_h if i < grid_size - 1 else h
#             start_w = j * cell_w
#             end_w = (j + 1) * cell_w if j < grid_size - 1 else w

#             cell_data = d_masked[start_h:end_h, start_w:end_w]
#             valid_cell = cell_data[~np.isnan(cell_data)]

#             if len(valid_cell) > 0:
#                 cell_means.append(np.mean(valid_cell))
#                 cell_stds.append(np.std(valid_cell))

#     # Calculate uniformity metrics
#     cell_means = np.array(cell_means)
#     cell_stds = np.array(cell_stds)

#     # Spatial uniformity (variation between cells)
#     spatial_cv = variation(cell_means)

#     # Local uniformity (average variation within cells)
#     local_cv = np.mean(cell_stds / (cell_means + 1e-10))

#     # Overall uniformity score
#     uniformity_score = spatial_cv + 0.5 * local_cv

#     # Apply log10 for easier interpretation
#     log_uniformity_score = np.log10(uniformity_score + 1e-10)

#     return float(-1 * log_uniformity_score)


def uniformity_statistical(
    fits_path: str,
    bpmask_path: str = None,
    grid_size: int = 32,
    bin_x: int = 1,
    bin_y: int = 1,
    alpha: float = 0.5,  # 0.5 for uncorrelated noise; lower if correlated structure dominates
):
    """
    Fast uniformity checking using statistical grid analysis.
    This divides the image into a grid and analyzes statistics of each cell.
    Fast and gives spatial uniformity information.
    Returns log10(uniformity_score) for easier interpretation.
    """

    # Load data
    d = fits.getdata(fits_path).astype(np.float32)
    n = bin_x * bin_y
    d = d / n  # convert sum-binned image to mean-equivalent units
    print("sum-bin")

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

    # --- (A) Grid correction for spatial_cv: keep physical scale fixed ---
    # If image is binned by (bin_y, bin_x), its dimensions typically shrink.
    # To keep the same "unbinned" footprint per cell, reduce grid counts accordingly.
    grid_y = max(1, int(round(grid_size / bin_y)))
    grid_x = max(1, int(round(grid_size / bin_x)))

    cell_h = h // grid_y
    cell_w = w // grid_x

    cell_means = []
    cell_stds = []

    for i in range(grid_y):
        for j in range(grid_x):
            sh = i * cell_h
            eh = (i + 1) * cell_h if i < grid_y - 1 else h
            sw = j * cell_w
            ew = (j + 1) * cell_w if j < grid_x - 1 else w

            cell = d_masked[sh:eh, sw:ew]
            valid = cell[~np.isnan(cell)]
            if valid.size:
                m = np.mean(valid)
                s = np.std(valid)
                cell_means.append(m)
                cell_stds.append(s)

    cell_means = np.asarray(cell_means)
    cell_stds = np.asarray(cell_stds)

    spatial_cv = variation(cell_means)

    local_cv = np.mean(cell_stds / (cell_means + 1e-10))

    # --- (B) Local correction: undo binning noise reduction ---
    local_cv *= n**alpha

    uniformity_score = spatial_cv + 0.5 * local_cv
    return float(-np.log10(uniformity_score + 1e-10))


def compute_masterframe_stats(
    filename: str,
    data: np.ndarray,
    *,
    dtype: str,
    device_id: int = 0,
    cropsize: int = 500,
    sig_data: np.ndarray | None = None,
    bpmask_path: str | None = None,
) -> fits.Header:
    """Compute per-masterframe pixel statistics into a fresh Header (no FITS re-read).

    Intended to run where ``data`` / ``sig_data`` are already in memory (CPU combine
    or GPU subprocess). DARK uniformity re-reads the bpmask file (``bpmask_path``)
    that was just written alongside the master.
    """
    header = fits.Header()
    data = np.asarray(data, dtype=np.float32, copy=False)
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
        if sig_data is None:
            datasig = fits.getdata(filename.replace("flat_", "flatsig_")).astype(np.float32)
        else:
            datasig = np.asarray(sig_data, dtype=np.float32, copy=False)
        s_mean, s_median, s_std = sigma_clipped_stats(datasig, device_id=device_id, sigma=3, maxiters=5)
        edge_var, _ = calculate_edge_variation(data)

        header["SIGMEAN"] = (s_mean, "3-sig clipped mean of the errormap")
        header["SIGMED"] = (s_median, "3-sig clipped median of the errormap")
        header["SIGSTD"] = (s_std, "3-sig clipped std of the errormap")
        header["EDGEVAR"] = (edge_var, "Edge variation of the image")

    elif dtype.upper() == "DARK":
        uniformity_score = uniformity_statistical(filename, bpmask_path=bpmask_path)
        header["UNIFORM"] = (uniformity_score, "Uniformity score")
        header["NTOTPIX"] = (data.shape[0] * data.shape[1], "Total number of pixels")

    return header


def prepare_masterframe_header(
    filename: str,
    data: np.ndarray,
    *,
    dtype: str,
    device_id: int = 0,
    sig_data: np.ndarray | None = None,
    bpmask_path: str | None = None,
) -> None:
    """Compute masterframe pixel statistics and merge them into the sibling ``.header`` text file."""
    stats = compute_masterframe_stats(
        filename,
        data,
        dtype=dtype,
        device_id=device_id,
        sig_data=sig_data,
        bpmask_path=bpmask_path,
    )
    update_header_file(filename, stats)


def record_masterframe_statistics(
    filename: str,
    header: fits.Header,
    # *,
    # dtype: str,
) -> fits.Header:
    """Merge raw QA + pixel statistics from the sibling ``.header`` text file into ``header``.

    The combine step (CPU or CUDA subprocess) writes all keys into the ``.header`` text file
    via ``prepare_raw_qa_header`` and ``prepare_masterframe_header``, so the parent just
    merges them without touching the master FITS again.
    """
    side_header = load_header_file(filename)
    if side_header is None:
        raise FileNotFoundError(
            f"Master-frame header text file not found next to {filename!r}; "
            "combine step did not run or failed to write stats."
        )
    for card in side_header.cards:
        header[card.keyword] = (card.value, card.comment)
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


def bin_image(
    img: np.ndarray,
    bin_x: int,
    bin_y: int | None = None,
    method: Literal["mean", "sum", "median"] = "mean",
):
    """Block binning by integer factors in X and Y (cropping remainder)."""
    img = np.asarray(img)
    if bin_y is None:
        bin_y = bin_x

    nx = int(bin_x)
    ny = int(bin_y)
    if img.ndim != 2:
        raise ValueError("img must be 2D")
    if nx <= 0 or ny <= 0:
        raise ValueError("bin_x and bin_y must be >= 1")

    h, w = img.shape

    h2 = (h // ny) * ny
    w2 = (w // nx) * nx
    if h2 != h or w2 != w:
        print("[WARNING] Image dimensions not divisible by the bin factors. Cropping remainder.")
        img = img[:h2, :w2]

    if method == "mean":
        return img.reshape(h2 // ny, ny, w2 // nx, nx).mean(axis=(1, 3))
    elif method == "sum":
        return img.reshape(h2 // ny, ny, w2 // nx, nx).sum(axis=(1, 3))
    elif method == "median":
        return np.median(img.reshape(h2 // ny, ny, w2 // nx, nx), axis=(1, 3))
    else:
        raise ValueError(f"Invalid method: {method}")
