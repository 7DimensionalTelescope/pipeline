from astropy.io import fits
from numba import njit, prange
from ..utils import add_suffix
from ..path.path import PathHandler
import numpy as np
import fitsio
from ..cuda.weight_map import calc_weight as gpu_calc_weight


def _load_calibration_data(d_m_file, f_m_file, sig_z_file, sig_f_file):
    """
    Load calibration arrays and metadata shared by CPU and GPU paths.
    """

    sig_z = fitsio.read(sig_z_file).astype(np.float32)
    # sig_z can seldom be exactly zero: floor at LSB/sqrt(12)
    np.maximum(sig_z, np.float32(1.0 / np.sqrt(12.0)), out=sig_z)
    d_m = fitsio.read(d_m_file).astype(np.float32)
    f_m = fitsio.read(f_m_file).astype(np.float32)
    sig_f = fitsio.read(sig_f_file).astype(np.float32)
    p_z = np.float32(fits.getval(sig_z_file, "NFRAMES"))
    p_d = np.float32(fits.getval(d_m_file, "NFRAMES"))
    p_f = np.float32(fits.getval(f_m_file, "NFRAMES"))
    egain = np.float32(fits.getval(d_m_file, "EGAIN"))

    return sig_z, d_m, f_m, sig_f, p_z, p_d, p_f, egain


def calc_weight_with_gpu(images, d_m_file, f_m_file, sig_z_file, sig_f_file, device_id=0, weight=True, out_names=None):
    """
    Execute the CuPy/CUDA-based weight-map calculation in-process.
    """
    from ..cuda.weight_map import calc_weight as gpu_calc_weight

    sig_z, d_m, f_m, sig_f, p_z, p_d, p_f, egain = _load_calibration_data(d_m_file, f_m_file, sig_z_file, sig_f_file)

    gpu_calc_weight(
        images,
        d_m,
        f_m,
        sig_z,
        sig_f,
        p_d,
        p_z,
        p_f,
        egain,
        weight=weight,
        device=device_id,
        out_names=out_names,
    )


def calc_weight_with_cpu(images, d_m_file, f_m_file, sig_z_file, sig_f_file, weight=True, out_names=None,
                         weight_store=None, zero_mask=None, **kwargs):
    from .weight_store import load_single_weight, persist_single_weight

    # calibration masters load lazily: an all-reusable group never touches them
    output = None
    masters = {"d": d_m_file, "f": f_m_file, "sz": sig_z_file, "sf": sig_f_file}

    out_names = out_names if out_names is not None else add_suffix(images, suffix="weight")

    # Two threads overlap NFS I/O with the kernel (read-ahead + write-behind); bounded so a
    # busy system queue is not oversubscribed.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = None
        nxt = pool.submit(fitsio.read, images[0])
        for i, outname in enumerate(out_names):
            image = nxt.result().astype(np.float32)
            if i + 1 < len(images):
                nxt = pool.submit(fitsio.read, images[i + 1])
            out = load_single_weight(PathHandler.single_weight_map(images[i]), masters) if weight_store else None
            if out is None:
                if output is None:
                    output = _load_calibration_data(d_m_file, f_m_file, sig_z_file, sig_f_file)
                out = optimized_parallel(image, *output)
                out[~np.isfinite(out)] = 0.0  # degenerate noise model -> weight 0, not inf
                if weight_store:
                    pool.submit(persist_single_weight, PathHandler.single_weight_map(images[i]), out.copy(), masters)
            if zero_mask is not None:
                # zero_badpix_weight without interpolation: the factory copy carries the
                # zeros; the persisted store copy above stays pristine by contract
                out = out.copy()
                out[zero_mask] = 0.0
            if pending is not None:
                pending.result()
            pending = pool.submit(fitsio.write, outname, out.astype(np.float32), clobber=True)
        if pending is not None:
            pending.result()


@njit(parallel=True)
def optimized_parallel(image, sig_z, dark, flat, sig_f, num_z, num_d, num_f, egain):
    """Pixel weight per paper appendix D, both Poisson terms clipped; r_p stays signed."""
    out = np.empty_like(flat)
    h, w = flat.shape
    for i in prange(h):
        for j in range(w):
            r_p = image[i, j]
            f_m = flat[i, j]
            sz2 = sig_z[i, j] * sig_z[i, j]
            sig_zm2 = sz2 / num_z  # D1
            sig_dm2 = (max(dark[i, j], 0.0) / egain + (1 + 1 / num_z) * sz2) / num_d  # D2
            sig_fm2 = sig_f[i, j] * sig_f[i, j] / num_f  # D3
            sig_r2 = max(r_p * f_m + dark[i, j], 0.0) / egain + sz2  # D4
            sig_rp2 = (sig_r2 + sig_zm2 + sig_dm2) / (f_m * f_m) + r_p * r_p * sig_fm2 / (f_m * f_m)  # D5
            out[i, j] = 1.0 / sig_rp2  # D6, sig_b ~ 0
    return out


# Unused alternatives, same clipping semantics, kept for reference
@njit(parallel=True)
def optimized_parallel_twoclip(image, sig_z, dark, flat, sig_f, num_z, num_d, num_f, egain):
    c_z = (1.0 + 1.0 / num_z) * (1.0 + 1.0 / num_d)
    c_science = 1.0 / egain
    c_dark = 1.0 / (egain * num_d)
    c_flat = 1.0 / num_f
    out = np.empty_like(flat)
    h, w = flat.shape
    for i in prange(h):
        r_p = image[i]
        f_m = flat[i]
        d_m = dark[i]
        science_poisson = np.maximum(r_p * f_m + d_m, 0.0)
        dark_poisson = np.maximum(d_m, 0.0)
        flat_error = r_p * sig_f[i]
        denom = (c_z * sig_z[i] * sig_z[i] + c_science * science_poisson
                 + c_dark * dark_poisson + c_flat * flat_error * flat_error)  # fmt: skip
        out[i] = f_m * f_m / denom
    return out


@njit(parallel=True)
def prepare_weight_maps(sig_z, dark, flat, sig_f, num_z, num_d, num_f, egain):
    """use it with optimized_parallel_precomputed"""
    inv_num_d = 1.0 / num_d
    z_coeff = egain * (1.0 + 1.0 / num_z) * (1.0 + inv_num_d)
    flat_error_coeff = egain / num_f
    A = np.empty_like(flat)
    B = np.empty_like(flat)
    C = np.empty_like(flat)
    N = np.empty_like(flat)
    h, w = flat.shape
    for i in prange(h):
        for j in range(w):
            d_m = dark[i, j]
            dark_poisson = d_m if d_m > 0.0 else 0.0
            a = z_coeff * sig_z[i, j] * sig_z[i, j] + inv_num_d * dark_poisson
            A[i, j] = a
            B[i, j] = a + d_m
            C[i, j] = flat_error_coeff * sig_f[i, j] * sig_f[i, j]
            N[i, j] = egain * flat[i, j] * flat[i, j]
    return A, B, C, N


@njit(parallel=True)
def optimized_parallel_precomputed(image, flat, A, B, C, N):
    """use it with prepare_weight_maps"""
    out = np.empty_like(flat)
    h, w = flat.shape
    for i in prange(h):
        for j in range(w):
            r_p = image[i, j]
            pb = r_p * flat[i, j] + B[i, j]
            if pb < A[i, j]:
                pb = A[i, j]
            out[i, j] = N[i, j] / (pb + C[i, j] * r_p * r_p)
    return out


# def calc_weight_with_cpu(images, d_m_file, f_m_file, sig_z_file, sig_f_file, weight=True, **kwargs):
#     sig_z, d_m, f_m, sig_f, p_z, p_d, p_f, egain = _load_calibration_data(d_m_file, f_m_file, sig_z_file, sig_f_file)
#     sig_b_squared = np.zeros_like(d_m, dtype=np.float32)

#     sig_zm = sig_z / np.sqrt(p_z)
#     sig_dm_sq = (d_m / egain + (1 + 1 / p_z) * sig_z**2) / p_d
#     sig_fm = sig_f / np.sqrt(p_f)

#     sig_r_sq = np.empty_like(d_m, dtype=np.float32)
#     sig_rp_sq = np.empty_like(d_m, dtype=np.float32)

#     out_names = add_suffix(images, suffix="weight")

#     for fname, outname in zip(images, out_names):
#         r_p = fits.getdata(fname).astype(np.float32)

#         # Use optimized C++ combined function if available (single pass, most efficient)
#         if _HAS_CPP:
#             result = np.empty_like(d_m, dtype=np.float32)
#             _compute_weight_combined_cpp(
#                 r_p, f_m, d_m, egain, sig_z, sig_zm, sig_dm_sq, f_m, sig_fm, sig_b_squared, weight, result
#             )
#         else:
#             # Fall back to step-by-step computation with Numba
#             _compute_sig_r(r_p, f_m, d_m, egain, sig_z, sig_r_sq)
#             _compute_sig_rp(sig_r_sq, sig_zm, sig_dm_sq, f_m, r_p, sig_fm, sig_rp_sq)
#             if weight:
#                 result = 1.0 / (sig_rp_sq + sig_b_squared)
#             else:
#                 result = np.sqrt(sig_rp_sq + sig_b_squared)

#         fits.writeto(outname, result.astype(np.float32), overwrite=True)


# @njit(parallel=True)
# def _compute_sig_r(sci, flat, dark, gain, sig_z, out):
#     """
#     out[i,j] = (max(sci[i,j] * flat[i,j] + dark[i,j], 0) / gain) + sig_z[i,j]**2
#     """
#     h, w = sci.shape
#     for i in prange(h):
#         for j in range(w):
#             poisson_component = sci[i, j] * flat[i, j] + dark[i, j]
#             clipped = poisson_component if poisson_component > 0.0 else 0.0
#             out[i, j] = clipped / gain + sig_z[i, j] * sig_z[i, j]


# @njit(parallel=True)
# def _compute_sig_rp(sig_r_squared, sig_zm, sig_dm_sq, f_m, r_p, sig_fm, out):
#     """
#     out[i,j] = (sig_r_squared + sig_zm**2 + sig_dm_sq) / f_m**2
#               + (r_p**2)*(sig_fm**2)/f_m**2
#     """
#     h, w = sig_r_squared.shape
#     for i in prange(h):
#         for j in range(w):
#             fm_sq = f_m[i, j] * f_m[i, j]
#             sig_zm_sq = sig_zm[i, j] * sig_zm[i, j]
#             sig_fm_sq = sig_fm[i, j] * sig_fm[i, j]
#             term1 = (sig_r_squared[i, j] + sig_zm_sq + sig_dm_sq[i, j]) / fm_sq
#             term2 = (r_p[i, j] * r_p[i, j]) * sig_fm_sq / fm_sq
#             out[i, j] = term1 + term2
