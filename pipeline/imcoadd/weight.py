from astropy.io import fits
from numba import njit, prange
from ..utils import add_suffix
import numpy as np
import fitsio
from ..cuda.weight_map import calc_weight as gpu_calc_weight

def _load_calibration_data(d_m_file, f_m_file, sig_z_file, sig_f_file):
    """
    Load calibration arrays and metadata shared by CPU and GPU paths.
    """

    sig_z = fitsio.read(sig_z_file).astype(np.float32)
    d_m = fitsio.read(d_m_file).astype(np.float32)
    f_m = fitsio.read(f_m_file).astype(np.float32)
    sig_f = fitsio.read(sig_f_file).astype(np.float32)
    p_z = np.float32(fits.getval(sig_z_file, "NFRAMES"))
    p_d = np.float32(fits.getval(d_m_file, "NFRAMES"))
    p_f = np.float32(fits.getval(f_m_file, "NFRAMES"))
    egain = np.float32(fits.getval(d_m_file, "EGAIN"))

    return sig_z, d_m, f_m, sig_f, p_z, p_d, p_f, egain


def calc_weight_with_gpu(images, d_m_file, f_m_file, sig_z_file, sig_f_file, device_id=0, weight=True):
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
    )


def calc_weight_with_cpu(images, d_m_file, f_m_file, sig_z_file, sig_f_file, weight=True, **kwargs):
    output = _load_calibration_data(d_m_file, f_m_file, sig_z_file, sig_f_file)

    out_names = add_suffix(images, suffix="weight")

    for fname, outname in zip(images, out_names):
        image = fitsio.read(fname).astype(np.float32)
        out = optimized_parallel(image, *output)
        fitsio.write(outname, out.astype(np.float32), clobber=True)


@njit(parallel=True)
def optimized_parallel(image, sig_z, dark, flat, sig_f, num_z, num_d, num_f, egain):
    constants = [(1 + 1 / num_z) * (1 + 1 / num_d), (1 + 1 / num_d) / egain, 1 / egain, 1 / num_f]
    out = np.empty_like(flat)
    h, w = flat.shape
    for i in prange(h):
        denom = (
            constants[0] * sig_z[i] * sig_z[i]
            + constants[1] * dark[i]
            + constants[2] * flat[i] * image[i]
            + constants[3] * (image[i] * sig_f[i]) ** 2
        )
        out[i] = flat[i] * flat[i] * (1 / denom)
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
