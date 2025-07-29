import os
from astropy.io import fits
from ..utils import swap_ext, add_suffix
from numba import njit, prange
import numpy as np
import subprocess
from ..const import SCRIPT_DIR


def calc_weight_with_subprocess(images, d_m_file, f_m_file, sig_z_file, sig_f_file, device_id=0):
    cmd = [
        "python",
        f"{SCRIPT_DIR}/cuda/weight_map.py",
        "-input",
        *images,
        "-d_m_file",
        d_m_file,
        "-f_m_file",
        f_m_file,
        "-sig_z_file",
        sig_z_file,
        "-sig_f_file",
        sig_f_file,
        "-device",
        f"{device_id}",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Error combining images: {result.stderr}")

    return None


def calc_weight_with_cpu(images, d_m_file, f_m_file, sig_z_file, sig_f_file, weight=True, **kwargs):

    mfg_data = []
    n_images = []
    for file in [sig_z_file, d_m_file, f_m_file]:
        data, header = fits.getdata(file, header=True)
        mfg_data.append(data.astype(np.float32))
        n_images.append(np.float32(header["NFRAMES"]))
        if file == d_m_file:
            egain = header["EGAIN"]  # e-/ADU

    sig_z, d_m, f_m = mfg_data
    p_z, p_d, p_f = n_images
    sig_f = fits.getdata(sig_f_file).astype(np.float32)
    sig_b = np.empty(d_m.shape, dtype=np.float32)
    
    sig_zm = sig_z / np.sqrt(p_z)
    sig_dm_sq = (d_m / egain + (1 + 1 / p_z) * sig_z**2) / p_d
    sig_fm = sig_f / np.sqrt(p_f)

    stacked = []
    for fname in images:
        data = fits.getdata(fname).astype(np.float32)
        stacked.append(data)
    
    H, W = data.shape

    sig_r_sq = np.empty((H, W), dtype=np.float32)
    sig_rp_sq = np.empty((H, W), dtype=np.float32)

    # process each image
    for idx in range(len(images)):
        r_p = stacked[idx]
        # compute sig_r_squared
        compute_sig_r(r_p, f_m, d_m, egain, sig_z, sig_r_sq)
        # compute sig_rp_squared
        compute_sig_rp(sig_r_sq, sig_zm, sig_dm_sq, f_m, r_p, sig_fm, sig_rp_sq)

        if weight:
            stacked[idx] = 1.0 / (sig_rp_sq + sig_b**2)
        else:
            stacked[idx] = np.sqrt(sig_rp_sq + sig_b**2)

    # write out
    out_names = add_suffix(images, suffix="weight")
    for idx, outname in enumerate(out_names):
        fits.writeto(outname, stacked[idx], overwrite=True)


@njit(parallel=True)
def compute_sig_r(sci, flat, dark, gain, sig_z, out):
    """
    out[i,j] = (max(sci[i,j] * flat[i,j] + dark[i,j], 0) / gain) + sig_z[i,j]**2
    """
    H, W = sci.shape
    for i in prange(H):
        for j in range(W):
            poisson_component = sci[i, j] * flat[i, j] + dark[i, j]
            clipped = poisson_component if poisson_component > 0.0 else 0.0
            out[i, j] = clipped / gain + sig_z[i, j] * sig_z[i, j]


@njit(parallel=True)
def compute_sig_rp(sig_r_squared, sig_zm, sig_dm_sq, f_m, r_p, sig_fm, out):
    """
    out[i,j] = (sig_r_squared + sig_zm**2 + sig_dm_sq) / f_m**2
              + (r_p**2)*(sig_fm**2)/f_m**2
    """
    H, W = sig_r_squared.shape
    for i in prange(H):
        for j in range(W):
            fm_sq = f_m[i, j] * f_m[i, j]
            sig_zm_sq = sig_zm[i, j] * sig_zm[i, j]
            sig_fm_sq = sig_fm[i, j] * sig_fm[i, j]
            term1 = (sig_r_squared[i, j] + sig_zm_sq + sig_dm_sq[i, j]) / fm_sq
            term2 = (r_p[i, j] * r_p[i, j]) * (sig_fm_sq) / fm_sq
            out[i, j] = term1 + term2

