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
    for idx in range(N):
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
            term2 = (r_p[i, j] * r_p[i, j]) * (sig_fm_sq[i, j]) / fm_sq
            out[i, j] = term1 + term2


# def pix_err_cupy(r_p, d_m, f_m, sig_b, sig_z, sig_f, p_d, p_z, p_f, G, weight=True):
#     import cupy as cp

#     nonneg = lambda x: cp.maximum(x, 0)
#     """
#     r: raw pixel value. r = f * r_p + d + z
#     r_p: preprocessed pixel value
#     z: bias
#     d: dark
#     f: flat
#     """
#     sig_r_squared = nonneg(f_m * r_p + d_m) / G + sig_z**2
#     sig_zm = sig_z / cp.sqrt(p_z)
#     sig_dm_squared = (d_m / G + (1 + 1 / p_z) * sig_z**2) / p_d
#     sig_fm = sig_f / cp.sqrt(p_f)

#     sig_rp_squared = (sig_r_squared + sig_zm**2 + sig_dm_squared) / f_m**2 + (r_p / f_m) ** 2 * sig_fm**2  # fmt: skip
#     if weight:
#         output = 1 / (sig_rp_squared + sig_b**2)
#     else:
#         output = cp.sqrt(sig_rp_squared + sig_b**2)
#     del sig_r_squared, sig_zm, sig_dm_squared, sig_fm, sig_rp_squared, sig_b
#     cp.get_default_memory_pool().free_all_blocks()
#     return output


# def unique(base_filename):
#     """
#     Modify base_filename by appending a number to make it unique.
#     If base_filename is 'image.png' and it exists, this function will return
#     'image_1.png', 'image_2.png', etc.
#     """
#     counter = 1
#     filename, file_extension = os.path.splitext(base_filename)
#     new_filename = base_filename

#     while os.path.exists(new_filename):
#         new_filename = f"{filename}_{counter}{file_extension}"
#         print(new_filename, "already exists")
#         counter += 1

#     # print('saving as', new_filename)
#     return new_filename


# @njit
# def pix_err_numba(r_p, d_m, f_m, sig_b, sig_z, sig_f, p_d, p_z, p_f, G, weight=True):
#     sig_r_squared = np.maximum(f_m * r_p + d_m, 0) / G + sig_z**2
#     sig_zm = sig_z / np.sqrt(p_z)
#     sig_dm_squared = (d_m / G + (1 + 1 / p_z) * sig_z**2) / p_d
#     sig_fm = sig_f / np.sqrt(p_f)

#     sig_rp_squared = (sig_r_squared + sig_zm**2 + sig_dm_squared) / f_m**2 + (r_p / f_m) ** 2 * sig_fm**2

#     if weight:
#         return 1 / (sig_rp_squared + sig_b**2)
#     else:
#         return np.sqrt(sig_rp_squared + sig_b**2)


# def pix_err_cupy(r_p, d_m, f_m, sig_b, sig_z, sig_f, p_d, p_z, p_f, G, weight=True):
#     """
#     r: raw pixel value. r = f * r_p + d + z
#     r_p: preprocessed pixel value
#     z: bias
#     d: dark
#     f: flat
#     """
#     nonneg = lambda x: cp.maximum(x, 0)
#     sig_r_squared = nonneg(f_m * r_p + d_m) / G + sig_z**2
#     sig_zm = sig_z / np.sqrt(p_z)
#     sig_dm_squared = (d_m / G + (1 + 1 / p_z) * sig_z**2) / p_d
#     sig_fm = sig_f / np.sqrt(p_f)

#     sig_rp_squared = (sig_r_squared + sig_zm**2 + sig_dm_squared) / f_m**2 + (r_p / f_m) ** 2 * sig_fm**2  # fmt: skip
#     if weight:
#         output = 1 / (sig_rp_squared + sig_b**2)
#     else:
#         output = cp.sqrt(sig_rp_squared + sig_b**2)
#     del sig_r_squared, sig_zm, sig_dm_squared, sig_fm, sig_rp_squared, sig_b
#     cp.get_default_memory_pool().free_all_blocks()
#     return output


# def pix_err_approx(xp, r_p, d_m, f_m, sig_b, sig_z, sig_f, p_d, p_z, p_f, G, weight=True):
#     sig_r_p_squared = r_p / G / f_m + sig_z**2 / f_m**2 + d_m / G / f_m**2 + r_p**2 * sig_f**2 / f_m**2
#     if weight:
#         return 1 / sig_r_p_squared
#     else:
#         return xp.sqrt(sig_r_p_squared)


# def calculate_background_sigma(bkg_file, G):
#     bkg = fits.getdata(bkg_file)
#     p_b = 128**2  # number of pixels used to estimate the background
#     return np.sqrt(bkg / G) / p_b


# def calculate_weight(config):
#     """deprecated"""
#     if config.settings.gpu_enabled:
#         import cupy as xp
#     else:
#         import numpy as xp

#     d_m = xp.asarray(fits.getdata(config.preprocess.mdark_file))
#     f_m = xp.asarray(fits.getdata(config.preprocess.mflat_file))
#     sig_z = xp.asarray(fits.getdata(config.preprocess.biassig_file))
#     sig_f = xp.asarray(fits.getdata(config.preprocess.flatsig_file))

#     p_z = fits.getheader(config.preprocess.biassig_file)["NFRAMES"]
#     p_d = fits.getheader(config.preprocess.mdark_file)["NFRAMES"]
#     p_f = fits.getheader(config.preprocess.flatsig_file)["NFRAMES"]

#     egain = fits.getheader(config.preprocess.mdark_file)["EGAIN"]  # e-/ADU

#     for i in range(len(config.file.processed_files)):
#         r_p_file = os.path.join(config.path.path_processed, config.file.processed_files[i])
#         r_p = xp.asarray(fits.getdata(r_p_file))

#         bkg_file = config.imstack.bkg_files[i]
#         bkgsub_file = config.imstack.bkgsub_files[i]
#         sig_b = xp.zeros_like(r_p)
#         # sig_b = calculate_background_sigma(bkg_file, egain)

#         # sig_b = xp.asarray(fits.getdata(sig_b_file))

#         weight_image = pix_err(xp, r_p, d_m, f_m, sig_b, sig_z, sig_f, p_d, p_z, p_f, egain, weight=True)

#         if hasattr(weight_image, "get"):  # if CuPy array
#             weight_image = weight_image.get()  # Convert to NumPy array

#         fits.writeto(
#             # os.path.join(config.path.path_processed, weight_file),
#             swap_ext(bkgsub_file, "weight.fits"),
#             data=weight_image,
#             overwrite=True,
#         )
