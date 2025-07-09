import os
from astropy.io import fits
from ..utils import swap_ext
from numba import njit
import numpy as np
import cupy as cp


def calc_weight(images, d_m, f_m, sig_z, sig_f, p_d, p_z, p_f, egain, weight=True, device=None):
    output = []
    cpu_buffer = np.empty(d_m.shape)
    sig_b = np.empty(d_m.shape, dtype=np.float32)

    if device and device != "CPU":
        with cp.cuda.Device(device):
            d_m, f_m = (
                cp.asarray(d_m),
                cp.asarray(f_m),
            )
            sig_b, sig_z, sig_f = (
                cp.asarray(sig_b),
                cp.asarray(sig_z),
                cp.asarray(sig_f),
            )
            p_d, p_z, p_f = cp.asarray(p_d), cp.asarray(p_z), cp.asarray(p_f)

    for o in images:
        cpu_buffer[:] = fits.getdata(o)

        if device and device != "CPU":
            with cp.cuda.Device(device):
                gpu_buffer = cp.asarray(cpu_buffer)
                gpu_buffer[:] = gpu_buffer.astype(cp.float32)
                gpu_buffer[:] = pix_err_cupy(
                    gpu_buffer, d_m, f_m, sig_b, sig_z, sig_f, p_d, p_z, p_f, egain, weight=weight
                )
                cpu_buffer[:] = cp.asnumpy(gpu_buffer)
                output.append(cpu_buffer.copy())
        else:
            sig_z = np.array(sig_z, dtype=np.float32)
            d_m = np.array(d_m, dtype=np.float32)
            f_m = np.array(f_m, dtype=np.float32)
            sig_f = np.array(sig_f, dtype=np.float32)
            cpu_buffer[:] = pix_err_numba(
                cpu_buffer, d_m, f_m, sig_b, sig_z, sig_f, p_d, p_z, p_f, egain, weight=weight
            )
            output.append(cpu_buffer.copy())

    if device and device != "CPU":
        with cp.cuda.Device(device):
            del gpu_buffer, d_m, f_m, sig_b, sig_z, sig_f, p_d, p_z, p_f
            cp.get_default_memory_pool().free_all_blocks()

    del cpu_buffer

    return output


def pix_err_cupy(r_p, d_m, f_m, sig_b, sig_z, sig_f, p_d, p_z, p_f, G, weight=True):
    nonneg = lambda x: cp.maximum(x, 0)
    """
    r: raw pixel value. r = f * r_p + d + z
    r_p: preprocessed pixel value
    z: bias
    d: dark
    f: flat
    """
    sig_r_squared = nonneg(f_m * r_p + d_m) / G + sig_z**2
    sig_zm = sig_z / cp.sqrt(p_z)
    sig_dm_squared = (d_m / G + (1 + 1 / p_z) * sig_z**2) / p_d
    sig_fm = sig_f / cp.sqrt(p_f)

    sig_rp_squared = (sig_r_squared + sig_zm**2 + sig_dm_squared) / f_m**2 + (r_p / f_m) ** 2 * sig_fm**2  # fmt: skip
    if weight:
        output = 1 / (sig_rp_squared + sig_b**2)
    else:
        output = cp.sqrt(sig_rp_squared + sig_b**2)
    del sig_r_squared, sig_zm, sig_dm_squared, sig_fm, sig_rp_squared, sig_b
    cp.get_default_memory_pool().free_all_blocks()
    return output


def unique(base_filename):
    """
    Modify base_filename by appending a number to make it unique.
    If base_filename is 'image.png' and it exists, this function will return
    'image_1.png', 'image_2.png', etc.
    """
    counter = 1
    filename, file_extension = os.path.splitext(base_filename)
    new_filename = base_filename

    while os.path.exists(new_filename):
        new_filename = f"{filename}_{counter}{file_extension}"
        print(new_filename, "already exists")
        counter += 1

    # print('saving as', new_filename)
    return new_filename


@njit
def pix_err_numba(r_p, d_m, f_m, sig_b, sig_z, sig_f, p_d, p_z, p_f, G, weight=True):
    sig_r_squared = np.maximum(f_m * r_p + d_m, 0) / G + sig_z**2
    sig_zm = sig_z / np.sqrt(p_z)
    sig_dm_squared = (d_m / G + (1 + 1 / p_z) * sig_z**2) / p_d
    sig_fm = sig_f / np.sqrt(p_f)

    sig_rp_squared = (sig_r_squared + sig_zm**2 + sig_dm_squared) / f_m**2 + (r_p / f_m) ** 2 * sig_fm**2

    if weight:
        return 1 / (sig_rp_squared + sig_b**2)
    else:
        return np.sqrt(sig_rp_squared + sig_b**2)


def pix_err_cupy(r_p, d_m, f_m, sig_b, sig_z, sig_f, p_d, p_z, p_f, G, weight=True):
    """
    r: raw pixel value. r = f * r_p + d + z
    r_p: preprocessed pixel value
    z: bias
    d: dark
    f: flat
    """
    nonneg = lambda x: cp.maximum(x, 0)
    sig_r_squared = nonneg(f_m * r_p + d_m) / G + sig_z**2
    sig_zm = sig_z / np.sqrt(p_z)
    sig_dm_squared = (d_m / G + (1 + 1 / p_z) * sig_z**2) / p_d
    sig_fm = sig_f / np.sqrt(p_f)

    sig_rp_squared = (sig_r_squared + sig_zm**2 + sig_dm_squared) / f_m**2 + (r_p / f_m) ** 2 * sig_fm**2  # fmt: skip
    if weight:
        output = 1 / (sig_rp_squared + sig_b**2)
    else:
        output = cp.sqrt(sig_rp_squared + sig_b**2)
    del sig_r_squared, sig_zm, sig_dm_squared, sig_fm, sig_rp_squared, sig_b
    cp.get_default_memory_pool().free_all_blocks()
    return output


def pix_err_approx(xp, r_p, d_m, f_m, sig_b, sig_z, sig_f, p_d, p_z, p_f, G, weight=True):

    sig_r_p_squared = r_p / G / f_m + sig_z**2 / f_m**2 + d_m / G / f_m**2 + r_p**2 * sig_f**2 / f_m**2
    if weight:
        return 1 / sig_r_p_squared
    else:
        return xp.sqrt(sig_r_p_squared)


def calculate_background_sigma(bkg_file, G):
    bkg = fits.getdata(bkg_file)
    p_b = 128**2  # number of pixels used to estimate the background
    return np.sqrt(bkg / G) / p_b


def calculate_weight(config):
    """deprecated"""
    if config.settings.gpu_enabled:
        import cupy as xp
    else:
        import numpy as xp

    d_m = xp.asarray(fits.getdata(config.preprocess.mdark_file))
    f_m = xp.asarray(fits.getdata(config.preprocess.mflat_file))
    sig_z = xp.asarray(fits.getdata(config.preprocess.biassig_file))
    sig_f = xp.asarray(fits.getdata(config.preprocess.flatsig_file))

    p_z = fits.getheader(config.preprocess.biassig_file)["NFRAMES"]
    p_d = fits.getheader(config.preprocess.mdark_file)["NFRAMES"]
    p_f = fits.getheader(config.preprocess.flatsig_file)["NFRAMES"]

    egain = fits.getheader(config.preprocess.mdark_file)["EGAIN"]  # e-/ADU

    for i in range(len(config.file.processed_files)):
        r_p_file = os.path.join(config.path.path_processed, config.file.processed_files[i])
        r_p = xp.asarray(fits.getdata(r_p_file))

        bkg_file = config.imstack.bkg_files[i]
        bkgsub_file = config.imstack.bkgsub_files[i]
        sig_b = xp.zeros_like(r_p)
        # sig_b = calculate_background_sigma(bkg_file, egain)

        # sig_b = xp.asarray(fits.getdata(sig_b_file))

        weight_image = pix_err(xp, r_p, d_m, f_m, sig_b, sig_z, sig_f, p_d, p_z, p_f, egain, weight=True)

        if hasattr(weight_image, "get"):  # if CuPy array
            weight_image = weight_image.get()  # Convert to NumPy array

        fits.writeto(
            # os.path.join(config.path.path_processed, weight_file),
            swap_ext(bkgsub_file, "weight.fits"),
            data=weight_image,
            overwrite=True,
        )
