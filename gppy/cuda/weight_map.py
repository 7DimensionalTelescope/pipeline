import argparse
from astropy.io import fits
import numpy as np
import cupy as cp
import os

# Reduction kernel
sig_r_kernel = cp.ElementwiseKernel(
    in_params="T sci, T flat, T dark, T gain, T sig_z",
    out_params="T sig_r_squared",
    operation="""
    T corrected_signal = sci * flat + dark;
    T clipped_signal = corrected_signal > 0 ? corrected_signal : 0;
    sig_r_squared = clipped_signal / gain + sig_z * sig_z;
    """,
    name="sig_r_kernel"
)

sig_rp_squared_kernel = cp.ElementwiseKernel(
    in_params="T sig_r_squared, T sig_zm, T sig_dm_squared, T f_m, T r_p, T sig_fm",
    out_params="T sig_rp_squared",
    operation="""
    T f_m2 = f_m * f_m;
    sig_rp_squared = (sig_r_squared + sig_zm * sig_zm + sig_dm_squared) / f_m2
                   + (r_p * r_p) * (sig_fm * sig_fm) / f_m2;
    """,
    name="sig_rp_squared_kernel"
)


def calc_weight(images, d_m, f_m, sig_z, sig_f, p_d, p_z, p_f, egain, weight=True, device=None):

    sig_b = np.empty(d_m.shape, dtype=np.float32)

    with cp.cuda.Device(device):
        c_d_m, c_f_m = cp.asarray(d_m).astype(cp.float32), cp.asarray(f_m).astype(cp.float32)

        c_sig_b, c_sig_z, c_sig_f = cp.asarray(sig_b).astype(cp.float32), cp.asarray(sig_z).astype(cp.float32), cp.asarray(sig_f).astype(cp.float32)
        
        data = [fits.getdata(o) for o in images]

        c_r_p = cp.asarray(data)
        c_r_p = c_r_p.astype(cp.float32)

        c_sig_zm = c_sig_z / cp.sqrt(cp.float32(p_z))
        c_sig_dm_squared = (c_d_m / egain + (1 + 1 / cp.float32(p_z)) * c_sig_z**2) / cp.float32(p_d)
        c_sig_fm = c_sig_f / cp.sqrt(cp.float32(p_f))

        c_sig_r_squared = sig_r_kernel(c_r_p, c_f_m, c_d_m, egain, c_sig_z)
        c_sig_rp_squared = sig_rp_squared_kernel(c_sig_r_squared, c_sig_zm, c_sig_dm_squared, c_f_m, c_r_p, c_sig_fm)

        if weight:
            c_r_p = 1 / (c_sig_rp_squared + c_sig_b**2)
        else:
            c_r_p = cp.sqrt(c_sig_rp_squared + c_sig_b**2)

        for i in range(len(data)):
            data[i][:] = cp.asnumpy(c_r_p[i])
        
        del c_d_m, c_f_m, c_sig_b, c_sig_z, c_sig_f
        del c_sig_zm, c_sig_dm_squared, c_sig_fm
        del c_sig_r_squared, c_sig_rp_squared

        cp.get_default_memory_pool().free_all_blocks()

    output_names = add_suffix(images, "weight")
    for i, o in enumerate(output_names):
        fits.writeto(o, data[i], overwrite=True)

    del data

def add_suffix(filename: str | list[str], suffix):
    if isinstance(filename, list):
        return [add_suffix(f, suffix) for f in filename]
    base, ext = os.path.splitext(filename)
    suffix = suffix if suffix.startswith("_") else f"_{suffix}"
    return f"{base}{suffix}{ext}"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compute weight maps for a batch of images")
    p.add_argument("-input", nargs="+", help="List of science images to process")
    p.add_argument("-d_m_file", required=True, help="Dark master FITS file")
    p.add_argument("-f_m_file", required=True, help="Flat master FITS file")
    p.add_argument("-sig_z_file", required=True, help="Sigma-Z FITS file")
    p.add_argument("-sig_f_file", required=True, help="Sigma-F FITS file")
    p.add_argument("-device", default="CPU", help="Device identifier for calc_weight")
    
    args = p.parse_args()
    uncalculated_images = args.input
    sig_z_file, d_m_file, sig_f_file, f_m_file = args.sig_z_file, args.d_m_file, args.sig_f_file, args.f_m_file

    mfg_data = []
    n_images = []
    for file in [sig_z_file, d_m_file, f_m_file]:
        data, header = fits.getdata(file, header=True)
        mfg_data.append(data.astype(np.float32))
        n_images.append(header["NFRAMES"])
        if file == d_m_file:
            egain = header["EGAIN"]  # e-/ADU

    sig_z, d_m, f_m = mfg_data
    p_z, p_d, p_f = n_images
    sig_f = fits.getdata(sig_f_file).astype(np.float32)

    # Run weight‐map calculation
    calc_weight(
        uncalculated_images, d_m, f_m, sig_z, sig_f, p_d, p_z, p_f, egain, weight=True, device=args.device
    )