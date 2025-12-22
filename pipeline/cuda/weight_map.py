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
    name="sig_r_kernel",
)

sig_rp_squared_kernel = cp.ElementwiseKernel(
    in_params="T sig_r_squared, T sig_zm, T sig_dm_squared, T f_m, T r_p, T sig_fm",
    out_params="T sig_rp_squared",
    operation="""
    T f_m_sq = f_m * f_m;
    sig_rp_squared = (sig_r_squared + sig_zm * sig_zm + sig_dm_squared) / f_m_sq
                   + (r_p * r_p) * (sig_fm * sig_fm) / f_m_sq;
    """,
    name="sig_rp_squared_kernel",
)

# Combined kernel that does everything in one pass (most efficient)
weight_combined_kernel = cp.ElementwiseKernel(
    in_params="T sci, T flat, T dark, T gain, T sig_z, T sig_zm, T sig_dm_squared, T f_m, T sig_fm, T sig_b_squared, bool weight",
    out_params="T out",
    operation="""
    // Step 1: compute sig_r_squared
    T corrected_signal = sci * flat + dark;
    T clipped_signal = corrected_signal > 0 ? corrected_signal : 0;
    T sig_r_squared = clipped_signal / gain + sig_z * sig_z;
    
    // Step 2: compute sig_rp_squared
    T f_m_sq = f_m * f_m;
    T sig_zm_sq = sig_zm * sig_zm;
    T sig_fm_sq = sig_fm * sig_fm;
    T r_p_sq = sci * sci;
    T sig_rp_squared = (sig_r_squared + sig_zm_sq + sig_dm_squared) / f_m_sq
                      + r_p_sq * sig_fm_sq / f_m_sq;
    
    // Step 3: compute final weight
    T sum = sig_rp_squared + sig_b_squared;
    if (weight) {
        out = 1.0 / sum;
    } else {
        out = sqrt(sum);
    }
    """,
    name="weight_combined_kernel",
)


def calc_weight(images, d_m, f_m, sig_z, sig_f, p_d, p_z, p_f, egain, weight=True, device=None):

    with cp.cuda.Device(device):
        sig_b = np.zeros(d_m.shape, dtype=np.float32)

        c_d_m, c_f_m = cp.asarray(d_m).astype(cp.float32), cp.asarray(f_m).astype(cp.float32)

        c_sig_b, c_sig_z, c_sig_f = (
            cp.asarray(sig_b).astype(cp.float32),
            cp.asarray(sig_z).astype(cp.float32),
            cp.asarray(sig_f).astype(cp.float32),
        )

        host_images = [fits.getdata(o).astype(np.float32) for o in images]
        c_r_p = cp.asarray(host_images, dtype=cp.float32)
        del host_images

        c_sig_zm = c_sig_z / cp.sqrt(cp.float32(p_z))
        c_sig_dm_squared = (c_d_m / egain + (1 + 1 / cp.float32(p_z)) * c_sig_z**2) / cp.float32(p_d)
        c_sig_fm = c_sig_f / cp.sqrt(cp.float32(p_f))

        # Use optimized combined kernel (single pass, most efficient)
        c_weight = weight_combined_kernel(
            c_r_p, c_f_m, c_d_m, egain, c_sig_z, c_sig_zm, c_sig_dm_squared, c_f_m, c_sig_fm, c_sig_b**2, weight
        )

        weight_results = cp.asnumpy(c_weight)

        del c_d_m, c_f_m, c_sig_b, c_sig_z, c_sig_f, c_r_p, c_weight
        del c_sig_zm, c_sig_dm_squared, c_sig_fm

        cp.get_default_memory_pool().free_all_blocks()

    output_names = add_suffix(images, "weight")

    for i, o in enumerate(output_names):
        fits.writeto(o, weight_results[i], overwrite=True)

    del weight_results


def _load_calibration_data(d_m_file, f_m_file, sig_z_file, sig_f_file):
    sig_z = fitsio.read(sig_z_file).astype(np.float32)
    d_m = fitsio.read(d_m_file).astype(np.float32)
    f_m = fitsio.read(f_m_file).astype(np.float32)
    sig_f = fitsio.read(sig_f_file).astype(np.float32)
    p_z = np.float32(fits.getval(sig_z_file, "NFRAMES"))
    p_d = np.float32(fits.getval(d_m_file, "NFRAMES"))
    p_f = np.float32(fits.getval(f_m_file, "NFRAMES"))
    egain = np.float32(fits.getval(d_m_file, "EGAIN"))
    return sig_z, d_m, f_m, sig_f, p_z, p_d, p_f, egain


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
    sig_z, d_m, f_m, sig_f, p_z, p_d, p_f, egain = _load_calibration_data(
        args.sig_z_file, args.d_m_file, args.sig_f_file, args.f_m_file
    )

    # Run weight‐map calculation
    calc_weight(uncalculated_images, d_m, f_m, sig_z, sig_f, p_d, p_z, p_f, egain, weight=True, device=args.device)
