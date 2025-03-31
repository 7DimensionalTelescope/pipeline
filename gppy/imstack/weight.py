import os
from astropy.io import fits
from ..utils import swap_ext


def pix_err(xp, r_p, d_m, f_m, sig_b, sig_z, sig_f, p_d, p_z, p_f, weight=True):
    nonneg = lambda x: xp.maximum(x, 0)
    sig_r = xp.sqrt(nonneg(r_p + d_m) + sig_z**2)
    sig_zm = sig_z / xp.sqrt(p_z)
    sig_dm = xp.sqrt((nonneg(d_m) + (1 + 1 / p_z) * sig_z**2) / p_d)
    sig_fm = sig_f / xp.sqrt(p_f)

    sig_rp = (sig_r**2 + sig_zm**2 + sig_dm**2) / f_m**2 + (r_p / f_m) ** 2 * sig_fm**2

    if weight:
        return 1 / (sig_rp**2 + sig_b**2)  # 1/sigma**2
    else:
        return xp.sqrt(sig_rp**2 + sig_b**2)  # sigma


def calculate_weight(config):
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

    # for r_p, sig_b, weight_file in zip(
    #     config.combine.bkgsub_files,
    #     config.combine.bkg_rms_files,
    #     config.file.weight_files,
    # ):
    for r_p_file, sig_b_file in zip(
        config.imstack.bkgsub_files, config.imstack.bkg_rms_files
    ):
        r_p = xp.asarray(fits.getdata(r_p_file))
        sig_b = xp.asarray(fits.getdata(sig_b_file))
        weight_image = pix_err(
            xp, r_p, d_m, f_m, sig_b, sig_z, sig_f, p_d, p_z, p_f, weight=True
        )

        if hasattr(weight_image, "get"):  # if CuPy array
            weight_image = weight_image.get()  # Convert to NumPy array

        fits.writeto(
            # os.path.join(config.path.path_processed, weight_file),
            swap_ext(r_p_file, "weight.fits"),
            data=weight_image,
            overwrite=True,
        )
