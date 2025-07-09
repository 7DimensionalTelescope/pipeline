import argparse
import sys
from astropy.io import fits
from ..utils import add_suffix  # same for add_suffix
from .weight import calc_weight


def main():
    p = argparse.ArgumentParser(description="Compute weight maps for a batch of images")
    p.add_argument("--d_m_file", required=True, help="Dark master FITS file")
    p.add_argument("--f_m_file", required=True, help="Flat master FITS file")
    p.add_argument("--sig_z_file", required=True, help="Sigma-Z FITS file")
    p.add_argument("--sig_f_file", required=True, help="Sigma-F FITS file")
    # p.add_argument("--p_d", type=int, required=True, help="NFRAMES in dark")
    # p.add_argument("--p_z", type=int, required=True, help="NFRAMES in sigma-Z")
    # p.add_argument("--p_f", type=int, required=True, help="NFRAMES in flat")
    p.add_argument("--device", default="CPU", help="Device identifier for calc_weight")
    p.add_argument("images", nargs="+", help="List of science images to process")

    uncalculated_images = args.images
    args = p.parse_args()
    sig_z_file, d_m_file, sig_f_file, f_m_file = args.sig_z_file, args.d_m_file, args.sig_f_file, args.f_m_file

    mfg_data = []
    mfg_frames = []
    for file in [sig_z_file, d_m_file, f_m_file]:
        data, header = fits.getdata(file, header=True)
        mfg_data.append(data)
        mfg_frames.append(header["NFRAMES"])
        if file == d_m_file:
            egain = header["EGAIN"]  # e-/ADU

    sig_z, d_m, f_m = mfg_data
    p_z, p_d, p_f = mfg_frames
    sig_f = fits.getdata(sig_f_file)

    # Run weight‐map calculation
    output = calc_weight(
        uncalculated_images, d_m, f_m, sig_z, sig_f, p_d, p_z, p_f, egain, weight=True, device=args.device
    )

    # Write out each weight‐map
    for img, weight_image in zip(uncalculated_images, output):
        outname = add_suffix(img, "weight")
        fits.writeto(outname, weight_image, overwrite=True)


if __name__ == "__main__":
    main()
