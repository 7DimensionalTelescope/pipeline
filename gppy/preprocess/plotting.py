from astropy.io import fits
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import matplotlib.colors as mcolors
from pathlib import Path
from PIL import Image
from .utils import read_link


def save_fits_as_png(
    image_data, output_path, stretch=True, log_scale=False, max_width=1000
):
    # Handle potential NaN or inf values
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if isinstance(image_data, str):
        image_data = fits.getdata(image_data)

    image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)

    # Optional logarithmic scaling
    if log_scale:
        image_data = np.log1p(image_data)

    # Optional stretching for better contrast
    if stretch:
        # # Percentile-based stretching
        # p1, p99 = np.percentile(image_data[np.isfinite(image_data)], (1, 99))
        # vmin, vmax = p1, p99

        from astropy.visualization import ZScaleInterval

        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(image_data[np.isfinite(image_data)])

        image_data = np.clip(image_data, vmin, vmax)

    # Normalize to 0-255 range for 8-bit image
    image_data = (
        (image_data - image_data.min()) / (image_data.max() - image_data.min()) * 255
    ).astype(np.uint8)

    # Convert to Pillow Image
    pil_image = Image.fromarray(image_data)

    # Resize image if it's larger than max_width
    width, height = pil_image.size
    if width > max_width:
        scale = max_width / width
        new_width = max_width
        new_height = int(height * scale)
        pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)

    # Save with compression
    pil_image.save(
        output_path, "PNG", optimize=True, compress_level=5  # Optimize compression
    )


def plot_bias(file, savefig=False):
    if ".link" in file:
        orig_file = read_link(file)
    data = fits.getdata(orig_file)
    header = fits.getheader(orig_file)
    fdata = data.ravel()

    plt.hist(
        fdata,
        bins=100,
        density=True,
        alpha=0.6,
        label="Data",
        log=True,
        histtype="step",
    )


    for i, key in enumerate(["CLIPMEAN", "CLIPMED", "CLIPSTD"]):
        plt.axvline(header[key], linestyle="--", color=f"C{i+1}", label=key)

    plt.axvspan(0, header["CLIPMIN"], color="gray", alpha=0.5, label="within CLIPMIN and CLIPMAX")
    plt.axvspan(header["CLIPMAX"], 1e5, color="gray", alpha=0.5)
    plt.yscale("log")
    plt.xlabel("ADU")
    plt.ylabel("Density")
    plt.title("Master Bias")

    plt.xlim(400, 600)
    plt.ylim(1e-6, 10)
    plt.legend()
    plt.tight_layout()

    if savefig:
        path = Path(file)
        os.makedirs(path.parent / "images", exist_ok=True)
        plt.savefig(path.parent / "images" / "master_bias_flatten.png")
        plt.clf()
        save_fits_as_png(data, path.parent / "images" / "master_bias.png")
    else:
        plt.show(block=False)


def plot_dark(file_dict, fmask=None, savefig=False, badpix=0):
    if isinstance(file_dict, str):
        file_dict = {0: file_dict}
    elif isinstance(file_dict, list):
        file_dict = {i: file for i, file in enumerate(file_dict)}

    for exposure, file in file_dict.items():
        if not os.path.exists(file):
            continue
        if ".link" in file:
            orig_file = read_link(file)
        data = fits.getdata(orig_file)
        header = fits.getheader(orig_file)
        fdata = data.ravel()
        fdata = fdata[fmask] if fmask is not None else fdata
        
        plt.hist(
            fdata,
            bins=30,
            density=True,
            alpha=0.6,
            label="Masked Data",
            histtype="step",
        )
        
        for i, key in enumerate(["CLIPMEAN", "CLIPMED", "CLIPSTD"]):
            plt.axvline(header[key], linestyle="--", color=f"C{i+1}", label=key)

        plt.axvspan(-100, header["CLIPMIN"], color="gray", alpha=0.5, label="within CLIPMIN and CLIPMAX")
        plt.axvspan(header["CLIPMAX"], 1000, color="gray", alpha=0.5)
        
        plt.xlim(-50, 50)
        plt.yscale("log")
        plt.xlabel("ADU")
        plt.ylabel("Density")
        plt.title(f"Master Dark (exp. {exposure}s)")

        plt.ylim(1e-6, 10)
        plt.legend()
        plt.tight_layout()
        if savefig:
            path = Path(file)
            os.makedirs(path.parent / "images", exist_ok=True)
            plt.savefig(path.parent / "images" / f"master_dark_flatten_{exposure}s.png")
            plt.clf()
            save_fits_as_png(
                data, path.parent / "images" / f"master_dark_{exposure}s.png"
            )
        else:
            plt.show(block=False)

        plot_dark_tail(fdata, exposure, file, savefig=savefig)


def plot_dark_tail(fdata, exposure, file, savefig=False):
    p99 = np.percentile(fdata, 99, method="nearest")
    p999 = np.percentile(fdata, 99.9, method="nearest")
    fdata = fdata[fdata > 0]

    plt.figure(figsize=(10, 6))
    plt.hist(
        fdata,
        bins=np.geomspace(0.8, 1e4, 20),
        density=False,
        alpha=0.6,
        label="Data (ADU > 0)",
        histtype="step",
    )
    plt.axvline(p99, color="gray", label="99th percentile")
    plt.axvline(p999, color="gray", ls=":", label="99.9th percentile")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("ADU")
    plt.ylabel("N")
    plt.title(f"Master Dark Tail (exp. {exposure}s)")

    plt.legend()
    plt.tight_layout()

    if savefig:
        path = Path(file)
        os.makedirs(path.parent / "images", exist_ok=True)
        plt.savefig(path.parent / "images" / f"master_dark_tail_{exposure}s.png")
        plt.clf()
    else:
        plt.show(black=False)


def plot_flat(file_dict, fmask=None, badpix=1, savefig=False):
    if isinstance(file_dict, str):
        file_dict = {0: file_dict}
    elif isinstance(file_dict, list):
        file_dict = {i: file for i, file in enumerate(file_dict)}

    for filt, file in file_dict.items():
        if not (os.path.exists(file)):
            continue
        if ".link" in file:
            orig_file = read_link(file)
        
        data = fits.getdata(orig_file)
        header = fits.getheader(orig_file)

        fdata = data.ravel()
        fdata = fdata[fmask] if fmask is not None else fdata

        plt.hist(
            fdata,
            bins=100,
            color="C0",
            histtype="step",
            label="Masked Data",
        )

        for i, key in enumerate(["CLIPMEAN", "CLIPMED"]):
            plt.axvline(header[key], linestyle="--", color=f"C{i+1}", label=key)
        
        plt.axvspan(0, header["CLIPMIN"], color="gray", alpha=0.5, label="within CLIPMIN and CLIPMAX")
        plt.axvspan(header["CLIPMAX"], 3, color="gray", alpha=0.5)
            
        plt.yscale("log")
        plt.xlabel("Normalized ADU")
        plt.ylabel("N")
        plt.title(f"Master Flat (filter: {filt})")
        plt.ylim(1e5,)
        plt.xlim(0, 1.5)
        plt.legend()
        plt.tight_layout()

        if savefig:
            path = Path(file)
            os.makedirs(path.parent / "images", exist_ok=True)
            plt.savefig(path.parent / "images" / f"master_flat_flatten_{filt}.png")
            plt.clf()
            save_fits_as_png(data, path.parent / "images" / f"master_flat_{filt}.png")
        else:
            plt.show()


def plot_bpmask(file_dict, ext=1, badpix=1, savefig=False):
    for exposure, file in file_dict.items():
        if not (os.path.exists(file)):
            continue
        if ".link" in file:
            file = read_link(file)
        data = fits.getdata(file, ext=ext)

        header = fits.getheader(file, ext=ext)
        if "BADPIX" in header.keys():
            badpix = header["BADPIX"]

        if badpix == 0:
            cmap = mcolors.ListedColormap(["red", "white"])
        elif badpix == 1:
            cmap = mcolors.ListedColormap(["white", "red"])

        plt.imshow(data, cmap=cmap, interpolation="nearest")
        plt.title(f"Hot pixel mask (exp. {exposure}s)")
        plt.xlabel("X (pixels)")
        plt.ylabel("Y (pixels)")

        if savefig:
            path = Path(file)
            os.makedirs(path.parent / "images", exist_ok=True)
            plt.savefig(path.parent / "images" / f"master_dark_bpmask_{exposure}s.png")
            plt.clf()
        else:
            plt.show(block=False)

    return data
