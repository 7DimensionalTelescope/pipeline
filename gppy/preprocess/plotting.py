from astropy.io import fits
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from pathlib import Path
from PIL import Image

from ..path import PathHandler


def save_fits_as_png(image_data, output_path, stretch=True, log_scale=False, max_width=1000):
    # Handle potential NaN or inf values
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if output_path.exists():
        return

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
    image_data = ((image_data - image_data.min()) / (image_data.max() - image_data.min()) * 255).astype(np.uint8)

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
    pil_image.save(output_path, "PNG", optimize=True, compress_level=5)  # Optimize compression


def plot_bias(file, savefig=False):
    path = Path(file)
    os.makedirs(path.parent / "figures", exist_ok=True)
    output_path = path.parent / "figures" / f"{path.stem}_hist.png"

    if output_path.exists():
        return

    data = fits.getdata(file)
    header = fits.getheader(file)
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

    plt.axvspan(
        0,
        header["CLIPMIN"],
        color="gray",
        alpha=0.5,
        label="within CLIPMIN and CLIPMAX",
    )
    plt.axvspan(header["CLIPMAX"], 1e5, color="gray", alpha=0.5)
    plt.yscale("log")
    plt.xlabel("ADU")
    plt.ylabel("Density")
    plt.title("Master Bias")

    plt.xlim(400, 600)
    plt.ylim(1e-6, 10)
    plt.legend()
    

    if savefig:
        plt.savefig(output_path)
        plt.clf()
        save_fits_as_png(data, path.parent / "figures" / f"{path.stem}.png")
    else:
        plt.show(block=False)


def plot_dark(file, fmask=None, savefig=False):
    path = Path(file)
    os.makedirs(path.parent / "figures", exist_ok=True)
    output_path = path.parent / "figures" / f"{path.stem}_hist.png"
    if output_path.exists():
        return
        
    data = fits.getdata(file)
    header = fits.getheader(file)
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

    plt.axvspan(
        -100,
        header["CLIPMIN"],
        color="gray",
        alpha=0.5,
        label="within CLIPMIN and CLIPMAX",
    )
    plt.axvspan(header["CLIPMAX"], 1000, color="gray", alpha=0.5)

    plt.xlim(-50, 50)
    plt.yscale("log")
    plt.xlabel("ADU")
    plt.ylabel("Density")
    plt.title(f"Master Dark")

    plt.ylim(1e-6, 10)
    plt.legend()
    
    if savefig:
        plt.savefig(output_path)
        plt.clf()
        save_fits_as_png(data, path.parent / "figures" / f"{path.stem}.png")
    else:
        plt.show(block=False)

    plot_dark_tail(fdata, file, savefig=savefig)


def plot_dark_tail(fdata, file, savefig=False):
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
    plt.title(f"Master Dark Tail")

    plt.legend()
    

    if savefig:
        path = Path(file)
        os.makedirs(path.parent / "figures", exist_ok=True)
        plt.savefig(path.parent / "figures" / f"{path.stem}_tail.png")
        plt.clf()
    else:
        plt.show(black=False)


def plot_flat(file, fmask=None, savefig=False):

    path = Path(file)
    os.makedirs(path.parent / "figures", exist_ok=True)
    output_path = path.parent / "figures" / f"{path.stem}_hist.png"
    if output_path.exists():
        return

    data = fits.getdata(file)
    header = fits.getheader(file)

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

    plt.axvspan(
        0,
        header["CLIPMIN"],
        color="gray",
        alpha=0.5,
        label="within CLIPMIN and CLIPMAX",
    )
    plt.axvspan(header["CLIPMAX"], 3, color="gray", alpha=0.5)

    plt.yscale("log")
    plt.xlabel("Normalized ADU")
    plt.ylabel("N")
    plt.title(f"Master Flat")
    plt.ylim(
        1e5,
    )
    plt.xlim(0, 1.5)
    plt.legend()
    

    if savefig:
        plt.savefig(output_path)
        plt.clf()
        save_fits_as_png(data, path.parent / "figures" / f"{path.stem}.png")
    else:
        plt.show()


def plot_bpmask(file, ext=1, badpix=1, savefig=False):
    path = Path(file)
    os.makedirs(path.parent / "figures", exist_ok=True)
    output_path = path.parent / "figures" / f"{path.stem}.png"
    data = fits.getdata(file, ext=ext)
    if output_path.exists():
        return data

    header = fits.getheader(file, ext=ext)
    if "BADPIX" in header.keys():
        badpix = header["BADPIX"]

    if badpix == 0:
        cmap = mcolors.ListedColormap(["red", "white"])
    elif badpix == 1:
        cmap = mcolors.ListedColormap(["white", "red"])

    plt.imshow(data, cmap=cmap, interpolation="nearest")
    plt.title(f"Hot pixel mask")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")

    if savefig:
        plt.savefig(output_path)
        plt.clf()
    else:
        plt.show(black=False)
    return data


def plot_sci(output_img):
    path = PathHandler(output_img)
    save_fits_as_png(fits.getdata(output_img), path.figure_dir_to_path / f"{path.stem[0]}.png")
