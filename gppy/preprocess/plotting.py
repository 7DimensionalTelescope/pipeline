from astropy.io import fits
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import matplotlib.colors as mcolors
from pathlib import Path
from PIL import Image


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
        # Percentile-based stretching
        p1, p99 = np.percentile(image_data[np.isfinite(image_data)], (1, 99))
        image_data = np.clip(image_data, p1, p99)

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
    data = fits.getdata(file)
    fdata = data.ravel()
    lower_cut = np.percentile(fdata, 0.5, method="nearest")
    upper_cut = np.percentile(fdata, 99.5, method="nearest")
    selected = fdata[(fdata > lower_cut) & (fdata < upper_cut)]

    mu, sigma = norm.fit(selected)

    fdata = data.ravel()

    # Generate fitted curve
    x = np.linspace(fdata.min(), fdata.max(), 200)
    pdf = norm.pdf(x, mu, sigma)

    plt.hist(
        fdata,
        bins=100,
        density=True,
        alpha=0.6,
        label="Data",
        log=True,
        histtype="step",
    )
    plt.axvspan(
        lower_cut, upper_cut, color="gray", alpha=0.3, label="99% Data", zorder=-1
    )
    plt.plot(
        x, pdf, "r-", lw=2, label=f"Gaussian Fit\n$\mu={mu:.2f}, \sigma={sigma:.2f}$"
    )
    plt.axvline(mu, color="r", linestyle="--", label="Mean")

    plt.yscale("log")
    plt.xlabel("ADU")
    plt.ylabel("Density")
    plt.title("Master Bias")

    plt.ylim(1e-7, 10)
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


def plot_dark(file_dict, savefig=False):
    if isinstance(file_dict, str):
        file_dict = {0: file_dict}
    elif isinstance(file_dict, list):
        file_dict = {i: file for i, file in enumerate(file_dict)}

    for exposure, file in file_dict.items():
        data = fits.getdata(file)
        fdata = data.ravel()
        mask = fits.getdata(file.replace("dark", "bpmask"))
        fmask = mask.ravel()
        fdata_selected = fdata[fmask == 0]
        fdata = fdata[fdata < np.percentile(fdata, 99.9)]

        mu, sigma = norm.fit(fdata_selected)
        mask_min, mask_max = fdata_selected.min(), fdata_selected.max()
        x = np.linspace(mask_min, mask_max, 200)
        pdf = norm.pdf(x, mu, sigma)

        plt.hist(
            fdata,
            bins=100,
            density=True,
            alpha=0.6,
            label="99.9% Data",
            histtype="step",
        )
        plt.axvspan(
            -200, mask_min, color="gray", alpha=0.3, label="Hot-pixels", zorder=-1
        )
        plt.axvspan(mask_max, 200, color="gray", alpha=0.3, zorder=-1)
        plt.plot(
            x,
            pdf,
            "r-",
            lw=2,
            label=f"Gaussian Fit\n$\mu={mu:.2f}, \sigma={sigma:.2f}$",
        )
        plt.axvline(mu, color="r", linestyle="--", label="Mean")
        plt.xlim(-200, 200)
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


def plot_flat(file_dict, mask=None, savefig=False):
    if isinstance(file_dict, str):
        file_dict = {0: file_dict}
    elif isinstance(file_dict, list):
        file_dict = {i: file for i, file in enumerate(file_dict)}

    if mask is not None:
        fmask = mask.ravel()
        fmask = fmask == 0
    else:
        fmask = None
    for filt, file in file_dict.items():
        if os.path.exists(file):
            path = Path(file)
            data = fits.getdata(file)
            fdata = data.ravel()
            plt.hist(
                fdata,
                bins=np.arange(0.1, 3, step=0.05),
                color="C0",
                histtype="step",
                label="Data",
            )
            if fmask is not None:
                plt.hist(
                    fdata[fmask],
                    bins=np.arange(0.1, 3, step=0.05),
                    color="C0",
                    alpha=0.5,
                    histtype="stepfilled",
                    label=f"Hot-pixel-removed",
                )
            plt.yscale("log")
            plt.xlabel("Normalized ADU")
            plt.ylabel("N")
            plt.title(f"Master Flat (filter: {filt})")

            plt.legend()
            plt.tight_layout()

            if savefig:
                path = Path(file)
                os.makedirs(path.parent / "images", exist_ok=True)
                plt.savefig(path.parent / "images" / f"master_flat_flatten_{filt}.png")
                plt.clf()
                save_fits_as_png(
                    data, path.parent / "images" / f"master_flat_{filt}.png"
                )
            else:
                plt.show()


def plot_bpmask(file_dict, savefig=False):
    for exposure, file in file_dict.items():
        data = fits.getdata(file)
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
