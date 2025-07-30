import os
import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas  # thread-safe, but savefig only.
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from ..path import PathHandler


def save_fits_as_png(image_data, output_path, stretch=True, log_scale=False, max_width=1000):
    from PIL import Image

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


def plot_bias(file, overwrite=False):
    if not (isinstance(file, str)):
        print("An image path (bias) is not properly defined.")
        return

    path = Path(file)
    os.makedirs(path.parent / "figures", exist_ok=True)
    output_path = path.parent / "figures" / f"{path.stem}_hist.png"

    # if output_path.exists() and not overwrite:
    #     return

    data = fits.getdata(file)
    header = fits.getheader(file)
    fdata = data.ravel()
    clipmin = int(header["CLIPMIN"])
    clipmax = int(header["CLIPMAX"])
    mn = fdata.min()
    mx = fdata.max()

    edges = np.unique(np.concatenate(([mn], np.arange(clipmin, clipmax + 1, 1), [mx])))
    fig = Figure(figsize=(10, 6))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)
    # plt.figure()
    ax.hist(fdata, bins=edges, density=True, alpha=0.6, label="Data", log=True, histtype="step")

    lses = ["--", "-."]
    for i, key in enumerate(["CLIPMEAN", "CLIPMED"]):
        ax.axvline(header[key], linestyle=lses[i], color=f"C{i+1}", label=key)

    label = r"outside 5 clipped $\sigma$"
    ax.axvspan(mn, header["CLIPMEAN"] - 5 * header["CLIPSTD"], color="gray", alpha=0.2, label=label)
    ax.axvspan(header["CLIPMEAN"] + 5 * header["CLIPSTD"], mx, color="gray", alpha=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("ADU")
    ax.set_ylabel("Density")
    ax.set_title("Master Bias")
    scope = (350, 700)  # (400, 600) (clipmin, clipmax)
    ax.set_xlim(*scope)
    # plt.ylim(1e-7, 10)
    ax.legend(loc=2)

    # Inset to show the right tail
    axins = inset_axes(ax, width="30%", height="30%", loc="upper right", borderpad=2)
    tail = fdata[(fdata >= scope[1])]
    axins.hist(tail, bins=100, density=False, histtype="step")
    axins.set_xlim(scope[1], mx + 100)
    # axins.set_yscale("log")
    axins.tick_params(axis="both", which="major", labelsize=8)
    axins.set_title("Right-tail zoom", fontsize=9)

    canvas.print_figure(output_path)
    # plt.savefig(output_path)
    # plt.close()
    save_fits_as_png(data, path.parent / "figures" / f"{path.stem}.png")


def plot_dark(file, flattened_mask=None):
    if not (isinstance(file, str)):
        print("An image path (dark) is not properly defined.")
        return

    path = Path(file)
    os.makedirs(path.parent / "figures", exist_ok=True)
    output_path = path.parent / "figures" / f"{path.stem}_hist.png"
    # if output_path.exists():
    #     return

    data = fits.getdata(file)
    header = fits.getheader(file)
    fdata = data.ravel()

    scope = (-170, 200)
    mn = fdata.min()
    mx = fdata.max()
    edges = np.unique(np.concatenate(([mn], np.arange(scope[0], scope[1] + 1, 1), [mx])))

    # fig, ax = plt.subplots(figsize=(10, 6))
    fig = Figure(figsize=(10, 6))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)
    axins = inset_axes(ax, width="40%", height="40%", loc="upper right", borderpad=2)

    # unmasked data
    ax.hist(fdata, bins=edges, density=True, alpha=0.6, label="Unmasked Data", histtype="step")
    plot_dark_tail_on_ax(fdata, ax=axins, i=0, mx=mx)

    # percentiles (unmasked)
    fdata_pos = fdata[fdata > 0]
    # p99 = np.percentile(fdata_pos, 99, method="nearest")
    # p999 = np.percentile(fdata_pos, 99.9, method="nearest")
    # ax.axvline(p99, color="k", linestyle=":", label="99th pct")
    # ax.axvline(p999, color="k", label="99.9th pct")
    p9973 = np.percentile(fdata_pos, 99.73, method="nearest")
    ax.axvline(p9973, color="k", ls="-", label="99.73th percentile (unmasked)")

    # masked data
    fdata = fdata[flattened_mask] if flattened_mask is not None else fdata
    ax.hist(fdata, bins=edges, density=True, alpha=0.6, label="Masked Data", histtype="step", linestyle="--")
    plot_dark_tail_on_ax(fdata, ax=axins, i=1, mx=mx)

    # percentiles (masked)
    fdata_pos = fdata[fdata > 0]
    if len(fdata_pos) > 0:
        p9973 = np.percentile(fdata_pos, 99.73, method="nearest")
        ax.axvline(p9973, color="k", ls=":", label="99.73th percentile (masked)")

    # mean and median lines
    lses = ["--", "-."]
    for i, key in enumerate(["CLIPMEAN", "CLIPMED"]):
        ax.axvline(header[key], linestyle=lses[i], color=f"C{i+1}", label=f"{key}: {header[key]:.4f}")

    # 5-sigma shade
    label = r"outside 5 clipped $\sigma$"
    ax.axvspan(mn, header["CLIPMEAN"] - 5 * header["CLIPSTD"], color="gray", alpha=0.1, label=label)
    ax.axvspan(header["CLIPMEAN"] + 5 * header["CLIPSTD"], mx, color="gray", alpha=0.1)

    ax.set_xlim(*scope)
    # plt.xscale("symlog")
    ax.set_yscale("log")
    ax.set_xlabel("ADU")
    ax.set_ylabel("Density")
    ax.set_title(f"Master Dark")

    # plt.ylim(1e-6, 10)
    ax.legend(loc=2)

    canvas.print_figure(output_path)  # writes to PNG
    # plt.savefig(output_path)
    # plt.close()
    save_fits_as_png(data, path.parent / "figures" / f"{path.stem}.png")

    # plot_dark_tail(fdata, file, savefig=savefig)


def plot_dark_tail_on_ax(fdata, ax, i=0, mx=None):
    fdata_tail = fdata[fdata > 200]
    lses = ["-", "--"]
    bins = np.unique(np.concatenate((np.geomspace(200, 2**15), np.geomspace(2**15, mx or 2 * 16 - 1, 10))))
    ax.hist(fdata_tail, bins=bins, histtype="step", alpha=0.6, linestyle=lses[i], color=f"C{i}")

    ax.set_xlim(200, 2**16 - 1)
    # ax.set_xscale("symlog")
    ax.set_yscale("log")
    ax.set_xlabel("ADU", fontsize=8)
    ax.set_ylabel("N", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=6)
    ax.set_title("Tail", fontsize=10)
    # ax.legend(fontsize=6)
    return ax


# def plot_dark_tail(fdata, file, savefig=False):
#     p99 = np.percentile(fdata, 99, method="nearest")
#     p999 = np.percentile(fdata, 99.9, method="nearest")
#     fdata = fdata[fdata > 0]

#     plt.figure(figsize=(10, 6))
#     plt.hist(
#         fdata,
#         bins=np.geomspace(0.8, 1e4, 20),
#         density=False,
#         alpha=0.6,
#         label="Data (ADU > 0)",
#         histtype="step",
#     )
#     plt.axvline(p99, color="gray", label="99th percentile")
#     plt.axvline(p999, color="gray", ls=":", label="99.9th percentile")
#     plt.xscale("log")
#     plt.yscale("log")
#     plt.xlabel("ADU")
#     plt.ylabel("N")
#     plt.title(f"Master Dark Tail")

#     plt.legend()

#     if savefig:
#         path = Path(file)
#         os.makedirs(path.parent / "figures", exist_ok=True)
#         plt.savefig(path.parent / "figures" / f"{path.stem}_tail.png")
#         plt.clf()
#     else:
#         plt.show(block=False)


def plot_flat(file, fmask=None):
    if not (isinstance(file, str)):
        print("An image path (flat) is not properly defined.")
        return
    path = Path(file)
    os.makedirs(path.parent / "figures", exist_ok=True)
    output_path = path.parent / "figures" / f"{path.stem}_hist.png"
    # if output_path.exists():
    #     return

    data = fits.getdata(file)
    header = fits.getheader(file)

    fig = Figure(figsize=(10, 6))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)

    fdata = data.ravel()
    # unmasked data
    ax.hist(fdata, bins=100, color="C0", histtype="step", label="Unmasked Data")

    fdata = fdata[fmask] if fmask is not None else fdata

    # masked data
    ax.hist(fdata, bins=100, color="C1", histtype="step", linestyle="--", label="Masked Data")

    lses = ["--", "-."]
    for i, key in enumerate(["CLIPMEAN", "CLIPMED"]):
        ax.axvline(header[key], linestyle=lses[i], color=f"C{i+1}", label=f"{key}: {header[key]:.4f}")

    # label = r"outside 5 clipped $\sigma$, inside CLIPMIN/CLIPMAX"
    label = r"outside CLIPMIN/CLIPMAX"
    x_min, x_max = ax.get_xlim()
    ax.axvspan(x_min, header["CLIPMIN"], color="gray", alpha=0.1, label=label)
    ax.axvspan(header["CLIPMAX"], x_max, color="gray", alpha=0.1)
    ax.axvspan(
        header["CLIPMEAN"] - 1 * header["CLIPSTD"],
        header["CLIPMEAN"] + 1 * header["CLIPSTD"],
        color="gray",
        alpha=0.3,
        label=r"inside 1 clipped $\sigma$",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Normalized ADU")
    ax.set_ylabel("N")
    ax.set_title(f"Master Flat")
    # ax.set_ylim(1e5)
    ax.set_xlim(x_min, x_max)
    # ax.set_xlim(0, 1.5)
    ax.legend()

    canvas.print_figure(output_path)
    # fig.savefig(output_path)
    # fig.close()

    save_fits_as_png(data, path.parent / "figures" / f"{path.stem}.png")


def plot_bpmask(file, ext=1, badpix=1):
    if not (isinstance(file, str)):
        print("An image path (bpmask) is not properly defined.")
        return
    path = Path(file)
    os.makedirs(path.parent / "figures", exist_ok=True)
    output_path = path.parent / "figures" / f"{path.stem}.png"
    data = fits.getdata(file, ext=ext)
    # if output_path.exists():
    #     return

    header = fits.getheader(file, ext=ext)
    if "BADPIX" in header.keys():
        badpix = header["BADPIX"]

    if badpix == 0:
        cmap = mcolors.ListedColormap(["red", "white"])
    elif badpix == 1:
        cmap = mcolors.ListedColormap(["white", "red"])

    # plt.figure()
    fig = Figure(figsize=(6.4, 4.8))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)

    ax.imshow(data, cmap=cmap, interpolation="nearest")
    ax.set_title(f"Hot pixel mask")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")

    fig.tight_layout()
    canvas.print_figure(output_path)
    # plt.savefig(output_path)
    # plt.close()


def plot_sci(input_img, output_img):
    if not (isinstance(input_img, str)):
        print("An image path (input_img) is not properly defined.")
        return
    if not (isinstance(output_img, str)):
        print("An image path (output_img) is not properly defined.")
        return
    path = PathHandler(output_img)
    save_fits_as_png(fits.getdata(input_img), path.figure_dir_to_path / f"{path.stem[0]}_raw.png")
    save_fits_as_png(fits.getdata(output_img), path.figure_dir_to_path / f"{path.stem[0]}.png")
