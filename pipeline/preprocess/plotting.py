import os
import numpy as np
import fitsio
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from PIL import Image, ImageEnhance
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas  # thread-safe, but savefig only.
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from ..path import PathHandler


def plot_outputs_exist(output_paths) -> bool:
    return all(os.path.exists(output_path) for output_path in output_paths)

def save_fits_as_figures(image_data, output_path, stretch=True, log_scale=False, max_width=1000, overwrite=False):
    if os.path.exists(output_path) and not overwrite:
        return

    if isinstance(image_data, str):
        image_data = fitsio.read(image_data)

    image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)

    # Optional logarithmic scaling
    if log_scale:
        image_data = np.log1p(image_data)

    # Optional stretching for better contrast
    if stretch:
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
    pil_image.save(output_path, "JPEG", quality=85, optimize=True)


def save_fits_with_contrast(image_data, output_path, max_width=1000, overwrite=False):
    if os.path.exists(output_path) and not overwrite:
        return None

    data = fitsio.read(image_data)

    # Get clipping parameters from header
    clipmed = fits.getval(image_data, "CLIPMED")
    clipstd = fits.getval(image_data, "CLIPSTD")

    lower_bound = clipmed - 3 * clipstd
    upper_bound = clipmed + 3 * clipstd

    clipped_data = np.clip(data, lower_bound, upper_bound)

    if upper_bound > lower_bound:
        normalized_data = ((clipped_data - lower_bound) / (upper_bound - lower_bound) * 255).astype(np.uint8)
    else:
        normalized_data = np.full_like(data, 128, dtype=np.uint8)

    # Create PIL image
    pil_image = Image.fromarray(normalized_data, mode="L")  # Grayscale

    # Apply contrast enhancement
    enhancer = ImageEnhance.Contrast(pil_image)
    enhanced_image = enhancer.enhance(2.0)  # Increase contrast by factor of 4

    width, height = enhanced_image.size

    if width > max_width:
        scale = max_width / width
        new_width = max_width
        new_height = int(height * scale)
        enhanced_image = enhanced_image.resize((new_width, new_height), Image.LANCZOS)

    enhanced_image.save(output_path, "JPEG", quality=85, optimize=True)

    return enhanced_image


def plot_bias(file, overwrite=False, dry_run: bool = False):
    if not (isinstance(file, str)):
        print("An image path (bias) is not properly defined.")
        return

    figures = PathHandler(file).preprocess.figures
    hist_path = figures._bias_hist
    image_path = figures._bias_image
    contrast_path = figures._bias_contrast
    if dry_run:
        print(hist_path)
        print(image_path)
        print(contrast_path)
        return
    if plot_outputs_exist((hist_path, image_path, contrast_path)) and not overwrite:
        return

    need_hist = overwrite or not os.path.exists(hist_path)
    need_image = overwrite or not os.path.exists(image_path)
    need_contrast = overwrite or not os.path.exists(contrast_path)
    data = None

    if need_hist or need_image:
        data = fitsio.read(file)

    if need_hist:
        header = fits.getheader(file)
        fdata = data.ravel()
        clipmin = int(header["CLIPMIN"])
        clipmax = int(header["CLIPMAX"])
        mn = fdata.min()
        mx = fdata.max()
        scope = (350, 700)  # (400, 600) (clipmin, clipmax)

        # edges = np.unique(np.concatenate(([mn], np.arange(clipmin, clipmax + 1, 1), [mx])))
        edges = np.unique(np.concatenate(([mn], np.arange(scope[0], scope[1] + 1, 1), [mx]))) + 0.5
        fig = Figure(figsize=(10, 6))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(1, 1, 1)
        # plt.figure()
        ax.hist(fdata, bins=edges, density=True, alpha=0.6, label="Data", log=True, histtype="step")

        lses = ["--", "-."]
        for i, key in enumerate(["CLIPMEAN", "CLIPMED"]):
            ax.axvline(header[key], linestyle=lses[i], color=f"C{i+1}", label=f"{key}: {header[key]:.4f}")

        label = r"outside 5 clipped $\sigma$"
        ax.axvspan(mn, header["CLIPMEAN"] - 5 * header["CLIPSTD"], color="gray", alpha=0.2, label=label)
        ax.axvspan(header["CLIPMEAN"] + 5 * header["CLIPSTD"], mx, color="gray", alpha=0.2)
        ax.set_yscale("log")
        ax.set_xlabel("ADU")
        ax.set_ylabel("Density")
        ax.set_title("Master Bias")
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

        canvas.print_figure(figures.bias_hist)

    if need_image:
        save_fits_as_figures(data if data is not None else file, figures.bias_image, overwrite=overwrite)

    if need_contrast:
        save_fits_with_contrast(file, figures.bias_contrast, overwrite=overwrite)


def plot_dark(file, flattened_mask=None, bpmask_file=None, overwrite=False, dry_run: bool = False):
    if not (isinstance(file, str)):
        print("An image path (dark) is not properly defined.")
        return

    figures = PathHandler(file).preprocess.figures
    hist_path = figures._dark_hist
    image_path = figures._dark_image
    contrast_path = figures._dark_contrast
    if dry_run:
        print(hist_path)
        print(image_path)
        print(contrast_path)
        return
    if plot_outputs_exist((hist_path, image_path, contrast_path)) and not overwrite:
        return

    need_hist = overwrite or not os.path.exists(hist_path)
    need_image = overwrite or not os.path.exists(image_path)
    need_contrast = overwrite or not os.path.exists(contrast_path)
    data = None

    if need_hist or need_image:
        data = fitsio.read(file)

    if need_hist:
        header = fits.getheader(file)
        fdata = data.ravel()
        if flattened_mask is None and bpmask_file:
            badpix = fits.getval(bpmask_file, "BADPIX", ext=1)
            if badpix is None:
                badpix = 1
            flattened_mask = fits.getdata(bpmask_file, ext=1) != badpix
            flattened_mask = flattened_mask.ravel()

        scope = (-170, 200)
        mn = fdata.min()
        mx = fdata.max()
        edges = np.unique(np.concatenate(([mn], np.arange(scope[0], scope[1] + 1, 1), [mx]))) + 0.5

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
        ax.legend(loc=2)

        canvas.print_figure(figures.dark_hist)

    if need_image:
        save_fits_as_figures(data if data is not None else file, figures.dark_image, overwrite=overwrite)

    if need_contrast:
        save_fits_with_contrast(file, figures.dark_contrast, overwrite=overwrite)


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
#         plt.savefig(path.parent / "figures" / f"{path.stem}_tail.jpg")
#         plt.clf()
#     else:
#         plt.show(block=False)


def plot_flat(file, fmask=None, bpmask_file=None, overwrite=False, dry_run: bool = False):
    if not (isinstance(file, str)):
        print("An image path (flat) is not properly defined.")
        return
    figures = PathHandler(file).preprocess.figures
    hist_path = figures._flat_hist
    image_path = figures._flat_image
    if dry_run:
        print(hist_path)
        print(image_path)
        return
    if plot_outputs_exist((hist_path, image_path)) and not overwrite:
        return

    need_hist = overwrite or not os.path.exists(hist_path)
    need_image = overwrite or not os.path.exists(image_path)
    data = None

    if need_hist or need_image:
        data = fitsio.read(file)

    if need_hist:
        header = fits.getheader(file)
        if fmask is None and bpmask_file:
            badpix = fits.getval(bpmask_file, "BADPIX", ext=1)
            if badpix is None:
                badpix = 1
            fmask = fits.getdata(bpmask_file, ext=1) != badpix
            fmask = fmask.ravel()

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
        ax.set_xlim(x_min, x_max)
        ax.legend()

        canvas.print_figure(figures.flat_hist)

    if need_image:
        save_fits_as_figures(data if data is not None else file, figures.flat_image, overwrite=overwrite)


def plot_bpmask(file, ext=1, badpix=1, overwrite=False, dry_run: bool = False):
    if not (isinstance(file, str)):
        print("An image path (bpmask) is not properly defined.")
        return
    figures = PathHandler(file).preprocess.figures
    output_path = figures._bpmask_image
    if dry_run:
        print(output_path)
        return
    if os.path.exists(output_path) and not overwrite:
        return

    data = fitsio.read(file, ext=ext)

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
    canvas.print_figure(figures.bpmask_image)


def plot_sci(input_img, output_img, is_too=False, overwrite=False, dry_run: bool = False):
    if not (isinstance(input_img, str)):
        print("An image path (input_img) is not properly defined.")
        return
    if not (isinstance(output_img, str)):
        print("An image path (output_img) is not properly defined.")
        return
    figures = PathHandler(output_img, is_too=is_too).preprocess.figures
    raw_output_path = figures._science_raw
    processed_output_path = figures._science_processed
    if dry_run:
        print(raw_output_path)
        print(processed_output_path)
        return

    if plot_outputs_exist((raw_output_path, processed_output_path)) and not overwrite:
        return

    save_fits_as_figures(input_img, figures.science_raw, overwrite=overwrite)
    save_fits_as_figures(output_img, figures.science_processed, overwrite=overwrite)
