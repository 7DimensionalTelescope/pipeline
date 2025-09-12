import numpy as np
from astropy.io import fits
import subprocess


def convolve_fft_subprocess(
    images,
    outout,
    kernels=None,
    mode="same",
    normalize_kernel=False,
    device=None,
    apply_edge_mask=False,
    method=None,
    delta_peeing=None,
):

    # base command
    cmd = [
        "python",
        f"{SCRIPT_DIR}/cuda/convolve_fft.py",
        "-input",
        *images,
        "-output",
        *output,
        "-kernels",
        None,
        "-mode",
        str(mode),
        "-normalize_kernel",
        normalize_kernel,
        "-apply_edge_mask",
        str(apply_edge_mask),
        "=method",
        method,
        "-delta_peeing",
        delta_peeing,
        "-device",
        str(device),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Error combining images: {result.stderr}")

    return None


def convolve_fft_cpu(
    images,
    output,
    kernels,
    mode="same",
    normalize_kernel=False,
    device=None,
    apply_edge_mask=False,
    method=None,
    delta_peeing=None,
):
    h, w = fits.getdata(images[0]).shape
    cpu_buffer = np.empty((h, w))

    for i, (image, kernel) in enumerate(zip(images, kernels)):
        cpu_buffer[:] = fits.getdata(image)

        cpu_buffer[:] = convolve_fft_with_astropy(cpu_buffer, kernel, normalize_kernel=normalize_kernel)

        if apply_edge_mask:
            edge_mask = get_edge_mask_cpu(cpu_buffer, kernel)
            cpu_buffer[~edge_mask] = 0

        header = add_conv_header(fits.getheader(image), delta_peeing[i], method)
        fits.writeto(
            output[i],
            data=cpu_buffer.copy(),
            header=header,
            overwrite=True,
        )


def convolve_fft_with_astropy(image, kernel, normalize_kernel=False):
    from astropy.convolution import convolve_fft

    return convolve_fft(image, kernel, normalize_kernel=normalize_kernel)


def get_edge_mask(weight_image, kernel, device_id=None):
    if device_id == "CPU" or device_id is None:
        return get_edge_mask_cpu(weight_image, kernel)
    else:
        return get_edge_mask_gpu(weight_image, kernel, device_id)


def get_edge_mask_cpu(weight_image, kernel):
    from scipy.ndimage import binary_dilation

    mask = weight_image != 0  # 0 for edge, 1 for inside

    size = np.shape(kernel)
    struct = np.ones(size, dtype=bool)
    dilated_mask = binary_dilation(mask, structure=struct)
    return dilated_mask  # mask for inside


def add_conv_header(header, delta_peeing, method):
    header["CONV"] = (method.upper(), "Method for seeing-match convolution")
    header['CONVSIZE"'] = (delta_peeing, "Convolution kernel FWHM in pixels")
    return header
