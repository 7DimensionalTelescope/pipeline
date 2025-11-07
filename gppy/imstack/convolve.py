import numpy as np
from astropy.io import fits
import subprocess

from ..const import SCRIPT_DIR


def convolve_fft_subprocess(
    images,
    output,
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
    ]

    # Add mode
    cmd.extend(["-mode", str(mode)])

    # Add normalize_kernel flag if True
    if normalize_kernel:
        cmd.append("-normalize-kernel")

    # Add apply_edge_mask flag if True
    if apply_edge_mask:
        cmd.append("-apply-edge-mask")

    # Add method if provided
    if method:
        cmd.extend(["-method", method])

    # Add delta_peeing if provided (handle both list and single value)
    if delta_peeing is not None:
        if isinstance(delta_peeing, (list, tuple)):
            cmd.extend(["-delta-peeing"] + [str(d) for d in delta_peeing])
        else:
            cmd.extend(["-delta-peeing", str(delta_peeing)])

    # Add device
    cmd.extend(["-device", str(device)])

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Error combining images: {result.stderr}")

    return None


import time

from astropy.convolution import Gaussian2DKernel


def convolve_fft_cpu(
    images,
    output,
    kernels=None,
    mode="same",
    normalize_kernel=False,
    device=None,
    apply_edge_mask=False,
    method=None,
    delta_peeing=None,
):
    h, w = fits.getdata(images[0]).shape
    cpu_buffer = np.empty((h, w))
    st = time.time()

    for i, (image, kernel) in enumerate(zip(images, kernels)):

        if method == "gaussian":
            sigma = kernel / (np.sqrt(8 * np.log(2)))
            kernel = Gaussian2DKernel(x_stddev=sigma).array.astype(np.float32, copy=False)
        cpu_buffer = fits.getdata(image)
        cpu_buffer_conv = convolve_fft_with_astropy(cpu_buffer, kernel, normalize_kernel=normalize_kernel)
        if apply_edge_mask:
            edge_mask = get_edge_mask_cpu(cpu_buffer_conv, kernel)
            cpu_buffer_conv[~edge_mask] = 0

        header = add_conv_header(fits.getheader(image), delta_peeing[i], method)
        fits.writeto(
            output[i],
            data=cpu_buffer_conv.copy(),
            header=header,
            overwrite=True,
        )


def convolve_fft_with_astropy(image, kernel, normalize_kernel=False):
    from astropy.convolution import convolve_fft

    return convolve_fft(
        image,
        kernel,
        normalize_kernel=normalize_kernel,
        nan_treatment="fill",  # cheaper than 'interpolate' if you can allow it
        boundary="fill",  # try 'wrap' if physically valid; can be faster
        fill_value=0.0,
    )


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
    header["CONVSIZE"] = (delta_peeing, "Convolution kernel FWHM in pixels")
    return header
