import numpy as np
import cupy as cp
from cupyx.scipy.signal import fftconvolve


def convolve_fft_gpu(image, kernel, mode="same", normalize_kernel=False, device_id=0):
    """
    Perform FFT convolution on the GPU using cupyx.scipy.signal.fftconvolve,
    as an alternative to astropy.convolution.convolve_fft.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.
    kernel : Kernel object (e.g., astropy.convolution.Gaussian2DKernel)
        Convolution kernel; its values are accessed via kernel.array.
        Its dimensions must be odd to prevent shifting of the convolved image.
    mode : str, optional
        Convolution mode: 'full', 'same', or 'valid'. Default is 'same'.
    normalize_kernel : bool, optional
        If True, the kernel is normalized so that its sum equals 1.

    Returns
    -------
    result : numpy.ndarray
        Convolved image as a NumPy array.
    """
    with cp.cuda.Device(device_id):
        # Convert image to float64 and transfer to GPU.
        image = cp.asarray(image, dtype=cp.float64)

        # Get kernel array and transfer to GPU. It must be ODD!
        kernel_array = cp.asarray(kernel.array, dtype=cp.float64)
        if normalize_kernel:
            kernel_array = kernel_array / cp.sum(kernel_array)

        # Optionally, handle any NaN values in the image.
        image = cp.nan_to_num(image, nan=0.0)

        # Perform FFT convolution on the GPU.
        result = fftconvolve(image, kernel_array, mode=mode)

        # Transfer the result back to the CPU.
        return cp.asnumpy(result)


def get_edge_mask(weight_image, kernel):
    from scipy.ndimage import binary_dilation

    mask = weight_image != 0  # 0 for edge, 1 for inside

    size = np.shape(kernel)

    struct = np.ones(size, dtype=bool)
    dilated_mask = binary_dilation(mask, structure=struct)
    return dilated_mask  # mask for inside


def add_conv_method(header, delta_peeing, method):
    header["CONV"] = (method.upper(), "Method for seeing-match convolution")
    header['CONVSIZE"'] = (delta_peeing, "Convolution kernel FWHM in pixels")
    return header


# Example
if __name__ == "__main__":
    from astropy.convolution import Gaussian2DKernel

    im = np.zeros((300, 300))
    conv_fwhm = 5  # fwhm as [pix] unit
    gauss_kernel = Gaussian2DKernel(x_stddev=conv_fwhm / (np.sqrt(8 * np.log(2))))
    convolved_im = convolve_fft_gpu(im, gauss_kernel)
