import numpy as np
import cupy as cp

def convolve_fft(image, kernel, mode="same", normalize_kernel=False, device_id=0):
    if device_id is "CPU":
        return convolve_fft_cpu(image, kernel, normalize_kernel)
    else:
        return convolve_fft_gpu(image, kernel, mode, normalize_kernel, device_id)

def convolve_fft_gpu(image, kernel, mode="same", normalize_kernel=False, device_id=0):

    from cupyx.scipy.signal import fftconvolve

    with cp.cuda.Device(device_id):
        image = cp.asarray(image, dtype=cp.float64)
        kernel_array = cp.asarray(kernel.array, dtype=cp.float64)
        if normalize_kernel:
            kernel_array = kernel_array / cp.sum(kernel_array)

        image = cp.nan_to_num(image, nan=0.0)
        result = fftconvolve(image, kernel_array, mode=mode)
        result = cp.asnumpy(result)
        del image, kernel_array
        cp.get_default_memory_pool().free_all_blocks()

        return result

def convolve_fft_cpu(image, kernel, normalize_kernel=False):
    from astropy.convolution import convolve_fft
    return convolve_fft(image, kernel, normalize_kernel=normalize_kernel)

def get_edge_mask(weight_image, kernel, device_id=None):
    if device_id is "CPU":
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

def get_edge_mask_gpu(weight_image, kernel, device_id=None):
    from cupyx.scipy.ndimage import binary_dilation

    with cp.cuda.Device(device_id):
        mask = cp.asarray(weight_image != 0)  # 0 for edge, 1 for inside

        size = cp.shape(kernel)

        struct = cp.ones(size, dtype=bool)
        dilated_mask = binary_dilation(mask, structure=struct)
        dilated_mask = cp.asnumpy(dilated_mask)
        del mask, struct
        cp.get_default_memory_pool().free_all_blocks()
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
