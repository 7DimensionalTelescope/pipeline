import argparse
import numpy as np
import cupy as cp
from cupyx.scipy.signal import fftconvolve
from cupyx.scipy.ndimage import binary_dilation
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel


def convolve_fft(
    images,
    outputs,
    kernels=None,
    mode="same",
    normalize_kernel=False,
    apply_edge_mask=False,
    method=None,
    delta_peeing=None,
    device=0,
):

    # Validate lengths
    n = len(images)

    for i in range(n):
        img_file = images[i]
        out_file = outputs[i]
        delta = delta_peeing[i] if delta_peeing is not None else None

        if kernels is not None:
            kern_file = kernels[i]
        else:
            kern_file = None

        data = fits.getdata(img_file).astype(np.float32)

        if kern_file:
            kernel_np = fits.getdata(kern_file).astype(np.float32)
        else:
            kernel_np = Gaussian2DKernel(x_stddev=delta / (np.sqrt(8 * np.log(2)))).array.astype(np.float32)

        with cp.cuda.Device(device):
            # Load data on host

            cdata = cp.asarray(data)
            kernel_array = cp.asarray(kernel_np)
            if normalize_kernel:
                kernel_array = kernel_array / cp.sum(kernel_array)

            # replace NaNs
            cp.nan_to_num(cdata, copy=False, nan=0.0)

            # perform convolution
            cdata[:] = fftconvolve(cdata, kernel_array, mode=mode)

            # optionally mask edges
            if apply_edge_mask:
                mask = cdata != 0
                mask[:] = binary_dilation(mask, structure=cp.ones(kernel_array.shape, dtype=bool))
                cdata[~mask] = 0

            # bring back to host
            data[:] = cp.asnumpy(cdata)

        # prepare header
        header = add_conv_header(fits.getheader(img_file), delta, method)
        # write result
        fits.writeto(out_file, data=data, header=header, overwrite=True)


def add_conv_header(header, delta_peeing, method):
    """
    Add convolution metadata to FITS header.
    """
    if method:
        header["CONV"] = (method.upper(), "Method for seeing-match convolution")
    if delta_peeing is not None:
        header["CONVSIZE"] = (delta_peeing, "Convolution kernel FWHM in pixels")
    return header


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convolve FITS images with kernels using GPU FFT")
    parser.add_argument("-input", nargs="+", required=True, help="Input FITS image file paths")
    parser.add_argument("-output", nargs="+", required=True, help="Output FITS file paths")
    parser.add_argument("-kernels", nargs="+", help="Input FITS kernel file paths")
    parser.add_argument(
        "-mode", default="same", choices=["full", "same", "valid"], help="Convolution mode (default: same)"
    )
    parser.add_argument(
        "-normalize-kernel", action="store_true", help="Normalize kernel to unit sum before convolution"
    )
    parser.add_argument("-apply-edge-mask", action="store_true", help="Zero out edge artifacts after convolution")
    parser.add_argument("-method", type=str, help="Label for convolution method to add to header")
    parser.add_argument("-delta-peeing", nargs="+", type=float, help="Kernel FWHM values (pixels) for header")
    parser.add_argument("-device", type=int, default=0, help="GPU device ID (default: 0)")
    args = parser.parse_args()
    print(args)
    convolve_fft(
        args.input,
        args.output,
        kernels=args.kernels,
        mode=args.mode,
        normalize_kernel=args.normalize_kernel,
        apply_edge_mask=args.apply_edge_mask,
        method=args.method,
        delta_peeing=args.delta_peeing,
        device=args.device,
    )
