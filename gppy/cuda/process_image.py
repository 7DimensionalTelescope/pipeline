from astropy.io import fits
import argparse
import cupy as cp
import numpy as np

# Reduction kernel
reduction_kernel = cp.ElementwiseKernel(
    in_params="T x, T b, T d, T f", out_params="T z", operation="z = (x - b - d) / f", name="reduction"
)

def process_image_with_cupy(
    obs, bias, dark, flat, output, device_id, **kwargs
):
    """median is GPU, std is CPU. Uses pinned memory for better host-GPU transfer performance."""

    with cp.cuda.Device(device_id):
        data = [fits.getdata(o) for o in obs]

        cbias = cp.asarray(fits.getdata(bias))
        cdark = cp.asarray(fits.getdata(dark))
        cflat = cp.asarray(fits.getdata(flat))

        cdata = cp.asarray(data)
        cdata = cdata.astype(cp.float32)

        cdata = reduction_kernel(cdata, cbias, cdark, cflat)

        for i in range(len(data)):
            data[i][:] = cp.asnumpy(cdata[i])

        del cbias, cdark, cflat, cdata
        cp.get_default_memory_pool().free_all_blocks()

    for o, d in zip(output, data):
        header = o.replace(".fits", ".header")
        fits.writeto(o, d, header=header, overwrite=True)

    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process FITS images with optional subtraction, normalization, and BPMask creation using CUDA."
    )

    parser.add_argument("-input", nargs="+", required=True, help="Input FITS image paths.")
    parser.add_argument("-bias", required=True, help="Output FITS for median image.")
    parser.add_argument("-dark", required=True, help="Output FITS for stddev image.")
    parser.add_argument("-flat", nargs="*", default=[], help="Subtract image paths.")
    parser.add_argument("-output", nargs="*", type=float, default=[], help="Scale factors for subtract images.")
    parser.add_argument("-device", action="store_true", help="Apply per-image normalization.")

    args = parser.parse_args()

    process_image_with_cupy(
        args.input,
        args.bias,
        args.dark, 
        args.flat, 
        args.output,
        device_id=args.device,
    )
