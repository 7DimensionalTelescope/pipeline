from astropy.io import fits
import argparse
import cupy as cp
import numpy as np
import gc

# Reduction kernel
reduction_kernel = cp.ElementwiseKernel(
    in_params="T x, T b, T d, T f", out_params="T z", operation="z = (x - b - d) / f", name="reduction"
)

def process_image_with_cupy(obs, bias, dark, flat, output, device_id):
    """median is GPU, std is CPU. Uses pinned memory for better host-GPU transfer performance."""

    with cp.cuda.Device(device_id):
        data = [fits.getdata(o).astype(np.float32) for o in obs]

        cbias = cp.asarray(fits.getdata(bias), dtype=cp.float32)
        cdark = cp.asarray(fits.getdata(dark), dtype=cp.float32)
        cflat = cp.asarray(fits.getdata(flat), dtype=cp.float32)

        cdata = cp.asarray(data)
        cdata = cdata.astype(cp.float32)

        cdata = reduction_kernel(cdata, cbias, cdark, cflat)
        for i in range(len(data)):
            data[i][:] = cp.asnumpy(cdata[i])

        del cbias, cdark, cflat, cdata
        cp.get_default_memory_pool().free_all_blocks()

    for i, o in enumerate(output):
        fits.writeto(o, data[i], overwrite=True)
    del data
    gc.collect()
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process FITS images."
    )

    parser.add_argument("-bias", type=str, required=True, help="BIAS FITS image path.")
    parser.add_argument("-dark", type=str, required=True, help="DARK FITS image path.")
    parser.add_argument("-flat", type=str, required=True,  help="FLAT image path.")
    parser.add_argument("-input", nargs="+", required=True, help="Input FITS image paths.")
    parser.add_argument("-output", nargs="+", required=True, help="Output FITS image paths.")
    parser.add_argument("-device", type=int, default=0, help="CUDA device ID.")

    args = parser.parse_args()

    process_image_with_cupy(
        args.input,
        args.bias,
        args.dark, 
        args.flat, 
        args.output,
        args.device,
    )
