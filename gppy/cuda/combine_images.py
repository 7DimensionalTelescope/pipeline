from astropy.io import fits
import numpy as np
import argparse
import sys


def combine_images_with_cupy(
    data, device_id=None, subtract=None, norm=False, make_bpmask=False, maxiters=5, sigma=3, bpmask_sigma=5, **kwargs
):
    """median is gpu, std is cpu"""
    import cupy as cp
    
    arr_shape = data[0].shape
    np_median = np.empty(arr_shape)
    np_std = np.empty(arr_shape)
    np_bpmask = np.empty(arr_shape)

    with cp.cuda.Device(device_id):

        cp_stack = cp.asarray(data, dtype=cp.float32)
        if subtract is not None:
            cp_subtract = cp.asarray(subtract, dtype=cp.float32)
            cp_subtract = cp.sum(cp_subtract, axis=0)
            cp_stack -= cp_subtract
            del cp_subtract
        if norm:
            cp_stack /= cp.median(cp_stack, axis=(1, 2), keepdims=True)
        cp_median = cp.median(cp_stack, axis=0)
        cp_std = cp.std(cp_stack, axis=0, ddof=1)

        np_std[:] = cp.asnumpy(cp_std)
        np_median[:] = cp.asnumpy(cp_median)

        if make_bpmask:
            cp_data_flat = cp_median.ravel()
            for _ in range(int(maxiters)):
                median_val = cp.median(cp_data_flat)
                std_val = cp.std(cp_data_flat)
                # Keep only pixels within +/- sigma * std of the median
                mask = cp.abs(cp_data_flat - median_val) < (sigma * std_val)
                cp_data_flat = cp_data_flat[mask]
                del mask

            # Final statistics on the clipped data
            median_val = cp.median(cp_data_flat)
            std_val = cp.std(cp_data_flat)

            cp_bpmask = cp.abs(cp_median - median_val) > bpmask_sigma * std_val  # 1 for bad, 0 for okay
            cp_bpmask = cp_bpmask.astype(cp.uint8)  # Convert to uint8
            np_bpmask[:] = cp.asnumpy(cp_bpmask).astype(np.uint8)
            del cp_data_flat, cp_bpmask, median_val, std_val
        else:
            np_bpmask = None

        del cp_stack, cp_median, cp_std
        cp.get_default_memory_pool().free_all_blocks()

    return np_median, np_std, np_bpmask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process FITS images with optional subtraction, normalization, and BPMask creation using CUDA."
    )

    parser.add_argument("-input", nargs="+", required=True, help="Input FITS image paths.")
    parser.add_argument("-median_out", required=True, help="Output FITS for median image.")
    parser.add_argument("-std_out", required=True, help="Output FITS for stddev image.")
    parser.add_argument("-subtract", nargs="*", default=[], help="Subtract image paths.")
    parser.add_argument("-scales", nargs="*", type=float, default=[], help="Scale factors for subtract images.")
    parser.add_argument("-norm", action="store_true", help="Apply per-image normalization.")
    parser.add_argument("-bpmask", help="Output FITS file for BPMask.")
    parser.add_argument("-bpmask_sigma", type=float, default=5.0, help="Sigma threshold for BPMask.")
    parser.add_argument("-device", type=int, default=0, help="CUDA device ID.")

    args = parser.parse_args()

    # Validate logic
    if args.subtract:
        if len(args.subtract) != len(args.scales):
            print("Error: Number of subtract images and scale factors must match.", file=sys.stderr)
            sys.exit(1)
        else:
            subtract = [fits.getdata(o) * args.scales[i] for i, o in enumerate(args.subtract)]
    else:
        subtract = None

    if args.bpmask is not None:
        make_bpmask = True
    else:
        make_bpmask = False

    data = [fits.getdata(o) for o in args.input]
    np_median, np_std, np_bpmask = combine_images_with_cupy(
        data,
        device_id=args.device,
        subtract=subtract,
        normalize=args.norm,
        make_bpmask=make_bpmask,
        bpmask_sigma=args.bpmask_sigma,
    )

    fits.writeto(args.median_out, np_median, overwrite=True)
    fits.writeto(args.std_out, np_median, overwrite=True)

    if make_bpmask:
        fits.writeto(args.bpmask, np_bpmask, overwrite=True)
