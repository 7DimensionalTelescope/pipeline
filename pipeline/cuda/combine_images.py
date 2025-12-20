import argparse
import sys
import cupy as cp
import numpy as np
import fitsio


def pinned_empty(shape, dtype=np.float32):
    size = np.prod(shape)
    nbytes = np.dtype(dtype).itemsize * size
    mem = cp.cuda.alloc_pinned_memory(nbytes)
    arr = np.frombuffer(mem, dtype=dtype, count=size).reshape(shape)
    return arr


def combine_images_with_cupy(
    data, device_id=None, subtract=None, norm=False, make_bpmask=False, maxiters=5, sigma=3, bpmask_sigma=5
):
    """median is GPU, std is CPU. Uses pinned memory for better host-GPU transfer performance."""

    arr_shape = data[0].shape

    with cp.cuda.Device(device_id):
        # Allocate host-pinned memory
        np_median = pinned_empty(arr_shape, dtype=np.float32)
        np_std = pinned_empty(arr_shape, dtype=np.float32)
        np_bpmask = pinned_empty(arr_shape, dtype=np.uint8) if make_bpmask else None

        # GPU computation
        cp_stack = cp.asarray(data)
        cp_stack = cp_stack.astype(cp.float32)

        if subtract is not None:
            cp_subtract = cp.asarray(subtract)
            cp_subtract = cp_subtract.astype(cp.float32)
            cp_stack -= cp.sum(cp_subtract, axis=0)
            del cp_subtract

        if norm:
            cp_stack /= cp.median(cp_stack, axis=(1, 2), keepdims=True)

        cp_median = cp.median(cp_stack, axis=0)  # coadded image. e.g., master dark
        cp_std = cp.std(cp_stack, axis=0, ddof=1)

        # Copy to pinned host memory
        cp.cuda.runtime.memcpyAsync(
            np_median.ctypes.data,
            cp_median.data.ptr,
            cp_median.nbytes,
            cp.cuda.runtime.memcpyDeviceToHost,
            cp.cuda.Stream.null.ptr,
        )
        cp.cuda.runtime.memcpyAsync(
            np_std.ctypes.data,
            cp_std.data.ptr,
            cp_std.nbytes,
            cp.cuda.runtime.memcpyDeviceToHost,
            cp.cuda.Stream.null.ptr,
        )
        cp.cuda.Stream.null.synchronize()

        if make_bpmask:
            cp_data_flat = cp_median.ravel()
            for _ in range(int(maxiters)):
                median_val = cp.median(cp_data_flat)
                std_val = cp.std(cp_data_flat, ddof=1)
                mask = cp.abs(cp_data_flat - median_val) < (sigma * std_val)
                cp_data_flat = cp_data_flat[mask]
                del mask

            median_val = cp.median(cp_data_flat)
            std_val = cp.std(cp_data_flat)
            # print(f"gpu median_val {median_val}, std_val {std_val}")

            cp_bpmask = (cp.abs(cp_median - median_val) > bpmask_sigma * std_val).astype(cp.int8)

            cp.cuda.runtime.memcpyAsync(
                np_bpmask.ctypes.data,
                cp_bpmask.data.ptr,
                cp_bpmask.nbytes,
                cp.cuda.runtime.memcpyDeviceToHost,
                cp.cuda.Stream.null.ptr,
            )
            cp.cuda.Stream.null.synchronize()

            del cp_bpmask, cp_data_flat, median_val, std_val

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
            subtract = [fitsio.read(o).astype(np.float32) * args.scales[i] for i, o in enumerate(args.subtract)]
    else:
        subtract = None

    if args.bpmask is not None:
        make_bpmask = True
    else:
        make_bpmask = False

    data = [fitsio.read(img).astype(np.float32) for img in args.input]

    np_median, np_std, np_bpmask = combine_images_with_cupy(
        data,
        device_id=args.device,
        subtract=subtract,
        norm=args.norm,
        make_bpmask=make_bpmask,
        bpmask_sigma=args.bpmask_sigma,
    )

    # Write output files using fitsio if available
    fitsio.write(args.median_out, np_median, clobber=True)
    fitsio.write(args.std_out, np_std, clobber=True)
    if make_bpmask:
        fitsio.write(args.bpmask, np_bpmask, clobber=True)
