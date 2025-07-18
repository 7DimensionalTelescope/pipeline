import argparse
from astropy.io import fits
import numpy as np
import cupy as cp
import os

def interpolate_masked_pixels(images, mask, window=1, method=None, badpix=None, output_paths=None, weight: bool = True, device=None):
    data = []
    data_weight = []

    for sci_path, wgt_path in images:
        data.append(fits.getdata(sci_path).astype(np.float32))
        if weight:
            data_weight.append(fits.getdata(wgt_path).astype(np.float32))

    with cp.cuda.Device(device):

        cmask = cp.asarray(mask)

        for i, subdata in enumerate(data):
            
            cdata = cp.asarray(subdata, dtype=cp.float32)
            
            if weight:
                cdata_weight = cp.asarray(data_weight[i], dtype=cp.float32)
            else:
                cdata_weight = None
            
            cdata, cdata_weight = interpolate_masked_pixels_gpu_vectorized_weight(
                cdata, cmask, window=window, method=method, badpix=badpix, weight=cdata_weight, device=device)

            data[i][:] = cp.asnumpy(cdata[i])

            if weight:
                data_weight[i][:] = cp.asnumpy(cdata_weight[i])

        # cleanup
        del cdata, cdata_weight, cmask
        cp.get_default_memory_pool().free_all_blocks()

    # save the result
    for idx, (sci_out, wgt_out) in enumerate(output_paths):
        # write science
        fits.writeto(
            sci_out,
            data=data[idx],
            header=add_bpx_method(fits.getheader(images[idx][0]), method),
            overwrite=True,
        )
        # write weight if applicable
        if weight:
            fits.writeto(
                wgt_out,
                data=data_weight[idx],
                header=add_bpx_method(fits.getheader(images[idx][1]), method),
                overwrite=True,
            )



def interpolate_masked_pixels_gpu_vectorized_weight(
    image,
    mask,
    weight=None,
    window=1,
    method="median",
    badpix=1,
    device=0
):
    """Interpolate masked pixels in image (and optional weight map) using a mean filter.

    Args:
        image (cp.ndarray): 2D input image.
        mask (cp.ndarray): Binary mask (1 = masked, 0 = valid) when badpix=1.
        weight (cp.ndarray, optional): Pixel weight map. If None, unweighted mean is used. It must be 1/VARIANCE.
        window (int): Radius of square window.
        method (str): Interpolation method. Options are "inverse_variance" or "median".

    Returns:
        interpolated_image (cp.ndarray): Image with interpolated masked pixels.
        interpolated_weight (cp.ndarray): Weight map with interpolated weights.
    """

    with cp.cuda.Device(device):
        H, W = image.shape
        assert image.shape == mask.shape
        if weight is not None:
            assert weight.shape == image.shape

        result = image.copy()
        weight_result = weight.copy() if weight is not None else None  # cp.ones_like(image)

        ys, xs = cp.where(mask == badpix)
        N = ys.shape[0]
        if N == 0:
            return result, weight_result

        # Create patch offsets
        dy, dx = cp.meshgrid(cp.arange(-window, window + 1), cp.arange(-window, window + 1), indexing="ij")
        dy = dy.ravel()
        dx = dx.ravel()

        patch_ys = ys[:, None] + dy[None, :]
        patch_xs = xs[:, None] + dx[None, :]

        # Mask out-of-bound locations
        in_bounds = (patch_ys >= 0) & (patch_ys < H) & (patch_xs >= 0) & (patch_xs < W)

        # Clip for safe indexing
        patch_ys_safe = cp.clip(patch_ys, 0, H - 1)
        patch_xs_safe = cp.clip(patch_xs, 0, W - 1)
        flat_indices = patch_ys_safe * W + patch_xs_safe

        flat_image = image.ravel()
        flat_mask = mask.ravel()
        patch_vals = flat_image[flat_indices]
        patch_mask = flat_mask[flat_indices]

        valid = (patch_mask == 1 - badpix) & in_bounds

        if weight is not None:
            flat_weight = weight.ravel()
            patch_weights = flat_weight[flat_indices]

            patch_weights = cp.where(valid, patch_weights, 0)
            patch_vals = cp.where(valid, patch_vals, cp.nan)

            if method == "inverse_variance":
                weighted_sum = cp.sum(patch_weights * patch_vals, axis=1)
                weight_total = cp.sum(patch_weights, axis=1)

                # if weight_total == 0, val is 0
                interp_vals = cp.where(weight_total > 0, weighted_sum / weight_total, 0)
                interp_weights = weight_total  # assuming weight = 1/var

            elif method == "median":
                # Compute median and MAD
                interp_vals = cp.nanmedian(patch_vals, axis=1)

                # Handle NaNs safely by using cp.isclose (for float stability)
                mask = cp.isclose(patch_vals, interp_vals[:, None], equal_nan=True)

                # For multiple matches per row, take first match
                def first_match_per_row(data, mask):
                    idx = cp.argmax(mask, axis=1)  # index of first True in each row
                    row_idx = cp.arange(mask.shape[0])
                    return data[row_idx, idx]

                interp_weights = first_match_per_row(patch_weights, mask)

            # Fill interpolated values and weights
            result[ys, xs] = interp_vals
            weight_result[ys, xs] = interp_weights

            return result, weight_result

        else:
            patch_vals = cp.where(valid, patch_vals, cp.nan)
            interp_vals = cp.nanmedian(patch_vals, axis=1)

            result[ys, xs] = interp_vals
            return result, None


def add_bpx_method(header, method):
    header["INTERP"] = (method.upper(), "Method for bad pixel interpolation")
    return header

def add_suffix(filename: str | list[str], suffix):
    if isinstance(filename, list):
        return [add_suffix(f, suffix) for f in filename]
    base, ext = os.path.splitext(filename)
    suffix = suffix if suffix.startswith("_") else f"_{suffix}"
    return f"{base}{suffix}{ext}"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Interpolate masked pixels in FITS images (and optional weight maps) using CuPy GPU acceleration."
    )
    parser.add_argument(
        '-input', 
        nargs='+',
        required=True,
        help='Paths to input science FITS files'
    )

    parser.add_argument(
        '-mask',
        required=True,
        help='Path to binary mask FITS file'
    )
    
    parser.add_argument(
        '-output',
        nargs='+',
        required=True,
        help='Paths for output science FITS files'
    )

    parser.add_argument(
        '-window',
        type=int,
        default=1,
        help='Radius of square window for interpolation'
    )
    parser.add_argument(
        '-method',
        choices=['inverse_variance', 'median'],
        default='median',
        help='Interpolation method'
    )
    parser.add_argument(
        '-badpix',
        type=int,
        default=1,
        help='Value in mask indicating bad pixels'
    )

    parser.add_argument(
        '-device',
        type=int,
        default=0,
        help='GPU device ID'
    )
    parser.add_argument(
        '-no-weight',
        dest='weight',
        action='store_false',
        help='Disable weight map processing'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # build tuples
    if args.weight:
        images = list(zip(args.input, add_suffix(args.input, "weight")))
        output_paths = list(zip(args.output, add_suffix(args.output, "weight")))
    else:
        images = [(sci, None) for sci in args.input]
        output_paths = [(out, None) for out in args.output]

    # load mask
    mask = fits.getdata(args.mask)

    # call main function
    interpolate_masked_pixels(
        images,
        mask,
        window=args.window,
        method=args.method,
        badpix=args.badpix,
        output_paths=output_paths,
        weight=args.weight,
        device=args.device
    )

 