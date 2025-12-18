from astropy.io import fits
import argparse
import cupy as cp
import numpy as np
import gc
import os
import math

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time

try:
    import fitsio

    FITSIO_AVAILABLE = True
except ImportError:
    FITSIO_AVAILABLE = False

# Reduction kernel
reduction_kernel = cp.ElementwiseKernel(
    in_params="T x, T s, T m", out_params="T z", operation="z = x * m - s", name="reduction"
)


def process_image_with_cupy(obs, bias, dark, flat, output, device_id):
    """median is GPU, std is CPU. Uses pinned memory for better host-GPU transfer performance."""

    # Write output files in parallel using threading (I/O bound operation)
    def write_one(o, data):
        header_file = o.replace(".fits", ".header")
        header = None
        if os.path.exists(header_file):
            with open(header_file, "r") as f:
                header = fits.Header.fromstring(f.read(), sep="\n")

        if FITSIO_AVAILABLE:
            # Use fitsio for faster writing
            if header is not None:
                # Convert astropy header to fitsio format
                fitsio.write(o, data, header=dict(header), clobber=True)
            else:
                fitsio.write(o, data, clobber=True)
        else:
            # Fallback to astropy
            fits.writeto(o, data, header=header, overwrite=True)

    with cp.cuda.Device(device_id):
        # Load images in parallel using threading (I/O bound operation)

        print("Start prepration: loading masterframes and creating multiplicative and subtractive arrays")
        print("Bias: ", bias)
        print("Dark: ", dark)
        print("Flat: ", flat)
        cbias = cp.asarray(fits.getdata(bias)[None, :, :])
        cbias = cbias.astype(cp.float32)
        cdark = cp.asarray(fits.getdata(dark)[None, :, :])
        cdark = cdark.astype(cp.float32)
        cflat = cp.asarray(fits.getdata(flat)[None, :, :])
        cflat = cflat.astype(cp.float32)

        multiplicative = 1.0 / cflat
        subtractive = (cbias + cdark) * multiplicative

        batch_size = 30
        max_workers = 3
        print(f"Prepration done. Starting batch processing with batch size {batch_size} and max workers {max_workers}")
        total_batches = (len(obs) + batch_size - 1) // batch_size

        cbatch_data = None

        for batch_idx, batch_start in enumerate(tqdm(range(0, len(obs), batch_size), desc="Processing batches"), 1):
            batch_end = min(batch_start + batch_size, len(obs))
            batch_obs = obs[batch_start:batch_end]
            batch_output = output[batch_start:batch_end]
            batch_num_images = len(batch_obs)

            # Load images in parallel using threading
            # Optimize: increase workers for I/O-bound operations, use fitsio for faster reading
            def load_and_convert(path):
                if FITSIO_AVAILABLE:
                    # Use fitsio for faster reading
                    return fitsio.read(path).astype(np.float32)
                else:
                    # Fallback to astropy with memory mapping
                    with fits.open(path, memmap=True) as hdul:
                        return hdul[0].data.astype(np.float32)

            # Timeout for loading: 1 second per image
            load_timeout = batch_num_images
            max_retries = 3

            for retry in range(max_retries):
                try:
                    start_time = time.time()
                    batch_data = []

                    with ThreadPoolExecutor(max_workers=3) as ex:
                        # Submit all tasks
                        future_to_path = {ex.submit(load_and_convert, path): path for path in batch_obs}

                        # Collect results with timeout check during loading
                        from concurrent.futures import as_completed

                        timeout_exceeded = False

                        for future in as_completed(future_to_path):
                            # Check timeout as each future completes
                            elapsed_time = time.time() - start_time
                            if elapsed_time > load_timeout:
                                timeout_exceeded = True
                                print(
                                    f"Loading timeout ({elapsed_time:.2f}s > {load_timeout}s) after {len(batch_data)}/{batch_num_images} images, retrying from beginning (attempt {retry + 1}/{max_retries})"
                                )
                                # Stop collecting results - exit loop immediately
                                # Remaining futures will complete in background but we won't wait
                                break

                            try:
                                result = future.result()
                                batch_data.append(result)
                            except Exception as e:
                                print(f"Error loading image: {e}")
                                # On error, stop and retry
                                batch_data = []
                                break

                        # If timeout exceeded or incomplete, retry
                        if timeout_exceeded or len(batch_data) != batch_num_images:
                            if retry < max_retries - 1:
                                continue  # Retry
                            else:
                                raise RuntimeError(
                                    f"Failed to load all images: got {len(batch_data)}/{batch_num_images} after {max_retries} attempts"
                                )

                    # Success - got all images within timeout
                    break

                except Exception as e:
                    if retry < max_retries - 1:
                        print(f"Error during loading: {e}, retrying from beginning (attempt {retry + 2}/{max_retries})")
                    else:
                        raise  # Re-raise on final retry

            print(
                f"[Batch {batch_idx}/{total_batches}] Transferring {batch_num_images} images to GPU memory (device {device_id})"
            )
            time_start = time.time()
            if cbatch_data is None:
                cbatch_data = cp.asarray(batch_data)
                cbatch_data = cbatch_data.astype(cp.float32)
            else:
                for i in range(batch_num_images):
                    cbatch_data[i] = cp.asarray(batch_data[i])
                    cbatch_data[i] = cbatch_data[i].astype(cp.float32)
            time_end = time.time()
            print(
                f"Time taken to transfer {batch_num_images} images to GPU memory: {time_end - time_start:.2f} seconds"
            )

            print(f"[Batch {batch_idx}/{total_batches}] Applying calibration reduction (bias, dark, flat correction)")
            time_start = time.time()
            cbatch_data = reduction_kernel(cbatch_data, subtractive, multiplicative)
            time_end = time.time()
            print(f"Time taken to apply calibration reduction: {time_end - time_start:.2f} seconds")

            print(f"[Batch {batch_idx}/{total_batches}] Transferring processed data from GPU to host memory")
            time_start = time.time()
            for i in range(batch_num_images):
                batch_data[i] = cp.asnumpy(cbatch_data[i])
            time_end = time.time()
            print(f"Time taken to transfer processed data from GPU to host memory: {time_end - time_start:.2f} seconds")

            print(f"[Batch {batch_idx}/{total_batches}] Writing {batch_num_images} calibrated images to disk")
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                items = list(zip(batch_output, batch_data))
                list(
                    tqdm(
                        ex.map(lambda item: write_one(item[0], item[1]), items),
                        total=len(items),
                        desc="Writing output",
                    )
                )
            print(f"[Batch {batch_idx}/{total_batches}] Batch processing completed")
            

            gc.collect()

        del cbias, cdark, cflat, cbatch_data
        cp.get_default_memory_pool().free_all_blocks()

    print(f"Processing complete: {len(obs)} images processed on GPU device {device_id}")
    print("Releasing GPU memory resources")
    gc.collect()
    print("Memory cleanup completed")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process FITS images.")

    parser.add_argument("-bias", type=str, required=True, help="BIAS FITS image path.")
    parser.add_argument("-dark", type=str, required=True, help="DARK FITS image path.")
    parser.add_argument("-flat", type=str, required=True, help="FLAT image path.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-input", nargs="+", help="Input FITS image paths.")
    input_group.add_argument(
        "-input-list", type=str, help="Text file containing input FITS image paths (one per line)."
    )
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument("-output", nargs="+", help="Output FITS image paths.")
    output_group.add_argument(
        "-output-list", type=str, help="Text file containing output FITS image paths (one per line)."
    )
    parser.add_argument("-device", type=int, default=0, help="CUDA device ID.")

    args = parser.parse_args()

    # Read input paths from file or use command-line arguments
    if args.input_list:
        with open(args.input_list, "r") as f:
            input_paths = [line.strip() for line in f if line.strip()]
    else:
        input_paths = args.input

    # Read output paths from file or use command-line arguments
    if args.output_list:
        with open(args.output_list, "r") as f:
            output_paths = [line.strip() for line in f if line.strip()]
    else:
        output_paths = args.output

    process_image_with_cupy(
        input_paths,
        args.bias,
        args.dark,
        args.flat,
        output_paths,
        args.device,
    )
