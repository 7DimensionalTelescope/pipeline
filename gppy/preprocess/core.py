import cupy as cp

# Define a 4-input, 1-output elementwise kernel
reduce_kernel = cp.ElementwiseKernel(
    in_params="T sci, T bias, T dark, T flat",
    out_params="T out",
    # arithmetic fused into one expression
    operation="out = (sci - bias - dark) / flat;",
    name="sci_reduction",
)

# Later, apply it to your stacked data:
corrected_stack = reduce_kernel(
    sci_stack, mbias[None, ...], mdark[None, ...], mflat[None, ...]  # shape (N, H, W)  # broadcast to (N, H, W)
)


import numpy as np
import cupy as cp
from cupy.cuda import Stream, alloc_pinned_memory


def async_load(filenames, batch_size=16):
    """Yields batches of raw data already on the GPU, with loading overlapped."""
    # Pre-allocate pinned host buffer and GPU buffer for one batch
    # assume all frames same shape (H, W) and dtype
    H, W = ...
    dtype = np.float32
    pinned_bytes = alloc_pinned_memory(batch_size * H * W * np.dtype(dtype).itemsize)
    pinned_buf = np.frombuffer(pinned_bytes, dtype=dtype).reshape(batch_size, H, W)
    gpu_buf = cp.empty((batch_size, H, W), dtype=dtype)

    load_stream = Stream(non_blocking=True)
    compute_stream = Stream()  # default stream for reduction

    # Prime first upload
    for i, fname in enumerate(filenames[:batch_size]):
        pinned_buf[i] = load_raw_cpu(fname)  # your CPU loader
    with load_stream:
        cp.cuda.runtime.memcpyAsync(
            gpu_buf.data.ptr,
            pinned_bytes.ptr,
            pinned_bytes.nbytes,
            cp.cuda.runtime.cudaMemcpyHostToDevice,
            load_stream.ptr,
        )
    load_stream.synchronize()  # ensure first batch on GPU

    # Now slide over the rest
    idx = batch_size
    while True:
        # launch compute on gpu_buf in compute_stream
        with compute_stream:
            batch_out = reduce_kernel(gpu_buf, mbias[None], mdark[None], mflat[None])
        # Kick off next load if any remain
        if idx < len(filenames):
            # refill pinned_buf with raw from CPU
            for j in range(batch_size):
                if idx + j < len(filenames):
                    pinned_buf[j] = load_raw_cpu(filenames[idx + j])
                else:
                    break
            with load_stream:
                cp.cuda.runtime.memcpyAsync(
                    gpu_buf.data.ptr,
                    pinned_bytes.ptr,
                    pinned_bytes.nbytes,
                    cp.cuda.runtime.cudaMemcpyHostToDevice,
                    load_stream.ptr,
                )
        # wait for compute to finish, yield result
        compute_stream.synchronize()
        yield batch_out.get()
        if idx >= len(filenames):
            break
        # swap streams (so compute_stream picks up next batch once copied)
        load_stream, compute_stream = compute_stream, load_stream
        idx += batch_size
