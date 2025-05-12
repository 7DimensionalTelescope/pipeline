from pathlib import Path
from astropy.io import fits

keys = ("EXPTIME", "GAIN", "XBINNING")
groups: dict[tuple, list[str]] = {}

for f in Path(".").glob("**/*.fits"):
    with fits.open(f, memmap=False) as hdul:
        hdr = hdul[0].header
    key = tuple(hdr.get(k) for k in keys)
    groups.setdefault(key, []).append(str(f))


import glob
import threading
import numpy as np
import cupy as cp
from cupy.cuda import Stream
from astropy.io import fits

image_list = glob.glob(f"{mfg.path_raw}/*{filt}*{exposure}*.fits")

# Reduction kernel
reduction_kernel = cp.ElementwiseKernel(
    in_params='T x, T b, T d, T f',
    out_params='T z',
    operation='z = (x - b - d) / f',
    name='reduction'
)

def load_image(filename):
    if filename.endswith(".link"):
        filename = read_link(filename)
    return fits.getdata(filename).astype(np.float32)

def estimate_posssible_batch_size(filename):
    H, W = load_image(filename).shape
    image_size = cp.float32().nbytes * H * W / 1024**2
    available_mem = cp.cuda.runtime.memGetInfo()[0] / 1024**2
    safe_mem = int(available_mem * 0.7)
    batch_size = max(1, safe_mem // image_size)
    return int(batch_size), image_size, available_mem

def process_batch_on_device(device_id, image_paths, bias_cpu, dark_cpu, flat_cpu, results):
    with cp.cuda.Device(device_id):
        load_stream = Stream(non_blocking=True)
        compute_stream = Stream()

        # Transfer bias/dark/flat to this GPU once
        with load_stream:
            bias = cp.asarray(bias_cpu, dtype=cp.float32)
            dark = cp.asarray(dark_cpu, dtype=cp.float32)
            flat = cp.asarray(flat_cpu, dtype=cp.float32)

        load_stream.synchronize()

        local_results = []
        for img_path in image_paths:
            with load_stream:
                image = cp.asarray(load_image(img_path), dtype=cp.float32)
            with compute_stream:
                reduced = reduction_kernel(image, bias, dark, flat)
                local_results.append(cp.asnumpy(reduced))

        compute_stream.synchronize()
        results[device_id] = local_results

# Setup per-device batch size
num_devices = cp.cuda.runtime.getDeviceCount()
batch_size = []
for i in range(num_devices):
    with cp.cuda.Device(i):
        batch_size.append(estimate_posssible_batch_size(image_list[0])[0])

# Distribute work
ratio = [b / sum(batch_size) for b in batch_size]
batch_dist = np.floor(np.array(ratio) * len(image_list)).astype(int)
if sum(batch_dist) < len(image_list):
    batch_dist[-1] += len(image_list) - sum(batch_dist)

# Bias, dark, flat (on CPU, assumed loaded)
bias_cpu = load_image(mfg.mbias_link)
dark_cpu = load_image(mfg.mdark_link[exposure])
flat_cpu = load_image(mfg.mflat_link[filt])

# Launch threads for each device
threads = []
results = [None] * num_devices
start_idx = 0

for i in range(num_devices):
    end_idx = start_idx + batch_dist[i]
    subset = image_list[start_idx:end_idx]
    t = threading.Thread(
        target=process_batch_on_device,
        args=(i, subset, bias_cpu, dark_cpu, flat_cpu, results)
    )
    t.start()
    threads.append(t)
    start_idx = end_idx

# Wait for all threads
for t in threads:
    t.join()

# Combine results
all_results = [item for sublist in results if sublist for item in sublist]

