from __future__ import annotations

import gc
import re
import time
import psutil
import threading
from contextlib import contextmanager
from datetime import datetime
from astropy.table import Table
from typing import Optional, Iterator, overload
import fcntl
from contextlib import contextmanager
import pynvml
import os
from collections import UserDict
from itertools import chain

import ctypes

from ..const import DISCOVERY_RAW_INVENTORY, GPU_DEVICES, SERVICES_TMP_DIR
from ..utils import collapse


_cached_cg_path = None  # cache for resolved cgroup path


def _discover_user_slice_dir():
    """
    Return the absolute /sys/fs/cgroup path for the current process.
    Reads /proc/self/cgroup (cgroup v2 unified line) directly.
    """
    uid = os.getuid()

    # 1) Read the actual cgroup path for this process from /proc/self/cgroup
    try:
        with open("/proc/self/cgroup", "r") as f:
            for line in f:
                # cgroup v2 unified hierarchy: "0::/some/path"
                if line.startswith("0::"):
                    rel = line.split("::", 1)[1].strip().lstrip("/")
                    candidate = os.path.join("/sys/fs/cgroup", rel)
                    if os.path.isdir(candidate):
                        return candidate
                    break
    except FileNotFoundError:
        pass

    # 2) Fallback: UID-derived user slice
    candidate = f"/sys/fs/cgroup/user.slice/user-{uid}.slice"
    if os.path.isdir(candidate):
        return candidate

    raise RuntimeError(
        f"Could not locate cgroup directory for UID {uid}. "
        f"If running in a container or non-systemd env, pass cg_path explicitly."
    )


def read_cgroup_mem(cg_path="auto"):
    """
    Return cgroup-v2 memory usage for a slice/cgroup.
    If cg_path == 'auto' or None, resolve the current account's user slice automatically:
      /sys/fs/cgroup/user.slice/user-<UID>.slice
    You can still pass a custom relative path like 'user.slice/user-10019.slice'
    or an absolute path under /sys/fs/cgroup.
    """
    global _cached_cg_path

    # Resolve and cache the cgroup directory
    if _cached_cg_path is None:
        if cg_path in (None, "auto"):
            cg_dir = _discover_user_slice_dir()
        else:
            cg_dir = cg_path
            if not cg_dir.startswith("/"):
                cg_dir = os.path.join("/sys/fs/cgroup", cg_dir.lstrip("/"))
        _cached_cg_path = cg_dir

    cg_dir = _cached_cg_path

    # Read memory.current
    with open(os.path.join(cg_dir, "memory.current"), "r") as f:
        mem_current = int(f.read().strip())

    # Read memory.max (may be "max")
    with open(os.path.join(cg_dir, "memory.max"), "r") as f:
        mem_max_raw = f.read().strip()
    mem_max = None if mem_max_raw == "max" else int(mem_max_raw)

    return {
        "bytes_current": mem_current,
        "bytes_max": mem_max,  # None == unlimited at this slice
        "gb_current": mem_current / (1024**3),
        "gb_max": None if mem_max is None else mem_max / (1024**3),
        "percent_of_cap": None if mem_max is None else (mem_current / mem_max) * 100.0,
        "path": cg_dir,
    }


def cleanup_memory() -> None:
    """
    Perform comprehensive memory cleanup across CPU and GPU.

    This function provides a robust mechanism for releasing
    unused memory resources in both CPU and GPU contexts.

    Key Operations:
    - Trigger Python garbage collection
    - Free CuPy default memory pool blocks
    - Free CuPy pinned memory pool blocks
    - Perform a second garbage collection to ensure complete memory release

    Ideal for:
    - Preventing memory leaks
    - Managing memory in long-running scientific computing tasks
    - Preparing for memory-intensive operations

    Notes:
    - Calls garbage collection twice to ensure thorough cleanup
    - Uses CuPy's memory pool management for GPU memory
    - Minimal performance overhead

    Example:
        >>> # Before starting a memory-intensive task
        >>> cleanup_memory()
    """
    import cupy as cp

    try:
        for device_id in range(cp.cuda.runtime.getDeviceCount()):
            with cp.cuda.Device(device_id):
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
    except:
        pass

    gc.collect()  # Initial garbage collection


@contextmanager
def monitor_memory_usage(
    interval: float = 1.0, logger: Optional = None, add_utilization=True, verbose: bool = False
) -> Table:
    """
    Context manager that monitors and logs memory usage every X seconds
    while running code inside the `with` block.
    Returns an astropy Table containing the usage history after the context ends.

    Parameters
    ----------
    interval : float, optional
        Time interval between measurements in seconds (default: 1.0)
    logger : logging.Logger, optional
        Logger instance to use for logging (default: None)
    add_utilization : bool, optional
        Whether to include GPU utilization data (default: True)
    verbose : bool, optional
        Whether to print/log usage in real-time (default: False)

    Returns
    -------
    astropy.table.Table
        Table containing timestamps and memory usage data

    Example
    -------
    with monitor_memory_usage(interval=2.0) as history:
        run_preprocess()
    history.write('memory_usage.csv', format='csv', overwrite=True)  # Save to file if needed
    """
    from .memory import MemoryMonitor

    # Create column names based on number of GPUs detected
    n_gpus = len(MemoryMonitor.current_gpu_memory_percent)
    column_names = ["time", "cpu_memory"] + [f"gpu{i}_memory" for i in range(n_gpus)]
    dtype = [object, float] + [float] * n_gpus

    if add_utilization:
        column_names += [f"gpu{i}_utilization" for i in range(n_gpus)]
        dtype += [float] * n_gpus

    usage_data = Table(names=column_names, dtype=dtype)

    # Set column descriptions
    usage_data["time"].description = "Measurement timestamp"
    usage_data["cpu_memory"].description = "CPU memory usage (%)"
    for i in range(n_gpus):
        usage_data[f"gpu{i}_memory"].description = f"GPU {i} memory usage (%)"
        if add_utilization:
            usage_data[f"gpu{i}_utilization"].description = f"GPU {i} utilization (%)"

    stop_thread = False

    def logging_thread() -> None:
        """
        Background thread for monitoring memory usage.

        Continuously monitors memory usage at specified intervals
        and logs the data to the usage table.
        """
        while not stop_thread:
            current_time = str(datetime.now())
            cpu_memory = MemoryMonitor.current_memory_percent
            gpu_memories = MemoryMonitor.current_gpu_memory_percent
            row = [current_time, cpu_memory] + gpu_memories

            if add_utilization:
                gpu_utilizations = MemoryMonitor.current_gpu_utilization
                row += gpu_utilizations

            usage_data.add_row(row)
            if verbose:
                usage_str = MemoryMonitor.log_memory_usage
                if logger:
                    logger.info(usage_str)
                else:
                    print(usage_str)

            time.sleep(interval)

    t = threading.Thread(target=logging_thread, daemon=True)
    t.start()

    try:
        yield usage_data
    finally:
        stop_thread = True
        t.join()


@contextmanager
def monitor_io_rate(interval: float = 1.0, logger: Optional = None, verbose: bool = False) -> Table:
    """
    Context manager that monitors disk I/O (read/write rate) every `interval` seconds.
    Returns an astropy Table with a history of rates.

    Parameters
    ----------
    interval : float
        Time interval between measurements (seconds)
    logger : Logger, optional
        Logger to use for logging (optional)
    verbose : bool
        Whether to print/log the I/O rate in real time

    Returns
    -------
    astropy.table.Table
        Table containing timestamps and I/O rates
    """

    io_data = Table(names=["time", "read_kbps", "write_kbps"], dtype=[object, float, float])
    io_data["time"].description = "Measurement timestamp"
    io_data["read_kbps"].description = "Disk read rate (KB/s)"
    io_data["write_kbps"].description = "Disk write rate (KB/s)"

    stop_thread = False

    def logging_thread():
        """
        Background thread for monitoring I/O rates.

        Continuously monitors disk I/O at specified intervals
        and logs the data to the I/O table.
        """
        prev = psutil.disk_io_counters()
        prev_time = time.time()
        while not stop_thread:
            time.sleep(interval)
            curr = psutil.disk_io_counters()
            curr_time = time.time()

            delta_time = curr_time - prev_time
            read_kbps = (curr.read_bytes - prev.read_bytes) / delta_time / 1024
            write_kbps = (curr.write_bytes - prev.write_bytes) / delta_time / 1024

            timestamp = str(datetime.now())
            io_data.add_row([timestamp, read_kbps, write_kbps])

            if verbose:
                msg = f"[{timestamp}] Read: {read_kbps:.2f} KB/s, Write: {write_kbps:.2f} KB/s"
                if logger:
                    logger.info(msg)
                else:
                    print(msg)

            prev, prev_time = curr, curr_time

    t = threading.Thread(target=logging_thread, daemon=True)
    t.start()

    try:
        yield io_data
    finally:
        stop_thread = True
        t.join()


def plot_history(history: Table, filename: Optional[str] = None, keys=None, ax=None, **kwargs) -> None:
    """
    Plot memory usage history.

    Parameters
    ----------
    history : astropy.table.Table
        Table containing memory usage history
    filename : str, optional
        File to save the plot to (default: None)
    keys : list, optional
        Specific columns to plot (default: all except 'time')
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes to plot on (default: None, creates new figure)
    **kwargs : dict
        Additional keyword arguments passed to matplotlib plot function
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    if ax is None:
        fig, ax = plt.subplots()

    if keys is None:
        keys = history.keys()
        keys.remove("time")
    times = pd.to_datetime(history["time"])
    time = (times - times[0]).total_seconds()

    for key in keys:
        ax.plot(time, history[key], label=key, **kwargs)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Usage (%)")
    ax.set_title("Usage History")
    ax.legend()
    return ax


class classmethodproperty:
    """
    A custom decorator that combines class method and property behaviors.

    Allows creating class-level properties that can be accessed
    without instantiating the class, while maintaining the
    flexibility of class methods.

    Typical use cases:
    - Generating computed class-level attributes
    - Providing dynamic class-level information
    - Implementing lazy-loaded class properties

    Attributes:
        func (classmethod): The underlying class method

    Example:
        class Example:
            @classmethodproperty
            def dynamic_property(cls):
                return compute_something_for_class()
    """

    def __init__(self, func):
        """
        Initialize the classmethodproperty decorator.

        Args:
            func (callable): The function to be converted to a class method property
        """
        self.func = classmethod(func)

    def __get__(self, instance, owner):
        """
        Retrieve the value of the class method property.

        Args:
            instance: The instance calling the property (ignored)
            owner: The class on which the property is defined

        Returns:
            The result of calling the class method
        """
        return self.func.__get__(instance, owner)()


def check_gpu_activity(device_id=None, gpu_threshold=None):
    """nvml indices of the GPU_DEVICES candidates cupy sees that pass the utilisation and free-memory limits."""
    return [device["index"] for device in _usable_devices(device_id)]


def conservative_worker_count(n_tasks: int, cap: int = 4) -> int:
    """Worker count that respects whatever else the machine is running.

    Sized from idle cores (cpu - 1min loadavg), halved for headroom, capped: the system
    queue may be processing in the background, and a few long-lived workers beat many
    short-lived ones (thread init and context switching cost more than they return).
    """
    cpu = os.cpu_count() or 1
    try:
        load = os.getloadavg()[0]
    except OSError:
        load = 0.0
    idle = max(1.0, cpu - load)
    return max(1, min(cap, int(idle // 2), n_tasks))


# ---------------------------------------------------------------------------------------------------------------------
# Host GPU policy: the devices .env GPU_DEVICES offers, one chosen per process, CPU fallback on any failure

GPU_MIN_FREE_BYTES = 8 << 30
GPU_MAX_UTILIZATION = 90  # per cent
GPU_RETRY_SECONDS = 30
CU_CTX_SCHED_BLOCKING_SYNC = 4

_gpu_lock = threading.Lock()
_gpu_state = {"resolved": False, "device": None, "nvml_index": None, "reason": None, "retry_at": 0.0}
_nvml_state = {"handles": None, "ordinals": None}
_warm_kernels = set()


def _nvml_handles():
    """pynvml device handles for this process, initialised once; [] when there is no driver."""
    if _nvml_state["handles"] is None:
        _nvml_state["handles"] = []
        try:
            pynvml.nvmlInit()
            _nvml_state["handles"] = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        except Exception:
            pass
    return _nvml_state["handles"]


def _cupy_ordinals() -> dict:
    """cupy device ordinal per PCI (domain, bus, device); CUDA_VISIBLE_DEVICES renumbers them, nvml does not."""
    if _nvml_state["ordinals"] is None:
        _nvml_state["ordinals"] = {}
        try:
            import cupy as cp

            for ordinal in range(cp.cuda.runtime.getDeviceCount()):
                prop = cp.cuda.runtime.getDeviceProperties(ordinal)
                _nvml_state["ordinals"][(prop["pciDomainID"], prop["pciBusID"], prop["pciDeviceID"])] = ordinal
        except Exception:
            pass
    return _nvml_state["ordinals"]


def nvml_devices() -> list[dict]:
    """Per device: nvml index, cupy ordinal (None when cupy cannot see it), memory bytes, utilisation (%), processes."""
    devices = []
    ordinals = _cupy_ordinals()
    for index, handle in enumerate(_nvml_handles()):
        try:
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            processes = len(pynvml.nvmlDeviceGetComputeRunningProcesses(handle))
            pci = pynvml.nvmlDeviceGetPciInfo(handle)
        except Exception:
            continue
        devices.append(
            dict(
                index=index,
                ordinal=ordinals.get((pci.domain, pci.bus, pci.device)),
                free=memory.free,
                used=memory.used,
                total=memory.total,
                utilization=utilization,
                processes=processes,
            )
        )
    return devices


def _usable_devices(device_id=None) -> list[dict]:
    """The GPU_DEVICES candidates cupy can see, below GPU_MAX_UTILIZATION and with GPU_MIN_FREE_BYTES free."""
    if GPU_DEVICES is None:
        return []
    usable = []
    for device in nvml_devices():
        if device["ordinal"] is None or (device_id is not None and device["index"] != device_id):
            continue
        if GPU_DEVICES != "auto" and device["index"] not in GPU_DEVICES:
            continue
        if device["free"] >= GPU_MIN_FREE_BYTES and device["utilization"] <= GPU_MAX_UTILIZATION:
            usable.append(device)
    return usable


def device_free_bytes(index: int) -> int:
    """Free memory of one device (nvml index) as the driver reports it; 0 when it cannot be read."""
    for device in nvml_devices():
        if device["index"] == index:
            return device["free"]
    return 0


def _blocking_sync(ordinal: int) -> bool:
    """Make the primary context block instead of spin on a device wait; must run before cupy creates the context."""
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
        device = ctypes.c_int()
        return (
            cuda.cuInit(0) == 0
            and cuda.cuDeviceGet(ctypes.byref(device), ordinal) == 0
            and cuda.cuDevicePrimaryCtxSetFlags(device, CU_CTX_SCHED_BLOCKING_SYNC) == 0
        )
    except Exception:
        return False


def gpu_disabled_reason() -> str | None:
    """Why this process is on the CPU, or None."""
    return _gpu_state["reason"]


def disable_gpu(reason: str) -> None:
    """Keep this process on the CPU from now on (an OOM or a CUDA error)."""
    with _gpu_lock:
        _gpu_state.update(resolved=True, device=None, reason=reason)


def pause_gpu(reason: str) -> None:
    """Leave the device for GPU_RETRY_SECONDS (a call over its time budget), then choose again."""
    with _gpu_lock:
        _gpu_state.update(resolved=False, device=None, reason=reason, retry_at=time.time() + GPU_RETRY_SECONDS)


def gpu_device() -> int | None:
    """The cupy device this process computes on, chosen from GPU_DEVICES by free memory and utilisation; None = CPU."""
    with _gpu_lock:
        if _gpu_state["resolved"]:
            return _gpu_state["device"]
        if time.time() < _gpu_state["retry_at"]:
            return None
        if GPU_DEVICES is None:
            _gpu_state.update(resolved=True, reason="GPU_DEVICES unset")
            return None
        usable = _usable_devices()
        if not usable:
            # a busy or full device is sampled again later; a missing driver or cupy is final
            final = not nvml_devices() or not _cupy_ordinals()
            _gpu_state.update(resolved=final, retry_at=time.time() + GPU_RETRY_SECONDS)
            seen = f"nvml sees {len(nvml_devices())}, cupy {len(_cupy_ordinals())}"
            _gpu_state["reason"] = f"no usable device (GPU_DEVICES={GPU_DEVICES!r}; {seen})"
            return None
        best = min(usable, key=lambda d: (d["utilization"], d["processes"], -d["free"]))
        _blocking_sync(best["ordinal"])
        _gpu_state.update(resolved=True, device=best["ordinal"], nvml_index=best["index"], reason=None)
        return best["ordinal"]


@contextmanager
def _device_slot(index: int, slots: int | None):
    """One of `slots` advisory flock slots of a device, shared by every user of the host; None when all are taken."""
    if not slots:
        yield True
        return
    handle = None
    try:
        lock_dir = os.path.join(SERVICES_TMP_DIR, "gpu_locks")
        os.makedirs(lock_dir, exist_ok=True)
        for k in range(slots):
            f = os.fdopen(os.open(os.path.join(lock_dir, f"gpu{index}.slot{k}"), os.O_RDONLY | os.O_CREAT, 0o666), "r")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                handle = f
                break
            except BlockingIOError:
                f.close()
    except OSError:
        handle = None
    try:
        yield handle
    finally:
        if handle is not None:
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()


@contextmanager
def gpu_session(need_bytes: int = 0, slots: int | None = None):
    """cupy bound to this process's device for one operation, or None (no device, memory or free slot)."""
    ordinal = gpu_device()
    if ordinal is None or (need_bytes and device_free_bytes(_gpu_state["nvml_index"]) < need_bytes):
        yield None
        return
    with _device_slot(_gpu_state["nvml_index"], slots) as slot:
        if slot is None:
            yield None
            return
        import cupy as cp

        try:
            cp.cuda.Device(ordinal).use()  # for this thread; no restore, so no second context on device 0
        except Exception as e:
            disable_gpu(f"device {ordinal}: {e!r}")
            yield None
            return
        try:
            yield cp
        finally:
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass


def on_gpu(func, *args, need_bytes=0, slots=None, budget_s=None, logger=None, name=None, **kwargs):
    """func(cp, *args, **kwargs) on the device, else None; a failure moves the process to the CPU, a slow call pauses."""
    name = name or func.__name__
    with gpu_session(need_bytes, slots) as cp:
        if cp is None:
            return None
        st = time.perf_counter()
        try:
            result = func(cp, *args, **kwargs)
            cp.cuda.Device().synchronize()
        except Exception as e:
            disable_gpu(f"{name}: {e!r}")
            if logger is not None:
                logger.warning(f"GPU {name} failed ({e!r}); this process continues on the CPU")
            return None
    elapsed = time.perf_counter() - st
    first = name not in _warm_kernels  # the first call compiles the kernels; the budget starts with the second
    _warm_kernels.add(name)
    if budget_s is not None and elapsed > budget_s and not first:
        pause_gpu(f"{name} took {elapsed:.1f} s (budget {budget_s:.1f} s)")
        if logger is not None:
            logger.warning(f"GPU {name} took {elapsed:.1f} s (budget {budget_s:.1f} s); CPU for {GPU_RETRY_SECONDS} s")
    return result


@contextmanager
def acquire_available_gpu(device_id=None, gpu_threshold=400, blocking=True, timeout=1):
    """
    Attempt to lock any available GPU(s) for up to `timeout` seconds.
    If no lock is acquired within the timeout, yields None.

    Args:
        device_id (int or None): Specific GPU ID to try; if None, try all available GPUs.
            If -1, force GPU usage: wait for a lock on any device GPU_DEVICES offers, busy or not.
        gpu_threshold: ignored.
        blocking (bool): Whether to block when attempting to acquire the lock.
        timeout (int | float): Total time (in seconds) to spend trying all GPUs.

    Yields:
        int or None: the cupy device ordinal if the lock was acquired; otherwise None.
    """
    # If CPU mode is requested, yield None immediately
    if device_id == "CPU":
        yield None
        return

    # Force GPU mode: if device_id == -1, try all GPUs regardless of availability
    force_gpu = device_id == -1
    if force_gpu:
        device_id = None  # Try all GPUs
        # Increase timeout for forced GPU mode
        timeout = max(timeout, 10)

    # Get list of GPUs whose memory usage is below the threshold
    available = _usable_devices(device_id)

    # forcing still stays within the host's GPU_DEVICES; it only skips the load predicate
    if force_gpu and not available and GPU_DEVICES is not None:
        available = [
            d
            for d in nvml_devices()
            if d["ordinal"] is not None and (GPU_DEVICES == "auto" or d["index"] in GPU_DEVICES)
        ]

    if not available:
        yield None
        return
    ordinal_of = {d["index"]: d["ordinal"] for d in available}
    available_gpus = [d["index"] for d in available]

    # one lock directory for every user of the host, so the two checkouts and ad-hoc runs arbitrate the same devices
    lock_dir = os.path.join(SERVICES_TMP_DIR, "gpu_locks")
    try:
        os.makedirs(lock_dir, exist_ok=True)
    except OSError:
        yield None
        return

    start_time = time.time()

    # Try each available GPU in turn
    for gpu_id in available_gpus:
        # If we've already spent the timeout, stop trying
        if time.time() - start_time >= timeout:
            break

        lock_path = os.path.join(lock_dir, f"gpu{gpu_id}.lock")
        try:
            lock_file = open(lock_path, "a+")
        except Exception:
            continue

        try:
            # Use blocking lock if forcing GPU usage, otherwise non-blocking
            if force_gpu and blocking:
                # Use blocking lock - will wait until GPU is available
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                # Success: yield the GPU ID
                yield ordinal_of[gpu_id]
                return
            else:
                # Use non-blocking lock to avoid futex waits
                # This eliminates the futex_wait_queue_me bottleneck
                flag = fcntl.LOCK_EX | fcntl.LOCK_NB
                try:
                    fcntl.flock(lock_file, flag)
                    # Success: yield the GPU ID
                    yield ordinal_of[gpu_id]
                    return
                except BlockingIOError:
                    # GPU is busy, try next one immediately
                    # No futex wait - fail fast and move on
                    continue

        finally:
            # Always release and close the lock file
            try:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
            except Exception:
                pass
            lock_file.close()

    # If no lock was acquired on any GPU, yield None
    yield None


class SortedGroupDict(UserDict):
    """
    A dictionary that has sorted preprocess, science, and cross-filter groups.

    Iterating over it gives sorted PreprocessGroups and ScienceGroups.
    Its values() method returns a sorted list of PreprocessGroups and ScienceGroups.

    The sorted order is:
    1. PreprocessGroup items first, sorted by number of sci_keys (descending)
    2. ScienceGroup items second, sorted by number of image_files (descending)
    3. CrossFilterGroup items last, sorted by number of source groups (descending)

    This ensures preprocessing groups with more science dependencies are processed first,
    and science groups with more images are processed first within their category.
    """

    @overload
    def __getitem__(self, key: int) -> PreprocessGroup | ScienceGroup: ...

    @overload
    def __getitem__(self, key: str) -> PreprocessGroup | ScienceGroup: ...

    def __getitem__(self, key: int | str) -> PreprocessGroup | ScienceGroup:
        if type(key) == int:
            return self.values()[key]
        else:
            return super().__getitem__(key)

    def __iter__(self) -> Iterator[PreprocessGroup | ScienceGroup]:
        """
        Returns an iterator over groups sorted by type and priority.

        Iteration order:
        1. PreprocessGroup items first, sorted by number of sci_keys (descending)
        2. ScienceGroup items second, sorted by number of image_files (descending)

        This ensures preprocessing groups with more science dependencies are processed first,
        and science groups with more images are processed first within their category.

        Yields:
            PreprocessGroup | ScienceGroup: Groups in sorted order.
        """
        return iter(self._get_sorted_values())

    def values(self) -> list[PreprocessGroup | ScienceGroup]:
        """
        Returns a list of groups sorted by type and priority.

        Sorting order:
        1. PreprocessGroup items first, sorted by number of sci_keys (descending)
        2. ScienceGroup items second, sorted by number of image_files (descending)

        This ensures preprocessing groups with more science dependencies are processed first,
        and science groups with more images are processed first within their category.

        Returns:
            list[PreprocessGroup | ScienceGroup]: Sorted list of groups.
        """
        return self._get_sorted_values()

    def items(self) -> list[tuple[str | None, PreprocessGroup | ScienceGroup]]:
        """
        Returns a list of (key, group) tuples sorted by type and priority.

        The key is a string extracted from the group's `key` attribute:
        - For PreprocessGroup: string from PathHandler.output_name (e.g., preproc config stem)
          or fallback format "mfg_{i}" (e.g., "mfg_0", "mfg_1")
        - For ScienceGroup: string from sci_dict keys, typically in format
          "nightdate_obj_filter" (e.g., "20250102_T08285_m425")
        - Returns None if the group doesn't have a `key` attribute (unlikely in practice)

        Returns:
            list[tuple[str | None, PreprocessGroup | ScienceGroup]]: Sorted list of
            (key, group) pairs, with PreprocessGroup items first, then ScienceGroup items.
        """
        sorted_values = self._get_sorted_values()
        return [(getattr(v, "key", None), v) for v in sorted_values]

    def _get_sorted_values(self) -> list[PreprocessGroup | ScienceGroup]:
        # Separate the dependency layers.
        preprocess_groups = []
        science_groups = []
        crossfilter_groups = []

        for value in self.data.values():
            if isinstance(value, PreprocessGroup):
                preprocess_groups.append(value)
            elif isinstance(value, ScienceGroup):
                science_groups.append(value)
            else:
                crossfilter_groups.append(value)

        # Sort PreprocessGroup by sci_keys length (descending)
        sorted_preprocess = sorted(preprocess_groups, key=lambda x: len(x.sci_keys), reverse=True)

        # Sort ScienceGroup by image_files length (descending)
        sorted_science = sorted(science_groups, key=lambda x: len(x.image_files), reverse=True)
        sorted_crossfilter = sorted(crossfilter_groups, key=lambda x: len(x.source_groups), reverse=True)

        return sorted_preprocess + sorted_science + sorted_crossfilter

    def __repr__(self):
        if len(self.values()) == 0:
            return "Group is empty"
        string = ""
        for value in self.values():
            string += str(value) + "\n"
        return string


class PreprocessGroup:
    def __init__(self, key):
        """
        Initialize a PreprocessGroup.

        Args:
            key: String identifier for this preprocessing group (e.g., from PathHandler.output_name).

        Attributes:
            sci_keys: List of strings representing ScienceGroup keys that depend on this
                preprocessing group. Used for scheduling dependencies and priority sorting.
                Each sci_key holds science images to be processed together.
        """
        self.key = key
        self._image_files = []
        self._config = None
        self.sci_keys: list[str] = []

    def __lt__(self, other):
        if isinstance(other, PreprocessGroup):
            # For PreprocessGroups, higher sci_keys count means higher priority
            # Reverse the comparison to make higher count come first
            return len(self.sci_keys) < len(other.sci_keys)
        else:
            # PreprocessGroup always comes before other types
            return False

    def __eq__(self, other):
        if hasattr(other, "sci_keys"):
            return len(self.sci_keys) == len(other.sci_keys)
        else:
            return False

    @property
    def config(self):
        if self._config is None:
            self.create_config()
        return self._config

    @property
    def image_files(self):
        return self._image_files

    def add_images(self, filepath):
        if isinstance(filepath, list):
            self._image_files.extend(filepath)
        elif isinstance(filepath, str):
            self._image_files.append(filepath)
        elif isinstance(filepath, tuple):
            _tmp_list = list(chain.from_iterable(filepath))
            self._image_files.extend(_tmp_list)
        else:
            raise ValueError("Invalid filepath type")

    def add_sci_keys(self, keys: str):
        self.sci_keys.append(keys)

    def create_config(
        self,
        overwrite: bool = False,
        is_too: bool = False,
        is_pipeline: bool = False,
        overwrite_preprocess: bool = False,
        **kwargs,
    ):
        from ..config import PreprocConfiguration

        overwrite = overwrite or overwrite_preprocess
        # print(
        #     f"PreprocessGroup {self.key} creating config with {len(self.image_files)} images; "
        #     f"{len(os.listdir(f'/proc/{os.getpid()}/fd'))} FDs under limit of {resource.getrlimit(resource.RLIMIT_NOFILE)} FDs"
        # )
        c = PreprocConfiguration(
            self.image_files,
            overwrite=overwrite,
            is_pipeline=is_pipeline,
            is_too=is_too,
        )

        self._config = c.config_file

        # Explicitly close logger file handlers so FDs are released immediately
        # rather than waiting for non-deterministic Logger.__del__ + gc to fire.
        try:
            if getattr(c, "logger", None) is not None and hasattr(c.logger, "cleanup"):
                c.logger.cleanup()
        finally:
            del c
            gc.collect()

    def __repr__(self):
        return f"PreprocessGroup ({self.key} used in {self.sci_keys} with {len(self.image_files)} images)"

    def cleanup(self):
        self._config = None
        self.sci_keys = []
        self._image_files = []


class ScienceGroup:
    def __init__(self, key):
        self.key = key
        self.image_files = []
        self._config = None
        self.multi_units = False

    @property
    def config(self):
        if self._config is None:
            self.create_config()
        return self._config

    def __lt__(self, other):
        if isinstance(other, PreprocessGroup):
            # ScienceGroup always comes after PreprocessGroup
            return False
        elif isinstance(other, ScienceGroup):
            # For ScienceGroups, higher image_files count means higher priority
            # Reverse the comparison to make higher count come first
            return len(self.image_files) < len(other.image_files)
        else:
            return True

    def __eq__(self, other):
        if not isinstance(other, ScienceGroup):
            return False
        return self.key == other.key

    def add_images(self, filepath):
        if isinstance(filepath, list):
            self.image_files.extend(filepath)
        elif isinstance(filepath, str):
            self.image_files.append(filepath)
        else:
            raise ValueError("Invalid filepath type")

    def create_config(self, overwrite=False, is_too=False, is_pipeline=False, is_multi_epoch=False, **kwargs):

        from ..config import SciProcConfiguration
        from ..path import PathHandler

        # print(
        #     f"PreprocessGroup {self.key} creating config with {len(self.image_files)} images; "
        #     f"{len(os.listdir(f'/proc/{os.getpid()}/fd'))} FDs under limit of {resource.getrlimit(resource.RLIMIT_NOFILE)} FDs"
        # )

        sci_yml = collapse(
            PathHandler(
                self.image_files, is_too=is_too, is_pipeline=is_pipeline, is_multi_epoch=is_multi_epoch
            ).sciproc_output_yml,
            raise_error=True,
        )
        if os.path.exists(sci_yml) and not overwrite:
            # If the config file already exists, load it
            c = SciProcConfiguration(sci_yml, write=True, is_too=is_too)  # 5 ms
            # c = SciProcConfiguration(sci_yml, write=True)  # 36 ms
        else:
            c = SciProcConfiguration(
                self.image_files,
                overwrite=overwrite,
                is_too=is_too,
                is_pipeline=is_pipeline,
                is_multi_epoch=is_multi_epoch,
            )
            # c = SciProcConfiguration.base_config(input_images=self.image_files, config_file=sci_yml, logger=True)
            # c = SciProcConfiguration.from_list(self.image_files)

        self._config = c.config_file

        # Explicitly close logger file handlers so FDs are released immediately
        # rather than waiting for non-deterministic Logger.__del__ + gc to fire.
        try:
            if getattr(c, "logger", None) is not None and hasattr(c.logger, "cleanup"):
                c.logger.cleanup()
        finally:
            del c
            gc.collect()

    def __repr__(self):
        return f"ScienceGroup({self.key} with {len(self.image_files)} images)"

    def cleanup(self):
        self._config = None
        self.image_files = []
        self.multi_units = False


class CrossFilterGroup:
    def __init__(self, key):
        self.key = key
        self.source_groups: list[ScienceGroup] = []
        self._config = None

    @property
    def config(self):
        if self._config is None:
            self.create_config()
        return self._config

    def add_source_group(self, group: ScienceGroup):
        if group not in self.source_groups:
            self.source_groups.append(group)

    def create_config(
        self,
        overwrite=False,
        is_too=False,
        is_pipeline=False,
        is_multi_epoch=False,
        config_suffix=None,
        **kwargs,
    ):
        from ..config import CrossFilterConfiguration

        config = CrossFilterConfiguration.from_science_configs(
            sorted(group.config for group in self.source_groups),
            overwrite=overwrite,
            is_too=is_too,
            is_pipeline=is_pipeline,
            is_multi_epoch=is_multi_epoch,
            config_suffix=config_suffix,
        )
        source_raw_images = sorted({image for group in self.source_groups for image in group.image_files})
        config.record_discovery(source_raw_images, DISCOVERY_RAW_INVENTORY)  # skip input completeness check
        self._config = config.config_file

        try:
            if getattr(config, "logger", None) is not None and hasattr(config.logger, "cleanup"):
                config.logger.cleanup()
        finally:
            del config
            gc.collect()

    def __repr__(self):
        return f"CrossFilterGroup({self.key} from {len(self.source_groups)} science groups)"

    def cleanup(self):
        self._config = None
        self.source_groups = []
