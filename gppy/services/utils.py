import gc
import numpy as np
import time
import psutil
import threading
from contextlib import contextmanager
from datetime import datetime
from astropy.table import Table

from typing import Optional

import fcntl
from contextlib import contextmanager
import pynvml

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


def get_best_gpu_device():
    from ..services.memory import MemoryMonitor

    percent = MemoryMonitor.current_gpu_memory_percent
    available = []
    for i, p in enumerate(percent):
        if not (acquire_available_gpu(i)) and p < 90:
            available.append(p)

    if len(available) == 0:
        return "CPU"
    else:
        return int(np.argmin(available))
    import pynvml

def check_gpu_activity(device_id=None, gpu_threshold=500):
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    available = []

    indices = [device_id] if device_id is not None else range(device_count)

    for i in indices:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_MB = meminfo.used / 1024 / 1024
        if used_MB < gpu_threshold:
            available.append(i)

    pynvml.nvmlShutdown()
    return available

@contextmanager
def acquire_available_gpu(device_id=None, gpu_threshold=500, blocking=True, timeout=1):
    """
    Context manager that locks an available GPU (based on memory and lock).

    Args:
        device_id (int or None): If given, only try to acquire this GPU.
        gpu_threshold (int): GPU memory usage threshold in MB.
        blocking (bool): Whether to block when trying to acquire lock.
        timeout (int): Timeout in seconds for lock acquisition.

    Yields:
        int or None: GPU ID if successfully locked, otherwise None.
    """
    if device_id == "CPU":
        yield None
    
    available_gpus = check_gpu_activity(device_id=device_id, gpu_threshold=gpu_threshold)

    for gpu_id in available_gpus:
        lock_path = f"/tmp/gpu_locks/gpu{gpu_id}.lock"
        try:
            lock_file = open(lock_path, "w")
        except Exception:
            continue  # Skip if lock file can't be opened

        t_start = time.time()

        while True:
            try:
                flag = fcntl.LOCK_EX
                if not blocking:
                    flag |= fcntl.LOCK_NB

                fcntl.flock(lock_file, flag)

                try:
                    yield gpu_id  # Success
                finally:
                    # Always unlock and close on exit, even on exception
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
                    lock_file.close()
                return
            except BlockingIOError:
                if not blocking or (time.time() - t_start) > timeout:
                    break
                time.sleep(0.1)

        lock_file.close()

    yield None