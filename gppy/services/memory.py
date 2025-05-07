from enum import Enum
from tkinter.constants import FALSE
from typing import Optional, Dict, List, Tuple
import time
import gc

from astropy.utils import state
import cupy as cp
import psutil
import threading
from contextlib import contextmanager
from datetime import datetime
from astropy.table import Table

from ..base import decorator
from . import utils


@contextmanager
def monitor_memory_usage(interval: float = 1.0, logger: Optional = None, verbose: bool = False) -> Table:
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

    # Create column names based on number of GPUs detected
    n_gpus = len(MemoryMonitor.current_gpu_memory_percent)
    column_names = ["time", "cpu_memory"] + [f"gpu{i}_memory" for i in range(n_gpus)]

    usage_data = Table(names=column_names, dtype=[object, float] + [float] * n_gpus)

    # Set column descriptions
    usage_data["time"].description = "Measurement timestamp"
    usage_data["cpu_memory"].description = "CPU memory usage (%)"
    for i in range(n_gpus):
        usage_data[f"gpu{i}_memory"].description = f"GPU {i} memory usage (%)"

    stop_thread = False

    def logging_thread() -> None:
        while not stop_thread:
            current_time = str(datetime.now())
            cpu_memory = MemoryMonitor.current_memory_percent
            gpu_memories = MemoryMonitor.current_gpu_memory_percent

            # Create row with timestamp, CPU memory, and all GPU memories
            row = [current_time, cpu_memory] + gpu_memories
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


class MemoryState(Enum):
    HEALTHY = ("healthy", "continue", None, 0)
    WARNING = ("warning", "cleanup", 70.0, 1)
    CRITICAL = ("critical", "pause", 85.0, 2)
    EMERGENCY = ("emergency", "stop", 95.0, 3)

    def __init__(self, state: str, action: str, threshold: Optional[float], order: int):
        self.state = state
        self.action = action
        self.threshold = threshold
        self.order = order


class MemoryMonitor:
    """
    Advanced memory monitoring and management system.

    Provides comprehensive tracking and intelligent management of memory resources
    across CPU and GPU devices. Offers real-time memory state detection,
    proactive cleanup strategies, and detailed reporting.

    Key Responsibilities:
    - Monitor CPU and GPU memory usage
    - Detect and classify memory states
    - Implement memory recovery strategies
    - Log and report memory usage

    Workflow:
    1. Continuously track memory usage
    2. Classify memory state
    3. Trigger appropriate recovery actions
    4. Provide detailed memory usage reports

    Args:
        logger: Logging instance for tracking memory events

    Attributes:
        logger: Logging system
        _memory_state (MemoryState): Current memory state

    Example:
        >>> monitor = MemoryMonitor(logger)
        >>> state, trigger = monitor.get_unified_state()
        >>> if state == MemoryState.WARNING:
        ...     monitor.handle_state(trigger, gpu_context, stop_callback)
    """

    def __init__(self, logger):
        self.logger = logger
        self._memory_state = {"CPU": MemoryState.HEALTHY, "GPU": MemoryState.HEALTHY}

    def __repr__(self):
        """
        Provide a concise string representation of the MemoryMonitor.

        Returns:
            str: Current memory state and usage summary
        """
        return f"MemoryMonitor(state={self.memory_state}, usage={self.log_memory_usage})"

    @classmethod
    def cleanup_memory(cls):
        """
        Perform system-wide memory cleanup.

        Calls utility function to release unused memory resources.
        """
        utils.cleanup_memory()

    @property
    def memory_state(self):
        """
        Get the current memory state.

        Returns:
            MemoryState: Current memory usage state
        """
        return self._memory_state

    def _initialize_gpu_devices(self) -> List[int]:
        """
        Safely initialize available GPU devices.

        Attempts to retrieve all CUDA-capable GPU devices.

        Returns:
            List[int]: List of available GPU device indices

        Notes:
            Falls back to CPU-only mode if GPU initialization fails
        """
        try:
            return list(range(cp.cuda.runtime.getDeviceCount()))
        except Exception as e:
            self.logger.warning(f"GPU initialization failed: {e}. Falling back to CPU only.")
            return []

    def get_unified_state(
        self,
    ) -> Tuple["MemoryState", str]:
        """
        Determine the most severe memory state across CPU and GPU.

        Evaluates memory usage for CPU and all available GPU devices,
        returning the most critical state and its source.

        Returns:
            Tuple[MemoryState, str]: Most severe memory state and its source (e.g., 'CPU', 'GPU0')

        Strategy:
        - Check CPU memory state
        - Check each GPU's memory state
        - Return the most severe state
        """
        states = {"CPU": MemoryState.HEALTHY, "GPU": MemoryState.HEALTHY}

        # Check CPU state
        if self.current_memory_percent >= MemoryState.EMERGENCY.threshold:
            states["CPU"] = MemoryState.EMERGENCY
        elif self.current_memory_percent >= MemoryState.CRITICAL.threshold:
            states["CPU"] = MemoryState.CRITICAL
        elif self.current_memory_percent >= MemoryState.WARNING.threshold:
            states["CPU"] = MemoryState.WARNING
        else:
            states["CPU"] = MemoryState.HEALTHY

        # Check each GPU state
        gpu_state = []
        for i, gpu_percent in enumerate(self.current_gpu_memory_percent):
            if gpu_percent >= MemoryState.EMERGENCY.threshold:
                gpu_state.append(MemoryState.EMERGENCY)
            elif gpu_percent >= MemoryState.CRITICAL.threshold:
                gpu_state.append(MemoryState.CRITICAL)
            elif gpu_percent >= MemoryState.WARNING.threshold:
                gpu_state.append(MemoryState.WARNING)
            else:
                gpu_state.append(MemoryState.HEALTHY)

        # Use the healthiest state
        try:
            sorted_gpu_states = sorted(gpu_state, key=lambda x: x.order, reverse=False)
            states["GPU"] = sorted_gpu_states[0]
        except:
            states["GPU"] = MemoryState.HEALTHY

        self._memory_state = states
        return states

    def should_recover(self, recovery_threshold=60.0) -> bool:
        """
        Check if memory usage has dropped below recovery threshold.

        Args:
            recovery_threshold (float, optional): Memory usage percentage
                below which recovery is considered successful. Defaults to 60.0.

        Returns:
            bool: Whether memory usage is below recovery threshold
        """
        return self.current_memory_percent <= recovery_threshold and any(
            gpu_percent <= recovery_threshold for gpu_percent in self.current_gpu_memory_percent
        )

    def handle_state(self, trigger_source, gpu_context, stop_callback) -> None:
        """
        Handle memory state based on current usage and trigger source.

        Implements different recovery strategies for various memory states:
        - WARNING: Light cleanup
        - CRITICAL: Aggressive memory recovery
        - EMERGENCY: Immediate process stoppage

        Args:
            trigger_source: Source of memory pressure (e.g., 'CPU', 'GPU0')
            gpu_context: Context manager for GPU operations
            stop_callback: Function to stop all processing
        """
        if self.memory_state[trigger_source] == MemoryState.WARNING:
            self._handle_warning(trigger_source, gpu_context)
        elif self.memory_state[trigger_source] == MemoryState.CRITICAL:
            self._handle_critical(trigger_source, gpu_context)
        elif self.memory_state[trigger_source] == MemoryState.EMERGENCY:
            self._handle_emergency(trigger_source, gpu_context, stop_callback)
            self.logger.critical(f"Emergency memory threshold exceeded on {trigger_source}. All processes stopped.")

    def _handle_warning(self, trigger_source: str, gpu_context) -> None:
        """
        Handle WARNING memory state with minimal intervention.

        Performs lightweight memory cleanup:
        - For GPU: Free memory pool blocks
        - For CPU: Trigger garbage collection

        Args:
            trigger_source (str): Source of memory pressure
            gpu_context: Context manager for GPU operations
        """
        if trigger_source.startswith("GPU"):
            # GPU-specific cleanup
            device = int(trigger_source[3:])
            with gpu_context(device):
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
        else:
            # CPU cleanup
            gc.collect()

    def _handle_critical(
        self,
        trigger_source: str,
        gpu_context,
    ) -> None:
        """
        Handle CRITICAL memory state with aggressive recovery.

        Implements a recovery loop that:
        - Frees GPU memory pools
        - Triggers system-wide memory cleanup
        - Waits and monitors until memory usage recovers

        Args:
            trigger_source (str): Source of memory pressure
            gpu_context: Context manager for GPU operations
        """
        while True:
            if trigger_source.startswith("GPU"):
                device = int(trigger_source[3:])
                with gpu_context(device):
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()

            utils.cleanup_memory()

            if self.should_recover():
                self.logger.info(
                    f"Memory recovered - CPU: {self.current_memory_percent:.2f}%, "
                    f"GPUs: {[f'{p:.2f}%' for p in self.current_gpu_memory_percent]}"
                )
                break

            self.logger.warning(
                f"Waiting for memory to recover - CPU: {self.current_memory_percent:.2f}%, "
                f"GPUs: {[f'{p:.2f}%' for p in self.current_gpu_memory_percent]}"
            )
            time.sleep(5)

    def _handle_emergency(self, trigger_source: str, gpu_context, stop_callback) -> None:
        """
        Handle EMERGENCY memory state with immediate stoppage.

        Performs comprehensive cleanup and halts all processing:
        - Free all GPU memory pools
        - Trigger system-wide memory cleanup
        - Stop all running processes

        Args:
            trigger_source (str): Source of memory pressure
            gpu_context: Context manager for GPU operations
            stop_callback: Function to stop all processing
        """
        # Attempt emergency cleanup
        if trigger_source.startswith("GPU"):
            for device in range(cp.cuda.runtime.getDeviceCount()):
                with gpu_context(device):
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()

        utils.cleanup_memory()
        stop_callback()

    @decorator.classmethodproperty
    def current_memory(cls):
        """
        Get current CPU memory usage statistics.

        Returns:
            Dict: Memory usage details including total, used, free memory, and percentage
        """
        used = psutil.Process().memory_info().rss / 1024 / 1024
        total = psutil.virtual_memory().total / 1024 / 1024
        return {
            "total": total,
            "used": used,
            "free": total - used,
            "percent": (used / total) * 100,
        }

    @decorator.classmethodproperty
    def current_gpu_memory(cls) -> Dict:
        """
        Get GPU memory statistics for all available devices.

        Returns:
            Dict: Memory usage details for each GPU device
        """
        gpu_stats = {}

        if cp.cuda.runtime.getDeviceCount() > 0:
            for device in range(cp.cuda.runtime.getDeviceCount()):
                try:
                    with cp.cuda.Device(device):
                        mem_info = cp.cuda.runtime.memGetInfo()
                        total = mem_info[1] / 1024 / 1024  # in MB
                        free = mem_info[0] / 1024 / 1024  # in MB
                        used = total - free
                        gpu_stats[f"device_{device}"] = {
                            "total": total,
                            "used": used,
                            "free": free,
                            "percent": (used / total) * 100,
                        }
                except:
                    continue
        return gpu_stats

    @decorator.classmethodproperty
    def current_memory_percent(cls):
        """
        Get current CPU memory usage percentage.

        Returns:
            float: Percentage of CPU memory used
        """
        return cls.current_memory["percent"]

    @decorator.classmethodproperty
    def current_gpu_memory_percent(cls):
        """
        Get current GPU memory usage percentages.

        Returns:
            List[float]: Memory usage percentage for each GPU device
        """
        gpu_percentages = [stats["percent"] for _, stats in cls.current_gpu_memory.items()]
        return gpu_percentages

    @decorator.classmethodproperty
    def log_memory_usage(cls):
        """
        Generate a comprehensive memory usage log string.

        Returns:
            str: Formatted string with CPU and GPU memory usage percentages
        """
        gpu_summary = [f"{device}: {percent:.2f}%" for device, percent in enumerate(cls.current_gpu_memory_percent)]
        gpu_info = f", GPU [{', '.join(gpu_summary)}]"
        return f"System [{cls.current_memory_percent:.2f}%]{gpu_info}"
