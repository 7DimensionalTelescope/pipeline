from enum import Enum
from typing import Optional, Dict, List, Tuple
import time
import gc
import psutil
import pynvml

from . import utils


class MemoryState(Enum):
    """
    Memory state classification for system monitoring.
    
    Defines different memory states with associated actions and thresholds:
    - HEALTHY: Normal operation, no action needed
    - WARNING: Elevated memory usage, cleanup recommended
    - CRITICAL: High memory usage, pause processing
    - EMERGENCY: Critical memory usage, stop all processing
    
    Each state includes:
    - state: Human-readable state name
    - action: Recommended action to take
    - threshold: Memory percentage threshold (None for HEALTHY)
    - order: Numeric ordering for comparison
    """
    HEALTHY = ("healthy", "continue", None, 0)
    WARNING = ("warning", "cleanup", 80.0, 1)
    CRITICAL = ("critical", "pause", 90.0, 2)
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

    def __init__(self, logger=None):
        if logger is None:
            from .logger import Logger

            self.logger = Logger(name="MemoryMonitor")
        else:
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

    @utils.classmethodproperty
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

    @utils.classmethodproperty
    def current_gpu_memory(cls) -> Dict:
        """
        Get GPU memory statistics for all available devices.

        Returns:
            Dict: Memory usage details for each GPU device
        """
        import pynvml

        gpu_stats = {}

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for device in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(device)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                total = mem_info.total / 1024 / 1024  # in MB
                used = mem_info.used / 1024 / 1024
                free = mem_info.free / 1024 / 1024

                gpu_stats[f"device_{device}"] = {
                    "total": total,
                    "used": used,
                    "free": free,
                    "percent": (used / total) * 100,
                }

        finally:
            pynvml.nvmlShutdown()

        return gpu_stats

    @utils.classmethodproperty
    def current_memory_percent(cls):
        """
        Get current CPU memory usage percentage.

        Returns:
            float: Percentage of CPU memory used
        """
        return cls.current_memory["percent"]

    @utils.classmethodproperty
    def current_gpu_memory_percent(cls):
        """
        Get current GPU memory usage percentages.

        Returns:
            List[float]: Memory usage percentage for each GPU device
        """
        gpu_percentages = [stats["percent"] for _, stats in cls.current_gpu_memory.items()]
        return gpu_percentages

    @utils.classmethodproperty
    def current_gpu_utilization(cls):
        """
        Get current GPU utilization percentages.

        Returns:
            List[float]: Utilization percentage for each GPU device
        """
        gpu_utils = []
        import pynvml

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utils.append(util.gpu)  # GPU utilization percentage (0–100)
        finally:
            pynvml.nvmlShutdown()

        return gpu_utils

    @utils.classmethodproperty
    def log_memory_usage(cls):
        """
        Generate a comprehensive memory usage log string.

        Returns:
            str: Formatted string with CPU and GPU memory usage percentages
        """
        gpu_summary = [f"{device}: {percent:.2f}%" for device, percent in enumerate(cls.current_gpu_memory_percent)]
        gpu_info = f", GPU [{', '.join(gpu_summary)}]"
        return f"System [{cls.current_memory_percent:.2f}%]{gpu_info}"
