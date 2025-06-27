from typing import Optional, Any, Callable, Union, Tuple
from enum import Enum
from datetime import datetime
import time
import sys
import itertools
from .utils import cleanup_memory
import cupy as cp
from contextlib import contextmanager

class Priority(Enum):
    """
    Task priority levels for workload management.

    Provides a hierarchical priority system to manage task execution:
    - LOW: Background or non-critical tasks
    - MEDIUM: Standard processing tasks
    - HIGH: Time-sensitive or critical tasks

    Allows fine-grained control over task scheduling and resource allocation.
    """
    LOW = 10
    MEDIUM = 5
    HIGH = 0

class Task:
    _id_counter = itertools.count(0)

    def __init__(
        self,
        func,
        args: Optional[tuple] = None,
        kwargs: Optional[dict] = None,
        priority: Optional[Priority] = Priority.MEDIUM,
        id: Optional[str] = None,
        starttime: Optional[datetime] = None,
        endtime: Optional[datetime] = None,
        gpu: Optional[bool] = False,
        device: Optional[int] = None,
        task_name: Optional[str] = None,
        status: str = "pending",
        result: Any = None,
        error: Optional[Exception] = None,
        cls: Optional[Any] = None,
        inherit_input: Optional[bool] = False,
    ):
        # Store function and class references
        self._func = func
        self._func_name = func.__name__ if func else None
        
        self.cls = cls
        
        # Store arguments
        self.args = args or ()
        self.kwargs = kwargs or {}
        
        # Task metadata
        self.id = id or f"task_{next(self._id_counter)}"
        self.priority = priority
        self.starttime = starttime or datetime.now()
        self.endtime = endtime or datetime.now()
        self.gpu = gpu
        self._device = device
        self.status = status
        self.result = result
        self.error = error
        self.time_priority = int(time.time() * 1000)
        self.inherit_input = inherit_input
        self.task_name = self._get_task_name(task_name)

    def __lt__(self, other):
        return self.sort_index < other.sort_index
    
    @property
    def device(self):
        if self._device is None:
            from .utils import get_best_gpu_device
            self._device = get_best_gpu_device()
        return self._device
    
    @property
    def func(self):
        return self._get_function()
    
    @property
    def sort_index(self):
        return (self.priority.value + 1) * 1000000 + self.time_priority
    
    def set_time_priority(self):
        self.time_priority = int(time.time() * 1000)  # milliseconds since epoch
    
    def _get_task_name(self, task_name):
        if task_name:
            return task_name
        elif self._func_name is None:
            return "unnamed_task"
        elif self.cls is None:
            return self._func_name
        else:
            return f"{type(self.cls).__name__}.{self._func_name}"

    def _get_function(self):
        """Retrieve the callable function for this task.
        
        Returns:
            Callable: The function to be executed
        """
        if self.cls is not None and self._func_name is not None:
            func = getattr(self.cls, self._func_name, None)
            if func is None:
                raise ValueError(f"Function '{self._func_name}' not found in class {type(self.cls).__name__}")
            return func
        elif self._func is not None:
            return self._func
        else:
            raise ValueError(f"No function reference available for task {self.id}")

    def execute(self):
        """Execute the task's function with the provided arguments.
        
        Returns:
            Task: The current task instance with updated status and result
            
        Raises:
            Exception: If task execution fails
        """
        self.status = "processing"
        self.starttime = datetime.now()
        try:
            # Execute function with appropriate context
            if self.gpu:
                with cp.cuda.Device(self.device):
                    self.result = self.func(device_id = self.device, *self.args, **self.kwargs)
            else:
                self.result = self.func(*self.args, **self.kwargs)
        except Exception as e:
            self.status = "failed"
            self.error = e
            try:
                import logging
                tmp_logger = logging.getLogger(self.task_name)
                tmp_logger.error(f"Task {self.id} failed with error: {str(e)}")
            except:
                print(e)
            raise
        self.status = "completed"
        self.endtime = datetime.now()
        return self.result

    def cleanup(self):
        """Cleanup resources after task execution.
        
        This method releases both CPU and GPU resources used by the task.
        """
        # Clear function references
        self._func = None
        # Clear arguments to free memory
        self.args = None
        self.kwargs = None
        
        # Clear result if it's a large object
        if hasattr(self.result, 'nbytes') and self.result.nbytes > 1e6:  # If result is large (>1MB)
            self.result = None
            
        # Clean up GPU memory if this was a GPU task
        if self.gpu:
            with cp.cuda.Device(self.device):
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                except Exception as e:
                    print(f"Warning: Failed to free GPU memory: {e}", file=sys.stderr)
        
        # General memory cleanup
        cleanup_memory()
        
    def __repr__(self):
        """Return a string representation of the task.
        
        Returns:
            str: String representation including ID, name, and status
        """
        cls_name = type(self.cls).__name__ if self.cls is not None else None
        task_name = f"{cls_name}.{self.task_name}" if cls_name else self.task_name
        
        return (
            f"Task(id={self.id}, "
            f"name={task_name}, "
            f"status={self.status}, "
            f"gpu={self.gpu}, "
            f"priority={self.priority.name if self.priority else 'N/A'})"
        )
        