from typing import List, Optional, Dict, Any, Callable, Union, Tuple
from enum import Enum
from datetime import datetime
import time
import sys
import itertools
import traceback
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
    LOW = 0
    MEDIUM = 1
    HIGH = 2

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
        self.task_name = task_name
        self.priority = priority
        self.starttime = starttime or datetime.now()
        self.endtime = endtime or datetime.now()
        self.gpu = gpu
        self.device = device
        self.status = status
        self.result = result
        self.error = error
        self.time_priority = int(time.time() * 1000)
        self.inherit_input = inherit_input
        
        # Set task name after initialization to avoid recursion
        if task_name is None:
            self.task_name = self._get_task_name()

    def __lt__(self, other):
        return self.sort_index < other.sort_index
    
    @property
    def func(self):
        return self._get_function()
    
    @property
    def sort_index(self):
        return (self.priority.value + 1) * 1000000 + self.time_priority
    
    def set_time_priority(self):
        self.time_priority = int(time.time() * 1000)  # milliseconds since epoch
    
    def _get_task_name(self):
        if self.task_name:
            return self.task_name
        if self._func_name is None:
            return "unnamed_task"
        if self.cls is None:
            return self._func_name
        return f"{type(self.cls).__name__}.{self._func_name}"

    def _get_function(self):
        """Retrieve the callable function for this task.
        
        Returns:
            Callable: The function to be executed
        """
        if self.cls is not None:
            return getattr(self.cls, self._func_name)
        elif self._func is not None:
            return self._func

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
            if self.gpu:
                with self.gpu_context():
                    self.result = self.func(*self.args, **self.kwargs)
            else:
                self.result = self.func(*self.args, **self.kwargs)
                
            self.status = "completed"
            return self
            
        except Exception as e:
            self.status = "failed"
            self.error = e
            self.endtime = datetime.now()
            error_msg = f"Task {self.id} failed: {str(e)}\n{traceback.format_exc()}"
            print(error_msg, file=sys.stderr)
            raise

    @contextmanager
    def gpu_context(self):
        """Context manager for GPU operations with proper device handling.
        
        Yields:
            None
            
        Raises:
            RuntimeError: If GPU operation fails
        """
        device = self.device or 0  # Default to device 0 if not specified
        try:
            with cp.cuda.Device(device):
                yield
                # Ensure all operations are completed before leaving the context
                cp.cuda.stream.get_current_stream().synchronize()
        except Exception as e:
            error_msg = f"GPU operation failed on device {device}: {e}"
            # Can't use logger here as it might not be picklable
            print(error_msg, file=sys.stderr)
            raise RuntimeError(error_msg) from e
            
    def __getstate__(self):
        # Return a dictionary of the object's state, excluding unpicklable attributes
        state = self.__dict__.copy()
        # Remove unpicklable attributes
        state['_func'] = None
        state['cls'] = None
        return state
        
    def __setstate__(self, state):
        # Restore the object's state
        self.__dict__.update(state)
        # Reinitialize unpicklable attributes if needed
        self._func = None  # Will be set by the worker process

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
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to a dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the task
        """
        return {
            'id': self.id,
            'task_name': self.task_name,
            'status': self.status,
            'priority': self.priority.name if self.priority else None,
            'gpu': self.gpu,
            'device': self.device,
            'starttime': self.starttime.isoformat() if self.starttime else None,
            'endtime': self.endtime.isoformat() if self.endtime else None,
            'duration': (self.endtime - self.starttime).total_seconds() 
                       if self.starttime and self.endtime else None,
            'error': str(self.error) if self.error else None
        }

class TaskTree:
    """
    Represents the processing pipeline for a single image, containing multiple tasks.
    tasks are processed sequentially.
    """
    _id_counter = itertools.count(0)

    def __init__(self, tasks: List[Task], id: str=None):
        self.id = id or f"tree_{next(self._id_counter)}"
        self.tasks = tasks
        self.status = "pending"
        
        self.results = {}
    
    def add_task(self, task: Task) -> None:
        """Add a processing task to this tree."""
        self.tasks.append(task)
    
    def get_next_task(self) -> Optional[Task]:
        """Get the next task to be processed."""
        if len(self.tasks) == 0:
            return None
        return self.tasks[0]
    
    def advance(self) -> None:
        """Move to the next task after current one completes."""
        self.tasks.pop(0)
        cleanup_memory()
        if len(self.tasks) == 0:
            self.status = "completed"
    
    def is_complete(self) -> bool:
        """Check if all tasks have been processed."""
        return self.status == "completed"
    
    def store_result(self, task_id: str, result: Any) -> None:
        """Store the result of a task execution."""
        self.results[task_id] = result
        
    def get_all_results(self) -> Dict[str, Any]:
        """Get all results from all tasks."""
        return self.results
        
    def __repr__(self):
        base = f"TaskTree(id={self.id}, status={self.status}, tasks={len(self.tasks)})"
        