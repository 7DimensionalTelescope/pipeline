from typing import List, Optional, Dict, Any, Callable
from enum import Enum
from datetime import datetime
import time
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
    LOW = 0
    MEDIUM = 1
    HIGH = 2

class Task:
    _id_counter = itertools.count(0)

    def __init__(
        self,
        func: Optional[Callable]=None,
        args: Optional[tuple] = (),
        kwargs: Optional[dict] = {},
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
        cls: Optional[Any] = None
    ):
        self._func = func
        self.cls = cls
        self.args = args or ()
        self.kwargs = kwargs or {}

        self.id = id or f"task_{next(self._id_counter)}"
        self.task_name = task_name or func.__name__
        
        self.priority = priority
        self.starttime = starttime or datetime.now()
        self.endtime = endtime or datetime.now()
        self.gpu = gpu
        self.device = device
        self.status = status
        self.result = result
        self.error = error
        timestamp_part = time.time() % 1  # Get just the decimal part
        self.sort_index = (self.priority.value + 1) * 1000000 + timestamp_part
    
    def __lt__(self, other):
        return self.sort_index < other.sort_index
        
    @property
    def func(self):
        return self._get_function()

    def _get_function(self):
        if self.cls is None:
            return self._func
        return getattr(self.cls, self._func.__name__)
    
    def execute(self):
        try:
            self.status = "processing"
            self.starttime = datetime.now()
            if self.gpu:
                with self.gpu_context():
                    self.result = self.func(*self.args, **self.kwargs)
            else:
                self.result = self.func(*self.args, **self.kwargs)
        except Exception as e:
            self.status = "failed"
            self.result = None
            self.error = e
            print(f"Task {self.id} failed with error: {e}")
        finally:
            self.status = "completed"
            self.endtime = datetime.now()
        return self

    @contextmanager
    def gpu_context(self):
        try:
            with cp.cuda.Device(self.device):
                yield
        except Exception as e:
            print(f"GPU operation failed on device {device}: {e}")
            raise

    def cleanup(self):
        """Cleanup resources after task execution."""
        self._func = None
        self.args = None
        self.kwargs = None
        self.cls = None
        cleanup_memory()
        
    def __repr__(self):
        if self.cls is None:
            return f"Task(id={self.id}, task_name={self.task_name}, status={self.status})"
        else:
            return f"Task(id={self.id}, task_name={type(self.cls).__name__}.{self.task_name}, status={self.status})"

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
        base = f"TaskTree(id={self.id}, status={self.status}\n"
        return base + "\n".join([str(task) for task in self.tasks])