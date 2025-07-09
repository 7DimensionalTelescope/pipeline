from typing import Optional, Any
from enum import Enum
from datetime import datetime
import itertools

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
    HIGH = 1
    PREPROCESS = 0

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
        task_name: Optional[str] = None,
        status: str = "pending",
        result: Any = None,
        error: Optional[Exception] = None,
    ):
        # Store function and class references
        self.func = func
        self.task_name = task_name or func.__name__

        # Store arguments
        self.args = args or ()
        self.kwargs = kwargs or {}
        
        # Task metadata
        self.id = id or f"task_{next(self._id_counter)}"
        self.priority = priority
        self.starttime = starttime or datetime.now()
        self.endtime = endtime or datetime.now()
        self.status = status
        self.result = result
        self.error = error

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

    def __repr__(self):
        """Return a string representation of the task.
        
        Returns:
            str: String representation including ID, name, and status
        """
        return (
            f"Task(id={self.id}, "
            f"name={self.task_name}, "
            f"status={self.status}, "
            f"priority={self.priority.name if self.priority else 'N/A'})"
        )
        