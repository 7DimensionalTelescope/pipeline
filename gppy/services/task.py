from typing import Optional, Any
from enum import Enum
from datetime import datetime
import itertools

class Priority(Enum):
    """
    Task priority levels for workload management.

    Provides a hierarchical priority system to manage task execution:
    - PREPROCESS: Highest priority for preprocessing tasks
    - HIGH: Time-sensitive or critical tasks
    - MEDIUM: Standard processing tasks
    - LOW: Background or non-critical tasks

    Allows fine-grained control over task scheduling and resource allocation.
    """
    PREPROCESS = 0
    HIGH = 1
    MEDIUM = 5
    LOW = 10

class Task:
    """
    A task container for pipeline processing with execution tracking.
    
    This class encapsulates a function call with its arguments, metadata,
    and execution state. It provides a standardized interface for task
    management in the pipeline system.
    
    Features:
    - Function execution with arguments
    - Priority-based scheduling
    - Execution status tracking
    - Result and error storage
    - Timing information
    - Unique task identification
    
    Args:
        func: Function to be executed
        args (tuple, optional): Positional arguments for the function
        kwargs (dict, optional): Keyword arguments for the function
        priority (Priority, optional): Task priority level (default: MEDIUM)
        id (str, optional): Unique task identifier
        starttime (datetime, optional): Task start time
        endtime (datetime, optional): Task end time
        task_name (str, optional): Descriptive name for the task
        status (str): Initial task status (default: "pending")
        result (Any): Task execution result
        error (Exception, optional): Exception if task failed
    
    Example:
        >>> task = Task(
        ...     func=process_image,
        ...     args=(image_path,),
        ...     kwargs={"output_dir": "/output"},
        ...     priority=Priority.HIGH,
        ...     task_name="Process Image A"
        ... )
        >>> result = task.execute()
    """
    
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
        """
        Execute the task's function with the provided arguments.
        
        Executes the stored function with the specified arguments and
        updates the task status, timing, and result information.
        
        Returns:
            Any: The result of the function execution
            
        Raises:
            Exception: If task execution fails (stored in self.error)
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
        """
        Return a string representation of the task.
        
        Returns:
            str: String representation including ID, name, status, and priority
        """
        return (
            f"Task(id={self.id}, "
            f"name={self.task_name}, "
            f"status={self.status}, "
            f"priority={self.priority.name if self.priority else 'N/A'})"
        )
        