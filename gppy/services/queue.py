from multiprocessing.pool import Pool
import multiprocessing as mp

import queue

import threading

import cupy as cp

import time
import signal
import os

from typing import Callable, Any, Optional, List, Dict, Union
from datetime import datetime
from ..logger import Logger
from .task import TaskTree, Task, Priority

from contextlib import contextmanager

import itertools

from .memory import MemoryState, MemoryMonitor

from .utils import cpu_callback_wrapper

signal.signal(signal.SIGINT, signal.SIG_IGN)

class AbruptStopException(Exception):
    """Custom exception to signal abrupt stop processing."""
    pass


class QueueManager:
    """
    Advanced task management system for parallel processing.

    Provides comprehensive task queuing, processing, and resource management
    capabilities. Supports both CPU and GPU task processing with:
    - Multi-priority task scheduling
    - Dynamic resource allocation
    - Memory state monitoring
    - Error tracking and logging

    Key Responsibilities:
    - Task submission and tracking
    - Parallel task execution
    - Resource allocation
    - Memory management
    - Error handling and logging

    Workflow:
    1. Initialize with configurable worker pool
    2. Submit tasks with priorities and resource requirements
    3. Automatically distribute tasks across CPU/GPU
    4. Monitor and manage system resources
    5. Track task status and results

    Args:
        max_workers (int, optional): Maximum number of CPU workers
        logger (Logger, optional): Custom logger instance
        **kwargs: Additional configuration parameters

    Example:
        >>> queue = QueueManager(max_workers=8)
        >>> queue.add_task(my_function, args=(param1, param2), priority=Priority.HIGH)
        >>> queue.wait_all_task_completion()
    """

    _id_counter = itertools.count(1)
    _gpu_device = 0

    def __init__(
        self,
        max_workers: Optional[int] = None,
        max_gpu_workers: int = 4,
        logger: Optional[Logger] = None,
        **kwargs,
    ):
        # Initialize logging
        if logger:
            self.logger = logger
        else:
            self.logger = Logger()

        self.logger.debug(f"Initialize QueueManager.")

        self.memory_monitor = MemoryMonitor(logger=self.logger)

        # Default CPU allocation
        self.total_cpu_worker = max_workers or mp.cpu_count() - 1

        # Single priority queue for CPU tasks
        self.cpu_queue = queue.Queue()
        self.cpu_high_priority_queue = queue.Queue()
        
        # Create multiple GPU worker threads
        self.gpu_devices = range(cp.cuda.runtime.getDeviceCount())
        self.gpu_queue = {device: queue.Queue() for device in self.gpu_devices}

        # Results and error handling
        self.tasks: List[Task] = []
        self.trees: List[TaskTree] = []
        self.results: List[Any] = []
        self.errors: List[Dict] = []
        self.lock = threading.Lock()

        # Memory tracking
        self.memory_history: List[Dict] = []
        self._memory_history_size = kwargs.pop("memory_history_size", 100)
        self.current_memory_state = {"CPU": MemoryState.HEALTHY, "GPU": MemoryState.HEALTHY}

        # Memory tracking
        self.initial_memory = self.memory_monitor.current_memory["used"]
        self.initial_gpu_memory = self.memory_monitor.current_gpu_memory

        self.peak_memory = {"CPU": self.initial_memory, "CPU_TIMESTAMP": datetime.now()}
        for device, stats in self.initial_gpu_memory.items():
            # Initialize max GPU memory for each device
            self.peak_memory[f"GPU_{device}"] = stats["used"]
            self.peak_memory[f"GPU_{device}_TIMESTAMP"] = datetime.now()

        # Register signal handlers
        signal.signal(signal.SIGTERM, self._handle_keyboard_interrupt)
        signal.signal(signal.SIGINT, self._handle_keyboard_interrupt)

        # Optional: Jupyter notebook interrupt handling
        try:
            get_ipython  # Check if running in Jupyter
            from ipykernel.kernelbase import Kernel
            Kernel.raw_interrupt_handler = self._jupyter_interrupt_handler
        except (NameError, ImportError):
            pass

        # Abrupt stop flag
        self._abrupt_stop_requested = threading.Event()

        self.logger.debug(f"{self.memory_monitor.log_memory_usage}")
        self.memory_history.append(
            {
                "timestamp": datetime.now(),
                "memory": self.memory_monitor.current_memory["used"],
                "gpu_memory": [
                    device["used"]
                    for _, device in self.memory_monitor.current_gpu_memory.items()
                ],
                "event": "Initialization",
            }
        )

        self._start_queue()
        self.logger.debug("QueueManager Initialization complete")

    def _start_queue(self):
        self.cpu_threads = []
        for _ in range(self.total_cpu_worker):
            t = threading.Thread(target=self._cpu_worker, daemon=True)
            t.start()
            self.cpu_threads.append(t)

        self.gpu_threads = []
        for device in self.gpu_devices:
            t = threading.Thread(target=self._gpu_worker, args=(device,), daemon=True)
            t.start()
            self.gpu_threads.append(t)

    def __exit__(self, exc_type, exc_val, _):
        """Context manager exit."""
        self.stop_processing()
        if exc_type:
            self.logger.error(f"Error during execution: {exc_val}")
            return False
        return True

    ######### Add task #########
    def add_task(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        priority: Priority = Priority.MEDIUM,
        gpu: bool = False,
        device: int = None,
        task_name: str = None,
    ) -> str:
        """
        Add a task to the processing queue with specified execution parameters.

        Allows flexible task submission with fine-grained control over:
        - Execution function
        - Arguments
        - Priority
        - Processing target (CPU/GPU)

        Args:
            func (Callable): Function to be executed
            args (tuple, optional): Positional arguments for the function
            kwargs (dict, optional): Keyword arguments for the function
            priority (Priority, optional): Task priority level. Defaults to MEDIUM.
            gpu (bool, optional): Execute on GPU. Defaults to False.
            device (int, optional): Specific GPU device. Defaults to 0.
            task_name (str, optional): Descriptive task name

        Returns:
            str: Unique task identifier for tracking

        Example:
            >>> queue.add_task(
            ...     process_data,
            ...     args=(dataset,),
            ...     priority=Priority.HIGH,
            ...     gpu=True
            ... )
        """

        if self._abrupt_stop_requested.is_set():
            self.logger.warning("Cannot add task. Abrupt stop is active.")
            return None

        # Generate a unique task ID
        task_id = f"t{next(self._id_counter)}"

        # Ensure kwargs is a dictionary
        kwargs = kwargs or {}

        # Create the task
        task = Task(
            func=func,
            args=args,
            kwargs=kwargs,
            id=task_id,
            task_name=task_name or func.__name__,
            priority=priority,
            starttime=datetime.now(),
            endtime=None,
            gpu=gpu,
            device=device,  
        )

        task.set_time_priority()
        # Add task to the task list
        self.tasks.append(task)

        # Put the task in the appropriate queue
        self.add_to_queue(task, None)

        self.logger.info(
            f"Added {'GPU' if gpu else 'CPU'} task {task.task_name} (id: {task_id}) with priority {priority}"
        )
        time.sleep(0.1)

        self.log_memory_stats(f"Task {task.task_name}(id: {task.id}) submitted")
        return task_id

    def add_tree(self, tree: TaskTree):
        """Add a TaskTree to the forest."""
        self.trees.append(tree)
        self.logger.info(f"Added tree {tree.id} with {len(tree.tasks)} branches")
        first_task = tree.tasks[0]
        first_task.set_time_priority()
        
        self.tasks.append(first_task)
        
        self.add_to_queue(first_task, tree)

        self.logger.info(
            f"Added {'GPU' if first_task.gpu else 'CPU'} task {first_task.task_name} (id: {first_task.id}) with priority {first_task.priority}"
        )
        self.log_memory_stats(f"Tree {tree.id} submitted")
        time.sleep(0.1)

    def _move_to_next_task(self, tree: TaskTree, cls: Any=None) -> None:
        """Queue the next task of a tree for processing."""

        tree.advance()

        if tree.is_complete():
            return

        task = tree.get_next_task()
        
        if task is None:
            return

        if cls is not None:
            if type(cls).__name__ == type(task.cls).__name__:
                task.cls = cls
        
        task.set_time_priority()

        self.tasks.append(task)

        self.add_to_queue(task, tree)

        self.logger.info(
            f"Added {'GPU' if task.gpu else 'CPU'} task {task.task_name} (id: {task.id}) with priority {task.priority}"
        )
        self.log_memory_stats(f"Task {task.task_name}(id: {task.id}) submitted")
        time.sleep(0.1)
        
    def add_to_queue(self, task, tree=None):
        """Add a task to the appropriate queue."""
        if task.gpu:
            if task.device is None:
                task.device = self._choose_gpu_device()
            self.gpu_queue[task.device].put((task, tree))
        else:
            if task.priority == Priority.HIGH:
                self.cpu_high_priority_queue.put((task, tree))
            else:
                self.cpu_queue.put((task, tree))

    ######### Task processing #########
    def _cpu_worker(self):
        """Distribute tasks based on priority and available resources."""
        while True:
            try:
                self._check_abrupt_stop()
                self.manage_memory_state()
                
                while (self.current_memory_state["CPU"] != MemoryState.EMERGENCY):
                    
                    try:
                        try:
                            task, tree = self.cpu_high_priority_queue.get(timeout=0.2)
                        except:
                            task, tree = self.cpu_queue.get(timeout=0.2)
                            
                        # Execute the task
                        try:
                            task.status = "processing"
                            task.execute()
                            self._task_callback(task, tree)
                        except Exception as e:
                            self.logger.error(f"Error processing task {task.id}: {e}")
                            task.status = "failed"
                            task.error = e
                            self.errors.append(e)
                        finally:
                            if isinstance(tree, TaskTree) and task.status == "completed":
                                self._move_to_next_task(tree, task.cls)
                            self.logger.debug(self.log_detailed_memory_report())
                    except queue.Empty:
                        continue
                    
            except AbruptStopException:
                self.logger.info("CPU worker stopped by abrupt stop.")
                break
            except Exception as e:
                self.logger.error(f"Error in CPU worker: {e}")

            time.sleep(0.1)

    def _task_callback(self, task: Task, tree: TaskTree = None):
        """Callback function for task completion."""
        try:
            with self.lock:
                for submitted_task in self.tasks:
                    if submitted_task.id == task.id:
                        submitted_task.status = "completed"
                        submitted_task.result = task.result
                        submitted_task.endtime = datetime.now()
                        submitted_task.error = None
                        if tree is None:
                            submitted_task.cleanup()
                            task.cleanup()
                        self.logger.info(f"{'GPU' if task.gpu else 'CPU'} task {task.task_name} (id: {task.id}) completed")
                        return submitted_task
                else:
                    # If no matching task is found, log a warning
                    self.logger.warning(
                        f"No matching task found for task_id: {task.id}"
                    )

                # Append result to results list
                self.results.append(task.result)

        except Exception as e:
            # Log and track any errors during task callback
            error_info = {"task_id": task.id, "error": e, "timestamp": datetime.now()}
            self.logger.error(f"Error in task callback for task {task.id}: {e}")
            self.errors.append(error_info)

    @contextmanager
    def gpu_context(self, device: int = None):
        """Context manager for safe GPU operations."""
        if device is None:
            device = self._choose_gpu_device()
            self.logger.info(f"Using GPU device {device}")
        try:
            with cp.cuda.Device(device):
                yield
        except Exception as e:
            self.logger.error(f"GPU operation failed on device {device}: {e}")
            raise

    def _choose_gpu_device(self):
        """Choose a GPU device to use."""
        import numpy as np

        gpu_memory = self.memory_monitor.current_gpu_memory_percent
        if gpu_memory[self._gpu_device] > 90:
            self._gpu_device += 1
        self._gpu_device += 1
        self._gpu_device = self._gpu_device%2
        self.logger.info(f"Selected GPU device {self._gpu_device}")
        return self._gpu_device
    
    def _gpu_worker(self, device: int):
        cp.cuda.Device(device).use()
        while True:
            try:
                self._check_abrupt_stop()
                self.manage_memory_state()
                
                while (self.current_memory_state["GPU"] != MemoryState.EMERGENCY) and (not self.gpu_queue[device].empty()):
                    task, tree = self.gpu_queue[device].get()
                    try:
                        task.status = "processing"
                        task.starttime = datetime.now()
                        with self.gpu_context(device):
                            updated_task = task.execute()
                        self._task_callback(updated_task, tree)
                        task.status = "completed"
                        task.endtime = datetime.now()
                        cp.get_default_memory_pool().free_all_blocks()
                    except Exception as e:
                        task.endtime = datetime.now()
                        task.status = "failed"
                        task.error = e
                        self.logger.error(f"GPU task {task.id} failed on device {device}: {e}")
                    finally:
                        self.gpu_queue[device].task_done()
                        # if isinstance(tree, TaskTree) and task.status == "completed":
                        #     self._move_to_next_task(tree, task.cls)
                        # self.logger.debug(self.log_detailed_memory_report())
                        if isinstance(tree, TaskTree) and task.status == "completed":
                            self._move_to_next_task(tree, updated_task.cls)
                        
                        self.log_memory_stats(f"GPU task {task.id} completed on device {device}")
            except AbruptStopException:
                self.logger.info(f"GPU worker process for device {device} stopped.")
                break
            except Exception as e:
                self.logger.error(f"Error in GPU worker process for device {device}: {e}")
            time.sleep(0.1)

    def wait_until_task_complete(
        self, task_id: Union[str, List[str]], timeout: Optional[float] = None
    ):
        """
        Wait until the specified task(s) complete or until timeout.

        Args:
            task_id: A single task ID, list of task IDs, or "all" to wait for all tasks
            timeout: Optional timeout in seconds

        Returns:
            True if all tasks completed, False if timeout occurred
        """
        start_time = time.time()

        # Special case for "all" keyword
        if task_id == "all":
            task_id = [task.id for task in self.tasks]
            tree_id = [tree.id for tree in self.trees]
        
        # Convert single task_id to a list for consistent handling
        if isinstance(task_id, str):
            if "task" in task_id:
                task_id = [task_id]
                tree_id = []
            elif "tree" in task_id:
                task_id = []
                tree_id = [task_id]

        # If no tasks to wait for, return immediately
        if not task_id and not tree_id:
            return True

        while task_id or tree_id:
            # Check if timeout has occurred
            if timeout is not None and time.time() - start_time > timeout:
                return False

            # Find tasks that are not yet completed
            remaining_tasks =[
                tid for tid in task_id
                if next((task for task in self.tasks if task.id == tid and task.status != "completed"), None) is not None
            ]

            remaining_trees = [
                tid for tid in tree_id 
                if next((tree for tree in self.trees if tree.id == tid and tree.status != "completed"), None) is not None
            ]
            
            # If all tasks are completed, exit
            if not remaining_tasks and not remaining_trees:
                return True
                
            # Small sleep to prevent tight loop
            if task_id == "all":
                time.sleep(10)
            else:
                time.sleep(1)

        return True

    def stop_processing(self, *args):
        """
        Gracefully stop all task processing.

        Can be used as a signal handler or manually called to halt processing.

        Args:
            *args: Signal arguments (ignored)

        Notes:
            - Closes and terminates all CPU process pool
            - Clears pending tasks in CPU and GPU queues
            - Logs memory usage and shutdown details
            - Provides a clean shutdown mechanism for the task processing system

        Raises:
            Exception: If an error occurs during the shutdown process
        """
        try:
            while not self.cpu_queue.empty():
                self.cpu_queue.get()
                self.cpu_queue.task_done()

            while not self.cpu_high_priority_queue.empty():
                self.cpu_high_priority_queue.get()
                self.cpu_high_priority_queue.task_done()


            for gpu_queue in self.gpu_queue:
                while not gpu_queue.empty():
                    gpu_queue.get()
                    gpu_queue.task_done()

            # Log shutdown details
            self.logger.info("Task processing stopped")
            self.log_memory_stats("Shutdown")

        except Exception as e:
            self.logger.error(f"Error during task processing shutdown: {e}")

    def _check_abrupt_stop(self):
        """
        Check if abrupt stop has been requested.
        
        Raises AbruptStopException if abrupt stop is active.
        Provides a mechanism to gracefully exit long-running tasks.
        """
        if self._abrupt_stop_requested.is_set():
            # Minimal logging for interruption
            self.logger.debug("Task interrupted by abrupt stop mechanism.")
            raise AbruptStopException("Task processing stopped by abrupt stop mechanism.")
    
    def abrupt_stop(self):
        if self._abrupt_stop_requested.is_set():
            os._exit(0)
            return
        self._abrupt_stop_requested.set()
        self.logger.warning("Abrupt stop initiated. Clearing task queues and stopping processes...")
        try:
            while not self.cpu_task.empty():
                self.cpu_task.get()
                self.cpu_task.task_done()
            for gpu_task in self.gpu_tasks:
                while not gpu_task.empty():
                    gpu_task.get()
                    gpu_task.task_done()
            self.cpu_pool.terminate()
            self.cpu_pool.join()

        except Exception as e:
            self.logger.error(f"Error during abrupt stop: {e}")
        finally:
            self._abrupt_stop_requested.clear()
            self.logger.info("Abrupt stop completed. All tasks cleared.")
            os._exit(0)


    def _handle_keyboard_interrupt(self, signum, frame):
        """Handle keyboard interrupt with abrupt stop mechanism."""
        self.logger.warning("Keyboard interrupt detected. Initiating abrupt stop...")
        self.abrupt_stop()

    def _jupyter_interrupt_handler(self, kernel, signum, frame):
        """Custom interrupt handler for Jupyter notebook."""
        self.logger.warning("Jupyter notebook interrupt detected. Initiating abrupt stop...")
        self.abrupt_stop()
        raise KeyboardInterrupt()

    ######### Memory related #########
    def manage_memory_state(self) -> None:
        """Manage memory state considering both CPU and GPU memory."""

        new_state = self.memory_monitor.get_unified_state()

        if new_state["CPU"] != self.current_memory_state["CPU"]:
            self.logger.info(
                f"Memory state changed from {self.current_memory_state['CPU'].state} to "
                f"{new_state['CPU'].state} (triggered by CPU)"
            )
            self.logger.warning(f"{self.memory_monitor.log_memory_usage}")
            self.current_memory_state["CPU"] = new_state["CPU"]

            if self.current_memory_state["CPU"] != MemoryState.HEALTHY:
                self.memory_monitor.handle_state(
                    trigger_source="CPU",
                    gpu_context=self.gpu_context,
                    stop_callback=self.stop_processing,
                )

        if new_state["GPU"] != self.current_memory_state["GPU"]:
            self.logger.info(
                f"Memory state changed from {self.current_memory_state['GPU'].state} to "
                f"{new_state['GPU'].state} (triggered by GPU)"
            )
            self.logger.warning(f"{self.memory_monitor.log_memory_usage}")
            self.current_memory_state["GPU"] = new_state["GPU"]

            if self.current_memory_state["GPU"] != MemoryState.HEALTHY:
                self.memory_monitor.handle_state(
                    trigger_source="GPU",
                    gpu_context=self.gpu_context,
                    stop_callback=self.stop_processing,
                )

    def log_detailed_memory_report(self):
        """Log memory usage report with multiprocessing."""
        # Detailed report for debug level
        log_message = ["Detailed memory usage report:"]
        log_message.extend(
            [
                "System Memory:",
                f"  Initial: {self.initial_memory:.1f} MB",
                f"  Peak: {self.peak_memory['CPU']:.1f} MB ({self.peak_memory['CPU_TIMESTAMP']})",
                f"  Current: {self.memory_monitor.current_memory['used']:.1f} MB ({self.memory_monitor.current_memory_percent:.1f}%)",
                f"  Total: {self.memory_monitor.current_memory['total']:.1f} MB",
            ]
        )

        gpu_stats = (
            self.memory_monitor.current_gpu_memory
        )  # GPU stats still in main process
        if gpu_stats:
            log_message.append("\nGPU Memory:")
            for device, stats in gpu_stats.items():
                log_message.extend(
                    [
                        f"  {device}:",
                        f"    Initial: {self.initial_gpu_memory[device]['used']:.1f} MB",
                        f"    Peak: {self.peak_memory[f'GPU_{device}']:.1f} MB ({self.peak_memory[f'GPU_{device}_TIMESTAMP']})",
                        f"    Current: {stats['used']:.1f} MB ({stats['percent']:.1f} %)",
                        f"    Total: {stats['total']:.1f} MB",
                    ]
                )

        self.logger.debug("\n".join(log_message))

    def log_memory_stats(self, stage: str = None) -> float:
        """
        Update memory statistics and optionally log the current stage.

        Args:
            stage: Optional description of the current processing stage

        Returns:
            float: Current memory usage in MB
        """
        current_memory = self.memory_monitor.current_memory["used"]
        if current_memory > self.peak_memory["CPU"]:
            self.peak_memory["CPU"] = current_memory
            self.peak_memory["CPU_TIMESTAMP"] = datetime.now()

        for device, stats in self.memory_monitor.current_gpu_memory.items():
            if stats["used"] > self.peak_memory[f"GPU_{device}"]:
                self.peak_memory[f"GPU_{device}"] = stats["used"]
                self.peak_memory[f"GPU_{device}_TIMESTAMP"] = datetime.now()

        if stage and self.logger:
            self.logger.debug(f"Memory at {stage}: {current_memory:.2f} MB")

        self.memory_history.append(
            {
                "timestamp": datetime.now(),
                "memory": current_memory,
                "gpu_memory": [
                    gpu["used"]
                    for _, gpu in self.memory_monitor.current_gpu_memory.items()
                ],
                "event": stage,
            }
        )

        if len(self.memory_history) > self._memory_history_size:
            self.memory_history.pop(0)

        self.logger.debug(f"Memory at {stage}: {self.memory_monitor.log_memory_usage}")

    def plot_memory_history(self):
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        # Extract data
        data = self.memory_history

        timestamps = [entry["timestamp"] for entry in data]
        memory_usage = [entry["memory"] for entry in data]
        plt.plot(timestamps, memory_usage, label="CPU Memory (MB)", marker="o")
        for i in range(len(data[0]["gpu_memory"])):
            gpu_memory = [entry["gpu_memory"][i] for entry in data]
            plt.plot(
                timestamps, gpu_memory, label=f"GPU device_{i} Memory (MB)", marker="s"
            )

        # Format timestamps on x-axis
        plt.gca().xaxis.set_major_formatter(
            mdates.DateFormatter("%H:%M:%S")
        )  # Show HH:MM:SS format
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())  # Auto spacing

        # Formatting
        plt.xlabel("Timestamp", fontsize=13)
        plt.ylabel("Memory Usage (MB)", fontsize=13)
        plt.title("Memory Usage Over Time", fontsize=13)
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True)

        # Show the plot
        plt.show()
