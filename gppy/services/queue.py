#from multiprocessing.pool import Pool
from tkinter.constants import TRUE
import multiprocess as mp
import queue
from queue import PriorityQueue
import threading

import cupy as cp

import time
import signal
import os

from typing import Callable, Any, Optional, List, Dict, Union
from datetime import datetime
from .logger import Logger
from .task import Task, Priority


from contextlib import contextmanager

import itertools

from .memory import MemoryState, MemoryMonitor
from ..utils import time_diff_in_seconds

signal.signal(signal.SIGINT, signal.SIG_IGN)

mp.set_start_method("spawn", force=True)

class AbruptStopException(Exception):
    """Custom exception to signal abrupt stop processing."""
    pass


class QueueManager:

    _id_counter = itertools.count(1)

    def __init__(
        self,
        max_workers: Optional[int] = None,
        logger: Optional[Logger] = None,
        print_debug: bool = False,
        save_result: bool = False,
        auto_start: bool = True,
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
        self.total_cpu_worker = max_workers or mp.cpu_count() - 30

        # Create CPU queues (thread-safe)
        self.cpu_queue = PriorityQueue()  # Priority queue for CPU tasks
        
        # Create GPU queues (one per device, thread-safe)
        self.gpu_devices = range(cp.cuda.runtime.getDeviceCount())
        self.gpu_queue = {device: queue.Queue() for device in self.gpu_devices}
        self.gpu_high_priority_queue = {device: queue.Queue() for device in self.gpu_devices}
        self._remaind_gpu_queue = 0 
        self._gpu_device = 0
        
        # Process pool for CPU tasks
        self.cpu_pool = mp.Pool(processes=self.total_cpu_worker)

        # Results and error handling
        self.tasks: List[Task] = []
        self.streams: List[Taskstream] = []
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
        # Abrupt stop flag
        self._abrupt_stop_requested = mp.Event()
        if auto_start:
            self._start_workers()

        self.print_debug = print_debug
        self.save_result = save_result
        self.logger.debug("QueueManager Initialization complete")
    
    def _start_workers(self):
        # Initialize task tracking
        self._stop_event = threading.Event()
        
        # Start the task distributor thread
        self.cpu_thread = threading.Thread(
            target=self._cpu_worker,
            daemon=True
        )
        self.cpu_thread.start()

        # Start GPU workers
        self.gpu_threads = []
        for device in self.gpu_devices:
            t = threading.Thread(
                target=self._gpu_worker, 
                args=(device,),
                daemon=True
            )
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
        inherit_input: bool = False,
    ) -> str:
        """
        Add a task to the processing queue with specified execution parameters.

        Allows flexible task submission with fine-grained control over:
        - Execution function
        - Arguments
        - Priority
        - Processing target (CPU/GPU)

        Args:
nherit_input             func (Callable): Function to be executed
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
        
        if isinstance(func, Task):
            task = func
            self.tasks.append(task)
            self.add_to_queue(task, None)
            return task.id

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
            inherit_input=inherit_input,
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

    def add_stream(self, stream):
        """Add a ReductionStream to the forest."""
        self.streams.append(stream)
        self.logger.info(f"Added stream {stream.id}")

        task = stream.get_task()
        task.set_time_priority()
        # Add task to the task list
        self.tasks.append(task)

        # Put the task in the appropriate queue
        self.add_to_queue(task, stream)

        self.logger.info(
            f"Added {'GPU' if task.gpu else 'CPU'} task {task.task_name} (id: {task.id}) with priority {task.priority}"
        )
        self.log_memory_stats(f"Stream {stream.id} submitted")
        time.sleep(0.1)

    def _move_to_next_task(self, stream) -> None:
        """Queue the next task of a stream for processing."""
        task = stream.get_task()

        if stream.is_complete():
            return 
        
        task.set_time_priority()

        self.tasks.append(task)

        self.add_to_queue(task, stream)

        self.logger.info(
            f"Added {'GPU' if task.gpu else 'CPU'} task {task.task_name} (id: {task.id}) with priority {task.priority}"
        )
        self.log_memory_stats(f"Task {task.task_name}(id: {task.id}) submitted")
        time.sleep(0.1)
        
    def add_to_queue(self, task, stream=None):
        """Add a task to the appropriate queue."""
        if task.gpu:
            if self._remaind_gpu_queue >= (len(self.gpu_devices)):
                task.gpu=False
                task._device = "CPU"
                task.kwargs["use_gpu"] = False
                self.cpu_queue.put((task.priority, task, stream))
                return
            if task.device is None:
                task._device = self._choose_gpu_device()
            if task.priority == Priority.HIGH:
                self.gpu_high_priority_queue[task.device].put((task, stream))
            else:
                self.gpu_queue[task.device].put((task, stream))
            with self.lock:
                self._remaind_gpu_queue += 1
        else:
            self.cpu_queue.put((task.priority, task, stream))

    ######### Task processing #########
    def _cpu_worker(self):
        """Distribute CPU tasks to the process pool."""

        def task_wrapper(task, stream):
            try:
                result = task.execute()
                return task, stream, result, None
            except Exception as e:
                return task, stream, None, str(e)

        def callback(result_tuple):
            task, stream, result, error = result_tuple
            self._handle_completed_task(task, stream, result, error)

        def errback(error):
            self._handle_completed_task(task, stream, None, str(error))

        while not self._abrupt_stop_requested.is_set():
            try:
                self._check_abrupt_stop()
                self.manage_memory_state(process_type="CPU")
                
                # First check high priority queue
                try:
                    # Unpack the priority queue item (priority, task, stream)
                    _, task, stream = self.cpu_queue.get(timeout=0.2)
                except queue.Empty:
                    task, stream = None, None
          
                if task is None:
                    time.sleep(0.2)
                    continue
                    
                task.status = "processing"
                task.starttime = datetime.now()
                
                try:
                    async_result = self.cpu_pool.apply_async(
                        task_wrapper, args=(task, stream), callback=callback, error_callback=errback
                    )
                except Exception as e:
                    import traceback
                    self.logger.error(f"Error in CPU worker: {e}")
                    errback(e)
                    time.sleep(0.5)
                    traceback.print_exc()
                    raise
            
            except AbruptStopException:
                self.logger.info(f"CPU worker process stopped.")
                break

            except Exception as e:
                self.logger.error(f"Error in CPU worker: {e}")
                time.sleep(0.5)
                raise
                

        self.logger.info("CPU worker thread exiting")


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
        gpu_memory = self.memory_monitor.current_gpu_memory_percent
        if self._gpu_device > len(gpu_memory):
            self._gpu_device = 0
        else:
            if gpu_memory[self._gpu_device] > 90:
                self._gpu_device += 1
            self._gpu_device += 1
            self._gpu_device = self._gpu_device%2
            self.logger.info(f"Selected GPU device {self._gpu_device}")
        return self._gpu_device

    def _gpu_worker(self, device: int):
        while not self._abrupt_stop_requested.is_set():
            try:
                self._check_abrupt_stop()
                self.manage_memory_state(process_type="GPU")
                
                if self.current_memory_state["GPU"] == MemoryState.EMERGENCY:
                    time.sleep(0.5)  # Wait during memory emergency
                    continue
                
                for task_queue in [self.gpu_high_priority_queue[device], self.gpu_queue[device]]:
                    try:
                        task, stream = task_queue.get(timeout=0.2)
                        with self.lock:
                            self._remaind_gpu_queue -= 1
                        break
                    except queue.Empty:
                        task, stream = None, None
                        continue
                
                if task is None:
                    time.sleep(0.2)
                    continue
                
                task.status = "processing"
                task.starttime = datetime.now()
                try:
                    with self.gpu_context(device):
                        result = task.execute()
                        self._handle_completed_task(task, stream, result, None)

                except Exception as e:
                    self._handle_completed_task(task, stream, None, str(e))

            except AbruptStopException:
                self.logger.info(f"GPU worker process for device {device} stopped.")
                break
            except Exception as e:
                self.logger.error(f"Error in GPU worker process for device {device}: {e}")
                time.sleep(0.5)
                raise

    def _handle_completed_task(self, task, stream, result, error):
        """Handle a completed task from either pool or GPU."""
        with self.lock:
            for submitted_task in self.tasks:
                if submitted_task.id == task.id:
                    if error:
                        submitted_task.status = "failed"
                    else:
                        submitted_task.status = "completed"
                        
                        if self.save_result:
                            submitted_task.result = result
                            self.results.append(result)
                        else:
                            submitted_task.result = None
                            result = None

                    task.endtime = datetime.now()
                    submitted_task.endtime = datetime.now()
                    submitted_task.error = error

                    task.cleanup()
                    submitted_task.cleanup()
                    
                    if stream is not None and task.error is None:
                        self._move_to_next_task(stream)    
                    break
            self.logger.info(f"Completed task {task.task_name}(id: {task.id}) in {time_diff_in_seconds(task.starttime, task.endtime)} seconds")
            self.log_memory_stats(f"Task {task.task_name}(id: {task.id}) completed")

            if self.print_debug:
                self.logger.debug(self.log_detailed_memory_report())
                self.print_the_number_of_processes()
                
    def print_the_number_of_processes(self):
        """Print the number of processes in each queue."""
        self.logger.info(
            f"CPU Queue: {self.cpu_queue.qsize()} tasks"
        )

        for device in self.gpu_devices:
            self.logger.info(
                f"GPU device {device}: {self.gpu_queue[device].qsize()} tasks"
            )
        

        n_processing = len([t for t in self.tasks if t.status == 'processing'])
        n_completed = len([t for t in self.tasks if t.status == 'completed'])
        n_failed = len([t for t in self.tasks if t.status == 'failed'])
        self.logger.info(f"Processing: {n_processing}, Completed: {n_completed}, Failed: {n_failed}")


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
            self.logger.info("Initiating graceful shutdown...")
            
            # Signal all threads to stop
            self._abrupt_stop_requested.set()
            
            # Clear CPU queues
            while not self.cpu_queue.empty():
                try:
                    self.cpu_queue.get_nowait()
                except queue.Empty:
                    break

            # Clear GPU queues
            for device, gpu_queue in self.gpu_queue.items():
                while not gpu_queue.empty():
                    try:
                        gpu_queue.get_nowait()
                    except queue.Empty:
                        break

            # Wait for pool tasks to complete
            self.logger.info("Waiting for process pool tasks to complete...")
            
            # Terminate the pool
            if hasattr(self, 'cpu_pool') and self.cpu_pool:
                self.cpu_pool.close()
                self.cpu_pool.terminate()
                self.cpu_pool.join()
                self.logger.info("Process pool terminated")

            # Log shutdown details
            self.logger.info("Task processing stopped")
            self.log_memory_stats("Shutdown")

        except Exception as e:
            self.logger.error(f"Error during task processing shutdown: {e}")
            raise

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
        """Immediately stop all processing and exit."""
        if self._abrupt_stop_requested.is_set():
            os._exit(0)
            return

        self._abrupt_stop_requested.set()
        self.logger.warning("Abrupt stop initiated. Terminating all processes...")
        
        try:
            # Clear CPU queues
            while not self.cpu_queue.empty():
                try:
                    self.cpu_queue.get_nowait()
                except queue.Empty:
                    break

            # Clear GPU queues
            for device in self.gpu_devices:
                while not self.gpu_queue[device].empty():
                    try:
                        self.gpu_queue[device].get_nowait()
                    except queue.Empty:
                        break

                while not self.gpu_high_priority_queue[device].empty():
                    try:
                        self.gpu_high_priority_queue[device].get_nowait()
                    except queue.Empty:
                        break

            # Terminate the process pool
            if hasattr(self, 'cpu_pool') and self.cpu_pool:
                self.cpu_pool.terminate()
                self.cpu_pool.join()

        except Exception as e:
            self.logger.error(f"Error during abrupt stop: {e}")
            raise
        finally:
            self.logger.info("Abrupt stop completed. Terminating process.")
            os._exit(0)

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

        if isinstance(task_id, str):
            if task_id == "all":
                task_ids = [task.id for task in self.tasks]
                stream_ids = [stream.id for stream in self.streams]
            elif task_id.startswith("stream"):
                task_ids = []
                stream_ids = [task_id]
            elif task_id.startswith("t"):
                task_ids = [task_id]
                stream_ids = []
            else:
                task_ids = [task_id]
                stream_ids = []
        elif isinstance(task_id, list):
            task_ids = [tid for tid in task_id if tid.startswith("t")]
            stream_ids = [tid for tid in task_id if tid.startswith("stream")]
        else:
            return True  # Invalid input

        while task_ids or stream_ids:
            if timeout is not None and time.time() - start_time > timeout:
                return False

            task_ids = [tid for tid in task_ids if any(task.id == tid and task.status != "completed" for task in self.tasks)]
            stream_ids = [tid for tid in stream_ids if any(stream.id == tid and not stream.is_complete() for stream in self.streams)]

            if not task_ids and not stream_ids:
                return True

            time.sleep(1)

        return True


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
    def manage_memory_state(self, process_type: str = "CPU") -> None:
        """Manage memory state considering both CPU and GPU memory."""

        new_state = self.memory_monitor.get_unified_state()

        if new_state[process_type] != self.current_memory_state[process_type]:
            self.logger.info(
                f"Memory state changed from {self.current_memory_state[process_type].state} to "
                f"{new_state[process_type].state} (triggered by {process_type})"
            )
            self.logger.warning(f"{self.memory_monitor.log_memory_usage}")
            self.current_memory_state[process_type] = new_state[process_type]

            if self.current_memory_state[process_type] != MemoryState.HEALTHY:
                self.memory_monitor.handle_state(
                    trigger_source=process_type,
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