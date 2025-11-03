import multiprocess as mp
import queue
import threading
import time
import signal
import os
from typing import Callable, Any, Optional, List, Dict, Union
from datetime import datetime
import itertools
import subprocess
from .memory import MemoryMonitor

from .logger import Logger
from .task import Task, Priority
from ..utils import time_diff_in_seconds


signal.signal(signal.SIGINT, signal.SIG_IGN)
mp.set_start_method("spawn", force=True)


class AbruptStopException(Exception):
    """
    Custom exception to signal abrupt stop processing.

    Raised when the queue manager needs to immediately halt all processing
    without waiting for graceful completion of tasks.
    """

    pass


class QueueManager:
    """
    Advanced task queue manager for parallel processing with priority scheduling.

    This class provides a comprehensive task management system that supports
    both CPU-based task processing and subprocess scheduling. It includes
    priority-based task scheduling, error handling, and graceful shutdown
    capabilities.

    Features:
    - Priority-based task scheduling (HIGH, MEDIUM, LOW, PREPROCESS)
    - Multi-process CPU task execution
    - Subprocess scheduling and monitoring
    - Comprehensive error handling and recovery
    - Graceful and abrupt shutdown mechanisms
    - Real-time task status tracking
    - Jupyter notebook compatibility

    Args:
        max_workers (int, optional): Maximum number of worker processes (default: 10)
        logger (Logger, optional): Custom logger instance
        save_result (bool): Whether to save task results (default: False)
        auto_start (bool): Whether to start workers immediately (default: False)
        **kwargs: Additional configuration options

    Example:
        >>> queue = QueueManager(max_workers=8, save_result=True)
        >>> task_id = queue.add_task(
        ...     process_data,
        ...     args=(dataset,),
        ...     priority=Priority.HIGH
        ... )
        >>> queue.wait_until_task_complete(task_id)
    """

    _id_counter = itertools.count(1)

    def __init__(
        self,
        max_workers: Optional[int] = None,
        logger: Optional[Logger] = None,
        save_result: bool = False,
        auto_start: bool = False,
        process_timeout: int = 7200,  # 2 hours default timeout
        **kwargs,
    ):
        # Initialize logging
        if logger:
            self.logger = logger
        else:
            self.logger = Logger("QueueManager")
            self.logger.set_output_file(
                f"/var/log/pipeline/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{os.getpid()}.log"
            )

        self.logger.debug(f"Initialize QueueManager.")

        # Default CPU allocation
        self.total_cpu_worker = max_workers or 10

        self.lock = threading.Lock()

        # Register signal handlers
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, self._handle_keyboard_interrupt)
            signal.signal(signal.SIGINT, self._handle_keyboard_interrupt)

        # Optional: Jupyter notebook interrupt handling
        try:
            get_ipython()  # Check if running in Jupyter
            from ipykernel.kernelbase import Kernel

            Kernel.raw_interrupt_handler = self._jupyter_interrupt_handler
        except (NameError, ImportError):
            pass

        # Abrupt stop flag
        self._abrupt_stop_requested = mp.Event()
        if auto_start:
            self._start_workers()
        else:
            self.ptype = None

        self.save_result = save_result
        self.process_timeout = process_timeout
        self.logger.debug("QueueManager Initialization complete")

    def _start_workers(self, process_type="scheduler"):
        """
        Initialize and start worker threads based on process type.

        Sets up the appropriate worker infrastructure for either task processing
        or subprocess scheduling.

        Args:
            process_type (str): Type of processing ("task" or "scheduler")
        """
        # Initialize task tracking
        self._stop_event = threading.Event()

        if process_type == "task":
            # Create CPU queues (thread-safe)
            self.processing_queue = queue.PriorityQueue()  # Priority queue for CPU tasks
            self.completion_queue = queue.Queue()

            # Process pool for CPU tasks
            self.pool = mp.Pool(processes=self.total_cpu_worker)

            # Results and error handling
            self.tasks: List[Task] = []
            self.results: List[Any] = []
            self.errors: List[Dict] = []

            # Start the task completion handler thread
            self.processing_thread = threading.Thread(target=self._task_worker, daemon=True)
            self.processing_thread.start()
            self.completion_thread = threading.Thread(target=self._task_completion_worker, daemon=True)
            self.completion_thread.start()
            self.ptype = "task"
        elif process_type == "scheduler":
            # Start the task scheduler thread
            self.processing_thread = threading.Thread(target=self._scheduler_worker, daemon=True)
            self.processing_thread.start()
            self.completion_thread = threading.Thread(target=self._scheduler_completion_worker, daemon=True)
            self.completion_thread.start()
            # Start health monitoring thread
            self.health_thread = threading.Thread(target=self._monitor_process_health, daemon=True)
            self.health_thread.start()
            self.ptype = "scheduler"
        else:
            self.ptype = None

    def __exit__(self, exc_type, exc_val, _):
        """
        Context manager exit method.

        Ensures proper cleanup when used as a context manager.

        Args:
            exc_type: Exception type if any
            exc_val: Exception value if any
            _: Traceback (ignored)

        Returns:
            bool: False if exception occurred, True otherwise
        """
        self.stop_processing()
        if exc_type:
            self.logger.error(f"Error during execution: {exc_val}", exc_info=True)
            return False
        return True

    ######### Add task #########
    def add_task(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        priority: Priority = Priority.MEDIUM,
        task_name: str = None,
    ) -> str:
        """
        Add a task to the processing queue with specified execution parameters.

        Allows flexible task submission with fine-grained control over:
        - Execution function
        - Arguments
        - Priority
        - Processing target (CPU/GPU)
        - Asynchronous submission

        Args:
            func (Callable): Function to be executed
            args (tuple, optional): Positional arguments for the function
            kwargs (dict, optional): Keyword arguments for the function
            priority (Priority, optional): Task priority level. Defaults to MEDIUM.
            task_name (str, optional): Descriptive task name

        Returns:
            str: Unique task identifier for tracking

        Example:
            >>> queue.add_task(
            ...     process_data,
            ...     args=(dataset,),
            ...     priority=Priority.HIGH,
            ...     async_submit=True
            ... )
        """

        if not (hasattr(self, "processing_thread")):
            self._start_workers(process_type="task")

        if self._abrupt_stop_requested.is_set():
            self.logger.warning("Cannot add task. Abrupt stop is active.")
            return None

        if isinstance(func, Task):
            task = func
            self.tasks.append(task)
            task_id = task.id
        else:
            # Generate a unique task ID
            task_id = f"task_{next(self._id_counter)}"

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
            )

            # Process synchronously
            self.tasks.append(task)

        self.processing_queue.put((task.priority, task))

        self.logger.info(f"Added task {task.task_name} (id: {task_id}) with priority {priority}")
        time.sleep(0.1)

        return task_id

    def add_scheduler(self, scheduler):
        """
        Add a scheduler for subprocess management.

        Args:
            scheduler: Scheduler instance for managing subprocess tasks
        """
        self.scheduler = scheduler
        self._active_processes = []
        if not (hasattr(self, "processing_thread")):
            self._start_workers(process_type="scheduler")

    ######### Task processing #########
    def _task_worker(self):
        """
        Distribute CPU tasks to the process pool.

        This worker thread continuously processes tasks from the priority queue,
        submitting them to the process pool for execution and handling results
        through the completion queue.
        """

        def task_wrapper(task):
            """
            Wrapper function for task execution in process pool.

            Args:
                task: Task instance to execute

            Returns:
                tuple: (task, result, error) where error is None if successful
            """
            try:
                result = task.execute()
                return task, result, None
            except Exception as e:
                return task, None, str(e)

        def callback(result_tuple):
            """
            Callback for successful task completion.

            Args:
                result_tuple: Tuple containing (task, result, error)
            """
            task, result, error = result_tuple
            self.completion_queue.put((task, result, error))

        def errback(error):
            """
            Error callback for failed task execution.

            Args:
                error: Exception that occurred during task execution
            """
            self.completion_queue.put((task, None, str(error)))

        while not self._abrupt_stop_requested.is_set():
            try:
                self._check_abrupt_stop()

                # First check high priority queue
                try:
                    # Unpack the priority queue item (priority, task)
                    _, task = self.processing_queue.get(timeout=0.2)
                except queue.Empty:
                    task = None

                if task is None:
                    time.sleep(0.2)
                    continue

                task.status = "processing"
                task.starttime = datetime.now()

                try:
                    async_result = self.pool.apply_async(
                        task_wrapper, args=(task,), callback=callback, error_callback=errback
                    )
                except Exception as e:
                    import traceback

                    self.logger.error(f"Error in processing worker: {e}")
                    errback(e)
                    time.sleep(0.5)
                    traceback.print_exc()
                    raise

            except AbruptStopException:
                self.logger.info(f"Processing worker stopped.")
                break

            except Exception as e:
                self.logger.error(f"Error in processing worker: {e}")
                time.sleep(0.5)
                raise

    def _task_completion_worker(self):
        """
        Handle task completion and result processing.

        This worker thread processes completed tasks from the completion queue,
        updates task status, and stores results if requested.
        """
        while not self._stop_event.is_set():
            try:
                task, result, error = self.completion_queue.get()
                with self.lock:
                    for submitted_task in self.tasks:
                        if submitted_task.id == task.id:
                            if error:
                                submitted_task.status = "failed"
                                task.status = "failed"
                            else:
                                submitted_task.status = "completed"
                                task.status = "completed"

                                if self.save_result:
                                    submitted_task.result = result
                                    self.results.append(result)
                                else:
                                    submitted_task.result = None
                                    result = None

                            task.endtime = datetime.now()
                            submitted_task.endtime = datetime.now()
                            submitted_task.error = error

                            self.logger.info(
                                f"Completed task {task.task_name} (id: {task.id}) in {time_diff_in_seconds(task.starttime, task.endtime)} seconds"
                            )
                            break
            except Exception as e:
                self.logger.error(f"Error in completion worker: {e}")
                time.sleep(0.2)
                continue

    def _create_subprocess(self, cmd):
        """
        Create subprocess with proper configuration and error handling.

        Args:
            cmd (list): Command to execute

        Returns:
            subprocess.Popen: Configured subprocess instance

        Raises:
            RuntimeError: If process fails to start
        """
        try:
            proc = subprocess.Popen(
                cmd,
                start_new_session=True,  # Ensure clean process group
            )

            # Verify process actually started
            time.sleep(0.1)
            if proc.poll() is not None:
                # Process died immediately
                stdout, stderr = proc.communicate()
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise RuntimeError(f"Process failed to start: {error_msg}")

            return proc

        except Exception as e:
            self.logger.error(f"Failed to create subprocess for {cmd}: {e}")
            raise

    def _scheduler_worker(self):
        """
        Worker thread for subprocess scheduling.

        Manages the execution of subprocess tasks by retrieving commands
        from the scheduler and launching them as separate processes.
        """

        while not self._abrupt_stop_requested.is_set():
            try:
                self._check_abrupt_stop()

                with self.lock:  # Thread-safe access to active processes
                    if len(self._active_processes) > self.total_cpu_worker:
                        time.sleep(1)
                        continue

                try:
                    cmd = self.scheduler.get_next_task()
                except:
                    time.sleep(1)
                    continue

                if cmd is None:
                    time.sleep(1)
                    continue

                try:
                    proc = self._create_subprocess(cmd)

                    # Extract config path from command for tracking
                    config = cmd[cmd.index("-config") + 1] if "-config" in cmd else "unknown"

                    with self.lock:  # Thread-safe modification
                        self._active_processes.append([config, proc, time.time()])

                    self.logger.info(f"Process with {os.path.basename(config)} (PID = {proc.pid}) submitted.")
                    time.sleep(0.5)

                except Exception as e:
                    import traceback

                    self.logger.error(f"Error in processing worker: {e}")
                    time.sleep(0.5)
                    traceback.print_exc()
                    # Don't raise - continue processing other tasks
                    continue

            except AbruptStopException:
                self.logger.info(f"Processing worker stopped.")
                break

            except Exception as e:
                self.logger.error(f"Error in processing worker: {e}")
                time.sleep(0.5)
                # Don't raise - continue processing
                continue

    def _scheduler_completion_worker(self):
        """
        Worker thread for monitoring subprocess completion.

        Continuously checks the status of active subprocesses and updates
        the scheduler when processes complete. Includes timeout handling
        and proper cleanup of failed processes.
        """
        while not self._stop_event.is_set():
            try:
                current_time = time.time()
                processes_to_remove = []

                with self.lock:  # Thread-safe access
                    for process in list(self._active_processes):  # (config_path, proc, start_time)
                        config, proc, start_time = process

                        if proc.poll() is not None:  # Process finished
                            pid = proc.pid
                            success = proc.returncode == 0

                            # Get process output for logging
                            try:
                                stdout, stderr = proc.communicate(timeout=1)
                                stdout_str = stdout.decode() if stdout else ""
                                stderr_str = stderr.decode() if stderr else ""
                            except subprocess.TimeoutExpired:
                                stdout_str = stderr_str = "Output collection timed out"
                            except Exception:
                                stdout_str = stderr_str = "Could not collect output"

                            if success:
                                self.logger.info(f"Process with {config} (PID = {pid}) completed successfully.")
                                if stdout_str.strip():
                                    self.logger.debug(f"Process {config} stdout: {stdout_str[:500]}...")
                            else:
                                self.logger.error(
                                    f"Process with {os.path.basename(config)} (PID = {pid}) failed with return code {proc.returncode}."
                                )
                                if stderr_str.strip():
                                    self.logger.error(f"Process {config} stderr: {stderr_str[:500]}...")

                            self.scheduler.mark_done(config, success=success)
                            processes_to_remove.append(process)

                        else:  # Process still running
                            # Check for timeout (default: 2 hours)
                            timeout_seconds = getattr(self, "process_timeout", 7200)  # 2 hours default
                            if current_time - start_time > timeout_seconds:
                                pid = proc.pid
                                self.logger.warning(
                                    f"Process with {os.path.basename(config)} (PID = {pid}) timed out after {timeout_seconds} seconds. Killing..."
                                )

                                try:
                                    # Try graceful termination first
                                    proc.terminate()
                                    try:
                                        proc.wait(timeout=10)  # Wait 10 seconds for graceful exit
                                    except subprocess.TimeoutExpired:
                                        # Force kill if graceful termination fails
                                        self.logger.warning(f"Force killing process {pid}")
                                        proc.kill()
                                        proc.wait()  # Wait for kill to complete

                                    self.logger.error(f"Process {config} (PID = {pid}) killed due to timeout.")
                                    self.scheduler.mark_done(config, success=False)
                                    processes_to_remove.append(process)

                                except Exception as kill_error:
                                    self.logger.error(f"Failed to kill timed out process {pid}: {kill_error}")
                                    # Still mark as failed even if we can't kill it
                                    self.scheduler.mark_done(config, success=False)
                                    processes_to_remove.append(process)

                # Remove completed/timed out processes
                for process in processes_to_remove:
                    if process in self._active_processes:
                        self._active_processes.remove(process)

                time.sleep(0.5)
            except Exception as e:
                self.logger.error(f"Error in completion worker: {e}")
                time.sleep(0.2)

    def _monitor_process_health(self):
        """
        Monitor processes for health issues and resource consumption.

        This method runs in a separate thread to monitor active processes
        for potential issues like excessive memory usage or unresponsive behavior.
        """
        while not self._stop_event.is_set():
            try:
                if not hasattr(self, "_active_processes") or not self._active_processes:
                    time.sleep(300)  # Check every minute if no processes
                    continue

                current_time = time.time()
                processes_to_check = []

                with self.lock:
                    processes_to_check = list(self._active_processes)

                for process in processes_to_check:
                    config, proc, start_time = process

                    if proc.poll() is not None:
                        continue  # Process already finished

                    runtime = current_time - start_time

                    # Check for long-running processes
                    if runtime > 3600:  # 1 hour
                        self.logger.warning(f"Process {config} has been running for {runtime/3600:.1f} hours")

                    # Check for excessive memory usage (if psutil is available)
                    try:
                        import psutil

                        try:
                            process_info = psutil.Process(proc.pid)
                            memory_mb = process_info.memory_info().rss / 1024 / 1024

                            if memory_mb > 8192:  # 8GB
                                self.logger.warning(f"Process {config} using {memory_mb:.0f}MB of memory")

                            # Check CPU usage over last 5 seconds
                            cpu_percent = process_info.cpu_percent(interval=1)
                            if cpu_percent > 400:  # Very high CPU usage
                                self.logger.warning(f"Process {config} using {cpu_percent:.1f}% CPU")

                        except psutil.NoSuchProcess:
                            # Process died between checks
                            continue

                    except ImportError:
                        # psutil not available, skip resource monitoring
                        pass

                time.sleep(60)  # Check every minute

            except Exception as e:
                self.logger.error(f"Error in process health monitoring: {e}")
                time.sleep(60)

    def stop_processing(self, *args):
        """
        Gracefully stop all task processing.

        Can be used as a signal handler or manually called to halt processing.

        Args:
            *args: Signal arguments (ignored)

        Notes:
            - Closes and terminates all CPU process pool
            - Stops all worker threads (submission, completion, CPU, GPU)
            - Clears pending tasks in all queues
            - Provides a clean shutdown mechanism for the task processing system

        Raises:
            Exception: If an error occurs during the shutdown process
        """
        try:
            if hasattr(self, "_stop_event") and self._stop_event.is_set():
                self.logger.warning("Stop already in progress")
                return

            self.logger.info("Initiating graceful shutdown...")

            # Signal all threads to stop
            if hasattr(self, "_stop_event"):
                self._stop_event.set()
            self._abrupt_stop_requested.set()

            if self.ptype == "schedule":
                self.logger.info("Stopping all active subprocesses...")
                for pid, proc in list(self._active_processes.items()):
                    try:
                        if proc.poll() is None:  # Still running
                            self.logger.info(f"Terminating process PID {pid}")
                            proc.terminate()
                            try:
                                proc.wait(timeout=5)  # Wait a few seconds to exit cleanly
                                self.logger.info(f"Process PID {pid} terminated gracefully.")
                            except subprocess.TimeoutExpired:
                                self.logger.warning(f"Force killing process PID {pid}")
                                proc.kill()
                        else:
                            self.logger.info(f"Process PID {pid} already finished.")
                    except Exception as e:
                        self.logger.error(f"Failed to stop PID {pid}: {e}")
                    finally:
                        del self._active_processes[pid]

            elif self.ptype == "task":
                # Clear queues
                while not self.processing_queue.empty():
                    try:
                        self.processing_queue.get_nowait()
                    except queue.Empty:
                        break

                while not self.completion_queue.empty():
                    try:
                        self.completion_queue.get_nowait()
                    except queue.Empty:
                        break

                # Wait for pool tasks to complete
                self.logger.info("Waiting for process pool tasks to complete...")

                # Terminate the pool
                self.cpu_pool.close()
                self.cpu_pool.terminate()
                self.cpu_pool.join()
                self.logger.info("Process pool terminated")

            else:
                return

            # Wait for all threads to finish
            self.processing_thread.join(timeout=2.0)
            self.completion_thread.join(timeout=2.0)
            if hasattr(self, "health_thread"):
                self.health_thread.join(timeout=2.0)

            # Log shutdown details
            self.logger.info("All task processing stopped")

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
        """
        Immediately stop all processing and exit.

        Forces immediate termination of all processes and threads,
        then exits the current process.
        """
        if self._abrupt_stop_requested.is_set():
            os._exit(0)
            return

        self._abrupt_stop_requested.set()
        self.logger.warning("Abrupt stop initiated. Terminating all processes...")

        try:
            self.stop_processing()
        except Exception as e:
            self.logger.error(f"Error during abrupt stop: {e}")
            raise
        finally:
            self.logger.info("Abrupt stop completed. Terminating process.")
            os._exit(0)

    def wait_until_task_complete(self, task_id: Union[str, List[str]], timeout: Optional[float] = None):
        """
        Wait until the specified task(s) complete or until timeout.

        Args:
            task_id: A single task ID, list of task IDs, or "all" to wait for all tasks
            timeout: Optional timeout in seconds

        Returns:
            True if all tasks completed, False if timeout occurred
        """
        start_time = time.time()

        if self.ptype == "scheduler":
            # for scheduler
            i = 0
            while not (self.scheduler.is_all_done()):
                if i % 6 == 0:
                    self.logger.info(self.scheduler.report_number_of_tasks())
                    self.logger.info(MemoryMonitor.log_memory_usage)
                    time.sleep(60)
                i += 1

        elif self.ptype == "task":

            if isinstance(task_id, str):
                if task_id == "all":
                    task_ids = [task.id for task in self.tasks]
                elif task_id.startswith("task"):
                    task_ids = [task_id]
                else:
                    task_ids = [task_id]
            elif isinstance(task_id, list):
                task_ids = [tid for tid in task_id if tid.startswith("t")]
            else:
                return True  # Invalid input

            # for task
            while task_id:
                if timeout is not None and time.time() - start_time > timeout:
                    return False

                task_ids = [
                    tid for tid in task_ids if any(task.id == tid and task.status != "completed" for task in self.tasks)
                ]

                if not task_ids:
                    return True

                time.sleep(10)

        return True

    def _handle_keyboard_interrupt(self, signum, frame):
        """
        Handle keyboard interrupt with abrupt stop mechanism.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        self.logger.warning("Keyboard interrupt detected. Initiating abrupt stop...")
        self.abrupt_stop()

    def _jupyter_interrupt_handler(self, kernel, signum, frame):
        """
        Custom interrupt handler for Jupyter notebook.

        Args:
            kernel: Jupyter kernel instance
            signum: Signal number
            frame: Current stack frame
        """
        self.logger.warning("Jupyter notebook interrupt detected. Initiating abrupt stop...")
        self.abrupt_stop()
        raise KeyboardInterrupt()

    def get_process_status(self):
        """
        Get current status of all active processes.

        Returns:
            dict: Dictionary containing process status information
        """
        if not hasattr(self, "_active_processes"):
            return {"active_processes": 0, "processes": []}

        with self.lock:
            processes_info = []
            current_time = time.time()

            for process in self._active_processes:
                config, proc, start_time = process
                runtime = current_time - start_time

                process_info = {
                    "config": config,
                    "pid": proc.pid,
                    "start_time": start_time,
                    "runtime_seconds": runtime,
                    "runtime_hours": runtime / 3600,
                    "status": "running" if proc.poll() is None else "finished",
                }

                # Add resource usage if psutil is available
                try:
                    import psutil

                    try:
                        process_obj = psutil.Process(proc.pid)
                        process_info["memory_mb"] = process_obj.memory_info().rss / 1024 / 1024
                        process_info["cpu_percent"] = process_obj.cpu_percent()
                    except psutil.NoSuchProcess:
                        process_info["status"] = "finished"
                except ImportError:
                    pass

                processes_info.append(process_info)

        return {
            "active_processes": len(processes_info),
            "processes": processes_info,
            "max_workers": self.total_cpu_worker,
            "process_timeout": getattr(self, "process_timeout", 7200),
        }
