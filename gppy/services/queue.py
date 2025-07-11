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

from .logger import Logger
from .task import Task, Priority
from ..utils import time_diff_in_seconds
from ..const import SCRIPT_DIR

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
        save_result: bool = False,
        auto_start: bool = False,
        **kwargs,
    ):
        # Initialize logging
        if logger:
            self.logger = logger
        else:
            self.logger = Logger()

        self.logger.debug(f"Initialize QueueManager.")

        # Default CPU allocation
        self.total_cpu_worker = max_workers or 10

        self.lock = threading.Lock()

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
        self._abrupt_stop_requested = mp.Event()
        if auto_start:
            self._start_workers()
        else:
            self.ptype = None

        self.save_result = save_result
        self.logger.debug("QueueManager Initialization complete")

    def _start_workers(self, process_type="scheduler"):
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
            # Start the task schduler thread
            self.processing_thread = threading.Thread(target=self._scheduler_worker, daemon=True)
            self.processing_thread.start()
            self.completion_thread = threading.Thread(target=self._scheduler_completion_worker, daemon=True)
            self.completion_thread.start()
            self.ptype = "schedule"
        else:
            self.ptype = None

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
            gpu (bool, optional): Execute on GPU. Defaults to False.
            device (int, optional): Specific GPU device. Defaults to 0.
            task_name (str, optional): Descriptive task name
            inherit_input (bool, optional): Whether to inherit input from previous task
            async_submit (bool, optional): Whether to submit the task asynchronously

        Returns:
            str: Unique task identifier for tracking

        Example:
            >>> queue.add_task(
            ...     process_data,
            ...     args=(dataset,),
            ...     priority=Priority.HIGH,
            ...     gpu=True,
            ...     async_submit=True
            ... )
        """

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

        if not (hasattr(self, "processing_thread")):
            self._start_workers(process_type="task")

        self.logger.info(f"Added task {task.task_name} (id: {task_id}) with priority {priority}")
        time.sleep(0.1)

        return task_id

    def add_scheduler(self, scheduler):

        self.scheduler = scheduler
        self._active_processes = []
        self._device_id = 0
        if not (hasattr(self, "processing_thread")):
            self._start_workers(process_type="scheduler")

    ######### Task processing #########
    def _task_worker(self):
        """Distribute CPU tasks to the process pool."""

        def task_wrapper(task):
            try:
                result = task.execute()
                return task, result, None
            except Exception as e:
                return task, None, str(e)

        def callback(result_tuple):
            task, result, error = result_tuple
            self.completion_queue.put((task, result, error))

        def errback(error):
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
        while not self._stop_event.is_set():
            try:
                task, result, error = self.completion_queue.get()
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

                self.logger.info(
                    f"Completed task {task.task_name} (id: {task.id}) in {time_diff_in_seconds(task.starttime, task.endtime)} seconds"
                )
            except Exception as e:
                self.logger.error(f"Error in completion worker: {e}")
                time.sleep(0.2)
                continue

    def _scheduler_worker(self):

        while not self._abrupt_stop_requested.is_set():
            try:
                self._check_abrupt_stop()

                if len(self._active_processes) > self.total_cpu_worker:
                    time.sleep(1)
                    continue

                try:
                    config, ptype = self.scheduler.get_next_task()
                except:
                    time.sleep(1)
                    continue

                try:
                    if ptype == "Masterframe":
                        cmd = [
                            f"{SCRIPT_DIR}/bin/preprocess",
                            "-config",
                            config,
                            "-device",
                            str(int(self._device_id % 2)),
                            "-only_with_sci",
                        ]
                    else:
                        cmd = [f"{SCRIPT_DIR}/bin/data_reduction", "-config", config]
                    proc = subprocess.Popen(cmd)
                    self._active_processes.append([config, proc])
                    self._device_id += 1
                    self.logger.info(f"Process ({ptype}) with {os.path.basename(config)} (PID = {proc.pid}) submitted.")
                    time.sleep(0.5)

                except Exception as e:
                    import traceback

                    self.logger.error(f"Error in processing worker: {e}")
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

    def _scheduler_completion_worker(self):
        while not self._stop_event.is_set():
            try:
                for process in list(self._active_processes):  # should store proc objects, not just PIDs
                    config, proc = process
                    if proc.poll() is not None:
                        pid = proc.pid
                        if proc.returncode == 0:
                            self.logger.info(f"Process with {config} (PID = {pid}) completed.")
                            self._active_processes.remove(process)
                            # Inform the scheduler that the task is done
                            self.scheduler.mark_done(config)
                        else:
                            self.logger.error(
                                f"Process with {os.path.basename(config)} (PID = {pid}) failed with return code {proc.returncode}."
                            )
                            self._active_processes.remove(process)
                time.sleep(0.5)
            except Exception as e:
                self.logger.error(f"Error in completion worker: {e}")
                time.sleep(0.2)
                continue

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
        """Immediately stop all processing and exit."""
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

        while task_id:
            if timeout is not None and time.time() - start_time > timeout:
                return False

            task_ids = [
                tid for tid in task_ids if any(task.id == tid and task.status != "completed" for task in self.tasks)
            ]

            if not task_ids:
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
