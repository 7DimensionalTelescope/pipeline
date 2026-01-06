import multiprocess as mp
import queue
import threading
import time
import signal
import os
import socket
from typing import Optional, Union
from datetime import datetime
import subprocess

from ..utils import time_diff_in_seconds
from ..const import QUEUE_SOCKET_PATH

from .memory import MemoryMonitor
from .logger import Logger
from .scheduler import Scheduler

signal.signal(signal.SIGINT, signal.SIG_IGN)
mp.set_start_method("spawn", force=True)

DEFAULT_MAX_WORKERS = 15
DEFAULT_WORKER_SLEEP_TIME = 0.5
DEFAULT_MONITOR_CHECK_INTERVAL = 60
DEFAULT_SOCKET_LISTENER_TIMEOUT = 1.0


class AbruptStopException(Exception):
    """
    Custom exception to signal abrupt stop processing.

    Raised when the queue manager needs to immediately halt all processing
    without waiting for graceful completion.
    """

    pass


class QueueManager:
    """
    Queue manager for subprocess scheduling.

    This class provides subprocess scheduling and monitoring with
    error handling and graceful shutdown capabilities.

    Features:
    - Subprocess scheduling and monitoring
    - Comprehensive error handling and recovery
    - Graceful and abrupt shutdown mechanisms
    - Jupyter notebook compatibility

    Args:
        max_workers (int, optional): Maximum number of worker processes (default: 10)
        logger (Logger, optional): Custom logger instance
        auto_start (bool): Whether to start workers immediately (default: False)
        **kwargs: Additional configuration options
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        logger: Optional[Logger] = None,
        auto_start: bool = True,
        monitor: bool = False,
        **kwargs,
    ):
        # Initialize logging
        if logger:
            self.logger = logger
        else:
            self.logger = Logger("QueueManager")
            if monitor:
                self.logger.set_output_file(
                    f"/var/log/pipeline/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_monitor_{os.getpid()}.log"
                )
            else:
                self.logger.set_output_file(
                    f"/var/log/pipeline/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{os.getpid()}.log"
                )

        self.logger.debug(f"Initialize QueueManager.")

        # Default CPU allocation
        self.total_cpu_worker = max_workers or DEFAULT_MAX_WORKERS

        self.lock = threading.Lock()

        self.scheduler = None
        self._active_processes = []

        # Wake event for socket-based wake mechanism
        self._wake_event = threading.Event()

        # Register signal handlers
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, self._handle_keyboard_interrupt)
            signal.signal(signal.SIGINT, self._handle_keyboard_interrupt)

        # Optional: Jupyter notebook interrupt handling
        try:
            from IPython import get_ipython

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

        # Start socket listener for wake messages
        self._start_socket_listener()

        self.logger.debug("QueueManager Initialization complete")

    def _start_workers(self, process_type="scheduler"):
        """
        Initialize and start worker threads for subprocess scheduling.

        Args:
            process_type (str): Type of processing (should be "scheduler")
        """
        self._stop_event = threading.Event()

        if process_type == "scheduler":
            # Start the scheduler thread
            self.processing_thread = threading.Thread(target=self._scheduler_worker, daemon=True)
            self.processing_thread.start()
            self.completion_thread = threading.Thread(target=self._scheduler_completion_worker, daemon=True)
            self.completion_thread.start()
            self.ptype = "scheduler"
        else:
            self.ptype = None

    def _start_socket_listener(self):
        """Start UNIX socket listener for wake messages."""
        if not QUEUE_SOCKET_PATH:
            self.logger.warning("QUEUE_SOCKET_PATH not set, socket listener disabled")
            return

        socket_dir = os.path.dirname(QUEUE_SOCKET_PATH)

        # Create socket directory if it doesn't exist
        try:
            os.makedirs(socket_dir, mode=0o755, exist_ok=True)
        except PermissionError:
            self.logger.warning(f"Cannot create socket directory {socket_dir}, socket listener may not work")
            return

        # Remove existing socket if it exists
        try:
            if os.path.exists(QUEUE_SOCKET_PATH):
                os.unlink(QUEUE_SOCKET_PATH)
        except OSError:
            pass

        # Start socket listener thread
        self.socket_listener_thread = threading.Thread(
            target=self._socket_listener, daemon=True, args=(QUEUE_SOCKET_PATH,)
        )
        self.socket_listener_thread.start()
        self.logger.debug(f"Socket listener started on {QUEUE_SOCKET_PATH}")

    def _socket_listener(self, socket_path):
        """Listen for wake messages on UNIX socket."""
        sock = None
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.bind(socket_path)
            # Set socket permissions (group readable/writable)
            try:
                os.chmod(socket_path, 0o660)
            except OSError:
                pass
            sock.listen(5)

            self.logger.debug(f"Socket listener ready on {socket_path}")

            while not self._stop_event.is_set():
                try:
                    sock.settimeout(1.0)  # Check stop event periodically
                    conn, addr = sock.accept()
                    with conn:
                        data = conn.recv(1024)
                        if data and b"wake" in data:
                            self.logger.debug("Received wake message")
                            self._wake_event.set()
                            self._wake_event.clear()  # Reset immediately for next wake
                except socket.timeout:
                    continue
                except OSError:
                    if not self._stop_event.is_set():
                        self.logger.error(f"Socket error on {socket_path}")
                    break
        except Exception as e:
            self.logger.error(f"Socket listener error: {e}")
        finally:
            if sock:
                try:
                    sock.close()
                    if os.path.exists(socket_path):
                        os.unlink(socket_path)
                except OSError:
                    pass

    def __enter__(self):
        """
        Context manager entry method.
        """
        return self

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

    def add_scheduler(self, scheduler: Scheduler):
        """
        Add a scheduler for subprocess management.

        Args:
            scheduler: Scheduler instance for managing subprocess tasks
        """
        try:
            if self.scheduler is None:
                self.scheduler = scheduler
            else:
                self.scheduler.add_schedule(scheduler)
        except Exception as e:
            self.logger.error(f"Error adding scheduler: {e}")
            raise

        if not (hasattr(self, "processing_thread")):
            self._start_workers(process_type="scheduler")

    def _create_process_from_command(self, cmd):
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
            time.sleep(DEFAULT_WORKER_SLEEP_TIME)
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

                job = None
                cmd = None
                job_index = None

                with self.lock:
                    current_usage = len(self._active_processes)

                if current_usage >= self.total_cpu_worker:
                    time.sleep(DEFAULT_WORKER_SLEEP_TIME)
                    continue
                elif self.scheduler is None or not self.scheduler.has_schedule:
                    self._wake_event.wait(timeout=DEFAULT_SOCKET_LISTENER_TIMEOUT)
                    continue
                else:
                    # Get next task and mark as Processing atomically within lock
                    job, cmd = self.scheduler.get_next_task()
                    if job is not None and cmd is not None:
                        job_index = job["index"]
                    else:
                        self._wake_event.wait(timeout=DEFAULT_SOCKET_LISTENER_TIMEOUT)
                        continue

                # Create process outside lock (this can take time)
                try:
                    proc = self._create_process_from_command(cmd)

                    # Extract config path from command for tracking
                    config = cmd[cmd.index("-config") + 1] if "-config" in cmd else "unknown"

                    with self.lock:  # Thread-safe modification
                        self._active_processes.append([config, proc, time.time(), job_index])
                    self.scheduler.set_pid(job_index, proc.pid)

                    self.logger.info(f"Process with {os.path.basename(config)} (PID = {proc.pid}) submitted.")
                    self.logger.debug(f"Command: {cmd}")
                    time.sleep(DEFAULT_WORKER_SLEEP_TIME)

                except Exception as e:
                    import traceback

                    self.logger.error(f"Error in processing worker: {e}")
                    time.sleep(DEFAULT_WORKER_SLEEP_TIME)
                    traceback.print_exc()
                    # Don't raise - continue processing other tasks
                    continue

            except AbruptStopException:
                self.logger.info(f"Processing worker stopped.")
                break

            except Exception as e:
                self.logger.error(f"Error in processing worker: {e}")
                time.sleep(DEFAULT_WORKER_SLEEP_TIME)
                # Don't raise - continue processing
                continue

    def _scheduler_completion_worker(self):
        """
        Worker thread for monitoring subprocess completion.

        Continuously checks the status of active subprocesses and updates
        the scheduler when processes complete. Includes proper cleanup of failed processes.
        """
        while not self._stop_event.is_set():
            try:
                with self.lock:  # Thread-safe access
                    for process in list(self._active_processes):  # (config_path, proc, start_time, job_index)
                        config, proc, start_time = process[:3]
                        job_index = process[3] if len(process) > 3 else None

                        if proc.poll() is not None:  # Process finished
                            pid = proc.pid
                            success = proc.returncode == 0

                            # Get process output for logging
                            try:
                                stdout, stderr = proc.communicate(timeout=1)
                                stdout_str = stdout.decode(errors="replace") if stdout else ""
                                stderr_str = stderr.decode(errors="replace") if stderr else ""
                            except subprocess.TimeoutExpired:
                                stdout_str = stderr_str = "Output collection timed out"
                            except Exception:
                                stdout_str = stderr_str = "Could not collect output"

                            if success:
                                self.logger.info(
                                    f"Process with {config} (PID = {pid}) completed successfully in {time_diff_in_seconds(start_time)} seconds."
                                )
                                if stdout_str and stdout_str.strip():
                                    self.logger.debug(f"Process {config} stdout: {stdout_str[:500]}...")
                            else:
                                self.logger.error(
                                    f"Process with {os.path.basename(config)} (PID = {pid}) failed with return code {proc.returncode}."
                                )
                                if stderr_str and stderr_str.strip():
                                    self.logger.error(f"Process {config} stderr: {stderr_str[:500]}...")

                            # Mark job as done using index
                            # Use try-finally to ensure process is removed even if mark_done fails
                            try:
                                if job_index is not None and self.scheduler is not None:
                                    self.scheduler.mark_done(job_index, success=success)
                            except Exception as e:
                                self.logger.error(f"Error marking job {job_index} as done: {e}", exc_info=True)
                            finally:
                                # Always remove process from active list, even if mark_done failed
                                if process in self._active_processes:
                                    self._active_processes.remove(process)

                time.sleep(DEFAULT_WORKER_SLEEP_TIME)
            except Exception as e:
                self.logger.error(f"Error in completion worker: {e}")
                time.sleep(DEFAULT_WORKER_SLEEP_TIME)

    def stop_processing(self, *args):
        """
        Gracefully stop all subprocess processing.

        Can be used as a signal handler or manually called to halt processing.

        Args:
            *args: Signal arguments (ignored)

        Notes:
            - Stops all worker threads
            - Provides a clean shutdown mechanism for the subprocess processing system

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

            if self.ptype == "scheduler":
                self.logger.info("Stopping all active subprocesses...")
                with self.lock:
                    active_processes = list(self._active_processes)

                for process in active_processes:
                    _, proc = process[:2]
                    pid = proc.pid
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
                        with self.lock:
                            if process in self._active_processes:
                                self._active_processes.remove(process)
            else:
                return

            # Wait for all threads to finish
            self.processing_thread.join(timeout=2.0)
            self.completion_thread.join(timeout=2.0)

            # Log shutdown details
            self.logger.info("All processing stopped")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            raise

    def _check_abrupt_stop(self):
        """
        Check if abrupt stop has been requested.

        Raises AbruptStopException if abrupt stop is active.
        Provides a mechanism to gracefully exit long-running processes.
        """
        if self._abrupt_stop_requested.is_set():
            # Minimal logging for interruption
            self.logger.debug("Processing interrupted by abrupt stop mechanism.")
            raise AbruptStopException("Processing stopped by abrupt stop mechanism.")

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

    def wait_until_task_complete(self, timeout: Optional[float] = None):
        """
        Wait until all scheduler tasks complete or until timeout.

        Args:
            timeout: Optional timeout in seconds

        Returns:
            True if all tasks completed, False if timeout occurred
        """
        start_time = time.time()

        if self.ptype == "scheduler" and self.scheduler is not None:
            # for scheduler
            i = 0
            while not (self.scheduler.is_all_done()):
                if (timeout is not None) and (time.time() - start_time > timeout):
                    return False
                if i % 6 == 0:
                    self.logger.info(f"Scheduler status: {self.scheduler.status()}")
                    self.logger.info(MemoryMonitor.log_memory_usage)
                    time.sleep(DEFAULT_MONITOR_CHECK_INTERVAL)
                else:
                    time.sleep(DEFAULT_MONITOR_CHECK_INTERVAL / 6.0)

                i += 1
            from .utils import cleanup_memory

            self.logger.info("All tasks completed")
            self.logger.info(f"Scheduler status: {self.scheduler}")
            self.logger.info(MemoryMonitor.log_memory_usage)
            cleanup_memory()

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


def clear_completed_schedules():
    """
    Clear completed schedules from the scheduler database.

    This function is designed to be called periodically (e.g., via systemd timer)
    to clean up completed jobs from the database.
    """
    from .scheduler import Scheduler
    from .logger import Logger

    logger = Logger("ClearCompletedSchedules")
    logger.info("Starting to clear completed schedules from database")

    try:
        scheduler = Scheduler(use_system_queue=True)
        before_count = len(scheduler.schedule)
        completed_count = len(scheduler.schedule[scheduler.schedule["status"] == "Completed"])

        scheduler.clear_schedule(all=False)  # Clear only completed schedules

        after_count = len(scheduler.schedule)
        cleared_count = before_count - after_count

        logger.info(f"Cleared {cleared_count} completed schedules (was {before_count}, now {after_count})")
        return cleared_count
    except Exception as e:
        logger.error(f"Error clearing completed schedules: {e}", exc_info=True)
        raise
