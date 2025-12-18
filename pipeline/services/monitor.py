import threading
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from typing import List, Callable, Dict, Any, Tuple
from .logger import Logger
import signal
from .queue import QueueManager


class Monitor(FileSystemEventHandler):
    """
    A file system monitor for detecting and processing new FITS files.

    This monitor watches a specified directory tree for new FITS files and
    triggers processing callbacks when files are detected. It includes
    debouncing functionality to batch multiple file events together.

    Features:
    - Recursive directory monitoring
    - FITS file detection and filtering
    - Debounced event processing
    - Multiple callback support
    - Thread-safe operation
    - Graceful shutdown handling

    Args:
        base_path (Path): Root directory to monitor for FITS files
        debounce_seconds (int): Time to wait before processing detected files (default: 5)

    Example:
        >>> monitor = Monitor(Path("/data/observations"), debounce_seconds=10)
        >>> monitor.add_callback(process_fits_files, output_dir="/processed")
        >>> observer = monitor.start()
        >>> # ... later ...
        >>> observer.stop()
        >>> observer.join()
    """

    def __init__(self, base_path: Path, debounce_seconds: int = 5):
        super().__init__()
        self.base_path = Path(base_path)
        self.callbacks: List[Tuple[Callable, dict]] = []
        self.logger = Logger("Monitor logger")
        self.debounce_seconds = debounce_seconds
        self._debounce_timer: threading.Timer | None = None
        self._pending_files = set()
        self._lock = threading.Lock()

        # Register signal handlers only in main thread
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, self._handle_keyboard_interrupt)
            signal.signal(signal.SIGINT, self._handle_keyboard_interrupt)

    def _handle_keyboard_interrupt(self, signum, frame):
        """
        Handle keyboard interrupt or termination signal.

        Logs the interruption and raises KeyboardInterrupt for proper cleanup.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        self.logger.warning("Keyboard interrupt or termination signal detected. Stopping monitor...")
        raise KeyboardInterrupt()

    def on_created(self, event):
        """
        Handle file creation events.

        Filters for FITS files in 7DT subdirectories and adds them to the
        pending files set with debounced processing.

        Args:
            event: File system event from watchdog
        """
        if not event.is_directory and event.src_path.endswith(".fits"):
            # Check if the file is in an immediate subdirectory starting with "7DT"
            file_path = Path(event.src_path)
            relative_path = file_path.relative_to(self.base_path)
            path_parts = relative_path.parts

            # Check if the first part (immediate subdirectory) starts with "7DT"
            if path_parts and path_parts[0].startswith("7DT"):
                with self._lock:
                    self._pending_files.add(event.src_path)
                    # Reset debounce timer
                    if self._debounce_timer is not None:
                        self._debounce_timer.cancel()
                    self._debounce_timer = threading.Timer(self.debounce_seconds, self._debounced_process)
                    self._debounce_timer.daemon = True
                    self._debounce_timer.start()

    def _debounced_process(self):
        """
        Process pending files after debounce period.

        This method is called after the debounce timer expires. It processes
        all pending files and calls registered callbacks with the file list.
        """
        with self._lock:
            files_to_process = list(self._pending_files)
            self._pending_files.clear()
            self._debounce_timer = None

        if files_to_process:
            self.logger.info(f"Found {len(files_to_process)} new .fits files")
            for callback, kwargs in self.callbacks:
                self.logger.info(f"Start to process the new {len(files_to_process)} files.")
                callback(files_to_process, **kwargs)
        else:
            self.logger.info("No .fits files to process after debounce period.")

    def add_callback(self, callback: Callable, **kwargs):
        """
        Add a callback function to be called when FITS files are detected.

        Args:
            callback (Callable): Function to call with detected files
            **kwargs: Additional keyword arguments to pass to the callback
        """
        self.callbacks.append((callback, kwargs))

    def start(self):
        """
        Start monitoring the base directory for FITS files.

        Creates and starts a file system observer that watches for new
        FITS files in the specified directory tree.

        Returns:
            Observer: The watchdog observer instance for later stopping
        """
        self.logger.info(f"Start to monitor {str(self.base_path)}")
        observer = Observer()
        observer.schedule(self, str(self.base_path), recursive=True)
        observer_thread = threading.Thread(target=observer.start)
        observer_thread.daemon = True
        observer_thread.start()
        return observer
