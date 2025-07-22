import threading
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from typing import List, Callable, Dict, Any, Tuple
from .logger import Logger
import signal

class Monitor(FileSystemEventHandler):
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
        self.logger.warning("Keyboard interrupt or termination signal detected. Stopping monitor...")
        raise KeyboardInterrupt()

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.fits'):
            with self._lock:
                self._pending_files.add(event.src_path)
                # Reset debounce timer
                if self._debounce_timer is not None:
                    self._debounce_timer.cancel()
                self._debounce_timer = threading.Timer(self.debounce_seconds, self._debounced_process)
                self._debounce_timer.daemon = True
                self._debounce_timer.start()

    def _debounced_process(self):
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
        self.callbacks.append((callback, kwargs))

    def start(self):
        self.logger.info(f"Start to monitor {str(self.base_path)}")
        observer = Observer()
        observer.schedule(self, str(self.base_path), recursive=True)
        observer_thread = threading.Thread(target=observer.start)
        observer_thread.daemon = True
        observer_thread.start()
        return observer