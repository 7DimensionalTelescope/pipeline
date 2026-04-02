import logging
import os
import re
import threading
from collections import defaultdict
from pathlib import Path

from ..const.observation import available_7dt_units
from ..path.utils import get_nightdate

logger = logging.getLogger(__name__)

_UNIT_IN_PATH_RE = re.compile(r"7DT\d{2}")


def unit_from_transfer_entry(path_or_name: str | Path) -> str | None:
    """Return 7DTxx from a transfer-history path/filename, or None if not present."""
    m = _UNIT_IN_PATH_RE.search(os.fspath(path_or_name))
    return m.group(0) if m else None


def normalize_raw_dirname(path_or_name: str | Path) -> str | None:
    raw = os.fspath(path_or_name).rstrip(os.sep)
    if not raw:
        return None

    candidate = os.path.basename(raw)
    if candidate.lower().endswith(".fits"):
        candidate = os.path.basename(os.path.dirname(raw))

    return candidate or None


def normalize_batch_key(path_or_name: str | Path) -> str | None:
    dirname = normalize_raw_dirname(path_or_name)
    if dirname is None:
        return None

    normalized = dirname[:-4] if dirname.endswith(".tar") else dirname

    if get_nightdate(normalized, use_dirname=False) is None:
        return None

    return normalized


def batch_family_key(path_or_name: str | Path) -> str | None:
    batch_key = normalize_batch_key(path_or_name)
    if batch_key is None:
        return None
    if batch_key.endswith("_ToO"):
        return batch_key[:-4]
    return batch_key


class TransferHistoryIndex:
    """Cache expected raw-data batches from the transfer history file."""

    def __init__(self, history_path: str | Path, rawdata_dir: str | Path):
        self.history_path = Path(history_path)
        self.rawdata_dir = Path(rawdata_dir)
        self._history_mtime_ns: int | None = None
        self._expected_dirs_by_nightdate: dict[str, frozenset[tuple[str, str]]] = {}
        self._indexed_nightdates: tuple[str, ...] = ()
        self._refresh(force=True)

    def _refresh(self, force: bool = False) -> None:
        try:
            if not self.history_path.exists():
                self._history_mtime_ns = None
                self._expected_dirs_by_nightdate = {}
                self._indexed_nightdates = ()
                return

            stat = self.history_path.stat()
            if not force and self._history_mtime_ns == stat.st_mtime_ns:
                return

            expected_dirs_by_nightdate = defaultdict(set)
            with self.history_path.open() as stream:
                for raw_line in stream:
                    if "|" not in raw_line:
                        continue

                    columns = [part.strip() for part in raw_line.strip().split("|")[1:-1]]
                    if not columns or columns[0] == "Filename":
                        continue

                    batch_dirname = normalize_batch_key(columns[0])
                    nightdate = get_nightdate(columns[0], use_dirname=False)
                    if batch_dirname and nightdate:
                        unit_in_entry = unit_from_transfer_entry(columns[0])
                        if unit_in_entry:
                            expected_dirs_by_nightdate[nightdate].add((unit_in_entry, batch_dirname))
                        else:
                            for unit in available_7dt_units:
                                expected_dirs_by_nightdate[nightdate].add((unit, batch_dirname))

            self._history_mtime_ns = stat.st_mtime_ns
            self._expected_dirs_by_nightdate = {
                nightdate: frozenset(dirnames) for nightdate, dirnames in expected_dirs_by_nightdate.items()
            }
            self._indexed_nightdates = tuple(sorted(self._expected_dirs_by_nightdate))
        except Exception as exc:
            logger.warning("Failed to read transfer history %s: %s", self.history_path, exc)
            self._history_mtime_ns = None
            self._expected_dirs_by_nightdate = {}
            self._indexed_nightdates = ()

    def _nightdate_in_range(self, nightdate: str) -> bool:
        if not self._indexed_nightdates:
            return False
        return self._indexed_nightdates[0] <= nightdate <= self._indexed_nightdates[-1]

    def _path_keys(self, path: str | Path) -> set[tuple[str, str]]:
        """
        Convert a raw dir or FITS path to the possible `(unit, batch_dirname)`
        keys used by the transfer-history lock.

        `_ToO` is retained in the exact key, but we also include its base batch
        family so `_ToO` arrivals can still be locked by a base history entry.
        """
        try:
            relative_path = Path(path).relative_to(self.rawdata_dir)
        except ValueError:
            return set()

        if len(relative_path.parts) < 2:
            return set()

        unit, dirname = relative_path.parts[:2]
        exact_key = normalize_batch_key(dirname)
        if exact_key is None:
            return set()

        keys = {(unit, exact_key)}
        family_key = batch_family_key(dirname)
        if family_key is not None:
            keys.add((unit, family_key))
        return keys

    def _present_batch_keys(
        self, expected_dirs: set[tuple[str, str]]
    ) -> tuple[set[tuple[str, str]], set[Path], set[Path]]:
        """
        Check only the exact history-expected unit/batch directories.

        This avoids broad scans under `RAWDATA_DIR` and separately tracks the
        optional `_ToO` sibling directories so release can include them.
        """
        present_dirs = set()
        release_dirs = set()
        too_dirs = set()

        for unit, batch_dirname in expected_dirs:
            base_dir = self.rawdata_dir / unit / batch_dirname
            if not base_dir.is_dir():
                continue

            present_dirs.add((unit, batch_dirname))
            release_dirs.add(base_dir)

            too_dir = self.rawdata_dir / unit / f"{batch_dirname}_ToO"
            if too_dir.is_dir():
                too_dirs.add(too_dir)

        return present_dirs, release_dirs, too_dirs

    def batch_files(self, nightdate: str) -> list[str]:
        expected_dirs = self._expected_dirs_by_nightdate.get(nightdate)
        if not expected_dirs:
            return []

        _, release_dirs, too_dirs = self._present_batch_keys(set(expected_dirs))
        files = set()

        for directory in sorted(release_dirs | too_dirs, key=str):
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        if entry.is_file() and entry.name.endswith(".fits"):
                            files.add(entry.path)
            except FileNotFoundError:
                continue

        return sorted(files)

    def should_lock_for(self, path: str | Path) -> bool:
        self._refresh()

        if not self.history_path.exists():
            print("history not exists")
            return False

        path_keys = self._path_keys(path)
        nightdate = get_nightdate(path)
        if not path_keys or nightdate is None:
            print("path_keys or nightdate is None")
            return False

        if not self._nightdate_in_range(nightdate):
            print("nightdate not in range")
            return False

        expected_dirs = self._expected_dirs_by_nightdate.get(nightdate)
        if not expected_dirs or not (path_keys & expected_dirs):
            print("expected_dirs or path_keys not in expected_dirs")
            return False

        return True

    def batch_status(self, nightdate: str) -> tuple[bool, set[tuple[str, str]], set[tuple[str, str]]]:
        """
        Return whether a nightdate is ready to release.

        Missing or out-of-range transfer history intentionally returns ready=True
        so the trigger falls back to the old behavior instead of blocking forever.
        """
        self._refresh()

        if not self.history_path.exists() or not self._nightdate_in_range(nightdate):
            return True, set(), set()

        expected_dirs = self._expected_dirs_by_nightdate.get(nightdate)
        if not expected_dirs:
            return True, set(), set()

        present_dirs, _, _ = self._present_batch_keys(set(expected_dirs))
        missing_dirs = set(expected_dirs) - present_dirs
        return len(missing_dirs) == 0, set(expected_dirs), missing_dirs


class BatchTriggerLock:
    """Hold trigger callbacks until a night batch is complete."""

    def __init__(
        self,
        callback,
        *,
        rawdata_dir: str | Path,
        history_path: str | Path,
        poll_seconds: int = 15,
    ):
        self.callback = callback
        self.history = TransferHistoryIndex(history_path=history_path, rawdata_dir=rawdata_dir)
        self.poll_seconds = poll_seconds
        self._pending_by_nightdate: dict[str, set[str]] = defaultdict(set)
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()
        self._dispatch_lock = threading.Lock()

    def __call__(self, image_paths):
        immediate_paths = []
        locked_paths = defaultdict(list)

        for image_path in image_paths:

            if self.history.should_lock_for(image_path):
                nightdate = get_nightdate(image_path)
                if nightdate is not None:
                    locked_paths[nightdate].append(image_path)
                    continue
            immediate_paths.append(image_path)

        for nightdate, paths in locked_paths.items():
            immediate_paths.extend(self._enqueue_or_release(nightdate, paths))

        self._dispatch(immediate_paths)

    def _dispatch(self, image_paths):
        if not image_paths:
            return

        with self._dispatch_lock:
            self.callback(sorted(set(image_paths)))

    def _drain_pending(self, nightdate: str) -> list[str]:
        with self._lock:
            pending_paths = sorted(self._pending_by_nightdate.pop(nightdate, set()))
            timer = self._timers.pop(nightdate, None)

        if timer is not None:
            timer.cancel()

        return pending_paths

    def _release_batch(self, nightdate: str) -> list[str]:
        pending_paths = set(self._drain_pending(nightdate))
        pending_paths.update(self.history.batch_files(nightdate))
        return sorted(pending_paths)

    def _enqueue_or_release(self, nightdate: str, image_paths: list[str]) -> list[str]:
        with self._lock:
            self._pending_by_nightdate[nightdate].update(image_paths)
            pending_count = len(self._pending_by_nightdate[nightdate])

        ready, expected_dirs, missing_dirs = self.history.batch_status(nightdate)
        if ready:
            return self._release_batch(nightdate)

        self._ensure_poll_timer(nightdate)
        logger.info(
            "Locking %d files for %s until all batch directories arrive (%d/%d present; missing=%s)",
            pending_count,
            nightdate,
            len(expected_dirs) - len(missing_dirs),
            len(expected_dirs),
            sorted(missing_dirs),
        )
        return []

    def _ensure_poll_timer(self, nightdate: str) -> None:
        with self._lock:
            if nightdate in self._timers:
                return

            timer = threading.Timer(self.poll_seconds, self._poll_pending_nightdate, args=(nightdate,))
            timer.daemon = True
            self._timers[nightdate] = timer
            timer.start()

    def _poll_pending_nightdate(self, nightdate: str) -> None:
        with self._lock:
            has_pending = nightdate in self._pending_by_nightdate

        if not has_pending:
            with self._lock:
                self._timers.pop(nightdate, None)
            return

        ready, expected_dirs, missing_dirs = self.history.batch_status(nightdate)
        if ready:
            pending_paths = self._release_batch(nightdate)
            if pending_paths:
                logger.info("Batch trigger lock released %d files for %s", len(pending_paths), nightdate)
                self._dispatch(pending_paths)
            return

        logger.info(
            "Still locking trigger for %s (%d/%d present; missing=%s)",
            nightdate,
            len(expected_dirs) - len(missing_dirs),
            len(expected_dirs),
            sorted(missing_dirs),
        )

        with self._lock:
            if nightdate not in self._pending_by_nightdate:
                self._timers.pop(nightdate, None)
                return

            timer = threading.Timer(self.poll_seconds, self._poll_pending_nightdate, args=(nightdate,))
            timer.daemon = True
            self._timers[nightdate] = timer
            timer.start()
