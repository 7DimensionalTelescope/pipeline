"""
File descriptor diagnostics.

Designed to make leaks easy to spot in long-running loops:

    - get_fd_info(detailed=True) -> dict with per-kind breakdown and
      most-duplicated file targets (the typical "logger leak" signal).
    - log_fd_info(...) -> formatted single-line log; auto-promotes to
      WARNING when usage crosses ``warn_threshold``.
    - FDTracker -> checkpoint-based delta reporter that highlights newly
      opened targets between calls, so 1000 iterations of "302/1024" become
      "+0 / +0 / +2 (new: 2x .../foo.log) / ..." instead.

All Linux-specific paths are guarded with /proc fallbacks via psutil.
"""

from __future__ import annotations

import os
import threading
import time
from collections import Counter
from typing import Optional

import psutil
import resource


_FD_DIR = "/proc/self/fd"


def _read_fd_targets() -> dict[str, str]:
    """Return ``{fd_id: target}`` for all FDs of the current process.

    On Linux this reads ``/proc/self/fd``; targets look like::

        /abs/path/to/file        # regular file
        socket:[12345]           # socket
        pipe:[67890]             # pipe
        anon_inode:[eventpoll]   # epoll / eventfd / timerfd / inotify ...
        /dev/null, /dev/pts/3    # char/special device

    Falls back to ``psutil`` on platforms without ``/proc``.
    FDs that disappear mid-scan are silently skipped.
    """
    targets: dict[str, str] = {}

    if os.path.isdir(_FD_DIR):
        try:
            entries = os.listdir(_FD_DIR)
        except OSError:
            entries = []
        for fd in entries:
            try:
                targets[fd] = os.readlink(os.path.join(_FD_DIR, fd))
            except OSError:
                # FD closed between listdir and readlink; skip.
                continue
        return targets

    # Non-Linux fallback (macOS / Windows).
    try:
        proc = psutil.Process()
        for f in proc.open_files():
            targets[str(f.fd)] = f.path
        try:
            for c in proc.net_connections(kind="all"):
                if c.fd != -1:
                    targets[str(c.fd)] = f"socket:[{c.laddr}->{c.raddr}]"
        except (psutil.AccessDenied, AttributeError):
            pass
    except Exception:
        pass

    return targets


def _classify(target: str) -> str:
    """Bucket an FD target string into a short category name."""
    if target.startswith("socket:"):
        return "socket"
    if target.startswith("pipe:"):
        return "pipe"
    if target.startswith("anon_inode:"):
        return "anon_inode"
    if target.startswith("/dev/"):
        return "device"
    return "file"


def _get_limits() -> tuple[int, int]:
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        return soft, hard
    except (OSError, AttributeError):
        return -1, -1


def _fast_count() -> int:
    """Cheap FD count (just ``listdir``, no symlink reads)."""
    try:
        return len(os.listdir(_FD_DIR))
    except (OSError, FileNotFoundError):
        try:
            return len(psutil.Process().open_files())
        except Exception:
            return -1


def get_fd_info(detailed: bool = False, top_n: int = 5) -> dict:
    """
    Gather FD usage info.

    Args:
        detailed: when True, also include ``by_kind`` (count per category)
            and ``top_targets`` (file paths opened more than once, useful
            for spotting duplicate opens of the same log file). This costs
            one ``readlink`` per FD; cheap unless FDs are in the thousands.
        top_n: cap on number of duplicated file targets to report.

    Returns:
        dict with keys: ``current``, ``soft_limit``, ``hard_limit``,
        ``percent_used``, ``pid``; plus ``by_kind`` and ``top_targets``
        when ``detailed=True``.
    """
    soft_limit, hard_limit = _get_limits()

    if detailed:
        targets = _read_fd_targets()
        fd_count = len(targets) if targets else -1
    else:
        fd_count = _fast_count()
        targets = None

    percent_used = (
        (fd_count / soft_limit * 100) if (fd_count > 0 and soft_limit > 0) else -1
    )

    info: dict = {
        "current": fd_count,
        "soft_limit": soft_limit,
        "hard_limit": hard_limit,
        "percent_used": percent_used,
        "pid": os.getpid(),
    }

    if detailed and targets:
        kinds = Counter(_classify(t) for t in targets.values())
        info["by_kind"] = dict(kinds)

        # Most-duplicated regular files - the strongest leak signal.
        file_targets = (t for t in targets.values() if _classify(t) == "file")
        most_common = Counter(file_targets).most_common(top_n)
        info["top_targets"] = [(p, c) for p, c in most_common if c > 1]

    return info


def _format_msg(info: dict, prefix: str = "", delta: Optional[int] = None) -> str:
    """Compose a one-line summary from an ``info`` dict (with optional delta)."""
    parts = [
        f"{prefix}FD: {info['current']}/{info['soft_limit']} "
        f"({info['percent_used']:.1f}%) [PID: {info['pid']}]"
    ]
    if delta is not None:
        parts[0] += f" Δ={delta:+d}"

    by_kind = info.get("by_kind")
    if by_kind:
        parts.append("by_kind: " + ", ".join(f"{k}={v}" for k, v in sorted(by_kind.items())))

    top = info.get("top_targets")
    if top:
        parts.append("dup_files: " + "; ".join(f"{c}x {p}" for p, c in top))

    new_targets = info.get("new_targets")
    if new_targets:
        parts.append("new: " + "; ".join(f"{c}x {p}" for p, c in new_targets))

    return " | ".join(parts)


def log_fd_info(
    logger=None,
    prefix: str = "",
    detailed: bool = True,
    warn_threshold: float = 80.0,
) -> dict:
    """
    Log FD usage in one structured line.

    Args:
        logger: object with ``.info`` / ``.warning``. Falls back to ``print``.
        prefix: optional message prefix (e.g. ``"[create_config] "``).
        detailed: include per-kind breakdown and duplicate-file report.
        warn_threshold: percent_used at or above which the message is
            emitted at WARNING level.

    Returns:
        The info dict (so callers can branch on it).
    """
    info = get_fd_info(detailed=detailed)
    msg = _format_msg(info, prefix=prefix)

    is_warn = info["percent_used"] >= warn_threshold

    if logger is not None:
        if is_warn and hasattr(logger, "warning"):
            logger.warning(msg)
        else:
            logger.info(msg)
    else:
        print(msg)

    return info


class FDTracker:
    """
    Track FD usage across checkpoints to expose leaks in long-running loops.

    Example:

        tracker = FDTracker(label="start")
        for i, item in enumerate(items):
            do_work(item)
            if i % 10 == 0:
                tracker.checkpoint(f"after {i+1} items")

    Each ``checkpoint`` call logs:
        - current count and percent of soft limit
        - delta since the previous checkpoint (``Δ=+N``)
        - newly opened targets aggregated by path (``new: 2x /path/foo.log``)
        - per-kind breakdown when ``detailed=True``
        - escalates to WARNING when delta is large or usage is high.
    """

    def __init__(self, label: str = "init", detailed: bool = True):
        self._snapshot = _read_fd_targets()
        self._last_count = len(self._snapshot)
        self._label = label
        self._detailed = detailed

    @property
    def count(self) -> int:
        return self._last_count

    def checkpoint(
        self,
        label: str,
        logger=None,
        warn_threshold: float = 80.0,
        delta_warn: int = 50,
    ) -> dict:
        """Log delta since previous checkpoint and return the info dict."""
        new_snap = _read_fd_targets()
        new_count = len(new_snap)
        delta = new_count - self._last_count

        info = get_fd_info(detailed=self._detailed)
        info["delta"] = delta
        info["delta_since"] = self._label

        if delta > 0:
            prev = Counter(self._snapshot.values())
            curr = Counter(new_snap.values())
            opened = (curr - prev).most_common(5)
            if opened:
                info["new_targets"] = opened

        prefix = f"[{label} <- {self._label}] "
        msg = _format_msg(info, prefix=prefix, delta=delta)

        is_warn = (
            info["percent_used"] >= warn_threshold or delta >= delta_warn
        )
        if logger is not None:
            if is_warn and hasattr(logger, "warning"):
                logger.warning(msg)
            else:
                logger.info(msg)
        else:
            print(msg)

        self._snapshot = new_snap
        self._last_count = new_count
        self._label = label
        return info


class PeakFDSampler:
    """
    Background-thread sampler that captures peak FD usage.

    Synchronous probes (``log_fd_info``, ``FDTracker.checkpoint``) only fire
    when the main thread reaches them, so they routinely miss the peak when
    work happens concurrently (e.g. under ``ThreadPoolExecutor``). This
    sampler runs in its own daemon thread, polling at a configurable
    interval and recording the maximum count it observes — plus a target
    snapshot taken at the moment the peak is set.

    Use as a context manager:

        with PeakFDSampler(interval=0.05) as sampler:
            with ThreadPoolExecutor(max_workers=30) as ex:
                ...

        sampler.report(logger=mylog)   # peak count, peak %, top dup files,
                                       # by_kind at peak, time-to-peak

    Notes:
        - Polling cost is one ``listdir(/proc/self/fd)`` per tick (cheap).
        - When a new peak is set, one full target snapshot is captured for
          diagnostics; this is gated by ``min_peak_delta`` to avoid
          re-snapshotting on tiny bumps.
        - Safe in threaded code; ``os.listdir`` does not contend with
          ``open()`` in worker threads.
        - Must be used in the same process as the work (forked children
          have their own FD tables).
    """

    def __init__(
        self,
        interval: float = 0.05,
        capture_targets_on_peak: bool = True,
        min_peak_delta: int = 5,
    ):
        self._interval = interval
        self._capture = capture_targets_on_peak
        self._min_delta = min_peak_delta

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Recorded state.
        self._start_time: Optional[float] = None
        self._peak_count: int = -1
        self._peak_at: Optional[float] = None
        self._peak_targets: Optional[dict[str, str]] = None
        self._sample_count: int = 0

    # ---- lifecycle -----------------------------------------------------

    def __enter__(self) -> "PeakFDSampler":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._start_time = time.time()
        self._peak_count = _fast_count()
        self._peak_at = self._start_time
        if self._capture:
            self._peak_targets = _read_fd_targets()
        self._thread = threading.Thread(
            target=self._run, name="PeakFDSampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=max(self._interval * 4, 0.5))
        self._thread = None

    # ---- sampling loop -------------------------------------------------

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            count = _fast_count()
            self._sample_count += 1
            if count > self._peak_count + self._min_delta:
                # Capture diagnostics first so the snapshot is taken near
                # the moment we observed the new peak.
                if self._capture:
                    try:
                        self._peak_targets = _read_fd_targets()
                    except Exception:
                        pass
                self._peak_count = count
                self._peak_at = time.time()
            elif count > self._peak_count:
                # Update count without re-snapshotting (cheap path).
                self._peak_count = count
                self._peak_at = time.time()

    # ---- query / report ------------------------------------------------

    @property
    def peak_count(self) -> int:
        return self._peak_count

    @property
    def peak_targets(self) -> Optional[dict[str, str]]:
        return self._peak_targets

    def peak_info(self) -> dict:
        """Build an info-dict in the same shape as ``get_fd_info``."""
        soft, hard = _get_limits()
        percent = (
            (self._peak_count / soft * 100)
            if (self._peak_count > 0 and soft > 0)
            else -1
        )
        info: dict = {
            "current": self._peak_count,
            "soft_limit": soft,
            "hard_limit": hard,
            "percent_used": percent,
            "pid": os.getpid(),
            "samples": self._sample_count,
            "elapsed_s": (
                (self._peak_at or 0) - (self._start_time or 0)
                if self._start_time
                else -1
            ),
        }
        targets = self._peak_targets
        if targets:
            kinds = Counter(_classify(t) for t in targets.values())
            info["by_kind"] = dict(kinds)
            file_targets = (t for t in targets.values() if _classify(t) == "file")
            most_common = Counter(file_targets).most_common(5)
            info["top_targets"] = [(p, c) for p, c in most_common if c > 1]
        return info

    def report(
        self,
        logger=None,
        prefix: str = "",
        warn_threshold: float = 80.0,
    ) -> dict:
        """Log a one-line peak summary and return the info dict."""
        info = self.peak_info()
        elapsed = info.get("elapsed_s", -1)
        suffix = (
            f" (peak at +{elapsed:.2f}s, {info['samples']} samples)"
            if elapsed and elapsed >= 0
            else ""
        )
        msg = _format_msg(info, prefix=f"{prefix}PEAK ") + suffix

        is_warn = info["percent_used"] >= warn_threshold

        if logger is not None:
            if is_warn and hasattr(logger, "warning"):
                logger.warning(msg)
            else:
                logger.info(msg)
        else:
            print(msg)
        return info
