"""Cross-process coordination for the in-memory coadd combine.

Two cooperating pieces, both living under ``const.HOST_LOCK_DIR`` (tmpfs -> cleared on
boot; a flock dies with its holder, so no stale-lock handling is needed anywhere):

- **Per-filesystem mutex**: the combine is I/O-bound on whatever filesystem holds its
  inputs, so two combines on the same filesystem only split bandwidth and break
  readahead. ``CombineSlot`` serializes them per ``st_dev`` while combines on different
  filesystems (NFS / sdc1 / data0) run concurrently.
- **Memory leases**: concurrent combines each size their strip stack from MemAvailable,
  which races. Every active combine writes a lease file with its planned bytes;
  ``active_lease_bytes`` sums the leases of still-alive PIDs so chunk sizing can
  subtract them (see calc._auto_chunk_h).
"""

import fcntl
import json
import os
import time

from ..const.environ import HOST_LOCK_DIR


def _fs_tag(path: str) -> str:
    """Human-readable, filesystem-unique tag: the mount point of the nearest existing
    ancestor ("lyman-data2", "data0", ...) -- one lock per filesystem, readable logs."""
    p = os.path.abspath(path)
    while not os.path.exists(p):
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    while not os.path.ismount(p):
        p = os.path.dirname(p)
    return p.strip("/").replace("/", "-") or "root"


def _lease_dir() -> str:
    d = os.path.join(HOST_LOCK_DIR, "leases")
    os.makedirs(d, exist_ok=True)
    return d


def active_lease_bytes(exclude_pid: int | None = None) -> int:
    """Sum of planned combine bytes leased by live processes (dead PIDs are cleaned up)."""
    total = 0
    for name in os.listdir(_lease_dir()):
        path = os.path.join(_lease_dir(), name)
        try:
            with open(path) as fp:
                lease = json.load(fp)
            pid = int(lease["pid"])
        except (OSError, ValueError, KeyError):
            continue
        if pid == exclude_pid:
            continue
        if not os.path.exists(f"/proc/{pid}"):
            try:
                os.remove(path)
            except OSError:
                pass
            continue
        total += int(lease.get("bytes", 0))
    return total


class CombineSlot:
    """Blocking flock on the input filesystem's combine slot + a memory lease.

    Usage::

        with CombineSlot(input_dir, logger=self.logger) as slot:
            chunk = _auto_chunk_h(..., reserved_bytes=slot.reserved_bytes)
            slot.lease(planned_bytes)
            ... combine ...
    """

    def __init__(self, anchor_path: str, logger=None):
        self.tag = _fs_tag(anchor_path)
        self.logger = logger
        self._fh = None
        self._lease_file = os.path.join(_lease_dir(), f"combine_{os.getpid()}.json")

    def __enter__(self):
        os.makedirs(HOST_LOCK_DIR, exist_ok=True)
        self._fh = open(os.path.join(HOST_LOCK_DIR, f"combine_{self.tag}.lock"), "w")
        try:
            fcntl.flock(self._fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return self
        except OSError:
            pass
        # blocking flock: the kernel wakes us the moment the holder releases -- no
        # polling latency. The thread only narrates the wait.
        t0 = time.time()
        if self.logger:
            self.logger.info(f"Combine slot {self.tag} is held by another process; blocking until released")
        stop = None
        if self.logger:
            import threading

            stop = threading.Event()

            def narrate():
                while not stop.wait(300.0):
                    self.logger.info(f"Combine slot {self.tag} still held; waiting ({time.time() - t0:.0f} s)")

            threading.Thread(target=narrate, daemon=True).start()
        try:
            fcntl.flock(self._fh, fcntl.LOCK_EX)
        finally:
            if stop is not None:
                stop.set()
        if self.logger:
            self.logger.info(f"Combine slot {self.tag} acquired after {time.time() - t0:.0f} s")
        return self

    @property
    def reserved_bytes(self) -> int:
        """Bytes currently leased by other live combines (for chunk sizing)."""
        return active_lease_bytes(exclude_pid=os.getpid())

    def lease(self, planned_bytes: int) -> None:
        with open(self._lease_file, "w") as fp:
            json.dump({"pid": os.getpid(), "bytes": int(planned_bytes), "tag": self.tag, "ts": time.time()}, fp)

    def __exit__(self, *exc):
        try:
            os.remove(self._lease_file)
        except OSError:
            pass
        if self._fh is not None:
            fcntl.flock(self._fh, fcntl.LOCK_UN)
            self._fh.close()
        return False


class NullSlot:
    """No-op stand-in for CombineSlot (small runs that must never queue)."""

    reserved_bytes = 0

    def __enter__(self):
        return self

    def lease(self, planned_bytes: int) -> None:
        pass

    def __exit__(self, *exc):
        return False
