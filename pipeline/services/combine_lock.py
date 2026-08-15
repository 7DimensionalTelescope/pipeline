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


_MEMINFO_KEYS = ("MemTotal", "MemFree", "MemAvailable", "Cached", "Committed_AS",
                 "AnonPages", "Shmem", "Unevictable", "SUnreclaim", "KernelStack",
                 "PageTables", "SwapTotal", "SwapFree")


def meminfo_bytes(keys=_MEMINFO_KEYS) -> dict:
    """Selected /proc/meminfo fields in bytes; 0 for anything unreadable."""
    wanted, out = set(keys), dict.fromkeys(keys, 0)
    try:
        with open("/proc/meminfo") as fp:
            for line in fp:
                key, _, rest = line.partition(":")
                if key in wanted:
                    out[key] = int(rest.split()[0]) * 1024
    except OSError:
        pass
    return out


def cgroup_memory() -> tuple[int, int]:
    """(binding memory ceiling, anonymous bytes under it), or (0, 0) when unconstrained.

    The limit usually sits on an ancestor slice rather than the process's own leaf cgroup,
    so every level is inspected and the tightest ceiling wins."""
    try:
        rel = open("/proc/self/cgroup").read().strip().rsplit(":", 1)[-1].lstrip("/")
    except OSError:
        return 0, 0
    node = os.path.join("/sys/fs/cgroup", rel)
    limit, holder = 0, None
    while True:
        for name in ("memory.high", "memory.max"):
            # unreadable or unparsable means "no ceiling here", never a crash: this runs on
            # hosts with cgroup v1, no cgroups, or none of it mounted
            try:
                value = int(open(os.path.join(node, name)).read().strip())
            except (OSError, ValueError):
                continue
            if limit == 0 or value < limit:
                limit, holder = value, node
        if os.path.realpath(node) == "/sys/fs/cgroup" or not node.startswith("/sys/fs/cgroup"):
            break
        node = os.path.dirname(node)
    if not limit:
        return 0, 0
    anon = 0
    try:
        for line in open(os.path.join(holder, "memory.stat")):
            key, _, value = line.partition(" ")
            if key == "anon":
                anon = int(value)
                break
    except (OSError, ValueError):
        pass
    return limit, anon


def host_irreclaimable_bytes(mem: dict | None = None) -> int:
    """Host RAM the kernel cannot reclaim (anon, tmpfs, mlocked, kernel), plus swapped-out
    pages that may return. Planned-but-untouched combine allocations are deliberately not
    in here -- the lease files carry those."""
    mem = mem or meminfo_bytes()
    return (mem["AnonPages"] + mem["Shmem"] + mem["Unevictable"] + mem["SUnreclaim"]
            + mem["KernelStack"] + mem["PageTables"] + (mem["SwapTotal"] - mem["SwapFree"]))


def memory_headroom_bytes(reserved_bytes: int = 0) -> int:
    """Bytes a combine may plan for: the tighter of its cgroup ceiling and host RAM.

    Both bind. The cgroup ceiling is what throttles this account (and differs per user --
    a dedicated slice override, the global one, or none at all), while the host is shared
    with every other account, whose slices may sum past physical RAM. Page cache is
    excluded from both: it is reclaimable, and the combine creates it itself, which is
    what made MemAvailable unusable as a budget.

    The host term counts irreclaimable pages, not Committed_AS: a memmapped combine
    promises its whole input set as private COW mappings, so Committed_AS runs past
    MemTotal while nearly all of it will never be dirtied (518 vs 504 GiB measured
    2026-08-14, 397 GiB of it one combine's own .fits maps, budget pinned to 0 and every
    combine to the strip floor -- fatal on NFS). Reading inputs moves MemFree to Cached
    and touches neither term here, so the budget no longer shrinks as combines run."""
    mem = meminfo_bytes()
    host = mem["MemTotal"] - host_irreclaimable_bytes(mem)
    limit, anon = cgroup_memory()
    headroom = min(limit - anon, host) if limit else host
    return max(0, headroom - int(reserved_bytes))


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
        peers = self.reserved_bytes
        with open(self._lease_file, "w") as fp:
            json.dump({"pid": os.getpid(), "bytes": int(planned_bytes), "tag": self.tag, "ts": time.time()}, fp)
        if self.logger:
            mem, (limit, anon) = meminfo_bytes(), cgroup_memory()
            budget = (
                f"cgroup ceiling {limit/2**30:.0f} GiB - anon {anon/2**30:.0f} GiB"
                if limit
                else f"host RAM {mem['MemTotal']/2**30:.0f} GiB - irreclaimable {host_irreclaimable_bytes(mem)/2**30:.0f} GiB"
            )
            self.logger.debug(
                f"Combine memory claim [{self.tag}]: this run {planned_bytes/2**30:.0f} GiB, "
                f"other combines {peers/2**30:.0f} GiB, total claimed {(planned_bytes + peers)/2**30:.0f} GiB "
                f"| headroom {memory_headroom_bytes(peers)/2**30:.0f} GiB ({budget}, peers reserved) "
                f"| host MemAvailable {mem['MemAvailable']/2**30:.0f} GiB "
                f"(free {mem['MemFree']/2**30:.0f} + cached {mem['Cached']/2**30:.0f}) -- not the budget"
            )

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
