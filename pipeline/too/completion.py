"""Decide whether a ToO night has finished processing, from the per-filter configs and the scheduler.

A night is finished when no filter can still change: every filter either produced difference
photometry (or ended it for lack of input), or its scheduler job failed. Filters whose fate the
scheduler no longer records (completed rows are cleared nightly, stashed rows never resume) are
settled by quiescence: nothing under the night has been written for QUIET_PERIOD.
"""

import glob
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

import yaml

QUIET_PERIOD = timedelta(hours=6)
_TAIL_BYTES = 4096
_NO_INPUT_MARK = "Skipping difference_photometry"


class FilterState(Enum):
    DONE = "done"
    FAILED = "failed"
    RUNNING = "running"
    STALLED = "stalled"
    UNKNOWN = "unknown"


_SCHEDULER_STATE = {
    "Completed": FilterState.DONE,
    "Failed": FilterState.FAILED,
    "Rejected": FilterState.FAILED,
    "Ready": FilterState.RUNNING,
    "Pending": FilterState.RUNNING,
    "Processing": FilterState.RUNNING,
    "Stashed": FilterState.STALLED,
    "Paused": FilterState.STALLED,
}


@dataclass(frozen=True)
class FilterStatus:
    filter: str
    config: str
    state: FilterState


@dataclass(frozen=True)
class NightStatus:
    base_path: str
    filters: tuple
    last_activity: datetime

    def names(self, state: FilterState) -> list:
        return sorted(f.filter for f in self.filters if f.state is state)

    def final_due(self, now: datetime, quiet: timedelta = QUIET_PERIOD) -> tuple:
        if not self.filters:
            return False, "no per-filter configs under base_path"
        if self.names(FilterState.RUNNING):
            return False, f"still queued or running: {', '.join(self.names(FilterState.RUNNING))}"
        if not self.names(FilterState.DONE):
            return False, "no filter reached difference photometry"
        open_filters = self.names(FilterState.STALLED) + self.names(FilterState.UNKNOWN)
        if not open_filters:
            return True, "every filter finished or failed"
        idle = now - self.last_activity
        if idle >= quiet:
            return True, f"quiet for {idle} with unsettled filters: {', '.join(open_filters)}"
        return False, f"waiting on {', '.join(open_filters)}; quiet for {idle} of {quiet}"

    def summary(self) -> str:
        lines = [f"Filters observed: {len(self.filters)}"]
        for state, label in (
            (FilterState.DONE, "Processed"),
            (FilterState.FAILED, "Failed"),
            (FilterState.STALLED, "Stalled in queue"),
            (FilterState.UNKNOWN, "Not processed"),
        ):
            names = self.names(state)
            if names:
                lines.append(f"  {label} ({len(names)}): {', '.join(names)}")
        return "\n".join(lines)


def _difference_photometry_ended(config: str) -> bool:
    try:
        with open(config) as f:
            flags = (yaml.safe_load(f) or {}).get("flag") or {}
        if flags.get("difference_photometry"):
            return True
    except (OSError, yaml.YAMLError):
        pass
    log = os.path.splitext(config)[0] + ".log"
    try:
        with open(log, "rb") as f:
            f.seek(max(0, os.path.getsize(log) - _TAIL_BYTES))
            return _NO_INPUT_MARK in f.read().decode("utf-8", "replace")
    except OSError:
        return False


def _scheduler_states(scheduler_db: str, base_path: str) -> dict:
    if not scheduler_db or not os.path.exists(scheduler_db):
        return {}
    conn = sqlite3.connect(f"file:{scheduler_db}?mode=ro", uri=True, timeout=10)
    try:
        rows = conn.execute(
            'SELECT config, status FROM scheduler WHERE config LIKE ? ORDER BY "index"',
            (base_path.rstrip("/") + "/%",),
        ).fetchall()
    finally:
        conn.close()
    return {config: status for config, status in rows}


def night_status(base_path: str, scheduler_db: str) -> NightStatus:
    base_path = base_path.rstrip("/")
    scheduled = _scheduler_states(scheduler_db, base_path)
    filters, last = [], datetime.fromtimestamp(0)
    for config in sorted(glob.glob(os.path.join(base_path, "*", "*_ToO_*.yml"))):
        filt = os.path.basename(os.path.dirname(config))
        for path in glob.glob(os.path.join(os.path.dirname(config), "*")):
            last = max(last, datetime.fromtimestamp(os.path.getmtime(path)))
        if _difference_photometry_ended(config):
            state = FilterState.DONE
        else:
            state = _SCHEDULER_STATE.get(scheduled.get(config), FilterState.UNKNOWN)
        filters.append(FilterStatus(filt, config, state))
    return NightStatus(base_path, tuple(filters), last)
