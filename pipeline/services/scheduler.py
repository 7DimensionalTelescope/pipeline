import json
import sqlite3
import socket
import os
import signal
import threading
import numpy as np
from contextlib import contextmanager
from datetime import datetime
from astropy.table import Table, vstack


from ..const import SCRIPTS_DIR, NUM_GPUS, SCHEDULER_DB_PATH, QUEUE_SOCKET_PATH
from ..const import SUCCESS_RETURN_CODE, FAILURE_RETURN_CODE, EMPTY_INPUT_AFTER_SANITY_REJECTION_RETURN_CODE
from .logger import get_high_level_task_logger, log_orchestration_stop


class Scheduler:

    _empty_schedule = Table(
        dtype=[
            ("index", int),
            ("config", object),
            ("config_type", object),
            ("input_type", object),
            ("is_ready", bool),
            ("priority", int),
            ("readiness", int),
            ("status", object),
            ("dependent_idx", list),
            ("pid", int),
            ("dispatch", object),
            ("kwargs", object),
            ("process_start", object),
            ("process_end", object),
        ]
    )

    # SQL ordering clause used consistently throughout
    _ORDER_BY = 'ORDER BY is_ready DESC, priority DESC, readiness DESC, "index" ASC'

    # Explicit column list — safe regardless of ALTER TABLE column order in existing DBs.
    _DB_COLUMNS = (
        "index",
        "config",
        "config_type",
        "input_type",
        "is_ready",
        "priority",
        "readiness",
        "status",
        "dependent_idx",
        "pid",
        "dispatch",
        "kwargs",
        "process_start",
        "process_end",
    )
    _SELECT_COLUMNS = ", ".join(f'"{col}"' if col == "index" else col for col in _DB_COLUMNS)
    _LOCAL_TASK_FILTER = "(dispatch IS NULL OR dispatch = '')"

    # Constants
    MAX_PREPROCESS = 3
    HIGH_PRIORITY_THRESHOLD = 10
    # Seconds a statement waits for a competing writer before raising "database is locked".
    # A bulk submission must never make the queue daemon drop a completion.
    DB_BUSY_TIMEOUT = 30.0
    # Serializes sqlite3.connect()/close() across threads. libsqlite3 (seen on 3.51.1) can
    # ABBA-deadlock in its unix VFS when one thread opens a WAL db while another closes it
    # (unixOpen/findReusableFd vs sqlite3WalClose/unixLock, gdb-verified 2026-08-08) — it
    # froze the queue daemon whole. Guards only open/close, never the transaction body.
    _SQLITE_OPEN_CLOSE_LOCK = threading.Lock()

    def __init__(
        self,
        schedule=None,
        use_system_queue=False,
        overwrite_schedule=False,
        db_path=None,
        **kwargs,
    ):
        self._db_path = db_path or SCHEDULER_DB_PATH
        self.use_system_queue = use_system_queue and self._db_path is not None

        self.overwrite_schedule = overwrite_schedule

        self._kwargs = kwargs

        if self.use_system_queue:
            self._schedule = None
            self._connection_check()
            if schedule is not None:
                self._save_table_to_db(self._validate_and_get_table(schedule))
            # Initialize processing_preprocess from database
            self.processing_preprocess = 0
        else:
            self._schedule = self._validate_and_get_table(schedule) if schedule is not None else self._empty_schedule
            # Initialize processing_preprocess from current schedule
            if schedule is not None:
                self.processing_preprocess = len(
                    self._schedule[
                        (self._schedule["status"] == "Processing") & (self._schedule["config_type"] == "preprocess")
                    ]
                )
            else:
                self.processing_preprocess = 0

    @classmethod
    def from_list(
        cls,
        list_of_configs,
        base_priority=1,
        use_system_queue=False,
        overwrite_schedule=False,
        overwrite=False,
        overwrite_data=False,
        overwrite_preprocess=False,
        overwrite_science=False,
        input_type=None,
        processes=None,
        extra_kwargs=None,
        **kwargs,
    ):
        """Create a scheduler from a list of configs. `extra_kwargs`: plain flags appended to every task's command line; never JSON (the kwargs round-trip mangles quotes)."""
        import re
        import copy

        list_of_configs = np.atleast_1d(list_of_configs)

        table = copy.deepcopy(cls._empty_schedule)

        for idx, config in enumerate(list_of_configs):
            if not (os.path.exists(config)):
                print(f"Warning: Config file {config} does not exist")
                continue

            basename = os.path.basename(config)

            # Determine task_type based on the config name
            discriminator = basename.split("_")[0]
            if bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", discriminator)):
                task_type = "preprocess"
                priority = base_priority + 1
            else:
                task_type = "science"
                priority = base_priority

            if task_type == "preprocess":
                scheduler_kwargs = ["-overwrite"] if (overwrite or overwrite_data or overwrite_preprocess) else []
            else:
                scheduler_kwargs = ["-processes"] + list(processes) if processes is not None else []
                if overwrite or overwrite_data or overwrite_science:
                    scheduler_kwargs.append("-overwrite")

            if extra_kwargs:
                scheduler_kwargs = scheduler_kwargs + list(extra_kwargs)

            table.add_row(
                [
                    idx,
                    config,
                    task_type,
                    input_type or "User-input",
                    True,
                    priority,
                    100,
                    "Ready",
                    [],
                    0,
                    "",
                    scheduler_kwargs,
                    "",
                    "",
                ]
            )

        return cls(schedule=table, use_system_queue=use_system_queue, overwrite_schedule=overwrite_schedule, **kwargs)

    def _connection_check(self):
        """Create scheduler table if it doesn't exist."""

        with self._db_connection() as conn:
            # WAL so a bulk submission's writes never lock out the daemon's readers.
            # Persists in the file itself; best-effort because it needs a moment with no other writer.
            try:
                conn.execute("PRAGMA journal_mode=WAL").fetchone()
            except sqlite3.Error as e:
                get_high_level_task_logger(__name__).debug(f"Could not set journal_mode=WAL: {e}")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scheduler (
                    "index" INTEGER PRIMARY KEY,
                    config TEXT NOT NULL,
                    config_type TEXT NOT NULL,
                    input_type TEXT NOT NULL,
                    is_ready INTEGER NOT NULL,
                    priority INTEGER NOT NULL,
                    readiness INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    dependent_idx TEXT,
                    pid INTEGER,
                    dispatch TEXT,
                    kwargs TEXT,
                    process_start TEXT,
                    process_end TEXT
                )
            """
            )

            # Migrate existing tables to add new columns if they don't exist
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(scheduler)")
            columns = [row[1] for row in cursor.fetchall()]
            if "dispatch" not in columns:
                if "external" in columns:
                    cursor.execute("ALTER TABLE scheduler RENAME COLUMN external TO dispatch")
                else:
                    cursor.execute("ALTER TABLE scheduler ADD COLUMN dispatch TEXT")

            # Without these the claim scans the whole table under the write lock.
            cursor.execute(
                'CREATE INDEX IF NOT EXISTS ix_scheduler_claim '
                'ON scheduler(status, is_ready DESC, priority DESC, readiness DESC, "index")'
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS ix_scheduler_gate "
                "ON scheduler(status, config_type, priority, input_type)"
            )

            conn.commit()

    def start_system_queue(self):
        """Send wake message to queue socket."""
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(QUEUE_SOCKET_PATH)
            sock.sendall(b"wake\n")
            sock.close()
        except (FileNotFoundError, ConnectionRefusedError, OSError):
            # Socket might not exist yet or queue manager not running - this is OK
            pass

    def _validate_and_get_table(self, schedule):
        """Validate and extract Table from various input types."""
        if isinstance(schedule, Table):
            if schedule.colnames == self._empty_schedule.colnames:
                return schedule
            raise ValueError("Invalid schedule type")
        elif isinstance(schedule, Scheduler):
            return schedule.schedule
        raise ValueError("Invalid schedule type")

    @contextmanager
    def _db_connection(self, timeout=None):
        """Context manager for database connections. `timeout` overrides DB_BUSY_TIMEOUT."""
        with Scheduler._SQLITE_OPEN_CLOSE_LOCK:
            conn = sqlite3.connect(
                self._db_path, timeout=self.DB_BUSY_TIMEOUT if timeout is None else timeout
            )
            conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            with Scheduler._SQLITE_OPEN_CLOSE_LOCK:
                conn.close()

    @staticmethod
    def _parse_kwargs(kwargs_raw):
        if kwargs_raw is None:
            return []
        if isinstance(kwargs_raw, list):
            return kwargs_raw
        if isinstance(kwargs_raw, str):
            try:
                return json.loads(kwargs_raw.replace("'", '"'))
            except json.JSONDecodeError:
                return []
        return []

    def _row_to_dict(self, row):
        """Convert a sqlite3.Row (or mapping) to dictionary."""
        if row is None:
            return None

        return {
            "index": row["index"],
            "config": row["config"],
            "config_type": row["config_type"],
            "input_type": row["input_type"],
            "is_ready": bool(row["is_ready"]),
            "priority": row["priority"],
            "readiness": row["readiness"],
            "status": row["status"],
            "dependent_idx": json.loads(row["dependent_idx"]) if row["dependent_idx"] else [],
            "pid": row["pid"],
            "dispatch": row["dispatch"] or "",
            "kwargs": self._parse_kwargs(row["kwargs"]),
            "process_start": row["process_start"],
            "process_end": row["process_end"],
        }

    def _rows_to_table(self, rows):
        """Convert database rows to astropy Table."""
        if not rows:
            return self._empty_schedule

        data = {col: [] for col in self._empty_schedule.colnames}
        for row in rows:
            row_dict = self._row_to_dict(row)
            for col in self._empty_schedule.colnames:
                data[col].append(row_dict[col])

        return Table(data, dtype=self._empty_schedule.dtype)

    def _append_ready_task_constraints(self, cursor, query, params):
        """Append concurrency filters shared by local and dispatched Ready-task selection."""
        cursor.execute(
            "SELECT COUNT(*) FROM scheduler WHERE status = ? AND config_type = ?",
            ("Processing", "preprocess"),
        )
        if cursor.fetchone()[0] >= self.MAX_PREPROCESS:
            query += " AND config_type != ?"
            params.append("preprocess")

        cursor.execute(
            "SELECT COUNT(*) FROM scheduler WHERE status = ? AND priority > ?",
            ("Processing", self.HIGH_PRIORITY_THRESHOLD),
        )
        if cursor.fetchone()[0] > 0:
            query += " AND priority > ?"
            params.append(self.HIGH_PRIORITY_THRESHOLD)
        else:
            cursor.execute(
                "SELECT COUNT(*) FROM scheduler WHERE status = ? AND LOWER(input_type) = ?",
                ("Processing", "too"),
            )
            if cursor.fetchone()[0] > 0:
                query += " AND LOWER(input_type) = ?"
                params.append("too")

        return query, params

    @staticmethod
    def _config_stem(config):
        """Config-file path -> process_status.name stem (basename without .yml)."""
        base = os.path.basename(str(config))
        if base.endswith(".yml"):
            return base[:-4]
        if base.endswith(".yaml"):
            return base[:-5]
        return base

    def _dependency_edges(self, table):
        """Extract config-level dependency edges from a schedule table.

        A parent row's ``dependent_idx`` lists the scheduler indices that
        depend on it, so each entry becomes an edge
        ``(derived_name, source_name, source_config_type)``.  Walking every row
        keeps this agnostic to how many config types form the chain (e.g. a
        future type depending on science is mirrored automatically once the
        blueprint records those dependent_idx links).
        """
        edges = []
        if table is None or len(table) == 0:
            return edges

        idx_to_stem = {int(row["index"]): self._config_stem(row["config"]) for row in table}

        for row in table:
            source_stem = self._config_stem(row["config"])
            source_type = str(row["config_type"])
            dependents = row["dependent_idx"] if row["dependent_idx"] is not None else []
            for child_idx in dependents:
                derived_stem = idx_to_stem.get(int(child_idx))
                if not derived_stem:
                    continue
                edges.append((derived_stem, source_stem, source_type))
        return edges

    def mirror_dependencies(self):
        """Best-effort mirror of the schedule's config dependencies to postgres.

        The SQLite schedule stays authoritative; this never raises so the
        scheduler keeps working when the database is offline.
        """
        try:
            table = self._get_table_from_db() if self.use_system_queue else self._schedule
            edges = self._dependency_edges(table)
            if not edges:
                return
            from .database.process_status_dependency import ProcessStatusDependency

            ProcessStatusDependency().replace_dependencies(edges)
        except Exception as e:
            try:
                get_high_level_task_logger(__name__).debug(
                    f"process_status_dependency mirror skipped: {e}"
                )
            except Exception:
                pass

    def _check_duplicates(self):
        """Check for duplicate configs in database. Logs warning but doesn't raise error."""
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT config, COUNT(*) as count 
                FROM scheduler 
                GROUP BY config 
                HAVING COUNT(*) > 1
            """
            )
            duplicates = cursor.fetchall()
            if duplicates:
                dup_configs = [row[0] for row in duplicates]
                print(f"Warning: Duplicate configs exist in the schedule: {dup_configs}")

    def __add__(self, other):
        current_table = self.schedule
        offset = max(current_table["index"]) if len(current_table) > 0 else -1

        other_table = other.schedule.copy() if isinstance(other, Scheduler) else other.copy()
        if not isinstance(other_table, Table):
            raise ValueError("Invalid schedule type")

        # Check for overwrite functionality
        overwrite_schedule = self.overwrite_schedule

        if len(current_table) > 0:
            # Get set of existing configs to filter duplicates
            existing_configs = set(current_table["config"])

            if overwrite_schedule:
                # Remove existing rows with duplicate configs
                duplicate_mask = [config in existing_configs for config in other_table["config"]]

                if any(duplicate_mask):
                    duplicate_configs = [
                        config for config, is_dup in zip(other_table["config"], duplicate_mask) if is_dup
                    ]

                    # Stop currently running duplicate tasks before replacement.
                    processing_duplicate_rows = current_table[
                        [
                            (config in duplicate_configs) and (status == "Processing") and (pid not in (None, 0))
                            for config, status, pid in zip(
                                current_table["config"], current_table["status"], current_table["pid"]
                            )
                        ]
                    ]
                    for row in processing_duplicate_rows:
                        self._terminate_process(row["pid"])

                    # Remove duplicate rows from current_table
                    current_table = current_table[
                        [config not in duplicate_configs for config in current_table["config"]]
                    ]

                    # Recalculate offset after removing duplicates
                    offset = max(current_table["index"]) if len(current_table) > 0 else -1

                    print(f"Replaced {len(duplicate_configs)} existing schedule(s) with new ones")
            else:
                # Filter out rows with duplicate configs (keep existing, ignore new duplicates)
                non_duplicate_mask = [config not in existing_configs for config in other_table["config"]]
                other_table = other_table[non_duplicate_mask] if any(non_duplicate_mask) else other_table[[]]

                if len(other_table) < len(other.schedule if isinstance(other, Scheduler) else other):
                    duplicate_count = len(other.schedule if isinstance(other, Scheduler) else other) - len(other_table)
                    print(f"Warning: Ignoring {duplicate_count} duplicate config(s) when adding schedule")

        # Adjust indices
        if len(other_table) > 0:
            other_table["index"] = other_table["index"] + offset + 1
            for i in range(len(other_table)):
                if other_table["dependent_idx"][i]:
                    other_table["dependent_idx"][i] = [idx + offset + 1 for idx in other_table["dependent_idx"][i]]

        # Combine tables
        if len(other_table) > 0:
            if len(current_table) > 0:
                combined_table = vstack([current_table, other_table])
            else:
                combined_table = other_table
        else:
            combined_table = current_table

        if self.use_system_queue:
            self._save_table_to_db(combined_table)
        else:
            self._schedule = combined_table

        return combined_table

    def __repr__(self):
        return self.status(with_table=True)

    def print_schedule(self):
        self.schedule.pprint_all()

    def status(self, with_table=False):
        schedule = self.schedule
        total_tasks = len(schedule)
        in_ready = len(schedule[schedule["status"] == "Ready"])
        in_pending = len(schedule[schedule["status"] == "Pending"])
        in_processing = len(schedule[schedule["status"] == "Processing"])
        in_completed = len(schedule[schedule["status"] == "Completed"])
        in_paused = len(schedule[schedule["status"] == "Paused"])
        is_preprocess = len(schedule[schedule["config_type"] == "preprocess"])
        is_science = len(schedule[schedule["config_type"] == "science"])
        is_failed = len(schedule[schedule["status"] == "Failed"])
        if with_table:
            schedule.pprint_all(max_lines=10)
        return (
            f"Scheduler with {total_tasks} (preprocess: {is_preprocess} and science: {is_science}) tasks: "
            f"{in_ready} ready, {in_pending} pending, {in_processing} processing, {in_paused} paused, "
            f"{is_failed} failed, and {in_completed} completed"
        )

    @property
    def schedule(self):
        if self.use_system_queue:
            self._check_duplicates()
            return self._get_table_from_db()
        else:
            table = self._schedule
            table.sort(["is_ready", "priority", "readiness"], reverse=True)
            # Check duplicates for in-memory table (log warning but don't raise)
            config = table["config"]
            vals, counts = np.unique(config, return_counts=True)
            dups = vals[counts > 1]
            if len(dups) > 0:
                print(f"Warning: Duplicate configs exist in the schedule: {dups}")
            return table

    def add_schedule(self, other):
        self + other
        return self

    @property
    def has_schedule(self):
        if self.use_system_queue:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM scheduler WHERE status != ?", ("Completed",))
                return cursor.fetchone()[0] > 0
        return len(self.schedule[self.schedule["status"] != "Completed"]) > 0

    def get_next_task(self):
        """Get the next task to process with priority and concurrency constraints."""
        if not self.has_schedule:
            return None

        if self.use_system_queue:
            row, cmd = self._get_next_task_db()
            if row is None:
                import subprocess

                subprocess.run([f"{SCRIPTS_DIR}/autostash"], check=True)
                return None, None
            else:
                return row, cmd
        else:
            return self._get_next_task_memory()

    def _get_next_task_db(self):
        """Get next task from database."""
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            try:
                query = (
                    f'SELECT "index" FROM scheduler WHERE status = ? AND {self._LOCAL_TASK_FILTER}'
                )
                params = ["Ready"]
                query, params = self._append_ready_task_constraints(cursor, query, params)
                query += f" {self._ORDER_BY} LIMIT 1"

                cursor.execute(query, tuple(params))
                index_row = cursor.fetchone()
                if not index_row:
                    conn.rollback()
                    return None, None

                task_index = index_row[0]
                process_start = datetime.now().isoformat()
                cursor.execute(
                    f'UPDATE scheduler SET status = ?, process_start = ?, process_end = ? '
                    f'WHERE "index" = ? AND status = ? AND {self._LOCAL_TASK_FILTER}',
                    ("Processing", process_start, "", task_index, "Ready"),
                )
                if cursor.rowcount == 0:
                    conn.rollback()
                    return None, None

                cursor.execute(
                    f'SELECT {self._SELECT_COLUMNS} FROM scheduler WHERE "index" = ?',
                    (task_index,),
                )
                row = cursor.fetchone()
                conn.commit()
            except Exception:
                conn.rollback()
                return None, None

        row = self._row_to_dict(row) if row else None
        if row is None:
            return None, None

        # This is dangerous
        # if row["priority"] == 0 and ["-overwrite"] not in row["kwargs"]:
        #     row["kwargs"].append("-overwrite")

        return row, self._generate_command(row["index"], row["kwargs"])

    def _get_next_task_memory(self):
        """Get next task from in-memory schedule."""
        ready_mask = (self.schedule["status"] == "Ready") & (
            (self.schedule["dispatch"] == "") | (self.schedule["dispatch"] == None)  # noqa: E711
        )
        ready_tasks = self.schedule[ready_mask]
        if len(ready_tasks) == 0:
            return None, None

        # Enforce preprocess limit
        if self.processing_preprocess >= self.MAX_PREPROCESS:
            ready_tasks = ready_tasks[ready_tasks["config_type"] != "preprocess"]

        # Check for high priority processing
        high_priority_processing = (
            len(
                self.schedule[
                    (self.schedule["status"] == "Processing")
                    & (self.schedule["priority"] > self.HIGH_PRIORITY_THRESHOLD)
                ]
            )
            > 0
        )

        if high_priority_processing:
            ready_tasks = ready_tasks[ready_tasks["priority"] > self.HIGH_PRIORITY_THRESHOLD]
        else:
            # Check for TOO processing
            too_processing = (
                len(
                    self.schedule[
                        (self.schedule["status"] == "Processing")
                        & (np.char.lower(self.schedule["input_type"].astype(str)) == "too")
                    ]
                )
                > 0
            )
            if too_processing:
                ready_tasks = ready_tasks[np.char.lower(ready_tasks["input_type"].astype(str)) == "too"]

        if len(ready_tasks) == 0:
            return None, None

        # Get first task (already sorted by _ORDER_BY in schedule property)
        row_dict = {col: ready_tasks[col][0] for col in ready_tasks.colnames}
        task_index = row_dict["index"]

        # Mark as Processing and set process_start
        mask = self._schedule["index"] == task_index

        self._schedule["status"][mask] = "Processing"
        self._schedule["process_start"][mask] = datetime.now().isoformat()
        self._schedule["process_end"][mask] = ""

        if row_dict.get("config_type") == "preprocess":
            self.processing_preprocess += 1

        scheduler_kwargs = row_dict["kwargs"]

        if row_dict["priority"] == 0 and ["-overwrite"] not in scheduler_kwargs:
            scheduler_kwargs.append("-overwrite")

        return row_dict, self._generate_command(task_index, scheduler_kwargs)

    @staticmethod
    def _orchestration_stop_reason(return_code) -> str:
        """Why a run ended without reporting an outcome of its own — for the config's log."""
        if isinstance(return_code, int) and return_code < 0:
            try:
                detail = f"killed by {signal.Signals(-return_code).name}"
            except ValueError:
                detail = f"killed by signal {-return_code}"

            if return_code == -signal.SIGTERM:
                detail += " (queue/trigger restart, schedule overwrite, or terminate_scheduler_tasks)"
            elif return_code == -signal.SIGKILL:
                detail += " (OOM killer or a forced kill)"
        else:
            detail = f"exited with return code {return_code}, which no stage produces"

        return (
            f"Marked Failed by the queue daemon: {detail}. Orchestration stop, not a scientific "
            f"verdict — the run was ended from outside and filed no report of its own."
        )

    def requeue_task(self, index, reason=None, timeout=None):
        """Put one unfinished Processing task back in the queue. True when the row changed."""
        return self.requeue_tasks([index], reason=reason, timeout=timeout) == 1

    def requeue_tasks(self, indices, reason=None, timeout=None):
        """
        Put unfinished Processing tasks back in the queue. Returns how many rows changed.

        For tasks cut off before they could report an outcome — a daemon shutdown, a crash.
        Mirrors what `update_process_status` does for dead PIDs, and only touches rows still
        marked Processing so a resolved row is never clobbered.

        One transaction for the whole set: called from `stop_processing`, where a per-row
        write could block on `DB_BUSY_TIMEOUT` each time and blow through systemd's
        `TimeoutStopSec` — which would get the daemon SIGKILLed mid-cleanup and strand the
        very rows it came to rescue. `timeout` bounds that wait; shutdown passes a short one.

        Notes are written after the commit, never while holding the write lock: they land on
        NFS, so a slow mount must not extend the transaction. If the process dies between the
        two, the rows are still correct and only the notes are missing.
        """
        indices = [int(index) for index in np.atleast_1d(indices) if index is not None]
        if not indices:
            return 0

        configs = []

        if self.use_system_queue:
            placeholders = ",".join("?" * len(indices))
            with self._db_connection(timeout=timeout) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f'SELECT config FROM scheduler WHERE "index" IN ({placeholders}) AND status = ? '
                    f'AND {self._LOCAL_TASK_FILTER}',
                    (*indices, "Processing"),
                )
                configs = [row[0] for row in cursor.fetchall()]
                if not configs:
                    return 0

                cursor.execute(
                    f'UPDATE scheduler SET status = ?, pid = 0, process_start = ?, dispatch = NULL '
                    f'WHERE "index" IN ({placeholders}) AND status = ? AND {self._LOCAL_TASK_FILTER}',
                    ("Ready", "", *indices, "Processing"),
                )
                changed = cursor.rowcount
                conn.commit()
        else:
            changed = 0
            for index in indices:
                mask = self._schedule["index"] == index
                if len(self._schedule[mask]) == 0 or self._schedule["status"][mask][0] != "Processing":
                    continue

                configs.append(self._schedule["config"][mask][0])
                self._schedule["status"][mask] = "Ready"
                self._schedule["pid"][mask] = 0
                self._schedule["process_start"][mask] = ""
                changed += 1

        if reason:
            for config in configs:
                log_orchestration_stop(config, reason)

        return changed

    def mark_done(self, index, return_code=True, timeout=None):
        if self.use_system_queue:
            self._mark_done_db(index, return_code, timeout=timeout)
        else:
            self._mark_done_memory(index, return_code)

    def _mark_done_db(self, index, return_code=True, timeout=None):
        orchestration_note = None
        with self._db_connection(timeout=timeout) as conn:
            cursor = conn.cursor()
            # Check if task is already marked as done to prevent duplicate processing
            cursor.execute(
                'SELECT status, dependent_idx, config_type, config FROM scheduler WHERE "index" = ?', (index,)
            )
            row = cursor.fetchone()
            if not row:
                return

            current_status, dependent_idx_json, config_type, config = row
            # If already marked as done, skip to prevent duplicate increments
            if current_status in ("Completed", "Failed", "Rejected"):
                return

            dependent_indices = json.loads(dependent_idx_json) if dependent_idx_json else []
            process_end = datetime.now().isoformat()

            # the WHERE's done-status guard is a compare-and-swap: one of two racing mark_done calls wins and promotes
            done_guard = " AND status NOT IN (?, ?, ?)"
            done_states = ("Completed", "Failed", "Rejected")
            if return_code==SUCCESS_RETURN_CODE:

                cursor.execute(
                    'UPDATE scheduler SET status = ?, pid = 0, dispatch = NULL, '
                    'process_end = ? WHERE "index" = ?' + done_guard,
                    ("Completed", process_end, index, *done_states),
                )
            elif return_code==FAILURE_RETURN_CODE:

                cursor.execute(
                    'UPDATE scheduler SET status = ?, readiness = ?, is_ready = ?, pid = 0, dispatch = NULL, '
                    'process_end = ? WHERE "index" = ?' + done_guard,
                    ("Failed", 0, 0, process_end, index, *done_states),
                )
            elif return_code==EMPTY_INPUT_AFTER_SANITY_REJECTION_RETURN_CODE:
                process_end = datetime.now().isoformat()
                cursor.execute(
                    'UPDATE scheduler SET status = ?, pid = 0, dispatch = NULL, '
                    'process_end = ? WHERE "index" = ?' + done_guard,
                    ("Rejected", process_end, index, *done_states),
                )
            else:
                # The run never reported for itself: killed by a signal (systemd stop, OOM
                # killer) or an exit code no stage produces. Orchestration, not science —
                # fail it here, because leaving it Processing strands the row forever.
                cursor.execute(
                    'UPDATE scheduler SET status = ?, readiness = ?, is_ready = ?, pid = 0, dispatch = NULL, '
                    'process_end = ? WHERE "index" = ?' + done_guard,
                    ("Failed", 0, 0, process_end, index, *done_states),
                )
                if cursor.rowcount:
                    orchestration_note = self._orchestration_stop_reason(return_code)

            # any outcome advances the still-Pending dependents; missing-input ones fail fast downstream
            if cursor.rowcount:
                for dep_idx in dependent_indices:
                    cursor.execute(
                        'SELECT readiness FROM scheduler WHERE "index" = ? AND status = ?', (dep_idx, "Pending")
                    )
                    dep_row = cursor.fetchone()
                    if dep_row:
                        new_readiness = dep_row[0] + 1

                        if new_readiness > 100:
                            new_readiness = 100

                        if new_readiness == 100:
                            cursor.execute(
                                'UPDATE scheduler SET readiness = ?, status = ?, is_ready = ? '
                                'WHERE "index" = ? AND status = ?',
                                (new_readiness, "Ready", 1, dep_idx, "Pending"),
                            )
                        else:
                            cursor.execute(
                                'UPDATE scheduler SET readiness = ? WHERE "index" = ? AND status = ?',
                                (new_readiness, dep_idx, "Pending"),
                            )

            conn.commit()

        # Only after the transaction is closed: this writes to a config log on NFS, and doing
        # it inside the transaction once held the write lock open indefinitely and wedged the
        # daemon. Nothing that touches a filesystem belongs above this line.
        if orchestration_note:
            log_orchestration_stop(config, orchestration_note)

    def _mark_done_memory(self, task_index, return_code=True):
        mask = self._schedule["index"] == task_index
        if len(self._schedule[mask]) == 0:
            return

        # Check if already marked as done to prevent duplicate processing
        row_dict = {col: self._schedule[col][mask][0] for col in self._schedule.colnames}
        current_status = row_dict["status"]

        if current_status == "Completed" or current_status == "Failed":
            return

        # Get task info
        config_type = row_dict["config_type"]
        dependent_indices = row_dict["dependent_idx"]

        if config_type == "preprocess":
            self.processing_preprocess -= 1
            if self.processing_preprocess < 0:
                self.processing_preprocess = 0

        if return_code==SUCCESS_RETURN_CODE:
            self._schedule["status"][mask] = "Completed"
            self._schedule["pid"][mask] = 0
            self._schedule["process_end"][mask] = datetime.now().isoformat()

        else:
            # Check if this is a retry (priority is already 0)
            self._schedule["pid"][mask] = 0
            self._schedule["status"][mask] = "Failed"
            self._schedule["readiness"][mask] = 0
            self._schedule["is_ready"][mask] = False
            self._schedule["process_end"][mask] = datetime.now().isoformat()

            if return_code not in (
                SUCCESS_RETURN_CODE,
                FAILURE_RETURN_CODE,
                EMPTY_INPUT_AFTER_SANITY_REJECTION_RETURN_CODE,
            ):
                log_orchestration_stop(row_dict["config"], self._orchestration_stop_reason(return_code))

        # any outcome advances the still-Pending dependents; missing-input ones fail fast downstream
        for dep_idx in dependent_indices:
            dep_mask = (self._schedule["index"] == dep_idx) & (self._schedule["status"] == "Pending")
            if not dep_mask.any():
                continue
            self._schedule["readiness"][dep_mask] += 1

            if self._schedule["readiness"][dep_mask] >= 100:
                self._schedule["readiness"][dep_mask] = 100
                self._schedule["status"][dep_mask] = "Ready"
                self._schedule["is_ready"][dep_mask] = True

    def list_of_ready_tasks(self):
        if self.use_system_queue:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"SELECT {self._SELECT_COLUMNS} FROM scheduler WHERE is_ready = 1 {self._ORDER_BY}"
                )
                return self._rows_to_table(cursor.fetchall())
        return self.schedule[self.schedule["is_ready"]]

    def set_pid(self, index, pid):
        if self.use_system_queue:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('UPDATE scheduler SET pid = ? WHERE "index" = ?', (pid, index))
                conn.commit()
        else:
            self._schedule["pid"][self._schedule["index"] == index] = pid

    def claim_next_dispatch_task(self, server_name, config_types=None, input_type=None):
        """
        Claim one Ready task for a worker host.

        ``config_types``: restrict to these config_type values, for a worker that cannot run
        every kind of task (a GPU-less host must never claim preprocess or coadd work).
        ``input_type``: relabel the claimed row, so both databases attribute it to this worker.
        A ToO row keeps its own label: `is_too` and the ToO exclusivity gate both key off it.

        Marks the row Processing, sets ``dispatch`` to ``server_name``, and leaves
        ``pid`` at 0 until :meth:`set_dispatch_pid` is called.

        Returns:
            dict | None: Task row from the origin scheduler DB, or None if nothing to claim.
        """
        if not self.use_system_queue:
            raise RuntimeError("claim_next_dispatch_task requires use_system_queue=True")

        server_name = str(server_name).strip()
        if not server_name:
            raise ValueError("server_name must be non-empty")

        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            try:
                query = (
                    f'SELECT "index" FROM scheduler WHERE status = ? AND {self._LOCAL_TASK_FILTER}'
                )
                params = ["Ready"]
                if config_types:
                    query += f" AND config_type IN ({','.join('?' for _ in config_types)})"
                    params.extend(config_types)
                query, params = self._append_ready_task_constraints(cursor, query, params)
                query += f" {self._ORDER_BY} LIMIT 1"

                cursor.execute(query, tuple(params))
                index_row = cursor.fetchone()
                if not index_row:
                    conn.rollback()
                    return None

                task_index = index_row[0]
                process_start = datetime.now().isoformat()
                relabel = ", input_type = CASE WHEN LOWER(input_type) = 'too' THEN input_type ELSE ? END"
                update_params = ["Processing", server_name, process_start, ""]
                if input_type:
                    update_params.append(input_type)
                update_params += [task_index, "Ready"]
                cursor.execute(
                    f'UPDATE scheduler SET status = ?, dispatch = ?, pid = 0, process_start = ?, process_end = ?'
                    f'{relabel if input_type else ""} '
                    f'WHERE "index" = ? AND status = ? AND {self._LOCAL_TASK_FILTER}',
                    tuple(update_params),
                )
                if cursor.rowcount == 0:
                    conn.rollback()
                    return None

                cursor.execute(
                    f'SELECT {self._SELECT_COLUMNS} FROM scheduler WHERE "index" = ?',
                    (task_index,),
                )
                row = cursor.fetchone()
                conn.commit()
            except Exception:
                conn.rollback()
                return None

        return self._row_to_dict(row) if row else None

    def set_dispatch_pid(self, index, server_name, pid):
        """
        Record the worker PID reported by a worker host for a borrowed task.

        Returns:
            bool: True when the row was updated.
        """
        if not self.use_system_queue:
            raise RuntimeError("set_dispatch_pid requires use_system_queue=True")

        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE scheduler SET pid = ? WHERE "index" = ? AND dispatch = ? AND status = ?',
                (int(pid), int(index), str(server_name), "Processing"),
            )
            updated = cursor.rowcount > 0
            conn.commit()
        return updated

    def release_dispatch_task(self, index, server_name):
        """
        Return a borrowed task to Ready on the origin scheduler without completing it.

        Returns:
            bool: True when the row was updated.
        """
        if not self.use_system_queue:
            raise RuntimeError("release_dispatch_task requires use_system_queue=True")

        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE scheduler SET status = ?, dispatch = NULL, pid = 0, process_start = ?, process_end = ? '
                'WHERE "index" = ? AND dispatch = ? AND status = ?',
                ("Ready", "", "", int(index), str(server_name), "Processing"),
            )
            updated = cursor.rowcount > 0
            conn.commit()
        return updated

    def is_all_done(self):
        if self.use_system_queue:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT COUNT(*) FROM scheduler WHERE status NOT IN (?, ?)",
                    ("Completed", "Failed"),
                )
                return cursor.fetchone()[0] == 0
        else:
            completed = self.schedule[self.schedule["status"] == "Completed"]
            failed = self.schedule[self.schedule["status"] == "Failed"]
            return (len(completed) + len(failed)) == len(self.schedule)

    def rerun_failed_tasks(self, overwrite=False, dates=None):
        """
        Rerun failed tasks by changing their status to Ready with readiness 100.

        Returns:
            int: Number of tasks that were updated

        Parameters:
            overwrite (bool): If True, add '-overwrite' to each row's kwargs so reruns overwrite
                existing outputs; the row's other kwargs (e.g. a cascade sweep's
                -master_frame_only -calib_types ...) are preserved either way.
            dates (str | Iterable[str] | None): If given, only rerun failed tasks whose config
                path contains one of these date strings (e.g. "2025-04-30" or
                ["2025-04-30", "2025-05-01"]). If None or empty, all failed tasks are rerun.
        """
        if dates is None:
            date_list = []
        elif isinstance(dates, str):
            date_list = [d.strip() for d in dates.split(",") if d.strip()]
        else:
            date_list = [str(d).strip() for d in dates if str(d).strip()]

        if self.use_system_queue:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                if date_list:
                    like_clause = " OR ".join(["config LIKE ?"] * len(date_list))
                    cursor.execute(
                        f'SELECT "index", kwargs FROM scheduler WHERE status = ? AND ({like_clause})',
                        ("Failed", *(f"%{d}%" for d in date_list)),
                    )
                else:
                    cursor.execute('SELECT "index", kwargs FROM scheduler WHERE status = ?', ("Failed",))
                rows = cursor.fetchall()
                for index, row_kwargs in rows:
                    retry_kwargs = str(self._retry_kwargs(row_kwargs, overwrite))
                    cursor.execute(
                        """UPDATE scheduler
                           SET status = ?, priority = ?, readiness = ?, is_ready = ?, pid = 0,
                               process_start = ?, process_end = ?, input_type = ?, kwargs = ?
                           WHERE "index" = ?""",
                        ("Ready", 0, 100, 1, "", "", "Reprocess", retry_kwargs, index),
                    )
                conn.commit()
                return len(rows)
        else:
            mask = self._schedule["status"] == "Failed"
            if date_list:
                configs = self._schedule["config"]
                date_mask = np.array(
                    [any(d in str(c) for d in date_list) for c in configs],
                    dtype=bool,
                )
                mask = mask & date_mask
            count = int(np.sum(mask))
            if count > 0:
                for idx in np.where(mask)[0]:
                    self._schedule["kwargs"][idx] = self._retry_kwargs(self._schedule["kwargs"][idx], overwrite)
                self._schedule["status"][mask] = "Ready"
                self._schedule["priority"][mask] = 1
                self._schedule["readiness"][mask] = 100
                self._schedule["is_ready"][mask] = True
                self._schedule["pid"][mask] = 0
                self._schedule["process_start"][mask] = ""
                self._schedule["process_end"][mask] = ""
                self._schedule["input_type"][mask] = "Reprocess"
            return count

    @staticmethod
    def _retry_kwargs(kwargs, overwrite):
        """A failed row keeps its kwargs on retry; '-overwrite' only when requested."""
        if isinstance(kwargs, str):
            try:
                kwargs = json.loads(kwargs.replace("'", '"'))
            except (ValueError, TypeError):
                kwargs = []
        retry = [k for k in (kwargs or []) if k != "-overwrite"]
        if overwrite:
            retry.append("-overwrite")
        return retry

    def clear_schedule(self, all=False):
        import signal

        if self.use_system_queue:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                # Get completed tasks to save before deleting
                if all:
                    cursor.execute(f"SELECT {self._SELECT_COLUMNS} FROM scheduler")
                else:
                    cursor.execute(
                        f"SELECT {self._SELECT_COLUMNS} FROM scheduler WHERE status = ?",
                        ("Completed",),
                    )

                completed_rows = cursor.fetchall()

                # Save completed tasks to file if any exist
                if completed_rows:
                    completed_table = self._rows_to_table(completed_rows)
                    self._save_completed_to_file(completed_table)

                # Get PIDs of tasks to be cleared before deleting
                if all:
                    cursor.execute(
                        f"SELECT pid FROM scheduler WHERE pid IS NOT NULL AND {self._LOCAL_TASK_FILTER}"
                    )
                else:
                    cursor.execute(
                        f"SELECT pid FROM scheduler WHERE status = ? AND pid IS NOT NULL AND {self._LOCAL_TASK_FILTER}",
                        ("Completed",),
                    )

                pids_to_kill = [row[0] for row in cursor.fetchall() if row[0] is not None]

                # Kill processes with those PIDs
                for pid in pids_to_kill:
                    try:
                        if pid != 0:
                            os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        # Process already dead, ignore
                        pass
                    except PermissionError:
                        # No permission to kill, log but continue
                        pass
                    except Exception as e:
                        # Other error, log but continue
                        pass

                # Now delete the schedules
                if all:
                    cursor.execute("DELETE FROM scheduler")
                else:
                    cursor.execute("DELETE FROM scheduler WHERE status = ?", ("Completed",))
                conn.commit()
        else:
            # For in-memory schedule, kill PIDs before clearing
            if all:
                schedule_to_clear = self._schedule
            else:
                schedule_to_clear = self._schedule[self._schedule["status"] == "Completed"]

            # Save completed tasks to file if any exist
            if len(schedule_to_clear) > 0:
                if all:
                    self._save_completed_to_file(self._schedule)
                else:
                    self._save_completed_to_file(schedule_to_clear)

            # Kill processes with PIDs
            import signal

            if "pid" in schedule_to_clear.colnames:
                for pid in schedule_to_clear["pid"]:
                    if pid != 0:
                        try:
                            os.kill(int(pid), signal.SIGTERM)
                        except (ProcessLookupError, PermissionError, ValueError, TypeError):
                            # Process already dead, no permission, or invalid PID, ignore
                            pass

            # Now clear the schedule
            self._schedule = self._empty_schedule if all else self._schedule[self._schedule["status"] != "Completed"]

    def _save_completed_to_file(self, table):
        """Save completed tasks table to /var/db/{date}.npy as astropy Table.
        If file exists, combine/append with existing data."""
        try:
            db_dir = os.path.dirname(self._db_path)
            date_str = datetime.now().strftime("%Y-%m-%d")
            file_path = os.path.join(db_dir, f"{date_str}.npy")

            # Normalize dependent_idx column to 1D object array for vstack compatibility
            def normalize_dependent_idx(tbl):
                if "dependent_idx" in tbl.colnames:
                    dep_idx_vals = tbl["dependent_idx"]
                    tbl.remove_column("dependent_idx")
                    dep_idx = [
                        (
                            list(val.flatten())
                            if isinstance(val, np.ndarray) and val.ndim > 1
                            else (list(val) if isinstance(val, (list, np.ndarray)) and len(val) > 0 else [])
                        )
                        for val in dep_idx_vals
                    ]
                    dep_idx_arr = np.empty(len(dep_idx), dtype=object)
                    dep_idx_arr[:] = dep_idx
                    tbl["dependent_idx"] = dep_idx_arr
                return tbl

            table = normalize_dependent_idx(table.copy())

            if os.path.exists(file_path):
                existing_table = normalize_dependent_idx(Table(np.load(file_path, allow_pickle=True)))
                combined_table = vstack([existing_table, table]) if len(existing_table) > 0 else table
            else:
                combined_table = table

            np.save(file_path, combined_table)
        except Exception as e:
            print(f"Warning: Failed to save completed tasks to file: {e}")
            np.save(file_path.replace(".npy", "_error.npy"), table, allow_pickle=True)

    def _generate_command(self, index, scheduler_kwargs, **kwargs):

        if self.use_system_queue:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT config, config_type, input_type FROM scheduler WHERE "index" = ?', (index,))
                row = cursor.fetchone()
                if not row:
                    raise ValueError(f"Task with index {index} not found")
                config, config_type, input_type = row
        else:
            # Find task by index in schedule
            mask = self._schedule["index"] == index
            if len(self._schedule[mask]) == 0:
                raise ValueError(f"Task with index {index} not found")
            config = self._schedule["config"][mask][0]
            config_type = self._schedule["config_type"][mask][0]
            input_type = self._schedule["input_type"][mask][0]

        return self.build_command(config, config_type, input_type, scheduler_kwargs)

    @staticmethod
    def build_command(config, config_type, input_type, scheduler_kwargs):
        """Reduction command line for one row; SCRIPTS_DIR is the CALLING host's, so a worker builds its own."""
        is_too = str(input_type).lower() == "too" or "_ToO_" in config

        if config_type == "preprocess":
            cmd = [f"{SCRIPTS_DIR}/preprocess", "-config", config, "-make_plots"]
        elif config_type == "science":
            cmd = [f"{SCRIPTS_DIR}/data_reduction", "-config", config]
        elif config_type == "debug":
            return [f"{SCRIPTS_DIR}/debug", "-config", config]
        else:
            raise ValueError(f"Invalid systemd queue config_type: {config_type}")

        if input_type:
            cmd.extend(["-input_type", str(input_type)])
        if is_too:
            cmd.append("-is_too")
        cmd.extend(scheduler_kwargs)
        return cmd

    def _get_table_from_db(self):
        """Create astropy Table from SQLite database."""
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT {self._SELECT_COLUMNS} FROM scheduler {self._ORDER_BY}")
            return self._rows_to_table(cursor.fetchall())

    def _save_table_to_db(self, table):
        """Save astropy Table to SQLite database. If data exists, append instead of overwriting."""
        # Check if table is empty
        if len(table) == 0:
            print("Warning: Attempted to save empty schedule to database")
            return

        with self._db_connection() as conn:
            cursor = conn.cursor()

            # Check if there's existing data
            cursor.execute("SELECT COUNT(*) FROM scheduler")
            existing_count = cursor.fetchone()[0]

            if existing_count > 0:
                # Load existing table to get max index and existing configs
                existing_table = self._get_table_from_db()

                # Get set of existing configs to filter duplicates
                existing_configs = set(existing_table["config"])

                # Filter out rows with duplicate configs
                overwrite_schedule = self.overwrite_schedule

                new_table = table.copy()
                # Create mask for non-duplicate rows

                if overwrite_schedule:
                    duplicate_mask = [config in existing_configs for config in new_table["config"]]

                    # Delete existing rows with duplicate configs from database
                    if any(duplicate_mask):
                        duplicate_configs = [
                            config for config, is_dup in zip(new_table["config"], duplicate_mask) if is_dup
                        ]
                        placeholders = ",".join(["?"] * len(duplicate_configs))
                        cursor.execute(
                            f'SELECT pid FROM scheduler WHERE status = ? AND pid IS NOT NULL AND config IN ({placeholders}) '
                            f'AND {self._LOCAL_TASK_FILTER}',
                            ("Processing", *duplicate_configs),
                        )
                        for (pid,) in cursor.fetchall():
                            self._terminate_process(pid)

                        cursor.execute(
                            "DELETE FROM scheduler WHERE config IN ({})".format(
                                ",".join(["?"] * len(duplicate_configs))
                            ),
                            duplicate_configs,
                        )
                        conn.commit()

                        # Update existing_table to reflect deleted rows for index calculation
                        existing_table = existing_table[
                            [config not in duplicate_configs for config in existing_table["config"]]
                        ]

                        get_high_level_task_logger(__name__).info(
                            "Replaced %d existing schedule(s) with new ones", len(duplicate_configs)
                        )

                    # Use all new rows (replacing existing ones)
                    filtered_table = new_table

                else:
                    non_duplicate_mask = [config not in existing_configs for config in new_table["config"]]
                    filtered_table = (
                        new_table[non_duplicate_mask] if any(non_duplicate_mask) else new_table[[]]
                    )  # Empty table if all duplicates

                    if len(filtered_table) < len(new_table):
                        duplicate_count = len(new_table) - len(filtered_table)
                        print(f"Warning: Ignoring {duplicate_count} duplicate config(s) when adding schedule")

                # Adjust indices in filtered table to avoid conflicts
                if len(existing_table) > 0 and len(filtered_table) > 0:
                    max_existing_idx = max(existing_table["index"])
                    offset = max_existing_idx + 1

                    # Adjust indices
                    filtered_table["index"] = filtered_table["index"] + offset

                    # Adjust dependent_idx references in the filtered table
                    for i in range(len(filtered_table)):
                        if filtered_table["dependent_idx"][i]:
                            filtered_table["dependent_idx"][i] = [
                                idx + offset for idx in filtered_table["dependent_idx"][i]
                            ]

                    # Insert only the new non-duplicate rows
                    table_to_insert = filtered_table
                elif len(filtered_table) > 0:
                    # No existing rows, insert the filtered table as-is
                    table_to_insert = filtered_table
                else:
                    # All rows were duplicates
                    table_to_insert = filtered_table

            else:
                # No existing data, just use the new table
                table_to_insert = table

            # Insert only the new rows (existing data remains untouched)
            if len(table_to_insert) == 0:
                # No rows to insert
                return

            for row in table_to_insert:
                try:
                    # Convert dependent_idx to list of Python ints (handle numpy int64)
                    dependent_idx = row["dependent_idx"] if row["dependent_idx"] else []
                    if dependent_idx:
                        # Convert numpy int64 to Python int for JSON serialization
                        dependent_idx = [int(idx) for idx in dependent_idx]
                    dependent_idx_json = json.dumps(dependent_idx) if dependent_idx else None

                    pid = row.get("pid") if "pid" in row.colnames else 0
                    dispatch = row.get("dispatch") if "dispatch" in row.colnames else ""
                    kwargs = row.get("kwargs") if "kwargs" in row.colnames else None
                    process_start = row.get("process_start") if "process_start" in row.colnames else None
                    process_end = row.get("process_end") if "process_end" in row.colnames else None

                    cursor.execute(
                        """INSERT INTO scheduler 
                           ("index", config, config_type, input_type, is_ready, priority, readiness, status, dependent_idx, pid, dispatch, kwargs, process_start, process_end)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            int(row["index"]),
                            str(row["config"]),
                            str(row["config_type"]),
                            str(row["input_type"]),
                            1 if row["is_ready"] else 0,
                            int(row["priority"]),
                            int(row["readiness"]),
                            str(row["status"]),
                            dependent_idx_json,
                            int(pid) if pid is not None else 0,
                            str(dispatch) if dispatch is not None else "",
                            str(kwargs) if kwargs is not None else None,
                            str(process_start) if process_start is not None else None,
                            str(process_end) if process_end is not None else None,
                        ),
                    )
                except Exception as e:
                    # Log the error but continue with other rows
                    print(f"Warning: Failed to insert row with index {row.get('index', 'unknown')}: {e}")
                    raise  # Re-raise to see what's wrong

            conn.commit()

        # Mirror config-level dependencies to postgres (best-effort; never fatal).
        self.mirror_dependencies()

    def update_process_status(self):

        if self.use_system_queue:
            return self._update_process_status_db()
        else:
            return self._update_process_status_memory()

    def _update_process_status_db(self):
        """Check and revert killed processes for database mode."""
        reverted_count = 0
        reclaimed_configs = []
        with self._db_connection() as conn:
            cursor = conn.cursor()
            # Get all tasks with PIDs that are in Processing status
            cursor.execute(
                f'SELECT "index", pid, config_type, config FROM scheduler '
                f'WHERE status = ? AND pid IS NOT NULL AND {self._LOCAL_TASK_FILTER}',
                ("Processing",),
            )
            processing_tasks = cursor.fetchall()

            for task_index, pid, config_type, config in processing_tasks:

                # Check if process is still alive
                if not self._is_task_process_alive(pid, config):
                    # Process is dead, revert to Ready state
                    cursor.execute(
                        'UPDATE scheduler SET status = ?, pid = 0, process_start = ? WHERE "index" = ?',
                        ("Ready", "", task_index),
                    )
                    reverted_count += 1
                    reclaimed_configs.append(config)

            conn.commit()

        # After the commit: the note is a courtesy, and must not hold the write lock.
        for config in reclaimed_configs:
            log_orchestration_stop(
                config,
                "Requeued as Ready: found Processing with a dead PID, so the run died without "
                "reporting (daemon killed, crash, or reboot). Orchestration stop, not a scientific "
                "verdict — it will start over from where the config's flags left it.",
            )

        return reverted_count

    def _update_process_status_memory(self):
        """Check and revert killed processes for in-memory mode."""
        reverted_count = 0
        # Get all tasks with PIDs that are in Processing status
        processing_mask = (self._schedule["status"] == "Processing") & (
            (self._schedule["pid"] != 0) & (self._schedule["pid"] != None)  # noqa: E711
        ) & (
            (self._schedule["dispatch"] == "") | (self._schedule["dispatch"] == None)  # noqa: E711
        )
        processing_tasks = self._schedule[processing_mask]

        for task in processing_tasks:
            pid = task["pid"]
            task_index = task["index"]
            config_type = task["config_type"]

            # Check if process is still alive
            if not self._is_task_process_alive(pid, task["config"]):
                # Process is dead, revert to Ready state
                mask = self._schedule["index"] == task_index
                self._schedule["status"][mask] = "Ready"
                self._schedule["pid"][mask] = 0
                self._schedule["process_start"][mask] = ""
                reverted_count += 1

        return reverted_count

    def terminate_scheduler_tasks(self):
        """
        SIGTERM every non-zero PID stored on the schedule and reset those rows.

        Affected rows are set to status ``Paused`` (not ``Ready``) so the queue will not
        immediately pick them up again. Use :meth:`resume_paused_scheduler_tasks` to move
        them back to ``Ready``. ``pid`` is cleared and ``process_start`` is reset.
        Does not delete rows or modify tasks that have no PID set.
        """
        if self.use_system_queue:
            terminated = 0
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f'SELECT "index", pid FROM scheduler WHERE pid IS NOT NULL AND pid != 0 '
                    f'AND {self._LOCAL_TASK_FILTER}'
                )
                for task_index, pid in cursor.fetchall():
                    self._terminate_process(pid)
                    cursor.execute(
                        'UPDATE scheduler SET status = ?, pid = 0, process_start = ? WHERE "index" = ?',
                        ("Paused", "", task_index),
                    )
                    terminated += 1
                conn.commit()
            return terminated

        terminated = 0
        mask = (self._schedule["pid"] != 0) & (self._schedule["pid"] != None) & (  # noqa: E711
            (self._schedule["dispatch"] == "") | (self._schedule["dispatch"] == None)  # noqa: E711
        )
        for task in self._schedule[mask]:
            self._terminate_process(task["pid"])
            idx_mask = self._schedule["index"] == task["index"]
            self._schedule["status"][idx_mask] = "Paused"
            self._schedule["pid"][idx_mask] = 0
            self._schedule["process_start"][idx_mask] = ""
            terminated += 1
        return terminated

    def resume_paused_scheduler_tasks(self):
        """
        Set every ``Paused`` task to ``Ready`` so :meth:`get_next_task` can run them again.
        """
        if self.use_system_queue:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE scheduler SET status = ? WHERE status = ?", ("Ready", "Paused"))
                n = cursor.rowcount
                conn.commit()
            return n

        mask = self._schedule["status"] == "Paused"
        n = int(np.sum(mask))
        if n:
            self._schedule["status"][mask] = "Ready"
        return n

    def _terminate_process(self, pid):
        """Terminate a process safely; ignore invalid or already-dead PIDs."""
        if pid in (None, 0):
            return
        try:
            os.kill(int(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, ValueError, TypeError):
            pass

    def _is_task_process_alive(self, pid, config):
        """
        True when `pid` is alive AND still running this task's config.

        `_is_process_alive` alone is not enough, for two reasons that both produced permanent
        orphans in production:

        1. **Zombies.** A finished-but-unreaped child still has a /proc entry, so `kill(pid, 0)`
           succeeds and it reads as running. Rows behind a zombie survived every reclaim ever
           run against them.
        2. **PID recycling.** After a reboot or a busy week a stored PID can belong to an
           unrelated live process, which would keep the row Processing forever.

        Errs toward "alive" whenever /proc cannot be read — leaving one orphan is far cheaper
        than reclaiming a task that is genuinely running and launching a duplicate reduction
        over the same outputs.
        """
        if not self._is_process_alive(pid):
            return False

        try:
            with open(f"/proc/{pid}/stat", "rb") as stat_file:
                # state is the field after the (possibly space-containing) comm in parentheses
                state = stat_file.read().rpartition(b")")[2].split()[0].decode()
            with open(f"/proc/{pid}/cmdline", "rb") as cmdline_file:
                cmdline = cmdline_file.read().decode(errors="replace")
        except (FileNotFoundError, ProcessLookupError):
            return False  # exited between the two checks
        except (OSError, IndexError):
            return True  # cannot tell (hidepid, permissions) — do not risk a duplicate run

        if state == "Z":
            return False  # exited already; only the unreaped entry is left

        if not config:
            return True

        return config in cmdline

    def _is_process_alive(self, pid):
        """
        Check if a process with the given PID is still alive.

        Args:
            pid: Process ID to check

        Returns:
            bool: True if process is alive, False otherwise
        """
        if pid is None or pid == 0:
            return False

        try:
            # Signal 0 doesn't actually send a signal, it just checks if process exists
            os.kill(int(pid), 0)
            return True
        except (ProcessLookupError, OSError):
            # Process doesn't exist or permission denied
            return False
        except (ValueError, TypeError):
            # Invalid PID
            return False

    def rerun_task(self, index: int):
        if self.use_system_queue:
            return self._rerun_task_from_db(index)
        else:
            return self._rerun_task_from_memory(index)

    def _rerun_task_from_db(self, index: int):
        """Rerun a task in database mode."""
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """UPDATE scheduler 
                   SET status = ?, readiness = ?, is_ready = ?, pid = 0, 
                       process_start = ?, process_end = ?, kwargs = ?
                   WHERE "index" = ?""",
                ("Ready", 100, 1, "", "", "['-overwrite']", index),
            )
            conn.commit()
            return True

    def _rerun_task_from_memory(self, index: int):
        """Rerun a task in in-memory mode."""
        mask = self._schedule["index"] == index
        self._schedule["status"][mask] = "Ready"
        return True

    def remove_task(self, index: int):
        """Remove a task from the schedule."""
        if self.use_system_queue:
            return self._remove_task_from_db(index)
        else:
            return self._remove_from_memory(index)

    def _remove_task_from_db(self, index: int):
        """Remove a task from the schedule in database mode."""
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM scheduler WHERE "index" = ?', (index,))
            conn.commit()
            return True

    def _remove_task_from_memory(self, index: int):
        """Remove a task from the schedule in in-memory mode."""
        mask = self._schedule["index"] == index
        self._schedule = self._schedule[~mask]
        return True

    def stash_task(self, index: int):
        """Stash the schedule."""
        if self.use_system_queue:
            return self._stash_task_from_db(index)
        else:
            return self._stash_task_from_memory(index)

    def _stash_task_from_db(self, index: int):
        """Stash the schedule in database mode."""
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE scheduler SET status = 'Stashed' WHERE \"index\" = ?", (index,))
            conn.commit()
            return True

    def _stash_task_from_memory(self, index: int):
        """Stash the schedule in in-memory mode."""
        mask = self._schedule["index"] == index
        self._schedule["status"][mask] = "Stashed"
        return True

    def recover_stashed_task(self, index: int):
        """Recover a stashed task from the schedule."""
        if self.use_system_queue:
            return self._recover_stashed_task_from_db(index)
        else:
            return self._recover_stashed_task_from_memory(index)

    def _recover_stashed_task_from_db(self, index: int):
        """Recover a stashed task from the schedule in database mode."""
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE scheduler SET status = 'Ready' WHERE \"index\" = ?", (index,))
            conn.commit()
            return True

    def _recover_stashed_task_from_memory(self, index: int):
        """Recover a stashed task from the schedule in in-memory mode."""
        mask = self._schedule["index"] == index
        self._schedule["status"][mask] = "Ready"
        return True

    def _get_failed_tasks_from_db(self):
        """Get all failed tasks from the schedule."""
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT \"index\", config FROM scheduler WHERE status = 'Failed'")
            failed_tasks = cursor.fetchall()

        failed_names = [(task[0], os.path.basename(task[1]).replace(".yml", "")) for task in failed_tasks]
        return np.asarray(failed_names)
