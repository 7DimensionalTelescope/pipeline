import json
import sqlite3
import socket
import os
import numpy as np
from contextlib import contextmanager
from datetime import datetime
from astropy.table import Table, vstack


from ..const import SCRIPTS_DIR, NUM_GPUS, SCHEDULER_DB_PATH, QUEUE_SOCKET_PATH


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
            ("kwargs", object),
            ("process_start", object),
            ("process_end", object),
        ]
    )

    # SQL ordering clause used consistently throughout
    _ORDER_BY = 'ORDER BY is_ready DESC, priority DESC, readiness DESC, "index" ASC'

    # Constants
    MAX_PREPROCESS = 3
    HIGH_PRIORITY_THRESHOLD = 10

    def __init__(
        self,
        schedule=None,
        use_system_queue=False,
        overwrite_schedule=False,
        **kwargs,
    ):
        self.use_system_queue = use_system_queue and SCHEDULER_DB_PATH is not None

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
    def from_list(cls, list_of_configs, base_priority=1, use_system_queue=False, **kwargs):
        """Create a scheduler from a list of configs."""
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

            # scheduler_kwargs = ["-overwrite"] if overwrite or overwrite_science else [] + ["-processes"] + processes
            if kwargs.pop("overwrite", False):
                scheduler_kwargs = ["-overwrite"]
            else:
                scheduler_kwargs = []

            table.add_row(
                [
                    idx,
                    config,
                    task_type,
                    "User-input",
                    True,
                    priority,
                    100,
                    "Ready",
                    [],
                    0,
                    scheduler_kwargs,
                    "",
                    "",
                ]
            )

        return cls(schedule=table, use_system_queue=use_system_queue, **kwargs)

    def _connection_check(self):
        """Create scheduler table if it doesn't exist."""

        with self._db_connection() as conn:
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
    def _db_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(SCHEDULER_DB_PATH)
        try:
            yield conn
        finally:
            conn.close()

    def _row_to_dict(self, row):
        """Convert database row tuple to dictionary."""

        if isinstance(row[10], str):
            kwargs_str = row[10].replace("'", '"')
            kwargs = json.loads(kwargs_str)
        elif isinstance(row[10], list):
            kwargs = row[10]
        else:
            kwargs = []

        return {
            "index": row[0],
            "config": row[1],
            "config_type": row[2],
            "input_type": row[3],
            "is_ready": bool(row[4]),
            "priority": row[5],
            "readiness": row[6],
            "status": row[7],
            "dependent_idx": json.loads(row[8]) if row[8] else [],
            "pid": row[9],
            "kwargs": kwargs,
            "process_start": row[11],
            "process_end": row[12],
        }

    def _rows_to_table(self, rows):
        """Convert database rows to astropy Table."""
        if not rows:
            return self._empty_schedule

        data = {col: [] for col in self._empty_schedule.colnames}

        for row in rows:

            if isinstance(row[10], str):
                kwargs_str = row[10].replace("'", '"')
                kwargs = json.loads(kwargs_str)
            elif isinstance(row[10], list):
                kwargs = row[10]
            else:
                kwargs = []

            data["index"].append(row[0])
            data["config"].append(row[1])
            data["config_type"].append(row[2])
            data["input_type"].append(row[3])
            data["is_ready"].append(bool(row[4]))
            data["priority"].append(row[5])
            data["readiness"].append(row[6])
            data["status"].append(row[7])
            data["dependent_idx"].append(json.loads(row[8]) if row[8] else [])
            data["pid"].append(row[9])
            data["kwargs"].append(kwargs)
            data["process_start"].append(row[11])
            data["process_end"].append(row[12])

        return Table(data, dtype=self._empty_schedule.dtype)

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
        is_preprocess = len(schedule[schedule["config_type"] == "preprocess"])
        is_science = len(schedule[schedule["config_type"] == "science"])
        is_failed = len(schedule[schedule["status"] == "Failed"])
        if with_table:
            schedule.pprint_all(max_lines=10)
        return f"Scheduler with {total_tasks} (preprocess: {is_preprocess} and science: {is_science}) tasks: {in_ready} ready, {in_pending} pending, {in_processing} processing, {is_failed} failed, and {in_completed} completed"

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
            return self._get_next_task_db()
        else:
            return self._get_next_task_memory()

    def _get_next_task_db(self):
        """Get next task from database."""
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            try:
                # Check preprocess count
                cursor.execute(
                    "SELECT COUNT(*) FROM scheduler WHERE status = ? AND config_type = ?", ("Processing", "preprocess")
                )
                preprocess = cursor.fetchone()[0]

                # Build query with constraints
                query = 'SELECT "index" FROM scheduler WHERE status = ?'
                params = ["Ready"]

                # Enforce preprocess limit
                if preprocess >= self.MAX_PREPROCESS:
                    query += " AND config_type != ?"
                    params.append("preprocess")

                # Check for high priority processing
                cursor.execute(
                    "SELECT COUNT(*) FROM scheduler WHERE status = ? AND priority > ?",
                    ("Processing", self.HIGH_PRIORITY_THRESHOLD),
                )
                if cursor.fetchone()[0] > 0:
                    query += " AND priority > ?"
                    params.append(self.HIGH_PRIORITY_THRESHOLD)
                else:
                    # Check for TOO processing
                    cursor.execute(
                        "SELECT COUNT(*) FROM scheduler WHERE status = ? AND LOWER(input_type) = ?",
                        ("Processing", "too"),
                    )
                    if cursor.fetchone()[0] > 0:
                        query += " AND LOWER(input_type) = ?"
                        params.append("too")

                query += f" {self._ORDER_BY} LIMIT 1"

                # Find next task
                cursor.execute(query, tuple(params))
                index_row = cursor.fetchone()
                if not index_row:
                    conn.rollback()
                    return None, None

                task_index = index_row[0]

                # Mark as Processing and set process_start
                process_start = datetime.now().isoformat()
                cursor.execute(
                    'UPDATE scheduler SET status = ?, process_start = ?, process_end = ? WHERE "index" = ? AND status = ?',
                    ("Processing", process_start, "", task_index, "Ready"),
                )
                if cursor.rowcount == 0:
                    conn.rollback()
                    return None, None

                # Get full task data
                cursor.execute('SELECT * FROM scheduler WHERE "index" = ?', (task_index,))
                row = cursor.fetchone()
                conn.commit()
            except Exception:
                conn.rollback()
                return None, None

        row = self._row_to_dict(row) if row else None
        if row is None:
            return None, None

        if row["priority"] == 0 and ["-overwrite"] not in row["kwargs"]:
            row["kwargs"].append("-overwrite")

        return row, self._generate_command(row["index"], row["kwargs"])

    def _get_next_task_memory(self):
        """Get next task from in-memory schedule."""
        ready_tasks = self.schedule[(self.schedule["status"] == "Ready")]
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

    def mark_done(self, index, success=True):
        if self.use_system_queue:
            self._mark_done_db(index, success)
        else:
            self._mark_done_memory(index, success)

    def _mark_done_db(self, index, success=True):
        with self._db_connection() as conn:
            cursor = conn.cursor()
            # Check if task is already marked as done to prevent duplicate processing
            cursor.execute('SELECT status, dependent_idx, config_type FROM scheduler WHERE "index" = ?', (index,))
            row = cursor.fetchone()
            if not row:
                return

            current_status, dependent_idx_json, config_type = row
            # If already marked as done, skip to prevent duplicate increments
            if current_status == "Completed" or current_status == "Failed":
                return

            dependent_indices = json.loads(dependent_idx_json) if dependent_idx_json else []

            if success:
                new_status = "Completed"
                # Mark as Completed, clear PID, and set process_end
                process_end = datetime.now().isoformat()
                cursor.execute(
                    'UPDATE scheduler SET status = ?, pid = 0, process_end = ? WHERE "index" = ?',
                    (new_status, process_end, index),
                )
                for dep_idx in dependent_indices:
                    cursor.execute('SELECT readiness FROM scheduler WHERE "index" = ?', (dep_idx,))
                    dep_row = cursor.fetchone()
                    if dep_row:
                        new_readiness = dep_row[0] + 1

                        if new_readiness > 100:
                            new_readiness = 100

                        if new_readiness == 100:
                            cursor.execute(
                                'UPDATE scheduler SET readiness = ?, status = ?, is_ready = ? WHERE "index" = ?',
                                (new_readiness, "Ready", 1, dep_idx),
                            )
                        else:
                            cursor.execute(
                                'UPDATE scheduler SET readiness = ? WHERE "index" = ?', (new_readiness, dep_idx)
                            )
            else:
                process_end = datetime.now().isoformat()
                cursor.execute(
                    'UPDATE scheduler SET status = ?, readiness = ?, is_ready = ?, pid = 0, process_end = ? WHERE "index" = ?',
                    ("Failed", 0, 0, process_end, index),
                )

            conn.commit()

    def _mark_done_memory(self, task_index, success=True):
        mask = self._schedule["index"] == task_index
        if len(self._schedule[mask]) == 0:
            return

        # Check if already marked as done to prevent duplicate processing
        row_dict = {col: self._schedule[col][mask][0] for col in self._schedule.colnames}
        current_status = row_dict["status"]

        if current_status == "Completed" or current_status == "Failed":
            return

        if success:
            # Get task info
            config_type = row_dict["config_type"]
            dependent_indices = row_dict["dependent_idx"]

            if config_type == "preprocess":
                self.processing_preprocess -= 1
                if self.processing_preprocess < 0:
                    self.processing_preprocess = 0

            self._schedule["status"][mask] = "Completed"
            self._schedule["pid"][mask] = 0
            self._schedule["process_end"][mask] = datetime.now().isoformat()

            for dep_idx in dependent_indices:
                dep_mask = self._schedule["index"] == dep_idx
                self._schedule["readiness"][dep_mask] += 1

                if self._schedule["readiness"][dep_mask] >= 100:
                    self._schedule["readiness"][dep_mask] = 100
                    self._schedule["status"][dep_mask] = "Ready"
                    self._schedule["is_ready"][dep_mask] = True

        else:
            # Check if this is a retry (priority is already 0)
            self._schedule["pid"][mask] = 0
            self._schedule["status"][mask] = "Failed"
            self._schedule["readiness"][mask] = 0
            self._schedule["is_ready"][mask] = False
            self._schedule["process_end"][mask] = datetime.now().isoformat()

    def list_of_ready_tasks(self):
        if self.use_system_queue:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT * FROM scheduler WHERE is_ready = 1 {self._ORDER_BY}")
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

    def rerun_failed_tasks(self):
        """
        Rerun all failed tasks by changing their status to Ready with priority 1 and readiness 100.

        Returns:
            int: Number of tasks that were updated
        """
        if self.use_system_queue:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """UPDATE scheduler 
                       SET status = ?, priority = ?, readiness = ?, is_ready = ?, pid = 0, 
                           process_start = ?, process_end = ?, input_type = ?, kwargs = ?
                       WHERE status = ?""",
                    ("Ready", 0, 100, 1, "", "", "Reprocess", "['-overwrite']", "Failed"),
                )
                conn.commit()
                return cursor.rowcount
        else:
            # Handle in-memory schedule
            mask = self._schedule["status"] == "Failed"
            count = np.sum(mask)
            if count > 0:
                self._schedule["status"][mask] = "Ready"
                self._schedule["priority"][mask] = 1
                self._schedule["readiness"][mask] = 100
                self._schedule["is_ready"][mask] = True
                self._schedule["pid"][mask] = 0
                self._schedule["process_start"][mask] = ""
                self._schedule["process_end"][mask] = ""
                self._schedule["input_type"][mask] = "Reprocess"
                self._schedule["kwargs"][mask] = "['-overwrite']"
            return count

    def clear_schedule(self, all=False):
        import signal

        if self.use_system_queue:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                # Get completed tasks to save before deleting
                if all:
                    cursor.execute("SELECT * FROM scheduler")
                else:
                    cursor.execute("SELECT * FROM scheduler WHERE status = ?", ("Completed",))

                completed_rows = cursor.fetchall()

                # Save completed tasks to file if any exist
                if completed_rows:
                    completed_table = self._rows_to_table(completed_rows)
                    self._save_completed_to_file(completed_table)

                # Get PIDs of tasks to be cleared before deleting
                if all:
                    cursor.execute("SELECT pid FROM scheduler WHERE pid IS NOT NULL")
                else:
                    cursor.execute("SELECT pid FROM scheduler WHERE status = ? AND pid IS NOT NULL", ("Completed",))

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
            db_dir = os.path.dirname(SCHEDULER_DB_PATH)
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

        is_too = str(input_type).lower() == "too" or "_ToO_" in config

        if config_type == "preprocess":
            cmd = [f"{SCRIPTS_DIR}/preprocess", "-config", config, "-make_plots"]
            if is_too:
                cmd.append("-is_too")
            cmd.extend(scheduler_kwargs)
        elif config_type == "science":
            cmd = [f"{SCRIPTS_DIR}/data_reduction", "-config", config]
            if is_too:
                cmd.append("-is_too")
            cmd.extend(scheduler_kwargs)
        elif config_type == "debug":
            cmd = [f"{SCRIPTS_DIR}/debug", "-config", config]
        else:
            raise ValueError(f"Invalid systemd queue config_type: {config_type}")

        return cmd

    def _get_table_from_db(self):
        """Create astropy Table from SQLite database."""
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM scheduler {self._ORDER_BY}")
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

                        print(f"Replaced {len(duplicate_configs)} existing schedule(s) with new ones")

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
                    kwargs = row.get("kwargs") if "kwargs" in row.colnames else None
                    process_start = row.get("process_start") if "process_start" in row.colnames else None
                    process_end = row.get("process_end") if "process_end" in row.colnames else None

                    cursor.execute(
                        """INSERT INTO scheduler 
                           ("index", config, config_type, input_type, is_ready, priority, readiness, status, dependent_idx, pid, kwargs, process_start, process_end)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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

    def update_process_status(self):

        if self.use_system_queue:
            return self._update_process_status_db()
        else:
            return self._update_process_status_memory()

    def _update_process_status_db(self):
        """Check and revert killed processes for database mode."""
        reverted_count = 0
        with self._db_connection() as conn:
            cursor = conn.cursor()
            # Get all tasks with PIDs that are in Processing status
            cursor.execute(
                'SELECT "index", pid, config_type FROM scheduler WHERE status = ? AND pid IS NOT NULL',
                ("Processing",),
            )
            processing_tasks = cursor.fetchall()

            for task_index, pid, config_type in processing_tasks:

                # Check if process is still alive
                if not self._is_process_alive(pid):
                    # Process is dead, revert to Ready state
                    cursor.execute(
                        'UPDATE scheduler SET status = ?, pid = 0, process_start = ? WHERE "index" = ?',
                        ("Ready", "", task_index),
                    )
                    reverted_count += 1

            conn.commit()
        return reverted_count

    def _update_process_status_memory(self):
        """Check and revert killed processes for in-memory mode."""
        reverted_count = 0
        # Get all tasks with PIDs that are in Processing status
        processing_mask = (self._schedule["status"] == "Processing") & (
            (self._schedule["pid"] != 0) & (self._schedule["pid"] != None)
        )
        processing_tasks = self._schedule[processing_mask]

        for task in processing_tasks:
            pid = task["pid"]
            task_index = task["index"]
            config_type = task["config_type"]

            # Check if process is still alive
            if not self._is_process_alive(pid):
                # Process is dead, revert to Ready state
                mask = self._schedule["index"] == task_index
                self._schedule["status"][mask] = "Ready"
                self._schedule["pid"][mask] = 0
                self._schedule["process_start"][mask] = ""
                reverted_count += 1

        return reverted_count

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
