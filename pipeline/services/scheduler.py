import json
import sqlite3
import socket
import os
from contextlib import contextmanager
from ..const import SCRIPT_DIR, NUM_GPUS, SCHEDULER_DB_PATH, QUEUE_SOCKET_PATH
from astropy.table import Table, vstack
import numpy as np


class Scheduler:

    _empty_schedule = Table(
        dtype=[
            ("index", int),
            ("config", str),
            ("type", str),
            ("input_type", str),
            ("is_ready", bool),
            ("priority", int),
            ("readiness", int),
            ("status", str),
            ("dependent_idx", list),
            ("pid", int),
            ("original_status", str),
        ]
    )

    # SQL ordering clause used consistently throughout
    _ORDER_BY = 'ORDER BY is_ready DESC, priority DESC, readiness DESC, "index" ASC'

    # Constants
    MAX_MASTERFRAMES = 6
    HIGH_PRIORITY_THRESHOLD = 10

    def __init__(self, schedule=None, use_system_queue=False, **kwargs):
        self._kwargs = kwargs
        self.use_system_queue = use_system_queue and SCHEDULER_DB_PATH is not None

        if self.use_system_queue:
            self._schedule = None
            self._connection_check()
            if schedule is not None:
                self._save_table_to_db(self._validate_and_get_table(schedule))
            # Initialize processing_masterframes from database
            self.processing_masterframes = 0
        else:
            self._schedule = self._validate_and_get_table(schedule) if schedule is not None else self._empty_schedule
            # Initialize processing_masterframes from current schedule
            if schedule is not None:
                self.processing_masterframes = len(
                    self._schedule[
                        (self._schedule["status"] == "Processing") & (self._schedule["type"] == "masterframe")
                    ]
                )
            else:
                self.processing_masterframes = 0

    def _connection_check(self):
        """Create scheduler table if it doesn't exist."""
        import os

        with self._db_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scheduler (
                    "index" INTEGER PRIMARY KEY,
                    config TEXT NOT NULL,
                    type TEXT NOT NULL,
                    input_type TEXT NOT NULL,
                    is_ready INTEGER NOT NULL,
                    priority INTEGER NOT NULL,
                    readiness INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    dependent_idx TEXT,
                    pid INTEGER,
                    original_status TEXT
                )
            """
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
    def _db_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(SCHEDULER_DB_PATH)
        try:
            yield conn
        finally:
            conn.close()

    def _row_to_dict(self, row):
        """Convert database row tuple to dictionary."""
        return {
            "index": row[0],
            "config": row[1],
            "type": row[2],
            "input_type": row[3],
            "is_ready": bool(row[4]),
            "priority": row[5],
            "readiness": row[6],
            "status": row[7],
            "dependent_idx": json.loads(row[8]) if row[8] else [],
            "pid": row[9] if len(row) > 9 else None,
            "original_status": row[10] if len(row) > 10 else None,
        }

    def _rows_to_table(self, rows):
        """Convert database rows to astropy Table."""
        if not rows:
            return self._empty_schedule

        data = {col: [] for col in self._empty_schedule.colnames}
        for row in rows:
            data["index"].append(row[0])
            data["config"].append(row[1])
            data["type"].append(row[2])
            data["input_type"].append(row[3])
            data["is_ready"].append(bool(row[4]))
            data["priority"].append(row[5])
            data["readiness"].append(row[6])
            data["status"].append(row[7])
            data["dependent_idx"].append(json.loads(row[8]) if row[8] else [])
            data["pid"].append(row[9] if len(row) > 9 and row[9] is not None else None)
            data["original_status"].append(row[10] if len(row) > 10 and row[10] is not None else None)

        return Table(data)

    def _check_duplicates(self):
        """Check for duplicate configs in database."""
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
                raise ValueError(f"Duplicate configs exist in the schedule: {dup_configs}")

    def __add__(self, other):
        current_table = self.schedule
        offset = max(current_table["index"]) if len(current_table) > 0 else -1

        other_table = other.schedule.copy() if isinstance(other, Scheduler) else other.copy()
        if not isinstance(other_table, Table):
            raise ValueError("Invalid schedule type")

        # Adjust indices
        other_table["index"] = other_table["index"] + offset + 1
        for i in range(len(other_table)):
            if other_table["dependent_idx"][i]:
                other_table["dependent_idx"][i] = [idx + offset + 1 for idx in other_table["dependent_idx"][i]]

        combined_table = vstack([current_table, other_table])

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
        total_jobs = len(schedule)
        in_ready = len(schedule[schedule["status"] == "Ready"])
        in_pending = len(schedule[schedule["status"] == "Pending"])
        in_processing = len(schedule[schedule["status"] == "Processing"])
        in_completed = len(schedule[schedule["status"] == "Completed"])
        is_master = len(schedule[schedule["type"] == "masterframe"])
        is_science = len(schedule[schedule["type"] == "science"])
        if with_table:
            schedule.pprint_all(max_lines=10)
        return f"Scheduler with {total_jobs} (masterframe: {is_master} and science: {is_science}) jobs: {in_ready} ready, {in_pending} pending, {in_processing} processing, and {in_completed} completed"

    @property
    def schedule(self):
        if self.use_system_queue:
            self._check_duplicates()
            return self._get_table_from_db()
        else:
            table = self._schedule
            table.sort(["is_ready", "priority", "readiness"], reverse=True)
            # Check duplicates for in-memory table
            config = table["config"]
            vals, counts = np.unique(config, return_counts=True)
            dups = vals[counts > 1]
            if len(dups) > 0:
                raise ValueError(f"Duplicate configs exist in the schedule: {dups}")
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

    def get_next_task(self, **kwargs):
        """Get the next task to process with priority and concurrency constraints."""
        if not self.has_schedule:
            return None

        if self.use_system_queue:
            return self._get_next_task_db(**kwargs)
        else:
            return self._get_next_task_memory(**kwargs)

    def _get_next_task_db(self, **kwargs):
        """Get next task from database."""
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            try:
                # Check masterframe count
                cursor.execute(
                    "SELECT COUNT(*) FROM scheduler WHERE status = ? AND type = ?", ("Processing", "masterframe")
                )
                masterframes = cursor.fetchone()[0]

                # Build query with constraints
                query = 'SELECT "index" FROM scheduler WHERE status = ?'
                params = ["Ready"]

                # Enforce masterframe limit
                if masterframes >= self.MAX_MASTERFRAMES:
                    query += " AND type != ?"
                    params.append("masterframe")

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

                # Find next job
                cursor.execute(query, tuple(params))
                index_row = cursor.fetchone()
                if not index_row:
                    conn.rollback()
                    return None, None

                job_index = index_row[0]

                # Mark as Processing
                cursor.execute(
                    'UPDATE scheduler SET status = ? WHERE "index" = ? AND status = ?',
                    ("Processing", job_index, "Ready"),
                )
                if cursor.rowcount == 0:
                    conn.rollback()
                    return None, None

                # Get full job data
                cursor.execute('SELECT * FROM scheduler WHERE "index" = ?', (job_index,))
                row = cursor.fetchone()
                conn.commit()
            except Exception:
                conn.rollback()
                return None, None

        row = self._row_to_dict(row) if row else None
        if row is None:
            return None, None

        overwrite = kwargs.get("overwrite", True)
        return row, self._generate_command(row["index"], overwrite=overwrite, **kwargs)

    def _get_next_task_memory(self, **kwargs):
        """Get next task from in-memory schedule."""
        ready_jobs = self.schedule[(self.schedule["status"] == "Ready")]
        if len(ready_jobs) == 0:
            return None, None

        # Enforce masterframe limit
        if self.processing_masterframes >= self.MAX_MASTERFRAMES:
            ready_jobs = ready_jobs[ready_jobs["type"] != "masterframe"]

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
            ready_jobs = ready_jobs[ready_jobs["priority"] > self.HIGH_PRIORITY_THRESHOLD]
        else:
            # Check for TOO processing
            too_processing = (
                len(
                    self.schedule[
                        (self.schedule["status"] == "Processing") & (self.schedule["input_type"].str.lower() == "too")
                    ]
                )
                > 0
            )
            if too_processing:
                ready_jobs = ready_jobs[ready_jobs["input_type"].str.lower() == "too"]

        if len(ready_jobs) == 0:
            return None, None

        # Get first job (already sorted by _ORDER_BY in schedule property)
        row_dict = {col: ready_jobs[col][0] for col in ready_jobs.colnames}
        job_index = row_dict["index"]

        # Mark as Processing
        mask = self._schedule["index"] == job_index
        self._schedule["status"][mask] = "Processing"

        if row_dict.get("type") == "masterframe":
            self.processing_masterframes += 1

        overwrite = kwargs.get("overwrite", True)
        return row_dict, self._generate_command(job_index, overwrite=overwrite, **kwargs)

    def _mark_done_memory(self, job_index, success=True):
        mask = self._schedule["index"] == job_index
        if len(self._schedule[mask]) == 0:
            return

        # Check if already marked as done to prevent duplicate processing
        row_dict = {col: self._schedule[col][mask][0] for col in self._schedule.colnames}
        current_status = row_dict["status"]

        if current_status == "Completed" or current_status == "Failed":
            return

        if success:
            # Get job info
            job_type = row_dict["type"]
            dependent_indices = row_dict["dependent_idx"]

            if job_type == "masterframe":
                self.processing_masterframes -= 1
                if self.processing_masterframes < 0:
                    self.processing_masterframes = 0

            self._schedule["status"][mask] = "Completed"
            self._schedule["pid"][mask] = 0

            for dep_idx in dependent_indices:
                dep_mask = self._schedule["index"] == dep_idx
                self._schedule["readiness"][dep_mask] += 1

                if self._schedule["readiness"][dep_mask] >= 100:
                    self._schedule["readiness"][dep_mask] = 100
                    self._schedule["status"][dep_mask] = "Ready"
                    self._schedule["is_ready"][dep_mask] = True

        else:
            # Check if this is a retry (priority is already 0)
            current_priority = self._schedule["priority"][mask][0]
            self._schedule["pid"][mask] = 0

            if current_priority == 0:
                # This is a retry that failed again - set readiness to 0, is_ready to false, and mark as Failed
                self._schedule["status"][mask] = "Failed"
                self._schedule["readiness"][mask] = 0
                self._schedule["is_ready"][mask] = False
            else:
                # First failure - set priority to 0 and status to Pending for retry
                self._schedule["status"][mask] = "Ready"
                self._schedule["priority"][mask] = 0

    def list_of_ready_jobs(self):
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

    def mark_done(self, index, success=True):
        if self.use_system_queue:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                # Check if job is already marked as done to prevent duplicate processing
                cursor.execute('SELECT status, dependent_idx, type FROM scheduler WHERE "index" = ?', (index,))
                row = cursor.fetchone()
                if not row:
                    return

                current_status, dependent_idx_json, job_type = row
                # If already marked as done, skip to prevent duplicate increments
                if current_status == "Completed" or current_status == "Failed":
                    return

                dependent_indices = json.loads(dependent_idx_json) if dependent_idx_json else []

                if success:
                    new_status = "Completed"
                    # Mark as Completed and clear PID
                    cursor.execute('UPDATE scheduler SET status = ?, pid = 0 WHERE "index" = ?', (new_status, index))
                else:
                    # Check if this is a retry (priority is already 0)
                    cursor.execute('SELECT priority FROM scheduler WHERE "index" = ?', (index,))
                    priority_row = cursor.fetchone()
                    current_priority = priority_row[0] if priority_row else None

                    if current_priority == 0:
                        # This is a retry that failed again - set readiness to 0, is_ready to false, and mark as Failed
                        cursor.execute(
                            'UPDATE scheduler SET status = ?, readiness = ?, is_ready = ?, pid = 0 WHERE "index" = ?',
                            ("Failed", 0, 0, index),
                        )
                    else:
                        # First failure - set priority to 0 and status to Ready for retry
                        cursor.execute(
                            'UPDATE scheduler SET status = ?, priority = ?, pid = 0 WHERE "index" = ?',
                            ("Ready", 0, index),
                        )

                # Only increment readiness for dependent jobs if the job completed successfully
                if success:
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

                conn.commit()
        else:
            self._mark_done_memory(index, success)

    def is_all_done(self):
        if self.use_system_queue:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM scheduler WHERE status != ?", ("Completed",))
                return cursor.fetchone()[0] == 0
        return len(self.schedule[self.schedule["status"] != "Completed"]) == 0

    def clear_schedule(self, all=False):
        import signal

        if self.use_system_queue:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                # Get PIDs of jobs to be cleared before deleting
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

    def _generate_command(self, index, **kwargs):
        kwargs = {**self._kwargs, **kwargs}

        if self.use_system_queue:
            with self._db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT config, type, input_type FROM scheduler WHERE "index" = ?', (index,))
                row = cursor.fetchone()
                if not row:
                    raise ValueError(f"Job with index {index} not found")
                config, job_type, input_type = row
        else:
            # Find job by index in schedule
            mask = self._schedule["index"] == index
            if len(self._schedule[mask]) == 0:
                raise ValueError(f"Job with index {index} not found")
            config = self._schedule["config"][mask][0]
            job_type = self._schedule["type"][mask][0]
            input_type = self._schedule["input_type"][mask][0]

        is_too = input_type.lower() == "too"
        overwrite = kwargs.get("overwrite", False)
        processes = kwargs.get("processes", ["astrometry", "photometry", "combine", "subtract"])

        if job_type == "masterframe":
            cmd = [f"{SCRIPT_DIR}/bin/preprocess", "-config", config, "-make_plots"]
            if is_too:
                cmd.append("-is_too")
            if overwrite:
                cmd.append("-overwrite")
            if kwargs.get("preprocess_kwargs", None):
                cmd.extend(["--preprocess_kwargs", json.dumps(kwargs["preprocess_kwargs"])])
        else:  # ScienceImage
            cmd = [f"{SCRIPT_DIR}/bin/data_reduction", "-config", config]
            if is_too:
                cmd.append("-is_too")
            cmd.append("-processes")
            cmd.extend(processes)
            if overwrite:
                cmd.append("-overwrite")

        return cmd

    def _get_table_from_db(self):
        """Create astropy Table from SQLite database."""
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM scheduler {self._ORDER_BY}")
            return self._rows_to_table(cursor.fetchall())

    def _save_table_to_db(self, table):
        """Save astropy Table to SQLite database."""
        with self._db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM scheduler")

            for row in table:
                dependent_idx_json = json.dumps(row["dependent_idx"]) if row["dependent_idx"] else None
                pid = row.get("pid") if "pid" in row.colnames else 0
                original_status = row.get("original_status") if "original_status" in row.colnames else None
                cursor.execute(
                    """INSERT INTO scheduler 
                       ("index", config, type, input_type, is_ready, priority, readiness, status, dependent_idx, pid, original_status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        int(row["index"]),
                        str(row["config"]),
                        str(row["type"]),
                        str(row["input_type"]),
                        1 if row["is_ready"] else 0,
                        int(row["priority"]),
                        int(row["readiness"]),
                        str(row["status"]),
                        dependent_idx_json,
                        int(pid) if pid is not None else 0,
                        str(original_status) if original_status is not None else None,
                    ),
                )

            conn.commit()
