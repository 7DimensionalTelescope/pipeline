import json
from collections import deque
from ..config.utils import get_filter_from_config
from ..const import SCRIPT_DIR, NUM_GPUS


class Scheduler:
    """
    A task scheduler for managing dependent and independent pipeline configurations.

    This scheduler manages the execution order of pipeline tasks based on dependencies.
    It handles master configurations that must complete before dependent configurations
    can be processed, and independent configurations that can run in parallel.

    The scheduler maintains separate queues for:
    - Master configurations (must complete first)
    - Dependent configurations (wait for master completion)
    - Independent configurations (can run in parallel)

    Features:
    - Dependency-based task scheduling
    - Automatic task status tracking
    - Command generation for different task types
    - Comprehensive status reporting

    Args:
        dependent_configs (dict): Dictionary mapping master configs to their dependent configs
        independent_configs (list): List of independent configs that can run in parallel
        **kwargs: Additional configuration options including:
            - processes (list): List of processes to run (default: ["astrometry", "photometry", "combine", "subtract"])
            - overwrite (bool): Whether to overwrite existing results (default: False)
            - device_id (int): Starting GPU device ID (default: 0)
            - max_devices (int): Maximum number of GPU devices available (default: 2)

    Example:
        >>> scheduler = Scheduler(
        ...     dependent_configs={
        ...         "master1.yml": ["dependent1.yml", "dependent2.yml"],
        ...         "master2.yml": ["dependent3.yml"]
        ...     },
        ...     independent_configs=["independent1.yml", "independent2.yml"]
        ... )
    """

    def __init__(self, dependent_configs={}, independent_configs=[], is_too=False, **kwargs):
        """
        kwargs can have preprocess_kwargs, processes, overwrite, device_id, max_devices
        """

        # Initialize master queue and status tracking
        masters = list(dependent_configs.keys())
        self.master_status = {self._key_from_path(p): None for p in masters}

        # Sort dependents to prioritize broadband filters (not starting with "m")
        sorted_dependents = {}
        for k, v in dependent_configs.items():
            sorted_deps = self._sort_configs_by_filter_priority(v)
            sorted_dependents[self._key_from_path(k)] = sorted_deps

        # Sort masters: those with broadband dependents first
        sorted_masters = sorted(masters, key=lambda m: (not self._has_broadband_dependents(m, sorted_dependents), m))
        self.master_queue = deque(sorted_masters)

        # Rebuild dependents dictionary in the same order as sorted masters
        ordered_dependents = {}
        for master in sorted_masters:
            master_key = self._key_from_path(master)
            ordered_dependents[master_key] = sorted_dependents[master_key]
        self.dependents = ordered_dependents

        # Sort independents to prioritize broadband filters (not starting with "m")
        sorted_independents = self._sort_configs_by_filter_priority(independent_configs)
        self.independents = deque(sorted_independents)
        self.original_independents = set(independent_configs)
        dependent_paths = [p for group in dependent_configs.values() for p in group]
        self.task_status = {self._key_from_path(p): None for p in masters + dependent_paths + list(independent_configs)}

        self.task_queue = deque()
        self.completed_tasks = set()
        self.skipped_tasks = set()

        # Configuration options
        self.preprocess_kwargs = kwargs.get("preprocess_kwargs", {})
        self.processes = kwargs.get("processes", ["astrometry", "photometry", "combine", "subtract"])

        self.overwrite = kwargs.get("overwrite", False)
        self.device_id = kwargs.get("device_id", 0)
        self.max_devices = kwargs.get("max_devices", NUM_GPUS)  # default to maximum number of GPUs

        self.is_too = is_too

        if not self.master_status:  # enqueue independents immediately if no masters
            while self.independents:
                self.task_queue.append(self.independents.popleft())

            # Sort masters to prioritize those with broadband filter dependents

    def _has_broadband_dependents(self, master_path, sorted_dependents):
        """Check if a master has any broadband filter dependents."""
        master_key = self._key_from_path(master_path)
        dependents = sorted_dependents.get(master_key, [])
        return any(self._is_broadband_filter(dep) for dep in dependents)

    def _is_broadband_filter(self, config: str) -> bool:
        """
        Check if a config uses a broadband filter (not starting with "m").

        Args:
            config: Path to configuration file

        Returns:
            bool: True if filter is broadband (doesn't start with "m"), False otherwise
        """
        filter_name = get_filter_from_config(config)
        if filter_name is None:
            return False
        return not filter_name.startswith("m")

    def _sort_configs_by_filter_priority(self, configs):
        """
        Sort configs to prioritize broadband filters (not starting with "m").

        Args:
            configs: List or iterable of config paths

        Returns:
            list: Sorted list with broadband filters first
        """
        return sorted(configs, key=lambda config: (not self._is_broadband_filter(config), config))

    def _key_from_path(self, path: str) -> str:
        """
        Extract master key like '2025-01-01_7DT11' from the full config path.

        Args:
            path (str): Full path to configuration file

        Returns:
            str: Extracted key from the filename without extension
        """
        if not path:
            raise ValueError("Path cannot be empty")
        return path.split("/")[-1].replace(".yml", "")

    def _generate_command(self, config, task_type):
        """
        Generate command for the given config and task type.

        Creates the appropriate command line arguments for different types
        of pipeline tasks (Masterframe vs ScienceImage processing).

        Args:
            config (str): Path to configuration file
            task_type (str): Type of task ("Masterframe" or "ScienceImage")

        Returns:
            list: Command line arguments as a list of strings
        """
        if task_type == "Masterframe":
            cmd = [
                f"{SCRIPT_DIR}/bin/preprocess",
                "-config",
                config,
                "-device",
                str(int(self.device_id % self.max_devices)),
                "-make_plots",
            ]
            if self.is_too: # fmt: skip
                cmd.append("-is_too")
            if self.overwrite:
                cmd.append("-overwrite")

            if self.preprocess_kwargs:
                cmd.extend(["--preprocess_kwargs", json.dumps(self.preprocess_kwargs)])

            self.device_id += 1
        else:  # ScienceImage
            cmd = [f"{SCRIPT_DIR}/bin/data_reduction", "-config", config]
            if self.is_too: # fmt: skip
                cmd.append("-is_too")
            # Add processes as individual arguments
            cmd.append("-processes")
            cmd.extend(self.processes)

            if self.overwrite:
                cmd.append("-overwrite")

        return cmd

    def update_queue(self, key=None, success=True):
        """
        Update the task queue based on task completion.

        Handles scheduling of dependent and independent tasks based on
        master task completion status.

        Args:
            key (str, optional): The task key (extracted from task_path).
                If None, updates queue based on current master_status.
            success (bool): Whether the task completed successfully.
                Only used when key is provided.

        Notes:
            - If key is provided: handles specific task completion
            - If key is None: re-evaluates queue based on current master_status
            - Schedules dependent tasks when master tasks succeed
            - Marks dependent tasks as skipped when master tasks fail
            - Marks all independents as skipped when any master fails
            - Schedules independents only when all masters have succeeded
        """
        if key is None:
            # Re-evaluate queue based on current master_status
            all_masters_succeeded = all(v is True for v in self.master_status.values())
            any_master_failed = any(v is False for v in self.master_status.values())

            if any_master_failed:
                # Mark all dependents of failed masters as skipped
                for master_key, master_success in self.master_status.items():
                    if master_success is False:
                        for dep in self.dependents.get(master_key, []):
                            dep_key = self._key_from_path(dep)
                            if self.task_status.get(dep_key) is None:
                                self.task_status[dep_key] = False
                                self.skipped_tasks.add(dep)

                # Mark all independents as skipped when any master fails
                while self.independents:
                    mu = self.independents.popleft()
                    mu_key = self._key_from_path(mu)
                    if self.task_status.get(mu_key) is None:
                        self.task_status[mu_key] = False
                        self.skipped_tasks.add(mu)

            if all_masters_succeeded:
                # Schedule all dependents of succeeded masters that haven't been scheduled yet
                for master_key, master_success in self.master_status.items():
                    if master_success is True:
                        for dep in self.dependents.get(master_key, []):
                            # Only add if not already in queue or completed
                            dep_key = self._key_from_path(dep)
                            if dep not in self.task_queue and dep_key not in self.completed_tasks:
                                self.task_queue.append(dep)

                # Schedule independents only if all masters succeeded
                while self.independents:
                    mu = self.independents.popleft()
                    self.task_queue.append(mu)
        else:
            # Handle specific task completion
            # Case 1: Masterframe
            if key in self.master_status:
                if success:
                    # Schedule dependent tasks when master succeeds
                    for dep in self.dependents.get(key, []):
                        self.task_queue.append(dep)
                else:
                    # Mark dependent tasks as failed when master fails
                    for dep in self.dependents.get(key, []):
                        dep_key = self._key_from_path(dep)
                        self.task_status[dep_key] = False
                        self.skipped_tasks.add(dep)

                    # Mark all independents as skipped when any master fails
                    while self.independents:
                        mu = self.independents.popleft()
                        mu_key = self._key_from_path(mu)
                        self.task_status[mu_key] = False
                        self.skipped_tasks.add(mu)

            # Schedule independents only if all masters succeeded
            if all(v is True for v in self.master_status.values()):
                while self.independents:
                    mu = self.independents.popleft()
                    self.task_queue.append(mu)

    def mark_done(self, task_path, success=True):
        """
        Mark a task as completed and schedule follow-up tasks if needed.

        This method updates the status of completed tasks and manages the
        scheduling of dependent tasks based on master task completion.

        Args:
            task_path (str): Path to the completed task configuration
            success (bool): Whether the task completed successfully

        Notes:
            - Updates task status and completion tracking
            - Schedules dependent tasks when master tasks succeed
            - Marks dependent tasks as skipped when master tasks fail
            - Manages independent task scheduling based on master completion
        """
        key = self._key_from_path(task_path)
        self.task_status[key] = success
        if success:
            self.completed_tasks.add(task_path)
        else:
            self.skipped_tasks.add(task_path)

        # Update master status if this is a master task
        if key in self.master_status:
            self.master_status[key] = success

        self.update_queue(key, success)

    def get_next_task(self):
        """
        Get next task command from the appropriate queue.

        Returns the next task to be executed, prioritizing master tasks
        over dependent and independent tasks.

        Returns:
            list or None: Command line arguments for the next task, or None if no tasks available
        """

        if self.is_too:
            priority = "high"
        else:
            priority = "normal"

        if self.master_queue:
            config = self.master_queue.popleft()
            cmd = self._generate_command(config, "Masterframe")
            return cmd, priority

        if self.task_queue:
            config = self.task_queue.popleft()
            cmd = self._generate_command(config, "ScienceImage")
            return cmd, priority

        return None, priority

    def is_all_done(self):
        """
        Check if all tasks have been completed.

        Returns:
            bool: True if all tasks are finished, False otherwise
        """

        return (
            not self.master_queue
            and not self.task_queue
            and not self.independents
            and all(v is not None for v in self.task_status.values())
        )

    def report_number_of_tasks(self):
        """
        Get progress report showing completion status by task type.

        Returns:
            dict: Dictionary with progress strings for master, dependent, and independent tasks
        """
        # Count total tasks
        total_masters = len(self.master_status)
        total_dependents = sum(len(deps) for deps in self.dependents.values())
        total_independents = len(self.original_independents)

        # Count completed tasks
        completed_masters = sum(1 for status in self.master_status.values() if status is not None)

        # Count dependent tasks that are completed (have non-None status)
        dependent_keys = set()
        for deps in self.dependents.values():
            dependent_keys.update(self._key_from_path(dep) for dep in deps)
        completed_dependents = sum(
            1 for task, status in self.task_status.items() if task in dependent_keys and status is not None
        )

        # Count independent tasks that are completed (have non-None status)
        independent_keys = set(self._key_from_path(ind) for ind in self.original_independents)
        completed_independents = sum(
            1 for task, status in self.task_status.items() if task in independent_keys and status is not None
        )

        # Count successful and failed tasks by category
        successful_masters = sum(1 for v in self.master_status.values() if v is True)
        failed_masters = sum(1 for v in self.master_status.values() if v is False)

        successful_dependents = sum(
            1 for task, status in self.task_status.items() if task in dependent_keys and status is True
        )
        failed_dependents = sum(
            1 for task, status in self.task_status.items() if task in dependent_keys and status is False
        )

        successful_independents = sum(
            1 for task, status in self.task_status.items() if task in independent_keys and status is True
        )
        failed_independents = sum(
            1 for task, status in self.task_status.items() if task in independent_keys and status is False
        )

        return {
            "master": f"{completed_masters} out of {total_masters} (success: {successful_masters}, failed: {failed_masters})",
            "dependent": f"{completed_dependents} out of {total_dependents} (success: {successful_dependents}, failed: {failed_dependents})",
            "independent": f"{completed_independents} out of {total_independents} (success: {successful_independents}, failed: {failed_independents})",
        }

    def status_of_queues(self):
        """
        Get status of all queues.

        Returns:
            dict: Dictionary containing status of all queues
        """
        return {
            "master": self.master_queue,
            "dependent": self.task_queue,
        }

    def get_failed_configs(self):
        """
        Return config identifiers that failed (status == False).
        """
        return [key for key, status in self.task_status.items() if status is False]

    def get_failed_or_skipped_configs(self):
        """
        Return full paths of configs that failed or were skipped.
        """
        return list(self.skipped_tasks)


# from astropy.table import Table, vstack
# import numpy as np
# from gppy.const import SCRIPT_DIR


# class Scheduler:

#     _empty_schedule = Table(
#         dtype=[
#             ("index", int),
#             ("config", str),
#             ("type", str),
#             ("input_type", str),
#             ("is_ready", bool),
#             ("priority", int),
#             ("readiness", int),
#             ("status", str),
#             ("dependent_idx", list),
#         ]
#     )

#     def __init__(self, schedule=None):
#         if schedule is not None:
#             if isinstance(schedule, Table):
#                 if schedule.colnames == self._empty_schedule.colnames:
#                     self._schedule = schedule
#                 else:
#                     raise ValueError("Invalid schedule type")
#             elif isinstance(schedule, Scheduler):
#                 self._schedule = schedule.schedule
#             else:
#                 raise ValueError("Invalid schedule type")
#         else:
#             self._schedule = self._empty_schedule

#     def __add__(self, other):
#         offset = max(self.schedule["index"])

#         if isinstance(other, Scheduler):
#             other_table = other.schedule.copy()
#         elif isinstance(other, Table):
#             other_table = other.copy()
#         else:
#             raise ValueError("Invalid schedule type")

#         # Adjust the index column by adding the offset
#         other_table["index"] = other_table["index"] + offset

#         # Adjust all values in dependent_idx by adding the offset to each value in the lists
#         for i in range(len(other_table)):
#             if other_table["dependent_idx"][i]:  # If the list is not empty
#                 other_table["dependent_idx"][i] = [idx + offset for idx in other_table["dependent_idx"][i]]

#         # Combine the tables
#         self._schedule = vstack([self.schedule, other_table])
#         return self._schedule

#     def __getitem__(self, index):
#         return self._schedule[self._schedule["index"] == index]

#     def __repr__(self):
#         total_jobs = len(self.schedule)
#         in_ready = len(self.schedule[self.schedule["status"] == "Ready"])
#         in_pending = len(self.schedule[self.schedule["status"] == "Pending"])
#         in_processing = len(self.schedule[self.schedule["status"] == "Processing"])
#         in_completed = len(self.schedule[self.schedule["status"] == "Completed"])
#         is_master = len(self.schedule[self.schedule["type"] == "masterframe"])
#         is_science = len(self.schedule[self.schedule["type"] == "science"])
#         self.schedule.pprint_all(max_lines=10)
#         return f"Scheduler with {total_jobs} (masterframe: {is_master} and science: {is_science}) jobs: \n {in_ready} ready, \n {in_pending} pending, \n {in_processing} processing, \n {in_completed} completed"

#     def print_schedule(self):
#         self.schedule.pprint_all()

#     @property
#     def schedule(self):

#         self._schedule.sort(["is_ready", "priority", "readiness"], reverse=True)

#         config = self._schedule["config"]
#         vals, counts = np.unique(config, return_counts=True)

#         dups = vals[counts > 1]
#         if len(dups) > 0:
#             raise ValueError(f"Duplicate configs exist in the schedule: {dups}")

#         return self._schedule

#     def add_schedule(self, other):
#         self._schedule = self + other
#         return self

#     def get_next_task(self):
#         schedule = self.schedule[self.schedule["status"] != "Completed"]
#         return schedule[0]

#     def get_next_cmd(self):
#         job = self.get_next_task()
#         return self._generate_command(job["index"])

#     def list_of_ready_jobs(self):
#         return self.schedule[self.schedule["is_ready"]]

#     def mark_done(self, index):
#         # Mark the current task as completed
#         mask = self._schedule["index"] == index
#         self._schedule["status"][mask] = "Completed"

#         # Get the dependent indices for this task
#         dependent_indices = self._schedule["dependent_idx"][mask][0]

#         # For each dependent index, increment its readiness by 1
#         for dep_idx in dependent_indices:
#             dep_mask = self._schedule["index"] == dep_idx
#             self._schedule["readiness"][dep_mask] += 1
#             if self._schedule["readiness"][dep_mask] >= 100:
#                 self._schedule["status"][dep_mask] = "Ready"
#                 self._schedule["is_ready"][dep_mask] = True

#     def is_all_done(self):
#         return len(self.schedule[self.schedule["status"] != "Completed"]) == 0

#     def clear_schedule(self, all=False):
#         if all:
#             self._schedule = self._empty_schedule

#         else:
#             self._schedule = self._schedule[self._schedule["status"] != "Completed"]

#     def _generate_command(self, index, **kwargs):

#         job = self[index]
#         overwrite = kwargs.get("overwrite", False)
#         config = job["config"][0]
#         is_too = job["input_type"][0] == "too"

#         if job["type"] == "masterframe":
#             cmd = [
#                 f"{SCRIPT_DIR}/bin/preprocess",
#                 "-config",
#                 config,
#                 "-make_plots",
#             ]
#             if is_too:
#                 cmd.append("-is_too")
#             if overwrite:
#                 cmd.append("-overwrite")

#             if kwargs.get("preprocess_kwargs", None):
#                 cmd.extend(["--preprocess_kwargs", json.dumps(kwargs["preprocess_kwargs"])])

#         else:  # ScienceImage
#             cmd = [f"{SCRIPT_DIR}/bin/data_reduction", "-config", config]

#             if is_too:
#                 cmd.append("-is_too")

#             cmd.append("-processes")
#             cmd.extend(self.processes)

#             if overwrite:
#                 cmd.append("-overwrite")

#         return cmd
