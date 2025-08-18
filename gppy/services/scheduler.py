from collections import deque
from ..const import SCRIPT_DIR


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
        preprocess_only (bool): If True, only run masterframe tasks (default: False)
        **kwargs: Additional configuration options including:
            - processes (list): List of processes to run (default: ["astrometry", "photometry", "combine", "subtract"])
            - overwrite (bool): Whether to overwrite existing results (default: False)
            - device_id (int): Starting GPU device ID (default: 0)

    Example:
        >>> scheduler = Scheduler(
        ...     dependent_configs={
        ...         "master1.yml": ["dependent1.yml", "dependent2.yml"],
        ...         "master2.yml": ["dependent3.yml"]
        ...     },
        ...     independent_configs=["independent1.yml", "independent2.yml"]
        ... )
    """

    def __init__(self, dependent_configs={}, independent_configs=[], preprocess_only=False, **kwargs):
        # Initialize master queue and status tracking
        masters = list(dependent_configs.keys())
        self.master_queue = deque(masters)
        self.master_status = {self._key_from_path(p): None for p in masters}

        # Convert dependents config to use extracted keys
        self.dependents = {self._key_from_path(k): v for k, v in dependent_configs.items()}

        # Only track independents if not in preprocess_only mode
        if not preprocess_only:
            self.independents = deque(independent_configs)
            self.original_independents = set(independent_configs)
        else:
            self.independents = deque()
            self.original_independents = set()

        # Task tracking
        self.task_queue = deque()
        self.completed_tasks = set()
        self.skipped_tasks = set()

        # Build task status tracking - only for tasks that will actually run
        dependent_paths = [p for group in dependent_configs.values() for p in group]
        if preprocess_only:
            # In preprocess_only mode, only track master tasks
            self.task_status = {self._key_from_path(p): None for p in masters}
        else:
            # Track all tasks in normal mode
            self.task_status = {
                self._key_from_path(p): None for p in masters + dependent_paths + list(independent_configs)
            }

        # Configuration options
        self.processes = kwargs.get("processes", ["astrometry", "photometry", "combine", "subtract"])
        self.overwrite = kwargs.get("overwrite", False)
        self.device_id = 0
        self.preprocess_only = preprocess_only

    def _key_from_path(self, path):
        """
        Extract master key like '2025-01-01_7DT11' from the full config path.

        Args:
            path (str): Full path to configuration file

        Returns:
            str: Extracted key from the filename without extension
        """
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
                str(int(self.device_id % 2)),
                "-make_plots",
            ]
            if self.overwrite:
                cmd.append("-overwrite")
            self.device_id += 1
        else:  # ScienceImage
            cmd = [
                f"{SCRIPT_DIR}/bin/data_reduction",
                "-config",
                config,
            ]
            # Add processes as individual arguments
            cmd.append("-processes")
            cmd.extend(self.processes)

            if self.overwrite:
                cmd.append("-overwrite")

        return cmd

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
        self.completed_tasks.add(task_path) if success else self.skipped_tasks.add(task_path)

        # Case 1: Masterframe
        if key in self.master_status:
            self.master_status[key] = success
            if success and not self.preprocess_only:
                # Schedule dependent tasks when master succeeds (only if not preprocess_only)
                for dep in self.dependents.get(key, []):
                    self.task_queue.append(dep)
            elif not success:
                # Mark dependent tasks as failed when master fails
                for dep in self.dependents.get(key, []):
                    self.task_status[self._key_from_path(dep)] = False
                    self.skipped_tasks.add(dep)

                # Mark all independents as skipped when any master fails
                while self.independents:
                    mu = self.independents.popleft()
                    self.task_status[self._key_from_path(mu)] = False
                    self.skipped_tasks.add(mu)

        # Schedule independents only if all masters succeeded and not in preprocess_only mode
        if not self.preprocess_only and all(v is True for v in self.master_status.values()):
            all_required_deps = [dep for k, v in self.master_status.items() if v for dep in self.dependents.get(k, [])]
            if all(self.task_status[self._key_from_path(dep)] is not None for dep in all_required_deps):
                while self.independents:
                    mu = self.independents.popleft()
                    self.task_queue.append(mu)

    def get_next_task(self):
        """
        Get next task command from the appropriate queue.

        Returns the next task to be executed, prioritizing master tasks
        over dependent and independent tasks.

        Returns:
            list or None: Command line arguments for the next task, or None if no tasks available
        """
        if self.master_queue:
            config = self.master_queue.popleft()
            cmd = self._generate_command(config, "Masterframe")
            return cmd
        if self.task_queue and not self.preprocess_only:
            config = self.task_queue.popleft()
            cmd = self._generate_command(config, "ScienceImage")
            return cmd
        return None

    def is_all_done(self):
        """
        Check if all tasks have been completed.

        Returns:
            bool: True if all tasks are finished, False otherwise
        """
        if self.preprocess_only:
            return not self.master_queue and all(v is not None for v in self.task_status.values())
        else:
            return (
                not self.master_queue
                and not self.task_queue
                and not self.independents
                and all(v is not None for v in self.task_status.values())
            )

    def report_status_detailed(self):
        """
        Get detailed status report of all tasks.

        Returns:
            dict: Dictionary containing lists of succeeded, failed, and pending tasks
        """
        return {
            "succeeded": [p for p, v in self.task_status.items() if v is True],
            "failed": [p for p, v in self.task_status.items() if v is False],
            "pending": [p for p, v in self.task_status.items() if v is None],
        }

    def report_status(self):
        """
        Get summary status report with task counts.

        Returns:
            dict: Dictionary containing counts of succeeded, failed, and pending tasks
        """
        succeeded_count = len([p for p, v in self.task_status.items() if v is True])
        failed_count = len([p for p, v in self.task_status.items() if v is False])
        pending_count = len([p for p, v in self.task_status.items() if v is None])

        return {
            "succeeded": succeeded_count,
            "failed": failed_count,
            "pending": pending_count,
        }

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
        completed_dependents = sum(
            1
            for task, status in self.task_status.items()
            if status is not None
            and task in [self._key_from_path(dep) for deps in self.dependents.values() for dep in deps]
        )
        completed_independents = sum(
            1
            for task, status in self.task_status.items()
            if status is not None and task in [self._key_from_path(ind) for ind in self.original_independents]
        )

        return {
            "master": f"{completed_masters} out of {total_masters}",
            "dependent": f"{completed_dependents} out of {total_dependents}",
            "independent": f"{completed_independents} out of {total_independents}",
        }
