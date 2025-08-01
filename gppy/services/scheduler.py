from collections import deque, defaultdict
from ..const import SCRIPT_DIR


class Scheduler:
    def __init__(self, dependent_configs={}, independent_configs=[], **kwargs):
        masters = list(dependent_configs.keys())
        self.master_queue = deque(masters)
        self.master_status = {self._key_from_path(p): None for p in masters}  # True/False/None
        # Convert dependents config to use extracted keys
        self.dependents = {self._key_from_path(k): v for k, v in dependent_configs.items()}
        self.independents = deque(independent_configs)
        self.original_independents = set(independent_configs)  # Keep track of original independents

        self.task_queue = deque()
        self.completed_tasks = set()
        self.skipped_tasks = set()

        dependent_paths = [p for group in dependent_configs.values() for p in group]
        # Use extracted keys for task_status instead of full paths
        self.task_status = {self._key_from_path(p): None for p in masters + dependent_paths + list(independent_configs)}

        self.processes = kwargs.get("processes", ["astrometry", "photometry", "combine", "subtract"])
        self.overwrite = kwargs.get("overwrite", False)
        self.device_id = 0

    def _key_from_path(self, path):
        """Extract master key like '2025-01-01_7DT11' from the full config path"""
        return path.split("/")[-1].replace(".yml", "")

    def _generate_command(self, config, task_type):
        """Generate command for the given config and task type"""
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
            cmd.extend(["-processes"] + self.processes)
            if self.overwrite:
                cmd.append("-overwrite")

        return cmd

    def mark_done(self, task_path, success=True):
        """Mark a task as completed and schedule follow-up if needed"""
        key = self._key_from_path(task_path)
        self.task_status[key] = success
        self.completed_tasks.add(task_path) if success else self.skipped_tasks.add(task_path)

        # Case 1: Masterframe
        if key in self.master_status:
            self.master_status[key] = success
            if success:
                for dep in self.dependents.get(key, []):

                    self.task_queue.append(dep)
            else:
                # If master failed, mark dependents as skipped
                for dep in self.dependents.get(key, []):
                    self.task_status[self._key_from_path(dep)] = False
                    self.skipped_tasks.add(dep)

                # Also mark all multiunits as skipped when any master fails
                while self.independents:
                    mu = self.independents.popleft()
                    self.task_status[self._key_from_path(mu)] = False
                    self.skipped_tasks.add(mu)

        # Case 2: Normal task (dependent or multiunit)
        # No action needed here; we just track its status

        # Schedule multiunits only if all masters succeeded
        if all(v is True for v in self.master_status.values()):
            all_required_deps = [dep for k, v in self.master_status.items() if v for dep in self.dependents.get(k, [])]
            if all(self.task_status[self._key_from_path(dep)] is not None for dep in all_required_deps):
                while self.independents:
                    mu = self.independents.popleft()
                    self.task_queue.append(mu)

    def get_next_task(self):
        """Get next task command"""
        if self.master_queue:
            config = self.master_queue.popleft()
            cmd = self._generate_command(config, "Masterframe")
            return cmd
        if self.task_queue:
            config = self.task_queue.popleft()
            cmd = self._generate_command(config, "ScienceImage")
            return cmd
        return None

    def is_all_done(self):
        return (
            not self.master_queue
            and not self.task_queue
            and not self.independents
            and all(v is not None for v in self.task_status.values())
        )

    def report_status(self):
        return {
            "succeeded": [p for p, v in self.task_status.items() if v is True],
            "failed": [p for p, v in self.task_status.items() if v is False],
            "pending": [p for p, v in self.task_status.items() if v is None],
        }

    def report_number_of_tasks(self):
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
