from collections import deque, defaultdict


class Scheduler:
    def __init__(self, masters, dependent_configs, multiunit_configs):
        self.master_queue = deque(masters)
        self.master_status = {self._key_from_path(p): None for p in masters}  # True/False/None
        self.dependents = dependent_configs  # key: master_key, value: list of dependent paths
        self.multiunits = deque(multiunit_configs)

        self.task_queue = deque()
        self.completed_tasks = set()
        self.skipped_tasks = set()

        dependent_paths = [p for group in dependent_configs.values() for p in group]
        self.task_status = {p: None for p in masters + dependent_paths + list(multiunit_configs)}

    def _key_from_path(self, path):
        """Extract master key like '2025-01-01_7DT11' from the full config path"""
        return path.split("/")[-1].replace(".yml", "")

    def mark_done(self, task_path, success=True):
        """Mark a task as completed and schedule follow-up if needed"""
        self.task_status[task_path] = success
        self.completed_tasks.add(task_path) if success else self.skipped_tasks.add(task_path)

        key = self._key_from_path(task_path)

        # Case 1: Masterframe
        if key in self.master_status:
            self.master_status[key] = success
            if success:
                for dep in self.dependents.get(key, []):
                    self.task_queue.append(dep)
            else:
                # If master failed, mark dependents as skipped
                for dep in self.dependents.get(key, []):
                    self.task_status[dep] = False
                    self.skipped_tasks.add(dep)
                
                # Also mark all multiunits as skipped when any master fails
                while self.multiunits:
                    mu = self.multiunits.popleft()
                    self.task_status[mu] = False
                    self.skipped_tasks.add(mu)

        # Case 2: Normal task (dependent or multiunit)
        # No action needed here; we just track its status

        # Schedule multiunits only if all masters succeeded
        if all(v is True for v in self.master_status.values()):
            all_required_deps = [dep for k, v in self.master_status.items() if v for dep in self.dependents.get(k, [])]
            if all(self.task_status[dep] is not None for dep in all_required_deps):
                while self.multiunits:
                    mu = self.multiunits.popleft()
                    self.task_queue.append(mu)

    def get_next_task(self):
        if self.master_queue:
            return self.master_queue.popleft(), "Masterframe"
        if self.task_queue:
            return self.task_queue.popleft(), "ScienceImage"
        return None

    def is_all_done(self):
        return not self.master_queue and not self.task_queue and all(v is not None for v in self.task_status.values())

    def report_status(self):
        return {
            "succeeded": [p for p, v in self.task_status.items() if v is True],
            "failed": [p for p, v in self.task_status.items() if v is False],
            "pending": [p for p, v in self.task_status.items() if v is None],
        }
