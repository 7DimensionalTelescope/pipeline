from collections import deque, defaultdict


class Scheduler:
    def __init__(self, masters, dependents, multiunits):
        self.master_queue = deque(masters)
        self.master_status = {self._key_from_path(p): False for p in masters}
        self.dependents = dependents  # key: master_name, value: list of dependent paths
        self.ready_dependents = defaultdict(deque)
        self.multiunits = deque(multiunits)
        self.completed_tasks = set()
        self.task_queue = deque()

    def _key_from_path(self, path):
        """Extract master key like '2025-01-01_7DT11' from the full config path"""
        return path.split("/")[-1].replace(".yml", "")

    def mark_done(self, task_path):
        """Mark a task as completed, and schedule any follow-ups"""
        self.completed_tasks.add(task_path)

        key = self._key_from_path(task_path)

        if key in self.master_status:
            self.master_status[key] = True

            for dep in self.dependents.get(key, []):
                self.task_queue.append(dep)

        if all(self.master_status.values()):
            all_deps_done = all(
                dep in self.completed_tasks for dep_list in self.dependents.values() for dep in dep_list
            )
            if all_deps_done:
                # Schedule final tasks
                while self.multiunits:
                    self.task_queue.append(self.multiunits.popleft())

    def get_next_task(self):
        """Fetch the next task to run"""
        if self.master_queue:
            return self.master_queue.popleft(), "Masterframe"
        if self.task_queue:
            return self.task_queue.popleft(), "ScienceImage"
        return None

    def is_all_done(self):
        return (
            not self.master_queue
            and not self.task_queue
            and all(self.master_status.values())
            and all(dep in self.completed_tasks for dep_list in self.dependents.values() for dep in dep_list)
            and all(task in self.completed_tasks for task in self.final_tasks)
        )
