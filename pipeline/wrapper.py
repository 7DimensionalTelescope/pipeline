from .services.blueprint import Blueprint
from .services.scheduler import Scheduler
from .services.queue import QueueManager


class DataReduction:

    def __init__(self, input_params=None, list_of_images=None, use_db=False, overwrite=False, **kwargs):

        master_frame_only = kwargs.get("master_frame_only", False)
        self.is_too = kwargs.get("is_too", False)
        self.base_priority = kwargs.get("base_priority", None)
        use_db = kwargs.get("use_db", True)

        self.blueprint = Blueprint(
            input_params,
            list_of_images=list_of_images,
            use_db=use_db,
            master_frame_only=master_frame_only,
            **kwargs,
        )
        self._created_config = False

    @property
    def schedule(self):
        return self.blueprint.schedule

    def create_config(self, overwrite=False, max_workers=50, is_too=False):
        if not self._created_config:
            self.blueprint.create_config(overwrite=overwrite, max_workers=max_workers, is_too=is_too)
            self._created_config = True
            self.blueprint.create_schedule(is_too=is_too, base_priority=self.base_priority)

    def run(
        self,
        *,
        overwrite=False,
        overwrite_config=False,
        overwrite_data=False,
        overwrite_preprocess=False,
        overwrite_science=False,
        max_workers=50,
        processes=["astrometry", "photometry", "combine", "subtract"],
        queue=None,
        preprocess_kwargs=None,
        is_too=False,
        use_system_queue=False,
    ):

        overwrite_config = overwrite_config or overwrite
        overwrite_data = overwrite_data or overwrite

        is_too = is_too or self.is_too

        if not self._created_config:
            self.create_config(overwrite=overwrite_config, max_workers=max_workers, is_too=is_too)
            self._created_config = True

        sc = Scheduler(
            self.blueprint.schedule,
            processes=processes,
            overwrite=overwrite_data,
            preprocess_kwargs=preprocess_kwargs,
            is_too=is_too,
            use_system_queue=use_system_queue,
            overwrite_preprocess=overwrite_preprocess,
            overwrite_science=overwrite_science,
        )
        if use_system_queue:
            sc.start_system_queue()
        else:
            queue = QueueManager()
            queue.add_scheduler(sc)
            queue.wait_until_task_complete()
