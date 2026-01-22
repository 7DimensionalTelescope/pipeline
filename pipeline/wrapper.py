from .services.blueprint import Blueprint
from .services.scheduler import Scheduler
from .services.queue import QueueManager


class DataReduction:

    def __init__(
        self,
        input_params=None,
        list_of_images=None,
        use_db=True,
        overwrite=False,
        master_frame_only=False,
        is_too=False,
        **kwargs,
    ):

        self.is_too = is_too

        self.blueprint = Blueprint(
            input_params,
            list_of_images=list_of_images,
            use_db=use_db,
            master_frame_only=master_frame_only,
            is_too=is_too,
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

    def run(
        self,
        *,
        overwrite=False,
        overwrite_config=False,
        overwrite_data=False,
        overwrite_preprocess=False,
        overwrite_science=False,
        overwrite_schedule=False,
        max_workers=50,
        base_priority=None,
        processes=["astrometry", "photometry", "coadd", "subtract"],
        queue=None,
        preprocess_kwargs=None,
        is_too=False,
        use_system_queue=False,
        input_type=None,
    ):

        overwrite_config = overwrite_config or overwrite
        overwrite_data = overwrite_data or overwrite

        is_too = is_too or self.is_too

        if not self._created_config:
            self.create_config(
                overwrite=overwrite_config,
                max_workers=max_workers,
                is_too=is_too,
            )
            self._created_config = True

        self.blueprint.create_schedule(
            is_too=is_too,
            base_priority=base_priority,
            processes=processes,
            overwrite=overwrite_data,
            overwrite_preprocess=overwrite_preprocess,
            overwrite_science=overwrite_science,
            preprocess_kwargs=preprocess_kwargs,
            input_type=input_type,
        )

        sc = Scheduler(
            self.blueprint.schedule,
            is_too=is_too,
            use_system_queue=use_system_queue,
            overwrite_schedule=overwrite_schedule,
        )
        if use_system_queue:
            sc.start_system_queue()
        else:
            queue = QueueManager()
            queue.add_scheduler(sc)
            queue.wait_until_task_complete()
