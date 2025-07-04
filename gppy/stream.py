from .config import SciProcConfiguration
from .astrometry import Astrometry
from .photometry import Photometry
from .imstack import ImStack
from .subtract import ImSubtract
from .services.task import Task, Priority
import copy

import itertools
class_mapping = {
    "astrometry": Astrometry,
    "single_photometry": Photometry,
    "combine": ImStack,
    "combined_photometry": Photometry,
    "subtraction": ImSubtract,
}

class ReductionStream:
    """
    Represents a sequential processing pipeline containing multiple tasks.
    Tasks are executed in order, with results tracked for each step.
    """
    _id_counter = itertools.count(0)

    def __init__(self, config: str):
        self.id = f"stream_{next(self._id_counter)}"
        self.status = "pending"
        self.current_class = None
        self.config_file = config
        config = SciProcConfiguration.from_config(self.config_file)
        try:
            self.id = config.config.name
        except:
            self.id = f"stream_{next(self._id_counter)}"
        self.load_class()

    @property
    def current_class_name(self):
        return self.current_class.__class__.__name__

    def load_class(self):
        config = SciProcConfiguration.from_config(self.config_file)
        for key in ["astrometry", "single_photometry", "combine", "combined_photometry", "subtraction"]:
            if not(getattr(config.config.flag, key)):
                self.status = "processing"
                break
            else:
                key = None
        if key is None:
            self.status = "completed"
        else:
            self.current_class = class_mapping[key](config)
            self.current_tasks = copy.copy(self.current_class.sequential_task)

    def get_task(self):
        if len(self.current_tasks) == 0:
            self.load_class()

        if len(self.current_tasks) !=0:
            _, func, use_gpu = self.current_tasks.pop(0)
            return Task(getattr(self.current_class, func), priority=Priority.MEDIUM, gpu=use_gpu, task_name = f"{self.current_class_name}.{func}")
        else:
            return None

    def is_complete(self) -> bool:
        """Check if all tasks have been processed."""
        return self.status == "completed"
    
    def __repr__(self):
        return f"ReductionStream(id={self.id}, status={self.status}, current_class={self.current_class_name})"
