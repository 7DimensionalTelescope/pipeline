import os
from concurrent.futures import ThreadPoolExecutor
from collections import UserDict
import time

from .utils import flatten
from .path import PathHandler
from .run import query_observations
from .config import PreprocConfiguration, SciProcConfiguration

from .services.queue import QueueManager

from itertools import chain
from .services.task import Task, Priority

from .services.scheduler import Scheduler


class SortedGroupDict(UserDict):
    """A dictionary that sorts its values when iterating."""

    def __getitem__(self, key):
        if type(key) == int:
            return self.values()[key]
        else:
            return super().__getitem__(key)

    def __iter__(self):
        # First sort by type (MasterframeGroup first, then ScienceGroup)
        # Then within each type, sort by their respective criteria
        return iter(self._get_sorted_values())

    def values(self):
        return self._get_sorted_values()

    def items(self):
        sorted_values = self._get_sorted_values()
        return [(getattr(v, "key", None), v) for v in sorted_values]

    def _get_sorted_values(self):
        # Separate MasterframeGroup and ScienceGroup
        masterframe_groups = []
        science_groups = []

        for value in self.data.values():
            if isinstance(value, MasterframeGroup):
                masterframe_groups.append(value)
            else:
                science_groups.append(value)

        # Sort MasterframeGroup by sci_keys length (descending)
        sorted_masterframe = sorted(masterframe_groups, key=lambda x: len(x.sci_keys), reverse=True)

        # Sort ScienceGroup by image_files length (descending)
        sorted_science = sorted(science_groups, key=lambda x: len(x.image_files), reverse=True)

        # Return MasterframeGroup first, then ScienceGroup
        return sorted_masterframe + sorted_science

    def __repr__(self):
        string = ""
        for value in self.values():
            string += str(value) + "\n"
        return string


class DataReduction:
    """overwrite=True to rewrite configs"""

    groups = SortedGroupDict()
    _unified_key_list = None  # Will be populated after initialization
    _key_usage_map = None  # Will track which keys are used in which groups
    input_params = None  # No input parameters for this method

    _multi_unit_config = set()

    def __init__(self, input_params, use_db=False, ignore_mult_date=False, **kwargs):
        self.input_params = input_params
        print("Globbing images with parameters:", input_params)

        self.list_of_images = query_observations(input_params, use_db=use_db, **kwargs)

        print(f"Found {len(self.list_of_images)} images.")
        if len(self.list_of_images) == 0:
            print("No images found")
            return
        print("Grouping images...")
        self.initialize(ignore_mult_date=ignore_mult_date)
        print("Blueprint initialized.")

    @classmethod
    def from_list(cls, list_of_images):
        if not all(f.endswith(".fits") for f in list_of_images):
            raise ValueError("Non-fits images in input")
        self = cls.__new__(cls)
        self.list_of_images = list_of_images
        self.initialize()
        print("Blueprint initialized from user-input list.")
        return self

    def initialize(self, ignore_mult_date=False):
        # [raw bdf, mframes, sci_dict]
        image_inventory = PathHandler.take_raw_inventory(self.list_of_images, ignore_mult_date=ignore_mult_date)

        if len(image_inventory) == 0:
            self.logger.warning(f"No group for wrapper out of {self.list_of_images}\nPossibly due to NUM_MIN_CALIB")

        for i, group in enumerate(image_inventory):
            try:
                # mfg_key = PathHandler(group[0][2][0]).config_stem
                sci_dict = group[2]
                flattened_group_0 = flatten(group[0])
                if sci_dict:
                    sample_file = flatten(next(iter(sci_dict.values())))[0]
                else:
                    sample_file = flattened_group_0[0]
                mfg_key = PathHandler(sample_file).config_stem

            except:
                mfg_key = f"mfg_{i}"
                print(f"Failed to extract mfg_key from {group}. Assigned a default key {mfg_key}")

            if mfg_key in self.groups:
                self.groups[mfg_key].add_images(flattened_group_0)
            else:
                mfg = MasterframeGroup(mfg_key)
                mfg.add_images(flattened_group_0)
                self.groups[mfg_key] = mfg

            for key, images in group[2].items():
                if key not in self.groups:
                    self.groups[key] = ScienceGroup(key)
                else:
                    self.groups[key].multi_units = True
                flattened_images = flatten(images[0])
                self.groups[key].add_images(flattened_images)
                self.groups[mfg_key].add_images(flattened_images)
                self.groups[mfg_key].add_sci_keys(key)

    def create_config(self, overwrite=False):
        kwargs = {"overwrite": overwrite}
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(group.create_config, **kwargs) for group in self.groups.values()]
            for f in futures:
                f.result()

    def config_list(self):
        dependent_configs = dict()
        multiunit_config = set()
        for _, group in self.groups.items():
            if isinstance(group, ScienceGroup):
                continue

            dependent_configs.setdefault(group.config, [])
            for scikey in group.sci_keys:
                sci_group = self.groups[scikey]
                if not (sci_group.multi_units):
                    dependent_configs[group.config].append(sci_group.config)
                else:
                    multiunit_config.add(sci_group.config)

        return dependent_configs, multiunit_config

    def process_all(
        self,
        preprocess_only=False,
        make_plots=False,
        overwrite=True,
        processes=["astrometry", "photometry", "combine", "subtract"],
        queue=None,
    ):
        if queue is None:
            from .services.queue import QueueManager

            queue = QueueManager()

        configs = self.config_list()
        sc = Scheduler(*configs, processes=processes, overwrite=overwrite, preprocess_only=preprocess_only)
        queue.add_scheduler(sc)
        queue.wait_until_task_complete("all")

        # self.queue = QueueManager()
        # masterframe_ids = []
        # for i, (key, group) in enumerate(self.groups.items()):
        #     if isinstance(group, MasterframeGroup):
        #         if i < 2:
        #             device_id = i
        #         else:
        #             device_id = "CPU"
        #         pre_task = group.get_task(device_id=device_id, only_with_sci=only_with_sci, make_plots=make_plots)
        #         self.queue.add_task(pre_task)
        #         masterframe_ids.append([pre_task, group])
        #         time.sleep(1)

        # if preprocess_only:
        #     self.queue.wait_until_task_complete("all")
        #     return

        # while True:
        #     for task, group in masterframe_ids:
        #         if task.status == "completed":
        #             for key in group.sci_keys:
        #                 sci_group = self.groups[key]
        #                 if not (sci_group.multi_units):
        #                     self.queue.add_task(sci_group.get_task())
        #                 else:
        #                     self._multi_unit_config.add(sci_group.config)
        #             masterframe_ids.remove([task, group])

        #     time.sleep(1)

        #     if len(masterframe_ids) == 0:
        #         break

        # for config in self._multi_unit_config:
        #     from .run import run_scidata_reduction

        #     sci_task = Task(
        #         run_scidata_reduction,
        #         kwargs={"config": config, "processes": ["astrometry", "photometry", "combine", "subtract"]},
        #         priority=Priority.MEDIUM,
        #     )
        #     self.queue.add_task(sci_task)

        # for key, group in self.groups.items():
        #     if isinstance(group, MasterframeGroup):
        #         pre_task = group.get_task(device_id=None, only_with_sci=False, make_plots=True, priority=Priority.LOW)
        #         self.queue.add_task(pre_task)

        # self.queue.wait_until_task_complete("all")


class MasterframeGroup:
    def __init__(self, key):
        self.key = key
        self._image_files = []
        self._config = None
        self.sci_keys = []

    def __lt__(self, other):
        if isinstance(other, MasterframeGroup):
            # For MasterframeGroups, higher sci_keys count means higher priority
            # Reverse the comparison to make higher count come first
            return len(self.sci_keys) < len(other.sci_keys)
        else:
            # MasterframeGroup always comes before other types
            return False

    def __eq__(self, other):
        if hasattr(other, "sci_keys"):
            return len(self.sci_keys) == len(other.sci_keys)
        else:
            return False

    @property
    def config(self):
        if self._config is None:
            self.create_config()
        return self._config

    @property
    def image_files(self):
        return self._image_files

    def add_images(self, filepath):
        if isinstance(filepath, list):
            self._image_files.extend(filepath)
        elif isinstance(filepath, str):
            self._image_files.append(filepath)
        elif isinstance(filepath, tuple):
            _tmp_list = list(chain.from_iterable(filepath))
            self._image_files.extend(_tmp_list)
        else:
            raise ValueError("Invalid filepath type")

    def add_sci_keys(self, keys):
        self.sci_keys.append(keys)

    def create_config(self, overwrite=False):
        c = PreprocConfiguration(self.image_files, overwrite=overwrite)
        self._config = c.config_file

    def get_task(self, device_id=None, only_with_sci=False, make_plots=True, priority=Priority.PREPROCESS, **kwargs):
        from .run import run_preprocess

        prep_task = Task(
            run_preprocess,
            kwargs={
                "config": self.config,
                "make_plots": make_plots,
                "only_with_sci": only_with_sci,
                "device_id": device_id,
            },
            priority=priority,
        )

        return prep_task

    def __repr__(self):
        return f"MasterframeGroup({self.key} used in {self.sci_keys} with {len(self.image_files)} images)"


class ScienceGroup:
    def __init__(self, key):
        self.key = key
        self.image_files = []
        self._config = None
        self.multi_units = False

    @property
    def config(self):
        if self._config is None:
            self.create_config()
        return self._config

    def __lt__(self, other):
        if isinstance(other, MasterframeGroup):
            # ScienceGroup always comes after MasterframeGroup
            return False
        elif isinstance(other, ScienceGroup):
            # For ScienceGroups, higher image_files count means higher priority
            # Reverse the comparison to make higher count come first
            return len(self.image_files) < len(other.image_files)
        else:
            return True

    def __eq__(self, other):
        if not isinstance(other, ScienceGroup):
            return False
        return self.key == other.key

    def add_images(self, filepath):
        if isinstance(filepath, list):
            self.image_files.extend(filepath)
        elif isinstance(filepath, str):
            self.image_files.append(filepath)
        else:
            raise ValueError("Invalid filepath type")

    def create_config(self, overwrite=False):
        sci_yml = PathHandler(self.image_files).sciproc_output_yml
        if os.path.exists(sci_yml) and not overwrite:
            # If the config file already exists, load it
            c = SciProcConfiguration.from_config(sci_yml, write=True)
        else:
            c = SciProcConfiguration(self.image_files)
        self._config = c.config_file

    def get_task(self, priority=Priority.MEDIUM, **kwargs):
        from .run import run_scidata_reduction

        sci_task = Task(
            run_scidata_reduction,
            kwargs={"config": self.config, "processes": ["astrometry", "photometry", "combine", "subtract"]},
            priority=priority,
        )

        return sci_task

    def __repr__(self):
        return f"ScienceGroup({self.key} with {len(self.image_files)} images)"
