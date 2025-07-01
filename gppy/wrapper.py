import os
import threading
from collections import UserDict
import time

from .utils import flatten
from .path import PathHandler
from .query import query_observations
from .config import PreprocConfiguration, SciProcConfiguration

from .services.queue import QueueManager

from itertools import chain


def glob_files_from_db(params):
    return []


def glob_files_by_param(keywords, **kwargs):
    if isinstance(keywords, dict):
        keywords = list(keywords.values())
    return query_observations(keywords, **kwargs)


class DataReduction:
    """overwrite=True to rewrite configs"""

    def __init__(self, input_params, use_db=False, **kwargs):
        self.groups = SortedGroupDict()  # use a sorted dictionary
        self._multi_unit_config = set()

        self._unified_key_list = None  # Will be populated after initialization
        self._key_usage_map = None  # Will track which keys are used in which groups

        self.input_params = input_params
        print("Globbing images with parameters:", input_params)

        if use_db:
            self.list_of_images = glob_files_from_db(input_params)
        else:
            self.list_of_images = glob_files_by_param(input_params, **kwargs)

        print(f"Found {len(self.list_of_images)} images.")
        print("Grouping images...")
        self.initialize()
        print("Blueprint initialized.")

        self.queue = QueueManager()

    @classmethod
    def from_list(cls, list_of_images):
        if not all(f.endswith(".fits") for f in list_of_images):
            raise ValueError("Non-fits images in input")
        self = cls.__new__(cls)
        self.list_of_images = list_of_images
        self.groups = SortedGroupDict()
        self._multi_unit_config = set()
        self._unified_key_list = None
        self._key_usage_map = None
        self.input_params = None  # No input parameters for this method
        self.queue = QueueManager()
        self.initialize()
        print("Blueprint initialized from user-input list.")
        return self

    def initialize(self):
        image_inventory = PathHandler.take_raw_inventory(self.list_of_images)  # [raw bdf, mframes, sci_dict]

        for i, group in enumerate(image_inventory):
            try:
                # mfg_key = PathHandler(group[0][2][0]).config_stem
                sci_dict = group[2]
                if sci_dict:
                    sample_file = flatten(next(iter(sci_dict.values())))[0]
                else:
                    sample_file = flatten(group[0])[0]
                mfg_key = PathHandler(sample_file).config_stem

            except:
                mfg_key = f"mfg_{i}"
                print(f"Failed to extract mfg_key from {group}. Assigned a default key {mfg_key}")

            if mfg_key in self.groups:
                self.groups[mfg_key].add_images(group[0])
            else:
                mfg = MasterframeGroup(mfg_key)
                mfg.add_images(group[0])
                self.groups[mfg_key] = mfg
            for key, images in group[2].items():
                if key not in self.groups:
                    self.groups[key] = ScienceGroup(key)
                else:
                    self.groups[key].multi_units = True
                self.groups[key].add_images(images[0])
                self.groups[mfg_key].add_images(images[0])
                self.groups[mfg_key].add_sci_keys(key)

    def create_config(self, overwrite=False):
        threads = []
        kwargs = {"overwrite": overwrite}
        for group in self.groups.values():
            t = threading.Thread(target=group.create_config, kwargs=kwargs)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    def process_all(self):
        from .stream import ReductionStream

        masterframe_ids = []
        for i, (key, group) in enumerate(self.groups.items()):
            if isinstance(group, MasterframeGroup):
                # Submit preprocess
                pre_task = group.get_tasks(device_id=i % 2, only_with_sci=True, make_plots=False)
                self.queue.add_task(pre_task)
                masterframe_ids.append([pre_task, group])

        while True:
            for task, group in masterframe_ids:
                if task.status == "completed":
                    for key in group.sci_keys:
                        sci_group = self.groups[key]
                        if not (sci_group.multi_units):
                            sci_stream = sci_group.get_stream()
                            self.queue.add_stream(sci_stream)
                        else:
                            self._multi_unit_config.add(sci_group.config)
                    masterframe_ids.remove([task, group])
            
            time.sleep(1)

            if len(masterframe_ids) == 0:
                break

        for config in self._multi_unit_config:
            self.queue.add_stream(ReductionStream(config))

        for key, group in self.groups.items():
            if isinstance(group, MasterframeGroup):
                pre_task = group.get_tasks(device_id=i % 2, only_with_sci=False, make_plots=True)
                self.queue.add_task(pre_task)

        self.queue.wait_until_task_complete("all")


class MasterframeGroup:
    def __init__(self, key):
        self.key = key
        self.image_files = []
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

    def add_images(self, filepath):
        if isinstance(filepath, list):
            self.image_files.extend(filepath)
        elif isinstance(filepath, str):
            self.image_files.append(filepath)
        elif isinstance(filepath, tuple):

            _tmp_list = list(chain.from_iterable(filepath))
            self.image_files.extend(_tmp_list)
        else:
            raise ValueError("Invalid filepath type")

    def add_sci_keys(self, keys):
        self.sci_keys.append(keys)

    def create_config(self, overwrite=False):
        c = PreprocConfiguration(self.image_files)
        self._config = c.config_file

    def get_tasks(self, device_id=None, only_with_sci=False, make_plots=True):
        from .run import get_preprocess_task

        pre_task = get_preprocess_task(
            self.config, device_id=device_id, only_with_sci=only_with_sci, make_plots=make_plots
        )
        return pre_task

    def run(self):
        from .run import run_preprocess

        return run_preprocess(self.config, make_plots=True)

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

    def get_stream(self):
        from .stream import ReductionStream

        return ReductionStream(self.config)

    def __repr__(self):
        return f"ScienceGroup({self.key} with {len(self.image_files)} images)"


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
