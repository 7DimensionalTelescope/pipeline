import os
import threading
from collections import UserDict

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
                print(f"Failed to extract mfg_key from {group}")

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

    def create_config(self):
        threads = []
        for group in self.groups.values():
            t = threading.Thread(target=group.create_config)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    def process_group(self, group, device_id=None):
        # Submit preprocess
        pre_task, plot_task = group.get_tasks(device_id=device_id)
        self.queue.add_task(pre_task)
        # Wait for this group's preprocess to finish
        self.queue.wait_until_task_complete(pre_task.id)
        self.queue.add_task(plot_task)

        for key in group.sci_keys:
            sci_group = self.groups[key]
            if not (sci_group.multi_units):
                sci_tree = sci_group.get_tree()
                self.queue.add_tree(sci_tree)
            else:
                self._multi_unit_config.add(sci_group.config)

    def process_all(self):
        threads = []
        for i, (key, group) in enumerate(self.groups.items()):
            if isinstance(group, MasterframeGroup):
                t = threading.Thread(target=self.process_group, args=(group, i % 2))
                t.start()
                threads.append(t)

        for t in threads:
            t.join()

        while len(self._multi_unit_config) > 0:
            from .run import get_scidata_reduction_tasktree

            config = self._multi_unit_config.pop()
            self.queue.add_tree(get_scidata_reduction_tasktree(config))

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

    def create_config(self):
        c = PreprocConfiguration(self.image_files)
        self._config = c.config_file

    def get_tasks(self, device_id=None):
        from .run import get_preprocess_task, get_make_plot_task

        pre_task = get_preprocess_task(self.config, device_id=device_id)
        plot_task = get_make_plot_task(self.config)
        return pre_task, plot_task

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

    def create_config(self):
        sci_yml = PathHandler(self.image_files).sciproc_output_yml
        if os.path.exists(sci_yml):
            # If the config file already exists, load it
            c = SciProcConfiguration.from_config(sci_yml, write=True)
        else:
            c = SciProcConfiguration(self.image_files)
        self._config = c.config_file

    def get_tree(self):
        from .run import get_scidata_reduction_tasktree

        return get_scidata_reduction_tasktree(self.config)

    def run(self):
        from .run import run_scidata_reduction

        return run_scidata_reduction(self.config)

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
