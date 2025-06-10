from pathlib import Path
import threading
from collections import UserDict

from ..path import PathHandler
from ..query import query_observations
from ..config import PreprocConfiguration, SciProcConfiguration

from ..services.queue import QueueManager

from itertools import chain

from ..run import run_preprocess_with_task, run_process_with_tree, run_make_plots
        

def glob_files_from_db(params):
    return []


def glob_files_by_param(keywords):
    if isinstance(keywords, dict):
        keywords = list(keywords.values())
    return query_observations(keywords)



class Blueprint:
    def __init__(self, input_params, use_db=False):
        self.groups = SortedGroupDict()  # use a sorted dictionary
        self._unified_key_list = None  # Will be populated after initialization
        self._key_usage_map = None  # Will track which keys are used in which groups

        self.input_params = input_params
        print("Globbing images with parameters:", input_params)

        if use_db:
            self.list_of_images = glob_files_from_db(input_params)
        else:
            self.list_of_images = glob_files_by_param(input_params)

        print(f"Found {len(self.list_of_images)} images.")
        print("Grouping images...")
        self.initialize()
        print("Blueprint initialized.")

        self.queue = QueueManager()
    
    def initialize(self):
        image_inventory = PathHandler.take_raw_inventory(self.list_of_images)
        
        for i, group in enumerate(image_inventory):
            mfg_key = f"mfg_{i}"
            mfg = MasterframeGroup(mfg_key)
            mfg.add_images(group[0])
            for key, images in group[2].items():
                if key not in self.groups:
                    self.groups[key] = ScienceGroup(key)
                else:
                    self.groups[key].multi_units = True
                self.groups[key].add_images(images[0])
                mfg.add_images(images[0])
                mfg.add_sci_keys(key)
            self.groups[mfg_key] = mfg
        
    def create_config(self):
        threads = []
        for group in self.groups.values():
            t = threading.Thread(target=group.create_config)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    def process_group(self, group, device_id=None):
        self._multi_unit_config = []
        # Submit preprocess
        pre_task = run_preprocess_with_task(group.config, device_id=device_id)
        self.queue.add_task(pre_task)
        # Wait for this group's preprocess to finish
        self.queue.wait_until_task_complete(pre_task.id)

        plot_task = run_make_plots(group.config)
        self.queue.add_task(plot_task)

        for key in group.sci_keys:
            sci_group = self.groups[key]
            if not(sci_group.multi_units):
                sci_tree = run_process_with_tree(sci_group.config)
                self.queue.add_tree(sci_tree)
            else:
                self._multi_unit_config.append(sci_group.config)

    def process_all(self):
        threads = []
        for i, (key, group) in enumerate(self.groups.items()):
            if key.startswith("mfg"):    
                t = threading.Thread(target=self.process_group, args=(group, i%2))
                t.start()
                threads.append(t)

        # Optionally, wait for all threads to finish
        for t in threads:
            t.join()
        
        if len(self._multi_unit_config) > 0:
            for config in self._multi_unit_config:
                multi_unit_tree = run_process_with_tree(config)
                self.queue.add_tree(multi_unit_tree)

class MasterframeGroup:
    def __init__(self, key):  
        self.key = key
        self.image_files = []
        self.config = None
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
        self.config = c.config_file

    def __repr__(self):
        return f"MasterframeGroup({self.key} used in {self.sci_keys} with {len(self.image_files)} images)"

class ScienceGroup:
    def __init__(self, key):
        self.key = key
        self.image_files = []
        self.config = None
        self.multi_units = False
        
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
        c = SciProcConfiguration(self.image_files)
        self.config = c.config_file

    def __repr__(self):
        return f"ScienceGroup({self.key} with {len(self.image_files)} images)"
        
class SortedGroupDict(UserDict):
    """A dictionary that sorts its values when iterating."""
    def __iter__(self):
        # First sort by type (MasterframeGroup first, then ScienceGroup)
        # Then within each type, sort by their respective criteria
        return iter(self._get_sorted_values())
    
    def values(self):
        return self._get_sorted_values()
    
    def items(self):
        sorted_values = self._get_sorted_values()
        return [(getattr(v, 'key', None), v) for v in sorted_values]
    
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
        for value in self.values():
            print(value)