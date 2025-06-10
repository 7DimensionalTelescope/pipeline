from pathlib import Path
import threading

from ..path import PathHandler
from ..query import query_observations
from ..config import PreprocConfiguration, SciProcConfiguration

from ..services.queue import QueueManager

def glob_files_from_db(params):
    return []


def glob_files_by_param(keywords):
    if isinstance(keywords, dict):
        keywords = list(keywords.values())
    return query_observations(keywords)


class Blueprint:
    def __init__(self, input_params, use_db=False):
        self.groups = {}  # use a dictionary

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

    def __getattr__(self, name):
        if name == "groups":
            return sorted(self.groups.values())
        elif name in self.__dict__:
            return self.__dict__[name]
        else:
            raise AttributeError(f"Attribute {name} not found")

    def initialize(self):
        for image in self.list_of_images:
            o = Path(image)
            identifier = o.parent.parts[-2:]
            if identifier not in self.groups:
                self.groups[identifier] = MasterframeGroup(str(o.parent))
            self.groups[identifier].add_image(str(o.absolute()))
    
    def create_config(self):
        for group in self.groups.values():
            group.create_preproc_config()
            group.create_sciproc_config()
    
    def get_group_by_index(self, i):
        return self.groups[list(self.groups.keys())[i]]
        
    
    def process_group(self, group, device_id=None):
        from ..run import run_preprocess_with_task, run_process_with_tree, run_make_plots
        
        # Submit preprocess
        pre_task = run_preprocess_with_task(group.masterframe_config, device_id=device_id)
        self.queue.add_task(pre_task)
        # Wait for this group's preprocess to finish
        self.queue.wait_until_task_complete(pre_task.id)

        plot_task = run_make_plots(group.masterframe_config)
        self.queue.add_task(plot_task)

        for config in group.science_configs:
            sci_tree = run_process_with_tree(config)
            self.queue.add_tree(sci_tree)

    def process_all(self):

        threads = []
        for i, group in enumerate(self.groups.values()):
            t = threading.Thread(target=self.process_group, args=(group, i%2))
            t.start()
            threads.append(t)

        # Optionally, wait for all threads to finish
        for t in threads:
            t.join()

class MasterframeGroup:
    
    def __init__(self, key):
        self.key = key  
        self.science_groups = set()
        self.image_files = []
        self.masterframe_config = None
        self.science_configs = []

    def __lt__(self, other):
        return self.usage_count < other.usage_count
    
    def __eq__(self, other):
        if not isinstance(other, MasterframeGroup):
            return False
        return self.key == other.key

    def __hash__(self):
        return hash(self.key)
    
    @property
    def usage_count(self):
        return len(self.image_files)

    def add_image(self, filepath):
        self.image_files.append(filepath)

    def create_preproc_config(self):
        c = PreprocConfiguration(self.image_files)
        self.masterframe_config = c.config_file        

    def create_sciproc_config(self, use_threads=True):
        

        if use_threads:
            lock = threading.Lock()
            threads = []
            def gen_config(file_path):
                sciproc = SciProcConfiguration(file_path)
                with lock:
                    self.science_configs.append(sciproc.config_file)
            
        for group in raw_groups:
            for file_list in group[2].values():
                if use_threads:
                    thread = threading.Thread(target=gen_config, args=(file_list[0],))
                    thread.start()
                    threads.append(thread)
                else:
                    c = SciProcConfiguration(file_list[0])
                    self.science_configs.append(c.config_file)
        
        if use_threads:
            for thread in threads:
                thread.join()

    def __repr__(self):
        string = f"<Masterframe Group {self.key} ({len(self.image_files)} images)"
        if self.image_files:
            string += "\n- Images:\n"
            for i, img in enumerate(self.image_files):
                string += f"  {i}: {img}\n"
        return string
