from ..path import PathHandler
from ..query import query_observations
from ..config import PreprocConfiguration, SciProcConfiguration
from pathlib import Path
import threading

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
    
    def process_sequence(self):
        pass

class MasterframeGroup:
    
    def __init__(self, key):
        self.key = key  
        self.science_groups = set()
        self.image_files = []

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
        return PreprocConfiguration(self.image_files)

    def create_sciproc_config(self):
        threads = []
        raw_groups = PathHandler.take_raw_inventory(self.image_files)
        for group in raw_groups:
            for file_list in group[2].values():
                thread = threading.Thread(target=SciProcConfiguration, args=(file_list[1],))
                thread.start()
                threads.append(thread)
        for thread in threads:
            thread.join()
    
    def __repr__(self):
        string = f"<Masterframe Group {self.key} ({len(self.image_files)} images)"
        if self.image_files:
            string += "\n- Images:\n"
            for i, img in enumerate(self.image_files):
                string += f"  {i}: {img}\n"
        return string
