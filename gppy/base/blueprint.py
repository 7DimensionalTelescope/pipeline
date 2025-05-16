import os
from collections import defaultdict, Counter
from socket import setdefaulttimeout
import matplotlib.pyplot as plt
import networkx as nx
from ..path import PathHandler
from ..query import query_observations


def glob_files_from_db(params):
    return []


def glob_files_by_param(keywords):
    if isinstance(keywords, dict):
        keywords = list(keywords.values())
    return query_observations(keywords)


class Blueprint:
    def __init__(self, input_params, use_db=False):
        self.input_params = input_params
        print("Globbing images...")

        if use_db:
            self.list_of_images = glob_files_from_db(input_params)
        else:
            self.list_of_images = glob_files_by_param(input_params)

        print(f"Found {len(self.list_of_images)} images.")
        print("Building dependency group...")
        self.initialize(self.list_of_images)
        print("Blueprint initialized.")

    def initialize(self, images):
        self.group = DependencyGroup()

        for file in images:
            try:
                p = PathHandler(file)
                types = p._names.types[0]
                self.group.register_image(file, types, p.obs_params)
            except Exception as e:
                print(f"Error with file {file}: {e}")
    
    def create_configs(self):
        pass

    def process_sequence(self):
        pass



class MasterframeGroup:
    
    def __init__(self, key):
        self.key = key  
        self.science_groups = set()
        self.usage_count = 0
        self.image_files = {}

    def add_image(self, filepath, types):
        self.image_files.setdefault(types, []).append(filepath)

    def add_usage(self, science_key):
        self.science_groups.add(science_key)
        self.usage_count += 1

    def __repr__(self):
        string = f"<Masterframe {self.key} (types={len(self.image_files.keys())} usage={self.usage_count})>"
        if self.image_files:
            string += "\n- Images:\n"
            for i, img in enumerate(self.image_files):
                string += f"  {i}: {img}\n"
        return string

class ScienceGroup:
    def __init__(self, target):
        self.target = target
        self.image_files = []

    def add_image(self, filepath):
        self.image_files.append(filepath)

    def assign_masterframe(self, masterframe):
        self.masterframe = masterframe

    def __repr__(self):
        string = f"<SCI {self.target} (N={len(self.image_files)})>\n"
        if self.masterframe:
            string += f"- Masterframe: {self.masterframe.key}\n"
        if self.image_files:
            string += f"- Images:\n"
            for i, img in enumerate(self.image_files):
                string += f"  {i}: {img}\n"
        return string

class DependencyGroup:
    def __init__(self):
        self.mfi_groups = defaultdict(MasterframeGroup)
        self.science_groups = defaultdict(set)

    def register_image(self, filepath, types, obs_params):
        # obs_params: (unit, nightdate, filter, obj, exposure)
        mfi_key = (obs_params['unit'], obs_params['nightdate'])

        if mfi_key not in self.mfi_groups:
            self.mfi_groups[mfi_key] = MasterframeGroup(mfi_key)
            self.science_groups[mfi_key] = dict()
        
        if types[1] == "science":
            if obs_params["obj"] not in self.science_groups[mfi_key]:
                self.science_groups[mfi_key][obs_params["obj"]] = ScienceGroup(obs_params["obj"])
                self.science_groups[mfi_key][obs_params["obj"]].assign_masterframe(self.mfi_groups[mfi_key])
            self.mfi_groups[mfi_key].add_usage(filepath)
            self.science_groups[mfi_key][obs_params["obj"]].add_image(filepath)
        else:
            self.mfi_groups[mfi_key].add_image(filepath, types[1])
        
    def sorted_mfi_list(self):
        return sorted(self.mfi_groups, key=lambda k: self.mfi_groups[k].usage_count)

    def sorted_sci_list(self, mfi_key):
        if isinstance(mfi_key, int):
            mfi_key = self.sorted_mfi_list()[mfi_key]
        return self.science_groups[mfi_key]

    def print_tree(self):
        sorted_mfis = self.sorted_mfi_list()
        for mfi_key in sorted_mfis:
            print(f"Masterframe: {mfi_key} (types={len(self.mfi_groups[mfi_key].image_files.keys())} usage={self.mfi_groups[mfi_key].usage_count})")
            sci_list = self.sorted_sci_list(mfi_key)
            for sci in sci_list.values():
                print(f"  └── SCI: {sci.target} (N={len(sci.image_files)})")
            print()
