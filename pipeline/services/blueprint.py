import os
from concurrent.futures import ThreadPoolExecutor
import gc
import numpy as np

from ..utils import flatten
from ..path import PathHandler


from itertools import chain

from ..config.utils import get_filter_from_config

from .utils import SortedGroupDict, MasterframeGroup, ScienceGroup


class Blueprint:
    """overwrite=True to rewrite configs"""

    def __init__(
        self,
        input_params=None,
        list_of_images=None,
        use_db=False,
        ignore_mult_date=False,
        master_frame_only=False,
        **kwargs,
    ):
        self.groups = SortedGroupDict()

        if input_params is not None or list_of_images is not None:
            self.input_params = input_params

            if list_of_images is None:
                assert input_params is not None
                from ..run import query_observations

                print("Globbing images with parameters:", input_params)
                self.list_of_images = query_observations(
                    input_params, use_db=use_db, master_frame_only=master_frame_only, **kwargs
                )
            else:
                self.list_of_images = list_of_images

            print(f"Found {len(self.list_of_images)} images.")

            if len(self.list_of_images) == 0:
                print("No images found")
                return

            print("Grouping images...")
            self.initialize(ignore_mult_date=ignore_mult_date)

        print("Blueprint initialized.")

        self._config_generated = False

    @classmethod
    def from_config(cls, config_list):
        config_list = np.atleast_1d(config_list)
        cls = cls.__new__(cls)

        dependent_configs = dict()
        independent_configs = set()

        for config in config_list:
            if os.path.exists(config):
                keywords = os.path.basename(config).split("_")
                if len(keywords) == 2:
                    dependent_configs.setdefault(config, [])
                    dependent_configs[config].append(config)
                elif len(keywords) == 3:
                    independent_configs.add(config)
                else:
                    raise ValueError(f"Invalid config file: {config}")
            else:
                raise ValueError(f"Config file does not exist: {config}")
        cls.dependent_configs = dependent_configs
        cls.independent_configs = independent_configs

        return cls

    @classmethod
    def from_list(cls, list_of_images: list[str], ignore_mult_date=False, is_too=False, **kwargs):
        # if not all(f.endswith(".fits") for f in list_of_images):
        #     raise ValueError("Non-fits images in input")
        if not list_of_images:
            raise ValueError("Empty list_of_images")

        if not all(isinstance(f, str) and f.endswith(".fits") for f in list_of_images):
            raise ValueError("Non-fits images in input")

        return cls(list_of_images=list_of_images, ignore_mult_date=ignore_mult_date, is_too=is_too, **kwargs)

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
                    self.groups[key].multi_units += 1
                flattened_images = flatten(images[0])
                self.groups[key].add_images(flattened_images)
                self.groups[mfg_key].add_images(flattened_images)
                self.groups[mfg_key].add_sci_keys(key)

    def create_config(self, overwrite=False, max_workers=50, is_too=False, priority=None):
        kwargs = {"overwrite": overwrite, "is_too": is_too}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(group.create_config, **kwargs) for group in self.groups.values()]
            for f in futures:
                f.result()
        del futures
        gc.collect()

        self._config_generated = True

    def create_schedule(self, is_too=False, priority=None, **kwargs):

        if not self._config_generated:
            self.create_config(
                overwrite=kwargs.get("overwrite", False), max_workers=kwargs.get("max_workers", 50), is_too=is_too
            )

        from astropy.table import Table

        schedule = Table(
            dtype=[
                ("index", int),
                ("config", str),
                ("type", str),
                ("input_type", str),
                ("is_ready", bool),
                ("priority", int),
                ("readiness", int),
                ("status", str),
                ("dependent_idx", list),
                ("pid", int),
                ("original_status", str),
            ]
        )

        idx = 0

        if priority is None:
            if is_too:
                base_priority = 5
                input_type = "ToO"
            else:
                base_priority = 0
                input_type = "Daily"
        else:
            base_priority = priority
            input_type = "User-input"

        input_type = kwargs.get("input_type", input_type)

        for group in self.groups:
            if isinstance(group, ScienceGroup):
                continue
            schedule.add_row(
                [idx, group.config, "masterframe", input_type, True, base_priority + 2, 100, "Ready", [], 0, "Ready"]
            )
            parent_idx = idx
            idx += 1

            for scikey in group.sci_keys:
                sci_group = self.groups[scikey]
                if sci_group.config in schedule["config"]:
                    continue

                filter_name = get_filter_from_config(sci_group.config)

                if filter_name.startswith("m"):
                    priority = base_priority + 1
                elif is_too:
                    priority = 11
                    schedule["priority"][parent_idx] = 12
                else:
                    priority = base_priority + 3
                    schedule["priority"][parent_idx] = base_priority + 4

                schedule.add_row(
                    [
                        idx,
                        sci_group.config,
                        "science",
                        input_type,
                        False,
                        priority,
                        99 - sci_group.multi_units,
                        "Pending",
                        [],
                        0,
                        "Pending",
                    ]
                )
                schedule["dependent_idx"][parent_idx].append(idx)
                idx += 1

        schedule.sort(["is_ready", "priority", "readiness"], reverse=True)

        self.schedule = schedule

    def cleanup(self):
        import gc

        for group in self.groups.values():
            group.cleanup()
            del group
        self.groups = SortedGroupDict()
        gc.collect()
