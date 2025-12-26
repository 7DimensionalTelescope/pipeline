import os
from concurrent.futures import ThreadPoolExecutor
import gc
import numpy as np

from ..utils import flatten
from ..path import PathHandler


from itertools import chain

from ..config.utils import get_filter_from_config

from .utils import SortedGroupDict, PreprocessGroup, ScienceGroup


class Blueprint:
    """overwrite=True to rewrite configs"""

    def __init__(
        self,
        input_params=None,
        list_of_images=None,
        use_db=False,
        master_frame_only=False,
        is_too=False,
        **kwargs,
    ):
        self.groups: SortedGroupDict = SortedGroupDict()

        self.is_too = is_too

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
            self.initialize()

        print("Blueprint initialized.")

        self._config_generated = False

    @classmethod
    def from_list(cls, list_of_images: list[str], is_too=False, **kwargs):
        # if not all(f.endswith(".fits") for f in list_of_images):
        #     raise ValueError("Non-fits images in input")
        if not list_of_images:
            raise ValueError("Empty list_of_images")

        if not all(isinstance(f, str) and f.endswith(".fits") for f in list_of_images):
            raise ValueError("Non-fits images in input")

        return cls(list_of_images=list_of_images, is_too=is_too, **kwargs)

    def initialize(self):
        # [raw bdf, mframes, sci_dict]
        image_inventory = PathHandler.take_raw_inventory(self.list_of_images)

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
                mfg = PreprocessGroup(mfg_key)
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

        is_too = is_too or self.is_too

        kwargs = {"overwrite": overwrite, "is_too": is_too}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(group.create_config, **kwargs) for group in self.groups.values()]
            for f in futures:
                f.result()
        del futures
        gc.collect()

        self._config_generated = True

    def create_schedule(self, is_too=False, base_priority=None, **kwargs):

        is_too = is_too or self.is_too

        if not self._config_generated:
            self.create_config(
                overwrite=kwargs.get("overwrite", False), max_workers=kwargs.get("max_workers", 50), is_too=is_too
            )

        from astropy.table import Table

        schedule = Table(
            dtype=[
                ("index", int),
                ("config", object),
                ("type", object),  # Preprocess or Science
                ("input_type", object),  # Daily or ToO
                ("is_ready", bool),  # True if the task is ready to be processed
                ("priority", int),  # Priority of the task
                ("readiness", int),  # 100 if the task is ready to be processed
                ("status", object),  # Ready, Pending, Processing, Completed
                ("dependent_idx", list),
                ("pid", int),  # Process ID
                ("original_status", object),  # Ready, Pending, Processing, Completed
                ("process_start", object),  # ISO format timestamp when processing started
                ("process_end", object),  # ISO format timestamp when processing ended
            ]
        )

        # priority definition
        # 0: Failed process

        # 1: Daily science medium band
        # 2: Daily preprocess medium band
        # 3: Daily science broad band
        # 4: Daily preprocess broad band

        # 6: ToO science medium band
        # 7: ToO preprocess medium band

        # 11: ToO science broad band
        # 12: ToO preprocess broad band

        idx = 0

        if base_priority is None:
            if is_too:
                base_priority = 6
                input_type = "ToO"
            else:
                base_priority = 3
                input_type = "Daily"
        else:
            base_priority = base_priority
            input_type = "User-input"

        input_type = kwargs.get("input_type", input_type)

        for group in self.groups:
            if isinstance(group, ScienceGroup):
                continue
            schedule.add_row(
                [
                    idx,
                    group.config,
                    "preprocess",
                    input_type,
                    True,
                    base_priority + 1,
                    100,
                    "Ready",
                    [],
                    0,
                    "Ready",
                    "",
                    "",
                ]
            )
            parent_idx = idx
            idx += 1

            for scikey in group.sci_keys:

                sci_group = self.groups[scikey]
                if sci_group.config in schedule["config"]:
                    existing_idx = schedule["index"][schedule["config"] == sci_group.config][0]
                    schedule["dependent_idx"][parent_idx].append(existing_idx)
                    continue

                filter_name = get_filter_from_config(sci_group.config)

                if filter_name.startswith("m"):
                    priority = base_priority
                elif is_too:
                    priority = 11
                    schedule["priority"][parent_idx] = 12

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
                        "",
                        "",
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
