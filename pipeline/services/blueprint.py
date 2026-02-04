import gc
from typing import List
from itertools import chain
from concurrent.futures import ThreadPoolExecutor

from ..utils import flatten
from ..path import PathHandler
from ..config.utils import get_filter_from_config
from ..const.observation import BROAD_FILTERS

from .utils import SortedGroupDict, PreprocessGroup, ScienceGroup
from .fd import log_fd_info

import json


class Blueprint:
    """overwrite=True to rewrite configs"""

    def __init__(
        self,
        input_params: List[str] = None,
        list_of_images: List[str] = None,
        use_db: bool = False,
        master_frame_only: bool = False,
        is_too: bool = False,
        **kwargs,
    ):
        self.groups = SortedGroupDict()

        self.is_too = is_too

        self.master_frame_only = master_frame_only

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
                sci_dict = group[2]
                flattened_group_0 = flatten(group[0])
                if sci_dict:
                    sample_file = flatten(next(iter(sci_dict.values())))[0]
                else:
                    sample_file = flattened_group_0[0]
                mfg_key = PathHandler(sample_file).output_name

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

    def create_config(self, overwrite=False, max_workers=30, is_too=False, priority=None):

        is_too = is_too or self.is_too

        # Log FD info before starting
        log_fd_info(prefix="[create_config] Before: ")

        kwargs = {"overwrite": overwrite, "is_too": is_too}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(group.create_config, **kwargs) for group in self.groups.values()]
            for i, f in enumerate(futures):
                f.result()
                # Log FD info periodically during processing
                if (i + 1) % 10 == 0 or i == len(futures) - 1:
                    log_fd_info(prefix=f"[create_config] After {i+1}/{len(futures)} configs: ")
        del futures
        gc.collect()

        # Log FD info after completion
        log_fd_info(prefix="[create_config] After cleanup: ")

        self._config_generated = True

    def create_schedule(
        self,
        is_too=False,
        base_priority=None,
        overwrite=False,
        overwrite_preprocess=False,
        overwrite_science=False,
        preprocess_kwargs=None,
        processes=["astrometry", "photometry", "coadd", "subtract"],
        input_type=None,
        **kwargs,
    ):
        """
        # priority definition:
        Preprocess = base_priority + 1
        SciProcess = base_priority
        --------------------------------

        0: Failed process

        1: User-input / Reprocess science
        2: User-input / Reprocess preprocess

        3: Daily science
        4: Daily preprocess

        6: ToO science medium band
        7: ToO preprocess medium band

        11: ToO science broad band
        12: ToO preprocess broad band
        """

        is_too = is_too or self.is_too

        if not self._config_generated:
            self.create_config(overwrite=overwrite, max_workers=kwargs.get("max_workers", 50), is_too=is_too)

        from astropy.table import Table

        schedule = Table(
            dtype=[
                ("index", int),
                ("config", object),
                ("config_type", object),  # Preprocess or Science
                ("input_type", object),  # Daily or ToO
                ("is_ready", bool),  # True if the task is ready to be processed
                ("priority", int),  # Priority of the task
                ("readiness", int),  # 100 if the task is ready to be processed
                ("status", object),  # Ready, Pending, Processing, Completed
                ("dependent_idx", list),
                ("pid", int),  # Process ID
                ("kwargs", object),  # overwrite, ...
                ("process_start", object),  # ISO format timestamp when processing started
                ("process_end", object),  # ISO format timestamp when processing ended
            ]
        )

        # priority definition
        # 0: Failed process

        # 1: User-input / Reprocess science
        # 2: User-input / Reprocess preprocess

        # 3: Daily science
        # 4: Daily preprocess

        # 6: ToO science medium band
        # 7: ToO preprocess medium band

        # 11: ToO science broad band
        # 12: ToO preprocess broad band

        idx = 0

        if base_priority is None and input_type in ["Daily", "ToO"]:
            if is_too:
                base_priority = 6
            else:
                base_priority = 3
        else:
            base_priority = base_priority or 1
            input_type = input_type or "User-input"

        input_type = kwargs.get("input_type", input_type)

        for group in self.groups:
            # group is PreprocessGroup
            if isinstance(group, ScienceGroup):
                continue

            scheduler_kwargs = (
                ["-overwrite"]
                if overwrite or overwrite_preprocess
                else [] + ["--preprocess_kwargs", json.dumps(preprocess_kwargs)] if preprocess_kwargs else []
            )

            # preproc gets priority +1 from base_priority
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
                    scheduler_kwargs,
                    "",
                    "",
                ]
            )
            parent_idx = idx
            idx += 1

            if self.master_frame_only:
                continue

            # add ScienceGroups that depend on this PreprocessGroup
            for scikey in group.sci_keys:

                sci_group = self.groups[scikey]
                if sci_group.config in schedule["config"]:
                    existing_idx = schedule["index"][schedule["config"] == sci_group.config][0]
                    schedule["dependent_idx"][parent_idx].append(existing_idx)
                    continue

                filter_name = get_filter_from_config(sci_group.config)

                # keep base_priority for medium-band (ToO & Daily) and Daily broadband
                priority = base_priority

                # highest priority for ToO broadband
                if is_too and filter_name in BROAD_FILTERS:
                    priority = 11  # sciprocess
                    schedule["priority"][parent_idx] = 12  # preprocess

                scheduler_kwargs = ["-overwrite"] if overwrite or overwrite_science else [] + ["-processes"] + processes

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
                        scheduler_kwargs,
                        "",
                        "",
                    ]
                )
                schedule["dependent_idx"][parent_idx].append(idx)
                idx += 1

        schedule.sort(["is_ready", "priority", "readiness"], reverse=True)

        self.schedule = schedule

    def cleanup(self):
        for group in self.groups.values():
            group.cleanup()
            del group
        self.groups = SortedGroupDict()
        gc.collect()
