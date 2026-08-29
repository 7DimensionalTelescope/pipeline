import gc
from typing import List, Literal
from itertools import chain
from concurrent.futures import ThreadPoolExecutor

from ..utils import collapse, flatten
from ..path import PathHandler
from ..path.name import NameHandler
from ..config.utils import get_filter_from_config
from ..const.run import DEFAULT_CROSSFILTER_PROCESSES, DEFAULT_SCIDATA_PROCESSES
from ..const.observation import BROAD_FILTERS

from .logger import get_high_level_task_logger
from .utils import CrossFilterGroup, SortedGroupDict, PreprocessGroup, ScienceGroup
from .fd import log_fd_info, FDTracker, PeakFDSampler

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
        is_pipeline: bool = False,
        enable_crossfilter: bool = True,
        crossfilter_suffix: str | None = None,
        **kwargs,
    ):
        self.groups = SortedGroupDict()

        self.is_too = is_too
        self.enable_crossfilter = enable_crossfilter
        self.crossfilter_suffix = crossfilter_suffix

        self.master_frame_only = master_frame_only

        if input_params is not None or list_of_images is not None:
            if input_params is not None and not isinstance(input_params, list):
                raise ValueError(f"input_params must be a list. Maybe you intended [{input_params}]?")
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
            self.initialize(is_pipeline=is_pipeline)

        print("Blueprint initialized.")

        self._config_generated = False

    @property
    def crossfilter_configs(self) -> list[str]:
        return [group.config for group in self.groups.values() if isinstance(group, CrossFilterGroup)]

    @classmethod
    def from_list(cls, list_of_images: list[str], is_too: bool = False, is_pipeline: bool = False, **kwargs):
        # if not all(f.endswith(".fits") for f in list_of_images):
        #     raise ValueError("Non-fits images in input")
        if not list_of_images:
            raise ValueError("Empty list_of_images")

        if not all(isinstance(f, str) and f.endswith(".fits") for f in list_of_images):
            raise ValueError("Non-fits images in input")

        return cls(list_of_images=list_of_images, is_too=is_too, is_pipeline=is_pipeline, **kwargs)

    def initialize(self, is_pipeline: bool = False):
        # [raw bdf, mframes, sci_dict]
        image_inventory = PathHandler.take_raw_inventory(
            self.list_of_images,
            is_too=self.is_too,
            is_pipeline=is_pipeline,
        )

        if len(image_inventory) == 0:
            get_high_level_task_logger(__name__).warning(
                f"No group for wrapper out of {self.list_of_images}\nPossibly due to NUM_MIN_CALIB"
            )

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

        if self.enable_crossfilter and not self.master_frame_only:
            science_groups = [group for group in self.groups.values() if isinstance(group, ScienceGroup)]
            for science_group in science_groups:
                name = NameHandler(science_group.image_files[0])
                nightdate = collapse(name.nightdate, raise_error=True)
                obj = collapse(name.obj, raise_error=True)
                key = f"crossfilter:{nightdate}:{obj}"
                if key not in self.groups:
                    self.groups[key] = CrossFilterGroup(key)
                self.groups[key].add_source_group(science_group)

    def create_config(
        self,
        overwrite=False,
        max_workers=30,
        is_too=False,
        is_pipeline=False,
        is_multi_epoch=False,
        overwrite_preprocess=False,
        priority=None,
        overwrite_crossfilter=None,
    ):
        """
        Create configs for all groups.
        - is_pipeline=True for normal pipeline runs (DataReduction / from_list).
        - is_pipeline=False (default) for ad-hoc / reprocess usage.
        """
        is_too = is_too or self.is_too

        log_fd_info(prefix="[create_config] Before: ")
        fd_tracker = FDTracker(label="create_config:start")

        kwargs = {
            "overwrite": overwrite,
            "is_too": is_too,
            "is_pipeline": is_pipeline,
            "is_multi_epoch": is_multi_epoch,
            "overwrite_preprocess": overwrite_preprocess,
        }
        # Background FD watcher
        with PeakFDSampler(interval=0.05) as fd_peak:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                upstream = [group for group in self.groups.values() if not isinstance(group, CrossFilterGroup)]
                futures = [executor.submit(group.create_config, **kwargs) for group in upstream]
                for i, f in enumerate(futures):
                    f.result()
                    if (i + 1) % 10 == 0 or i == len(futures) - 1:
                        fd_tracker.checkpoint(f"create_config:{i+1}/{len(futures)}")
                crossfilter = [group for group in self.groups.values() if isinstance(group, CrossFilterGroup)]
                crossfilter_kwargs = dict(
                    kwargs, overwrite=overwrite if overwrite_crossfilter is None else overwrite_crossfilter
                )
                futures = [
                    executor.submit(group.create_config, config_suffix=self.crossfilter_suffix, **crossfilter_kwargs)
                    for group in crossfilter
                ]
                for f in futures:
                    f.result()
            del futures
            gc.collect()

        fd_peak.report(prefix="[create_config] ")
        fd_tracker.checkpoint("create_config:after_cleanup")

        self._config_generated = True

    def create_schedule(
        self,
        is_too=False,
        base_priority=None,
        overwrite=False,
        overwrite_preprocess=False,
        overwrite_science=False,
        overwrite_crossfilter=False,
        preprocess_kwargs=None,
        processes=DEFAULT_SCIDATA_PROCESSES,
        crossfilter_processes=DEFAULT_CROSSFILTER_PROCESSES,
        input_type: Literal["Daily", "ToO", "Reprocess", "User-input"] = None,
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
                ("input_type", object),  # Daily / ToO / Reprocess / User-input
                ("is_ready", bool),  # True if the task is ready to be processed
                ("priority", int),  # Priority of the task
                ("readiness", int),  # 100 if the task is ready to be processed
                ("status", object),  # Ready, Pending, Processing, Completed
                ("dependent_idx", list),
                ("pid", int),  # Process ID
                ("dispatch", object),  # worker host running the task, empty for the local host
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
                input_type = "ToO"
            else:
                base_priority = 3
                input_type = "Daily"
        else:
            base_priority = base_priority or 1
            input_type = input_type or "User-input"

        input_type = kwargs.get("input_type", input_type)

        for group in self.groups:
            # group is PreprocessGroup
            if isinstance(group, ScienceGroup | CrossFilterGroup):
                continue

            scheduler_kwargs = ["-overwrite"] if (overwrite or overwrite_preprocess) else []
            if preprocess_kwargs:
                scheduler_kwargs = scheduler_kwargs + ["--preprocess_kwargs", json.dumps(preprocess_kwargs)]

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
                    "",
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

                scheduler_kwargs = ["-overwrite"] if overwrite or overwrite_science else []
                if processes != DEFAULT_SCIDATA_PROCESSES:
                    scheduler_kwargs = scheduler_kwargs + ["-processes"] + processes

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
                        "",
                        scheduler_kwargs,
                        "",
                        "",
                    ]
                )
                schedule["dependent_idx"][parent_idx].append(idx)
                idx += 1

        for group in self.groups.values():
            if not isinstance(group, CrossFilterGroup):
                continue
            parent_indices = [
                int(schedule["index"][schedule["config"] == source.config][0])
                for source in group.source_groups
                if source.config in schedule["config"]
            ]
            scheduler_kwargs = ["-overwrite"] if overwrite or overwrite_crossfilter else []
            if crossfilter_processes != DEFAULT_CROSSFILTER_PROCESSES:
                scheduler_kwargs += ["-processes"] + crossfilter_processes
            schedule.add_row(
                [
                    idx,
                    group.config,
                    "crossfilter",
                    input_type,
                    not parent_indices,
                    base_priority,
                    100 - len(parent_indices),
                    "Ready" if not parent_indices else "Pending",
                    [],
                    0,
                    "",
                    scheduler_kwargs,
                    "",
                    "",
                ]
            )
            for parent_idx in parent_indices:
                parent_row = next(i for i, value in enumerate(schedule["index"]) if value == parent_idx)
                schedule["dependent_idx"][parent_row].append(idx)
            idx += 1

        schedule.sort(["is_ready", "priority", "readiness"], reverse=True)

        self.schedule = schedule

    def cleanup(self):
        for group in self.groups.values():
            group.cleanup()
            del group
        self.groups = SortedGroupDict()
        gc.collect()
