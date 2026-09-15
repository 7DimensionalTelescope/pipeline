from __future__ import annotations
import os
import glob
import time
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from .. import __version__
from ..errors import ConfigurationError
from ..utils import clean_up_folder, clean_up_sciproduct, atleast_1d, time_diff_in_seconds, collapse
from ..utils.header import get_header
from ..path.path import PathHandler
from ..services.logger import Logger
from ..const.sciproc import SCIPROCESS_REGISTRY

from .base import BaseConfig
from .utils import get_key, merge_dicts

if TYPE_CHECKING:
    from ._sciproc_stubs import SciProcNode


class SciProcConfiguration(BaseConfig):
    if TYPE_CHECKING:
        node: SciProcNode

    def __init__(
        self,
        input: list[str] | str | dict = None,
        logger: bool | Logger = None,
        write=True,  # False for PhotometrySingle
        verbose=True,
        overwrite=False,
        working_dir: str | None = None,
        is_pipeline=False,
        is_too=False,
        is_multi_epoch=False,
        overwrite_config_sections: list[str] = None,
        **kwargs,
    ):
        st = time.time()
        self.write = write

        self._handle_input(
            input,
            logger,
            verbose,
            working_dir=working_dir,
            is_pipeline=is_pipeline,
            is_too=is_too,
            is_multi_epoch=is_multi_epoch,
            overwrite=overwrite,
            **kwargs,
        )

        if not self._initialized:
            self.logger.info("Initializing configuration")
            self.initialize(is_too=is_too, is_multi_epoch=is_multi_epoch, is_pipeline=is_pipeline, **kwargs)
            self.logger.info(f"'SciProcConfiguration' initialized in {time_diff_in_seconds(st)} seconds")
            self.logger.info(f"Writing configuration to file: {os.path.basename(self.config_file)}")
            self.logger.debug(f"Full path to the configuration file: {self.config_file}")

        # fill in missing keys, even though initialized
        self.fill_missing_from_yaml()

        if overwrite_config_sections:
            self.overwrite_config_sections(overwrite_config_sections)

        if not os.path.exists(self.config_file) or overwrite:
            self.write_config()
        self.logger.info("Completed to load configuration")

    @property
    def name(self):
        if hasattr(self, "config_file") and self.config_file is not None:
            return os.path.splitext(os.path.basename(self.config_file))[0]
        elif hasattr(self, "path"):
            return os.path.basename(self.path.sciproc_output_yml).replace(".yml", "")
        elif hasattr(self.node, "name"):
            return self.node.name
        else:
            return None

    def _handle_input(
        self,
        input,
        logger,
        verbose,
        working_dir=None,
        is_pipeline=False,
        is_too=False,
        is_multi_epoch=False,
        overwrite=False,
        **kwargs,
    ):
        # list of science images
        if isinstance(input, list) or (isinstance(input, str) and input.endswith(".fits")):
            self.input_files = sorted(input)
            self.path = PathHandler(
                input, working_dir=working_dir, is_pipeline=is_pipeline, is_too=is_too, is_multi_epoch=is_multi_epoch
            )
            config_source = collapse(self.path.sciproc_base_yml, raise_error=True)
            log_file = self.path.sciproc_output_log

            self.logger = self._setup_logger(
                logger,
                name=self.name,
                log_file=log_file,
                verbose=verbose,
                overwrite=self.write,
            )
            self.logger.info("Generating 'SciProcConfiguration' from the 'base' configuration")
            self.logger.debug(f"Configuration source: {config_source}")
            self.logger.debug(f"PathHandler settings: {self.path.settings}")
            super().__init__(config_source=config_source, write=self.write, **kwargs)
            self.node.logging.file = log_file

        # path of a config file
        elif isinstance(input, str | dict):
            config_source = input
            super().__init__(config_source=config_source, write=self.write, **kwargs)
            # working_dir = os.path.dirname(config_source) if isinstance(config_source, str) else None
            self.path = self._set_pathhandler_from_config(
                working_dir=working_dir,
                is_pipeline=get_key(self.node.settings, "is_pipeline", False),
                is_too=get_key(self.node.settings, "is_too", False),
                is_multi_epoch=get_key(self.node.settings, "is_multi_epoch", False),
                config_file=config_source if isinstance(config_source, str) else None,
            )
            self.node.logging.file = self.path.sciproc_output_log

            if isinstance(config_source, str):
                self.config_file = config_source  # use the filename as is
            self.logger = self._setup_logger(
                logger,
                name=self.name,
                log_file=self.node.logging.file if self.write else None,
                verbose=verbose,
                overwrite=overwrite,
            )
            self._initialized = True
            self.logger.info("Loading 'SciProcConfiguration' from an exisiting file or dictionary")
            self.logger.debug(f"Configuration source: {config_source}")
            self.logger.debug(f"PathHandler settings: {self.path.settings}")

        else:
            raise ValueError("Input must be a list of image files, a configuration file path, or a configuration dictionary.")  # fmt: skip

        # used by write_config
        if not hasattr(self, "config_file"):
            self.config_file = self.path.sciproc_output_yml  # used by write_config

        return

    def _set_pathhandler_from_config(
        self, working_dir=None, is_pipeline=False, is_too=False, is_multi_epoch=False, config_file=None
    ):
        # mind the check order
        kwargs = {
            "working_dir": working_dir,
            "is_pipeline": is_pipeline,
            "is_too": is_too,
            "is_multi_epoch": is_multi_epoch,
            "config_file": config_file,
            "factory_scratch": get_key(self.node.settings, "factory_scratch"),
        }
        if hasattr(self.node, "input"):
            if hasattr(self.node.input, "calibrated_images") and self.node.input.calibrated_images:
                return PathHandler(self.node.input.calibrated_images, **kwargs)

            if hasattr(self.node.input, "processed_dir") and self.node.input.processed_dir:
                f = os.path.join(self.node.input.processed_dir, "**.fits")
                return PathHandler(sorted(glob.glob(f)), **kwargs)

            if hasattr(self.node.input, "coadd_image") and self.node.input.coadd_image:
                return PathHandler(self.node.input.coadd_image, **kwargs)

        raise ValueError("Configuration does not contain valid input files or directories to create PathHandler.")

    def initialize(self, write=False, is_pipeline=False, is_too=False, is_multi_epoch=False):
        """Fill in universal info, filenames, settings."""

        override_yml = self._override_yml(is_too, is_multi_epoch)
        if override_yml:
            self.logger.info(f"Overriding base configuration with {override_yml}")
            self.override_from_yaml(override_yml)

        self.node.info.creation_version = __version__
        self.node.info.creation_datetime = datetime.now().isoformat()
        self.node.info.file = self.config_file
        self.node.name = self.node.name or self.name

        self.node.input.calibrated_images = atleast_1d(self.path.processed_images)

        if is_too and is_pipeline:
            from .toodb import update_too_times

            update_too_times(self, self.input_files)

        self.node.input.output_dir = self.path.output_dir
        # self.node.imcoadd.coadd_image = self.path.imcoadd._coadd_image
        # self.node.input.coadd_image = self.node.imcoadd.coadd_image

        self.node.settings.is_pipeline = is_pipeline
        self.node.settings.is_too = is_too
        self.node.settings.is_multi_epoch = is_multi_epoch
        self._define_settings(self.input_files[0])
        # self.input_files = self.node.input.calibrated_images

        self._initialized = True

    def _override_yml(self, is_too=False, is_multi_epoch=False) -> str | None:
        if is_too:
            return self.path.sciproc_too_override_yml
        if is_multi_epoch:
            return self.path.sciproc_multi_epoch_override_yml
        return None

    def overwrite_config_sections(self, sections: list[str]) -> bool:
        """Rebuild the named science sections from the template, keeping input_images and runtime_version; clear flags from the first affected stage onward."""
        sections = list(atleast_1d(sections))
        known = [spec.config_section for spec in SCIPROCESS_REGISTRY.specs]
        if unknown := [s for s in sections if s not in known]:
            raise ConfigurationError.ValueError(f"Unknown config sections {unknown}; choose from {sorted(set(known))}")
        if not (self.write and get_key(self.node.settings, "is_pipeline", False)):
            self.logger.warning("overwrite_config_sections skipped: requires write=True and settings.is_pipeline=True")
            return False
        template = self.read_config(collapse(self.path.sciproc_base_yml, raise_error=True))
        override_yml = self._override_yml(
            get_key(self.node.settings, "is_too", False), get_key(self.node.settings, "is_multi_epoch", False)
        )
        if override_yml and os.path.exists(override_yml):
            merge_dicts(template, self.read_config(override_yml))  # same layering as initialize
        before = deepcopy(self._config_in_dict)
        for section in sections:
            old = self._config_in_dict.get(section) or {}
            fresh = deepcopy(template[section])
            if "input_images" in fresh:
                fresh["input_images"] = old.get("input_images")
            fresh["runtime_version"] = old.get("runtime_version") or get_key(self.node.info, "runtime_version")
            self._config_in_dict[section] = fresh
        specs = SCIPROCESS_REGISTRY.specs
        first = min(i for i, spec in enumerate(specs) if spec.config_section in sections)
        for spec in specs[first:]:
            self._config_in_dict["flag"][spec.name] = False
        if self._config_in_dict == before:
            return False
        self._rebuilding = True
        try:
            self._make_nodes()
        finally:
            self._rebuilding = False
        self.write_config()
        for section in sections:
            self.logger.info(
                f"Overwrote '{section}' from the template (recorded runtime_version={self._config_in_dict[section]['runtime_version']!r})"
            )
        return True

    def _define_settings(self, input_file_sample):
        try:
            # skip single frame combine for Deep mode
            raw_header_sample = get_header(input_file_sample)
            try:
                obsmode = raw_header_sample["OBSMODE"]
            except KeyError:
                self.logger.warning("OBSMODE keyword not found in the header. Defaulting to 'spec'.")
                obsmode = "spec"
            # self.config.obs.obsmode = obsmode
            self.node.settings.coadd = False if obsmode.lower() == "deep" else True
        except Exception as e:
            self.logger.warning(f"Failed to define settings: {e}")

    @classmethod
    def user_config(
        cls,
        input_images: list[str] | str = None,
        working_dir: str = None,
        config_file: str = None,
        write: bool = True,
        logger: bool | Logger = True,
        verbose: bool = True,
        is_pipeline: bool = False,
        is_too: bool = False,
        is_multi_epoch: bool = False,
        config_suffix: str = None,
        factory_scratch: str = None,
        config_name_policy: Literal["error", "last"] = "error",
        **kwargs,
    ):
        """
        SciProcConfiguration for user-input images.

        Args:
        - input_images: list of science images
        - working_dir: PathHandler's working_dir
        - config_file: path to save this configuration to
        - write: write configuration to file. False to skip writing.
        - logger: False to turn off logger, True to use default logger, or Logger instance to use custom logger
        - verbose: verbose level
        - is_pipeline: you want it False unless trying to modify existing pipeline product
        - is_too: flag for ToO observations, which have a dedicated save location
        - config_suffix: appended to the auto-generated config stem, e.g. "median"
        - config_name_policy: "error" to raise an error, other options to resolve the degeneracy
        """

        logger = False if not write else logger
        input_images = sorted([os.path.abspath(image) for image in atleast_1d(input_images)])
        # path = PathHandler(input_images, working_dir=working_dir or os.getcwd(), is_too=is_too)
        path = PathHandler(
            input_images,
            working_dir=working_dir,
            is_pipeline=is_pipeline,
            is_too=is_too,
            is_multi_epoch=is_multi_epoch,
            config_file=config_file,
            config_suffix=config_suffix,
            factory_scratch=factory_scratch,
        )
        self = cls.base_config(write=write)
        self.input_files = input_images
        self.path = path
        self.config_file = self.path.sciproc_output_yml
        if isinstance(self.config_file, list):
            if config_name_policy == "error":
                raise ConfigurationError.GroupingError(
                    "Inhomogeneous input images; config name is not uniquely defined. Use force_creation=True to use the last one."
                )
            elif config_name_policy == "last":
                print(f"[WARNING] config name is not uniquely defined. Using the last one.")
                self.config_file = collapse(sorted(self.path.sciproc_output_yml)[::-1], force=True)
            else:
                raise ConfigurationError.ValueError(f"Invalid config name policy: {config_name_policy}")

        if logger is True:
            log_file = self.path.sciproc_output_log
            if isinstance(log_file, list):
                print(f"[WARNING] log filename is not uniquely defined. Using the last one.")
                log_file = collapse(sorted(log_file)[::-1], force=True)
            self.node.logging.file = log_file
            self.logger = cls._setup_logger(
                name=self.name,
                log_file=self.node.logging.file,
                verbose=verbose,
                overwrite=write,
                **kwargs,
            )
        elif isinstance(logger, Logger):
            self.logger = logger
        else:
            self.logger = None

        if not self.input_files:
            return self

        self.initialize(write=write, is_pipeline=is_pipeline, is_too=is_too, is_multi_epoch=is_multi_epoch)
        self.node.settings.factory_scratch = factory_scratch
        if self.write:  # defined in base_config
            self.write_config(force=True)

        return self

    @classmethod
    def reset_config(cls, config_path: str, write: bool = True) -> "SciProcConfiguration":
        """
        Rebuild the config at ``config_path`` from ``sciproc_base.yml``, preserving
        only the essentials that define *which* config it is:

        - ``settings.is_too`` / ``settings.is_pipeline`` / ``settings.is_multi_epoch``
        - ``input.calibrated_images`` (the seed image list)

        Everything else (``flag``, ``info``, ``logging``, ``astrometry``,
        ``photometry``, ``imcoadd``, ``imsubtract``, ...) is rebuilt by
        :meth:`user_config`, which runs :meth:`initialize` on the seed images.

        Use this to recover from stale per-stage state (e.g. ``imcoadd.input_images: []``
        left over from a previous sanity-rejected run) before re-running the pipeline.
        """
        old = cls(config_path)
        calibrated_images = atleast_1d(get_key(old.node.input, "calibrated_images", default=[]) or [])
        if len(calibrated_images) == 0:
            raise ConfigurationError.ValueError(
                f"Cannot reset {config_path}: input.calibrated_images is empty. "
                "The config has no seed image list to rebuild from."
            )
        is_too = bool(get_key(old.node.settings, "is_too", default=False))
        is_pipeline = bool(get_key(old.node.settings, "is_pipeline", default=False))
        is_multi_epoch = bool(get_key(old.node.settings, "is_multi_epoch", default=False))

        return cls.user_config(
            input_images=list(calibrated_images),
            config_file=config_path,
            write=write,
            is_too=is_too,
            is_pipeline=is_pipeline,
            is_multi_epoch=is_multi_epoch,
        )
