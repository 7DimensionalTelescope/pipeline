from __future__ import annotations
import os
import glob
import time
from pathlib import Path
from datetime import datetime
from typing import Literal

from .. import __version__
from ..errors import ConfigurationError
from ..utils import clean_up_folder, clean_up_sciproduct, atleast_1d, time_diff_in_seconds, collapse
from ..utils.header import get_header
from ..path.path import PathHandler
from ..services.logger import Logger

from .base import BaseConfig
from .utils import get_key


class SciProcConfiguration(BaseConfig):
    def __init__(
        self,
        input: list[str] | str | dict = None,
        logger=None,
        write=True,  # False for PhotometrySingle
        verbose=True,
        overwrite=False,
        is_too=False,
        is_multi_epoch=False,
        **kwargs,
    ):
        st = time.time()
        self.write = write

        self._handle_input(input, logger, verbose, is_too=is_too, **kwargs)

        if not self._initialized:
            self.logger.info("Initializing configuration")
            self.initialize(is_too=is_too, is_multi_epoch=is_multi_epoch, **kwargs)
            self.logger.info(f"'SciProcConfiguration' initialized in {time_diff_in_seconds(st)} seconds")
            self.logger.info(f"Writing configuration to file: {os.path.basename(self.config_file)}")
            self.logger.debug(f"Full path to the configuration file: {self.config_file}")

        # fill in missing keys, even though initialized
        self.fill_missing_from_yaml()

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

    def _handle_input(self, input, logger, verbose, is_too=False, **kwargs):
        # list of science images
        if isinstance(input, list) or (isinstance(input, str) and input.endswith(".fits")):
            self.input_files = sorted(input)
            self.path = PathHandler(input, is_too=is_too)
            config_source = self.path.sciproc_base_yml
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
            super().__init__(config_source=config_source, write=self.write, **kwargs)
            self.node.logging.file = log_file

        # path of a config file
        elif isinstance(input, str | dict):
            config_source = input
            super().__init__(config_source=config_source, write=self.write, **kwargs)
            # working_dir = os.path.dirname(config_source) if isinstance(config_source, str) else None
            is_too = is_too or get_key(self.node.settings, "is_too", False)
            self.path = self._set_pathhandler_from_config(is_too=is_too)
            self.node.logging.file = self.path.sciproc_output_log

            if isinstance(config_source, str):
                self.config_file = config_source  # use the filename as is
            self.logger = self._setup_logger(
                logger,
                name=self.name,
                log_file=self.node.logging.file,
                verbose=verbose,
                overwrite=False,
            )
            self._initialized = True
            self.logger.info("Loading 'SciProcConfiguration' from an exisiting file or dictionary")
            self.logger.debug(f"Configuration source: {config_source}")

        else:
            raise ValueError("Input must be a list of image files, a configuration file path, or a configuration dictionary.")  # fmt: skip

        # used by write_config
        if not hasattr(self, "config_file"):
            self.config_file = self.path.sciproc_output_yml  # used by write_config

        return

    def _set_pathhandler_from_config(self, working_dir=None, is_too=False):
        # mind the check order
        if hasattr(self.node, "input"):
            if hasattr(self.node.input, "calibrated_images") and self.node.input.calibrated_images:
                return PathHandler(self.node.input.calibrated_images, working_dir=working_dir, is_too=is_too)

            if hasattr(self.node.input, "processed_dir") and self.node.input.processed_dir:
                f = os.path.join(self.node.input.processed_dir, "**.fits")
                return PathHandler(sorted(glob.glob(f)), working_dir=working_dir, is_too=is_too)

            if hasattr(self.node.input, "coadd_image") and self.node.input.coadd_image:
                return PathHandler(self.node.input.coadd_image, working_dir=working_dir, is_too=is_too)

        raise ValueError("Configuration does not contain valid input files or directories to create PathHandler.")

    def initialize(self, write=False, is_pipeline=True, is_too=False, is_multi_epoch=False):
        """Fill in universal info, filenames, settings."""

        if is_too:
            override_yml = self.path.sciproc_too_override_yml
            self.logger.info(f"Overriding base configuration with {override_yml}")
            self.override_from_yaml(override_yml)
        elif is_multi_epoch:
            override_yml = self.path.sciproc_multi_epoch_override_yml
            self.logger.info(f"Overriding base configuration with {override_yml}")
            self.override_from_yaml(override_yml)

        self.node.info.version = __version__
        self.node.info.creation_datetime = datetime.now().isoformat()
        self.node.info.file = self.config_file
        self.node.name = self.node.name or self.name

        self.node.input.calibrated_images = atleast_1d(PathHandler(self.input_files, is_too=is_too).processed_images)

        if is_too and is_pipeline:
            from .toodb import update_too_times

            update_too_times(self, self.input_files)

        self.node.input.output_dir = self.path.output_dir

        self.node.settings.is_pipeline = is_pipeline and self.path.is_pipeline
        self.node.settings.is_too = is_too
        self._define_settings(self.input_files[0])
        # self.input_files = self.node.input.calibrated_images

        self._initialized = True

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
        - config_name_policy: "error" to raise an error, other options to resolve the degeneracy
        """

        logger = False if not write else logger
        input_images = sorted([os.path.abspath(image) for image in atleast_1d(input_images)])
        path = PathHandler(input_images, working_dir=working_dir or os.getcwd(), is_too=is_too)
        self = cls.base_config(write=write)
        self.input_files = input_images
        self.path = path
        self.config_file = config_file or self.path.sciproc_output_yml
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

        if not self.input_files:
            return self

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

        self.initialize(write=write, is_pipeline=is_pipeline, is_too=is_too, is_multi_epoch=is_multi_epoch)
        if self.write:  # defined in base_config
            self.write_config(force=True)

        return self
