import os
import sys
from datetime import datetime
import glob
import time

from .. import __version__
from ..utils import clean_up_folder, flatten, time_diff_in_seconds
from ..path.path import PathHandler
from ..const import CalibType
from .base import BaseConfig
from .utils import get_key


class PreprocConfiguration(BaseConfig):

    def __init__(
        self,
        input: list[str] | str | dict = None,
        logger=None,
        write=True,
        overwrite=False,
        verbose=True,
        is_too=False,
        **kwargs,
    ):
        st = time.time()
        self.write = write
        self._handle_input(input, logger, verbose, is_too=is_too, **kwargs)

        if not self._initialized:
            self.logger.info("Initializing configuration")
            self.initialize(is_too=is_too)
            self.logger.info(f"'PreprocConfiguration' initialized in {time_diff_in_seconds(st)} seconds")
            self.logger.info(f"Writing configuration to file")
            self.logger.debug(f"Configuration file: {self.config_file}")

        self.write_config()
        self.logger.info("Completed to load configuration")

    @property
    def name(self):
        if hasattr(self, "node") and hasattr(self.node, "name"):
            return self.node.name
        elif hasattr(self, "path"):
            return self.path.output_name
        else:
            return None

    @classmethod
    def user_config(cls, **kwargs):
        print("[WARNING] Not implemented yet. Returning base config...\n")  # TODO
        self = cls.base_config()
        self.node.settings.is_pipeline = False
        return self

    def initialize(self, is_too=False):
        if is_too:
            self.logger.info(f"Overriding preproc base configuration with {self.path.preproc_too_override_yml}")
            self.override_from_yaml(self.path.preproc_too_override_yml)

        self.node.info.creation_version = __version__
        self.node.info.creation_datetime = datetime.now().isoformat()
        self.node.info.file = self.config_file
        self.node.name = self.path.output_name

        if self.input_files:
            masterframe_images = set()
            science_images = set()
            for file in self.input_files:
                if any(calib_type in file for calib_type in CalibType):
                    masterframe_images.add(file)
                else:
                    science_images.add(file)

        self.node.input.masterframe_images = list(masterframe_images)
        self.node.input.science_images = list(science_images)
        self.node.input.raw_dir = self.input_dir
        self._initialized = True

    def _handle_input(self, input, logger, verbose, is_too=False, **kwargs):

        # List of FITS files
        if isinstance(input, list):
            if len(input) < 1:
                self.logger.error("No input images")
                sys.exit(0)
            # sci_images = PathHandler(input).pick_type("science")
            # print(sci_images)
            # self.path = PathHandler(sorted(sci_images)[-1])  # in case of multiple dates, use the later date
            self.path = PathHandler(input, is_too=is_too)  # in case of multiple dates, use the later date
            config_source = self.path.preproc_base_yml
            config_output = self.path.preproc_output_yml
            log_file = self.path.preproc_output_log

            if not isinstance(config_source, str):
                raise ValueError(f"PreprocConfiguration ill-defined: {config_source}")
            self.logger = self._setup_logger(logger, name=self.name, log_file=log_file, verbose=verbose)
            self.logger.info("Generating 'PreprocConfiguration' from the 'base' configuration")
            self.logger.debug(f"Configuration source: {config_source}")
            self.input_files = input
            self.input_dir = None
            super().__init__(config_source=config_source, write=self.write, is_too=is_too, **kwargs)
            self.node.logging.file = log_file

        # Configuration file path
        elif (isinstance(input, str) and input.endswith(".yml")) or isinstance(input, dict):
            config_source = input
            super().__init__(config_source=config_source, write=self.write, is_too=is_too, **kwargs)
            self._initialized = True
            self.path = self._set_pathhandler_from_config(is_too=is_too or get_key(self.node.settings, "is_too", False))
            config_output = self.path.preproc_output_yml
            log_file = self.path.preproc_output_log
            print(self.path._input_files)

            self.logger = self._setup_logger(
                logger, name=self.node.name, log_file=log_file, verbose=verbose, overwrite=False
            )
            self.logger.info("Loading configuration from an exisiting file or dictionary")
            self.logger.debug(f"Configuration source: {config_source}")

        # Directory containing FITS files
        # TODO: redirect it to cls
        elif isinstance(input, str) and os.path.isdir(input):
            sample_file = self._has_fits_file(input)

            self.path = PathHandler(sample_file)
            config_source = self.path.preproc_base_yml
            config_output = self.path.preproc_output_yml
            log_file = self.path.preproc_output_log

            self.logger = self._setup_logger(logger, name=self.name, log_file=log_file, verbose=verbose)
            self.logger.info("Loading 'PreprocConfiguration' from an exisiting file or dictionary")
            self.logger.debug(f"Configuration source: {config_source}")
            self.input_dir = input
            self.input_files = None
            super().__init__(config_source=config_source, write=self.write, is_too=is_too, **kwargs)
            self.node.logging.file = log_file

        else:
            raise ValueError("Input must be a list of FITS files or a directory containing FITS files")

        self.config_file = config_output  # used by write_config

        return

    def _set_pathhandler_from_config(self, is_too=False):
        # mind the check order
        if hasattr(self.node, "input"):
            if hasattr(self.node.input, "science_images") and self.node.input.science_images:
                return PathHandler(self.node.input.science_images[0], is_too=is_too)

            elif hasattr(self.node.input, "masterframe_images") and self.node.input.masterframe_images:
                return PathHandler(flatten(self.node.input.masterframe_images)[0], is_too=is_too)

            elif hasattr(self.node.input, "raw_dir") and self.node.input.raw_dir:
                f = os.path.join(self.node.input.raw_dir, "**.fits")
                return PathHandler(sorted(glob.glob(f))[0], is_too=is_too)

        raise ValueError("Configuration does not contain valid input files or directories to create PathHandler.")

    def _has_fits_file(self, folder_path):
        with os.scandir(folder_path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith(".fits"):
                    return entry.path
        return False
