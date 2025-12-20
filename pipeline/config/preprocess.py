import os
import sys
from datetime import datetime
import glob
from .. import __version__
from ..utils import clean_up_folder, flatten, time_diff_in_seconds
from ..path.path import PathHandler
from .base import BaseConfig
import time
from ..const import CalibType


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
            self.initialize()
            self.logger.info(f"'PreprocConfiguration' initialized in {time_diff_in_seconds(st)} seconds")
            self.logger.info(f"Writing configuration to file")
            self.logger.debug(f"Configuration file: {self.config_file}")

        # Cleaning Factory is only for SciProcConfiguration
        # if overwrite:
        #     self.logger.info("Deleting the factory directory to overwrite")
        #     factory_dir = self.path.factory_dir
        #     self.logger.debug(f"Factory directory: {factory_dir}")
        #     if not isinstance(factory_dir, str):
        #         raise ValueError(f"Multiple directories; aborting cleaning: {factory_dir}")
        #     clean_up_folder(factory_dir)
        #     # clean_up_folder(self.path.masterframe_dir)
        #     # clean_up_folder(self.path.preproc_output_dir)

        self.write_config()
        self.logger.info("Completed to load configuration")

    @property
    def name(self):
        if hasattr(self, "path"):
            # return os.path.basename(self.path.preproc_output_yml).replace(".yml", "")
            return self.path.output_name
        elif hasattr(self.config, "name"):
            return self.config.name
        else:
            return None

    @classmethod
    def base_config(cls):
        return

    def initialize(self):
        self.config.info.version = __version__
        self.config.info.creation_datetime = datetime.now().isoformat()
        self.config.name = os.path.basename(self.path.preproc_output_yml).replace(".yml", "")
        if self.input_files:
            masterframe_images = set()
            science_images = set()
            for file in self.input_files:
                if any(calib_type in file for calib_type in CalibType):
                    masterframe_images.add(file)
                else:
                    science_images.add(file)

        self.config.input.masterframe_images = list(masterframe_images)
        self.config.input.science_images = list(science_images)
        self.config.input.raw_dir = self.input_dir
        self._initialized = True

    def _handle_input(self, input, logger, verbose, is_too=False, **kwargs):

        if isinstance(input, list):  # List of FITS files
            if len(input) < 1:
                self.logger.warning("No input images")
                sys.exit(0)
            # sci_images = PathHandler(input).pick_type("science")
            # print(sci_images)
            # self.path = PathHandler(sorted(sci_images)[-1])  # in case of multiple dates, use the later date
            self.path = PathHandler(input, is_too=is_too)  # in case of multiple dates, use the later date
            config_source = self.path.preproc_base_yml
            if not isinstance(config_source, str):
                raise ValueError(f"PreprocConfiguration ill-defined: {config_source}")
            self.logger = self._setup_logger(
                logger,
                name=self.name,
                log_file=self.path.preproc_output_log,
                verbose=verbose,
            )
            self.logger.info("Generating 'PreprocConfiguration' from the 'base' configuration")
            self.logger.debug(f"Configuration source: {config_source}")
            self.input_files = input
            self.input_dir = None
            super().__init__(config_source=config_source, write=self.write, is_too=is_too, **kwargs)
        elif isinstance(input, str) and os.path.isdir(input):  # Directory containing FITS files
            sample_file = self._has_fits_file(input)
            self.path = PathHandler(sample_file)
            config_source = self.path.preproc_base_yml
            self.logger = self._setup_logger(
                logger,
                name=self.name,
                log_file=self.path.preproc_output_log,
                verbose=verbose,
            )
            self.logger.info("Loading 'PreprocConfiguration' from an exisiting file or dictionary")
            self.logger.debug(f"Configuration source: {config_source}")
            self.input_dir = input
            self.input_files = None
            super().__init__(config_source=config_source, write=self.write, is_too=is_too, **kwargs)
        elif (isinstance(input, str) and ".yml" in input) or isinstance(input, dict):  # Configuration file path
            config_source = input
            super().__init__(config_source=config_source, write=self.write, is_too=is_too, **kwargs)
            self.path = self._set_pathhandler_from_config()
            self.logger = self._setup_logger(
                logger,
                name=self.config.name,
                log_file=self.path.preproc_output_log,
                verbose=verbose,
                overwrite=False,
            )
            self._initialized = True
            self.logger.info("Loading configuration from an exisiting file or dictionary")
            self.logger.debug(f"Configuration source: {config_source}")
        else:
            raise ValueError("Input must be a list of FITS files or a directory containing FITS files")

        self.config.logging.file = self.path.preproc_output_log
        self.config_file = self.path.preproc_output_yml  # used by write_config
        return

    def _set_pathhandler_from_config(self):
        # mind the check order
        if hasattr(self.config, "input"):
            if hasattr(self.config.input, "science_images") and self.config.input.science_images:
                return PathHandler(self.config.input.science_images[0])

            elif hasattr(self.config.input, "masterframe_images") and self.config.input.masterframe_images:
                return PathHandler(flatten(self.config.input.masterframe_images)[0])

            elif hasattr(self.config.input, "raw_dir") and self.config.input.raw_dir:
                f = os.path.join(self.config.input.raw_dir, "**.fits")
                return PathHandler(sorted(glob.glob(f))[0])

        raise ValueError("Configuration does not contain valid input files or directories to create PathHandler.")

    def _has_fits_file(self, folder_path):
        with os.scandir(folder_path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith(".fits"):
                    return entry.path
        return False
