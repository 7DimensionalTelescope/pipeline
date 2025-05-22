import os
from datetime import datetime
from .. import __version__
from ..utils import (
    clean_up_folder,
)
from ..path.path import PathHandler
from .base import BaseConfig
import time
from ..const import CalibType


class PreprocConfiguration(BaseConfig):

    def __init__(
        self,
        input_files: str | list[str] = None,
        config_source: str | dict = None,
        logger=None,
        write=True,
        overwrite=False,
        verbose=True,
        **kwargs,
    ):
        st = time.time()
        self.write = write

        if isinstance(input_files, list):
            if ".fits" in input_files[0] and os.path.exists(input_files[0]):
                sample_file = input_files[0]
                self.input_files = input_files
                self.input_dir = None
            else:
                raise ValueError("Input list must be a list of FITS files")
        elif isinstance(input_files, str) and os.path.isdir(input_files):
            sample_file = self._has_fits_file(input_files)
            self.input_dir = input_files
            self.input_files = None
        else:
            raise ValueError("Input must be a list of FITS files or a directory containing FITS files")

        if not sample_file:
            raise FileNotFoundError("Input directory does not contain any FITS files")

        self.path = PathHandler(sample_file)
        self.config_file = self.path.preproc_output_yml
        self.log_file = self.path.preproc_output_log
        self.name = os.path.basename(self.config_file).replace(".yml", "")

        self.logger = self._setup_logger(logger, name=self.name, log_file=self.log_file, verbose=verbose)

        if overwrite:
            self.logger.info("Overwriting masterframe, processed, and factory directories")
            clean_up_folder(self.path.masterframe_dir)
            clean_up_folder(self.path.preproc_output_dir)

        if config_source:
            self._initialized = True
        else:
            config_source = self.path.preproc_base_yml
            self._initialized = False

        self.logger.info("Loading configuration")
        self.logger.debug(f"Configuration source: {config_source}")
        super().__init__(config_source=config_source, **kwargs)

        if not self._initialized:
            self.logger.info("Initializing configuration")
            self.initialize()

        self.logger.info(f"Writing configuration to file")
        self.logger.debug(f"Configuration file: {self.config_file}")
        self.write_config()

        self.logger.info(f"PreprocConfiguration initialized")
        self.logger.debug(f"PreprocConfiguration initialization took {time.time() - st:.2f} seconds")

    def initialize(self):
        self.config.info.version = __version__
        self.config.info.creation_datetime = datetime.now().isoformat()
        self.config.name = self.name
        if self.input_files:
            self.config.input.masterframe_files = []
            self.config.input.science_files = []
            for file in self.input_files:
                if any(calib_type in file for calib_type in CalibType):
                    self.config.input.masterframe_files.append(file)
                else:
                    self.config.input.science_files.append(file)

        self.config.input.raw_dir = self.input_dir
        self._initialized = True

    def _has_fits_file(self, folder_path):
        with os.scandir(folder_path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith(".fits"):
                    return entry.path
        return False
