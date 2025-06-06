import yaml
import os
import re
import glob
import json
from datetime import datetime
from .. import __version__
from ..utils import (
    header_to_dict,
    to_datetime_string,
    find_raw_path,
    get_camera,
    clean_up_folder,
    clean_up_sciproduct,
    swap_ext,
    most_common_in_list,
    get_header,
)
from .utils import merge_dicts
from ..const import HEADER_KEY_MAP, STRICT_KEYS, ANCILLARY_KEYS
from ..path.path import PathHandler
from .base import BaseConfig
import time


class SciProcConfiguration(BaseConfig):
    def __init__(
        self,
        input: list[str] | str | dict = None,
        logger=None,
        write=True,  # False for PhotometrySingle
        overwrite=False,
        verbose=True,
        **kwargs,
    ):
        st = time.time()
        self.write = write

        self._handle_input(input, logger, verbose, **kwargs)

        if not self._initialized:
            self.logger.info("Initializing configuration")
            self.initialize()

        if overwrite:
            self.logger.info("Overwriting factory_dir")
            clean_up_folder(self.path.factory_dir)
            clean_up_sciproduct(self.path.output_dir)

        self.logger.info(f"Writing configuration to file")
        self.logger.debug(f"Configuration file: {self.config_file}")
        self.write_config()

        self.logger.info(f"SciProcConfiguration initialized")
        self.logger.debug(f"SciProcConfiguration initialization took {time.time() - st:.2f} seconds")

    @property
    def name(self):
        if hasattr(self, "path"):
            return os.path.basename(self.path.sciproc_output_log).replace(".log", "")
        elif hasattr(self.config, "name"):
            return self.config.name
        else:
            return None
        
    def _handle_input(self, input, logger, verbose, **kwargs):
        if isinstance(input, list):
            self.input_files = input
            self.path = PathHandler(input)
            config_source = self.path.sciproc_base_yml
            self.logger = self._setup_logger(
                logger,
                name=self.name,
                log_file=self.path.sciproc_output_log,
                verbose=verbose,
            )
            self.logger.info("Generating a configuration from the 'base' configuration")
            self.logger.debug(f"Configuration source: {config_source}")
            super().__init__(config_source=config_source, write=self.write, **kwargs)
            self.config.logging.file = self.path.sciproc_output_log
        elif isinstance(input, str | dict):
            config_source = input
            super().__init__(config_source=config_source, write=self.write, **kwargs)
            self.path = self._set_pathhandler_from_config()
            self.config.logging.file = self.path.sciproc_output_log
            self.logger = self._setup_logger(
                logger,
                name=self.name,
                log_file=self.config.logging.file,
                verbose=verbose,
            )
            self._initialized = True
            self.logger.info("Loading configuration from an exisiting file or dictionary")
            self.logger.debug(f"Configuration source: {config_source}")
            
        else:
            raise ValueError("Input must be a list of image files, a configuration file path, or a configuration dictionary.")  # fmt: skip
          # used by write_config
        self.config_file = self.path.sciproc_output_yml  # used by write_config
        return

    def _set_pathhandler_from_config(self):
        # mind the check order
        if hasattr(self.config, "input"):
            if hasattr(self.config.input, "calibrated_images") and self.config.input.calibrated_images:
                return PathHandler(self.config.input.calibrated_images)

            if hasattr(self.config.input, "processed_dir") and self.config.input.processed_dir:
                f = os.path.join(self.config.input.processed_dir, "**.fits")
                return PathHandler(sorted(glob.glob(f)))

            if hasattr(self.config.input, "stacked_image") and self.config.input.stacked_image:
                return PathHandler(self.config.input.stacked_image)

        raise ValueError("Configuration does not contain valid input files or directories to create PathHandler.")

    def initialize(self):
        """Fill in obs info, name, paths."""

        self.config.info.version = __version__
        self.config.info.creation_datetime = datetime.now().isoformat()
        self.config.name = self.name

        self.config.input.calibrated_images = PathHandler(self.input_files).processed_images
        self.config.input.output_dir = self.path.output_dir

        # self.set_input_output()
        # self.check_masterframe_status()

        self._add_metadata()
        self._define_settings(self.input_files[0])
        self.input_files = self.config.input.calibrated_images
        self._initialized = True

    def _add_metadata(self):  # for webpage

        # metadata_path = os.path.join(self._processed_dir, name, "metadata.ecsv")
        metadata_path = os.path.join(self.path.metadata_dir, "metadata.ecsv")
        if not os.path.exists(metadata_path):
            with open(metadata_path, "w") as f:
                f.write("# %ECSV 1.0\n")
                f.write("# ---\n")
                f.write("# datatype:\n")
                f.write("# - {name: obj, datatype: string}\n")
                f.write("# - {name: filter, datatype: string}\n")
                f.write("# - {name: unit, datatype: string}\n")
                f.write("# - {name: n_binning, datatype: int64}\n")
                f.write("# - {name: gain, datatype: int64}\n")
                f.write("# meta: !!omap\n")
                f.write("# - {created: " + datetime.now().isoformat() + "}\n")
                f.write("# schema: astropy-2.0\n")
                f.write("object unit filter n_binning gain\n")

        # observation_data = [
        #     str(self.config.obs.obj),
        #     str(self.config.obs.unit),
        #     str(self.config.obs.filter),
        #     str(self.config.obs.n_binning),
        #     str(self.config.obs.gain),
        # ]
        # new_line = f"{' '.join(observation_data)}\n"
        # with open(metadata_path, "a") as f:
        #     f.write(new_line)

    def _define_settings(self, input_file):
        # use local astrometric reference catalog for tile observations
        # self.config.settings.local_astref = bool(re.fullmatch(r"T\d{5}", self.config.obs.obj))

        # skip single frame combine for Deep mode
        raw_header_sample = get_header(input_file)
        obsmode = raw_header_sample["OBSMODE"]
        # self.config.obs.obsmode = obsmode
        self.config.settings.daily_stack = False if obsmode.lower() == "deep" else True
