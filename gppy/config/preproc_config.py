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
    swap_ext,
    most_common_in_list,
    merge_dicts,
    get_header,
)
from ..const import HEADER_KEY_MAP, STRICT_KEYS, ANCILLARY_KEYS
from ..path.path import PathHandler
from .base import ConfigurationMixin, ConfigurationInstance


class PreprocConfiguration(ConfigurationMixin):
    def __init__(
        self,
        input_files: list[str] = None,
        config_source: str | dict = None,
        logger=None,
        write=True,  # False for PhotometrySingle
        overwrite=False,
        verbose=True,
        **kwargs,
    ):
        self.write = write
        self.path = PathHandler(input_files)
        self.config_file = self.path.preproc_output_yml
        self.log_file = self.path.preproc_output_log
        self._initialized = False

        if config_source:
            self._initialized = True
        else:
            config_source = self.path.preproc_base_yml
            # config_source = self._find_config_file()

        self._load_config(config_source, **kwargs)

        if not self._initialized:
            self.initialize(input_files)

        if overwrite:
            clean_up_folder(self.path.output_dir)
            clean_up_folder(self.path.factory_dir)

        self.logger = self._setup_logger(logger, verbose=verbose)

        self.write_config()

        self.config.flag.configuration = True
        self.logger.info(f"PreprocConfiguration initialized")
        self.logger.debug(f"Configuration file: {self.config_file}")

    @classmethod
    def from_dict(cls, config_dict, write=False, **kwargs):
        # config_dict['file']
        return cls(config_source=config_dict, write=write, **kwargs)

    def _load_config(self, config_source, **kwargs):
        # Load configuration from file or dict
        self._loaded = False

        if isinstance(config_source, str):
            input_dict = self.read_config(config_source)
        elif isinstance(config_source, dict):
            input_dict = config_source
        else:
            raise TypeError("Invalid config_source type")

        self._config_in_dict = input_dict

        self.config = ConfigurationInstance(self)

        self._update_with_kwargs(kwargs)
        self._make_instance(self._config_in_dict)

        self._loaded = True

    def _find_config_file(self):
        """Find the configuration file in the processed directory."""

        output_config_file = self.path.sciproc_output_yml
        if os.path.exists(output_config_file):
            self._initialized = True
            return output_config_file  # s[0]
        else:
            self._initialized = False
            return self.path.base_yml

    def initialize(self, input_files):
        """Fill in obs info, name, paths."""

        self.config.info.version = __version__
        self.config.info.creation_datetime = datetime.now().isoformat()

        # Set core observation details
        # self.config.obs.unit = obs_params["unit"]
        # self.config.obs.nightdate = obs_params["nightdate"]
        # self.config.obs.obj = obs_params["obj"]
        # self.config.obs.filter = obs_params["filter"]
        # self.config.obs.n_binning = obs_params["n_binning"]
        # self.config.obs.gain = obs_params["gain"]
        # self.config.obs.pixscale = self.config.obs.pixscale * float(obs_params["n_binning"])  # For initial solve
        self.config.name = self.path._output_name
        self.config.file.input_files = input_files

        self.raw_header_sample = get_header(input_files[0])
        # self.set_input_output()
        # self.check_masterframe_status()

        # self._generate_links()
        self._define_settings()
        self._initialized = True

    def _define_settings(self):
        pass
