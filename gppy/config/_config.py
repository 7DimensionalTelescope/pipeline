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
from .base import ConfigurationInstance


class Configuration:
    """
    DEPRECATED: Use SciProcConfiguration instead.
    Comprehensive configuration management system for 7DT observation data.

    Handles dynamic configuration loading, modification, and persistence across
    different stages of data processing. Provides flexible initialization,
    metadata extraction, and configuration file generation.

    Key Features:
    - Dynamic configuration instance creation
    - Nested configuration support
    - Automatic path generation
    - Metadata extraction from observation headers
    - Configuration file versioning
    """

    def __init__(
        self,
        obs_params: dict = None,
        # input_files: list = None,
        config_source: str | dict = None,
        logger=None,
        write=True,  # False for PhotometrySingle
        overwrite=False,
        return_base=False,  # for base_config
        verbose=True,
        **kwargs,
    ):
        """
        Initialize configuration with comprehensive observation metadata.

        Args:
            obs_params (dict, optional): Dictionary of observation parameters
            config_source (str|dict, optional): Custom configuration source
            **kwargs: Additional configuration parameter overrides
        """
        self.write = write

        if return_base:
            self.path = PathHandler()
            config_source = config_source or self.path.base_yml
            self._initialized = False

        elif config_source:
            self._initialized = True
        else:
            self.path = PathHandler(obs_params)

            if overwrite:
                # Default config source if overwrite
                config_source = self.path.base_yml
                self._initialized = False
            else:
                config_source = self._find_config_file()

        self._load_config(config_source, **kwargs)

        if return_base:
            return

        if not self._initialized:
            self.initialize(obs_params)

        if overwrite:
            # clean_up_folder(self.path.output_dir)
            clean_up_folder(self.path.factory_dir)
            # clean_up_folder(self.path.daily_stacked_dir)

        self.logger = self._setup_logger(logger, verbose=verbose)

        self.write_config()

        self.config.flag.configuration = True
        self.logger.info(f"Configuration initialized")
        # self.logger.debug(f"Configuration file: {self.config_file}")

    def __repr__(self):
        return self.config.__repr__()

    @classmethod
    def base_config(cls, working_dir=None, config_file=None, **kwargs):
        """Return the base (base.yml) ConfigurationInstance."""
        working_dir = working_dir or os.getcwd()

        instance = cls(return_base=True, **kwargs)  # working_dir=working_dir,
        config = instance.config
        config.name = "user-input"
        # instance.path = PathHandler(instance, working_dir=working_dir)  # make _data_type 'user-input'

        if config_file:
            # working_dir is ignored if config_file is absolute path
            config_file = os.path.join(working_dir, config_file) if working_dir else config_file
            if os.path.exists(config_file):
                with open(config_file, "r") as f:
                    new_config = yaml.load(f, Loader=yaml.FullLoader)

                instance._config_in_dict = Configuration._merge_dicts(instance._config_in_dict, new_config)
                instance._make_instance(instance._config_in_dict)
            else:
                raise FileNotFoundError("Provided Configuration file does not exist")

        instance._initialized = True
        # return config
        return instance

    @classmethod
    def from_obs(cls, obs, **kwargs):
        return cls(obs_params=obs.obs_params, **kwargs)

    @classmethod
    def from_dict(cls, config_dict, write=False, **kwargs):
        # config_dict['file']
        return cls(config_source=config_dict, write=write, **kwargs)

    @staticmethod
    def _merge_dicts(base: dict, updates: dict) -> dict:
        """
        Recursively merge the updates dictionary into the base dictionary.

        For each key in the updates dictionary:
        - If the key exists in base and both values are dictionaries,
          then merge these dictionaries recursively.
        - Otherwise, set or override the value in base with the one from updates.

        Args:
            base (dict): The original base configuration dictionary.
            updates (dict): The new configuration dictionary with updated values.

        Returns:
            dict: The merged dictionary containing updates from the new configuration.
        """
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = Configuration._merge_dicts(base[key], value)
            else:
                base[key] = value
        return base

    def _setup_logger(self, logger=None, overwrite=True, verbose=True):
        if logger is None:
            from ..services.logger import Logger

            logger = Logger(name=self.config.name, slack_channel="pipeline_report")

        if self.path.file_dep_initialized and self.write:
            log_file = self.path.output_log
            self.config.logging.file = log_file
            logger.set_output_file(log_file, overwrite=overwrite)
            logger.set_format(self.config.logging.format)
            logger.set_pipeline_name(self.path.output_name)

        if not (verbose):
            logger.set_level("WARNING")

        return logger

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

        # once config has file.raw_files, reinitialize PathHandler
        if self.is_initialized:
            # if hasattr(self.config, "file") and self.config.file is not None:
            # self.path.add_fits(self.config.file.raw_files)
            self.path = PathHandler(self)
            # self.path = PathHandler(self.config.file.raw_files)

        self._loaded = True

    def _find_config_file(self):
        """Find the configuration file in the processed directory."""

        # base_dir = self.path.output_dir
        # config_files = glob.glob(f"{base_dir}/*.yml")
        # if len(config_files) == 0:
        output_config_file = self.path.output_yml
        if os.path.exists(output_config_file):
            self._initialized = True
            return output_config_file  # s[0]
        else:
            self._initialized = False
            return self.path.base_yml

    def initialize(self, obs_params):
        """Fill in obs info, name, paths."""

        self._legacy_name_support(obs_params)

        # Set core observation details
        self.config.obs.unit = obs_params["unit"]
        self.config.obs.nightdate = obs_params["nightdate"]
        self.config.obs.obj = obs_params["obj"]
        self.config.obs.filter = obs_params["filter"]
        self.config.obs.n_binning = obs_params["n_binning"]
        self.config.obs.gain = obs_params["gain"]
        self.config.obs.pixscale = self.config.obs.pixscale * float(obs_params["n_binning"])  # For initial solve
        # self.config.name = f"{obs_params['nightdate']}_{obs_params['n_binning']}x{obs_params['n_binning']}_gain{obs_params['gain']}_{obs_params['obj']}_{obs_params['unit']}_{obs_params['filter']}"
        # self.config.name = f"{obs_params['nightdate']}_{obs_params['obj']}_{obs_params['filter']}_{obs_params['unit']}_{obs_params['n_binning']}x{obs_params['n_binning']}_gain{obs_params['gain']}"
        self.config.name = self.path.output_name
        self.config.info.creation_datetime = datetime.now().isoformat()

        self._glob_raw_images()
        self.check_image_coherency()  # outside self.initialize to use self.logger
        self.set_input_output()
        # self.check_masterframe_status()
        self._generate_links()

        self._add_metadata()
        # self._generate_links()
        self._define_settings()
        self._initialized = True

    @staticmethod
    def _legacy_name_support(obs_params):
        if "nightdate" not in obs_params and "date" in obs_params:
            obs_params["nightdate"] = obs_params["date"]
        if "object" not in obs_params and "obj" in obs_params:
            obs_params["object"] = obs_params["obj"]

    @property
    def is_initialized(self):
        return self._initialized

    @property
    def is_loaded(self):
        return self._loaded

    @property
    def config_in_dict(self):
        """Return the configuration dictionary."""
        return self._config_in_dict

    def _make_instance(self, input_dict):
        """
        Transform configuration dictionary into nested, dynamic instances.

        Args:
            input_dict (dict): Hierarchical configuration dictionary
        """

        for key, value in input_dict.items():
            if isinstance(value, dict):
                nested_dict = {}
                instances = ConfigurationInstance(self, key)
                for subkey, subvalue in value.items():
                    nested_dict[subkey] = subvalue
                    setattr(instances, subkey, subvalue)
                setattr(self.config, key, instances)
                self._config_in_dict[key] = nested_dict
            else:
                setattr(self.config, key, value)
                self._config_in_dict[key] = value

    def _update_config_in_dict(self, section, key, value):
        """Update configuration dictionary with new key-value pair."""
        target = self._config_in_dict[section] if section else self._config_in_dict
        target[key] = value

    def _update_with_kwargs(self, kwargs):
        """Merge additional configuration parameters."""
        # for key, value in kwargs.items():
        #     key = key.lower()
        #     if key in self._config_in_dict:
        #         self._config_in_dict[key] = value
        #     else:
        #         for section_dict in self._config_in_dict.values():
        #             if isinstance(section_dict, dict) and key in section_dict:
        #                 section_dict[key] = value
        #                 break

        lower_kwargs = {key.lower(): value for key, value in kwargs.items()}
        self._config_in_dict = Configuration._merge_dicts(self._config_in_dict, lower_kwargs)  # fmt:skip

    def _glob_raw_images(self):
        """Create and set output directory paths for processed data."""

        # obsdata is ill-defined and finding it involves searching
        path_raw = find_raw_path(
            self.config.obs.unit,
            self.config.obs.nightdate,
            self.config.obs.n_binning,
            self.config.obs.gain,
        )
        fname = self._obsdata_basename(self.config.obs)
        template = f"{path_raw}/{fname}"
        fits_files = sorted(glob.glob(template))

        raw_files = []
        raw_headers = []
        for fits_file in fits_files:
            header_in_dict = self._load_dict_header(fits_file)
            if header_in_dict["GAIN"] == self.config.obs.gain:
                raw_files.append(fits_file)
                raw_headers.append(header_in_dict)

        self._raw_files = raw_files
        self.raw_headers = raw_headers
        self.raw_header_sample = raw_headers[0]

    def check_image_coherency(self, **kwarg):
        """
        If incoherent, let the pipeline run for a subgroup of images.
        Then initialize pipeline from another manually copied config
        to do run_scidata_reduction.
        """
        _is_coherent = True
        obs_config_keys = STRICT_KEYS | ANCILLARY_KEYS
        # obs_config_keys.update(ANCILLARY_KEYS)

        # check all headers and write info to config.obs
        for config_key in obs_config_keys:
            if config_key == "nightdate":
                # info = [self.obs_params for header_in_dict in self.raw_headers]
                continue

            elif config_key == "camera":
                # Identify Camera from image size
                info = [get_camera(header_in_dict) for header_in_dict in self.raw_headers]

            else:
                header_key = HEADER_KEY_MAP[config_key]
                info = [header_in_dict[header_key] for header_in_dict in self.raw_headers]

            # collapse the list if coherent
            if len(set(info)) == 1:
                info = info[0]
            elif len(set(info)) == 0:
                raise ValueError(f"Input image information empty: {config_key}")

            # use the most common value if a strict key is incoherent
            elif config_key in STRICT_KEYS:

                # self.logger.warning(f"Incoherent Key {config_key}: {info}")  # logger undefined yet

                _is_coherent = False

                num, dominant_info = most_common_in_list(info)
                filtered_files = [f for f, val in zip(self._raw_files, info) if val == dominant_info]
                self._raw_files = filtered_files

                info = dominant_info
            # save ancillary key as list
            else:
                pass

            setattr(self.config.obs, config_key, info)

        self.config.obs.coherent_input = _is_coherent

    def set_input_output(self):
        self.path.add_fits(self._raw_files)  # file_dependent_common_paths work afterwards
        self.config.file.raw_files = self.path.raw_images
        self.config.file.processed_files = self.path.processed_images

    @staticmethod
    def _obsdata_basename(config):
        # ex) '7DT11_20250102_014829_T00139_m425_1x1_100.0s_0001.fits'
        # template = f"{path_raw}/*{self.config.obs.obj}_{self.config.obs.filter}_{self.config.obs.n_binning}*.fits"
        # return f"{config.unit}_{config.nightdate}_*_{config.obj}_{config.filter}_{config.n_binning}x{config.n_binning}_{config.exposure}.fits"
        return f"{config.unit}_*_*_{config.obj}_{config.filter}_{config.n_binning}x{config.n_binning}_*.fits"

    @staticmethod
    def _load_dict_header(fits_file):
        """use .head file if available"""
        header_file = swap_ext(fits_file, ".head")

        if os.path.exists(header_file):
            header_in_dict = header_to_dict(header_file)
        else:
            from astropy.io import fits

            header_in_dict = dict(fits.getheader(fits_file))
        return header_in_dict

    def _add_metadata(self):  # for webpage only

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

        observation_data = [
            str(self.config.obs.obj),
            str(self.config.obs.unit),
            str(self.config.obs.filter),
            str(self.config.obs.n_binning),
            str(self.config.obs.gain),
        ]
        new_line = f"{' '.join(observation_data)}\n"
        with open(metadata_path, "a") as f:
            f.write(new_line)

    def _generate_links(self):
        # Define pointer fpaths to master frames
        # legacy gppy used tool.calculate_average_date_obs('DATE-OBS')
        path_fdz = self.path.masterframe_dir  # master_frame/date_bin_gain/unit
        date_utc = to_datetime_string(self.config.obs.obstime[0], date_only=True)
        self.config.preprocess.mbias_link = os.path.join(
            path_fdz, f"bias_{date_utc}_{self.config.obs.camera}.link"
        )  # 7DT01/bias_20250102_C3.link
        self.config.preprocess.mdark_link = os.path.join(
            path_fdz,
            f"dark_{date_utc}_{int(self.config.obs.exptime)}s_{self.config.obs.camera}.link",  # mind exp can be list
        )  # 7DT01/flat_20250102_100_C3.link
        self.config.preprocess.mflat_link = os.path.join(
            path_fdz,
            f"flat_{date_utc}_{self.config.obs.filter}_{self.config.obs.camera}.link",
        )  # 7DT01/flat_20250102_m625_C3.link

    def check_masterframe_status(self):
        # fill mbias_file if on-date masterframe exists. otherwise leave it empty.
        pass

    def _define_settings(self):
        # use local astrometric reference catalog for tile observations
        self.config.settings.local_astref = bool(re.fullmatch(r"T\d{5}", self.config.obs.obj))

        # skip single frame combine for Deep mode
        obsmode = self.raw_header_sample["OBSMODE"]
        self.config.obs.obsmode = obsmode

        self.config.settings.daily_stack = False if obsmode.lower() == "deep" else True

    def read_config(self, config_file):
        """Read configuration from YAML file."""
        with open(config_file, "r") as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    def write_config(self):
        """
        Write current configuration to a YAML file.

        Generates a configuration filename using observation details:
        - Checks if configuration is initialized
        - Creates a filename with unit, object, datetime, filter, and exposure
        - Writes configuration dictionary to the output path
        """

        if not self.is_initialized or not self.write:
            return

        self._config_in_dict["info"]["last_update_datetime"] = datetime.now().isoformat()

        config_file = self.path.output_yml
        self.config_file = config_file

        with open(self.config_file, "w") as f:
            yaml.dump(self.config_in_dict, f, sort_keys=False)
