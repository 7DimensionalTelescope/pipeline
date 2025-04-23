import yaml
import os
import re
import glob
import json
from datetime import datetime
from astropy.table import Table
from .utils import (
    header_to_dict,
    to_datetime_string,
    find_raw_path,
    define_output_dir,
    get_camera,
    parse_exptime,
    clean_up_folder,
    swap_ext,
)
from .const import (
    FACTORY_DIR,
    PROCESSED_DIR,
    MASTER_FRAME_DIR,
    REF_DIR,
    STACKED_DIR,
    DAILY_STACKED_DIR,
)


class Configuration:
    """
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
        obs_params=None,
        config_source=None,
        logger=None,
        overwrite=False,
        return_base=False,
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
        if return_base:
            config_source = config_source or os.path.join(REF_DIR, "base.yml")
            # self._initialized = True
        elif overwrite:
            # Default config source if not provided
            if config_source is None:
                config_source = os.path.join(REF_DIR, "base.yml")
            self._initialized = False
        else:
            config_source = self._find_config_file(obs_params, **kwargs)
            self.config_file = config_source

        self._load_config(config_source, **kwargs)

        if return_base:
            return  # self  # init must return None

        if not self._initialized:
            self.initialize(obs_params)

        if overwrite:
            clean_up_folder(self.config.path.path_processed)
            clean_up_folder(self.config.path.path_factory)
            clean_up_folder(self.config.path.path_stacked)

        self.logger = self._setup_logger(logger, verbose=verbose)
        self.write_config()

        self.config.flag.configuration = True
        self.logger.info(f"Configuration initialized")
        self.logger.debug(f"Configuration file: {self.config_file}")

    def __repr__(self):
        return self.config.__repr__()

    @classmethod
    def base_config(cls, working_dir=None, config_file=None, **kwargs):
        """Return the base (base.yml) configuration instance."""
        # config = cls(return_base=True, **kwargs).config  # never overwrite here
        instance = cls(return_base=True, **kwargs)
        config = instance.config  # configuration instance from the new object
        config.name = "user-input"

        if working_dir is not None:
            config.path.path_processed = working_dir
            factory_dir = os.path.join(working_dir, "factory")
            os.makedirs(factory_dir, exist_ok=True)
            config.path.path_factory = factory_dir

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

        return config

    @classmethod
    def from_obs(cls, obs, **kwargs):
        return cls(obs.obs_params, **kwargs)

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
            from .logger import Logger

            logger = Logger(name=self.config.name, slack_channel="pipeline_report")

        filename = f"{self.output_name}.log"
        log_file = os.path.join(self.config.path.path_processed, filename)
        self.config.logging.file = log_file
        logger.set_output_file(log_file, overwrite=overwrite)
        logger.set_format(self.config.logging.format)
        logger.set_pipeline_name(self.output_name)
        if not (verbose):
            logger.set_level("WARNING")

        return logger

    def _load_config(self, config_source, **kwargs):
        # Load configuration from file or dict
        self._loaded = False
        input_dict = self.read_config(config_source) if isinstance(config_source, str) else config_source

        self._config_in_dict = input_dict

        self.config = ConfigurationInstance(self)

        self._output_prefix = kwargs.pop("path_processed", PROCESSED_DIR)
        self._update_with_kwargs(kwargs)
        self._make_instance(self._config_in_dict)
        self._loaded = True

    def _find_config_file(self, obs_params, **kwargs):
        """Find the configuration file in the processed directory."""
        base_dir = kwargs.get("path_processed", PROCESSED_DIR)
        tmp_path = define_output_dir(
            obs_params["date"],
            obs_params["n_binning"],
            obs_params["gain"],
            obj=obs_params["obj"],
            unit=obs_params["unit"],
            filt=obs_params["filter"],
        )
        base_dir = os.path.join(base_dir, tmp_path)
        config_files = glob.glob(f"{base_dir}/*.yml")
        if len(config_files) == 0:
            self._initialized = False
            return os.path.join(REF_DIR, "base.yml")
        else:
            self._initialized = True
            return config_files[0]

    def initialize(self, obs_params):
        """Fill in obs info, name, paths."""
        # Set core observation details
        self.config.obs.unit = obs_params["unit"]
        self.config.obs.date = obs_params["date"]
        self.config.obs.object = obs_params["obj"]
        self.config.obs.filter = obs_params["filter"]
        self.config.obs.n_binning = obs_params["n_binning"]
        self.config.obs.gain = obs_params["gain"]
        self.config.obs.pixscale = self.config.obs.pixscale * float(obs_params["n_binning"])  # For initial solve
        self.config.name = f"{obs_params['date']}_{obs_params['n_binning']}x{obs_params['n_binning']}_gain{obs_params['gain']}_{obs_params['obj']}_{obs_params['unit']}_{obs_params['filter']}"
        self.config.info.creation_datetime = datetime.now().isoformat()

        self._define_paths()
        self._define_files()
        self._define_settings()
        self._initialized = True

    @property
    def is_initialized(self):
        return self._initialized

    @property
    def is_loaded(self):
        return self._loaded

    @property
    def output_name(self):
        """
        Generate a standardized output filename for calibrated data.

        The filename follows a specific naming convention that includes:
        - Prefix 'calib_'
        - Observation unit identifier
        - Object name
        - Observation datetime (formatted as YYYYMMDD_HHMMSS)
        - Optional additional parameters (if applicable)

        Returns:
            str: Formatted filename for the calibrated data product
                 Example: calib_7DT11_T00139_20250102_014643_m425_100.0.fits

        Raises:
            AttributeError: If the configuration is not fully initialized
        """
        return (
            f"calib_{self.config.obs.unit}_{self.config.obs.object}_"
            f"{to_datetime_string(self.config.obs.datetime[0])}_"
            f"{self.config.obs.filter}_{self.config.obs.exposure[0]}"
        )

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

    def _define_paths(self):
        """Create and set output directory paths for processed data."""
        _date_dir = define_output_dir(self.config.obs.date, self.config.obs.n_binning, self.config.obs.gain)

        rel_path = os.path.join(
            _date_dir,
            self.config.obs.object,
            self.config.obs.unit,
            self.config.obs.filter,
        )
        fdz_rel_path = os.path.join(
            _date_dir,
            self.config.obs.unit,
        )

        path_processed = os.path.join(self._output_prefix, rel_path)
        path_factory = os.path.join(FACTORY_DIR, rel_path)
        path_fdz = os.path.join(MASTER_FRAME_DIR, fdz_rel_path)
        path_stacked_prefix = STACKED_DIR if self.config.name == "user-input" else DAILY_STACKED_DIR
        path_stacked = os.path.join(path_stacked_prefix, rel_path)

        os.makedirs(path_processed, exist_ok=True)
        os.makedirs(path_fdz, exist_ok=True)
        os.makedirs(path_factory, exist_ok=True)
        # os.makedirs(path_stacked, exist_ok=True)  # make dir in imstack

        self.config.path.path_processed = path_processed
        self.config.path.path_stacked = path_stacked
        self.config.path.path_factory = path_factory
        self.config.path.path_fdz = path_fdz
        self.config.path.path_raw = find_raw_path(
            self.config.obs.unit,
            self.config.obs.date,
            self.config.obs.n_binning,
            self.config.obs.gain,
        )
        self.config.path.path_sex = os.path.join(REF_DIR, "srcExt")
        self._add_metadata(_date_dir)

    def _add_metadata(self, name):

        metadata_path = os.path.join(self._output_prefix, name, "metadata.ecsv")
        if not os.path.exists(metadata_path):
            with open(metadata_path, "w") as f:
                f.write("# %ECSV 1.0\n")
                f.write("# ---\n")
                f.write("# datatype:\n")
                f.write("# - {name: object, datatype: string}\n")
                f.write("# - {name: unit, datatype: string}\n")
                f.write("# - {name: filter, datatype: string}\n")
                f.write("# - {name: n_binning, datatype: int64}\n")
                f.write("# - {name: gain, datatype: int64}\n")
                f.write("# meta: !!omap\n")
                f.write("# - {created: " + datetime.now().isoformat() + "}\n")
                f.write("# schema: astropy-2.0\n")
                f.write("object unit filter n_binning gain\n")

        observation_data = [
            str(self.config.obs.object),
            str(self.config.obs.unit),
            str(self.config.obs.filter),
            str(self.config.obs.n_binning),
            str(self.config.obs.gain),
        ]
        new_line = f"{' '.join(observation_data)}\n"
        with open(metadata_path, "a") as f:
            f.write(new_line)

    def _define_files(self):
        s = f"{self.config.path.path_raw}/*{self.config.obs.object}_{self.config.obs.filter}_{self.config.obs.n_binning}*.fits"  # obsdata/7DT11/*T00001*.fits

        fits_files = sorted(glob.glob(s))

        # Extract header metadata
        header_mapping = {
            "ra": "RA",
            "dec": "DEC",
            "datetime": "DATE-OBS",
            "exposure": "EXPOSURE",
        }

        for config_key, header_key in header_mapping.items():
            setattr(self.config.obs, config_key, [])

        raw_files = []
        for fits_file in fits_files:
            header_file = swap_ext(fits_file, ".head")
            if os.path.exists(header_file):
                header_in_dict = header_to_dict(header_file)
            else:
                from astropy.io import fits

                header_in_dict = dict(fits.getheader(fits_file))

            if header_in_dict["GAIN"] == self.config.obs.gain:
                raw_files.append(fits_file)
                for config_key, header_key in header_mapping.items():
                    getattr(self.config.obs, config_key).append(header_in_dict[header_key])
            self.raw_header_sample = header_in_dict

        self.config.file.raw_files = raw_files
        self.config.file.processed_files = [
            os.path.join(
                self.config.path.path_processed,
                f"calib_{self.config.obs.unit}_{self.config.obs.object}_"
                f"{to_datetime_string(datetime)}_"
                f"{self.config.obs.filter}_{int(exp)}s.fits",
            )
            for datetime, exp in zip(self.config.obs.datetime, self.config.obs.exposure)
        ]

        # Identify Camera from image size
        self.config.obs.camera = get_camera(self.raw_header_sample)

        # Define pointer fpaths to master frames
        path_fdz = self.config.path.path_fdz  # master_frame/date_bin_gain/unit
        date_utc = to_datetime_string(self.config.obs.datetime[0], date_only=True)
        # legacy gppy used tool.calculate_average_date_obs('DATE-OBS')
        self.config.preprocess.mbias_link = os.path.join(
            path_fdz, f"bias_{date_utc}_{self.config.obs.camera}.link"
        )  # 7DT01/bias_20250102_C3.link
        self.config.preprocess.mdark_link = os.path.join(
            path_fdz,
            f"dark_{date_utc}_{int(self.config.obs.exposure[0])}s_{self.config.obs.camera}.link",
        )  # 7DT01/flat_20250102_100_C3.link
        self.config.preprocess.mflat_link = os.path.join(
            path_fdz,
            f"flat_{date_utc}_{self.config.obs.filter}_{self.config.obs.camera}.link",
        )  # 7DT01/flat_20250102_m625_C3.link

    def _define_settings(self):
        # use local astrometric reference catalog for tile observations
        self.config.settings.local_astref = bool(re.fullmatch(r"T\d{5}", self.config.obs.object))

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

        if not self.is_initialized:
            return

        self._config_in_dict["info"]["last_update_datetime"] = datetime.now().isoformat()

        filename = f"{self.output_name}.yml"

        config_file = os.path.join(self.config.path.path_processed, filename)
        self.config_file = config_file

        with open(config_file, "w") as f:
            yaml.dump(self.config_in_dict, f)


class ConfigurationInstance:
    def __init__(self, parent_config=None, section=None):
        self._parent_config = parent_config
        self._section = section

    def __setattr__(self, name, value):
        if name.startswith("_"):
            return super().__setattr__(name, value)

        if self._parent_config:
            # Update the configuration dictionary
            if self._section:
                # For nested configurations
                if self._section not in self._parent_config.config_in_dict:
                    self._parent_config.config_in_dict[self._section] = {}
                self._parent_config.config_in_dict[self._section][name] = value
            else:
                # For top-level configurations
                self._parent_config.config_in_dict[name] = value

            # Always write config if initialized
            if (
                hasattr(self._parent_config, "is_initialized")
                and self._parent_config.is_initialized
                and self._parent_config.is_loaded
            ):
                self._parent_config.write_config()

        super().__setattr__(name, value)

    def __repr__(self, indent_level=0):
        indent = "  " * indent_level
        repr_lines = []

        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue

            # Handle nested ConfigurationInstance
            if isinstance(v, ConfigurationInstance):
                repr_lines.append(f"{indent}  {k}:")
                repr_lines.append(v.__repr__(indent_level + 1))
            elif isinstance(v, dict):
                repr_lines.append(f"{indent}  {k}:")
                for dict_k, dict_v in v.items():
                    repr_lines.append(f"{indent}    {dict_k}: {dict_v}")
            else:
                repr_lines.append(f"{indent}  {k}: {v}")

        return "\n".join(repr_lines)
