import yaml
import os
from datetime import datetime
from .. import __version__
from .. import const
from .utils import (
    merge_dicts,
)


class BaseConfig:

    def __init__(self, config_source=None, write=True, **kwargs) -> None:
        self._initialized = False
        self.write = write

        self._load_config(config_source, **kwargs)

    def __repr__(self):
        return self.config.__repr__()

    @property
    def is_initialized(self):
        return self._initialized

    @property
    def config_in_dict(self):
        """Return the configuration dictionary."""
        return self._config_in_dict

    @classmethod
    def from_dict(cls, input, write=False, **kwargs):
        return cls(input, write=write, **kwargs)

    @classmethod
    def from_file(cls, input, write=False, **kwargs):
        return cls(input, write=write, **kwargs)

    @classmethod
    def from_base(cls, config_type, **kwargs):
        if config_type == "preprocess":
            config_file = os.path.join(const.REF_DIR, "preproc_base.yml")
            return cls.from_file(config_file, **kwargs)
        elif config_type == "sciprocess":
            config_file = os.path.join(const.REF_DIR, "sciproc_base.yml")
            return cls.from_file(config_file, **kwargs)
        else:
            raise ValueError(f"Invalid config_type: {config_type}")

    @classmethod
    def base_config(cls, config_type=None, config_file=None, config_dict=None, working_dir=None, **kwargs):
        """Return the base (base.yml) ConfigurationInstance."""
        working_dir = working_dir or os.getcwd()
        if config_file is not None:
            config_file = os.path.join(working_dir, config_file) if working_dir else config_file
            if os.path.exists(config_file):
                config = cls.from_file(config_file=config_file, **kwargs)
            else:
                raise FileNotFoundError("Provided Configuration file does not exist")
        elif config_type is not None:
            config = cls.from_base(config_type, **kwargs)
        elif config_dict is not None:
            config = cls.from_dict(config_dict=config_dict, **kwargs)
        else:
            raise ValueError("Either config_file, config_type or config_dict must be provided")

        config.name = "user-input"
        config._initialized = True
        return config

    def _load_config(self, config_source, **kwargs):
        if isinstance(config_source, str):
            input_dict = self.read_config(config_source)
        elif isinstance(config_source, dict):
            input_dict = config_source
        else:
            raise TypeError("Invalid config_source type: must be str or dict")

        self._config_in_dict = input_dict

        self.config = ConfigurationInstance(self)

        self._update_with_kwargs(kwargs)
        self._make_instance()

    def _update_with_kwargs(self, kwargs):
        """Merge additional configuration parameters."""
        lower_kwargs = {key.lower(): value for key, value in kwargs.items()}
        self._config_in_dict = merge_dicts(self._config_in_dict, lower_kwargs)

    @staticmethod
    def read_config(config_file):
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

        with open(self.config_file, "w") as f:
            yaml.dump(self._config_in_dict, f, sort_keys=False)  # , default_flow_style=False)

    def _make_instance(self):
        """
        Transform configuration dictionary into nested, dynamic instances.
        """
        for key, value in self._config_in_dict.items():
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

    @classmethod
    def _setup_logger(cls, logger=None, name=None, log_file=None, verbose=True, overwrite=True, **kwargs):
        if logger is None:
            from ..services.logger import Logger

            logger = Logger(name=name, slack_channel="pipeline_report")

        logger.set_output_file(log_file, overwrite=overwrite)
        if "log_format" in kwargs:
            logger.set_format(kwargs.pop("log_format"))

        if not (verbose):
            logger.set_level("WARNING")

        return logger

    def extract_single_image_config(self, i: int):
        """return a Configuration with i-th element of all lists in the config dict"""
        from copy import deepcopy

        # Deep copy to avoid mutation
        config_dict = deepcopy(self.config_in_dict)

        # Recursively reduce all list values to i-th element
        config_dict = self.config.select_from_lists(config_dict, i)

        return BaseConfig(config_source=config_dict, write=False)
        # return Configuration.from_dict(config_dict, write=False)


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
                # and self._parent_config.is_loaded
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

    def to_dict(self):
        import copy

        result = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            if isinstance(v, ConfigurationInstance):
                result[k] = v.to_dict()
            else:
                result[k] = copy.deepcopy(v)
        return result

    def extract_single_image_config(self, i: int):
        """Returns ConfigurationInstance"""
        return self._parent_config.extract_single_image_config(i).config

    @staticmethod
    def select_from_lists(obj, i):
        # if isinstance(obj, dict):
        #     return {k: ConfigurationInstance.select_from_lists(v, i) for k, v in obj.items()}
        # elif isinstance(obj, list):
        #     try:
        #         return [obj[i]]  # wrap the selected value back in a list; pipeline can work the same way
        #     except IndexError:
        #         raise IndexError(f"Index {i} out of bounds for list: {obj}")
        # else:
        #     return obj

        exclude_keys = {"logging", "settings", "info"}
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                if k in exclude_keys:
                    result[k] = v
                else:
                    result[k] = ConfigurationInstance.select_from_lists(v, i)
            return result

        elif isinstance(obj, list):
            try:
                return [obj[i]]  # pick i-th element (wrapped back in a list)
            except IndexError:
                raise IndexError(f"Index {i} out of bounds for list: {obj}")

        else:
            return obj
