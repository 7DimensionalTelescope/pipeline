import yaml
import os
from datetime import datetime
from .. import __version__
from ..utils import (
    merge_dicts,
)
from ..path.path import PathHandler


class ConfigurationMixin:
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

    def __repr__(self):
        return self.config.__repr__()

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
        self._config_in_dict = merge_dicts(self._config_in_dict, lower_kwargs)

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

                instance._config_in_dict = merge_dicts(instance._config_in_dict, new_config)
                instance._make_instance(instance._config_in_dict)
            else:
                raise FileNotFoundError("Provided Configuration file does not exist")

        instance._initialized = True
        # return config
        return instance

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

    def _setup_logger(self, logger=None, overwrite=True, verbose=True):
        if logger is None:
            from .services.logger import Logger

            logger = Logger(name=self.config.name, slack_channel="pipeline_report")

        if self.write:
            self.config.logging.file = self.log_file
            logger.set_output_file(self.log_file, overwrite=overwrite)
            logger.set_format(self.config.logging.format)
            logger.set_pipeline_name(self.path._output_name)

        if not (verbose):
            logger.set_level("WARNING")

        return logger

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

        with open(self.config_file, "w") as f:
            yaml.dump(self.config_in_dict, f, sort_keys=False)


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

    # def to_dict(self):
    #     """shallow"""
    #     result = {}
    #     for k, v in self.__dict__.items():
    #         if k.startswith("_"):
    #             continue
    #         result[k] = v.to_dict() if isinstance(v, ConfigurationInstance) else v
    #     return result

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

    # def extract_single_image_config(self, i: int):
    #     from copy import deepcopy

    #     config_dict = deepcopy(self._parent_config.config_in_dict)
    #     sections = [
    #         config_dict.get("obs", {}),
    #         config_dict["file"].get("raw_images", {}),
    #         config_dict["file"].get("processed_images", {}),
    #     ]
    #     for section in sections:
    #         for k, v in section.items():
    #             if isinstance(v, list):
    #                 section[k] = v[i]
    #     return Configuration(config_source=config_dict, write=False)

    @staticmethod
    def select_from_lists(obj, i):
        if isinstance(obj, dict):
            return {k: ConfigurationInstance.select_from_lists(v, i) for k, v in obj.items()}
        elif isinstance(obj, list):
            try:
                # return obj[i]
                return [obj[i]]  # wrap the selected value back in a list; guard against slicing
            except IndexError:
                raise IndexError(f"Index {i} out of bounds for list: {obj}")
        else:
            return obj

    def extract_single_image_config(self, i: int):
        """return Configuration, not ConfigurationInstance"""
        from copy import deepcopy

        # Deep copy to avoid mutation
        config_dict = deepcopy(self._parent_config.config_in_dict)

        # Recursively reduce all list values to i-th element
        config_dict = self.select_from_lists(config_dict, i)

        return Configuration(config_source=config_dict, write=False)
        # return Configuration.from_dict(config_dict, write=False)
