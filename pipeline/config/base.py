import os
import yaml
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Self

from .. import __version__
from .utils import merge_dicts, merge_missing
from ..path.path import PathHandler


class BaseConfig(ABC):

    def __init__(self, config_source=None, write=True, **kwargs) -> None:
        # Don't pass is_too here. Child classes should handle it.
        self._initialized = False
        self.write = write

        self._load_config(config_source, **kwargs)

    def __repr__(self):
        if hasattr(self, "node"):
            return self.node.__repr__()
        else:
            return f"BaseConfig"

    @property
    def is_initialized(self):
        return self._initialized

    @property
    def config_in_dict(self):
        """Return the configuration dictionary."""
        return self._config_in_dict

    @classmethod
    def from_dict(cls, input, **kwargs):
        """No self.config_file, write is always False."""
        # return cls(input, write=write, **kwargs)
        self = cls.__new__(cls)
        self._load_config(config_source=input)
        self.config_file = None
        self.write = False
        self._initialized = True
        return self

    @classmethod
    def from_config(cls, input: str, write=True, is_too=False, **kwargs) -> Self:
        """
        Deprecated. Move away from it.

        Much faster (4.8 ms) than SciProcConfiguration(input, write=write) (36 ms)
        as it defines PathHandler with only the first file, and skips writing
        to disk during initialization.
        """
        print("[DeprecationWarning] Use SciProcConfiguration(input, write=write) instead.")
        # return cls(input, write=write, **kwargs)
        self = cls.__new__(cls)
        self._load_config(config_source=input)
        self.config_file = input
        # initialize PathHandler with the first group of input images
        input_dict = self.node.input.to_dict()
        input_images = next(iter(input_dict.values())) or None  # if empty, use None
        self.path = PathHandler(input_images, is_too=is_too)
        self.write = write
        self._initialized = True
        return self

    @classmethod
    def base_config(cls, write=False):
        self = cls.__new__(cls)

        # same as BaseConfig __init__
        self._initialized = False
        self.write = write

        if cls.__name__ == "PreprocConfiguration":
            self._load_config(PathHandler().preproc_base_yml)
        elif cls.__name__ == "SciProcConfiguration":
            self._load_config(PathHandler().sciproc_base_yml)
        else:
            raise ValueError(f"Invalid class name: {cls.__name__}")

        # self._initialized = True
        return self

    @abstractmethod
    def user_config(cls, **kwargs):
        raise NotImplementedError("User config is not implemented for this class")

    @abstractmethod
    def _set_pathhandler_from_config(self, is_too=False):
        pass

    @abstractmethod
    def initialize(self):
        """
        Responsible for filling in universal info & writing for the first time.
        """
        pass

    def _load_config(self, config_source, **kwargs):
        if isinstance(config_source, str):
            input_dict = self.read_config(config_source)
        elif isinstance(config_source, dict):
            input_dict = config_source
        else:
            raise TypeError("Invalid config_source type: must be str or dict")

        self._config_in_dict = input_dict

        self.node = ConfigNode(self)

        self._update_with_kwargs(**kwargs)

        self._make_nodes()

    def _update_with_kwargs(self, **kwargs):
        """Merge additional configuration parameters."""
        lower_kwargs = {key.lower(): value for key, value in kwargs.items()}
        self._config_in_dict = merge_dicts(self._config_in_dict, lower_kwargs)

    @staticmethod
    def read_config(config_file):
        """Read configuration from YAML file."""
        with open(config_file, "r") as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    def write_config(self, force=False):
        """
        Writes current configuration to a YAML file.

        Writes only if:
        - Configuration is initialized
        - Write is True
        """

        if not self.write and not force:
            return

        # CAVEAT: self._config_in_dict updates are not seen by ConfigNode instances. Use it sparingly.
        self._config_in_dict["info"]["runtime_version"] = __version__
        self._config_in_dict["info"]["last_update_datetime"] = datetime.now().isoformat()

        # print(f"Writing configuration to file: {self.config_file}")

        with open(self.config_file, "w") as f:
            yaml.dump(self._config_in_dict, f, sort_keys=False)  # , default_flow_style=False)

    def _make_nodes(self):
        """
        Transform configuration dictionary into nested, dynamic instances of ConfigNode.
        """
        for key, value in self._config_in_dict.items():
            if isinstance(value, dict):
                nested_dict = {}
                instances = ConfigNode(self, key)
                for subkey, subvalue in value.items():
                    nested_dict[subkey] = subvalue
                    setattr(instances, subkey, subvalue)
                setattr(self.node, key, instances)
                self._config_in_dict[key] = nested_dict
            else:
                setattr(self.node, key, value)
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
        from .sciprocess import SciProcConfiguration

        # Deep copy to avoid mutation
        config_dict = deepcopy(self.config_in_dict)

        # Recursively reduce all list values to i-th element
        config_dict = self.node.select_from_lists(config_dict, i)

        # return BaseConfig(config_source=config_dict, write=False)
        return SciProcConfiguration.from_dict(config_dict)

    def fill_missing_from_yaml(self, base_yaml: str = None, exclude_top_level=None):
        """
        Add ONLY missing keys to the current config from a YAML file of defaults.
        Returns a list of dotted-key paths that were added.
        """

        if exclude_top_level is None:
            exclude_top_level = {"name", "process_id", "info", "logging"}

        base_yaml = base_yaml or getattr(self.path, "sciproc_base_yml", None)
        if not base_yaml or not os.path.exists(base_yaml):
            return

        with open(base_yaml, "r") as f:
            base_defaults = yaml.load(f, Loader=yaml.FullLoader) or {}

        # mutate backing dict in place
        merge_missing(self._config_in_dict, base_defaults, exclude_top_level=exclude_top_level)

        # Rebuild instances once without spamming writes
        self._rebuilding = True
        try:
            self._make_nodes()  # make ConfigNode reflect _config_in_dict changes
        finally:
            self._rebuilding = False

        return

    def override_from_yaml(self, override_yaml: str = None):
        """
        Override existing keys in the current config from a YAML file.
        Unlike fill_missing_from_yaml, this will override existing values.
        """

        if not override_yaml or not os.path.exists(override_yaml):
            return

        with open(override_yaml, "r") as f:
            override_dict = yaml.load(f, Loader=yaml.FullLoader) or {}

        # mutate backing dict in place
        merge_dicts(self._config_in_dict, override_dict)

        # Rebuild instances once without spamming writes
        self._rebuilding = True
        try:
            self._make_nodes()  # make ConfigNode reflect _config_in_dict changes
        finally:
            self._rebuilding = False

        return


class ConfigNode:
    def __init__(self, parent_config=None, section=None):
        self._parent_config: BaseConfig = parent_config
        self._section = section

    def __setattr__(self, name, value):
        if name.startswith("_"):
            return super().__setattr__(name, value)

        if self._parent_config:
            # Check current value before updating
            if self._section:
                # For nested configurations
                current_value = None
                if self._section in self._parent_config.config_in_dict:
                    current_value = self._parent_config.config_in_dict[self._section].get(name)
            else:
                # For top-level configurations
                current_value = self._parent_config.config_in_dict.get(name)

            # Only update and write if value has changed
            if current_value != value:
                # Update the configuration dictionary
                if self._section:
                    # Ensure section exists
                    if self._section not in self._parent_config.config_in_dict:
                        self._parent_config.config_in_dict[self._section] = {}
                    self._parent_config.config_in_dict[self._section][name] = value
                else:
                    self._parent_config.config_in_dict[name] = value

                # Write config if initialized and not explicitly suppressed
                if (
                    self._parent_config
                    and self._parent_config.is_initialized
                    and not getattr(self._parent_config, "_rebuilding", False)
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
            if isinstance(v, ConfigNode):
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
            if isinstance(v, ConfigNode):
                result[k] = v.to_dict()
            else:
                result[k] = copy.deepcopy(v)
        return result

    def extract_single_image_config(self, i: int):
        """Returns ConfigurationInstance"""
        return self._parent_config.extract_single_image_config(i).node

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
                    result[k] = ConfigNode.select_from_lists(v, i)
            return result

        elif isinstance(obj, list):
            # If list is empty, return empty list (valid configuration value)
            if len(obj) == 0:
                return []
            try:
                return [obj[i]]  # pick i-th element (wrapped back in a list)
            except IndexError:
                raise IndexError(f"Index {i} out of bounds for list: {obj}")

        else:
            return obj
