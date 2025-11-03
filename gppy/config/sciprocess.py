import os
import glob
import time
from datetime import datetime

from .. import __version__
from ..const import PipelineError
from ..utils import clean_up_folder, clean_up_sciproduct, get_header, atleast_1d, time_diff_in_seconds, collapse
from ..path.path import PathHandler
from .base import BaseConfig


class SciProcConfiguration(BaseConfig):
    def __init__(
        self,
        input: list[str] | str | dict = None,
        logger=None,
        write=True,  # False for PhotometrySingle
        clear_dirs=False,  # clear factory_dir and output_dir
        verbose=True,
        overwrite=False,
        **kwargs,
    ):
        st = time.time()
        self.write = write

        self._handle_input(input, logger, verbose, **kwargs)

        if not self._initialized:
            self.logger.info("Initializing configuration")
            self.initialize(**kwargs)
            self.logger.info(f"'SciProcConfiguration' initialized in {time_diff_in_seconds(st)} seconds")
            self.logger.info(f"Writing configuration to file: {os.path.basename(self.config_file)}")
            self.logger.debug(f"Full path to the configuration file: {self.config_file}")

        if clear_dirs:
            self.logger.info("Overwriting factory_dir first")
            clean_up_folder(self.path.factory_dir)
            clean_up_sciproduct(self.path.output_dir)

        if not os.path.exists(self.config_file) or overwrite:
            self.write_config()
        self.logger.info("Completed to load configuration")

    # @classmethod
    # def base_config(cls, input_images=None, config_file=None, config_dict=None, working_dir=None, **kwargs):
    #     """Return the base (base.yml) ConfigurationInstance."""
    #     working_dir = working_dir or os.getcwd()
    #     if config_file is not None:
    #         config_file = os.path.join(working_dir, config_file) if working_dir else config_file
    #         if os.path.exists(config_file):
    #             config = cls.from_config(config_file=config_file, **kwargs)
    #         else:
    #             raise FileNotFoundError("Provided Configuration file does not exist")
    #     elif config_dict is not None:
    #         config = cls.from_dict(config_dict=config_dict, **kwargs)
    #     else:
    #         raise ValueError("Either config_file, config_type or config_dict must be provided")

    #     config.name = "user-input"
    #     config._initialized = True
    #     return config

    @classmethod
    def base_config(
        cls,
        input_images=None,
        working_dir=None,
        config_file=None,
        write=True,
        logger=None,
        verbose=True,
        force_creation=False,
        **kwargs,
    ):
        """Base configuration for user-input. Mind config.settings.is_pipeline is set to False"""
        # self = cls.__new__(cls)

        input_images = sorted([os.path.abspath(image) for image in atleast_1d(input_images)])
        path = PathHandler(input_images, working_dir=working_dir or os.getcwd())
        self = cls.from_config(path.sciproc_base_yml)
        self.path = path
        self.config_file = config_file or self.path.sciproc_output_yml
        if isinstance(self.config_file, list):
            if force_creation:
                print(f"[WARNING] config name is not uniquely defined. Using the last one.")
                self.config_file = collapse(sorted(self.path.sciproc_output_yml)[::-1], force=True)
            else:
                raise PipelineError(
                    "Inhomogeneous input images; config name is not uniquely defined. Use force_creation=True to use the last one."
                )
        # config_source = self.path.sciproc_base_yml
        # super().__init__(self, config_source=config_source, write=write, **kwargs)
        # self.initialize(is_pipeline=False)

        if logger is True:
            log_file = self.path.sciproc_output_log
            if isinstance(log_file, list):
                print(f"[WARNING] log filename is not uniquely defined. Using the last one.")
                log_file = collapse(sorted(log_file)[::-1], force=True)
            self.config.logging.file = log_file
            self.logger = cls._setup_logger(
                name=self.name,
                log_file=self.config.logging.file,
                verbose=verbose,
                overwrite=write,
                **kwargs,
            )

        elif logger:
            self.logger = logger
        else:
            self.logger = None

        self.config.input.calibrated_images = atleast_1d(input_images)
        # self.config.name = "user-input"
        self.config.name = self.name
        self.config.settings.is_pipeline = False
        self.config._initialized = True
        return self

    @classmethod
    def from_list(cls, input_images, working_dir=None, is_pipeline=False, **kwargs):
        # self.input_files = sorted(input)
        # self.path = PathHandler(input)
        # config_source = self.path.sciproc_base_yml
        # self.logger = self._setup_logger(
        #     logger,
        #     name=self.name,
        #     log_file=self.path.sciproc_output_log,
        #     verbose=verbose,
        #     overwrite=self.write,
        # )
        # self.logger.info("Generating 'SciProcConfiguration' from the 'base' configuration")
        # self.logger.debug(f"Configuration source: {config_source}")
        # super().__init__(config_source=config_source, write=self.write, **kwargs)
        # self.config.info.file = config_source
        # self.config.logging.file = self.path.sciproc_output_log

        self = cls.base_config(input_images=input_images, working_dir=working_dir, logger=True, **kwargs)
        # emulate constructor’s initialize path
        self._initialized = False
        self.input_files = atleast_1d(input_images)
        self.initialize(is_pipeline=is_pipeline)
        return self

    @property
    def name(self):
        if hasattr(self, "config_file") and self.config_file is not None:
            return os.path.splitext(os.path.basename(self.config_file))[0]
        elif hasattr(self, "path"):
            return os.path.basename(self.path.sciproc_output_yml).replace(".yml", "")
        elif hasattr(self.config, "name"):
            return self.config.name
        else:
            return None

    def _handle_input(self, input, logger, verbose, **kwargs):
        if isinstance(input, list) or (isinstance(input, str) and input.endswith(".fits")):  # list of science images
            self.input_files = sorted(input)
            self.path = PathHandler(input)
            config_source = self.path.sciproc_base_yml
            self.logger = self._setup_logger(
                logger,
                name=self.name,
                log_file=self.path.sciproc_output_log,
                verbose=verbose,
                overwrite=self.write,
            )
            self.logger.info("Generating 'SciProcConfiguration' from the 'base' configuration")
            self.logger.debug(f"Configuration source: {config_source}")
            super().__init__(config_source=config_source, write=self.write, **kwargs)
            self.config.info.file = config_source
            self.config.logging.file = self.path.sciproc_output_log
            # raise PipelineError("Initializing 'SciProcConfiguration' from a list of images is not supported anymore. Please use 'SciProcConfiguration.base_config' instead.")

        elif isinstance(input, str | dict):  # path of a config file
            config_source = input
            super().__init__(config_source=config_source, write=self.write, **kwargs)
            # working_dir = os.path.dirname(config_source) if isinstance(config_source, str) else None
            self.path = self._set_pathhandler_from_config()  # working_dir=working_dir)
            self.config.logging.file = self.path.sciproc_output_log
            self.logger = self._setup_logger(
                logger,
                name=self.name,
                log_file=self.config.logging.file,
                verbose=verbose,
                overwrite=False,
            )
            self._initialized = True
            self.logger.info("Loading 'SciProcConfiguration' from an exisiting file or dictionary")
            self.logger.debug(f"Configuration source: {config_source}")

        else:
            raise ValueError("Input must be a list of image files, a configuration file path, or a configuration dictionary.")  # fmt: skip
        # used by write_config
        self.config_file = self.path.sciproc_output_yml  # used by write_config
        return

    def _set_pathhandler_from_config(self, working_dir=None):
        # mind the check order
        if hasattr(self.config, "input"):
            if hasattr(self.config.input, "calibrated_images") and self.config.input.calibrated_images:
                return PathHandler(self.config.input.calibrated_images, working_dir=working_dir)

            if hasattr(self.config.input, "processed_dir") and self.config.input.processed_dir:
                f = os.path.join(self.config.input.processed_dir, "**.fits")
                return PathHandler(sorted(glob.glob(f)), working_dir=working_dir)

            if hasattr(self.config.input, "stacked_image") and self.config.input.stacked_image:
                return PathHandler(self.config.input.stacked_image, working_dir=working_dir)

        raise ValueError("Configuration does not contain valid input files or directories to create PathHandler.")

    def initialize(self, is_pipeline=True):
        """Fill in universal info, filenames, settings."""

        self.config.info.version = __version__
        self.config.info.creation_datetime = datetime.now().isoformat()
        self.config.name = self.config.name or self.name

        self.config.input.calibrated_images = atleast_1d(PathHandler(self.input_files).processed_images)
        self.config.input.output_dir = self.path.output_dir

        # self.set_input_output()
        # self.check_masterframe_status()

        if is_pipeline:
            self.config.settings.is_pipeline = True
        self._define_settings(self.input_files[0])
        self.input_files = self.config.input.calibrated_images
        self._initialized = True

    def _define_settings(self, input_file):
        # use local astrometric reference catalog for tile observations
        # self.config.settings.local_astref = bool(re.fullmatch(r"T\d{5}", self.config.obs.obj))

        try:
            # skip single frame combine for Deep mode
            raw_header_sample = get_header(input_file)
            try:
                obsmode = raw_header_sample["OBSMODE"]
            except KeyError:
                self.logger.warning("OBSMODE keyword not found in the header. Defaulting to 'spec'.")
                obsmode = "spec"
            # self.config.obs.obsmode = obsmode
            self.config.settings.daily_stack = False if obsmode.lower() == "deep" else True
        except Exception as e:
            self.logger.warning(f"Failed to define settings: {e}")
