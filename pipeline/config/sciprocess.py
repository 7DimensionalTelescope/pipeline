from __future__ import annotations
import os
from pathlib import Path
import glob
import time
from datetime import datetime
from astropy.io import fits

from .. import __version__
from ..const import PipelineError
from ..utils import clean_up_folder, clean_up_sciproduct, get_header, atleast_1d, time_diff_in_seconds, collapse
from ..path.path import PathHandler
from .base import BaseConfig
from ..services.database.too import TooDB
from ..path.name import NameHandler


class SciProcConfiguration(BaseConfig):
    def __init__(
        self,
        input: list[str] | str | dict = None,
        logger=None,
        write=True,  # False for PhotometrySingle
        clear_dirs=False,  # clear factory_dir and output_dir
        verbose=True,
        overwrite=False,
        is_too=False,
        **kwargs,
    ):
        st = time.time()
        self.write = write

        self._handle_input(input, logger, verbose, is_too=is_too, **kwargs)

        if not self._initialized:
            self.logger.info("Initializing configuration")
            self.initialize(is_too=is_too, **kwargs)
            self.logger.info(f"'SciProcConfiguration' initialized in {time_diff_in_seconds(st)} seconds")
            self.logger.info(f"Writing configuration to file: {os.path.basename(self.config_file)}")
            self.logger.debug(f"Full path to the configuration file: {self.config_file}")

        if clear_dirs:
            self.logger.info("Overwriting factory_dir first")
            clean_up_folder(self.path.factory_dir)
            clean_up_sciproduct(self.path.output_dir)

        # fill in missing keys
        self.fill_missing_from_yaml()

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
        input_images: list[str] | str = None,
        working_dir: str = None,
        config_file: str = None,
        write: bool = True,
        logger: bool | "Logger" = None,
        verbose: bool = True,
        force_creation: bool = False,
        is_too: bool = False,
        **kwargs,
    ):
        """Base configuration for user-input. Mind config.settings.is_pipeline is set to False"""
        # self = cls.__new__(cls)

        input_images = sorted([os.path.abspath(image) for image in atleast_1d(input_images)])
        path = PathHandler(input_images, working_dir=working_dir or os.getcwd(), is_too=is_too)
        self = cls.from_config(path.sciproc_base_yml, is_too=is_too)
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
            self.node.logging.file = log_file
            self.logger = cls._setup_logger(
                name=self.name,
                log_file=self.node.logging.file,
                verbose=verbose,
                overwrite=write,
                **kwargs,
            )

        elif logger:
            self.logger = logger
        else:
            self.logger = None

        self.node.input.calibrated_images = atleast_1d(input_images)
        # self.config.name = "user-input"
        self.node.name = self.name
        self.node.settings.is_pipeline = is_too
        self.fill_missing_from_yaml()
        self.node.info.version = __version__
        self._initialized = True
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
        elif hasattr(self.node, "name"):
            return self.node.name
        else:
            return None

    def _handle_input(self, input, logger, verbose, is_too=False, **kwargs):
        if isinstance(input, list) or (isinstance(input, str) and input.endswith(".fits")):  # list of science images
            self.input_files = sorted(input)
            self.path = PathHandler(input, is_too=is_too)
            config_source = self.path.sciproc_base_yml
            log_file = self.path.sciproc_output_log

            self.logger = self._setup_logger(
                logger,
                name=self.name,
                log_file=log_file,
                verbose=verbose,
                overwrite=self.write,
            )
            self.logger.info("Generating 'SciProcConfiguration' from the 'base' configuration")
            self.logger.debug(f"Configuration source: {config_source}")
            super().__init__(config_source=config_source, write=self.write, is_too=is_too, **kwargs)
            self.node.logging.file = log_file
            # raise PipelineError("Initializing 'SciProcConfiguration' from a list of images is not supported anymore. Please use 'SciProcConfiguration.base_config' instead.")

        elif isinstance(input, str | dict):  # path of a config file
            config_source = input
            super().__init__(config_source=config_source, write=self.write, is_too=is_too, **kwargs)
            # working_dir = os.path.dirname(config_source) if isinstance(config_source, str) else None
            is_too = is_too or self.node.settings.is_too
            self.path = self._set_pathhandler_from_config(is_too=is_too)  # working_dir=working_dir)
            self.node.logging.file = self.path.sciproc_output_log

            if isinstance(config_source, str):
                self.config_file = config_source  # use the filename as is
            self.logger = self._setup_logger(
                logger,
                name=self.name,
                log_file=self.node.logging.file,
                verbose=verbose,
                overwrite=False,
            )
            self._initialized = True
            self.logger.info("Loading 'SciProcConfiguration' from an exisiting file or dictionary")
            self.logger.debug(f"Configuration source: {config_source}")

        else:
            raise ValueError("Input must be a list of image files, a configuration file path, or a configuration dictionary.")  # fmt: skip

        # used by write_config
        if not hasattr(self, "config_file"):
            self.config_file = self.path.sciproc_output_yml  # used by write_config

        return

    def _update_too_times(self, input_file):
        """Update ToO database with transfer_time from input file creation time."""

        # Try to find and update the ToO record
        too_db = TooDB()

        input_files = atleast_1d(input_file)

        earliest_time = None
        observation_time = None
        for input_file in input_files:
            if os.path.exists(input_file):
                file_time = datetime.fromtimestamp(os.path.getctime(input_file))
                if earliest_time is None or file_time < earliest_time:
                    earliest_time = file_time

                # Parse DATE-OBS - handle ISO format with T separator and milliseconds
                # DATE-OBS in FITS is UTC (as per FITS standard), convert to KST for storage
                try:
                    date_obs_str = fits.getval(input_file, "DATE-OBS")
                    try:
                        # Try ISO format first (handles 'T' separator and milliseconds)
                        obs_time = datetime.fromisoformat(date_obs_str.replace("Z", "").replace("+00:00", ""))
                    except ValueError:
                        # Fall back to space-separated format
                        try:
                            obs_time = datetime.strptime(date_obs_str, "%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            # Try with milliseconds
                            obs_time = datetime.strptime(date_obs_str, "%Y-%m-%dT%H:%M:%S.%f")

                    # Convert from UTC to KST (Asia/Seoul, UTC+9)
                    import pytz

                    obs_time_utc = pytz.UTC.localize(obs_time)
                    kst = pytz.timezone("Asia/Seoul")
                    obs_time = obs_time_utc.astimezone(kst)

                    if observation_time is None or obs_time < observation_time:
                        observation_time = obs_time
                except (KeyError, ValueError) as e:
                    # DATE-OBS not found or unparseable, skip
                    if hasattr(self, "logger") and self.logger:
                        self.logger.debug(f"Could not parse DATE-OBS from {input_file}: {e}")
                    continue

        # First, try using config_file if available
        if hasattr(self, "config_file") and self.config_file:
            # Handle case where config_file might be a list
            config_file = self.config_file
            base_path = str(Path(config_file).parent.parent)
            if isinstance(config_file, list):
                config_file = collapse(sorted(config_file)[::-1], force=True) if config_file else None

            if config_file:
                try:
                    too_data = too_db.read_too_data(config_file=config_file)
                    if too_data and too_db.too_id:
                        too_db.update_too_data(
                            too_id=too_db.too_id, transfer_time=earliest_time, observation_time=observation_time
                        )
                        too_db.update_too_data(too_id=too_db.too_id, base_path=base_path)
                        if hasattr(self, "logger") and self.logger:
                            self.logger.info(f"Updated ToO transfer_time to {earliest_time.isoformat()}")

                        too_db.send_initial_notice_email(too_db.too_id)

                except Exception as e:
                    if hasattr(self, "logger") and self.logger:
                        self.logger.warning(f"Failed to update ToO transfer_time: {e}")

    def _set_pathhandler_from_config(self, working_dir=None, is_too=False):
        # mind the check order
        if hasattr(self.node, "input"):
            if hasattr(self.node.input, "calibrated_images") and self.node.input.calibrated_images:
                return PathHandler(self.node.input.calibrated_images, working_dir=working_dir, is_too=is_too)

            if hasattr(self.node.input, "processed_dir") and self.node.input.processed_dir:
                f = os.path.join(self.node.input.processed_dir, "**.fits")
                return PathHandler(sorted(glob.glob(f)), working_dir=working_dir, is_too=is_too)

            if hasattr(self.node.input, "stacked_image") and self.node.input.stacked_image:
                return PathHandler(self.node.input.stacked_image, working_dir=working_dir, is_too=is_too)

        raise ValueError("Configuration does not contain valid input files or directories to create PathHandler.")

    def initialize(self, is_pipeline=True, is_too=False):
        """Fill in universal info, filenames, settings."""

        self.node.info.version = __version__
        self.node.info.creation_datetime = datetime.now().isoformat()
        self.node.info.file = self.config_file
        self.node.name = self.node.name or self.name

        self.node.input.calibrated_images = atleast_1d(PathHandler(self.input_files, is_too=is_too).processed_images)

        if is_too:
            # base_name = os.path.basename(self.config_file).replace(".yml", "").replace(".yaml", "")
            # min_time = NameHandler.calculate_too_time(self.input_files)
            # self.config_file = self.config_file.replace(base_name, f"{base_name}_ToO_{min_time}")
            # self.node.name = os.path.splitext(os.path.basename(self.config_file))[0]
            self._update_too_times(self.input_files)

        self.node.input.output_dir = self.path.output_dir

        # self.set_input_output()
        # self.check_masterframe_status()

        if is_pipeline:
            self.node.settings.is_pipeline = True
        self._define_settings(self.input_files[0])
        self.input_files = self.node.input.calibrated_images

        self.fill_missing_from_yaml()

        self._initialized = True

    def _define_settings(self, input_file):
        try:
            # skip single frame combine for Deep mode
            raw_header_sample = get_header(input_file)
            try:
                obsmode = raw_header_sample["OBSMODE"]
            except KeyError:
                self.logger.warning("OBSMODE keyword not found in the header. Defaulting to 'spec'.")
                obsmode = "spec"
            # self.config.obs.obsmode = obsmode
            self.node.settings.daily_stack = False if obsmode.lower() == "deep" else True
        except Exception as e:
            self.logger.warning(f"Failed to define settings: {e}")
