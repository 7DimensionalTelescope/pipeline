from typing import Any, Union
from abc import ABC, abstractmethod
import glob
from ..config import PreprocConfiguration, SciProcConfiguration, ConfigurationInstance
from ..path.path import PathHandler
from .logger import Logger, LockingFileHandler
from .queue import QueueManager
import warnings
import logging

class BaseSetup(ABC):
    def __init__(
        self,
        config: Union[str, PreprocConfiguration, SciProcConfiguration] = None,
        logger: Logger = None,
        queue: Union[bool, Any] = False,
    ) -> None:
        """Initialize the astrometry module.

        Args:
            config: Configuration object or path to config file
            logger: Custom logger instance (optional)
            queue: QueueManager instance or boolean to enable parallel processing
        """
        # Setup PathHandler
        self.path = self._setup_path(config)

        # Setup Configuration
        self.config = self._setup_config(config)

        # Setup log
        self._logger = self._setup_logger(logger, self.config)

        # Setup queue
        self.queue = self._setup_queue(queue)

    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)

    @property
    def logger(self):
        return self._setup_logger(self._logger, self.config)

    def _setup_path(self, config):
        if isinstance(config, PreprocConfiguration | SciProcConfiguration):
            return config.path
        elif isinstance(config, ConfigurationInstance):
            return PathHandler(config)
        elif isinstance(config, str):
            warnings.warn("String path is deprecated. Assume SciProcConfiguration.")
            return PathHandler(SciProcConfiguration(config_source=config))
        else:
            raise ValueError("No information to initialize PathHandler")

    def _setup_config(self, config):
        if isinstance(config, PreprocConfiguration) or isinstance(config, SciProcConfiguration):
            return config.config
        elif isinstance(config, str):
            warnings.warn("String path is deprecated. Assume SciProcConfiguration.")
            return SciProcConfiguration(config_source=config).config
        elif isinstance(config, ConfigurationInstance):
            return config
        else:
            raise ValueError("Invalid configuration object")

    def _setup_logger(self, logger, config):
        
        if isinstance(logger, Logger) and any(isinstance(handler, LockingFileHandler) for handler in logger.logger.handlers):
            logger.set_output_file(config.logging.file, overwrite=False)
            self._logger = logger
            return logger
        else:
            list_of_logger = logging.Logger.manager.loggerDict
            if config.name in list_of_logger.keys():
                tmp_logger = logging.getLogger(config.name)
                if any(isinstance(handler, LockingFileHandler) for handler in tmp_logger.handlers):
                    self._logger = tmp_logger
                    return tmp_logger

            tmp_logger = Logger(name=config.name)
            tmp_logger.set_output_file(config.logging.file, overwrite=False)
            tmp_logger.set_format(config.logging.format)
            self._logger = tmp_logger
            return tmp_logger

    def _setup_queue(self, queue):

        if isinstance(queue, QueueManager):
            queue.logger = self.logger
            return queue
        elif queue:
            return QueueManager(logger=self.logger)
        else:
            return None

    @classmethod
    @abstractmethod
    def from_list(self):
        pass

    @classmethod
    def from_text_file(cls, image):
        return cls.from_list([image])

    @classmethod
    def from_dir(cls, dir_path):
        image_list = glob.glob(f"{dir_path}/*.fits")
        return cls.from_list(image_list)

    # @classmethod
    # def from_text_file(cls, imagelist_file):
    #     input_images = inputlist_parser(imagelist_file)
    #     cls.from_list(input_images)
