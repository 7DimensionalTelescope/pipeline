from typing import Any, Union
from abc import ABC, abstractmethod
import glob
from ..config import Configuration, ConfigurationInstance
from ..base.path import PathHandler
from .logger import Logger
from .queue import QueueManager


class BaseSetup(ABC):
    def __init__(
        self,
        config: Union[str, Any] = None,
        logger: Any = None,
        queue: Union[bool, Any] = False,
    ) -> None:
        """Initialize the astrometry module.

        Args:
            config: Configuration object or path to config file
            logger: Custom logger instance (optional)
            queue: QueueManager instance or boolean to enable parallel processing
        """
        self.path = self._setup_path(config)

        # Setup Configuration
        self.config = self._setup_config(config)

        # Setup log
        self.logger = self._setup_logger(logger, config)

        # Setup queue
        self.queue = self._setup_queue(queue)

    def _setup_path(self, config):
        if isinstance(config, Configuration):
            return config.path
        elif isinstance(config, str):
            return PathHandler(Configuration(config_source=config))
        else:
            raise ValueError("No information to initialize PathHandler")

    def _setup_config(self, config):

        if isinstance(config, Configuration):
            return config.config
        elif isinstance(config, str):
            return Configuration(config_source=config).config
        elif isinstance(config, ConfigurationInstance):
            return config
        else:
            raise ValueError("Invalid configuration object")

    def _setup_logger(self, logger, config):

        if isinstance(logger, Logger):
            return logger
        elif hasattr(config, "logger") and isinstance(config.logger, Logger):
            return config.logger
        else:
            return Logger(name=config.config.name, slack_channel="pipeline_report")

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

    @classmethod
    def from_text_file(cls, imagelist_file):
        input_images = inputlist_parser(imagelist_file)
        cls.from_list(input_images)

    def flagging(self):
        setattr(self.config.flag, self._flag_name, True)
