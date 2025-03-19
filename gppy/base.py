
from .config import Configuration
from .services.queue import QueueManager
from .logger import Logger
from typing import Any, Union
from abc import ABC, abstractmethod
import glob

class BaseSetup(ABC):
    def __init__(
        self,
        config: Union[str, Any] = None,
        logger: Any = None,
        queue: Union[bool, QueueManager] = False,
    ) -> None:
        """Initialize the astrometry module.

        Args:
            config: Configuration object or path to config file
            logger: Custom logger instance (optional)
            queue: QueueManager instance or boolean to enable parallel processing
        """
        # Load Configuration
        if isinstance(config, str):  # In case of File Path
            self.config = Configuration(config_source=config).config
        elif hasattr(config, "config"):
            self.config = config.config  # for easy access to config
        else:
            self.config = config

        # Setup log
        self.logger = self._setup_logger(logger, config)

        # Setup queue
        self.queue = self._setup_queue(queue)
    
    def _setup_logger(self, logger, config):
        if isinstance(logger, Logger):
            return logger
        elif hasattr(config, "logger") and config.logger is not None:
            return config.logger
        else:
            return Logger(name="7DT pipeline logger", slack_channel="pipeline_report")

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
    def from_file(cls, image):
        return cls.from_list([image])

    @classmethod
    def from_dir(cls, dir_path):
        image_list = glob.glob(f"{dir_path}/*.fits")
        return cls.from_list(image_list)
