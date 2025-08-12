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
    """
    Abstract base class for pipeline setup and configuration management.
    
    This class provides a unified interface for setting up pipeline components
    including path handling, configuration management, logging, and queue processing.
    It serves as the foundation for both preprocessing and science processing modules.
    
    Features:
    - Flexible configuration handling (file paths, config objects)
    - Automatic logger setup with file locking
    - Optional queue manager integration
    - Path handler initialization
    - Context manager support for cleanup
    
    Args:
        config: Configuration object, file path, or ConfigurationInstance
        logger: Custom logger instance (optional)
        queue: QueueManager instance or boolean to enable parallel processing
    
    Example:
        >>> setup = BaseSetup(
        ...     config="path/to/config.yml",
        ...     logger=custom_logger,
        ...     queue=True
        ... )
    """
    
    def __init__(
        self,
        config: Union[str, PreprocConfiguration, SciProcConfiguration] = None,
        logger: Logger = None,
        queue: Union[bool, Any] = False,
    ) -> None:
        """Initialize the pipeline setup module.

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
        """
        Get the configured logger instance.
        
        Returns:
            Logger: The configured logger with proper file handling
        """
        return self._setup_logger(self._logger, self.config)

    def _setup_path(self, config):
        """
        Initialize the path handler based on configuration.
        
        Args:
            config: Configuration object or file path
            
        Returns:
            PathHandler: Configured path handler instance
            
        Raises:
            ValueError: If no valid configuration information is provided
        """
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
        """
        Initialize the configuration object.
        
        Args:
            config: Configuration object or file path
            
        Returns:
            ConfigurationInstance: The configuration instance
            
        Raises:
            ValueError: If invalid configuration object is provided
        """
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
        """
        Initialize the logger with proper file handling.
        
        Creates or reuses a logger with LockingFileHandler to ensure
        thread-safe logging to files.
        
        Args:
            logger: Existing logger instance or None
            config: Configuration instance containing logging settings
            
        Returns:
            Logger: Configured logger instance
        """
        
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
        """
        Initialize the queue manager for parallel processing.
        
        Args:
            queue: QueueManager instance, boolean, or None
            
        Returns:
            QueueManager or None: Configured queue manager or None if disabled
        """
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
        """
        Abstract method for creating setup from a list of inputs.
        
        Must be implemented by subclasses to handle specific input types.
        """
        pass

    @classmethod
    def from_text_file(cls, image):
        """
        Create setup from a single text file.
        
        Args:
            image (str): Path to text file containing input list
            
        Returns:
            BaseSetup: Configured setup instance
        """
        return cls.from_list([image])

    @classmethod
    def from_dir(cls, dir_path):
        """
        Create setup from all FITS files in a directory.
        
        Args:
            dir_path (str): Directory path containing FITS files
            
        Returns:
            BaseSetup: Configured setup instance
        """
        image_list = glob.glob(f"{dir_path}/*.fits")
        return cls.from_list(image_list)

    # @classmethod
    # def from_text_file(cls, imagelist_file):
    #     input_images = inputlist_parser(imagelist_file)
    #     cls.from_list(input_images)
