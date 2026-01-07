import os
import sys
from datetime import datetime
import glob
import time

from .. import __version__
from ..utils import clean_up_folder, flatten, time_diff_in_seconds
from ..path.path import PathHandler
from ..const import CalibType
from .base import BaseConfig


class DebugConfiguration(BaseConfig):
    def __init__(
        self,
        input: list[str] | str | dict = None,
        logger=None,
        write=True,
        overwrite=False,
        verbose=True,
        is_too=False,
        **kwargs
    ):
        super().__init__(input, logger, write, overwrite, verbose, is_too, **kwargs)
