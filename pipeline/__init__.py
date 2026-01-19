from __future__ import annotations
import warnings

from .version import __version__
from .utils.config_integrity import verify_config_hashes

"""
Py7DT: Automated Pipeline for Astronomical Image Processing

Py7DT is a modern astronomical data reduction pipeline for optical images 
from the 7-Dimensional Telescope (7DT). 
It handles data reduction, astrometric calibration, stacking, photometric
calibration, image subtraction, and automated transient detection.

- inherits from gpPy-GPU developed by Dr. Gregory S.H. Paek (2023)

Core Developers: Donghwan Hyun, Dr. Donggeun Tak
"""

__package__ = "pipeline"

# Ignore common warnings that are not harmful
warnings.filterwarnings("ignore", message=".*datfix.*")
warnings.filterwarnings("ignore", message=".*pmsafe.*")
warnings.filterwarnings("ignore", message=".*partition.*")


# config version check


def _run_config_check_once() -> None:
    """Run the config hash check exactly once per process."""
    # Using a module-level flag to ensure this only runs once.
    global _CONFIG_CHECK_DONE
    if _CONFIG_CHECK_DONE:
        return

    verify_config_hashes()
    _CONFIG_CHECK_DONE = True


# initialize the guard flag
_CONFIG_CHECK_DONE = False

# run once on import time
_run_config_check_once()
