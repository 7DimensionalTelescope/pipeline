from __future__ import annotations
import warnings

from .version import __version__
from .config_integrity import verify_config_hashes

"""
gpPy: Automated Pipeline for Astronomical Image Processing

gpPy is a multi-threaded pipeline for processing optical and near-infrared (NIR) images from
IMSNG/GECKO and 7DT facilities. It handles data reduction, astrometric calibration, stacking, photometric
calibration, image subtraction, and automated transient detection using GPU and CPU multiprocessing.

- Developed by Dr. Gregory Peak (2018)
- First public release: September 1, 2023
- Major renovation: February 2025 (7DT pipeline integration)

Current maintainers: Donghwan Hyun, WonHyeong Lee
Contributors: Dr. Gregory Peak, Dr. Donggeun Tak

Contact: gregorypaek94_at_g_mail
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
