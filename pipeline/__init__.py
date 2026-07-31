from __future__ import annotations
import os

# Thread-pool caps. Must be set before numpy/numba load: OpenBLAS reads its count at
# library load time, numba at first import. Uncapped, every process starts 64 BLAS
# threads (6.7 -> 0.17 CPU-s per import once capped) and numba runs prange on all 128
# cores, which oversubscribes at DEFAULT_MAX_WORKERS concurrency. The pipeline does no
# large linear algebra, so BLAS pools are pure overhead; numba prange kernels saturate
# near 8. setdefault: an explicit env var from systemd or the shell still wins.
for _threads_var, _threads_default in (
    ("OMP_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("VECLIB_MAXIMUM_THREADS", "1"),
    ("NUMBA_NUM_THREADS", "8"),
):
    os.environ.setdefault(_threads_var, _threads_default)

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
