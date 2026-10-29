from .environ import *
from .observation import *
from .products import *
from .run import *
from .system_resources import *

# set umask 0022 -> 0002
import os
import sys
from ..utils.umask import set_umask

# TODO: This can leak and cause unwanted behavior in threading environments.
# Find ways to separate this when the pipeline is used as a library
UMASK = os.environ.get("UMASK", None)
_PIPELINE_UMASK_SENTINEL_ATTR = "_pipeline_umask_applied"
if UMASK and not getattr(sys, _PIPELINE_UMASK_SENTINEL_ATTR, False):
    set_umask(UMASK)
    setattr(sys, _PIPELINE_UMASK_SENTINEL_ATTR, True)  # cache to run only once


from . import observation as _observation


def __getattr__(name: str):
    if name in _observation._FILTER_CONSTANTS:
        return getattr(_observation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
