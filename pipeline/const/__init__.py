from .environ import *
from .observation import *
from .system_resources import *
from .errors import *

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
