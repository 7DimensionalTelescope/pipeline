import os
from pipeline.const import *


def dir_is_nonempty(path: str) -> bool:
    """Return True iff path exists, is a directory, and contains at least one entry."""
    try:
        with os.scandir(path) as it:
            return any(True for _ in it)  # stops at first entry
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return False


dirs_to_check_existence = [
    RAWDATA_DIR,
    FACTORY_DIR,
    MASTER_FRAME_DIR,
    PROCESSED_DIR,
    COADD_DIR,
    # TOO_DIR,
    # TOO_FACTORY_DIR,
    ASTRM_CUSTOM_REF_DIR,
    SCAMP_QUERY_DIR,
]

dirs_to_check_contents = list(REQUISITE_DIRS)


for dir in dirs_to_check_existence:
    if not os.path.isdir(dir) or not os.path.exists(dir):
        print(f"Directory {dir} does not exist or is not a directory.")
        exit(1)

for dir in dirs_to_check_contents:
    if not dir_is_nonempty(dir):
        print(f"Directory {dir} is empty, inaccessible, or does not exist.")
        exit(1)


print("Installation check passed")
