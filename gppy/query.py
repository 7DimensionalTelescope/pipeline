import os
from .const import RAWDATA_DIR, PROCESSED_DIR
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from . import const
import glob

def query_observations(include_keywords, exclude_keywords=None, DATA_DIR=const.RAWDATA_DIR, with_calib=True):
    
    default_exclude_keywords = ["test", "shift"] + const.CalibType

    include_keywords = list(np.atleast_1d(include_keywords))
    if exclude_keywords is not None:
        exclude_keywords = list(np.atleast_1d(exclude_keywords))
        exclude_keywords = exclude_keywords + default_exclude_keywords
    else:
        exclude_keywords = default_exclude_keywords

    flagging = lambda x: x.endswith(".fits") and not any(excl in x for excl in exclude_keywords)

    def search_obs(unit):
        result = []
        paths = []
        unit_path = os.path.join(DATA_DIR, unit)
        for dirpath, _, filenames in os.walk(unit_path):
            if "/tmp/" in dirpath:
                continue

            if len(filenames) == 0:
                continue

            flag_dir = all(keyword in dirpath for keyword in include_keywords)
            if flag_dir:
                result.extend(
                    [
                    os.path.join(dirpath, fname)
                    for fname in filenames 
                    if flagging(fname)
                    ]
                )
                continue

            matched_files = [fname for fname in filenames if flagging(fname)]
            if not matched_files:
                continue

            flag_files = [all(keyword in os.path.join(dirpath, fname) for keyword in include_keywords) for fname in matched_files]
            if any(flag_files):
                result.extend(
                    [
                        os.path.join(dirpath, fname)
                        for matched, fname in zip(flag_files, matched_files)
                        if matched and flagging(fname)
                    ]
                )
                paths.append(dirpath)

        calibs = []
        paths = list(set(paths))
        if with_calib:
            for path in paths:
                for calib in const.CalibType:
                    calibs.extend(glob.glob(os.path.join(path, f"*{calib}*.fits")))

        calibs = set(calibs)
        result.extend(calibs)
        return result

    units = []
    for kw in include_keywords:
        if re.match(r"(7DT\d\d)", kw):
            units.append(re.match(r"(7DT\d\d)", kw).group())
            include_keywords.remove(kw)

    if len(units) == 0:
        units = [f"7DT{i:02d}" for i in range(20)]

    output = []
    with ThreadPoolExecutor(max_workers=min(20, len(units))) as executor:
        futures = [executor.submit(search_obs, unit) for unit in units]
        for future in as_completed(futures):
            output.extend(future.result())

    return output




# def query_observations(include_keywords, datatype="processed", exclude_keywords=None, **kwargs):
#     """
#     Recursively searches for .fits files in RAWDATA_DIR or PROCESSED_DATA.

#     Files are returned if they:
#     - contain at least one of the include_keywords (if provided), and
#     - do not contain any of the exclude_keywords (if provided).

#     Parameters:
#     include_keywords (list of str): Keywords that must appear in the file path or name.
#     exclude_keywords (list of str): Keywords that must not appear in the file path or name.
#                                     Default is ["bias", "dark", "flat"].
#     **kwargs: Additional keyword arguments.

#     Returns:
#     list: List of paths to matching FITS files.
#     """
#     if exclude_keywords is None:
#         exclude_keywords = ["bias", "dark", "flat"]

#     matching_files = []

#     if kwargs.get("DATA_DIR"):
#         DATA_DIR = kwargs.get("DATA_DIR")
#     elif datatype == "processed":
#         DATA_DIR = PROCESSED_DIR
#     elif datatype == "raw":
#         DATA_DIR = RAWDATA_DIR
#     else:
#         raise ValueError("Invalid datatype. Must be 'processed' or 'raw'.")

#     include_keywords = np.atleast_1d(include_keywords)
#     exclude_keywords = np.atleast_1d(exclude_keywords)

#     for dirpath, _, filenames in os.walk(DATA_DIR):
#         for filename in fnmatch.filter(filenames, "*.fits"):
#             full_path = os.path.join(dirpath, filename)
#             full_path_lower = full_path.lower()

#             if any(keyword.lower() in full_path_lower for keyword in exclude_keywords):
#                 continue

#             if any(keyword.lower() not in full_path_lower for keyword in include_keywords):
#                 continue

#             matching_files.append(full_path)

#     return matching_files
