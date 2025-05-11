import os
import fnmatch
from .const import RAWDATA_DIR, PROCESSED_DIR
from tqdm import tqdm

def query_observations(include_keywords, datatype="processed", exclude_keywords=None):
    """
    Recursively searches for .fits files in RAWDATA_DIR or PROCESSED_DATA.
    
    Files are returned if they:
    - contain at least one of the include_keywords (if provided), and
    - do not contain any of the exclude_keywords (if provided).

    Parameters:
    include_keywords (list of str): Keywords that must appear in the file path or name.
    exclude_keywords (list of str): Keywords that must not appear in the file path or name.
                                    Default is ["bias", "dark", "flat"].

    Returns:
    list: List of paths to matching FITS files.
    """
    if exclude_keywords is None:
        exclude_keywords = ["bias", "dark", "flat"]

    matching_files = []

    if datatype == "processed":
        DATA_DIR = PROCESSED_DIR
    elif datatype == "raw":
        DATA_DIR = RAWDATA_DIR

    for dirpath, _, filenames in os.walk(DATA_DIR):
        for filename in fnmatch.filter(filenames, "*.fits"):
            full_path = os.path.join(dirpath, filename)
            full_path_lower = full_path.lower()
            
            # Check for exclude keywords first
            if any(keyword.lower() in full_path_lower for keyword in exclude_keywords):
                continue
            
            # Check for include keywords, if provided
            if any(keyword.lower() not in full_path_lower for keyword in include_keywords):
                continue

            matching_files.append(full_path)
    
    return matching_files

