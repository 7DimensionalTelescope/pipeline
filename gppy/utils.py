import os
import glob
import time
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from astropy.io import fits
from collections import Counter
from collections.abc import Iterable

from .const import FACTORY_DIR, ALL_GROUP_KEYS
import numpy as np

from .header import read_header_file


def atleast_1d(x):
    return [x] if not isinstance(x, list) else x


def flatten(nested, max_depth=None):
    """
    Flatten a nested list/tuple up to max_depth levels.

    Args:
        nested (list or tuple): input sequence
        max_depth (int or None): how many levels to flatten.
            None means “flatten completely”.

    Returns:
        list: flattened up to the requested depth
    """

    def _flatten(seq, depth):
        for el in seq:
            if isinstance(el, (list, tuple)) and (max_depth is None or depth < max_depth):
                yield from _flatten(el, depth + 1)
            else:
                yield el

    return list(_flatten(nested, 0))


def most_common_in_dict(counts: dict):
    best = None
    best_count = -1
    for key, cnt in counts.items():
        if cnt > best_count:
            best, best_count = key, cnt
    return best_count, best


def most_common_in_list(seq: list):
    if not seq:
        return None

    counts = {}
    for item in seq:
        counts[item] = counts.get(item, 0) + 1

    return most_common_in_dict(counts)


def clean_up_folder(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
        return
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)  # Remove file or symlink
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Remove directory and its contents
        except Exception as e:
            print(f"Failed to delete {item_path}: {e}")


def clean_up_factory():
    clean_up_folder(FACTORY_DIR)


def clean_up_sciproduct(root_dir: str | Path, suffixes=(".log", "_cat.fits", ".png")) -> None:
    root = Path(root_dir)
    for path in root.rglob("*"):
        if path.is_file() and any(path.name.endswith(s) for s in suffixes):
            try:
                path.unlink()
            except Exception as e:
                raise RuntimeError(f"Failed to delete {path}: {e}") from e


def lapse(explanation="elapsed", print_output=True):
    """
    Measure and report elapsed time using a global checkpoint.

    A utility function for performance tracking and logging elapsed time
    between function calls. It supports various time unit representations
    and optional console output.

    Args:
        explanation (str, optional): Description for the elapsed time report.
            Defaults to "elapsed".
        print_output (bool, optional): Whether to print the time report.
            Defaults to True.

    Returns:
        float: Elapsed time in seconds

    Usage:
        >>> lapse("Start")  # Initializes the timer
        >>> # Do some work
        >>> lapse("Task completed")  # Prints elapsed time
    """
    from timeit import default_timer as timer

    global _dhutil_lapse_checkpoint  # Global Checkpoint
    # Initialize if not yet defined
    if "_dhutil_lapse_checkpoint" not in globals():
        _dhutil_lapse_checkpoint = None

    current_time = timer()

    if _dhutil_lapse_checkpoint is None:  # Initialize if it's the first call
        _dhutil_lapse_checkpoint = current_time
    else:
        elapsed_time = current_time - _dhutil_lapse_checkpoint

        if elapsed_time < 60:
            dt, unit = elapsed_time, "seconds"
        elif elapsed_time > 3600:
            dt, unit = elapsed_time / 3600, "hours"
        else:
            dt, unit = elapsed_time / 60, "minutes"

        _dhutil_lapse_checkpoint = current_time  # Update the checkpoint

        print_str = f"{dt:.3f} {unit} {explanation}"
        # print(print_str)  # log the elapsed time at INFO level

        if print_output:
            print(print_str, end="\n")  # log the elapsed time
        return elapsed_time  # in seconds


def force_symlink(src, dst):
    """
    Remove the existing symlink `dst` if it exists, then create a new symlink.
    """
    try:
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
    except FileNotFoundError:
        pass

    # print(f"current working directory: {os.getcwd()}")
    # print(f"symlinking {src} to {dst}")
    os.symlink(src, dst)
    # os.symlink(os.path.realpath(src), dst)
    # os.system(f"ln -s {src} {dst}")


def time_diff_in_seconds(datetime1, datetime2=None, return_float=False):
    if datetime2 is None:
        datetime2 = time.time()
    if isinstance(datetime1, datetime):
        datetime1 = datetime1.timestamp()
    if isinstance(datetime2, datetime):
        datetime2 = datetime2.timestamp()
    time_diff = datetime2 - datetime1

    if return_float:
        return abs(time_diff)
    else:
        return f"{abs(time_diff):.2f}"


def swap_ext(file_path: str | list[str], new_ext: str) -> str:
    """
    Swap the file extension of a given file path.

    Args:
        file_path (str): The original file path.
        new_ext (str): The new extension (with or without a leading dot).

    Returns:
        str: The file path with the swapped extension.
    """
    if isinstance(file_path, list):
        return [swap_ext(f, new_ext) for f in file_path]
    base, _ = os.path.splitext(file_path)
    new_ext = new_ext if new_ext.startswith(".") else f".{new_ext}"
    return base + new_ext


def add_suffix(filename: str | list[str], suffix):
    """
    Add a suffix to the filename before the extension.

    Args:
        filename (str): The original filename.
        suffix (str): The suffix to add.

    Returns:
        str: The modified filename with the suffix added.
    """
    if isinstance(filename, list):
        return [add_suffix(f, suffix) for f in filename]
    base, ext = os.path.splitext(filename)
    suffix = suffix if suffix.startswith("_") else f"_{suffix}"
    return f"{base}{suffix}{ext}"


def equal_on_keys(d1: dict, d2: dict, keys: list):
    # return all(d1[k] == d2[k] for k in keys)
    return all(d1.get(k) == d2.get(k) for k in keys)  # None if key missing


def collapse(seq: list | dict[list], keys=ALL_GROUP_KEYS, raise_error=False, force=False):
    """
    If seq is non-empty and every element equals the first one,
    return the first element; else return seq unchanged.
    """
    if isinstance(seq, list):
        if seq:
            first = seq[0]
        else:
            # raise ValueError("Uncollapsible: input is empty list")
            return seq  # just return empty list
    # dict[str, list]
    elif isinstance(seq, dict):
        return {k: collapse(v) if isinstance(v, list) else v for k, v in seq.items()}
    else:
        return seq

    # list[dict]
    if isinstance(first, dict):
        common_keys = [k for k in keys if all(k in d for d in seq)]
        if common_keys and all(d[k] == first[k] for d in seq for k in common_keys):
            return first
    # list[str, int, float, ...]
    else:
        if all(x == first for x in seq) or force:
            return first

    if raise_error:
        raise ValueError(f"Uncollapsible: input is not homogeneous: {seq}")
    else:
        return seq


def get_header(filename: str | Path, force_return=False) -> dict | fits.Header:
    """
    Get the header of a FITS file.

    Args:
        filename (str | Path): Path to the FITS file or a .head file

    Returns:
        dict | fits.Header: Header of the FITS file
    """
    filename = str(filename)
    imhead_file = swap_ext(filename, "head")

    if os.path.exists(imhead_file):
        # Read the header from the text file
        return read_header_file(imhead_file)
    elif os.path.exists(filename):
        from astropy.io import fits

        return fits.getheader(swap_ext(filename, "fits"))
    else:
        if force_return:
            return {}
        raise FileNotFoundError(f"File not found: {filename}")


def get_header_key(header_file, key, default=None):
    header = get_header(header_file)
    if hasattr(header, key):
        return header[key]
    else:
        return default


def get_basename(file_path):
    return os.path.basename(file_path)


def read_text_file(file, start_row=0):
    """
    Read a text file and return a list of lines.
    """
    with open(file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f][start_row:]
        return lines
