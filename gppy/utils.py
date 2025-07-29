import os
import re
import glob
import time
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from astropy.io import fits
from collections import Counter
from collections.abc import Iterable

from .const import FACTORY_DIR, RAWDATA_DIR, HEADER_KEY_MAP, ALL_GROUP_KEYS
import numpy as np


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
        print(print_str)  # log the elapsed time at INFO level

        if print_output:
            print(print_str, end="\n")  # log the elapsed time
        return elapsed_time  # in seconds


def update_padded_header(target_fits, header_new):
    """
    Update a FITS file's header with header_new (scamp or photometry output).
    header_new can be either astropy.io.fits.Header or dict.

    CAVEAT: This overwrites COMMENTs adjacent to the padding

    Args:
        target_fits (str): Path to the target FITS file to be updated
        header_new (dict or Header): Header object with info to be added

    Note:
        - Modifies the target FITS file in-place
        - Preserves existing non-COMMENT header entries
        - Appends or replaces header cards from the input header
    """

    with fits.open(target_fits, mode="update") as hdul:
        header = hdul[0].header
        cards = header.cards
        for i in range(len(cards) - 1, -1, -1):
            if cards[i][0] != "COMMENT":  # i is the last non-comment idx
                break

        # format new header for iteration
        if isinstance(header_new, fits.Header):
            cardpack = header_new.cards
        elif isinstance(header_new, dict):  # (key, value) or (key, (value, comment))
            cardpack = [
                (key, *value) if isinstance(value, tuple) else (key, value) for key, value in header_new.items()
            ]
        else:
            raise ValueError("Unsupported Header format for updating padded Header")

        # Expects (key, value, comment)
        for j, card in enumerate(cardpack):
            if i + j <= len(cards) - 1:
                del header[i + j]
                header.insert(i + j, card)
            else:
                header.append(card, end=True)


# def update_padded_header(target_fits, header_new):
#     """
#     Update a FITS file's header with header_new (scamp or photometry output).
#     header_new can be either astropy.io.fits.Header or dict.

#     This version will override any existing key/value pairs in-place,
#     and append truly new cards just before the trailing COMMENTs.

#     Args:
#         target_fits (str): Path to the target FITS file to be updated
#         header_new (dict or Header): Header object or mapping with info to be added

#     Note:
#         - Modifies the target FITS file in-place
#         - Preserves existing non-COMMENT header entries (except overridden ones)
#         - Appends only those cards whose keys did not exist before
#     """
#     with fits.open(target_fits, mode="update") as hdul:
#         header: fits.Header = hdul[0].header

#         # Find the index of the last non-COMMENT card
#         last_non_comment = None
#         for idx, card in enumerate(header.cards):
#             if card.keyword != "COMMENT":
#                 last_non_comment = idx

#         insert_pos = (last_non_comment + 1) if last_non_comment is not None else len(header.cards)

#         # Normalize header_new into a list of (key, value, comment) tuples
#         if isinstance(header_new, fits.Header):
#             new_cards = [(c.keyword, c.value, c.comment) for c in header_new.cards]
#         elif isinstance(header_new, dict):
#             new_cards = []
#             for key, val in header_new.items():
#                 if isinstance(val, tuple):
#                     new_cards.append((key, val[0], val[1]))
#                 else:
#                     new_cards.append((key, val, None))
#         else:
#             raise ValueError("Unsupported Header format for updating padded Header")

#         # Apply each new card: override if exists, otherwise append in padding
#         for key, value, comment in new_cards:
#             if key in header:  # override existing
#                 # header.set handles both update and insertion
#                 header.set(key, value, comment or header.comments.get(key))
#             else:
#                 # insert before COMMENTS
#                 header.insert(insert_pos, (key, value, comment), after=False)
#                 insert_pos += 1  # shift insertion point forward


def force_symlink(src, dst):
    """
    Remove the existing symlink `dst` if it exists, then create a new symlink.
    """
    try:
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
    except FileNotFoundError:
        pass

    os.symlink(src, dst)


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
        return header_to_dict(imhead_file)
    elif os.path.exists(filename):
        from astropy.io import fits

        return fits.getheader(swap_ext(filename, "fits"))
    else:
        if force_return:
            return {}
        raise FileNotFoundError(f"File not found: {filename}")


def header_to_dict(file_path):
    """
    Parse a FITS header text file and convert it into a dictionary.

    This function reads a text file containing a FITS (Flexible Image Transport System)
    header generated by the `imhead` command, extracts key-value pairs from each line,
    and stores them in a dictionary.

    The function handles the following cases:
    - String values enclosed in single quotes are stripped of quotes and whitespace.
    - Numerical values are converted to integers or floats when possible.
    - Boolean values (`T` and `F` in FITS format) are converted to Python `True` and `False`.
    - Comments after the `/` character are ignored.

    Args:
        file_path (str): Path to the text file containing the FITS header.

    Returns:
        dict: A dictionary containing the parsed header, where keys are the FITS
        header keywords and values are the corresponding parsed values.

    Example:
        Given a FITS header file with the following lines:
            SIMPLE  = T / file does conform to FITS standard
            BITPIX  = 8 / number of bits per data pixel
            NAXIS   = 0 / number of data axes
            EXTEND  = T / FITS dataset may contain extensions

        The function will return:
        {
            "SIMPLE": True,
            "BITPIX": 8,
            "NAXIS": 0,
            "EXTEND": True
        }
    """
    # Regular expression to match FITS header format
    fits_pattern = re.compile(r"(\S+)\s*=\s*(.+?)(?:\s*/\s*(.*))?$")

    fits_dict = {}
    # Read the FITS header from the text file
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            match = fits_pattern.match(line)
            if match:
                key, value, comment = match.groups()
                value = value.strip()

                # Handle string values enclosed in single quotes
                if value.startswith("'") and value.endswith("'"):
                    value = value.strip("'").strip()

                # Convert numerical values
                else:
                    try:
                        if "." in value:
                            value = float(value)  # Convert to float if it contains a decimal
                        else:
                            value = int(value)  # Convert to integer otherwise
                    except ValueError:
                        pass  # Leave as string if conversion fails

                # Convert boolean values (T/F in FITS format)
                if value == "T":
                    value = True
                elif value == "F":
                    value = False

                fits_dict[key] = value

    return fits_dict


def get_basename(file_path):
    return os.path.basename(file_path)
