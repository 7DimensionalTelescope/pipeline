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


def clean_up_sciproduct(root_dir: str | Path, suffixes=(".log", "_cat.fits", "jpg", ".png")) -> None:
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
    stem, _ = os.path.splitext(file_path)
    new_ext = new_ext if new_ext.startswith(".") else f".{new_ext}"
    return stem + new_ext


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
    stem, ext = os.path.splitext(filename)
    suffix = suffix if suffix.startswith("_") else f"_{suffix}"
    return f"{stem}{suffix}{ext}"


def drop_suffix(filename: str | list[str], suffix: str | None = None):
    """
    Remove a suffix from the filename before the extension.
    If no suffix is provided, remove the last underscore section.

    Args:
        filename (str | list[str]): The original filename(s).
        suffix (str | None): The suffix to remove (without extension).
                             If None, drops the last '_suffix'.

    Returns:
        str | list[str]: The modified filename(s).
    """
    if isinstance(filename, list):
        return [drop_suffix(f, suffix) for f in filename]

    stem, ext = os.path.splitext(filename)

    if suffix is None:
        if "_" in stem:
            stem = stem.rsplit("_", 1)[0]
    else:
        suffix = suffix if suffix.startswith("_") else f"_{suffix}"
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]

    return f"{stem}{ext}"


def unique_filename(fpath: str):
    if not os.path.exists(fpath):
        return fpath
    else:
        idx = 0
        while True:
            idx += 1
            fpath_new = add_suffix(fpath, f"{idx}")
            if not os.path.exists(fpath_new):
                break
        return fpath_new


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


# def remove_ansi_escape_sequences(command: str) -> str:
#     """uses sed to remove ansi escape sequences from a command"""
#     return os.popen(f"sed 's/\x1b\[[0-9;]*m//g'").read()


# def ansi_clean(cmd: str, use_bash_ansi_c: bool = True) -> str:
#     """
#     Wrap a shell command so its stdout+stderr are filtered through sed to strip ANSI escape codes.

#     - cmd: full shell command (may or may not already have redirection like '>> file 2>&1')
#     - clean: if False, returns cmd unchanged
#     - use_bash_ansi_c: if True, uses Bash ANSI-C quoting ($'...') for portable ESC.
#                        If your environment isn't bash, set this to False.

#     Returns: modified command string.
#     """
#     import re

#     # sed expression: remove ESC [ ... letters (cursor/formatting controls)
#     if use_bash_ansi_c:
#         # Requires bash (ANSI-C quoting for \x1B).
#         sed_expr = r"$'s/\x1B\[[0-9;]*[[:alpha:]]//g'"
#     else:
#         # Works on many GNU/BSD seds that accept \033 inside single quotes.
#         sed_expr = r"'s/\033\[[0-9;]*[[:alpha:]]//g'"

#     sed_pipe = f"sed -E {sed_expr}"

#     # If the command already redirects to a file, we want:
#     #   <left> 2>&1 | sed ... >> <file>
#     # i.e., move the 2>&1 before the pipe and keep the same target file.
#     m = re.search(r"(.*?)(\s*)(>>?|)(\s*)([^>|]*?)(\s*)(2>&1)?\s*$", cmd)
#     # This regex tries to capture the final redirection if present:
#     # 1: left part (command proper)
#     # 3: redir operator '' or '>' or '>>'
#     # 5: redir target (filename) (may contain spaces; best-effort)
#     # 7: '2>&1' if present at the end

#     if m and m.group(3):  # has '>' or '>>'
#         left = m.group(1).rstrip()
#         redir = m.group(3)
#         target = m.group(5).strip()
#         # Ensure we don't duplicate an empty filename
#         target_part = f" {target}" if target else ""
#         return f"{left} 2>&1 | {sed_pipe} {redir}{target_part}".strip()
#     else:
#         # No explicit file redirection at the end; just pipe and keep existing behavior.
#         # If the original cmd already had its own inner pipes, this simply appends our filter.
#         # Ensure stderr joins stdout so both are cleaned.
#         return f"{cmd} 2>&1 | {sed_pipe}"


def ansi_clean(cmd: str) -> str:
    """
    Wrap a shell command so its stdout+stderr are filtered through sed to strip ANSI escape codes.

    - cmd: full shell command (may or may not already have redirection like '>> file 2>&1')

    Returns: modified command string that is portable under /bin/sh (no bash features).
    """
    import re

    # Portable ESC declaration (POSIX sh): put this in front of the command
    esc_decl = "ESC=$(printf '\\033'); "

    # Portable, locale-stable sed filter (ERE via -E; works on GNU & BSD sed)
    # Use double quotes so ${ESC} expands; escape the '[' inside the Python string.
    sed_pipe = 'LC_ALL=C sed -E "s/${ESC}\\[[0-9;]*[[:alpha:]]//g"'

    # If the command already redirects to a file at the end, restructure to:
    #   <left> 2>&1 | sed ... >> <file>
    m = re.search(r"(.*?)(\s*)(>>?|)(\s*)([^>|]*?)(\s*)(2>&1)?\s*$", cmd)
    # Groups:
    # 1: left side (the command proper)
    # 3: redir operator '' or '>' or '>>'
    # 5: redir target (filename) (best-effort; may be empty)
    # 7: trailing '2>&1' if present

    if m and m.group(3):  # has '>' or '>>'
        left = m.group(1).rstrip()
        redir = m.group(3)
        target = (m.group(5) or "").strip()
        target_part = f" {target}" if target else ""
        # Join stderr before piping so both streams get cleaned
        return f"{esc_decl}{left} 2>&1 | {sed_pipe} {redir}{target_part}".strip()
    else:
        # No terminal redirection: pipe cleaned output to stdout
        return f"{esc_decl}{cmd} 2>&1 | {sed_pipe}"


def lupton_asinh(img: np.array, sky: float, noise: float, hi_clip=None, k_soft=3.0, vmin_sigma=-1.0):
    """
    img: raw image
    sky, noise: from estimate_sky_noise_annulus
    hi_clip: value to clip the top end (e.g. 99.7th percentile). If None, compute.
    k_soft: softening in σ; smaller -> more compression of bright core
    vmin_sigma: map sky + vmin_sigma*noise to display 0
    """
    if hi_clip is None:
        hi_clip = np.percentile(img, 99.7)

    soft = k_soft * noise  # the “Q*σ” softening scale
    # shift by sky and clamp low values a bit below sky
    shifted = img - sky
    shifted = np.maximum(shifted, vmin_sigma * noise)
    shifted = np.minimum(shifted, hi_clip - sky)

    # Lupton-style scaling to [0,1]
    num = np.arcsinh(shifted / soft)
    den = np.arcsinh((hi_clip - sky) / soft)
    scaled = num / den
    return np.clip(scaled, 0, 1)
