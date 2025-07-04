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


def unique(seq, *, return_counts=False, return_index=False, return_inverse=False):
    """
    Pure-Python version of np.unique for 1D sequences.

    Parameters
    ----------
    seq : iterable
        Input sequence.
    return_counts : bool, optional
        If True, also return the counts of each unique element.
    return_index : bool, optional
        If True, also return the index of the first occurrence of each unique element in the original sequence.
    return_inverse : bool, optional
        If True, also return an array that can be used to reconstruct the original sequence from the unique values.

    Returns
    -------
    unique_vals : list
        Sorted list of unique elements.
    counts : list, optional
        List of counts corresponding to each unique element.
    indices : list, optional
        List of first‐occurrence indices for each unique element.
    inverse : list, optional
        List of indices such that unique_vals[inverse[i]] == seq[i].
    """
    # Record first‐occurrence index
    first_idx = {}
    for i, v in enumerate(seq):
        if v not in first_idx:
            first_idx[v] = i

    # Count occurrences
    cnt = Counter(seq)

    # Unique sorted values
    uniques = sorted(cnt)

    out = [uniques]

    if return_counts:
        out.append([cnt[v] for v in uniques])

    if return_index:
        out.append([first_idx[v] for v in uniques])

    if return_inverse:
        # Map each value to its position in the uniques list
        pos = {v: i for i, v in enumerate(uniques)}
        out.append([pos[v] for v in seq])

    # Unpack appropriately
    if len(out) == 1:
        return out[0]
    return tuple(out)


def atleast_1d(x):
    return [x] if not isinstance(x, list) else x


# def flatten(seq):
#     """
#     Recursively flatten any nested lists/tuples into a single flat list.
#     Strings and bytes are treated as atomic.
#     """
#     flat = []
#     for item in seq:
#         if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
#             flat.extend(flatten(item))
#         else:
#             flat.append(item)
#     return flat


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


def clean_up_factory():
    clean_up_folder(FACTORY_DIR)


def clean_up_folder(path):
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


def clean_up_sciproduct(root_dir: str | Path, suffixes=(".log", "_cat.fits", ".png")) -> None:
    root = Path(root_dir)
    for path in root.rglob("*"):
        if path.is_file() and any(path.name.endswith(s) for s in suffixes):
            try:
                path.unlink()
            except Exception as e:
                raise RuntimeError(f"Failed to delete {path}: {e}") from e


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


def check_params(img):
    """makes obs_params, which will be deprecated in the unified MFG-preproc scheme"""
    try:
        params = parse_key_params_from_filename(img)[0]
    except:
        try:
            params = parse_key_params_from_header(img)[0]
        except:
            raise ValueError("No parameters found in the image file names or headers.")
    if not params:
        raise ValueError("No parameters found in the image file names or headers.")
    return params


def parse_key_params_from_filename(img):
    if isinstance(img, str):
        path = Path(img)
    elif isinstance(img, Path):
        path = img
    else:
        raise TypeError("Input must be a string or Path.")

    # e.g., 7DT11_20250102_050704_T00223_m425_1x1_100.0s_0001.fits
    pattern = r"^(7DT\d{2})_(\d{8}_\d{6})_([a-zA-Z0-9]+)_([a-zA-Z0-9]+)_(\dx\d)_(\d+\.?\d*)s_([0-9]+)"
    match = re.match(pattern, path.stem)

    if match:
        units, datetime_string, target, filt, formatted_n_binning, exposure, image_number = match.groups()
        date = subtract_half_day(datetime_string)
    else:
        pattern = r"([a-zA-Z0-9]+)_([a-zA-Z0-9]+)_(7DT\d{2})_(\d+\.?\d*)s_(\d{8}_\d{6})"
        match = re.match(pattern, path.stem)
        if match:
            target, filt, units, exposure, datetime_string = match.groups()
            date = subtract_half_day(datetime_string)

    gain = re.findall("(gain[0-9]+)", path.abspath())

    if gain:
        gain = gain[-1]
    else:
        gain = None

    info = {
        "nightdate": date,
        "obstime": datetime_string,
        "filter": filt,
        "obj": target,
        "unit": units,
        "exposure": exposure,
        "n_binning": int(formatted_n_binning[0]),
        "gain": gain,
    }

    # deprecated key support
    info["date"] = info["nightdate"]
    info["datetime"] = info["obstime"]

    file_type = "master_image" if any(s in str(path.stem) for s in ["BIAS", "DARK", "FLAT"]) else "sci_image"

    return info, file_type


def parse_key_params_from_header(filename: str | Path) -> None:
    """
    Extract target information from a FITS filename.

    Args:
        file_path (Path): Path to the FITS file

    Returns:
        tuple: Target name and filter, or None if parsing fails
    """
    filename = str(filename)  # in case filename is pathlib Path
    info = {}
    header = fits.getheader(filename)

    for attr, key in HEADER_KEY_MAP.items():
        if key == "DATE-LOC":
            header_date = datetime.fromisoformat(header[key])
            adjusted_date = header_date - timedelta(hours=12)
            final_date = adjusted_date.date()
            info[attr] = final_date.isoformat()
        else:
            info[attr] = header[key]

    info["nightdate"] = subtract_half_day(to_datetime_string(info["obstime"]))

    file_type = "master_image" if any(s in filename for s in ["BIAS", "DARK", "FLAT"]) else "sci_image"

    return info, file_type


def subtract_half_day(timestr: str) -> str:
    if len(timestr) == 8:
        dt = datetime.strptime(timestr, "%Y%m%d")
    else:
        dt = datetime.strptime(timestr, "%Y%m%d_%H%M%S")
    new_dt = dt - timedelta(hours=15)  # -15h for winter, not -12h
    return new_dt.strftime("%Y-%m-%d")


def to_datetime_string(datetime_str, date_only=False):
    """
    Args:
        datetime_str (str): ISO-formatted datetime string (e.g., '2025-02-07T16:44:41+09:00')
        date_only (bool, optional): If True, returns only the date. Defaults to False.

    Returns:
        str: Formatted datetime string
            - With date_only=False: 'YYYYMMDD_HHMMSS' (e.g., '20250207_164441')
            - With date_only=True: 'YYYYMMDD' (e.g., '20250207')

    Example:
        >>> to_datetime_string('2025-02-07T16:44:41Z')
        '20250207_164441'
        >>> to_datetime_string('2025-02-07T16:44:41Z', date_only=True)
        '20250207'
    """
    dt = datetime.fromisoformat(datetime_str.replace("Z", "+00:00"))

    # Format to desired output
    if date_only:
        return dt.strftime("%Y%m%d")
    else:
        return dt.strftime("%Y%m%d_%H%M%S")


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


def get_camera(header):
    """
    Determine the camera type based on image dimensions.

    Identifies the camera model by examining the number of pixels in the first axis.
    Supports two camera types: C3 and C5.
    Returns UnKnownCam if the file does not exist or the header is missing NAXIS1.

    Support for the overscan area of C5 is to be added.

    Args:
        header (dict or str): Either a header dictionary or a path to a .head file

    Returns:
        str: Camera type ('C3', 'C5', or 'UnnknownCam')

    Example:
        >>> get_camera({'NAXIS1': 9576, 'NAXIS2': 6388})
        'C3'
        >>> get_camera('/path/to/header.head')
        'C5'
    """
    if isinstance(header, list):
        return [get_camera(s) for s in header]
    elif type(header) == dict or isinstance(header, fits.Header):
        pass
    elif isinstance(header, (str, Path)):
        header = get_header(header, force_return=True)
    else:
        raise TypeError("Input of get_camera must be a dictionary, fits.Header, or a file path.")

    # if header["NAXIS1"] == 9576:  # NAXIS2 6388
    #     return "C3"
    # elif header["NAXIS1"] == 14208:  # NAXIS2 10656
    #     return "C5"
    if header and "NAXIS1" in header:
        if 9576 % header["NAXIS1"] == 0:  # NAXIS2 6388
            return "C3"
        elif 14208 % header["NAXIS1"] == 0:  # NAXIS2 10656
            return "C5"
        else:
            return "UnknownCam"
    else:
        return "UnknownCam"  # None  # None makes the length masterframe_basename different.


def get_gain(fpath):
    key = HEADER_KEY_MAP["gain"]

    def parse_from_path(path: str) -> int:
        m = re.search(r"gain(\d+)", path)
        if not m:
            return None
        else:
            return int(m.group(1))

    return parse_from_path(fpath) or get_header(fpath, force_return=True).get(key, None)


def find_raw_path(unit, nightdate, n_binning, gain):
    """
    Locate the raw data directory for a specific observation.

    Searches for raw data directories with increasing specificity:
    1. By unit and date
    2. By unit, date, and gain
    3. By unit, date, binning, and gain

    Args:
        unit (str): Observation unit identifier
        date (str): Observation date
        n_binning (int): Pixel binning factor
        gain (float): Detector gain setting

    Returns:
        str: Path to the raw data directory

    Raises:
        ValueError: If no matching data directory is found
    """
    from .const import RAWDATA_DIR

    raw_data_folder = glob.glob(f"{RAWDATA_DIR}/{unit}/{nightdate}*")

    if len(raw_data_folder) > 1:
        raw_data_folder = glob.glob(f"{RAWDATA_DIR}/{unit}/{nightdate}*_gain{gain}*")
        if len(raw_data_folder) > 1:
            raw_data_folder = glob.glob(f"{RAWDATA_DIR}/{unit}/{nightdate}_{n_binning}x{n_binning}_gain{gain}*")

    elif len(raw_data_folder) == 0:
        raise ValueError("No data folder found")

    return raw_data_folder[0]


def parse_exptime(filename, return_type="float"):
    """
    Extract exposure time from a filename.

    Args:
        filename (str): Filename containing exposure time
        return_type (type, optional): Return type for exposure time. Defaults to float.

    Returns:
        float or int: Exposure time extracted from the filename

    Example:
        >>> parse_exptime('calib_7DT11_T00139_20250102_014643_m425_100s.fits')
        100.0
        >>> parse_exptime(calib_7DT11_T00139_20250102_014643_m425_100s.fits', return_type=int)
        100
    """
    match = re.search(r"_(\d+\.?\d*)s", filename)
    if match:
        exptime = float(match.group(1))
        return int(exptime) if return_type == int else exptime
    return None  # Return None if no match is found


def define_output_dir(date, n_binning, gain, obj=None, unit=None, filt=None):
    """
    *deprecated*

    Generate a standardized output directory name.

    Args:
        date (str): Observation date
        n_binning (int): Pixel binning factor
        gain (float): Detector gain setting

    Returns:
        str: Formatted output directory name

    Example:
        >>> define_output_dir('20250207', 2, 1.0)
        '20250207_2x2_gain1.0'
    """
    if obj:
        if unit:
            if filt:
                return f"{date}_{n_binning}x{n_binning}_gain{gain}/{obj}/{unit}/{filt}"
            else:
                return f"{date}_{n_binning}x{n_binning}_gain{gain}/{obj}/{unit}"
        else:
            return f"{date}_{n_binning}x{n_binning}_gain{gain}/{obj}"
    else:
        return f"{date}_{n_binning}x{n_binning}_gain{gain}"


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


def read_scamp_header(file):
    """
    Read a SCAMP output HEAD file, normalizing unicode and correcting WCS types.

    Args:
        file (str): Path to the header file

    Returns:
        fits.Header: Processed and cleaned FITS header with corrected WCS types

    Note:
        - Removes non-ASCII characters
        - Converts WCS projection type from TAN to TPV
    """
    import unicodedata

    with open(file, "r", encoding="utf-8") as f:
        content = f.read()

    # Clean non-ASCII characters
    cleaned_string = unicodedata.normalize("NFKD", content).encode("ascii", "ignore").decode("ascii")

    # Correct CTYPE (TAN --> TPV)
    hdr = fits.Header.fromstring(cleaned_string, sep="\n")
    hdr["CTYPE1"] = ("RA---TPV", "WCS projection type for this axis")
    hdr["CTYPE2"] = ("DEC--TPV", "WCS projection type for this axis")
    return hdr


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


def get_derived_product_path(image, subdir="catalogs", ext=".cat.fits"):
    """
    Deprecated
    Without kwargs, returns catalog path
    """
    input_file_base = os.path.basename(image)
    output_file = os.path.join(os.path.dirname(image), subdir, swap_ext(input_file_base, ext))
    return output_file


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


def check_obs_file(params):
    for key, v in params.items():
        if v is None:
            params[key] = "*"

    path = RAWDATA_DIR + f'/{params["unit"]}/{params["date"]}_gain{params["gain"]}/'
    filename = (
        f'{params["unit"]}_*_{params["obj"]}_{params["filter"]}_{params["n_binning"]}x{params["n_binning"]}*.fits'
    )
    files = glob.glob(path + filename)
    if len(files) == 0:
        return False
    else:
        return files


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


def parse_list_file(imagelist_file):
    if os.path.exists(imagelist_file):
        print(f"{imagelist_file} found!")
    else:
        print(f"Not Found {imagelist_file}!")
    from astropy.table import Table

    input_table = Table.read(imagelist_file, format="ascii")
    # input_table = Table.read(imagelist_file_to_stack, format="ascii.commented_header")
    files = [f for f in input_table["file"].data]
    return files


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


def get_basename(file_path):
    return os.path.basename(file_path)


def update_header_by_overwriting(filename, header):
    data = fits.getdata(filename)
    fits.writeto(filename, data, header=header, overwrite=True)

def write_header_into_file(filename, header):
    if filename.endswith(".fits"):
        filename = filename.replace(".fits", ".header")
        
    with open(filename, "w") as f:
        f.write(header.tostring(sep='\n')) 