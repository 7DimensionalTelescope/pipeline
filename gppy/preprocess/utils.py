import os
import time
import gc
import cupy as cp
from astropy.io import fits
from contextlib import contextmanager
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import uuid


def sigma_clipped_stats_cupy(cp_data, sigma=3, maxiters=5, minmax=False):
    """
    Approximate sigma-clipping using CuPy.
    Computes mean, median, and std after iteratively removing outliers
    beyond 'sigma' standard deviations from the median.

    Parameters
    ----------
    cp_data : cupy.ndarray
        Flattened CuPy array of image pixel values.
    sigma : float
        Clipping threshold in terms of standard deviations.
    maxiters : int
        Maximum number of clipping iterations.

    Returns
    -------
    mean_val : float
        Mean of the clipped data (as a GPU float).
    median_val : float
        Median of the clipped data (as a GPU float).
    std_val : float
        Standard deviation of the clipped data (as a GPU float).
    """
    # Flatten to 1D for global clipping
    cp_data = cp_data.ravel()

    for _ in range(maxiters):
        median_val = cp.median(cp_data)
        std_val = cp.std(cp_data)
        # Keep only pixels within +/- sigma * std of the median
        mask = cp.abs(cp_data - median_val) < (sigma * std_val)
        cp_data = cp_data[mask]

    # Final statistics on the clipped data
    mean_val = cp.mean(cp_data)
    median_val = cp.median(cp_data)
    std_val = cp.std(cp_data)

    # Convert results back to Python floats on the CPU
    # return float(mean_val), float(median_val), float(std_val)
    if minmax:
        return mean_val, median_val, std_val, cp_data.min(), cp_data.max()
    return mean_val, median_val, std_val


def write_link(fpath, content):
    """path to the link, and the path link is pointing"""
    with open(fpath, "w") as file:
        file.write(content)


@contextmanager
def load_data_gpu(fpath, ext=None):
    """Load data into GPU memory with automatic cleanup."""
    data = cp.asarray(fits.getdata(fpath, ext=ext), dtype="float32")
    try:
        yield data  # Provide the loaded data to the block
    finally:
        del data  # Free GPU memory when the block is exited
        gc.collect()  # Force garbage collection
        cp.get_default_memory_pool().free_all_blocks()


class FileCreationHandler(FileSystemEventHandler):
    def __init__(self, target_file):
        self.target_file = os.path.basename(target_file)
        self.created = False

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(self.target_file):
            self.created = True


def wait_for_masterframe(file_path, timeout=1800):
    """
    Wait for a file to be created using watchdog with timeout.

    Args:
        file_path (str): Path to the file to watch for
        timeout (int): Maximum time to wait in seconds (default: 1800 seconds / 30 minutes)

    Returns:
        bool: True if file was created, False if timeout occurred
    """
    # First check if file already exists
    if os.path.exists(file_path):
        return True

    directory = os.path.dirname(file_path) or "."
    handler = FileCreationHandler(file_path)
    observer = Observer()
    observer.schedule(handler, directory, recursive=False)
    observer.start()

    try:
        start_time = time.time()
        while not handler.created:
            if time.time() - start_time > timeout:
                return False
            time.sleep(1)
        return True
    finally:
        observer.stop()
        observer.join()


def read_link(link, timeout=1800):
    """
    Check if the link exists using watchdog, wait for it if it doesn't, and then read its content.

    Args:
        link (str): The file path to check and read.
        timeout (int, optional): Maximum time (in seconds) to wait for the file. Defaults to 1200.

    Returns:
        str: The content of the file.

    Raises:
        FileNotFoundError: If the file is not found within the timeout period.
        KeyboardInterrupt: If the user interrupts the waiting process.
    """
    try:
        # Use wait_for_masterframe to watch for the file
        if not wait_for_masterframe(link, timeout=timeout):
            raise FileNotFoundError(
                f"File '{link}' was not created within {timeout} seconds."
            )

        # Small delay to ensure file is fully written
        time.sleep(0.1)

        # Read and return the file content
        with open(link, "r") as f:
            return f.read().strip()

    except KeyboardInterrupt:
        print("KeyboardInterrupt while watching for a link. Exiting...")
        raise  # Re-raise the exception to terminate the following processes


def link_to_file(link):
    """Reformat link filename, not reading it"""
    import re

    pattern = r"\.link$"
    if re.match(pattern, link):
        return os.path.splitext(link)[0] + ".fits"
    else:
        raise ValueError("Not a link")


def search_with_date_offsets(template, max_offset=300, future=False):
    """
    Search for files based on a template, modifying embedded dates with offsets.
    future=False includes the current date

    Args:
        template (str): Template string with embedded dates (e.g., "/path/.../2025-01-01/.../20250102/...").
        max_offset (int, optional): Maximum number of days to offset (both positive and negative). Defaults to 2.

    Returns:
        str: A path to a closest existing master calibration frame file.
    """
    import re
    from datetime import datetime, timedelta

    # Regex to match dates in both YYYY-MM-DD and YYYYMMDD formats
    date_pattern = re.compile(r"\b\d{4}-\d{2}-\d{2}|\d{8}")

    # Extract all date strings from the template
    dates_in_template = date_pattern.findall(template)
    if not dates_in_template:
        raise ValueError("No date found in the template string.")

    date_night, date_utc = sorted(set(dates_in_template))

    # Parse dates into datetime objects
    date_night_format = "%Y-%m-%d"
    date_utc_format = "%Y%m%d"
    date_night_dt = datetime.strptime(date_night, date_night_format)
    date_utc_dt = datetime.strptime(date_utc, date_utc_format)

    if future:
        # Generate symmetric offsets: -1, +1, -2, +2, ..., up to max_offset
        offsets = [offset for i in range(1, max_offset + 1) for offset in (-i, i)]
    else:
        offsets = [-i for i in range(1, max_offset + 1)]
        offsets = [0] + offsets  # Include the original dates (offset 0)

    # Iterate through offsets
    for offset in offsets:
        # Adjust both dates by the offset
        adjusted_date_night_dt = date_night_dt + timedelta(days=offset)
        adjusted_date_utc_dt = date_utc_dt + timedelta(days=offset)

        # Format the adjusted dates
        adjusted_date_night = adjusted_date_night_dt.strftime(date_night_format)
        adjusted_date_utc = adjusted_date_utc_dt.strftime(date_utc_format)

        # Replace both dates in the template
        modified_path = template.replace(date_night, adjusted_date_night).replace(
            date_utc, adjusted_date_utc
        )

        # Check if the modified path exists
        if os.path.exists(modified_path):
            return modified_path

    # If no file is found, return None
    return None


def record_statistics(data, header, cropsize=500):
    mean, median, std, min, max = sigma_clipped_stats_cupy(
        data, sigma=3, maxiters=5, minmax=True
    )  # gpu vars
    header["CLIPMEAN"] = (float(mean), "3-sig clipped mean of the pixel values")
    header["CLIPMED"] = (float(median), "3-sig clipped median of the pixel values")
    header["CLIPSTD"] = (float(std), "3-sig clipped standard deviation of the pixels")
    header["CLIPMIN"] = (float(min), "3-sig clipped minimum of the pixel values")
    header["CLIPMAX"] = (float(max), "3-sig clipped maximum of the pixel values")

    height, width = data.shape
    start_row = (height - cropsize) // 2
    start_col = (width - cropsize) // 2

    # Slice the central 500x500 area
    cropped_data = data[
        start_row : start_row + cropsize, start_col : start_col + cropsize
    ]
    mean, median, std = sigma_clipped_stats_cupy(cropped_data, sigma=3, maxiters=5)
    header["CENCLPMN"] = (float(mean), f"3-sig clipped mean of center {cropsize}x{cropsize}")  # fmt: skip
    header["CENCLPMD"] = (float(median), f"3-sig clipped median of center {cropsize}x{cropsize}")  # fmt: skip
    header["CENCLPSD"] = (float(std), f"3-sig clipped std of center {cropsize}x{cropsize}")  # fmt: skip

    # header["CENCMIN"] = float(min)
    # header["CENCMAX"] = float(max)

    return header


def get_saturation_level(header, mbias_file, mdark_file, mflat_file):
    bitpix = header["BITPIX"]
    if bitpix > 0:
        # 	Positive BITPIX value indicates unsigned integer
        # 	Assuming the maximum value is 2**bitpix - 1
        maxval = 2**bitpix - 1
    else:
        raise ValueError("BITPIX value is not positive.")

    try:
        z = fits.getheader(mbias_file)["CLIPMED"]
        d = fits.getheader(mdark_file)["CLIPMED"]
        f = fits.getheader(mflat_file)["CENCLPMD"]
        satur_level = (maxval - z - d) / f

    except KeyError as e:
        print(f"Master frame header lacks a required key: {e}")
        print("Using default saturation level.")
        satur_level = maxval * 0.9

    return satur_level


def write_IMCMB_to_header(header, inputlist, full_path=False):
    """this function was copied from the package eclaire"""
    if inputlist is not None:
        llist = len(inputlist)

        # define the key format
        if llist <= 999:
            key = "IMCMB{:03d}"
        else:
            key = "IMCMB{:03X}"
            comment = "IMCMB keys are written in hexadecimal."
            # header.append("COMMENT", comment)  # original eclaire line
            header.add_comment(comment)

        # write the keys
        for i, f in enumerate(inputlist, 1):
            header[key.format(i)] = f if full_path else os.path.basename(f)
    return header


def add_image_id(header, key="IMAGEID"):
    """Add a unique image ID to the header."""
    header[key] = uuid.uuid4().hex
    header.comments[key] = "Unique ID of the image"
    return header


# def calculate_average_date_obs(date_obs_list):
#     import numpy as np
#     from astropy.time import Time

#     t = Time(date_obs_list, format="isot", scale="utc")
#     avg_time = np.mean(t.jd)
#     avg_time = Time(avg_time, format="jd", scale="utc")

#     # 'YYYY-MM-DDTHH:MM:SS'
#     avg_time_str = avg_time.isot

#     return avg_time_str


# def isot_to_mjd(time):  # 20181026 to 2018-10-26T00:00:00:000 to MJD form
#     from astropy.time import Time

#     yr = time[0:4]  # year
#     mo = time[4:6]  # month
#     da = time[6:8]  # day
#     isot = yr + "-" + mo + "-" + da + "T00:00:00.000"  # 	ignore hour:min:sec
#     t = Time(isot, format="isot", scale="utc")  # 	transform to MJD
#     return t.mjd
