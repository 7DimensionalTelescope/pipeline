import os
from glob import glob
import time
from astropy.io import fits

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import uuid

from ..path.path import PathHandler
from .checker import Checker
from ..header import write_header_file


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

    # Iterate through offsets - try all combinations of date offsets
    for offset_night in offsets:
        for offset_utc in offsets:
            # Adjust dates independently
            adjusted_date_night_dt = date_night_dt + timedelta(days=offset_night)
            adjusted_date_utc_dt = date_utc_dt + timedelta(days=offset_utc)

            # Format the adjusted dates
            adjusted_date_night = adjusted_date_night_dt.strftime(date_night_format)
            adjusted_date_utc = adjusted_date_utc_dt.strftime(date_utc_format)

            # Replace both dates in the template
            modified_path = template.replace(date_night, adjusted_date_night).replace(date_utc, adjusted_date_utc)

            # Check if the modified path exists
            # if os.path.exists(modified_path):
            #     return modified_path
            if "*" in template:
                # If there's a *, glob for all matches
                matches = glob(modified_path)
                if matches:
                    if len(matches) == 1:
                        if Checker().sanity_check(matches[0]):
                            return matches[0]
                    else:
                        return PathHandler(matches).get_minimum("exptime")
            else:
                if os.path.exists(modified_path):
                    return modified_path

    # If no file is found, return None
    return None


def get_saturation_level(header, mbias_file, mdark_file, mflat_file):
    bitpix = header["BITPIX"]
    if bitpix > 0:
        # 	Positive BITPIX value indicates unsigned integer
        # 	Assuming the maximum value is 2**bitpix - 1
        maxval = 2**bitpix - 1
    else:
        raise ValueError("BITPIX value is not positive.")

    try:
        z = fits.getval(mbias_file, "CLIPMED")
        d = fits.getval(mdark_file, "CLIPMED")
        f = fits.getval(mflat_file, "CENCLPMD")
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


def update_header_by_overwriting(filename, header, bitpix=-32):

    with fits.open(filename, mode="update") as hdul:
        if hdul[0].header["BITPIX"] == -32:
            for key in ["BZERO", "BSCALE"]:
                try:
                    del header[key]
                except:
                    continue

        for key in ["SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2"]:
            header[key] = hdul[0].header[key]

        hdul[0].header = header
        hdul.flush()


# def update_header_by_overwriting(filename, header, bitpix=-32):

#     if bitpix == -32:
#         for key in ["BZERO", "BSCALE"]:
#             try:
#                 del header[key]
#             except:
#                 continue

#     data = fits.getdata(filename)

#     fits.writeto(filename, data, header=header, overwrite=True)


def write_header(filename, header):
    for key in ["BZERO", "BSCALE"]:
        try:
            del header[key]
        except:
            continue

    header["BITPIX"] = -32

    if filename.endswith(".fits"):
        filename = filename.replace(".fits", ".header")

    write_header_file(filename, header)
