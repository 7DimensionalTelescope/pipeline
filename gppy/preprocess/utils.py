import os
import time
from astropy.io import fits

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import uuid


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

    # Iterate through offsets
    for offset in offsets:
        # Adjust both dates by the offset
        adjusted_date_night_dt = date_night_dt + timedelta(days=offset)
        adjusted_date_utc_dt = date_utc_dt + timedelta(days=offset)

        # Format the adjusted dates
        adjusted_date_night = adjusted_date_night_dt.strftime(date_night_format)
        adjusted_date_utc = adjusted_date_utc_dt.strftime(date_utc_format)

        # Replace both dates in the template
        modified_path = template.replace(date_night, adjusted_date_night).replace(date_utc, adjusted_date_utc)

        # Check if the modified path exists
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


def add_padding(header, n, copy_header=False):
    """
    Add empty COMMENT entries to a FITS header to ensure specific block sizes.

    This function helps manage FITS header sizes by adding padding comments.
    Useful for maintaining specific header block structures required by
    astronomical data processing tools.

    Args:
        header (fits.Header): Input FITS header
        n (int): Target number of 2880-byte blocks
        copy_header (bool, optional): If True, operates on a copy of the header.
            Defaults to False. Note: Using True is slower.

    Returns:
        fits.Header: Header with added padding comments

    Note:
        - Each COMMENT is 80 bytes long
        - The total header size must be a multiple of 2880 bytes
    """
    if copy_header:
        import copy

        header = copy.deepcopy(header)

    info_size = len(header.cards) * 80

    target_size = (n - 1) * 2880  # fits header size is a multiple of 2880 bytes
    padding_needed = target_size - info_size
    num_comments = padding_needed // 80  # (each COMMENT is 80 bytes)

    # CAVEAT: END also uses one line.
    # for _ in range(num_comments - 1):  # <<< full n-1 2880-byte blocks
    for _ in range(num_comments):  # <<< marginal n blocks
        header.add_comment(" ")

    return header


def remove_padding(header):
    """
    Remove COMMENT padding from a FITS header.

    Strips all trailing COMMENT entries, returning a header with only
    significant entries.

    Args:
        header (fits.Header): Input FITS header with potential padding

    Returns:
        fits.Header: Header with padding comments removed

    Note:
        This method is primarily useful for header inspection and may not
        be directly applicable for header updates.
    """
    # Extract all header cards
    cards = list(header.cards)

    # Find the last non-COMMENT entry
    for i in range(len(cards) - 1, -1, -1):
        if cards[i][0] != "COMMENT":  # i is the last non-comment idx
            break

    return header[: i + 1]

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

def write_header(filename, header):
    for key in ["BZERO", "BSCALE"]:
        try:
            del header[key]
        except:
            continue

    header["BITPIX"] = -32
    
    if filename.endswith(".fits"):
        filename = filename.replace(".fits", ".header")

    with open(filename, "w") as f:
        f.write(header.tostring(sep="\n"))

# def update_header_by_overwriting(filename, header, bitpix=-32):

#     if bitpix == -32:
#         for key in ["BZERO", "BSCALE"]:
#             try:
#                 del header[key]
#             except:
#                 continue
    
#     data = fits.getdata(filename)

#     fits.writeto(filename, data, header=header, overwrite=True)
    