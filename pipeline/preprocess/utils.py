import os
import time
import uuid
from glob import glob
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from astropy.io import fits
import fitsio
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from ..path.path import PathHandler
from ..path.name import NameHandler
from ..services.checker import Checker
from ..utils.header import write_header_file, get_header


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


def tolerant_search(template, dtype, max_offset=30, future=False):

    searched = search_with_date_offsets(template, max_offset=max_offset, future=future)
    # if found right away
    if searched:
        return searched

    # if dark, try other units (still same camera serial)
    if dtype == "dark":
        path = PathHandler(template)
        path.name.unit = "*"
        new_template = path.preprocess.masterframe
        searched = search_with_date_offsets(new_template, max_offset=max_offset, future=future)
        if searched:
            return searched

        # # try other exptimes. this requires scaling; NYI
        # path.name.exptime = "*"
        # new_template = path.preprocess.masterframe
        # searched = search_with_date_offsets(new_template, max_offset=max_offset, future=future)
        # if searched:
        #     return searched

    # still not found
    return None


def search_with_date_offsets(template, max_offset=30, future=False):
    """
    Search for files based on a template, modifying embedded dates with offsets.
    future=False includes the current date

    Args:
        template (str): Template string with embedded dates (e.g., "/path/.../2025-01-01/.../20250102/...").
        max_offset (int, optional): Maximum number of days to offset (both positive and negative). Defaults to 2.
            originally 300 for early 7DT flat, later 30 days.

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
        adjusted_date_night_dt = date_night_dt + timedelta(days=offset_night)
        adjusted_date_night = adjusted_date_night_dt.strftime(date_night_format)
        # Calculate UTC date properly using datetime arithmetic instead of string manipulation
        adjusted_date_utc_dt = date_utc_dt + timedelta(days=offset_night)
        adjusted_date_utc = adjusted_date_utc_dt.strftime(date_utc_format)
        modified_path = template.replace(date_night, adjusted_date_night).replace(date_utc, adjusted_date_utc)
        # for offset_utc in offsets:
        #     # Adjust dates independently
        #     adjusted_date_night_dt = date_night_dt + timedelta(days=offset_night)
        #     adjusted_date_utc_dt = date_utc_dt + timedelta(days=offset_utc)

        #     # Format the adjusted dates
        #     adjusted_date_night = adjusted_date_night_dt.strftime(date_night_format)
        #     adjusted_date_utc = adjusted_date_utc_dt.strftime(date_utc_format)

        #     # Replace both dates in the template
        #     modified_path = template.replace(date_night, adjusted_date_night).replace(date_utc, adjusted_date_utc)

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
                        continue
                else:
                    min_exptime_image = PathHandler(matches).get_minimum("exptime")
                    if Checker().sanity_check(min_exptime_image):
                        return min_exptime_image
                    else:
                        continue
        else:
            if os.path.exists(modified_path):
                if Checker().sanity_check(modified_path):
                    return modified_path
                else:
                    continue

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


def get_zdf_from_header_IMCMB(image):
    header = get_header(image)
    zdf_candidates = [v for k, v in header.items() if "IMCMB" in k]  # [z, d, f]
    zdf = []
    for master_frame_type in ["bias", "dark", "flat"]:
        for i, typ in enumerate(NameHandler(zdf_candidates).type):
            if typ[0] in "master" and typ[1] == master_frame_type:
                zdf.append(zdf_candidates[i])
                break
        else:
            raise ValueError(f"{master_frame_type} not correctly found from header IMCMB of {image}")
    return zdf


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


def sanitize_header(header: fits.Header) -> fits.Header:
    for key in ["BZERO", "BSCALE"]:
        try:
            del header[key]
        except Exception:
            continue

    header["BITPIX"] = -32
    return header


def write_header(filename, header):
    # for key in ["BZERO", "BSCALE"]:
    #     try:
    #         del header[key]
    #     except:
    #         continue

    # header["BITPIX"] = -32

    if filename.endswith(".fits"):
        filename = filename.replace(".fits", ".header")

    write_header_file(filename, header)


def ensure_mjd_in_header(header, logger=None):
    """
    Ensure the FITS header has MJD. If missing but DATE-OBS exists,
    compute MJD from DATE-OBS (UTC) and write it into the header.

    Parameters
    ----------
    header : astropy.io.fits.Header
        FITS header object to update.
    logger : logging.Logger, optional
        Logger instance to record messages. If None, falls back to print.

    Returns
    -------
    header : astropy.io.fits.Header
        Updated header with MJD included.
    """

    def chatter(msg: str, level: str = "debug"):
        """Internal logging helper."""
        if logger is not None:
            return getattr(logger, level)(msg)
        else:
            print(f"[ensure_mjd_in_header:{level.upper()}] {msg}")

    if "MJD" not in header:
        if "DATE-OBS" in header:
            try:
                from astropy.time import Time

                mjd = Time(header["DATE-OBS"], format="isot", scale="utc").mjd
                header["MJD"] = (mjd, "Modified Julian Date derived from DATE-OBS")
                chatter(f"Added MJD={mjd:.5f} (from DATE-OBS={header['DATE-OBS']})")
            except Exception as e:
                chatter(f"Failed to convert DATE-OBS='{header.get('DATE-OBS')}' to MJD: {e}", "error")
        else:
            chatter("Header missing both MJD and DATE-OBS; cannot compute MJD.", "error")
    else:
        # chatter("Header already contains MJD; no action taken.", "debug")
        pass

    return header


def read_fits_image(path, use_memmap=False):
    """
    Read FITS image using fitsio (faster) or astropy as fallback.

    Args:
        path: Path to FITS file
        use_memmap: If True, use memory mapping for large files (only for astropy fallback)

    Returns:
        numpy array as float32
    """
    try:
        # Use fitsio for faster reading
        data = fitsio.read(path)
        data = data.astype(np.float32)
    except:
        # Fallback to astropy
        data = fits.getdata(path, memmap=use_memmap)
        data = data.astype(np.float32)
    return data


def read_fits_image(path):
    data = fitsio.read(path)
    data = data.astype(np.float32)
    return data


def read_fits_images(input_paths, output_paths, max_workers=10):
    pairs = list(zip(input_paths, output_paths))
    pairs = sorted(pairs, key=lambda x: x[0])  # safe

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        data = list(executor.map(read_fits_image, [in_path for in_path, _ in pairs]))

    # data = [read_fits_image(in_path) for in_path, _ in pairs]
    in_paths = [in_path for in_path, _ in pairs]
    out_paths = [out_path for _, out_path in pairs]
    return data, in_paths, out_paths


def write_fits_image(output_path, processed_data):
    """Write processed image to disk using the header pre-generated on disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    header_file = output_path.replace(".fits", ".header")
    header = None
    if os.path.exists(header_file):
        with open(header_file, "r") as f:
            header = fits.Header.fromstring(f.read(), sep="\n")

    fits.writeto(output_path, processed_data, header=header, overwrite=True)
    # fitsio.write(output_path, processed_data, header=list(header.cards), clobber=True)


def write_fits_images(paths, data, max_workers=10):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(write_fits_image, paths, data))

    # for output_path, subdata in zip(paths, data):
    #     write_fits_image(output_path, subdata)
