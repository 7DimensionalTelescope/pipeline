from multiprocessing import Queue
import re
import time
from pathlib import Path
import copy

from .services.queue import QueueManager, Priority
from .data import CalibrationData, ObservationDataSet
from .run import run_scidata_reduction_with_tree, run_masterframe_generator_with_tree
from .const import available_7dt_units
from .utils import check_obs_file


def reprocess_folder(
    folder,
    overwrite=False,
    sync_units=False,
    queue=None,
    processes=["preprocess", "astrometry", "photometry", "combine", "subtract"],
    **kwargs,
):
    """
    Reprocess all FITS files in a given folder and its subfolders.

    This function performs a comprehensive scan of the input folder, identifying
    and processing astronomical data folders. It handles both calibration and
    observation data, using a queue-based parallel processing approach.

    Processing steps:
    1. Validate input folder
    2. Identify valid data subfolders
    3. Initialize calibration and observation data handlers
    4. Scan and add FITS files to data handlers
    5. Process calibration and observation data
    6. Wait for queue to complete processing
    7. Report any processing errors

    Args:
        folder (str): Path to the folder containing data to be reprocessed
        overwrite (bool, optional): Flag to enable overwriting of existing
            processed files. Defaults to False.

    Raises:
        RuntimeError: If any errors occur during folder processing
        ValueError: If the input folder is not a valid directory

    Example:
        >>> reprocess_folder('/path/to/data/folder')
        Finished processing files in /path/to/data/folder

    Note:
        - Supports folders with naming patterns like:
          YYYY-MM-DD(_ToO)?(_NxN)?_gainX
        - Uses QueueManager for parallel processing
        - Skips folders without FITS files
    """
    if queue is None:
        queue = QueueManager(max_workers=60, gpu_workers=10)

    # Convert folder to Path object
    folder_path = Path(folder)

    # Check if folder exists
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Error: {folder} is not a valid directory")
        return

    # Track errors during processing
    # processing_errors = []

    # Find all potential data folders
    data_folders = []

    # Check if the input folder itself matches the data folder pattern
    if re.match(r"\d{4}-\d{2}-\d{2}(_ToO)?(_\dx\d)?_gain\d+", folder_path.name):
        data_folders.append(folder_path)

    # If not, search subfolders
    data_folders.extend(
        [
            f
            for f in folder_path.iterdir()
            if f.is_dir()
            and re.match(r"\d{4}-\d{2}-\d{2}(_ToO)?(_\dx\d)?_gain\d+", f.name)
        ]
    )

    # If no folders found, print a warning
    if not data_folders:
        print(f"No valid data folders found in {folder_path}")
        return

    # Process each data folder
    for data_folder in data_folders:
        # try:
        # Initialize data handlers for the folder
        calib_data = CalibrationData(data_folder)
        obs_dataset = ObservationDataSet()

        # Find and process all FITS files in the folder
        fits_files = list(data_folder.glob("*.fits"))

        if not fits_files:
            print(f"No FITS files found in {data_folder}")
            continue

        for fits_file in fits_files:
            calib_data.add_fits_file(fits_file)
            obs_dataset.add_fits_file(fits_file)

        # Process calibration data if exists
        if calib_data.has_calib_files() and not calib_data.processed:
            
            tree = run_masterframe_generator_with_tree(
                calib_data.obs_params,
                overwrite=overwrite,
                priority=Priority.HIGH,
            )
            if tree is not None:
                queue.add_tree(tree)

            time.sleep(0.1)

        # Process observation data
        for obs in obs_dataset.get_unprocessed():
            obs_dataset.mark_as_processed(obs.identifier)

            priority = Priority.HIGH if obs.too else Priority.MEDIUM

            tree = run_scidata_reduction_with_tree(obs.obs_params, overwrite=overwrite, processes=processes, priority=priority)
            if tree is not None:
                queue.add_tree(tree)

            time.sleep(0.1)

        if sync_units:
            original_unit = copy.copy(obs.unit)
            for unit in available_7dt_units:
                if unit != original_unit:
                    obs.unit = unit
                    obs.filter = None
                    files = check_obs_file(obs.obs_params)
                    if files:
                        reprocess_folder(
                            Path(files[0]).parent,
                            overwrite=overwrite,
                            queue=queue,
                            processes=processes,
                            sync_units=False,
                        )

    # Wait for queue to complete processing
    queue.wait_until_task_complete("all")
    
    print(queue.trees)
    # Print summary of processing
    print(f"Finished processing files in {folder}")

    return queue