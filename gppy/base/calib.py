from datetime import datetime
from typing import List, Dict
from pathlib import Path
import re
import glob


class CalibrationData:
    """
    Represents and manages calibration data files (BIAS, DARK, FLAT).

    Provides specialized handling for calibration data, including:
    - Categorizing calibration files by type
    - Tracking calibration parameters
    - Managing processing state

    Attributes:
        calib_files (Dict[str, List[Path]]): Mapping of calibration types to file lists
        calib_params (Dict[str, dict]): Calibration parameters for each type
        _processed (bool): Internal flag to track processing state

    Methods:
        has_calib_files(): Check if any calibration files exist
        add_fits_file(): Add a FITS file to the calibration dataset
        mark_as_processed(): Mark the calibration dataset as processed

    Example:
        Automatically categorizes BIAS, DARK, and FLAT calibration files
        and tracks their metadata for further processing.
    """

    def __init__(self, folder_path: str or Path):
        if isinstance(folder_path, str):
            if ".fits" in folder_path:
                folder_path = Path(folder_path).parent
            else:
                folder_path = Path(folder_path)

        self.folder_path = folder_path

        self.calib_files: Dict[str, List[Path]] = {
            calib_type: [] for calib_type in ["BIAS", "DARK", "FLAT"]
        }
        self.calib_params: Dict[str, dict] = {
            calib_type: {} for calib_type in ["BIAS", "DARK", "FLAT"]
        }
        self._processed = False

        self.fits_files: List[Path] = []

    def add_fits_files(self):
        """
        Add FITS files to the calibration dataset.

        Parses the filename to determine calibration type and extracts metadata.

        Returns:
            bool: Whether any files were successfully added to the dataset
        """
        added_files = False
        fits_paths = glob.glob(str(self.folder_path) + "/*.fits")
        for fits_path in fits_paths:
            if self.add_fits_file(Path(fits_path)):
                added_files = True

        return added_files

    @property
    def processed(self):
        """
        Check if the calibration dataset has been processed.

        Returns:
            bool: Processing state of the calibration dataset
        """
        return self._processed

    def has_calib_files(self) -> bool:
        """
        Check if any calibration files exist in the dataset.

        Returns:
            bool: True if calibration files are present, False otherwise
        """
        return any(len(files) > 0 for files in self.calib_files.values())

    def add_fits_file(self, fits_path: Path) -> bool:
        """
        Add a FITS file to the calibration dataset.

        Parses the filename to determine calibration type and extracts metadata.

        Args:
            fits_path (Path): Path to the FITS file

        Returns:
            bool: Whether the file was successfully added to the dataset
        """
        filename = fits_path.name
        pattern = (
            r"(\w+)_(\d{8})_(\d{6})_([^_]+)_([^_]+)_(\d+x\d+)_(\d+\.\d+)s_\d+\.fits"
        )
        match = re.match(pattern, filename)

        if match:
            calib_type = None
            for ctype in ["BIAS", "DARK", "FLAT"]:
                if ctype in match.group(4):
                    calib_type = ctype
                    break

            if calib_type:
                self.fits_files.append(fits_path)
                self.calib_files[calib_type].append(fits_path)

                if not self.calib_params[calib_type]:
                    self._parse_info(match)
                    self.calib_params[calib_type] = {
                        "filter": self.filter,
                        "binning": self.n_binning,
                        "exposure": self.exposure,
                    }
                return True
        return False

    def _parse_info(self, match) -> None:
        """
        Parse common metadata from filename using regex match.

        Extracts:
        - Precise observation timestamp
        - Observation filter
        - Detector binning
        - Exposure time

        Args:
            match (re.Match): Regex match object from filename parsing
        """
        self.unit = self.folder_path.parent.name

        # Parse folder name (e.g., 2001-02-23_gain2750 or 2001-02-23_ToO_gain2750)
        folder_parts = self.folder_path.name.split("_")
        self.date = folder_parts[0]

        if "ToO" in self.folder_path.name:
            self.too = True
        else:
            self.too = False

        # Find the gain part, handling potential additional components
        gain_part = next(
            (part for part in folder_parts if part.startswith("gain")), None
        )
        if gain_part is None:
            print(f"Could not find gain in folder name: {self.folder_path.name}")
        else:
            self.gain = int(re.search(r"gain(\d+)", gain_part).group(1))

        date_str = f"{match.group(2)}_{match.group(3)}"
        self.datetime = datetime.strptime(date_str, "%Y%m%d_%H%M%S")

        filter_str = match.group(5)
        self.filter = f"m{filter_str}" if filter_str.isdigit() else filter_str

        binning = match.group(6)
        self.n_binning = int(binning.split("x")[0])
        self.exposure = float(match.group(7))

    def mark_as_processed(self):
        """
        Mark the calibration dataset as processed.

        Sets the internal processed flag to True, indicating
        that all necessary processing has been completed.
        """
        self._processed = True

    @property
    def obs_params(self):
        return {
            "date": self.date,
            "unit": self.unit,
            "n_binning": self.n_binning,
            "gain": self.gain,
        }

    @property
    def name(self):
        return f"{self.date}_{self.n_binning}x{self.n_binning}_gain{self.gain}_{self.unit}_masterframe"

    def generate_masterframe(self):
        from ..preprocess import MasterFrameGenerator

        master = MasterFrameGenerator(self.obs_params)
        master.run()
