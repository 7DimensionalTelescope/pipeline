from datetime import datetime, timedelta
from typing import Optional

import itertools
from pathlib import Path

from astropy.io import fits
import re

import glob

from ..utils import parse_key_params_from_header

class ObservationData:

    _id_counter = itertools.count(1)

    """
    Represents and manages astronomical observation data files.

    Provides specialized handling for observation data, including:
    - Tracking observation targets
    - Managing processed target states
    - Supporting multiple targets and filters

    Attributes:
        target (Optional[str]): Current observation target
        _processed (set): Set of processed target information

    Methods:
        add_fits_file(): Add a FITS file to the observation dataset
        get_unprocessed(): Retrieve unprocessed targets
        mark_as_processed(): Mark specific targets as processed

    Example:
        Tracks observation targets, filters, and processing state
        for complex astronomical observation datasets.
    """

    def __init__(self, file_path: str or Path):
        self.target: Optional[str] = None

        self._id = next(self._id_counter)


        if isinstance(file_path, str):
            file_path = Path(file_path)

        self.file_path = file_path

        if "calib" in file_path.name:
            self._file_type = "processed"

        else:
            self._file_type = "raw"

        info = parse_key_params_from_header(self.file_path)[0]
        self._set_attr_from_dict(info)
        self.too = False

    @classmethod
    def from_pattern(cls, pattern: str):
        """
        Create an ObservationData instance from a file pattern.
        
        Args:
            pattern: A string pattern like 'path_to_fits_folder/*.fits'
            
        Returns:
            ObservationData: An instance with all matching files
            
        Raises:
            ValueError: If multiple files match the pattern
        """
        matching_files = glob.glob(pattern)
        file_set = set()
        
        if not matching_files:
            raise ValueError(f"No files match pattern: {pattern}")
        
        if len(matching_files) > 1:
            for file in matching_files:
                file_set.add(parse_key_params_from_header(file)[0])

            if len(file_set) != 1:
                error_message = f"Multiple files match pattern: {pattern}\n"
                error_message += "Matching files:\n"
                for file in matching_files:
                    error_message += f"- {file}\n"
                raise ValueError(error_message)
            else:
                return cls(matching_files[0])
        
        # If we reach here, there's exactly one matching file
        return cls(matching_files[0])

    @property
    def id(self):
        return self._id

    @classmethod
    def _parse_info_from_header(cls, filename) -> None:
        """
        Extract target information from a FITS filename.

        Args:
            file_path (Path): Path to the FITS file

        Returns:
            tuple: Target name and filter, or None if parsing fails
        """
        info = {}
        header = fits.getheader(filename)
        for attr, key in zip(["exposure", "gain", "filter", "date", "obj", "unit", "n_binning"], \
                             ["EXPOSURE", "GAIN", "FILTER", "DATE-LOC", "OBJECT", "TELESCOP", "XBINNING"]):  # fmt:skip
            if key == "DATE-LOC":
                header_date = datetime.fromisoformat(header[key])
                adjusted_date = header_date - timedelta(hours=12)
                final_date = adjusted_date.date()
                info[attr] = final_date.isoformat()
            else:
                info[attr] = header[key]

        return info

    def _set_attr_from_dict(self, info):
        for attr, value in info.items():
            setattr(self, attr, value)

    @property
    def identifier(self):
        return (self.obj, self.filter)

    @property
    def obs_params(self):
        return {
            "date": self.date,
            "unit": self.unit,
            "gain": self.gain,
            "obj": self.obj,
            "filter": self.filter,
            "n_binning": self.n_binning,
        }

    @property
    def name(self):
        return f"{self.date}_{self.n_binning}x{self.n_binning}_gain{self.gain}_{self.obj}_{self.unit}_{self.filter}"

    def run_calibaration(self, save_path=None, verbose=True):
        from ..config import Configuration
        from ..preprocess import Calibration

        self.config = Configuration(self.obs_params, overwrite=True, verbose=verbose)
        if save_path:
            self.config.config.path.path_processed = save_path
        calib = Calibration(self.config)
        calib.run()

    def run_astrometry(self):
        from ..astrometry import Astrometry

        if hasattr(self, "config"):
            astro = Astrometry(self.config)
        else:
            astro = Astrometry.from_file(self.file_path)
        astro.run()

    def run_photometry(self):
        from ..photometry import Photometry

        if hasattr(self, "config"):
            phot = Photometry(self.config)
        else:
            phot = Photometry.from_file(self.file_path)
        phot.run()


class ObservationDataSet(ObservationData):

    def __init__(self, folder_path: str or Path):
        self._processed = set()
        self.obs_list = []

    def add_fits_file(self, fits_path: Path) -> bool:
        """
        Add a FITS file to the observation dataset.

        Parses the filename to extract target and observation metadata.

        Args:
            fits_path (Path): Path to the FITS file

        Returns:
            bool: Whether the file was successfully added to the dataset
        """
        pattern = (
            r"(\w+)_(\d{8})_(\d{6})_([^_]+)_([^_]+)_(\d+x\d+)_(\d+\.\d+)s_\d+\.fits"
        )
        match = re.match(pattern, fits_path.name)

        if match:
            if all(ctype not in fits_path.name for ctype in ["BIAS", "DARK", "FLAT"]):
                obs = ObservationData(fits_path)
                self.obs_list.append(obs)
                return True
            return False
        else:
            return False

    @property
    def processed(self):
        """
        Get the set of processed targets.

        Returns:
            set: Set of processed target information
        """
        return self._processed

    def get_unprocessed(self) -> set:
        """
        Retrieve targets that have not yet been processed.

        Returns:
            set: Set of unprocessed target information
        """
        for obs in self.obs_list:
            if obs.identifier not in self._processed:
                yield obs

    def mark_as_processed(self, obs_identifier):
        """
        Mark a specific target as processed.

        Args:
            obs_identifier (Tuple[str, str]): Identifier of the observation to mark as processed
        """
        self._processed.add(obs_identifier)
