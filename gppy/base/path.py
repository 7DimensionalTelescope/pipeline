import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from ..utils import check_params, get_camera, define_output_dir, Path7DS
from .. import const


def raw_dir_from_processed(str):

    return


image_unique_keys = ["date", "filter", "obj", "unit", "exposure"]


class PathHandler:
    def __init__(self, input):
        self._data_type = None  # propagate user-input in config.name
        if isinstance(input, str):
            if input.endswith(".fits"):
                self._input_files = [Path(input)]
            elif input.endswith(".yml"):
                from ..config import Configuration

                self._input_files = Configuration(input).config.file.raw_files
                self._input_files = [Path(file) for file in files]
            else:
                raise ValueError("Input must be a .fits file or a .yml file.")
        elif isinstance(input, Path):
            self._input_files = [input]
        elif isinstance(input, list):
            self._input_files = [Path(img) for img in input]
        elif hasattr(input, "config"):
            files = input.config.file.raw_files
            self._input_files = [Path(file) for file in files]
            self._data_type = input.config.name
            # else:
            #     raise ValueError("Input object must be Configuration")
        else:
            raise TypeError("Input must be a string, Path, or list of strings/Paths.")

        if not (self.is_present(self._input_files)):
            raise FileNotFoundError(f"Path does not exist: {self._input_files}")

        self.select_output_dir()
        self.define_common_path()
        self.define_specific_path()

    def __getattr__(self, name):
        if name.endswith("_to_string"):
            tmp_name = name.replace("_to_string", "")
            if getattr(self, tmp_name) is not None:
                if isinstance(getattr(self, tmp_name), Path):
                    return str(getattr(self, tmp_name).absolute())
                elif isinstance(getattr(self, tmp_name), str):
                    return getattr(self, tmp_name)
                else:
                    return getattr(self, tmp_name)
        elif name.endswith("_to_path"):
            temp_name = name.replace("_to_path", "")
            if getattr(self, temp_name) is not None:
                if isinstance(getattr(self, temp_name), str):
                    return Path(getattr(self, temp_name))
                elif isinstance(getattr(self, temp_name), Path):
                    return getattr(self, temp_name)
                else:
                    return getattr(self, tmp_name)
        elif name in self.__dict__:
            var = getattr(self, name)
            # Create the directory if nonexisting
            if isinstance(var, str) or isinstance(var, Path):
                os.makedirs(str(var), exist_ok=True)
            return getattr(self, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __getattribute__(self, name):
        """Make sure accessed dirs exist"""
        # delegate to super first
        value = super().__getattribute__(name)

        # custom extras
        if isinstance(value, (str, Path)):
            os.makedirs(str(value), exist_ok=True)
        return value

    def __repr__(self):
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items() if not k.startswith("_"))

    def is_present(self, path):
        paths = np.atleast_1d(path)
        return all([p.exists() for p in paths])

    def select_output_dir(self):
        # date_utc = Path7DS(self._input_files[0]).date  # utc date. e.g., "20250101"
        date_folder = check_params(self._input_files[0])["date"]
        if date_folder < "20260101":
            self.output_parent_dir = const.PROCESSED_DIR
        else:
            raise ValueError("Predefined date cap reached: consider moving to another disk.")

    def define_common_path(self):

        if const.RAWDATA_DIR in str(self._input_files[0]):
            params = check_params(self._input_files[0])
            self.data_type = self._data_type or "raw"
            self.raw_images = [str(file.absolute()) for file in self._input_files]
            _relative_path = os.path.join(params["date"], params["obj"], params["filter"], params["unit"])
            self.output_dir = os.path.join(self.output_parent_dir, _relative_path)
            self.masterframe_dir = os.path.join(const.MASTER_FRAME_DIR, params["date"], params["unit"])
            self.factory_dir = os.path.join(const.FACTORY_DIR, _relative_path)
            self.metadata_dir = os.path.join(self.output_parent_dir, params["date"])
            processed_file_stems = [switch_raw_name_order(file.stem) for file in self._input_files]
            self.processed_images = [
                os.path.join(self.output_dir, "images", file_stem + ".fits") for file_stem in processed_file_stems
            ]
        elif self.output_parent_dir in str(self._input_files[0]):
            self.data_type = self._data_type or "processed"
            self.processed_images = [str(file.absolute()) for file in self._input_files]
            # self.processed_file_stems = [file.stem for file in self._input_files]
            self.factory_dir = os.path.join(const.FACTORY_DIR, *Path(self._input_files[0]).parts[-6:-3])
            self.output_dir = self._input_files[0].parent.parent
        else:
            self.data_type = self._data_type or "user-input"
            print("User input data type detected. Assume the input is a list of processed images.")
            self.processed_images = [str(file.absolute()) for file in self._input_files]
            # self.processed_file_stems = [file.stem for file in self._input_files]
            self.output_dir = self._input_files[0].parent.parent
            self.factory_dir = self._input_files[0].parent.parent / "factory"

        self.image_dir = os.path.join(self.output_dir, "images")
        self.figure_dir = os.path.join(self.output_dir, "figures")
        self.daily_stacked_dir = os.path.join(self.output_dir, "stacked")
        self.subtracted_dir = os.path.join(self.output_dir, "subtracted")
        self.ref_sex_dir = Path(os.path.join(const.REF_DIR, "srcExt"))

        self.astrometry_ref_ris_dir = Path("/lyman/data1/factory/catalog/gaia_dr3_7DT")
        self.astrometry_ref_query_dir = Path("/lyman/data1/factory/ref_scamp")
        self.photometry_ref_ris_dir = Path("/lyman/data1/factory/ref_cat")  # divided by RIS tiles
        self.photometry_ref_gaia_dir = Path("/lyman/data1/Calibration/7DT-Calibration/output/Calibration_Tile")
        self.subtraction_ref_image_dir = Path("/lyman/data1/factory/ref_frame")

    # def path_preprocess(self):
    #     pass

    # def path_astromety(self):
    #     pass

    # def path_photometry(self):
    #     pass

    # def path_stacking(self):
    #     pass

    # def path_subtraction(self):
    #     pass

    def define_specific_path(self):
        # self.preprocess = PathPreprocess(path_processed=self.output_dir)
        # self.astrometry = PathAstrometry(path_processed=self.output_dir)
        pass


def switch_raw_name_order(name):
    parts = name.split("_")
    return "_".join(parts[3:5] + parts[0:1] + [format_subseconds(parts[6])] + parts[1:3])


def format_subseconds(sec: str):
    """100.0s -> 100s, 0.1s -> 0pt100s"""
    s = float(sec[:-1])
    integer_second = int(s)
    if integer_second != 0:
        return f"{integer_second}s"

    # if subsecond
    millis = int(abs(s) * 1000 + 0.5)  # round to nearest ms
    return f"0pt{millis:03d}s"


class PathPreprocess(PathHandler):
    """
    Gives you attributes such as

        path.preprocess.mbias_link      (file - no mkdir)
        path.preprocess.intermediate    (directory - mkdir on first access)

    without building anything up-front.  Everything is resolved the first
    time you touch the attribute.
    """

    # everything we can serve and how it should be built
    _spec = {
        "mbias_link": lambda self: self.path_fdz / f"bias_{self._date}_{self._cam}.link",
        "mdark_link": lambda self: self.path_fdz / f"dark_{self._date}_{self._exp}s_{self._cam}.link",
        "mflat_link": lambda self: self.path_fdz / f"flat_{self._date}_{self._filt}_{self._cam}.link",
    }

    # -----------------------------------------------------------------

    def __init__(self, path_processed):
        date_utc = "test"
        filt = "test"
        camera = "test"
        exposure = "test"

        _date_dir = define_output_dir(self.config.obs.date, self.config.obs.n_binning, self.config.obs.gain)
        self.path_fdz = Path(_date_dir) / self.config.obs.unit

    def __getattr__(self, item):
        """lazy magic"""
        if item not in self._spec:
            raise AttributeError(item)

        path = self._spec[item](self)  # build the Path
        path.mkdir(parents=True, exist_ok=True)
        return path

    # pretty printing
    def __repr__(self):
        keys = sorted(k for k in (*self.__dict__, *self._spec))
        return "\n".join(f"  {k}: {getattr(self, k)}" for k in keys)


class PathAstrometry(PathHandler):
    _all = {
        "mbias_link": lambda self: self.path_fdz / f"bias_{self._date}_{self._cam}.link",
        "mdark_link": lambda self: self.path_fdz / f"dark_{self._date}_{self._exp}s_{self._cam}.link",
        "mflat_link": lambda self: self.path_fdz / f"flat_{self._date}_{self._filt}_{self._cam}.link",
    }

    # -----------------------------------------------------------------

    def __init__(self, path_processed):
        self.path_processed = Path(path_processed)

    def __getattr__(self, item):
        """lazy magic"""
        if item not in self._all:
            raise AttributeError(item)

        path = self._all[item](self)  # build the Path
        path.mkdir(parents=True, exist_ok=True)
        return path

    # pretty printing
    def __repr__(self):
        keys = sorted(k for k in (*self.__dict__, *self._all))
        return "\n".join(f"  {k}: {getattr(self, k)}" for k in keys)
