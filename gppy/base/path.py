from pathlib import Path
from .. import const
import numpy as np
from ..utils import parse_key_params_from_header#, parse_key_params_from_filename
import re
import os
image_unique_keys = ["date", "filter", "obj", "unit", "exposure"]

class PathHandler:

    base_path = Path("/lyman/data2/")

    def __init__(self, input):
        if isinstance(input, str):
            self._input_file = [Path(input)]
        elif isinstance(input, Path):
            self._input_file = [input]
        elif isinstance(input, list):
            self._input_file = [Path(img) for img in input]
        else:
            raise TypeError("Input must be a string, Path, or list of strings/Paths.")
        
        if not(self.is_present(self._input_file)):
            raise FileNotFoundError(f"Path does not exist: {self._input_file}")
        
        self.define_common_path()

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
            return getattr(self, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __repr__(self):
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items() if not k.startswith("_"))
    
    def is_present(self, path):
        paths = np.atleast_1d(path)
        return all([p.exists() for p in paths])
        
    def define_common_path(self):
        if const.RAWDATA_DIR in str(self._input_file[0]):
            params = check_params(self._input_file[0])
            self.data_type = "raw"
            self.raw_image= [str(file.absolute()) for file in self._input_file]
            _tmp_relpath = os.path.join(params["date"], params["obj"], params["filter"], params["unit"])
            self.output_dir = os.path.join(const.PROCESSED_DIR, _tmp_relpath)
            self.factory_dir = os.path.join(const.FACTORY_DIR, _tmp_relpath)
            self.masterframe_dir = os.path.join(const.FACTORY_DIR, _tmp_relpath)
            self.file_prefix = [
                switch_name_order(file.stem) for file in self._input_file
            ]
            self.processed_image = [
                os.path.join(self.output_dir, "images", file_prefix+".fits") for file_prefix in self.file_prefix
            ]
        elif const.PROCESSED_DIR in str(self._input_file[0]):
            self.data_type = "processed"
            self.processed_image = [str(file.absolute()) for file in self._input_file]
            self.file_prefix = [file.stem for file in self._input_file]
            self.factory_dir = os.path.join(const.FACTORY_DIR, *Path(self._input_file[0]).parts[-6:-3])
            self.output_dir = self._input_file[0].parent.parent
        else:
            self.data_type = "user-input"
            print("User input data type detected. Assume the input is a list of processed images.")
            self.processed_image = [str(file.absolute()) for file in self._input_file]
            self.file_prefix = [file.stem for file in self._input_file]
            self.output_dir = self._input_file[0].parent.parent
            self.factory_dir = self._input_file[0].parent.parent / "factory"
        
        self.image_dir = os.path.join(self.output_dir, "images")
        self.figure_dir = os.path.join(self.output_dir, "figures")
        self.daily_stacked_dir = os.path.join(self.output_dir, "stacked")
        self.subtracted_dir = os.path.join(self.output_dir, "subtracted")
        self.ref_sex_dir = Path(os.path.join(const.REF_DIR, "srcExt"))
  
    def path_preprocess(self):
        pass

    def path_astromety(self):
        pass

    def path_photometry(self):
        pass

    def path_stacking(self):
        pass

def switch_name_order(name):
    """
    Switch the order of the name based on the given order.

    Args:
        name (str): The name to be switched.
        order (list): The order to switch the name.

    Returns:
        str: The switched name.
    """
    parts = name.split("_")
    return "_".join(parts[3:5] + parts[0:1] + parts[6:7] + parts[1:3])


def check_params(img):
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
    
    pattern = r"^(7DT\d{2})_(\d{8}_\d{6})_([a-zA-Z0-9]+)_([a-zA-Z0-9]+)_(\dx\d)_(\d+\.?\d*)s_([0-9]+)"
    match = re.match(pattern, path.stem)
        
    if match:
        units, date_string, target, filt, n_binning, exposure, _ = match.groups()
        date = subtract_half_day(date_string)
    else:
        pattern = r"([a-zA-Z0-9]+)_([a-zA-Z0-9]+)_(7DT\d{2})_(\d+\.?\d*)s_(\d{8}_\d{6})"
        match = re.match(pattern, path.stem)
        if match:
            target, filt, units, exposure, date_string = match.groups()
            date = subtract_half_day(date_string)

    info = {
        "date": date,
        "filter": filt,
        "obj": target,
        "unit": units,
        "exposure": exposure
    }

    file_type = "master_image" if any(s in str(path.stem) for s in ["BIAS", "DARK", "FLAT"]) else "sci_image"

    return info, file_type

from datetime import datetime, timedelta

def subtract_half_day(timestr: str) -> str:
    dt = datetime.strptime(timestr, "%Y%m%d_%H%M%S")
    new_dt = dt - timedelta(hours=12)
    return new_dt.strftime("%Y-%m-%d")