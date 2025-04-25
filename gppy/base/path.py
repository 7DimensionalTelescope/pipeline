from pathlib import Path
from .. import const
import numpy as np
from ..utils import parse_key_params_from_header, parse_key_params_from_filename
import re
import os
image_unique_keys = ["date", "gain", "filter", "obj", "unit", "n_binning", "exposure"]

class PathHandler:

    def __init__(self, input):
        if isinstance(input, str):
            self._input_file = [Path(input)]
        elif isinstance(input, Path):
            self._input_file = [input]
        elif isinstance(input, list):
            self._input_file = [Path(img) for img in input]
        else:
            raise TypeError("Input must be a string, Path, or list of strings/Paths.")
        
        if not(self.is_present(self.input_file)):
            raise FileNotFoundError(f"Path does not exist: {self.input_file}")
        
        self.check_params()

        self._datatype = self.check_datatype()
        
    def __getattr__(self, name):
        if name.startswith("__"):
            return None
        if hasattr(self, "_"+name) :
            return getattr(self, "_"+name)
        else:
            return getattr(self, name)
    
    def is_present(self, path):
        paths = np.atleast_1d(path)
        return all([p.exists() for p in paths])
        
    def check_datatype(self):
        if const.RAWDATA_DIR in str(self.input_file[0]):
            self._raw_image= [str(file.absolute()) for file in self.input_file]
            _tmp_relpath = os.path.join(self.date, self.obj, self.filter, self.unit)
            self._output_dir = Path(os.path.join(const.PROCESSED_DIR, _tmp_relpath))
            self._factory_dir = Path(os.path.join(const.FACTORY_DIR, _tmp_relpath))
            self._file_prefix = [
                switch_name_order(file.stem) for file in self.input_file
            ]
            self._processed_image = [
                os.path.join(self.output_dir, "images", file_prefix, ".fits") for file_prefix in self._file_prefix
            ]
            return "raw"
        elif const.PROCESSED_DIR in str(self.input_file[0]):
            self._processed_image = [str(file.absolute()) for file in self.input_file]
            self._file_prefix = [file.stem for file in self.input_file]
            
            #self._factory_dir = Path(os.path.join(const.FACTORY_DIR, self.date, self.obj, self.filter, self.unit))
            return "processed"
        else:
            return "user-input"
    
    def define_common_path(self):
        self._output_dir = self._processed_image[0].parent.parent

    def check_params(self):
        try:
            params = [parse_key_params_from_filename(img)[0] for img in self.input_file]
        except:
            try:
                params = [parse_key_params_from_header(img)[0] for img in self.input_file]
            except:
                raise ValueError("No parameters found in the image file names or headers.")
        if not params:
            raise ValueError("No parameters found in the image file names or headers.")
        def all_dicts_equal(lst):
            return all(d == lst[0] for d in lst)
        if not all_dicts_equal(params):
            raise ValueError("Not all images have the same parameters.")
        else:
            for key in image_unique_keys:
                setattr(self, "_"+key, params[0][key])

    def path_configuration(self):
        pass
  
    def path_preprocess(self):
        pass

    def path_astromety(self):
        pass

    def path_photometry(self):
        pass

    def path_stacking(self):
        pass



    def define_paths(self):
        """Create and set output directory paths for processed data."""
        _date_dir = define_output_dir(self.config.obs.date, self.config.obs.n_binning, self.config.obs.gain)

        rel_path = os.path.join(
            _date_dir,
            self.config.obs.object,
            self.config.obs.unit,
            self.config.obs.filter,
        )
        fdz_rel_path = os.path.join(
            _date_dir,
            self.config.obs.unit,
        )

        path_processed = os.path.join(self._output_prefix, rel_path)
        path_factory = os.path.join(FACTORY_DIR, rel_path)
        path_fdz = os.path.join(MASTER_FRAME_DIR, fdz_rel_path)
        path_stacked_prefix = STACKED_DIR if self.config.name == "user-input" else DAILY_STACKED_DIR
        path_stacked = os.path.join(path_stacked_prefix, rel_path)

        os.makedirs(path_processed, exist_ok=True)
        os.makedirs(path_fdz, exist_ok=True)
        os.makedirs(path_factory, exist_ok=True)
        # os.makedirs(path_stacked, exist_ok=True)  # make dir in imstack

        self.config.path.path_processed = path_processed
        self.config.path.path_stacked = path_stacked
        self.config.path.path_factory = path_factory
        self.config.path.path_fdz = path_fdz
        self.config.path.path_raw = find_raw_path(
            self.config.obs.unit,
            self.config.obs.date,
            self.config.obs.n_binning,
            self.config.obs.gain,
        )
        self.config.path.path_sex = os.path.join(REF_DIR, "srcExt")

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
