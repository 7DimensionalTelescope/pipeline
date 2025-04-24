from .decorator import autoproperties
from pathlib import Path
from .. import const
import numpy as np

@autoproperties
class PathHandler:

    def __init__(self, input):
        if isinstance(input, str):
            self._base_path = [Path(input)]
        elif isinstance(input, Path):
            self._base_path = [input]
        elif isinstance(input, list):
            self._base_path = [Path(img) for img in input]
        else:
            raise TypeError("Input must be a string, Path, or list of strings/Paths.")
        
        if not(self.is_present(self.base_path)):
            raise FileNotFoundError(f"Path does not exist: {self.base_path}")

        self._datatype = self.check_datatype()

    def is_present(self, path):
        paths = np.atleast_1d(path)
        for path in paths:
            if not path.exists():
                return False
        return True
    
    def check_datatype(self):
        if const.RAWDATA_DIR in str(self.base_path[0]):
            return "raw"
        elif const.PROCESSED_DIR in str(self.base_path[0]):
            return "processed"
        else:
            return "user-input"
