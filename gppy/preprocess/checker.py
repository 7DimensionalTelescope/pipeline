import json
from astropy.io import fits

class Checker:
    def __init__(self):
        self.criteria = self.load_criteria()

    def load_criteria(self):
        try:
            criteria_file = __file__.replace('checker.py', 'masterframe_criteria.json')
            with open(criteria_file, 'r') as f:
                self.criteria = json.load(f)
                return self.criteria
        except FileNotFoundError:
            raise RuntimeError(f"Criteria file not found: {criteria_file}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON in criteria file: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load criteria: {e}")

    def apply_criteria(self, file_path: str=None, header: dict=None, dtype: str=None):
        if dtype is None:
            if "bias" in file_path:
                dtype = "BIAS"
            elif "dark" in file_path:
                dtype = "DARK"
            elif "flat" in file_path:
                dtype = "FLAT"
            else:
                raise ValueError(f"Unknown dtype: {file_path}")
                
        if not(hasattr(self, "criteria")):
            self.load_criteria()
        
        criteria = self.criteria[dtype.upper()]
        
        flag = True

        if header is None:
            if file_path is not None:
                header = fits.getheader(file_path)
            else:
                raise ValueError("Either file_path or header must be provided")

        for key, value in criteria.items():
            if value["criteria"] == "neq":
                if header[key] == value["value"]:
                    flag = False
                    break
            elif value["criteria"] == "eq":
                if header[key] != value["value"]:
                    flag = False
                    break
            elif value["criteria"] == "gte":
                if header[key] < value["value"]:
                    flag = False
                    break 
            elif value["criteria"] == "gt":
                if header[key] <= value["value"]:
                    flag = False
                    break
            elif value["criteria"] == "lte":
                if header[key] > value["value"]:
                    flag = False
                    break
            elif value["criteria"] == "lt":
                if header[key] >= value["value"]:
                    flag = False
                    break
        header["SANITY"] = flag

        return flag, header

    def sanity_check(self, file_path: str=None, header: dict=None, dtype: str=None):
        if file_path is not None:
            header = fits.getheader(file_path)
        elif header is not None:
            pass
        else:
            raise ValueError("Either file_path or header must be provided")

        if "SANITY" in header:
            return header["SANITY"]
        else:
            return self.apply_criteria(file_path, dtype)[0]