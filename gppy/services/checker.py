import json
from astropy.io import fits
from .. import const


class Checker:
    def __init__(self, dtype=None):
        self.dtype = dtype
        self.criteria = self.load_criteria()

    def load_criteria(self, dtype="masterframe"):
        try:
            if dtype.upper() in ["BIAS", "DARK", "FLAT", "MASTERFRAME"]:
                dtype = "masterframe"
            else:
                dtype = "science"
            criteria_file = const.REF_DIR + f"/qa/{dtype.lower()}.json"
            with open(criteria_file, "r") as f:
                self.criteria = json.load(f)
                return self.criteria
        except FileNotFoundError:
            raise RuntimeError(f"Criteria file not found: {criteria_file}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON in criteria file: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load criteria: {e}")

    def apply_criteria(self, file_path: str = None, header: dict = None, dtype: str = None):
        if dtype is None:
            if self.dtype is not None:
                dtype = self.dtype
            else:
                if "bias" in file_path:
                    dtype = "BIAS"
                elif "dark" in file_path:
                    dtype = "DARK"
                elif "flat" in file_path:
                    dtype = "FLAT"
                else:
                    dtype = "SCIENCE"

        if not (hasattr(self, "criteria")):
            self.load_criteria(dtype=dtype)

        criteria = self.criteria[dtype.upper()]
        print(file_path)

        flag = True

        if header is None:
            if file_path is not None:
                header = fits.getheader(file_path)
                print(header)
            else:
                raise ValueError("Either file_path or header must be provided")

        for key, value in criteria.items():
            if key not in header and dtype.upper() == "SCIENCE":
                continue

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
            elif value["criteria"] == "within":
                if header[key] < value["value"][0] or header[key] > value["value"][1]:
                    flag = False
                    break

        header["SANITY"] = (flag, "Sanity flag")

        return flag, header

    def sanity_check(self, file_path: str = None, header: dict = None, dtype: str = None):

        if file_path is not None:
            try:
                sanity = fits.getval(file_path, "SANITY")
                if sanity is not None:
                    return sanity
            except:
                pass
            return self.apply_criteria(file_path=file_path, dtype=dtype)[0]
        elif header is not None:
            pass
        else:
            raise ValueError("Either file_path or header must be provided")
