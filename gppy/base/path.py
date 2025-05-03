import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union, TYPE_CHECKING
import numpy as np


# from ..utils import parse_key_params_from_header, parse_key_params_from_filename, get_camera, define_output_dir, Path7DS
from ..utils import check_params
from .. import const

if TYPE_CHECKING:
    from gppy.config import Configuration  # just for type hinting. actual import will cause circular import error


class AutoMkdirMixin:
    _mkdir_exclude = set()  # subclasses can override this

    def __getattribute__(self, name):
        """
        CAVEAT: This runs every time attr is accessed. Keep it short.
        Make sure accessed dirs exist
        """
        if name.startswith("_"):  # Bypass all custom logic for private attributes
            return object.__getattribute__(self, name)

        value = object.__getattribute__(self, name)

        # Skip excluded attributes
        if name in object.__getattribute__(self, "_mkdir_exclude"):
            return value

        if isinstance(value, (str, Path)):
            self._mkdir(value)
        elif isinstance(value, list) and all(isinstance(p, (str, Path)) for p in value):
            for p in value:
                self._mkdir(p)

        return value

    def _mkdir(self, value):
        p = Path(value).expanduser()  # understands ~/
        d = p.parent if p.suffix else p  # ensure directory

        # Use mixin's own per-instance cache
        created_dirs = object.__getattribute__(self, "_created_dirs_cache")

        if d not in created_dirs and not d.exists():  # check cache first for performance
            d.mkdir(parents=True, exist_ok=True)
            created_dirs.add(d)

    def __init_subclass__(cls):
        # Ensure subclasses have their own directory cache
        cls._created_dirs_cache = set()


class PathHandler(AutoMkdirMixin):
    def __init__(self, input: Union[str, Path, list, dict, "Configuration"], working_dir=None):
        self._config = None

        # input is obs_param
        if isinstance(input, dict):
            self._input_file = None
            self._data_type = None
            self.obs_params = input

        # input is config
        elif hasattr(input, "config"):
            self._input_file = [Path(file) for file in input.config.file.raw_files]

            self._data_type = input.config.name  # propagate user-input
            self.obs_params = input.config_in_dict["obs"]
            self._config = input.config

        # input is a fits file
        elif isinstance(input, str) or isinstance(input, Path):
            self._input_file = [Path(input)]
            self._data_type = None
            self.obs_params = check_params(self._input_file)

        # input is a fits file list
        elif isinstance(input, list):
            self._input_file = [Path(img) for img in input]
            self._data_type = None
            self.obs_params = None
            self.obs_params = check_params(self._input_file[0])

        else:
            raise TypeError(f"Input must be a path (str | Path), a list of paths, obs_params (dict), or Configuration.")

        # self._file_indep_initialized = False
        self._file_dep_initialized = False
        self.select_output_dir(working_dir=working_dir)

        # if not self._file_indep_initialized:
        self.define_file_independent_common_paths()

        if not self._file_dep_initialized and self._input_file:
            self.define_file_dependent_common_paths()

        # if self._file_indep_initialized and self._file_dep_initialized:
        self.define_specific_paths()

    def __getattr__(self, name):
        """
        Below runs when name is not in __dict__.
        (1) If file-dependent paths have not been built yet, build them.
        (2) Retry the lookup - if the attribute was created by the builder we
            return it; otherwise fall through to the convenience “_to_*” hooks.
        """
        # ---------- 1. Lazy initialization ----------
        if not self._file_dep_initialized and self._input_file:
            self._file_dep_initialized = True  # set the flag first to prevent accidental recursion
            try:
                self.define_file_dependent_common_paths()
            except Exception:
                # roll back if the builder blew up
                self._file_dep_initialized = False
                raise

            # after building, see whether that gave us the requested attr
            if name in self.__dict__:
                return self.__dict__[name]

        # ---------- 2. “Syntactic-sugar” logic ----------
        if name.endswith("_to_string"):
            base = name[:-10]
            val = getattr(self, base)
            return str(val) if isinstance(val, (Path, str)) else val

        if name.endswith("_to_path"):
            base = name[:-7]
            val = getattr(self, base)
            return Path(val) if isinstance(val, str) else val

        # ---------- 3. Still not found → real error ----------
        raise AttributeError(f"{self.__class__.__name__!s} has no attribute {name!r}")

    def __repr__(self):
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items() if not k.startswith("_"))

    def is_present(self, path):
        paths = np.atleast_1d(path)
        return all([p.exists() for p in paths])

    def select_output_dir(self, working_dir=None):
        # date_utc = Path7DS(self._input_files[0]).date  # utc date. e.g., "20250101"
        # date_folder = check_params(self._input_file[0])["nightdate"]
        if working_dir:
            self.output_parent_dir = working_dir
            self.factory_parent_dir = os.path.join(working_dir, "factory")
        else:
            if self.obs_params["nightdate"] < "20260101":
                self.output_parent_dir = const.PROCESSED_DIR
                self.factory_parent_dir = const.FACTORY_DIR
            else:
                raise ValueError("Predefined date cap reached: consider moving to another disk.")

    def define_file_independent_common_paths(self):

        _relative_path = os.path.join(
            self.obs_params["nightdate"], self.obs_params["obj"], self.obs_params["filter"], self.obs_params["unit"]
        )
        self.output_dir = os.path.join(self.output_parent_dir, _relative_path)
        self.factory_dir = os.path.join(self.factory_parent_dir, _relative_path)
        self.metadata_dir = os.path.join(self.output_parent_dir, self.obs_params["nightdate"])

        # directories
        self.image_dir = os.path.join(self.output_dir, "images")
        self.figure_dir = os.path.join(self.output_dir, "figures")
        self.daily_stacked_dir = os.path.join(self.output_dir, "stacked")
        self.subtracted_dir = os.path.join(self.output_dir, "subtracted")
        self.ref_sex_dir = os.path.join(const.REF_DIR, "srcExt")

        # files
        self.base_yml = os.path.join(const.REF_DIR, "base.yml")
        self.output_name = f"{self.obs_params['obj']}_{self.obs_params['filter']}_{self.obs_params['unit']}_{self.obs_params['nightdate']}"
        self.output_yml = os.path.join(self.output_dir, self.output_name + ".yml")
        self.output_log = os.path.join(self.output_dir, self.output_name + ".log")

        self._file_indep_initialized = True

    def add_fits(self, files: str | Path | list):
        if isinstance(files, list):
            self._input_file = [Path(f) for f in files]
        else:
            self._input_file = Path(files)

    def define_file_dependent_common_paths(self):
        if not (self.is_present(self._input_file)):
            raise FileNotFoundError(f"Not all paths exist: {self._input_file}")

        if const.RAWDATA_DIR in str(self._input_file[0]):
            self.data_type = self._data_type or "raw"
            self.raw_images = [str(file.absolute()) for file in self._input_file]
            self.masterframe_dir = os.path.join(
                f"{const.MASTER_FRAME_DIR}",
                f"{self.obs_params['nightdate']}_{self.obs_params['n_binning']}x{self.obs_params['n_binning']}_gain{self.obs_params['gain']}",
                self.obs_params["unit"],
            )
            processed_file_stems = [switch_raw_name_order(file.stem) for file in self._input_file]
            self.processed_images = [
                os.path.join(self.output_dir, "images", file_stem + ".fits") for file_stem in processed_file_stems
            ]
        elif self.output_parent_dir in str(self._input_file[0]):
            self.data_type = self._data_type or "processed"
            self.processed_images = [str(file.absolute()) for file in self._input_file]
            # self.processed_file_stems = [file.stem for file in self._input_files]
            self.factory_dir = os.path.join(const.FACTORY_DIR, *Path(self._input_file[0]).parts[-6:-3])
            self.output_dir = self._input_file[0].parent.parent
        else:
            self.data_type = self._data_type or "user-input"
            print("User input data type detected. Assume the input is a list of processed images.")
            self.processed_images = [str(file.absolute()) for file in self._input_file]
            # self.processed_file_stems = [file.stem for file in self._input_files]
            self.output_dir = self._input_file[0].parent.parent
            self.factory_dir = self._input_file[0].parent.parent / "factory"

        self._file_dep_initialized = True
        pass

    def define_specific_paths(self):
        self.astrometry = PathAstrometry(self, self._config)
        self.photometry = PathPhotometry(self)
        self.imstack = PathImstack(self)
        self.imsubtract = PathImsubtract(self)


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
    _spec = {
        "mbias_link": lambda self: self.path_fdz / f"bias_{self._date}_{self._cam}.link",
        "mdark_link": lambda self: self.path_fdz / f"dark_{self._date}_{self._exp}s_{self._cam}.link",
        "mflat_link": lambda self: self.path_fdz / f"flat_{self._date}_{self._filt}_{self._cam}.link",
    }

    def __init__(self, path_processed):
        # _date_dir = define_output_dir(self.config.obs.date, self.config.obs.n_binning, self.config.obs.gain)
        _date_dir = os.path.join(self.config.obs.date, self.config.obs.n_binning, self.config.obs.gain)
        self.path_fdz = Path(_date_dir) / self.config.obs.unit


class PathAstrometry(AutoMkdirMixin):
    _mkdir_exclude = {"ref_ris_dir", "ref_query_dir"}

    def __init__(self, parent: PathHandler, config=None):
        self._parent = parent

        # Default values
        self.ref_ris_dir = "/lyman/data1/factory/catalog/gaia_dr3_7DT"
        self.ref_query_dir = "/lyman/data1/factory/ref_scamp"

        # Apply config overrides if provided
        if config and hasattr(config, "path"):
            for key, val in config.astrometry.path.items():
                setattr(self, key, val)

    # def __repr__(self):
    #     return self.tmp_dir

    @property
    def tmp_dir(self):
        return os.path.join(self._parent.factory_dir, "astrometry")


class PathPhotometry(AutoMkdirMixin):
    _mkdir_exclude = {"ref_ris_dir", "ref_gaia_dir"}

    def __init__(self, parent: PathHandler, config=None):
        self._parent = parent

        self.ref_ris_dir = "/lyman/data1/factory/ref_cat"  # divided by RIS tiles
        self.ref_gaia_dir = "/lyman/data1/Calibration/7DT-Calibration/output/Calibration_Tile"

        # Apply config overrides if provided
        if config and hasattr(config, "path"):
            for key, val in config.photometry.path.items():
                setattr(self, key, val)

    @property
    def tmp_dir(self):
        return os.path.join(self._parent.factory_dir, "photometry")


class PathImstack(AutoMkdirMixin):
    def __init__(self, parent: PathHandler):
        self._parent = parent

    @property
    def tmp_dir(self):
        return os.path.join(self._parent.factory_dir, "imstack")


class PathImsubtract(AutoMkdirMixin):
    _mkdir_exclude = {"ref_image_dir"}

    def __init__(self, parent: PathHandler, config=None):
        self._parent = parent

        self.ref_image_dir = "/lyman/data1/factory/ref_frame"

        # Apply config overrides if provided
        if config and hasattr(config, "path"):
            for key, val in config.imsubtract.path.items():
                setattr(self, key, val)

    @property
    def tmp_dir(self):
        return os.path.join(self._parent.factory_dir, "imsubtract")
