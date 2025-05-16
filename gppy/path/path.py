import os
from pathlib import Path
from typing import Union, TYPE_CHECKING
import numpy as np
from .. import const
from ..utils import check_params, add_suffix, swap_ext, collapse
from .name import NameHandler


if TYPE_CHECKING:
    from gppy.config import Configuration  # just for type hinting. actual import will cause circular import error


class AutoMkdirMixin:
    """This makes sure accessed dirs exist. Prepend _ to variables to prevent mkdir"""

    _mkdir_exclude = set()  # subclasses can override this

    def __init_subclass__(cls):
        # Ensure subclasses have their own created-directory cache
        cls._created_dirs_cache = set()

    def __getattribute__(self, name):
        """CAVEAT: This runs every time attr is accessed. Keep it short."""
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


class PathHandler(AutoMkdirMixin):
    def __init__(self, input: Union[str, Path, list, dict, "Configuration"] = None, *, working_dir=None):
        """Input homogeneous images"""
        self._config = None
        self._input_files = None
        self._data_type = None
        self.obs_params = {}

        if input is not None:
            self._handle_input(input)

        self.select_output_dir(working_dir=working_dir)

        self._file_indep_initialized = False
        self.define_info_agnostic_paths()
        self.define_file_independent_paths()

        self._file_dep_initialized = False
        if not self._file_dep_initialized and self._input_files:
            self.define_file_dependent_paths()

        # if self._file_indep_initialized and self._file_dep_initialized:
        self.define_operation_paths()

    def _handle_input(self, input):
        """init with obs_parmas and config are ad-hoc. Will be changed to always take filenames"""
        # input is obs_param
        if isinstance(input, dict):
            self.obs_params = input

        # input is ConfigurationInstance
        elif hasattr(input, "obs"):
            if input.file.raw_files is not None:
                self._input_files = [os.path.abspath(file) for file in input.file.raw_files]
            elif input.file.processed_files is not None:
                self._input_files = [os.path.abspath(file) for file in input.file.processed_files]
            self._data_type = input.name  # propagate user-input
            self.obs_params = input.obs.to_dict()
            self._config = input

        # input is Configuration
        elif hasattr(input, "config"):
            if input.config.file.raw_files is not None:
                self._input_files = [os.path.abspath(file) for file in input.config.file.raw_files]
            elif input.config.file.processed_files is not None:
                self._input_files = [os.path.abspath(file) for file in input.config.file.processed_files]
            self._data_type = input.config.name  # propagate user-input
            self.obs_params = input.config_in_dict["obs"]
            self._config = input.config

        # input is a fits file list; the only method to keep
        elif isinstance(input, list) or isinstance(input, str) or isinstance(input, Path):
            input = list(np.atleast_1d(input))
            self._names = NameHandler(input)
            self._input_files = [os.path.abspath(img) for img in input]
            self._data_type = self._names.types
            # self.obs_params = check_params(self._input_files[0])
            obs_params = collapse(self._names.to_dict(), keys=const.SCIENCE_GROUP_KEYS)
            if isinstance(obs_params, list):
                raise ValueError("PathHandler input is incoherent w.r.t. SCIENCE_GROUP_KEYS.")
            self.obs_params = obs_params

        else:
            raise TypeError(f"Input must be a path (str | Path), a list of paths, obs_params (dict), or Configuration.")

    def __getattr__(self, name):
        """
        Below runs when name is not in __dict__.
        (1) If file-dependent paths have not been built yet, build them.
        (2) Retry the lookup - if the attribute was created by the builder we
            return it; otherwise fall through to the convenience “_to_*” hooks.
        """
        # ---------- 1. Lazy initialization ----------
        if not self._file_dep_initialized and self._input_files:
            self._file_dep_initialized = True  # set the flag first to prevent accidental recursion
            try:
                self.define_file_dependent_paths()
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
            base = name[:-8]
            val = getattr(self, base)
            return Path(val) if isinstance(val, (Path, str)) else val

        if name.endswith("_collapse") or name.endswith("_squeeze") or name.endswith("_compact"):
            if name.endswith("_collapse"):
                base = name[:-9]
            else:
                base = name[:-8]
            val = getattr(self, base)
            if isinstance(val, list):
                return collapse(val)
            if isinstance(val, dict):
                return {k: collapse(v) for k, v in val.items() if isinstance(v, list)}
            else:
                return val

        # ---------- 3. Still not found -> real error ----------
        raise AttributeError(f"{self.__class__.__name__!s} has no attribute {name!r}")

    def __repr__(self):
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items() if not k.startswith("_"))

    def is_present(self, path):
        paths = np.atleast_1d(path)
        return all([Path(p).exists() for p in paths])

    def select_output_dir(self, working_dir=None):
        """
        CWD if user-input. Assume pipeline paths otherwise.
        obs_params check is ad-hoc
        """
        if self._input_files:
            _file_dir = str(Path(self._input_files[0]).absolute().parent)
            _pipeline_dirs = {const.RAWDATA_DIR, const.PROCESSED_DIR}
            _not_pipeline_dir = not any(s in _file_dir for s in _pipeline_dirs)
        else:
            _not_pipeline_dir = False

        # insufficient info or outside-pipeline input
        if not self.obs_params or working_dir or _not_pipeline_dir:
            working_dir = working_dir or (os.path.dirname(self._input_files[0]) if self._input_files else os.getcwd())

            self._output_parent_dir = working_dir
            tmp_dir = os.path.join(working_dir, "tmp")
            self.factory_parent_dir = tmp_dir
            self.factory_dir = tmp_dir
            self._assume_pipeline = False

        else:
            from datetime import date

            datestring = self.obs_params.get("nightdate") or date.today().strftime("%Y%m%d")
            if datestring < "20260101":
                self._output_parent_dir = const.PROCESSED_DIR
                self.factory_parent_dir = const.FACTORY_DIR
            else:
                raise ValueError("nightdate cap reached: consider moving to another disk.")
            self._assume_pipeline = True

    @property
    def output_parent_dir(self):
        return self._output_parent_dir

    @property
    def file_dep_initialized(self):
        """Safe from AutoMkdirMixin as it's a bool."""
        return self._file_dep_initialized

    @property
    def assume_pipeline(self):
        return self._assume_pipeline

    def define_info_agnostic_paths(self):
        self.ref_sex_dir = os.path.join(const.REF_DIR, "srcExt")
        self.base_yml = os.path.join(const.REF_DIR, "base.yml")
        self.output_yml = os.path.join(
            self._output_parent_dir, "config.yml"
        )  # overridden in define_file_independent_paths()
        # self.imstack_base_yml
        # self.phot_base_yml

    def define_file_independent_paths(self):

        if self._assume_pipeline:  # and self.obs_params and self.obs_params.get("nightdate"):
            _relative_path = os.path.join(
                self.obs_params["nightdate"], self.obs_params["obj"], self.obs_params["filter"], self.obs_params["unit"]
            )
            self.output_dir = os.path.join(self.output_parent_dir, _relative_path)
            self.factory_dir = os.path.join(self.factory_parent_dir, _relative_path)
            self.metadata_dir = os.path.join(self.output_parent_dir, self.obs_params["nightdate"])

            # directories
            self.image_dir = os.path.join(self.output_dir, "images")
            self.daily_stacked_dir = os.path.join(self.output_dir, "stacked")
            self.subtracted_dir = os.path.join(self.output_dir, "subtracted")

            # files
            self._output_name = f"{self.obs_params['obj']}_{self.obs_params['filter']}_{self.obs_params['unit']}_{self.obs_params['nightdate']}"
            self.output_yml = os.path.join(self.output_dir, self._output_name + ".yml")
            self.output_log = os.path.join(self.output_dir, self._output_name + ".log")

        else:
            self.output_dir = self.output_parent_dir
            self.factory_dir = self.factory_parent_dir

        self.figure_dir = os.path.join(self.output_dir, "figures")

        self._file_indep_initialized = True

    def add_fits(self, files: str | Path | list):
        if isinstance(files, list):
            self._input_files = [str(f) for f in files]
        else:
            self._input_files = str(files)

    def define_file_dependent_paths(self):
        """use utils.Path7DS for bidirectional handling"""

        names = NameHandler(self._input_files)

        if self._assume_pipeline:
            if not (self.is_present(self._input_files)):
                raise FileNotFoundError(f"Not all input paths exist: {self._input_files}")

            if const.RAWDATA_DIR in str(self._input_files[0]):  # raw pipeline input
                # self.data_type = self._data_type or "raw"  # interferes with Mkdir

                self.raw_images = self._input_files  # unnecessary

                self.masterframe_dir = os.path.join(
                    f"{const.MASTER_FRAME_DIR}",
                    f"{self.obs_params['nightdate']}",
                    self.obs_params["unit"],
                )
                if names._single:
                    self.processed_images = os.path.join(self.output_dir, names.conjugate)
                else:
                    self.processed_images = [os.path.join(self.output_dir, f) for f in names.conjugate]

            elif self.output_parent_dir in str(self._input_files[0]):  # processed pipeline input
                # self.data_type = self._data_type or "processed"
                # self.processed_images = [str(file.absolute()) for file in self._input_files]
                self.factory_dir = os.path.join(const.FACTORY_DIR, *Path(self._input_files[0]).parts[-6:-3])
                self.output_dir = str(Path(self._input_files[0]).parent.parent)

            else:  # user input
                # self.data_type = self._data_type or "user-input"
                print("User input data type detected. Assume the input is a list of processed images.")
                self.processed_images = [str(file.absolute()) for file in self._input_files]
                # self.processed_file_stems = [file.stem for file in self._input_files]
                self.output_dir = str(Path(self._input_files[0]).parent.parent)
                self.factory_dir = str(Path(self._input_files[0]).parent.parent / "factory")
        else:
            pass

        self._file_dep_initialized = True

    def define_operation_paths(self):
        self.preprocess = PathPreprocess(self, self._config)
        self.astrometry = PathAstrometry(self, self._config)
        self.photometry = PathPhotometry(self, self._config)
        self.imstack = PathImstack(self, self._config)
        self.imsubtract = PathImsubtract(self, self._config)

    @property
    def conjugate(self) -> str | list[str]:
        names = NameHandler(self._input_files)
        basenames = names.conjugate
        types = names.types

        if not isinstance(basenames, list):  # single path
            basenames, types, single = [basenames], [types], True
        else:  # list of paths
            single = False

        paths = []
        for bn, t in zip(basenames, types):
            if t == "raw_image":
                # original was raw → conjugate is processed
                root = self.image_dir
            else:
                # original was processed → conjugate is raw
                root = const.RAWDATA_DIR
            paths.append(os.path.abspath(os.path.join(root, bn)))

        return paths[0] if single else paths

    @classmethod
    def from_grouped_files(cls, on_date_calib):
        from collections import Counter

        triples = [
            (tuple(on_date_bias), tuple(on_date_dark), tuple(on_date_flat))
            for flag, on_date_bias, on_date_dark, on_date_flat in on_date_calib
            if flag == True
        ]
        counts = Counter(triples)

        _tmp_list = []
        # in ascending order of count
        for (bias_files, dark_files, flat_files), cnt in sorted(counts.items(), key=lambda item: item[1]):
            _tmp_list.append(
                (
                    PathHandler(list(bias_files)).preprocess.bias,
                    PathHandler(list(dark_files)).preprocess.dark,
                    PathHandler(list(flat_files)).preprocess.flat,
                )
            )

        return _tmp_list


class PathPreprocess(AutoMkdirMixin):
    def __init__(self, parent: PathHandler, config=None):
        self._parent = parent

        # Apply config overrides if provided
        if config and hasattr(config, "path"):
            for key, val in config.preprocess.path.items():
                setattr(self, key, val)

    def __repr__(self):
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items() if not k.startswith("_"))

    @property
    def bias(self):
        names = NameHandler(self._parent._input_files)
        return collapse(
            [
                os.path.join(self._parent.masterframe_dir, s)
                for typ, s in zip(names.types, names.masterframe_basename)
                if typ[1] == "bias"
            ]
        )  # it may not be collapsed if input is incoherent

    @property
    def dark(self):
        names = NameHandler(self._parent._input_files)
        return collapse(
            [
                os.path.join(self._parent.masterframe_dir, s)
                for typ, s in zip(names.types, names.masterframe_basename)
                if typ[1] == "dark"
            ]
        )

    @property
    def flat(self):
        names = NameHandler(self._parent._input_files)
        return collapse(
            [
                os.path.join(self._parent.masterframe_dir, s)
                for typ, s in zip(names.types, names.masterframe_basename)
                if typ[1] == "flat"
            ]
        )


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

    def __repr__(self):
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items() if not k.startswith("_"))

    @property
    def tmp_dir(self):
        return os.path.join(self._parent.factory_dir, "astrometry")

    @property
    def input(self):
        return self._parent.processed_images


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

    def __repr__(self):
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items() if not k.startswith("_"))

    @property
    def tmp_dir(self):
        return os.path.join(self._parent.factory_dir, "photometry")

    @property
    def prep_catalog(self):
        input = self._parent._input_files
        if isinstance(input, list):
            return [os.path.join(self.tmp_dir, swap_ext(add_suffix(s, "prep"), "cat")) for s in input]
        else:
            return os.path.join(self.tmp_dir, swap_ext(add_suffix(input, "prep"), "cat"))

    @property
    def main_catalog(self):
        """intermediate sextractor output"""
        input = self._parent._input_files
        if isinstance(input, list):
            return [os.path.join(self.tmp_dir, swap_ext(s, "cat")) for s in input]
        else:
            return os.path.join(self.tmp_dir, swap_ext(input, "cat"))

    @property
    def final_catalog(self):
        """final pipeline output catalog"""
        input = self._parent._input_files
        if isinstance(input, list):
            return [os.path.join(self.tmp_dir, add_suffix(s, "cat")) for s in input]
        else:
            return os.path.join(self.tmp_dir, add_suffix(input, "cat"))

    def __getattr__(self):
        # run file-dependent path definitions once?
        pass


class PathImstack(AutoMkdirMixin):
    def __init__(self, parent: PathHandler, config=None):
        self._parent = parent

        # Apply config overrides if provided
        if config and hasattr(config, "path"):
            for key, val in config.imstack.path.items():
                setattr(self, key, val)

    def __repr__(self):
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items() if not k.startswith("_"))

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

    def __repr__(self):
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items() if not k.startswith("_"))

    @property
    def tmp_dir(self):
        return os.path.join(self._parent.factory_dir, "imsubtract")
