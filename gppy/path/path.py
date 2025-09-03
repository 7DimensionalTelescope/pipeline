import os
from glob import glob
from pathlib import Path
from typing import Union
import numpy as np
from .. import const
from .utils import find_raw_path
from ..utils import add_suffix, swap_ext, collapse, atleast_1d
from .name import NameHandler
from .utils import format_exptime
from .utils import broadcast_join_pure as bjoin


class SingletonUnpackMixin:
    """Automatically unpacks singleton lists when _single is True"""

    def __getattribute__(self, name):
        if name.startswith("_"):
            return object.__getattribute__(self, name)

        value = object.__getattribute__(self, name)

        # Unpack singleton lists if _single is True
        if (
            "_single" in self.__dict__
            and object.__getattribute__(self, "_single")
            and isinstance(value, list)
            and len(value) == 1
        ):
            value = value[0]

        return value


class AutoCollapseMixin:
    """Automatically collapses the output when it is a list of uniformly
    releated elemements"""

    # Define which attributes should be collapsed
    _collapse_exclude = {}  # "output_name", "name", "preprocess"}
    _collapse_include = {"masterframe"}  # "output_dir", "image_dir", "factory_dir", "stacked_dir"}

    def __getattribute__(self, name):
        # if name.startswith("_"):
        #     return object.__getattribute__(self, name)

        # value = object.__getattribute__(self, name)
        value = super().__getattribute__(name)

        if name.startswith("_"):
            return value

        # print("collapse", name, value)

        # Collapse if explicitly included or path-like list
        if name not in self._collapse_exclude and (
            name in self._collapse_include
            or (isinstance(value, list) and all(isinstance(v, (str, Path)) for v in value))
        ):
            # print("being collapsed", name, value)
            return collapse(value)

        return value

    # def __getattribute__(self, name):
    #     if name.startswith("_"):
    #         return object.__getattribute__(self, name)

    #     value = object.__getattribute__(self, name)
    #     # Only collapse specific attributes or path-like lists
    #     should_collapse = (
    #         (name in getattr(self, "_collapse_include", set()) and name not in self._collapse_exclude)
    #         or (isinstance(value, list) and all(isinstance(p, (str, Path)) for p in value))
    #         or (isinstance(value, list) and "_yml" in name)
    #         or (isinstance(value, list) and "_dir" in name)
    #         or (isinstance(value, list) and "_log" in name)
    #     )

    #     if should_collapse:
    #         return collapse(value)

    #     return value


class AutoMkdirMixin:
    """This makes sure accessed dirs exist. Prepend _ to variables to prevent mkdir"""

    _mkdir_exclude = {"output_name", "config_stem", "name", "changelog_dir"}  # subclasses can override this

    def __init_subclass__(cls):
        # Ensure subclasses have their own created-directory cache
        cls._created_dirs_cache = set()

    def __getattribute__(self, name):
        """CAVEAT: This runs every time attr is accessed. Keep it short."""
        # if name.startswith("_"):  # Bypass all custom logic for private attributes
        #     return object.__getattribute__(self, name)

        # value = object.__getattribute__(self, name)

        value = super().__getattribute__(name)

        if name.startswith("_"):
            return value

        # print("mkdir", name)

        # Skip excluded attributes
        if name in object.__getattribute__(self, "_mkdir_exclude"):
            return value

        # Handle vectorized paths
        if isinstance(value, list) and all(isinstance(p, (str, Path)) for p in value):
            for p in value:
                self._mkdir(p)
        elif isinstance(value, (str, Path)):
            self._mkdir(value)

        return value

    # def __getattr__(self, name):
    #     if name.startswith("_"):  # Bypass all custom logic for private attributes
    #         return object.__getattribute__(self, name)

    #     value = object.__getattribute__(self, name)

    #     if name in object.__getattribute__(self, "_mkdir_exclude"):
    #         return value

    #     if isinstance(value, list) and all(isinstance(p, (str, Path)) for p in value):
    #         for p in value:
    #             self._mkdir(p)
    #     elif isinstance(value, (str, Path)):
    #         self._mkdir(value)

    #     return value

    def _mkdir(self, value):
        p = Path(value).expanduser()  # understands ~/
        d = p.parent if p.suffix else p  # ensure directory

        # Use mixin's own per-instance cache
        created_dirs = object.__getattribute__(self, "_created_dirs_cache")

        if d not in created_dirs and not d.exists():  # check cache first for performance
            d.mkdir(parents=True, exist_ok=True)
            created_dirs.add(d)


# AutoMkdirMixin super() allows AutoCollapseMixin to run first.
class PathHandler(AutoMkdirMixin, AutoCollapseMixin):  # SingletonUnpackMixin, Check MRO: PathHandler.mro()
    """
    A comprehensive path handler for 7DT pipeline.
    It defines source and destination file paths in all stages of the pipeline
    for the given input images, their related paths, reference directores and
    files, and related products like the weight images and photometric catalogs.

    It enables access to all attributes to NameHandler properties with optimized
    caching and lazy initialization. It also provides convenient syntactic sugar
    such as `output_dir_to_string`, `output_dir_to_path`, etc.

    Currently lacks masterframe support
    """

    def __init__(self, input: Union[str, Path, list[str | Path]] = None, *, working_dir=None):
        self._name_cache = {}  # Cache for NameHandler properties
        self._file_indep_initialized = False
        self._file_dep_initialized = False
        self._config = None
        self._config = None
        self._input_files: list[str] = None

        self._handle_input(input)
        self.select_output_dir(working_dir=working_dir)

        self.define_file_independent_paths()

        if not self._file_dep_initialized and self._input_files:
            self.define_file_dependent_paths()

        # if self._file_indep_initialized and self._file_dep_initialized:
        self.define_operation_paths()

    def _handle_input(self, input):
        """init with obs_parmas and config are ad-hoc. Will be changed to always take filenames"""

        if input is None:
            return
        # elif isinstance(input, list) or isinstance(input, (str, Path)):

        # Normalize input to list
        if isinstance(input, list):
            pass
        elif isinstance(input, (str, Path)):
            input = [input]
        elif isinstance(input, np.ndarray):
            input = list(input)
        else:
            raise ValueError("Invalid PathHandler input type.")

        self._input_files = [os.path.abspath(img) for img in input]
        if input:
            try:
                self.name = NameHandler(input)
            except Exception as e:
                raise ValueError(f"NameHandler failure: not pipeline file.\n{input!r}:\n{e}")
        self._single = self.name._single

    def __getattr__(self, name):
        """
        Below runs when name is not in __dict__.
        (1) If file-dependent paths have not been built yet, build them.
        (2) Retry the lookup - if the attribute was created by the builder we
            return it; otherwise fall through to the convenience “_to_*” hooks.
        """
        # Check if it's a NameHandler property and cache it
        nh = self.__dict__.get("name", None)
        if nh and hasattr(nh, name):
            return self._get_cached_namehandler_property(name)

        # Lazy initialization
        # if not self._file_dep_initialized and self._input_files:
        #     self._file_dep_initialized = True  # set the flag first to prevent accidental recursion
        #     try:
        #         self.define_file_dependent_paths()
        #     except Exception:
        #         # roll back if the builder blew up
        #         self._file_dep_initialized = False
        #         raise

        #     if name in self.__dict__:
        #         # Temporarily return uniform list as a scalar
        #         returned_attr = self.__dict__[name]
        #         if isinstance(returned_attr, list):
        #             if all(val == returned_attr[0] for val in returned_attr):
        #                 return returned_attr[0]
        #         return returned_attr

        # Syntactic sugar for vectorized results
        if name.endswith("_to_string"):
            base = name[:-10]
            val = getattr(self, base)
            return self._vectorize_conversion(val, str)

        if name.endswith("_to_path"):
            base = name[:-8]
            val = getattr(self, base)
            return self._vectorize_conversion(val, Path)

        if name.endswith("_collapse") or name.endswith("_squeeze") or name.endswith("_compact"):
            suffix_len = 9 if name.endswith("_collapse") else 8
            base = name[:-suffix_len]
            val = getattr(self, base)
            return collapse(val)

        raise AttributeError(f"{self.__class__.__name__!s} has no attribute {name!r}")

    def _vectorize_conversion(self, val, converter):
        """Apply converter to vectorized values"""
        if isinstance(val, list):
            return [converter(v) if isinstance(v, (Path, str)) else v for v in val]
        elif isinstance(val, (Path, str)):
            return converter(val)
        return val

    # def __repr__(self):
    #     return "\n".join(f"{k}: {v}" for k, v in sorted(self.__dict__.items()) if not k.startswith("_"))

    def __repr__(self):
        lines = []
        for k, v in sorted(self.__dict__.items()):
            if not k.startswith("_"):
                try:
                    v = collapse(v)
                except:
                    pass
                lines.append(f"{k}: {v}")

            # if k.startswith("_") and not k.startswith("__"):
            #     public_name = k[1:]
            #     # if public_name not in self.__dict__:
            #     try:
            #         collapsed = collapse(v)
            #         lines.append(f"{public_name}: {collapsed}")
            #     except Exception:
            #         pass  # Skip if collapse fails
        return "\n".join(lines)

    def is_present(self, path):
        """Check if vectorized paths exist"""
        if isinstance(path, list):
            return [Path(p).exists() for p in path]
        else:
            return Path(path).exists()

    def select_output_dir(self, working_dir=None):
        """
        Vectorized output directory selection
        CWD if user-input. Assume pipeline paths otherwise.
        """
        if not self._input_files:
            self._output_parent_dir = [working_dir or os.getcwd()]
            self._preproc_output_dir = self._output_parent_dir
            self._factory_parent_dir = [os.path.join(self._output_parent_dir[0], "tmp")]
            self._is_pipeline = [False]

        else:
            # Process each file independently
            self._output_parent_dir = []
            self._preproc_output_dir = []
            self._factory_parent_dir = []
            self._is_pipeline = []

            for i, input_file in enumerate(self._input_files):
                file_dir = str(Path(input_file).absolute().parent)
                not_pipeline_dir = not any(s in file_dir for s in const.PIPELINE_DIRS)

                if working_dir or not_pipeline_dir:
                    output_parent_dir = working_dir or os.path.dirname(input_file)
                    self._output_parent_dir.append(output_parent_dir)
                    self._factory_parent_dir.append(os.path.join(output_parent_dir, "tmp"))
                    self._is_pipeline.append(False)
                else:
                    from datetime import date

                    nightdate = self._get_name_property_at_index("nightdate", i)
                    if isinstance(nightdate, list):
                        current_nightdate = nightdate[i] if i < len(nightdate) else nightdate[0]
                    else:
                        current_nightdate = nightdate or date.today().strftime("%Y%m%d")

                    if current_nightdate < "20260101":
                        output_parent_dir = const.PROCESSED_DIR
                        self._output_parent_dir.append(output_parent_dir)
                        self._factory_parent_dir.append(const.FACTORY_DIR)
                        self._is_pipeline.append(True)
                    else:
                        raise ValueError(
                            f"nightdate cap reached for file {input_file}: consider moving to another disk."
                        )

                preproc_output_dir = os.path.join(output_parent_dir, self._get_name_property_at_index("nightdate", i))
                self._preproc_output_dir.append(preproc_output_dir)

        # Store as lists
        self.output_parent_dir = self._output_parent_dir
        self.preproc_output_dir = self._preproc_output_dir
        self.factory_parent_dir = self._factory_parent_dir

    def _get_cached_namehandler_property(self, prop_name: str):
        """
        Get NameHandler property with lazy caching.
        First access fetches from NameHandler and caches it.
        Subsequent accesses return cached value.

        returns singleton list for single-file case
        """
        if prop_name not in self._name_cache:
            if not hasattr(self, "name") or self.name is None:
                return None

            # Get property from NameHandler for the first time
            prop_value = getattr(self.name, prop_name, None)

            # Normalize to list for consistent indexing
            if self._input_files and prop_value is not None:
                if isinstance(prop_value, list):
                    # Ensure list length matches input files
                    if len(prop_value) == 1 and len(self._input_files) > 1:
                        # Single value for multiple files - broadcast
                        self._name_cache[prop_name] = prop_value * len(self._input_files)
                    else:
                        self._name_cache[prop_name] = prop_value
                else:
                    # Single value - broadcast to all files
                    self._name_cache[prop_name] = [prop_value] * len(self._input_files)
            else:
                # No input files or None value
                self._name_cache[prop_name] = prop_value

        return self._name_cache[prop_name]

    def _get_name_property_at_index(self, prop_name: str, index: int):
        """Get cached NameHandler property value at specific index. Can handle single input case"""
        cached_prop = self._get_cached_namehandler_property(prop_name)

        if cached_prop is None:
            return None
        elif isinstance(cached_prop, list):
            return cached_prop[index] if index < len(cached_prop) else cached_prop[0]
        else:
            return cached_prop

    @property
    def file_dep_initialized(self):
        """Safe from AutoMkdirMixin as it's a bool."""
        return self._file_dep_initialized

    def define_file_independent_paths(self):
        self.ref_sex_dir = os.path.join(const.REF_DIR, "srcExt")

        self.sciproc_base_yml = os.path.join(const.REF_DIR, "sciproc_base.yml")
        self.preproc_base_yml = os.path.join(const.REF_DIR, "preproc_base.yml")
        self.changelog_dir = os.path.join(const.REF_DIR, "InstrumEvent")
        # self.instrum_status_dict = const.INSTRUM_STATUS_DICT

        self._file_indep_initialized = True

    def get_changelog(self, unit=1):
        return os.path.join(self.changelog_dir, f"changelog_unit{unit}.txt")

    @property
    def preproc_output_yml(self) -> str:
        config_stems = self.config_stem if hasattr(self, "config_stem") and self.config_stem else "preproc_config"
        # return [os.path.join(d, f"{s}.yml") for d, s in zip(self._preproc_output_dir, config_stems)]
        return bjoin(self._preproc_output_dir, [f"{s}.yml" for s in atleast_1d(config_stems)])

    @property
    def preproc_output_log(self):
        return swap_ext(self.preproc_output_yml, "log")
        # if isinstance(self.preproc_output_yml, str):
        #     return swap_ext(self.preproc_output_yml, "log")
        # else:
        #     return [swap_ext(s, "log") for s in self.preproc_output_yml]

    @property
    def sciproc_output_yml(self):
        # yml_basenames = [
        #     "_".join([obj, filte, unit, date]) + ".yml"
        #     for obj, filte, unit, date in zip(self.name.obj, self.name.filter, self.name.unit, self.name.date)
        # ]
        if self._input_files:
            yml_basenames = []
            for i in range(len(self._input_files)):
                obj = self._get_name_property_at_index("obj", i)
                filte = self._get_name_property_at_index("filter", i)
                # unit = self._get_property_at_index("unit", i)
                # date = self._get_property_at_index("date", i)
                nightdate = self._get_name_property_at_index("nightdate", i)

                yml_basenames.append("_".join([obj, filte, nightdate]) + ".yml")

            return bjoin(self._output_dir, yml_basenames)
        return bjoin(self.output_parent_dir, "sciproc_config.yml")

    @property
    def sciproc_output_log(self):
        if isinstance(self.sciproc_output_yml, str):
            return swap_ext(self.sciproc_output_yml, "log")
        else:
            return [swap_ext(s, "log") for s in self.sciproc_output_yml]

    @property
    def output_name(self) -> str:
        return collapse(self._config_stem)

    def add_fits(self, files: str | Path | list):
        if isinstance(files, list):
            self._input_files = [str(f) for f in files]
        else:
            self._input_files = [files]

    def define_file_dependent_paths(self):
        self._output_dir = []
        self._factory_dir = []
        self._single_dir = []
        self._config_stem = []
        # raw_images = []
        # processed_images = []
        self._masterframe_dir = []
        self._figure_dir = []
        self._daily_stacked_dir = []
        self._subtracted_dir = []
        self._stacked_dir = []
        self._metadata_dir = []

        for i in range(len(self._input_files)):
            # Get properties for this specific file
            nightdate = self._get_name_property_at_index("nightdate", i)
            unit = self._get_name_property_at_index("unit", i)
            obj = self._get_name_property_at_index("obj", i)
            filte = self._get_name_property_at_index("filter", i)
            typ = self._get_name_property_at_index("type", i)

            # Masterframe directory
            masterframe_dir = os.path.join(const.MASTER_FRAME_DIR, nightdate, unit)
            self._masterframe_dir.append(masterframe_dir)

            config_stem = "_".join([nightdate, unit])  # for preproc config
            self._config_stem.append(config_stem)

            if "calibrated" in typ or "raw" in typ:
                if self._is_pipeline[i]:
                    # Within pipeline processing
                    relative_path = os.path.join(nightdate, obj, filte)
                    output_dir = os.path.join(self._output_parent_dir[i], relative_path)

                    self._factory_dir.append(os.path.join(self._factory_parent_dir[i], relative_path))
                    self._single_dir.append(os.path.join(output_dir, "singles"))
                    self._stacked_dir.append(os.path.join(const.STACKED_DIR, obj, filte))
                    self._metadata_dir.append(os.path.join(self._output_parent_dir[i], nightdate))
                else:
                    # Outside pipeline
                    output_dir = self._output_parent_dir[i]
                    self._factory_dir.append(self._factory_parent_dir[i])
                    self._single_dir.append(output_dir)
                    self._stacked_dir.append(output_dir)

                self._output_dir.append(output_dir)
                self._daily_stacked_dir.append(os.path.join(output_dir, "stacked"))
                self._subtracted_dir.append(os.path.join(output_dir, "subtracted"))
                self._figure_dir.append(os.path.join(self._output_dir[-1], "figures"))

            elif "master" in typ:
                self._output_dir.append(self._output_parent_dir[i])
                self._factory_dir.append(self._factory_parent_dir[i])
                self._single_dir.append(self._output_parent_dir[i])
            else:
                raise ValueError("Unrecognized type for PathHandling")

        # Store all as lists without collapsing
        self.output_dir = collapse(self._output_dir)
        self.factory_dir = collapse(self._factory_dir)
        self.single_dir = collapse(self._single_dir)
        self.figure_dir = collapse(self._figure_dir)
        self.masterframe_dir = collapse(self._masterframe_dir)
        self.config_stem = collapse(self._config_stem)

        if self._daily_stacked_dir:
            self.daily_stacked_dir = self._daily_stacked_dir
        if self._subtracted_dir:
            self.subtracted_dir = self._subtracted_dir
        if self._stacked_dir:
            self.stacked_dir = self._stacked_dir
        if self._metadata_dir:
            self.metadata_dir = self._metadata_dir

        self._file_dep_initialized = True

    def define_operation_paths(self):
        self.preprocess = PathPreprocess(self, self._config)
        self.astrometry = PathAstrometry(self, self._config)
        self.photometry = PathPhotometry(self, self._config)
        self.imstack = PathImstack(self, self._config)
        self.imsubtract = PathImsubtract(self, self._config)

    @property
    def conjugate(self) -> str | list[str]:
        """None signals nonexistent file(s)"""
        paths = []
        for i, input in enumerate(self._input_files):
            typ = self._get_name_property_at_index("type", i)
            basename = self._get_name_property_at_index("conjugate_basename", i)

            if "raw" in typ[0]:
                # original was raw → conjugate is processed
                root = self._single_dir[i]
                paths.append(os.path.join(root, basename))
            elif "calibrated" in typ[0]:
                # original was processed → conjugate is raw
                unit = self._get_name_property_at_index("unit", i)
                nightdate = self._get_name_property_at_index("nightdate", i)
                n_binning = self._get_name_property_at_index("n_binning", i)
                gain = self._get_name_property_at_index("gain", i)
                root = find_raw_path(unit, nightdate, n_binning, gain)
                paths.append(os.path.join(root, basename))
            else:
                paths.append(input)

        return paths
        # return paths[0] if self._single else paths

    @property
    def raw_images(self):
        # return [i if "raw" in typ[0] else c for typ, i, c in zip(self.name.type, self._input_files, self.conjugate)]
        paths = []
        for i, input in enumerate(self._input_files):
            basename = self._get_name_property_at_index("raw_basename", i)
            unit = self._get_name_property_at_index("unit", i)
            nightdate = self._get_name_property_at_index("nightdate", i)
            n_binning = self._get_name_property_at_index("n_binning", i)
            gain = self._get_name_property_at_index("gain", i)
            root = find_raw_path(unit, nightdate, n_binning, gain)
            raw_image_template = os.path.join(root, basename)
            globbed = glob(raw_image_template)
            if len(globbed) == 0:
                raise FileNotFoundError(f"No raw image found for {raw_image_template}")
            elif len(globbed) > 1:
                raise FileExistsError(f"Multiple raw images found for {raw_image_template}")
            raw_image = globbed[0]
            paths.append(raw_image)

        return paths

    @property
    def processed_images(self):
        """blindly tries to construct processed_image name"""
        # """Returns input as is if given calib frames"""
        # return [c if "raw" in typ else i for typ, i, c in zip(self.name.type, self._input_files, self.conjugate)]
        paths = []
        for i, input in enumerate(self._input_files):
            basename = self._get_name_property_at_index("processed_basename", i)
            root = self._single_dir[i]
            paths.append(os.path.join(root, basename))

        return paths

    @property
    def catalog(self):
        # return add_suffix(self.processed_images, "cat")
        return add_suffix(self._input_files, "cat")

    @property
    def weight(self):
        # return add_suffix(self.processed_images, "weight")
        return add_suffix(self._input_files, "weight")

    @property
    def obs_params(self):
        return self.name.to_dict()

    def pick_type(self, typ):
        return self.name.pick_type(typ)

    def filter_by(self, attr, val):
        attrs = getattr(self.name, attr)
        return [f for f, a in zip(self._input_files, attrs) if a == val]

    def get_minimum(self, attr):
        attrs = getattr(self.name, attr)
        if isinstance(attrs, list):
            idx = attrs.index(min(attrs))
            return self._input_files[idx]
        else:
            return self._input_files[0]

    # @classmethod
    # def from_grouped_calib(cls, sci_files, on_date_calib):
    #     from collections import Counter

    #     triples = [
    #         (tuple(on_date_bias), tuple(on_date_dark), tuple(on_date_flat))
    #         for flag, on_date_bias, on_date_dark, on_date_flat in on_date_calib
    #         if flag == True
    #     ]
    #     counts = Counter(triples)

    #     result = []
    #     # in ascending order of count
    #     for (bias_files, dark_files, flat_files), cnt in sorted(
    #         counts.items(), key=lambda item: item[1]
    #     ):  # reverse=True
    #         raw_bias = list(bias_files)
    #         raw_dark = list(dark_files)
    #         raw_flat = list(flat_files)
    #         master_bias = PathHandler(raw_bias).preprocess.bias
    #         master_dark = PathHandler(raw_dark).preprocess.dark
    #         master_flat = PathHandler(raw_flat).preprocess.flat

    #         result.append(((raw_bias, raw_dark, raw_flat), (master_bias, master_dark, master_flat)))

    #     return result

    @classmethod
    def take_raw_inventory(cls, files: list[str], lone_calib=True, ignore_mult_date=False):
        return cls.build_preproc_input(
            *NameHandler.find_calib_for_sci(files), lone_calib=lone_calib, ignore_mult_date=ignore_mult_date
        )

    @classmethod
    def build_preproc_input(
        cls, sci_files, on_date_calib, off_date_calib=None, lone_calib=True, ignore_mult_date=False
    ):
        """
        Group science files by their associated on-date calibration sets.

        Parameters
        ----------
        sci_files : list
            List of science file identifiers (e.g. file paths), parallel to `on_date_calib`.
        on_date_calib : list of tuples
            Each element is (on_date_flag, bias_list, dark_list, flat_list), where
            `on_date_flag` is True if on-date calibration exists.
        revisit : list of strings
            Feed the list all files for the date again to pick raw calib frames
            not paired with on-date science images.

        Returns
        -------
        list of 3-tuples
            Each element is structured as
            (
                (raw_bias, raw_dark, raw_flat),
                (master_bias, master_dark, master_flat),
                {
                    (obj, filter, unit, n_binning): ([raw_sci], [processed_sci],
                    (): ([], []),
                    ...
                }
            )
            - For on-date groups (sorted by increasing group size):
                • raw_* lists are the original bias/dark/flat file paths
                • master_* are the processed calibration frames via `PathHandler(...).preprocess.*`
                • sci_groups is the list of science files sharing that calibration triple
            - For off-date entries (appended last):
                • raw_* are empty lists `([], [], [])`
                • master_* are search templates to lookup in `masterframe_dir`
                • sci_groups is a singleton list containing that science file
        """
        from collections import defaultdict

        def get_dict_key(sci_group):
            """Use only the values as tuple keys"""
            tuple_key = tuple(
                v
                for k, v in collapse(
                    # NameHandler.parse_params(sci_group, keys=const.SURVEY_SCIENCE_GROUP_KEYS), raise_error=True
                    NameHandler.parse_params(sci_group, keys=const.TRANSIENT_SCIENCE_GROUP_KEYS),
                    raise_error=True,
                ).items()
            )
            # return tuple_key
            return "_".join(tuple_key)

        # dict with master calib 3-tuples as keys, dict of raw bdf and sci as values
        calib_map = defaultdict(lambda: {"bias": None, "dark": None, "flat": None, "sci": []})

        for (bias, dark, flat), sci in zip(on_date_calib, sci_files):
            mbias, mdark, mflat = cls(sci[0]).preprocess.masterframe  # [0]: trust the grouping

            key = tuple((mbias, mdark, mflat))  # (tuple(bias), tuple(dark), tuple(flat))
            entry = calib_map[key]
            entry["sci"].append(sci)
            # stash the raw lists once
            if entry["bias"] is None:
                entry["bias"] = list(bias)
                entry["dark"] = list(dark)
                entry["flat"] = list(flat)

        #
        result = []

        # for _key, entry in sorted(calib_map.items(), key=lambda kv: len(kv[1]["sci"])):  # sort by number of sci groups
        # sort by # of sci frames
        for _key, entry in sorted(
            calib_map.items(), key=lambda kv: sum(len(inner) for inner in kv[1]["sci"]), reverse=True
        ):

            sci_dict = {}
            for sci_group in entry["sci"]:
                sci_group = sorted(sci_group)
                key = get_dict_key(sci_group)
                sci_dict[key] = (sci_group, atleast_1d(cls(sci_group).conjugate))

            raw_bias = entry["bias"]
            raw_dark = entry["dark"]
            raw_flat = entry["flat"]
            # If too few raw calib frames, ignore them.
            if len(raw_bias) < const.NUM_MIN_CALIB:
                raw_bias = []
            if len(raw_dark) < const.NUM_MIN_CALIB:
                raw_dark = []
            if len(raw_flat) < const.NUM_MIN_CALIB:
                raw_flat = []

            sample_file = next(iter(sci_dict.values()))[0][0]
            mbias, mdark, mflat = cls(sample_file).preprocess.masterframe  # trust the grouping

            result.append(
                [
                    [sorted(raw_bias), sorted(raw_dark), sorted(raw_flat)],
                    [mbias, mdark, mflat],
                    sci_dict,  # a dict of tuples of lists: ([science images in this on‐date group], [processed images])
                ]
            )

        # raw calibration frames with no corresponding on-date science frames
        if lone_calib and off_date_calib and any(l for l in off_date_calib):
            off_date_bias_groups, off_date_dark_groups, off_date_flat_groups = off_date_calib
            for off_date_bias_group in off_date_bias_groups:
                if len(off_date_bias_group) < const.NUM_MIN_CALIB:
                    continue

                mbias = cls.ensure_unique(cls(off_date_bias_group).preprocess.masterframe, off=not ignore_mult_date)
                # preprocess.mbias works too
                result.append([[sorted(off_date_bias_group), [], []], [mbias, "", ""], dict()])

            for off_date_dark_group in off_date_dark_groups:
                if len(off_date_dark_group) < const.NUM_MIN_CALIB:
                    continue

                mdark = cls.ensure_unique(cls(off_date_dark_group).preprocess.masterframe, off=not ignore_mult_date)
                mbias = cls.ensure_unique(
                    cls(off_date_dark_group).preprocess.mbias, off=not ignore_mult_date
                )  # mbias needed for mdark generation

                # use pre-generated mbias saved to disk, even if on-date
                result.append([[[], sorted(off_date_dark_group), []], [mbias, mdark, ""], dict()])

            for off_date_flat_group in off_date_flat_groups:
                if len(off_date_flat_group) < const.NUM_MIN_CALIB:
                    continue

                mflat = cls.ensure_unique(cls(off_date_flat_group).preprocess.masterframe, off=not ignore_mult_date)
                self = cls(off_date_flat_group)
                self.name.exptime = ["*"] * len(off_date_flat_group)  # * is a glob wildcard
                mdark = cls.ensure_unique(self.preprocess.mdark, off=not ignore_mult_date)
                mbias = cls.ensure_unique(cls(off_date_flat_group).preprocess.mbias, off=not ignore_mult_date)
                result.append([[[], [], sorted(off_date_flat_group)], [mbias, mdark, mflat], dict()])

        return result

    @staticmethod
    def ensure_unique(mframe, off=False):
        """
        CAVEAT: this is ad-hoc.
        Using this relies on that the inputs are properly grouped by nightdate,
        and only date of files are erroneous.

        You may turn it off for future data as of 2025-07-15.
        """
        if off:
            return mframe

        if isinstance(mframe, list):
            mframe_selected = sorted(mframe)[-1]
            print(f"[WARNING] Degenerate output filenames: {mframe};\nUsing the last: {mframe_selected}")
            return mframe_selected
        return mframe

    @classmethod
    def get_group_info(cls, raw_group):
        exptime, filter = "_", "_"
        mdark = raw_group[1][1]
        mflat = raw_group[1][2]
        if mdark:
            exptime = collapse(PathHandler(mdark).exptime)
        if mflat:
            filter = collapse(PathHandler(mflat).filter)
        return f"{filter}: {exptime}"

    @classmethod
    def weight_map_input(cls, mzdf_list: list[str]):
        """
        Input is a list of basenames of [mbias, mdark, mflat]

        Returns d_m_file, f_m_file, sig_z_file, sig_f_file
        """
        # z_m_file, d_m_file, f_m_file = (cls(s).preprocess.masterframe for s in mzdf_list)  # basename to full path
        z_m_file, d_m_file, f_m_file = cls(mzdf_list).preprocess.masterframe  # with vectorized PathHandler
        sig_z_file = z_m_file.replace("bias", "biassig")
        sig_f_file = f_m_file.replace("flat", "flatsig")
        return d_m_file, f_m_file, sig_z_file, sig_f_file

    @classmethod
    def get_bpmask(cls, input: Union[str, Path, "fits.Header"]) -> str | list[str]:
        from astropy.io import fits

        if isinstance(input, str | Path):
            name = NameHandler(input)
            if name.type[0] == "master" and (name.type[1] == "dark" or name.type[1] == "darksig"):
                input = cls(input).preprocess.masterframe  # ensure full path to master dark, not darksig
                return input.replace("dark", "bpmask")

            elif name.type[0] == "calibrated":
                header = fits.getheader(input)

        elif isinstance(input, fits.Header):
            header = input

        elif isinstance(input, list):
            return collapse([cls.get_bpmask(f) for f in input])

        else:
            raise ValueError("Invalid input to find bpmask")

        calibs = [v for k, v in header.items() if "IMCMB" in k]
        mdark = NameHandler(calibs).pick_type("master_dark")
        assert isinstance(mdark, str)
        mdark = cls(mdark).preprocess.masterframe
        return mdark.replace("dark", "bpmask")


class PathPreprocess(AutoMkdirMixin, AutoCollapseMixin):
    def __init__(self, parent: PathHandler, config=None):
        self._parent = parent

        # Apply config overrides if provided
        if config and hasattr(config, "path"):
            for key, val in config.preprocess.path.items():
                setattr(self, key, val)

    def __repr__(self):
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items() if not k.startswith("_"))

    @property
    def _masterframe_dir(self):
        """returns list-wrapped masterframe_dir"""
        if isinstance(self._parent.masterframe_dir, str):
            return [self._parent.masterframe_dir] * len(self._parent.name.masterframe_basename)
        return self._parent.masterframe_dir

    @property
    def mbias(self):
        # PathHandlerDeprecated Behavior
        # """Given mixed raw calib images, pick up only bias frames and give their
        # masterframe counterparts"""
        # result = []
        # for typ, d, s in zip(self._parent.name.type, self._masterframe_dir, self._parent.name.masterframe_basename):
        #     if typ[1] == "bias":
        #         result.append(os.path.join(d, s))
        #     elif typ[1] == "science":
        #         result.append(os.path.join(d, s[0]))
        #     else:
        #         result.append(None)

        # return result

        """Given a master frame, generates the name of mbias needed to create it"""
        return bjoin(self._parent._masterframe_dir, self._parent.name.mbias_basename)

    @property
    def mdark(self):
        # result = []
        # for typ, d, s in zip(
        #     self._parent.name.type, self._parent._masterframe_dir, self._parent.name.masterframe_basename
        # ):
        #     if typ[1] == "dark":
        #         result.append(os.path.join(d, s))
        #     # if typ[1] == 'dark' or typ[1] =='flat':
        #     #     pass
        #     elif typ[1] == "science":
        #         result.append(os.path.join(d, s[1]))
        #     else:
        #         result.append(None)

        # return result
        return bjoin(self._parent._masterframe_dir, self._parent.name.mdark_basename)

    @property
    def mflat(self):
        # result = []
        # for typ, d, s in zip(
        #     self._parent.name.type, self._parent._masterframe_dir, self._parent.name.masterframe_basename
        # ):
        #     if typ[1] == "flat":
        #         result.append(os.path.join(d, s))
        #     # if typ[1] == 'dark' or typ[1] =='flat':
        #     #     pass
        #     elif typ[1] == "science":
        #         result.append(os.path.join(d, s[2]))
        #     else:
        #         result.append(None)

        # return result
        return bjoin(self._parent._masterframe_dir, self._parent.name.mflat_basename)

    @property
    def masterframe(self):
        """
        Deletages NameHandler.masterframe_basename to make a full absolute path
        tuple(z, d, f) if science, just list[str] | str if calib
        """

        if self._parent._single:
            typ = self._parent.name.type
            if typ[1] == "science":
                return [os.path.join(self._masterframe_dir[0], s) for s in self._parent.name.masterframe_basename]
            else:
                return os.path.join(self._masterframe_dir[0], self._parent.name.masterframe_basename)

        result = []
        for typ, mfdir, basename in zip(
            self._parent.name.type, self._masterframe_dir, self._parent.name.masterframe_basename
        ):
            if typ[1] == "science":
                result.append([os.path.join(mfdir, s) for s in basename])
            else:
                result.append(os.path.join(mfdir, basename))
        return result

        # return bjoin(self._parent.masterframe_dir, self._parent.name.masterframe_basename)


class PathAstrometry(AutoMkdirMixin):
    _mkdir_exclude = {"ref_ris_dir", "ref_query_dir"}

    def __init__(self, parent: PathHandler, config=None):
        self._parent = parent

        # Default values
        self.ref_ris_dir = "/lyman/data2/factory/ref_scamp/gaia_dr3_7DT"  # "/lyman/data1/factory/catalog/gaia_dr3_7DT"
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
    def astrefcat(self):
        import re

        # use local astrefcat if tile obs
        match = re.search(r"T\d{5}", self._parent.name.obj)
        if match:
            astrefcat = os.path.join(self.ref_ris_dir, f"{match.group()}.fits")
        else:
            astrefcat = None
        return astrefcat

    # @property
    # def input_files(self):
    #     return self._parent.processed_images

    # @property
    # def solvefield_outputs(self):
    #     exts = ["solved", "axy", "corr", "match", "rdls", "wcs"]  # -indx.xyls?
    #     return tuple([swap_ext(image, ext) for ext in exts] for image in self.input_files)

    # @property
    # def catalog(self):
    #     # return (add_suffix(add_suffix(inim, 'prep'), "cat") for inim in self.input_files)
    #     return [add_suffix(inim, "cat") for inim in self.input_files]


class PathPhotometry(AutoMkdirMixin, AutoCollapseMixin):
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
        input = self._parent.name.basename
        return bjoin(self.tmp_dir, add_suffix(add_suffix(input, "prep"), "cat"))

    @property
    def main_catalog(self):
        """intermediate sextractor output"""
        input = self._parent.name.basename
        return bjoin(self.tmp_dir, swap_ext(input, "cat"))

    @property
    def final_catalog(self):
        """final pipeline output catalog"""
        input = self._parent._input_files
        return add_suffix(input, "cat")

    # def __getattr__(self, name):
    #     # run file-dependent path definitions once?
    #     pass


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

    @property
    def stacked_image_basename(self):
        # names = NameHandler(self._parent._input_files)
        names = self._parent.name
        _ = collapse(names.type, raise_error=True)  # ensure input images are coherent
        total_exptime = np.sum(names.exptime)
        # use the datetime of the last image
        unit = collapse(names.unit, force=True)
        fname = f"{names.obj_collapse}_{names.filter_collapse}_{unit}_{names.datetime[-1]}_{format_exptime(total_exptime, type='stacked')}_coadd.fits"
        return fname

    @property
    def daily_stacked_image(self):
        return os.path.join(self._parent.daily_stacked_dir, self.stacked_image_basename)

    @property
    def stacked_image(self):
        # return os.path.join(collapse(self._parent.stacked_dir, raise_error=True), self.stacked_image_basename)
        return bjoin(self._parent.stacked_dir, self.stacked_image_basename)

    # Todo: add weight images


class PathImsubtract(AutoMkdirMixin, AutoCollapseMixin):
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

    @property
    def diffim(self):
        # return bjoin(self._parent.output_dir, "difference", add_suffix(self._parent._input_files, "diff"))
        input = os.path.basename(self._parent.processed_images)  # Define a new PathHandler with stacked_image as input
        return bjoin(self._parent.output_dir, "difference", add_suffix(input, "diff"))
