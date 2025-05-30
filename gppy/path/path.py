import os
from pathlib import Path
from typing import Union, TYPE_CHECKING
import numpy as np
from .. import const
from ..utils import add_suffix, swap_ext, collapse, find_raw_path
from .name import NameHandler
from .utils import format_exptime
from .utils import broadcast_join_pure as bjoin


if TYPE_CHECKING:
    from gppy.config import Configuration  # just for type hinting. actual import will cause circular import error


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
    _collapse_include = {}  # "output_dir", "image_dir", "factory_dir", "stacked_dir"}

    def __getattribute__(self, name):
        if name.startswith("_"):
            return object.__getattribute__(self, name)

        value = object.__getattribute__(self, name)

        # Collapse if explicitly included or path-like list
        if name not in self._collapse_exclude and (
            name in self._collapse_include
            or (isinstance(value, list) and all(isinstance(v, (str, Path)) for v in value))
        ):
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

    _mkdir_exclude = {"output_name", "config_stem", "name"}  # subclasses can override this

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


class PathHandler(AutoCollapseMixin, AutoMkdirMixin):  # SingletonUnpackMixin, Check MRO: PathHandler.mro()
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

    def __init__(self, input: Union[str, Path, list[str | Path]] = None, *, working_dir=None, check_coherence=True):
        self._name_cache = {}  # Cache for NameHandler properties
        self._handle_input(input, check_coherence=check_coherence)
        self.select_output_dir(working_dir=working_dir)

        self.define_file_independent_paths()

        if not self._file_dep_initialized and self._input_files:
            self.define_file_dependent_paths()

        # if self._file_indep_initialized and self._file_dep_initialized:
        self.define_operation_paths()

    def _handle_input(self, input, check_coherence=True):
        """init with obs_parmas and config are ad-hoc. Will be changed to always take filenames"""
        self._config = None
        self._input_files: list[str] = None
        self._file_indep_initialized = False
        self._file_dep_initialized = False

        if input is None:
            return

        # elif isinstance(input, list) or isinstance(input, (str, Path)):

        # Normalize input to list
        if not isinstance(input, list):
            input = [input]
        self._input_files = [os.path.abspath(img) for img in input]
        self.name = NameHandler(input)
        self._single = self.name._single

    def __getattr__(self, name):
        """
        Below runs when name is not in __dict__.
        (1) If file-dependent paths have not been built yet, build them.
        (2) Retry the lookup - if the attribute was created by the builder we
            return it; otherwise fall through to the convenience “_to_*” hooks.
        """
        # # Optimized attribute access with whitelist checking
        # if name in self._EXPOSED_ATTRS:
        #     return getattr(self, f"_{name}")

        # Check if it's a NameHandler property and cache it
        nh = self.__dict__.get("name", None)
        if nh and hasattr(nh, name):
            return self._get_cached_namehandler_property(name)

        # # Delegate to NameHandler first evading AutoMkdirMixin
        # names = self.__dict__.get("names", None)
        # if names and hasattr(names, name):
        #     return getattr(self.names, name)

        # Lazy initialization
        if not self._file_dep_initialized and self._input_files:
            self._file_dep_initialized = True  # set the flag first to prevent accidental recursion
            try:
                self.define_file_dependent_paths()
            except Exception:
                # roll back if the builder blew up
                self._file_dep_initialized = False
                raise

            if name in self.__dict__:
                # Temporarily return uniform list as a scalar
                returned_attr = self.__dict__[name]
                if isinstance(returned_attr, list):
                    if all(val == returned_attr[0] for val in returned_attr):
                        return returned_attr[0]
                return returned_attr

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
            # return self._vectorize_collapse(val)
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
            self._within_pipeline = [False]
            return

        # Process each file independently
        self._output_parent_dir = []
        self._preproc_output_dir = []
        self._factory_parent_dir = []
        self._within_pipeline = []

        for i, input_file in enumerate(self._input_files):
            file_dir = str(Path(input_file).absolute().parent)
            not_pipeline_dir = not any(s in file_dir for s in const.PIPELINE_DIRS)

            if working_dir or not_pipeline_dir:
                output_parent_dir = working_dir or os.path.dirname(input_file)
                self._output_parent_dir.append(output_parent_dir)
                self._factory_parent_dir.append(os.path.join(output_parent_dir, "tmp"))
                self._within_pipeline.append(False)
            else:
                from datetime import date

                nightdate = self._get_property_at_index("nightdate", i)
                if isinstance(nightdate, list):
                    current_nightdate = nightdate[i] if i < len(nightdate) else nightdate[0]
                else:
                    current_nightdate = nightdate or date.today().strftime("%Y%m%d")

                if current_nightdate < "20260101":
                    output_parent_dir = const.PROCESSED_DIR
                    self._output_parent_dir.append(output_parent_dir)
                    self._factory_parent_dir.append(const.FACTORY_DIR)
                    self._within_pipeline.append(True)
                else:
                    raise ValueError(f"nightdate cap reached for file {input_file}: consider moving to another disk.")

            preproc_output_dir = os.path.join(output_parent_dir, self._get_property_at_index("nightdate", i))
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

    def _get_property_at_index(self, prop_name: str, index: int):
        """Get cached NameHandler property value at specific index"""
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

        # for non-pipeline input; overridden in define_file_independent_paths()

        # self.imstack_base_yml
        # self.phot_base_yml
        self._file_indep_initialized = True

    @property
    def preproc_output_yml(self) -> str:
        config_stems = self._config_stem if hasattr(self, "_config_stem") and self._config_stem else "preproc_config"
        return [os.path.join(d, f"{s}.yml") for d, s in zip(self._preproc_output_dir, config_stems)]

    @property
    def preproc_output_log(self):
        # return swap_ext(self.preproc_output_yml, "log")
        if isinstance(self.preproc_output_yml, str):
            return swap_ext(self.preproc_output_yml, "log")
        else:
            return [swap_ext(s, "log") for s in self.preproc_output_yml]

    @property
    def sciproc_output_yml(self):
        config_stems = self._config_stem if hasattr(self, "_config_stem") and self._config_stem else "sciproc_config"
        return [os.path.join(d, f"{s}.yml") for d, s in zip(self._output_dir, config_stems)]

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
        self._image_dir = []
        self._config_stem = []
        # raw_images = []
        # processed_images = []
        self._masterframe_dir = []
        self._figure_dir = []

        for i, input_file in enumerate(self._input_files):
            # Get properties for this specific file
            nightdate = self._get_property_at_index("nightdate", i)
            unit = self._get_property_at_index("unit", i)
            obj = self._get_property_at_index("obj", i)
            filte = self._get_property_at_index("filter", i)
            # data_type = self._get_property_at_index("type", i)

            # Masterframe directory
            masterframe_dir = os.path.join(const.MASTER_FRAME_DIR, nightdate, unit)
            self._masterframe_dir.append(masterframe_dir)

            config_stem = "_".join([nightdate, unit])
            self._config_stem.append(config_stem)

            if self._within_pipeline[i]:
                # Within pipeline processing
                relative_path = os.path.join(nightdate, obj, filte)
                output_dir = os.path.join(self._output_parent_dir[i], relative_path)
                factory_dir = os.path.join(self._factory_parent_dir[i], relative_path)
                image_dir = os.path.join(output_dir, "images")

                self._output_dir.append(output_dir)
                self._factory_dir.append(factory_dir)
                self._image_dir.append(image_dir)

                # # Handle processed images based on data type
                # if "raw" in data_type:
                #     conjugate = self._get_property_at_index("conjugate", i)
                #     processed_image = os.path.join(image_dir, conjugate)
                #     processed_images.append(processed_image)
                #     raw_images.append(input_file)
                # elif "calibrated" in data_type:
                #     conjugate = self._get_property_at_index("conjugate", i)
                #     raw_image = os.path.join(image_dir, conjugate)
                #     raw_images.append(raw_image)
                #     processed_images.append(input_file)
                # else:
                #     processed_images.append(str(Path(input_file).absolute()))
            else:
                # Outside pipeline
                self._output_dir.append(self._output_parent_dir[i])
                self._factory_dir.append(self._factory_parent_dir[i])
                self._image_dir.append(self._output_parent_dir[i])
                # raw_images.append(str(Path(input_file).absolute()))

            self._figure_dir.append(os.path.join(self._output_dir[-1], "figures"))

        # Store all as lists without collapsing
        self.output_dir = collapse(self._output_dir)
        self.factory_dir = collapse(self._factory_dir)
        self.image_dir = collapse(self._image_dir)
        self.masterframe_dir = collapse(self._masterframe_dir)
        self.figure_dir = collapse(self._figure_dir)
        self.config_stem = collapse(self._config_stem)
        # if raw_images:
        #     self.raw_images = raw_images
        # if processed_images:
        #     self.processed_images = processed_images

        # Generate additional directories
        self._generate_additional_dirs(self._output_dir)

        self._file_dep_initialized = True

    def _generate_additional_dirs(self, output_dirs):
        """Generate additional directories like daily_stacked_dir, etc."""
        self._daily_stacked_dir = []
        self._subtracted_dir = []
        self._stacked_dir = []
        self._metadata_dir = []

        for i, output_dir in enumerate(output_dirs):
            if self._within_pipeline[i]:
                self._daily_stacked_dir.append(os.path.join(output_dir, "stacked"))
                self._subtracted_dir.append(os.path.join(output_dir, "subtracted"))

                obj = self._get_property_at_index("obj", i)
                filter_name = self._get_property_at_index("filter", i)
                nightdate = self._get_property_at_index("nightdate", i)

                self._stacked_dir.append(os.path.join(const.STACKED_DIR, obj, filter_name))
                self._metadata_dir.append(os.path.join(self._output_parent_dir[i], nightdate))
            else:
                self._daily_stacked_dir.append(None)
                self._subtracted_dir.append(None)
                self._stacked_dir.append(None)
                self._metadata_dir.append(None)

        # Store as lists, filtering out None values where appropriate
        self.daily_stacked_dir = self._daily_stacked_dir
        self.subtracted_dir = self._subtracted_dir
        self.stacked_dir = self._stacked_dir
        self.metadata_dir = self._metadata_dir

    def define_operation_paths(self):
        self.preprocess = PathPreprocess(self, self._config)
        self.astrometry = PathAstrometry(self, self._config)
        self.photometry = PathPhotometry(self, self._config)
        self.imstack = PathImstack(self, self._config)
        self.imsubtract = PathImsubtract(self, self._config)

    @property
    def conjugate(self) -> str | list[str]:
        """None signals nonexistent file(s)"""
        basenames = self.name.conjugate
        types = self.type

        if self._single:
            basenames, types = [basenames], [types]

        paths = []
        for i, (input, basename, typ) in enumerate(zip(self._input_files, basenames, types)):
            if "raw" in typ:
                # original was raw → conjugate is processed
                root = self._image_dir[i]
                paths.append(os.path.join(root, basename))
            elif "calibrated" in typ:
                # original was processed → conjugate is raw
                unit = self._get_property_at_index("unit", i)
                nightdate = self._get_property_at_index("nightdate", i)
                n_binning = self._get_property_at_index("n_binning", i)
                gain = self._get_property_at_index("gain", i)
                root = find_raw_path(unit, nightdate, n_binning, gain)
                paths.append(os.path.join(root, basename))
            else:
                paths.append(input)

        return paths[0] if self._single else paths

    @property
    def raw_images(self):
        return [i if "raw" in typ else c for typ, i, c in zip(self.name.type, self._input_files, self.conjugate)]

    @property
    def processed_images(self):
        """Returns input as is if given calib frames"""
        return [c if "raw" in typ else i for typ, i, c in zip(self.name.type, self._input_files, self.conjugate)]

    @property
    def obs_params(self):
        return self.name.to_dict()

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
    def take_raw_inventory(cls, files: list[str]):
        return cls.build_preproc_input(*NameHandler.find_calib_for_sci(files))

    @classmethod
    def build_preproc_input(cls, sci_files, on_date_calib, off_date_calib=None):
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
                    NameHandler.parse_params(sci_group, keys=const.SURVEY_SCIENCE_GROUP_KEYS), raise_error=True
                ).items()
            )
            # return tuple_key
            return "_".join(tuple_key)

        # dict with master calib 3-tuples as keys, dict of raw bdf and sci as values
        calib_map = defaultdict(lambda: {"sci": [], "bias": None, "dark": None, "flat": None})

        for sci, (bias, dark, flat) in zip(sci_files, on_date_calib):
            mbias, mdark, mflat = cls(sci[0]).preprocess.masterframe  # trust the grouping

            key = tuple((mbias, mdark, mflat))  # (tuple(bias), tuple(dark), tuple(flat))
            entry = calib_map[key]
            entry["sci"].append(sci)
            # stash the raw lists once
            if entry["bias"] is None:
                entry["bias"] = list(bias)
                entry["dark"] = list(dark)
                entry["flat"] = list(flat)

        result = []

        # for _key, entry in sorted(calib_map.items(), key=lambda kv: len(kv[1]["sci"])):  # sort by # of sci groups
        # sort by # of sci frames
        for _key, entry in sorted(calib_map.items(), key=lambda kv: sum(len(inner) for inner in kv[1]["sci"])):

            sci_dict = {}
            for sci_group in entry["sci"]:
                key = get_dict_key(sci_group)
                sci_dict[key] = (sci_group, cls(sci_group).conjugate)

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

            result.append(
                (
                    (raw_bias, raw_dark, raw_flat),
                    (mbias, mdark, mflat),
                    sci_dict,  # a dict of tuples of lists: ([science images in this on‐date group], [processed images])
                )
            )

        if off_date_calib and any(l for l in off_date_calib):
            off_date_bias_groups, off_date_dark_groups, off_date_flat_groups = off_date_calib
            for off_date_bias_group in off_date_bias_groups:
                mbias = cls(off_date_bias_group).preprocess.bias
                result.append(
                    (
                        (off_date_bias_group, [], []),
                        (mbias, [], []),
                        dict(),
                    )
                )

            for off_date_dark_group in off_date_dark_groups:
                mdark = cls(off_date_dark_group).preprocess.dark
                result.append(
                    (
                        ([], off_date_dark_group, []),
                        ([], mdark, []),
                        dict(),
                    )
                )

            for off_date_flat_group in off_date_flat_groups:
                mflat = cls(off_date_flat_group).preprocess.flat
                result.append(
                    (
                        ([], [], off_date_flat_group),
                        ([], [], mflat),
                        dict(),
                    )
                )

        return result

    @classmethod
    def weight_map_input(cls, zdf_list: list[str]):
        """Returns d_m_file, f_m_file, sig_z_file, sig_f_file"""
        z_m_file, d_m_file, f_m_file = (cls(s).preprocess.masterframe for s in zdf_list)
        sig_z_file = z_m_file.replace("bias", "biassig")
        sig_f_file = f_m_file.replace("flat", "flatsig")
        return d_m_file, f_m_file, sig_z_file, sig_f_file


###############################################################################


class PathHandlerDeprecated(AutoMkdirMixin):
    def __init__(
        self, input: Union[str, Path, list, dict, "Configuration"] = None, *, working_dir=None, check_coherence=True
    ):
        """Input homogeneous images"""
        self._config = None
        self._input_files: list[str] = None
        self._data_type = None
        self.obs_params = {}
        self._file_indep_initialized = False
        self._file_dep_initialized = False

        self._handle_input(input, check_coherence=check_coherence)
        self.select_output_dir(working_dir=working_dir)

        self.define_file_independent_paths()

        if not self._file_dep_initialized and self._input_files:
            self.define_file_dependent_paths()

        # if self._file_indep_initialized and self._file_dep_initialized:
        self.define_operation_paths()

    def _handle_input(self, input, check_coherence=True):
        """init with obs_parmas and config are ad-hoc. Will be changed to always take filenames"""

        if input is None:
            pass

        # input is a fits file list; the only method to keep
        elif isinstance(input, list) or isinstance(input, (str, Path)):
            input = list(np.atleast_1d(input))
            self.names = NameHandler(input)
            self._input_files = [os.path.abspath(img) for img in input]
            self._data_type = self.names.type  # you can also use self.types from getattr delegate
            obs_params = collapse(self.names.to_dict(keys=const.PATH_KEYS), keys=const.SURVEY_SCIENCE_GROUP_KEYS)

            if check_coherence and isinstance(obs_params, list):
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
        # 1) Delegate any NameHandler property directly without running AutoMkdirMixin
        names = self.__dict__.get("names", None)  # safe check
        if names and hasattr(names, name):
            # return collapse(getattr(self._names, name))
            return getattr(self.names, name)  # use syntactic sugar _collapse in NameHandler

        # ---------- 2. Lazy initialization ----------
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

        # ---------- 3. “Syntactic-sugar” logic ----------
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

        # ---------- 4. Still not found -> real error ----------
        raise AttributeError(f"{self.__class__.__name__!s} has no attribute {name!r}")

    def __repr__(self):
        return "\n".join(f"{k}: {v}" for k, v in sorted(self.__dict__.items()) if not k.startswith("_"))

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
            _not_pipeline_dir = not any(s in _file_dir for s in const.PIPELINE_DIRS)
        else:
            _not_pipeline_dir = False

        # insufficient info or outside-pipeline input
        if not self.obs_params or working_dir or _not_pipeline_dir:
            working_dir = working_dir or (os.path.dirname(self._input_files[0]) if self._input_files else os.getcwd())

            self._output_parent_dir = working_dir
            tmp_dir = os.path.join(working_dir, "tmp")
            self._factory_parent_dir = tmp_dir
            self.factory_dir = tmp_dir
            self._within_pipeline = False

        else:
            from datetime import date

            datestring = self.obs_params.get("nightdate") or date.today().strftime("%Y%m%d")
            if datestring < "20260101":
                self._output_parent_dir = const.PROCESSED_DIR
                self.output_parent_dir = self._output_parent_dir
                self._factory_parent_dir = const.FACTORY_DIR
                self.factory_parent_dir = self._factory_parent_dir
            else:
                raise ValueError("nightdate cap reached: consider moving to another disk.")
            self._within_pipeline = True

    @property
    def file_dep_initialized(self):
        """Safe from AutoMkdirMixin as it's a bool."""
        return self._file_dep_initialized

    def define_file_independent_paths(self):
        self.ref_sex_dir = os.path.join(const.REF_DIR, "srcExt")

        self.sciproc_base_yml = os.path.join(const.REF_DIR, "sciproc_base.yml")
        self.preproc_base_yml = os.path.join(const.REF_DIR, "preproc_base.yml")

        # for non-pipeline input; overridden in define_file_independent_paths()
        self.sciproc_output_yml = os.path.join(self._output_parent_dir, "preproc_config.yml")
        self.preproc_output_yml = os.path.join(self._output_parent_dir, "sciproc_config.yml")

        # self.imstack_base_yml
        # self.phot_base_yml
        self._file_indep_initialized = True

    # def add_fits(self, files: str | Path | list):
    #     if isinstance(files, list):
    #         self._input_files = [str(f) for f in files]
    #     else:
    #         self._input_files = [files]

    def define_file_dependent_paths(self):

        names = NameHandler(self._input_files)

        # define masterframe_dir regardless of _within_pipeline
        self.masterframe_dir = os.path.join(
            f"{const.MASTER_FRAME_DIR}",
            self.obs_params["nightdate"],
            self.obs_params["unit"],
        )

        if self._within_pipeline:  # and self.obs_params and self.obs_params.get("nightdate"):
            # if not (self.is_present(self._input_files)):
            #     raise FileNotFoundError(f"Not all input paths exist: {self._input_files}")

            # preprocess-related paths
            # _relative_path = os.path.join(self.obs_params["nightdate"], self.obs_params["unit"])
            preproc_output_dir = os.path.join(self._output_parent_dir, self.obs_params["nightdate"])
            self.preproc_output_dir = preproc_output_dir
            config_stem = "_".join([self.obs_params["nightdate"], self.obs_params["unit"]])
            # config_stem = self.obs_params["nightdate"]
            self.preproc_output_yml = os.path.join(preproc_output_dir, config_stem + ".yml")
            self.preproc_output_log = os.path.join(preproc_output_dir, config_stem + ".log")

            # sciproc-related paths
            # _relative_path = os.path.join(self.obs_params["nightdate"], self.obs_params["unit"], self.obs_params["obj"], self.obs_params["filter"])  # fmt:skip
            _relative_path = os.path.join(self.obs_params["nightdate"], self.obs_params["obj"], self.obs_params["filter"])  # fmt:skip
            self._output_dir = os.path.join(self._output_parent_dir, _relative_path)
            self.output_dir = self._output_dir
            self.factory_dir = os.path.join(self._factory_parent_dir, _relative_path)
            self.metadata_dir = os.path.join(self._output_parent_dir, self.obs_params["nightdate"])
            image_dir = os.path.join(self._output_dir, "images")
            self.image_dir = image_dir
            self.daily_stacked_dir = os.path.join(self._output_dir, "stacked")
            self.subtracted_dir = os.path.join(self._output_dir, "subtracted")

            self.stacked_dir = os.path.join(const.STACKED_DIR, self.obs_params["obj"], self.obs_params["filter"])

            config_stem = self.names.config_stem_collapse
            if not isinstance(config_stem, str):
                raise ValueError("Incoherent input: configuration basename is not uniquely defined")
            self.output_name = config_stem
            self.sciproc_output_yml = os.path.join(self._output_dir, config_stem + ".yml")
            self.sciproc_output_log = os.path.join(self._output_dir, config_stem + ".log")

            # raw pipeline images as input
            if "raw" in self._data_type[0]:  # cheating; works for _single
                # self.data_type = self._data_type or "raw"  # interferes with Mkdir

                self.raw_images = self._input_files

                if names._single:
                    self.processed_images = os.path.join(image_dir, names.conjugate)
                else:
                    self.processed_images = [os.path.join(image_dir, f) for f in names.conjugate]

            # processed pipeline images as input
            # elif self.output_parent_dir in str(self._input_files[0]):
            elif "calibrated" in self._data_type[0]:
                self.processed_images = self._input_files if not self.names._single else str(self._input_files[0])

            else:  # user input
                # self.data_type = self._data_type or "user-input"
                print("User input data type detected. Assume the input is a list of processed images.")
                self.processed_images = [str(file.absolute()) for file in self._input_files]
                # self.processed_file_stems = [file.stem for file in self._input_files]
                self._output_dir = str(Path(self._input_files[0]).parent.parent)
                self.factory_dir = str(Path(self._input_files[0]).parent.parent / "factory")

        # outside pipeline
        else:
            self._output_dir = self._output_parent_dir
            self.factory_dir = self._factory_parent_dir

        self.figure_dir = os.path.join(self._output_dir, "figures")

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
        types = names.type

        if not isinstance(basenames, list):  # single path
            basenames, types, single = [basenames], [types], True
        else:  # list of paths
            single = False

        paths = []
        for bn, t in zip(basenames, types):
            if "raw" in t:
                # original was raw → conjugate is processed
                root = self.image_dir
            else:
                # original was processed → conjugate is raw
                root = const.RAWDATA_DIR
            paths.append(os.path.abspath(os.path.join(root, bn)))

        return paths[0] if single else paths

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
    def take_raw_inventory(cls, files: list[str]):
        return cls.build_preproc_input(*NameHandler.find_calib_for_sci(files))

    @classmethod
    def build_preproc_input(cls, sci_files, on_date_calib):
        """
        Group science files by their associated on-date calibration sets.

        Parameters
        ----------
        sci_files : list
            List of science file identifiers (e.g. file paths), parallel to `on_date_calib`.
        on_date_calib : list of tuples
            Each element is (on_date_flag, bias_list, dark_list, flat_list), where
            `on_date_flag` is True if on-date calibration exists.

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

        # Build a map from each (bias,dark,flat) tuple -> sci_groups
        calib_map = defaultdict(lambda: {"sci": [], "bias": None, "dark": None, "flat": None})
        off_date_groups = []

        for sci, (on_date_flag, bias, dark, flat) in zip(sci_files, on_date_calib):
            if on_date_flag:
                key = (tuple(bias), tuple(dark), tuple(flat))
                entry = calib_map[key]
                entry["sci"].append(sci)
                # stash the raw lists once
                if entry["bias"] is None:
                    entry["bias"] = list(bias)
                    entry["dark"] = list(dark)
                    entry["flat"] = list(flat)
            else:
                # off‐date: no calibration, just itself
                off_date_groups.append(sci)

        def get_key(sci_group):
            """Use only the values as tuple keys"""
            tuple_key = tuple(
                v
                for k, v in collapse(
                    NameHandler.parse_params(sci_group, keys=const.SURVEY_SCIENCE_GROUP_KEYS), raise_error=True
                ).items()
            )
            # return tuple_key
            return "_".join(tuple_key)

        result = []

        # off-date groups first: no processing time
        for sci_group in off_date_groups:
            mbias = collapse(cls(sci_group).preprocess.bias, raise_error=True)
            mdark = collapse(cls(sci_group).preprocess.dark, raise_error=True)
            mflat = collapse(cls(sci_group).preprocess.flat, raise_error=True)
            key = get_key(sci_group)
            result.append(
                (
                    ([], [], []),  # empty if no raw bias/dark/flat -> search them in masterframe_dir
                    (mbias, mdark, mflat),  # master bdf search template
                    {key: (sci_group, cls(sci_group).conjugate)},  # singleton science file
                    # {key: PathHandler(sci_group).conjugate},
                )
            )

        # on-date groups: sorted by increasing sci group numbers for each bdf triple
        for _key, entry in sorted(calib_map.items(), key=lambda kv: len(kv[1]["sci"])):
            raw_bias = entry["bias"]
            raw_dark = entry["dark"]
            raw_flat = entry["flat"]

            mbias = collapse(cls(raw_bias).preprocess.bias, raise_error=True)
            mdark = collapse(cls(raw_dark).preprocess.dark, raise_error=True)
            mflat = collapse(cls(raw_flat).preprocess.flat, raise_error=True)

            sci_dict = {}
            for sci_group in entry["sci"]:
                key = get_key(sci_group)
                sci_dict[key] = (sci_group, cls(sci_group).conjugate)

            result.append(
                (
                    (raw_bias, raw_dark, raw_flat),
                    (mbias, mdark, mflat),
                    sci_dict,  # a dict of tuples: (science images in this on‐date group, processed images)
                )
            )

        return result

    @classmethod
    def weight_map_input(cls, zdf_list: list[str]):
        """Returns d_m_file, f_m_file, sig_z_file, sig_f_file"""
        z_m_file, d_m_file, f_m_file = (cls(s).preprocess.masterframe for s in zdf_list)
        sig_z_file = z_m_file.replace("bias", "biassig")
        sig_f_file = f_m_file.replace("flat", "flatsig")
        return d_m_file, f_m_file, sig_z_file, sig_f_file


class PathPreprocess(AutoCollapseMixin, AutoMkdirMixin):
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
        if isinstance(self._parent.masterframe_dir, str):
            return [self._parent.masterframe_dir] * len(self._parent.name.masterframe_basename)
        return self._parent.masterframe_dir

    @property
    def bias(self):
        """Given mixed raw calib images, pick up only bias frames and give their
        masterframe counterparts"""
        # names = NameHandler(self._parent._input_files)
        # return os.path.join(self._parent.masterframe_dir, names.masterframe_basename[0])
        result = []
        for typ, d, s in zip(self._parent.name.type, self._masterframe_dir, self._parent.name.masterframe_basename):
            if typ[1] == "bias":
                result.append(os.path.join(d, s))
            # if typ[1] == 'dark' or typ[1] =='flat':
            #     pass
            elif typ[1] == "science":
                result.append(os.path.join(d, s[0]))
            else:
                result.append(None)

        return result
        # return [
        #     os.path.join(d, s) if typ[1] == "bias" else os.path.join(d, s[0])
        #     for typ, d, s in zip(
        #         self._parent.name.type, self._parent.masterframe_dir, self._parent.name.masterframe_basename
        #     )
        # ]

    @property
    def dark(self):
        # names = NameHandler(self._parent._input_files)
        # return [
        #     (
        #         os.path.join(self._parent.masterframe_dir, s)
        #         if typ[1] == "dark"
        #         else os.path.join(self._parent.masterframe_dir, s[1])
        #     )
        #     for typ, s in zip(names.type, names.masterframe_basename)
        # ]
        result = []
        for typ, d, s in zip(self._parent.name.type, self._masterframe_dir, self._parent.name.masterframe_basename):
            if typ[1] == "dark":
                result.append(os.path.join(d, s))
            # if typ[1] == 'dark' or typ[1] =='flat':
            #     pass
            elif typ[1] == "science":
                result.append(os.path.join(d, s[1]))
            else:
                result.append(None)

        return result

    @property
    def flat(self):
        # names = NameHandler(self._parent._input_files)
        # return [
        #     (
        #         os.path.join(self._parent.masterframe_dir, s)
        #         if typ[1] == "flat"
        #         else os.path.join(self._parent.masterframe_dir, s[2])
        #     )
        #     for typ, s in zip(names.type, names.masterframe_basename)
        # ]
        result = []
        for typ, d, s in zip(self._parent.name.type, self._masterframe_dir, self._parent.name.masterframe_basename):
            if typ[1] == "flat":
                result.append(os.path.join(d, s))
            # if typ[1] == 'dark' or typ[1] =='flat':
            #     pass
            elif typ[1] == "science":
                result.append(os.path.join(d, s[2]))
            else:
                result.append(None)

        return result

    @property
    def masterframe(self):
        """Generates on-date master zdf for sci input"""

        if self._parent._single:
            typ = self._parent.name.type
            if typ[1] == "science":
                return [os.path.join(self._masterframe_dir[0], s) for s in self._parent.name.masterframe_basename]
            else:
                return os.path.join(self._masterframe_dir[0], self._parent.name.masterframe_basename)

        result = []
        for typ, mfdir, ss in zip(
            self._parent.name.type, self._masterframe_dir, self._parent.name.masterframe_basename
        ):
            if typ[1] == "science":
                result.append([os.path.join(mfdir, s) for s in ss])
            else:
                result.append(os.path.join(mfdir, ss))
        return result

        # return bjoin(self._parent.masterframe_dir, self._parent.name.masterframe_basename)


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
        input = self._parent.basename
        if isinstance(input, list):
            return [os.path.join(self.tmp_dir, add_suffix(add_suffix(s, "prep"), "cat")) for s in input]
        else:
            return os.path.join(self.tmp_dir, add_suffix(add_suffix(input, "prep"), "cat"))

    @property
    def main_catalog(self):
        """intermediate sextractor output"""
        input = self._parent.basename
        if isinstance(input, list):
            return [os.path.join(self.tmp_dir, swap_ext(s, "cat")) for s in input]
        else:
            return os.path.join(self.tmp_dir, swap_ext(input, "cat"))

    @property
    def final_catalog(self):
        """final pipeline output catalog"""
        input = self._parent.basename
        if isinstance(input, list):
            return [os.path.join(self._parent.image_dir, add_suffix(s, "cat")) for s in input]
        else:
            return os.path.join(self._parent.image_dir, add_suffix(input, "cat"))

    def __getattr__(self, name):
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

    @property
    def stacked_image_basename(self):
        names = NameHandler(self._parent._input_files)
        _ = collapse(names.type, raise_error=True)  # ensure input images are coherent
        total_exptime = np.sum(names.exptime)
        fname = f"{names.obj_collapse}_{names.filter_collapse}_{names.unit_collapse}_{names.unit_collapse}_{names.datetime[-1]}_{format_exptime(total_exptime, type='stacked')}_coadd.fits"
        return fname

    @property
    def daily_stacked_image(self):
        return os.path.join(self._parent.daily_stacked_dir, self.stacked_image_basename)

    @property
    def stacked_image(self):
        return os.path.join(collapse(self._parent.stacked_dir, raise_error=True), self.stacked_image_basename)


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
