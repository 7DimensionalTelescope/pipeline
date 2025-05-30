import os
from collections import defaultdict
from pathlib import Path
from astropy.io import fits
from ..utils import subtract_half_day, get_camera, get_gain, get_header, equal_on_keys, collapse
from .. import const
from .utils import strip_binning, format_binning, strip_exptime, format_exptime, strip_gain


# # class Path7DS:
# class NameHandlerDeprecated:
#     """
#     Parser for 7DT fits file names that relies on strig split("_"), which is
#     faster than a full regex.
#     obsdata: 7DT11_20250102_050704_T00223_m425_1x1_100.0s_0001.fits
#     (outdated) processed: calib_7DT11_T09282_20241017_060927_m425_100.fits
#     processed: T09282_m425_7DT11_100s_20241017_060927.fits
#     """

#     def __init__(self, filename: str | Path):
#         if isinstance(filename, Path):
#             filename = str(filename)
#         elif not isinstance(filename, str):
#             raise TypeError("Input must be str or pathlib.Path")

#         # pathlib-like attr
#         self.path = os.path.abspath(filename)
#         self.basename = os.path.basename(filename)
#         self.stem, self.ext = os.path.splitext(self.basename)
#         if self.ext != ".fits":
#             raise ValueError("Not a FITS file")
#         self.parts = self.stem.split("_")

#         self._n_binning = None  # will be filled lazily
#         self.exists = os.path.exists(self.path)

#         if self.type == "raw_image":
#             self.parse_raw()
#         else:
#             self.parse_processed()

#     def __repr__(self):
#         return str(self.file)

#     @property
#     def type(self):
#         # raw
#         if self.stem.startswith("7DT"):
#             return "raw_image"
#         # processsed
#         else:
#             if "subt" in self.stem:
#                 image_type = "subtracted_image"
#             elif "coadd" in self.stem:
#                 image_type = "coadded_image"
#             else:
#                 image_type = "processed_image"

#             product_type = ""
#             if self.stem.endswith("_cat.fits"):
#                 product_type = "catalog"
#             if self.stem.endswith("_weight.fits"):
#                 product_type = "weight"

#             return "_".join([image_type, product_type])

#     @property
#     def n_binning(self) -> int:
#         """lazy"""
#         if self._n_binning is not None:
#             return self._n_binning
#         header = fits.getheader(self.path)
#         return header["XBINNING"]

#     @property
#     def datetime(self):
#         return self.date + "_" + self.hms

#     def nightdate(self):
#         return subtract_half_day(self.datetime)

#     @property
#     def raw_basename(self):
#         return f"{self.unit}_{self.date}_{self.hms}_{self.obj}_{self.filter}_{format_binning(self.n_binning)}_{format_exptime(self.exptime, type=self.type)}.fits"

#     @property
#     def processed_basename(self):
#         return f"{self.obj}_{self.filter}_{self.unit}_{self.date}_{self.hms}_{format_exptime(self.exptime, type=self.type)}.fits"

#     @property
#     def conjugate(self):
#         if self.type == "raw_image":
#             return self.processed_basename
#         else:
#             return self.raw_basename

#     def parse_raw(self):
#         """e.g., 7DT11_20250102_050704_T00223_m425_1x1_100.0s_0001.fits"""
#         self.unit = self.parts[0]
#         self.date = self.parts[1]
#         self.hms = self.parts[2]
#         self.obj = self.parts[3]
#         self.filter = self.parts[4]
#         self._n_binning = strip_binning(self.parts[5])
#         self.exptime = strip_exptime(self.parts[6])

#     def parse_processed(self):
#         self.obj = self.parts[0]
#         self.filter = self.parts[1]
#         self.unit = self.parts[2]
#         self.exptime = strip_exptime(self.parts[3])
#         self.date = self.parts[4]
#         self.hms = self.parts[5]


# vectorized version of above
class NameHandler:
    """
    Parser for 7DT fits file names that relies on strig split("_"), which is
    faster than a full regex.

    Accepts:
      - single filename (str or Path)
      - list/tuple of filenames (mix of str/Path OK)

    All attributes (`date`, `hms`, `obj`, etc.) will be either a single
    value or a list of values, matching your input.
    """

    def __init__(self, filenames: str | Path | list[str] | list[Path]):
        # --- 1. Normalize input to a list of strings ---
        if isinstance(filenames, (str, Path)):
            files = [str(filenames)]
            self._single = True
        elif isinstance(filenames, (list, tuple)):
            files = [str(f) for f in filenames]
            self._single = len(files) == 1
        else:
            raise TypeError("Input must be str, Path, or list/tuple of them")

        # --- 2. Build the filesystem-related attributes as lists ---
        self.abspath = [os.path.abspath(f) for f in files]
        self.basename = [os.path.basename(p) for p in self.abspath]
        self.stem, self.ext = (list(x) for x in zip(*(os.path.splitext(b) for b in self.basename)))
        # if any(ext != ".fits" for ext in self.ext):
        #     raise ValueError("One or more inputs are not FITS files")

        self.parts = [stem.split("_") for stem in self.stem]
        self.exists = [os.path.exists(p) for p in self.abspath]

        # --- 3. Determine raw vs processed for each input ---
        self.type = [self._detect_type(stem) for stem in self.stem]

        # --- 4. Parse each file into its components ---
        units, dates, hmses, objs, filters, nbinnings, exptimes, gains, cameras = [], [], [], [], [], [], [], [], []
        for parts, typ in zip(self.parts, self.type):
            if "raw" in typ:
                parsing_func = self._parse_raw
            elif "master" in typ:
                parsing_func = self._parse_master
            else:
                parsing_func = self._parse_processed

            unit, date, hms, obj, filte, nbin, exptime, gain, camera = parsing_func(parts)

            units.append(unit)
            dates.append(date)
            hmses.append(hms)
            objs.append(obj)
            filters.append(filte)
            nbinnings.append(nbin)
            exptimes.append(exptime)
            gains.append(gain)
            cameras.append(camera)

        if self._single:
            self.abspath = self.abspath[0]
            self.basename = self.basename[0]
            self.stem = self.stem[0]
            self.ext = self.ext[0]
            self.parts = self.parts[0]
            self.type = self.type[0]
            self.exists = self.exists[0]

        # --- 5. Attach them as lists (or scalar if single) ---
        self.unit = units if not self._single else units[0]
        self.date = dates if not self._single else dates[0]
        self.hms = hmses if not self._single else hmses[0]
        self.obj = objs if not self._single else objs[0]
        self.filter = filters if not self._single else filters[0]
        self._n_binning = nbinnings if not self._single else nbinnings[0]
        self._gain = gains if not self._single else gains[0]
        self._camera = cameras if not self._single else cameras[0]
        self.exptime = exptimes if not self._single else exptimes[0]
        self.exposure = self.exptime

    def __repr__(self):
        # when list: show first few
        if hasattr(self, "_single") and not self._single:
            return f"<NameHandler of {len(self.abspath) if not self._single else 1} files>"
        return f"<NameHandler {self.basename}>"

    @staticmethod
    def _detect_type(stem: str) -> tuple:
        """Classify a filename stem into a 5-component tuple.

        Order of components:
        0. raw / master / calibrated
        1. bias / dark / flat / science
        2. None / single / coadded       (None when a master frame)
        3. None / difference             (None when not a diff)
        4. image / weight / catalog
        """
        types = ()

        # raw/master/processed
        if stem.startswith("7DT"):
            types += ("raw",)
        elif stem.startswith(("bias", "dark", "flat")):
            types += ("master",)
        else:
            types += ("calibrated",)  # processed")

        # calib/sci
        stem_lower = stem.lower()
        if "bias" in stem_lower:
            types += ("bias",)
        elif "dark" in stem_lower:
            types += ("dark",)
        elif "flat" in stem_lower:
            types += ("flat",)
        else:
            types += ("science",)

        # single/coadd
        if "master" in types:
            types += (None,)
        elif "coadd" in stem:
            types += ("coadded",)
        else:
            types += ("single",)

        # diff
        if "diff" in stem:
            types += ("difference",)  # subtracted")
        else:
            types += (None,)

        # image/weight/cat
        if stem.endswith("_cat"):
            types += ("catalog",)
        elif stem.endswith("_weight"):
            types += ("weight",)
        else:
            types += ("image",)

        return types

    # @property
    # def types_str(self):
    #     """Return the type as a string."""
    #     if self._single:
    #         return "_".join(self.types)
    #     else:
    #         return ["_".join([t for t in types if t]) for types in self.types]

    def __getattr__(self, name):
        # syntactic sugar
        if name.endswith("_to_string"):
            base = name[:-10]
            val = getattr(self, base)
            if isinstance(val, list):
                # return joined type names
                if len(val) > 1:
                    return ["_".join([t for t in types if t]) for types in val]
                else:
                    return "_".join(val)

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

        raise AttributeError(f"{self.__class__.__name__!s} has no attribute {name!r}")

    @property
    def datetime(self):
        # if list-mode, this concatenates elementwise
        if self.hms:
            if isinstance(self.date, list):
                return [d + "_" + h if h else None for d, h in zip(self.date, self.hms)]
            return self.date + "_" + self.hms
        else:
            raise ValueError("Unable to construct datetime: HMS unavailable")

    @property
    def nightdate(self):
        # works in both modes
        if isinstance(self.date, list):
            return [subtract_half_day(dt) for dt in self.date]
        return subtract_half_day(self.date)

    @property
    def n_binning(self):
        key = const.HEADER_KEY_MAP["n_binning"]

        # single‐file mode
        if getattr(self, "_single", False):
            # if we already parsed a binning, return it
            if self._n_binning is not None:
                return self._n_binning
            # otherwise read it from the one FITS header
            else:
                return get_header(self.abspath, force_return=True).get(key, None)

        # multi‐file mode: build a list, using header only where needed
        nbins = []
        for nb, p in zip(self._n_binning, self.abspath):
            if nb is not None:
                nbins.append(nb)
            else:
                nbins.append(get_header(p, force_return=True).get(key, None))

        return nbins

    @property
    def gain(self):
        """
        returns None if inexistent
        self._gain is [None, None, ...]
        """
        if getattr(self, "_single", False):
            return get_gain(self.abspath)
        else:
            return [get_gain(p) for p in self.abspath]

    @property
    def camera(self):
        """
        returns None if inexistent
        self._camera is [None, None, ...]
        """
        if (self._single and self._camera is not None) or (
            not self._single and all(c is not None for c in self._camera)
        ):
            return self._camera

        if getattr(self, "_single", False):
            return get_camera(self.abspath)
        else:
            return [get_camera(u) for u in self.abspath]

    @property
    def pixscale(self):
        if getattr(self, "_single", False):
            return self.n_binning * const.PIXSCALE
        else:
            return [nbin * const.PIXSCALE for nbin in self.n_binning]

    @property
    def raw_basename(self):
        def make(unit, date, hms, obj, filte, nbin, exptime):
            return f"{unit}_{date}_{hms}_{obj}_{filte}_{format_binning(nbin)}_{format_exptime(exptime, type=self.type)}.fits"

        if getattr(self, "_single", False):
            return make(self.unit, self.date, self.hms, self.obj, self.filter, self.n_binning, self.exptime)

        return [
            make(unit, date, hms, obj, filte, nbin, exptime)
            for unit, date, hms, obj, filte, nbin, exptime in zip(
                self.unit, self.date, self.hms, self.obj, self.filter, self.n_binning, self.exptime
            )
        ]

    @staticmethod
    def _parse_raw(parts):
        # returns (unit, date, hms, obj, filter, n_binning, exptime)
        unit = parts[0]
        date = parts[1]
        hms = parts[2]
        obj = parts[3]
        filt = parts[4]
        nb = strip_binning(parts[5])
        exptime = strip_exptime(parts[6])
        gain = None
        camera = None
        return unit, date, hms, obj, filt, nb, exptime, gain, camera

    @property
    def processed_basename(self):
        def make(obj, filte, unit, date, hms, exptime):
            return f"{obj}_{filte}_{unit}_{date}_{hms}_{format_exptime(exptime, type=self.type)}.fits"

        if getattr(self, "_single", False):
            return make(self.obj, self.filter, self.unit, self.date, self.hms, self.exptime)

        return [
            make(obj, filte, unit, date, hms, exptime)
            for obj, filte, unit, date, hms, exptime in zip(
                self.obj, self.filter, self.unit, self.date, self.hms, self.exptime
            )
        ]

    @staticmethod
    def _parse_processed(parts):
        # returns (unit, date, hms, obj, filter, n_binning, exptime)
        obj = parts[0]
        filt = parts[1]
        unit = parts[2]
        date = parts[3]
        hms = parts[4]
        exptime = strip_exptime(parts[5])
        # we don’t have binning in processed names, so you can either
        # leave it None or derive from header later:
        nb = None
        gain = None
        camera = None
        return unit, date, hms, obj, filt, nb, exptime, gain, camera

    @property
    def masterframe_basename(self):  # , typ, quality):
        """works for mixed calibration file types"""

        def make(typ, filte, unit, date, exptime, nbin, gain, camera):
            if typ == "bias":
                quality = None
            elif typ == "dark":
                quality = format_exptime(exptime, type=typ)
            elif typ == "flat":
                quality = filte
            elif typ == "science":
                quality = [None, format_exptime(exptime, type=typ), filte]
            else:
                raise ValueError("Invalid type for masterframe_basename")

            if not isinstance(quality, list):  # master calib name for raw calib input
                infolist = [typ, quality, unit, date, format_binning(nbin), f"gain{gain}", camera]
                return "_".join([s for s in infolist if s is not None]) + ".fits"

            else:  # master calib name for raw sci input
                calib_bundles = []
                for typ, q in zip(["bias", "dark", "flat"], quality):
                    infolist = [typ, q, unit, date, format_binning(nbin), f"gain{gain}", camera]
                    calib_bundles.append("_".join([s for s in infolist if s is not None]) + ".fits")
                return tuple(calib_bundles)

        if getattr(self, "_single", False):
            return make(
                self.type[1], self.filter, self.unit, self.date, self.exptime, self.n_binning, self.gain, self.camera
            )

        return [
            make(typ[1], filte, unit, date, exptime, nbin, gain, camera)
            for typ, filte, unit, date, exptime, nbin, gain, camera in zip(
                self.type, self.filter, self.unit, self.date, self.exptime, self.n_binning, self.gain, self.camera
            )
        ]

    @staticmethod
    def _parse_master(parts):
        # returns (unit, date, hms, obj, filter, n_binning, exptime)
        obj = parts[0]
        if len(parts) == 6:
            offset = 1  # bias
        else:
            offset = 0  # dark, flat

        exptime = None
        filt = None
        if not offset:
            quality = parts[1]
            if quality.endswith("s"):
                exptime = strip_exptime(quality)
            else:
                filt = quality

        unit = parts[2 - offset]
        date = parts[3 - offset]
        nb = strip_binning(parts[4 - offset])
        gain = strip_gain(parts[5 - offset])
        camera = parts[6 - offset]
        hms = None
        return unit, date, hms, obj, filt, nb, exptime, gain, camera

    @property
    def config_stem(self):
        def make(obj, filte, unit, date):
            return f"{obj}_{filte}_{unit}_{date}"

        if getattr(self, "_single", False):
            return make(self.obj, self.filter, self.unit, self.date)

        return [
            make(obj, filte, unit, date) for obj, filte, unit, date, in zip(self.obj, self.filter, self.unit, self.date)
        ]

    @property
    def conjugate(self):
        """switch between raw and processed (existing) file paths"""
        single = getattr(self, "_single", False)

        # wrap scalar basenames into lists when needed
        raw_list = [self.raw_basename] if single else self.raw_basename
        proc_list = [self.processed_basename] if single else self.processed_basename

        # pick the “other” basename for each entry
        conj_list = [
            proc_bn if "raw" in typ else raw_bn for typ, raw_bn, proc_bn in zip(self.type, raw_list, proc_list)
        ]

        # unwrap for single-file mode
        return conj_list[0] if single else conj_list

    def to_dict(self, keys=None):
        keys = keys or const.ALL_GROUP_KEYS

        # grab the raw attributes for each key
        values = [getattr(self, key) for key in keys]

        if getattr(self, "_single", False):
            # single-file: zip the keys to their scalar values
            return dict(zip(keys, values))

        # multi-file: each attribute is a list; zip them to rows of dicts
        return [dict(zip(keys, row)) for row in zip(*values)]

    def get_grouped_files(self, keys=None):
        """
        Uses ALL_GROUP_KEYS by default, but you can specify other keys
        Output is a dict where keys are tuples of (type, obs_params) and values are lists of file paths.

        e.g.,
        name = NameHandler(flist)
        grouped_files = name.get_grouped_files()
        typ = next(iter(grouped_files))[0][0]
        obs_params = dict(next(iter(grouped_files))[0][1])
        """
        if not isinstance(self.abspath, list) or len(self.abspath) <= 1:
            return self.abspath[0]

        # groups = defaultdict(list)
        #     groups[key].append(str(f))
        groups = dict()
        for typ, f, obs_params in zip(self.type, self.abspath, self.to_dict(keys=keys)):
            key = (typ, tuple(obs_params.items()))  # sorted()  # check type to differentiate raw/proc with same params
            # key = tuple(obs_params.items())
            groups.setdefault(key, []).append(str(f))

        return groups

    @classmethod
    def parse_params(cls, files, keys=None):
        names = cls(files)
        return names.to_dict(keys=keys)

    @classmethod
    def find_calib_for_sci(cls, files) -> (list[str], list[tuple], (list[list], list[list], list[list])):
        """
        e.g., files = glob("/data/pipeline_reform/obsdata_test/7DT11/2025-01-01_gain2750/*.fits")
        sci_files, on_date_calib = NameHandler.find_calib_for_sci(flist)

        Returns
        -------
            sci_files
                list of science image files
            on_date_calib
                list of tuples of (on_date_bias, on_date_dark, on_date_flat)
                which corresponds to sci_files
            off_date_calib
                tuple of respectively grouped (bias, dark, flat)
        """
        if not isinstance(files, list) or len(files) <= 1:
            return files

        names = cls(files)
        grouped_files = names.get_grouped_files()

        bias, dark, flat, sci = {}, {}, {}, {}

        for k, v in grouped_files.items():  # k[0] is file type tuple, k[1] is obs_param dict
            if k[0][1] == "bias":
                bias[k] = v
            if k[0][1] == "dark":
                dark[k] = v
            if k[0][1] == "flat":
                flat[k] = v
            if k[0][1] == "science":
                sci[k] = v

        sci_files = []
        on_date_calib = []
        used_bias_keys = set()
        used_dark_keys = set()
        used_flat_keys = set()

        # On-date calib frames
        for key, val in sci.items():
            key_sci = dict(key[1])
            sci_files.append(val)

            # On-date calibration collection with group tracking
            on_bias, on_dark, on_flat = [], [], []

            for k, v in bias.items():
                if equal_on_keys(dict(k[1]), key_sci, const.BIAS_GROUP_KEYS):
                    on_bias.extend(v)
                    used_bias_keys.add(k)

            for k, v in dark.items():
                if equal_on_keys(dict(k[1]), key_sci, const.DARK_GROUP_KEYS):
                    on_dark.extend(v)
                    used_dark_keys.add(k)

            for k, v in flat.items():
                if equal_on_keys(dict(k[1]), key_sci, const.FLAT_GROUP_KEYS):
                    on_flat.extend(v)
                    used_flat_keys.add(k)

            on_date_calib.append((on_bias, on_dark, on_flat))

        # Off-date calib frames (not used by any on-date science images)
        unused_bias_keys = [k for k in bias.keys() if k not in used_bias_keys]  # order-preserving
        unused_dark_keys = [k for k in dark.keys() if k not in used_dark_keys]
        unused_flat_keys = [k for k in flat.keys() if k not in used_flat_keys]

        unused_bias_dict = defaultdict(list)
        unused_dark_dict = defaultdict(list)
        unused_flat_dict = defaultdict(list)

        for key_tuple in unused_bias_keys:
            bias_key_dict = dict(key_tuple[1])
            bias_key = tuple({k: v for k, v in bias_key_dict.items() if k in const.BIAS_GROUP_KEYS}.items())
            unused_bias_dict[bias_key].extend(bias[key_tuple])

        for key_tuple in unused_dark_keys:
            dark_key_dict = dict(key_tuple[1])
            dark_key = tuple({k: v for k, v in dark_key_dict.items() if k in const.DARK_GROUP_KEYS}.items())
            unused_dark_dict[dark_key].extend(dark[key_tuple])

        for key_tuple in unused_flat_keys:
            flat_key_dict = dict(key_tuple[1])
            flat_key = tuple({k: v for k, v in flat_key_dict.items() if k in const.FLAT_GROUP_KEYS}.items())
            unused_flat_dict[flat_key].extend(flat[key_tuple])

        grouped_unused_bias = [v for v in unused_bias_dict.values()]
        grouped_unused_dark = [v for v in unused_dark_dict.values()]
        grouped_unused_flat = [v for v in unused_flat_dict.values()]

        off_date_calib = (grouped_unused_bias, grouped_unused_dark, grouped_unused_flat)

        # off_date_calib = []
        # for key, val in sci.items():
        #     sci_files.append(val)
        #     key_sci = dict(key[1])

        #     # on-date collection
        #     on_date_bias = [
        #         item for k, v in bias.items() if equal_on_keys(dict(k[1]), key_sci, const.BIAS_GROUP_KEYS) for item in v
        #     ]  # get a flattened list, not [[], [], ..., []]

        #     on_date_dark = [
        #         item for k, v in dark.items() if equal_on_keys(dict(k[1]), key_sci, const.DARK_GROUP_KEYS) for item in v
        #     ]

        #     on_date_flat = [
        #         item for k, v in flat.items() if equal_on_keys(dict(k[1]), key_sci, const.FLAT_GROUP_KEYS) for item in v
        #     ]

        #     on_date_calib.append((on_date_bias, on_date_dark, on_date_flat))

        return sci_files, on_date_calib, off_date_calib
