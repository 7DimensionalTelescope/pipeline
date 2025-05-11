import os
from pathlib import Path
from astropy.io import fits
from ..utils import subtract_half_day
from .utils import strip_binning, format_binning, strip_exptime, format_exptime


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


# # vectorized version of above
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
        self.path = [os.path.abspath(f) for f in files]
        self.basename = [os.path.basename(p) for p in self.path]
        stems_exts = [os.path.splitext(b) for b in self.basename]
        self.stem, self.ext = zip(*stems_exts)
        if any(ext != ".fits" for ext in self.ext):
            raise ValueError("One or more inputs are not FITS files")

        self.parts = [stem.split("_") for stem in self.stem]
        self.exists = [os.path.exists(p) for p in self.path]

        # --- 3. Determine raw vs processed for each input ---
        self.types = [self._detect_type(stem) for stem in self.stem]

        # --- 4. Parse each file into its components ---
        units, dates, hmses, objs, filters, nbinnings, exptimes = [], [], [], [], [], [], []
        for parts, typ in zip(self.parts, self.types):
            if typ == "raw_image":
                unit, date, hms, obj, filte, nbin, exptime = self._parse_raw_parts(parts)
            else:
                unit, date, hms, obj, filte, nbin, exptime = self._parse_processed_parts(parts)
            units.append(unit)
            dates.append(date)
            hmses.append(hms)
            objs.append(obj)
            filters.append(filte)
            nbinnings.append(nbin)
            exptimes.append(exptime)

        # --- 5. Attach them as lists (or scalar if single) ---
        self.unit = units if not self._single else units[0]
        self.date = dates if not self._single else dates[0]
        self.hms = hmses if not self._single else hmses[0]
        self.obj = objs if not self._single else objs[0]
        self.filter = filters if not self._single else filters[0]
        self._n_binning = nbinnings if not self._single else nbinnings[0]
        self.exptime = exptimes if not self._single else exptimes[0]
        self.exposure = self.exptime
        self.exp = self.exptime

    def __repr__(self):
        # when list: show first few
        if hasattr(self, "_single") and not self._single:
            return f"<NameHandler of {len(self.path)} files>"
        return f"<NameHandler {self.basename}>"

    @staticmethod
    def _detect_type(stem: str) -> str:
        # raw
        if stem.startswith("7DT"):
            return "raw_image"
        # processed
        if "subt" in stem:
            image_type = "subtracted_image"
        elif "coadd" in stem:
            image_type = "coadded_image"
        else:
            image_type = "processed_image"

        product = ""
        if stem.endswith("_cat"):
            product = "_catalog"
        elif stem.endswith("_weight"):
            product = "_weight"

        return image_type + product

    @staticmethod
    def _parse_raw_parts(parts):
        # returns (unit, date, hms, obj, filter, n_binning, exptime)
        unit = parts[0]
        date = parts[1]
        hms = parts[2]
        obj = parts[3]
        filt = parts[4]
        nb = strip_binning(parts[5])
        exptime = strip_exptime(parts[6])
        return unit, date, hms, obj, filt, nb, exptime

    @staticmethod
    def _parse_processed_parts(parts):
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
        return unit, date, hms, obj, filt, nb, exptime

    @property
    def datetime(self):
        # if list-mode, this concatenates elementwise
        if isinstance(self.date, list):
            return [d + "_" + h for d, h in zip(self.date, self.hms)]
        return self.date + "_" + self.hms

    @property
    def nightdate(self):
        # works in both modes
        if isinstance(self.datetime, list):
            return [subtract_half_day(dt) for dt in self.datetime]
        return subtract_half_day(self.datetime)

    # @property
    # def n_binning(self):
    #     if isinstance(self._n_binning, list):
    #         return self._n_binning
    #     if self._n_binning is not None:
    #         return self._n_binning

    #     from astropy.io import fits

    #     header = fits.getheader(self.path)
    #     return header["XBINNING"]

    @staticmethod
    def _get_binning_from_header(fpath):
        from astropy.io import fits

        if not os.path.exists(fpath):
            raise FileNotFoundError("Supply an existing path to a processed image to get a conjugate of it.")
        hdr = fits.getheader(fpath)
        return hdr["XBINNING"]

    @property
    def n_binning(self):
        # single‐file mode
        if getattr(self, "_single", False):
            # if we already parsed a binning, return it
            if self._n_binning is not None:
                return self._n_binning
            # otherwise read it from the one FITS header
            else:
                return self._get_binning_from_header(self.path[0])

        # multi‐file mode: build a list, using header only where needed
        nbins = []
        for nb, p in zip(self._n_binning, self.path):
            if nb is not None:
                nbins.append(nb)
            else:
                nbins.append(self._get_binning_from_header(p))

        return nbins

    @property
    def raw_basename(self):
        def make(unit, date, hms, obj, filte, nbin, exptime):
            return f"{unit}_{date}_{hms}_{obj}_{filte}_{format_binning(nbin)}_{format_exptime(exptime, type=self.types)}.fits"

        if getattr(self, "_single", False):
            return make(self.unit, self.date, self.hms, self.obj, self.filter, self.n_binning, self.exptime)

        return [
            make(unit, date, hms, obj, filte, nbin, exptime)
            for unit, date, hms, obj, filte, nbin, exptime in zip(
                self.unit, self.date, self.hms, self.obj, self.filter, self.n_binning, self.exptime
            )
        ]

    @property
    def processed_basename(self):
        def make(obj, filte, unit, date, hms, exptime):
            return f"{obj}_{filte}_{unit}_{date}_{hms}_{format_exptime(exptime, type=self.types)}.fits"

        if getattr(self, "_single", False):
            return make(self.obj, self.filter, self.unit, self.date, self.hms, self.exptime)

        return [
            make(obj, filte, unit, date, hms, exptime)
            for obj, filte, unit, date, hms, exptime in zip(
                self.obj, self.filter, self.unit, self.date, self.hms, self.exptime
            )
        ]

    # @property
    # def conjugate(self):
    #     # get lists of basenames (or wrap scalars)
    #     raw_list = [self.raw_basename] if getattr(self, "_single", False) else self.raw_basename
    #     proc_list = [self.processed_basename] if getattr(self, "_single", False) else self.processed_basename

    #     conj_paths = []
    #     for typ, path, raw_bn, proc_bn in zip(self.type, self.path, raw_list, proc_list):
    #         # pick the opposite and join its directory
    #         target = proc_bn if typ == "raw_image" else raw_bn
    #         conj_paths.append(target)

    #     return conj_paths[0] if getattr(self, "_single", False) else conj_paths

    # @property
    # def conjugate(self):
    #     """Currently not working in the processed -> raw direction"""
    #     single = getattr(self, "_single", False)

    #     # helpers to wrap scalars as one‐element lists
    #     def wrap(x):
    #         return [x] if single else x

    #     types = wrap(self.type)
    #     paths = wrap(self.path)
    #     units = wrap(self.unit)
    #     dates = wrap(self.date)
    #     hmses = wrap(self.hms)
    #     objs = wrap(self.obj)
    #     filters = wrap(self.filter)
    #     nbins = wrap(self.n_binning)
    #     exps = wrap(self.exptime)

    #     conj_paths = []
    #     for typ, path, unit, date, hms, obj, filte, nbin, exptime in zip(
    #         types, paths, units, dates, hmses, objs, filters, nbins, exps
    #     ):
    #         if typ == "raw_image":
    #             # for raw → processed
    #             basename = f"{obj}_{filte}_{unit}_{date}_{hms}_{exptime}.fits"
    #         else:
    #             # for processed → raw
    #             basename = f"{unit}_{date}_{hms}_{obj}_{filte}_{nbin}x{nbin}_{exptime}.fits"
    #         conj_paths.append(basename)

    #     return conj_paths[0] if single else conj_paths

    @property
    def conjugate(self):
        single = getattr(self, "_single", False)

        # wrap scalar basenames into lists when needed
        raw_list = [self.raw_basename] if single else self.raw_basename
        proc_list = [self.processed_basename] if single else self.processed_basename

        # pick the “other” basename for each entry
        conj_list = [
            proc_bn if typ == "raw_image" else raw_bn for typ, raw_bn, proc_bn in zip(self.types, raw_list, proc_list)
        ]

        # unwrap for single-file mode
        return conj_list[0] if single else conj_list
