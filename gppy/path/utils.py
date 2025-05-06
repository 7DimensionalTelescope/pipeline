import os
from astropy.io import fits


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


# class NameHandler:
class Path7DS:
    """
    Parser for 7DT fits file names
    obsdata: 7DT11_20250102_050704_T00223_m425_1x1_100.0s_0001.fits
    (outdated) processed: calib_7DT11_T09282_20241017_060927_m425_100.fits
    processed: T09282_m425_7DT11_100s_20241017_060927.fits
    """

    def __init__(self, file: str):
        file = str(file)
        self.path = os.path.abspath(file)
        self.basename = os.path.basename(file)
        self.stem, self.ext = os.path.splitext(self.basename)
        if self.ext != ".fits":
            raise ValueError("Not a FITS file")
        self.parts = self.stem.split("_")

        self._n_binning = None  # will be filled lazily
        self.exists = os.path.exists(self.path)

        if self.type == "raw_image":
            self.parse_obsdata()
        else:
            self.parse_processed()

    def __repr__(self):
        return str(self.file)

    @property
    def type(self):
        cat_suffix = "_cat.fits"
        weight_suffix = "_weight.fits"

        # raw
        if self.stem.startswith("7DT"):
            return "raw_image"
        # processsed
        else:
            if "subt" in self.stem:
                image_type = "subtracted_image"
            elif "coadd" in self.stem:
                image_type = "coadded_image"
            else:
                image_type = "processed_image"

            if self.stem.endswith(cat_suffix):
                product_type = "_catalog"
            if self.stem.endswith(weight_suffix):
                product_type = "_weight"

            return image_type + product_type

    @property
    def n_binning(self) -> int:
        """lazy"""
        if self._n_binning is not None:
            return self._n_binning
        header = fits.getheader(self.file)
        return header["XBINNING"]

    @property
    def datetime(self):
        return self.date + "_" + self.hms

    @property
    def raw_basename(self):
        return f"{self.unit}_{self.date}_{self.hms}_{self.obj}_{self.filter}_{self.n_binning}x{self.n_binning}_{self.exptime}.fits"

    @property
    def processed_basename(self):
        return f"{self.obj}_{self.filter}_{self.unit}_{self.exptime}_{self.date}_{self.hms}.fits"

    @property
    def conjugate(self):
        if self.type == "raw_image":
            return self.processed_basename
        else:
            return self.raw_basename

    def parse_obsdata(self):
        self.unit = self.parts[0]
        self.date = self.parts[1]
        self.hms = self.parts[2]
        self.obj = self.parts[3]
        self.filter = self.parts[4]
        self._n_binning = int(self.parts[5][0])
        self.exptime = self.parts[6]

    def parse_processed(self):
        self.obj = self.parts[0]
        self.filter = self.parts[1]
        self.unit = self.parts[2]
        self.exptime = self.parts[3]
        self.date = self.parts[4]
        self.hms = self.parts[5]
