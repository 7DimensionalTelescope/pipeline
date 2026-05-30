"""In-memory snapshot of input image headers for ImCoadd.

``InputHeaderSet`` reads each input FITS header exactly once at construction
and exposes mask-aware aggregate quantities and a ``coadd_header`` property.
Downstream coadd code never re-reads FITS for header info; it mutates the
snapshot headers in place (e.g. ``zpscale`` stamps FLXSCALE, ``bkgsub`` stamps
BACKTYPE) so the snapshot stays the single source of truth.
"""

from __future__ import annotations

import os
from typing import Any, Iterable

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.table import Table
from astropy.coordinates import Angle

from ..const import REF_DIR
from ..utils import get_basename
from ..utils.tile import is_ris_tile, find_ris_tile
from .const import CORE_KEYS, HOMOGENEOUS_KEYS


class InputHeaderSet:
    """Snapshot of input image headers with a selection mask.

    Headers are kept mutable so processing steps can stamp per-image cards
    (FLXSCALE, BACKTYPE, ...) onto the snapshot. All aggregates and
    ``coadd_header`` reflect only the unmasked entries.
    """

    def __init__(self, names: list[str], headers: list[fits.Header]):
        if len(names) != len(headers):
            raise ValueError(f"names ({len(names)}) and headers ({len(headers)}) length mismatch")
        self._names: list[str] = list(names)
        self._headers: list[fits.Header] = list(headers)
        self._mask: np.ndarray = np.ones(len(self._names), dtype=bool)

    @classmethod
    def from_files(cls, paths: list[str]) -> "InputHeaderSet":
        return cls(
            names=[get_basename(p) for p in paths],
            headers=[fits.getheader(p) for p in paths],
        )

    # ---------- mask API ----------

    @property
    def mask(self) -> np.ndarray:
        """Copy of the active-input mask (True = included)."""
        return self._mask.copy()

    def set_mask(self, mask: Iterable[bool]) -> None:
        mask = np.asarray(list(mask), dtype=bool)
        if mask.shape != self._mask.shape:
            raise ValueError(f"mask shape {mask.shape} != expected {self._mask.shape}")
        self._mask = mask

    def exclude(self, image_names: Iterable[str]) -> None:
        drop = set(image_names)
        self._mask = np.array([n not in drop for n in self._names], dtype=bool) & self._mask

    def reset_mask(self) -> None:
        self._mask = np.ones(len(self._names), dtype=bool)

    # ---------- accessors (respect mask) ----------

    @property
    def names(self) -> list[str]:
        return [n for n, m in zip(self._names, self._mask) if m]

    @property
    def headers(self) -> list[fits.Header]:
        return [h for h, m in zip(self._headers, self._mask) if m]

    def __getitem__(self, key: int | str) -> fits.Header:
        if isinstance(key, int):
            image_name = self.names[key]
            return self._headers[self._names.index(image_name)]
        if isinstance(key, str):
            if key not in self.names:
                raise KeyError(f"image {key!r} not in unmasked set")
            return self._headers[self._names.index(key)]
        raise TypeError(f"InputHeaderSet indices must be int or str, not {type(key).__name__}")

    def __len__(self) -> int:
        return int(self._mask.sum())

    def __iter__(self):
        return iter(self.headers)

    def values(self, key: str) -> list:
        """Per-image values of ``key`` over the unmasked set (None where missing)."""
        return [h.get(key) for h in self.headers]

    def unique(self, key: str) -> list:
        """Distinct non-None values of ``key`` across unmasked inputs."""
        return list(set(v for v in self.values(key) if v is not None))

    def aggregate(self, key: str) -> Any:
        """Single representative value over unmasked inputs.

        Dispatch by belonging:
            - ``HOMOGENEOUS_KEYS``: assumed uniform -> first non-None value.
            - otherwise (inhomogeneous / unknown): derive a representative
              (numeric mean, "MIXED" for strings; None on type-heterogeneity).
        Returns None if the key is absent everywhere."""
        vals = self.values(key)
        non_none = [v for v in vals if v is not None]
        if not non_none:
            return None
        if key in HOMOGENEOUS_KEYS:
            return non_none[0]
        if all(v == non_none[0] for v in non_none):
            return non_none[0]
        # exclude bool (subclass of int) to avoid averaging True/False
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_none):
            return float(np.mean(non_none))
        if all(isinstance(v, str) for v in non_none):
            return "MIXED"
        return None

    def check_uniqueness(self, keys: Iterable[str], logger=None) -> None:
        """Log a warning for each ``key`` whose value differs across unmasked inputs."""
        for key in keys:
            if len(self.unique(key)) > 1:
                msg = f"Multiple {key} found. Using the first one."
                if logger is not None:
                    logger.warning(msg)
                else:
                    print(msg)

    # ---------- representative quantities (unmasked subset) ----------

    @property
    def obj(self) -> str:
        return self.unique("OBJECT")[0]

    @property
    def filter(self) -> str:
        return self.unique("FILTER")[0]

    @property
    def camera_gain(self):
        return self.unique("GAIN")[0]

    @property
    def total_exptime(self) -> float:
        return float(np.sum(self.values("EXPTIME")))

    @property
    def mean_mjd(self) -> float:
        return float(np.mean(self.values("MJD")))

    @property
    def mean_dateloc(self) -> str | None:
        """JD-averaged DATE-LOC as ISO string; None if absent."""
        vals = [v for v in self.values("DATE-LOC") if v is not None]
        if not vals:
            return None
        jds = [Time(v).jd for v in vals]
        return Time(float(np.mean(jds)), format="jd").isot

    @property
    def coadd_satur_level(self) -> float | None:
        """Conservative saturation: min over SATURATE * FLXSCALE.
        Requires FLXSCALE to have been stamped (e.g. by ``zpscale``)."""
        satur = self.values("SATURATE")
        flx = self.values("FLXSCALE")
        if any(s is None for s in satur) or any(f is None for f in flx):
            return None
        return float(np.min([s * f for s, f in zip(satur, flx)]))

    @property
    def coadd_egain(self) -> float | None:
        """Effective EGAIN for the coadd. Requires FLXSCALE stamped."""
        egain = self.values("EGAIN")
        flx = self.values("FLXSCALE")
        if any(e is None for e in egain) or any(f is None for f in flx):
            return None
        return float(np.sum([e / f for e, f in zip(egain, flx)]))

    @property
    def coadd_backtype(self) -> str | None:
        """BACKTYPE for the coadd; "MIXED" if inputs disagree; None before bkgsub stamps it."""
        return self.aggregate("BACKTYPE")

    # ---------- deprojection center for SWarp ----------

    @property
    def is_tile_center(self) -> bool:
        """True if OBJECT matches a RIS tile (predefined sky-grid center)."""
        return is_ris_tile(self.obj)

    @property
    def deprojection_center(self) -> str:
        """RA,Dec string for SWarp's CENTER_TYPE=MANUAL.
        Tile objects use the predefined sky-grid center; otherwise fall back
        to the first unmasked frame's OBJCTRA/OBJCTDEC."""
        obj = self.obj
        # 	Tile object (e.g. T01026)
        if self.is_tile_center:
            skygrid_table = Table.read(os.path.join(REF_DIR, "skygrid.fits"))
            idx_tile = skygrid_table["tile"] == find_ris_tile(obj)
            ra = Angle(skygrid_table["ra"][idx_tile][0], unit="deg")
            objra = ra.to_string(unit="hourangle", sep=":", pad=True)
            dec = Angle(skygrid_table["dec"][idx_tile][0], unit="deg")
            objdec = dec.to_string(unit="degree", sep=":", pad=True, alwayssign=True)
        # 	Non-Tile object
        else:
            objra = self[0]["OBJCTRA"]
            objdec = self[0]["OBJCTDEC"]
            # images from <~ 2024-03 has space-separated OBJCTRA/OBJCTDEC
            objra = objra.replace(" ", ":")
            objdec = objdec.replace(" ", ":")
        return f"{objra},{objdec}"

    # ---------- coadd metadata header ----------

    @property
    def coadd_header(self) -> fits.Header:
        """Coadd metadata header, freshly built from the unmasked snapshot.
        Carries no WCS; callers (e.g. coadd_with_numpy) overlay it onto a
        WCS-bearing base. Keys absent or value-None are skipped."""
        # 	Get Header info
        mjd = self.mean_mjd
        dateobs = Time(mjd, format="mjd").isot
        jd = Time(mjd, format="mjd").jd
        # gain = (2 / 3) * len(self) * self.camera_gain

        header = fits.Header()

        # 	Get Select Header Keys from Base Image
        # Aggregate from unmasked inputs: common value if all agree; mean for
        # numerics; "MIXED" for strings; drop on heterogeneous types.
        for key in CORE_KEYS:
            val = self.aggregate(key)
            if val is not None:
                header[key] = val

        # 	Additional Header Information
        # Specially handled: time keys (uniform mean), totals, and coadd-specific aggregates.
        keywords_to_update = {
            "DATE-OBS": (dateobs, "Time of observation (UTC) for coadded image"),
            "DATE-LOC": (self.mean_dateloc, "Time of observation (local) for coadded image"),
            "EXPTIME":  (self.total_exptime, "[s] Total exposure duration for coadded image"),
            "EXPOSURE": (self.total_exptime, "[s] Total exposure duration for coadded image"),
            "MJD":      (mjd, "Modified Julian Date at start of observations for coadded image"),
            "MJD-OBS":  (mjd, "Modified Julian Date at start of observations for coadded image"),
            "JD":       (jd, "Julian Date at start of observations for coadded image"),
            "SKYVAL":   (0, "SKY MEDIAN VALUE (Subtracted)"),
            "EGAIN":    (self.coadd_egain, "Effective EGAIN for coadded image (e-/ADU)"),  # swarp calculates it as GAIN, but irreproducible.
            "GAIN":     (self.camera_gain, "Gain from the camera configuration"),
            "SATURATE": (self.coadd_satur_level, "Conservative saturation level for coadded image"),  # let swarp handle this
            "BACKTYPE": (self.coadd_backtype, "Background subtraction type for coadded image"),
        }  # fmt: skip
        for key, (value, comment) in keywords_to_update.items():
            if value is not None:
                header[key] = (value, comment)

        # 	Names of coadded single images
        for nn, name in enumerate(self.names):
            header[f"IMG{nn:0>5}"] = (name, "single exposures")

        from .. import __version__

        header["PIPE_VER"] = (str(__version__), "Last Run Sciproc Pipeline Version")

        return header
