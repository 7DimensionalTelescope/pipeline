import os
import logging
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from astropy.io import fits

from ..const import STABLE_TYPE_RTNMAP
from ..path.path import PathHandler
from ..path.cam_tracker import get_camera_serial
from ..services.database import free_query
from ..services.database.sync import sync_image
from .utils import write_IMCMB_to_header, add_image_id, set_pipe_ver_in_header


@lru_cache(maxsize=1)
def _unit_night_cameras() -> pd.DataFrame:
    """Camera serial of every unit-night with a raw frame."""
    query = " UNION ".join(
        f"SELECT u.name, n.date FROM {table} f JOIN facility_unit u ON u.id = f.unit_id"
        " JOIN survey_night n ON n.id = f.night_id WHERE strpos(f.file_path, 'tmp') = 0"
        for table in ("survey_scienceframe", "survey_biasframe", "survey_darkframe", "survey_flatframe")
    )
    df = pd.DataFrame(free_query(query), columns=["unit", "nightdate"])
    df["nightdate"] = df.nightdate.astype(str)
    df["serial"] = [str(get_camera_serial(int(unit[3:]), night)) for unit, night in zip(df.unit, df.nightdate)]
    return df.sort_values(["nightdate", "unit"]).reset_index(drop=True)


def camera_nights(camera_serial: str | int, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Unit-nights of a camera by cam_tracker, within [start, end]."""
    df = _unit_night_cameras()
    df = df[df.serial == str(camera_serial)]
    if start is not None:
        df = df[df.nightdate >= start]
    if end is not None:
        df = df[df.nightdate <= end]
    return df.reset_index(drop=True)


def bias_sequences(
    camera_serial: str | int,
    gain: int = 2750,
    set_temp: float = -10.0,
    n_binning: int = 1,
    start: str | None = None,
    end: str | None = None,
) -> list[pd.DataFrame]:
    """Complete 9-frame bias sequences of a camera in time order."""
    nights = camera_nights(camera_serial, start, end)
    if nights.empty:
        return []
    rows = free_query(
        "SELECT DISTINCT ON (u.name, f.obstime) f.id, u.name, n.date, f.obstime, f.ccdtemp, f.egain, f.file_path"
        " FROM survey_biasframe f JOIN facility_unit u ON u.id = f.unit_id JOIN survey_night n ON n.id = f.night_id"
        " WHERE strpos(f.file_path, 'tmp') = 0 AND f.gain = %s AND f.binning_x = %s AND f.binning_y = %s"
        " AND f.set_ccdtemp = %s AND abs(f.ccdtemp - %s) <= 0.5 AND n.date BETWEEN %s AND %s"
        " ORDER BY u.name, f.obstime, n.date",
        [gain, n_binning, n_binning, set_temp, set_temp, nights.nightdate.min(), nights.nightdate.max()],
    )
    frames = pd.DataFrame(rows, columns=["id", "unit", "nightdate", "obstime", "ccdtemp", "egain", "file_path"])
    frames["nightdate"] = frames.nightdate.astype(str)
    frames["obstime"] = pd.to_datetime(frames.obstime, utc=True)
    frames = frames.merge(nights[["unit", "nightdate"]], on=["unit", "nightdate"])
    sequences = []
    for _, group in frames.groupby(["unit", "nightdate"]):
        group = group.sort_values("obstime").reset_index(drop=True)
        gap = group.obstime.diff().dt.total_seconds().to_numpy()[1:]
        for idx in np.split(np.arange(len(group)), np.flatnonzero(gap > 120) + 1):
            if len(idx) == 9:
                sequences.append(group.iloc[idx].reset_index(drop=True))
    return sorted(sequences, key=lambda s: s.obstime.iloc[0])


def select_sequences(
    sequences: list[pd.DataFrame], nseq: int = 20, logger=None
) -> list[tuple[pd.DataFrame, np.ndarray]]:
    """Evenly spread sequences whose frames all pass the strip QA, with their strip variance."""
    logger = logger or logging.getLogger(__name__)
    if not sequences:
        return []
    picks = np.unique(np.linspace(0, len(sequences) - 1, nseq).astype(int))
    chosen = []
    for i, stop in zip(picks, list(picks[1:]) + [len(sequences)]):
        for j in range(i, stop):
            var, good = _strip(sequences[j])
            if good.all():
                chosen.append((sequences[j], var))
                break
            logger.info(f"rtnmap: {_label(sequences[j])} has {int((~good).sum())} bad frames; next sequence")
        else:
            logger.warning(f"rtnmap: no complete sequence from {_label(sequences[i])} on")
    return chosen


def self_check(variances: list[np.ndarray]) -> np.ndarray:
    """Share of the other sequences whose flagged pixels (variance > 4 x median) match this one's, cosine >= 0.30."""
    if len(variances) == 1:
        return np.ones(1)
    flags = np.stack([(var > 4 * np.median(var))[:, ::4].ravel() for var in variances]).astype(np.float32)
    norm = np.sqrt(flags.sum(axis=1))
    cosine = flags @ flags.T / np.outer(norm, norm)
    return ((cosine >= 0.30).sum(axis=1) - 1) / (len(flags) - 1)


def build_rtnmap(
    camera_serial: str | int,
    gain: int = 2750,
    set_temp: float = -10.0,
    n_binning: int = 1,
    nseq: int = 20,
    start: str | None = None,
    end: str | None = None,
    register: bool = False,
    overwrite: bool = False,
    max_workers: int = 4,
    logger=None,
) -> str:
    """RTN map of a camera from its bias sequences: select, self-check, accumulate, write, optionally register."""
    logger = logger or logging.getLogger(__name__)
    nights = camera_nights(camera_serial, start, end)
    if nights.empty:
        raise ValueError(f"No nights of camera {camera_serial} between {start} and {end}")
    start = start or nights.nightdate.min()
    path = PathHandler.rtnmap(camera_serial, gain, n_binning, start)
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(path)
    chosen = select_sequences(bias_sequences(camera_serial, gain, set_temp, n_binning, start, end), nseq, logger)
    if not chosen:
        raise ValueError(f"No complete bias sequence of camera {camera_serial} at gain {gain}, {set_temp} C")

    share = self_check([var for _, var in chosen])
    kept, history = [], []
    for (frames, var), s in zip(chosen, share):
        if s >= 0.5:
            kept.append((frames, var))
        else:
            logger.warning(f"rtnmap: {_label(frames)} dropped, self-check matches {s:.0%} of the other sequences")
            history.append(f"Dropped {_label(frames)}: self-check matches {s:.0%} of the others")
    if not kept:
        raise ValueError(f"Every sequence of camera {camera_serial} failed the self-check")
    sigtyp = float(np.median([np.sqrt(np.median(var) / _chi2_median_factor(len(f))) for f, var in kept]))

    sequences = [frames for frames, _ in kept]
    raw_header = fits.getheader(sequences[0].file_path[0])
    shape = (raw_header["NAXIS2"], raw_header["NAXIS1"])
    blocks = [(y0, min(y0 + 400, shape[0])) for y0 in range(0, shape[0], 400)]
    ssw, dof, nexc, nframe = _accumulate(sequences, sigtyp, blocks, shape, max_workers, logger)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = ssw / dof[:, None]
    vartyp = float(np.nanmedian(v))

    frames = pd.concat(sequences, ignore_index=True)
    write_rtnmap(
        path,
        v / vartyp,
        nexc,
        np.broadcast_to(nframe[:, None], shape),
        list(frames.file_path),
        camera_serial=camera_serial,
        gain=gain,
        n_binning=n_binning,
        egain=float(np.median(frames.egain)),
        set_temp=set_temp,
        units=sorted(nights.unit.unique()),
        nights=list(frames.nightdate),
        nseq=len(sequences),
        sigtyp=sigtyp,
        vartyp=vartyp,
        sanity=len(sequences) >= nseq / 2,
        history=history,
        overwrite=overwrite,
    )
    logger.info(f"rtnmap: wrote {path}")
    if register:
        sync_image(path)
    return path


def write_rtnmap(
    path: str,
    ratio: np.ndarray,
    nexc: np.ndarray,
    nframe: np.ndarray,
    ingredients: list[str],
    *,
    camera_serial: str | int,
    gain: int,
    n_binning: int,
    egain: float,
    set_temp: float,
    units: list[str],
    nights: list[str],
    nseq: int,
    sigtyp: float,
    vartyp: float,
    sanity: bool,
    history: list[str] = (),
    overwrite: bool = False,
    dry_run: bool = False,
) -> fits.Header:
    """Write the RTN map FITS: R primary, NEXC and NFRAME compressed extensions."""
    finite = np.count_nonzero(np.isfinite(ratio))
    header = fits.Header()
    header["IMAGETYP"] = (STABLE_TYPE_RTNMAP.upper(), "Random-telegraph and excess read-noise map")
    header["CAM_ID"] = (str(camera_serial), "Camera serial")
    header["GAIN"] = (int(gain), "Gain mode")
    header["XBINNING"] = (int(n_binning), "Binning factor in x")
    header["YBINNING"] = (int(n_binning), "Binning factor in y")
    header["EGAIN"] = (float(egain), "[e-/ADU] Electronic gain of the frames")
    header["SET-TEMP"] = (float(set_temp), "[C] CCD set temperature")
    header["NIGHTBEG"] = (min(nights), "First nightdate of the frames")
    header["NIGHTEND"] = (max(nights), "Last nightdate of the frames")
    header["UNITS"] = (",".join(units), "Units the camera was on")
    header["NSEQ"] = (int(nseq), "Bias sequences used")
    header["NFRAMES"] = (len(ingredients), "Raw frames used")
    header["NNIGHTS"] = (len(set(nights)), "Nights of the frames")
    header["SIGTYP"] = (float(sigtyp), "[ADU] Typical read noise")
    header["VARTYP"] = (float(vartyp), "[ADU^2] Median per-pixel variance")
    header["EXCTHR"] = (_threshold(sigtyp, 9), "[ADU] Excursion threshold, 9 frames")
    for k in (2, 4, 8, 16):
        header[f"FRACR{k}"] = (np.count_nonzero(ratio > k) / finite, f"Fraction of pixels with R > {k}")
    header["FRACEXC"] = (
        int(nexc.sum(dtype=np.int64)) / int(nframe.sum(dtype=np.int64)),
        "Fraction of samples beyond EXCTHR",
    )
    header["SANITY"] = (bool(sanity), "Half the requested sequences passed self-check")
    set_pipe_ver_in_header(header)
    for line in history:
        header.add_history(line)
    if dry_run:
        return header

    write_IMCMB_to_header(header, ingredients)
    for key in header:
        if key.startswith("IMCMB"):
            header.comments[key] = "Raw frame"
    add_image_id(header)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fits.HDUList(
        [
            fits.PrimaryHDU(np.asarray(ratio, dtype=np.float32), header=header),
            fits.CompImageHDU(np.asarray(nexc, dtype=np.int16), name="NEXC"),
            fits.CompImageHDU(np.asarray(nframe, dtype=np.int16), name="NFRAME"),
        ]
    ).writeto(path, overwrite=overwrite)
    return header


def _accumulate(sequences: list[pd.DataFrame], sigtyp: float, blocks, shape, max_workers: int = 4, logger=None):
    """Pooled sums of squares, dof per row, excursion counts and frames per row over sequences."""
    logger = logger or logging.getLogger(__name__)
    ssw = np.zeros(shape, np.float64)
    nexc = np.zeros(shape, np.int16)
    dof = np.zeros(shape[0], np.int64)
    nframe = np.zeros(shape[0], np.int16)
    for frames in sequences:
        paths = list(frames.file_path)
        with ThreadPoolExecutor(max_workers) as ex:
            results = list(ex.map(lambda b: _block(_read_rows(paths, *b), sigtyp), blocks))
        for (y0, y1), (s, c, n, sticky) in zip(blocks, results):
            if sticky.any():
                logger.info(f"rtnmap: {_label(frames)} rows {y0}-{y1}: {int(sticky.sum())} sticky frames dropped")
            if n:
                ssw[y0:y1] += s
                nexc[y0:y1] += c
                dof[y0:y1] += n - 1
                nframe[y0:y1] += n
    return ssw, dof, nexc, nframe


def _block(stack: np.ndarray, sigtyp: float):
    """Sum of squares, excursion count and frames kept of one sequence block; sticky-bit frames dropped."""
    r, _, x = _residuals(stack)
    odd, even = r[:, :, 1::2], r[:, :, 0::2]
    sticky = ((odd > 118) & (odd < 138)).mean(axis=(1, 2)) - ((even > 118) & (even < 138)).mean(axis=(1, 2)) > 1e-4
    if sticky.any():
        if np.count_nonzero(~sticky) < 5:
            return None, None, 0, sticky
        r, _, x = _residuals(stack[~sticky])
    n = len(x)
    return x.var(axis=0, ddof=1) * (n - 1), (np.abs(r) > _threshold(sigtyp, n)).sum(axis=0), n, sticky


def _strip(frames: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Variance of the middle 64 rows and frame QA: sigma 3 %, level 1.5 ADU, excursions 20 %, CCD temperature 0.5 C."""
    stack = _read_rows(list(frames.file_path))
    n = len(stack)
    r, _, x = _residuals(stack)
    var = x.var(axis=0, ddof=1)
    sigma = np.array([_clipped_std(rj.ravel()) for rj in r]) / _loo_std(n)
    level = np.median(stack.reshape(n, -1), axis=1)
    excess = (np.abs(r) > _threshold(np.sqrt(np.median(var) / _chi2_median_factor(n)), n)).mean(axis=(1, 2))
    temp = frames.ccdtemp.to_numpy()
    good = (
        (np.abs(sigma / np.median(sigma) - 1) <= 0.03)
        & (np.abs(level - np.median(level)) <= 1.5)
        & (np.abs(excess / np.median(excess) - 1) <= 0.2)
        & (np.abs(temp - np.median(temp)) <= 0.5)
    )
    return var, good


def _read_rows(paths: list[str], y0: int | None = None, y1: int | None = None) -> np.ndarray:
    """Rows y0:y1 of raw frames as uint16; the middle 64 rows when None."""

    def read(path):
        with fits.open(path, uint=True, memmap=False) as hdul:
            hdu = next(h for h in hdul if h.size)
            mid = hdu.shape[0] // 2
            return hdu.section[mid - 32 : mid + 32] if y0 is None else hdu.section[y0:y1]

    with ThreadPoolExecutor(len(paths)) as ex:
        return np.stack(list(ex.map(read, paths)))


def _label(frames: pd.DataFrame) -> str:
    return f"{frames.unit.iloc[0]} {frames.nightdate.iloc[0]} {frames.obstime.iloc[0]:%H:%M:%S} UTC"


def _threshold(sigtyp: float, n: int) -> float:
    return float(5 * sigtyp * _loo_std(n))


def _row_offsets(d: np.ndarray) -> np.ndarray:
    """Clipped row mean of a deviation cube (n, ny, nx), clip 3 x frame MAD sigma."""
    med = np.median(d, axis=2, keepdims=True)
    s = 1.4826 * np.median(np.abs(d - med), axis=(1, 2), keepdims=True)
    w = np.abs(d - med) < 3 * np.maximum(s, 1.0)
    return (d * w).sum(axis=2, keepdims=True) / np.maximum(w.sum(axis=2, keepdims=True), 1)


def _residuals(stack: np.ndarray):
    """Leave-one-out median residuals, per-pixel median level and row-corrected frames."""
    x = stack.astype(np.float32)
    level = np.median(x, axis=0)
    x -= _row_offsets(x - level)
    r = x - _loo_median(x)
    r -= _row_offsets(r)
    return r, level, x


def _loo_median(x: np.ndarray) -> np.ndarray:
    """Per-pixel median of the other frames, for every frame."""
    n = x.shape[0]
    s = np.sort(x, axis=0)
    out = np.empty_like(x, dtype=np.float32)
    m = n // 2
    if n % 2:
        lo, mid, hi = s[m - 1], s[m], s[m + 1]
        for j in range(n):
            out[j] = np.where(x[j] < mid, 0.5 * (mid + hi), np.where(x[j] > mid, 0.5 * (lo + mid), 0.5 * (lo + hi)))
    else:
        lo, hi = s[m - 1], s[m]
        for j in range(n):
            out[j] = np.where(x[j] > lo, lo, hi)
    return out


@lru_cache
def _loo_std(n: int) -> float:
    """Std of x - leave-one-out median for a unit Gaussian, numerically."""
    x = np.random.default_rng(12345).standard_normal((n, 400000)).astype(np.float32)
    return float((x - _loo_median(x)).std())


def _chi2_median_factor(n: int) -> float:
    from scipy.stats import chi2

    return float(chi2.median(n - 1) / (n - 1))


def _clipped_std(a: np.ndarray, k: float = 3.0, iters: int = 5) -> float:
    s = float(a.std())
    for _ in range(iters):
        s = float(a[np.abs(a) < k * s].std())
    return s / 0.98658
