import os
import gc
import fcntl
import time as time_module
from time import time
from tqdm import tqdm
from glob import glob
from pathlib import Path
from functools import lru_cache
from contextlib import contextmanager


import numpy as np
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.coordinates import Angle
import healpy as hp
from spherical_geometry.polygon import SphericalPolygon

from ..io.cfitsldac import write_ldac
from .utils import build_wcs
from ..const import GAIA_ROOT_DIR, ASTRM_CUSTOM_REF_DIR

# import dhutil as dh

# only for astro zoom
# import ligo.skymap.plot


# Removed _get_gaia_healpix_files() - no longer needed, using in-memory healpix queries


@contextmanager
def _file_lock(lock_path, timeout=300, poll_interval=0.1):
    """
    Context manager for file-based locking with timeout.

    Parameters:
    -----------
    lock_path : str
        Path to the lock file
    timeout : float
        Maximum time to wait for lock acquisition (seconds)
    poll_interval : float
        Time between lock acquisition attempts (seconds)
    """
    lock_file = None
    lock_acquired = False
    start_time = time_module.time()

    try:
        # Create lock file directory if needed
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)

        # Try to acquire lock with timeout
        while time_module.time() - start_time < timeout:
            try:
                lock_file = open(lock_path, "w")
                # Non-blocking lock attempt
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Write PID to lock file for debugging
                lock_file.write(f"{os.getpid()}\n")
                lock_file.flush()
                os.fsync(lock_file.fileno())
                lock_acquired = True
                break
            except BlockingIOError:
                # Lock is held by another process, wait and retry
                if lock_file:
                    lock_file.close()
                    lock_file = None
                time_module.sleep(poll_interval)
            except FileNotFoundError:
                # Directory might not exist yet, retry
                if lock_file:
                    lock_file.close()
                    lock_file = None
                time_module.sleep(poll_interval)

        if not lock_acquired:
            raise TimeoutError(f"Could not acquire lock {lock_path} within {timeout} seconds")

        yield

    finally:
        # Release lock
        if lock_file and lock_acquired:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
            except Exception:
                pass
            lock_file.close()
            # Remove lock file (best effort)
            try:
                os.remove(lock_path)
            except Exception:
                pass


def _wait_for_file(file_path, timeout=300, poll_interval=0.5):
    """
    Wait for a file to appear, with timeout.

    Parameters:
    -----------
    file_path : str
        Path to the file to wait for
    timeout : float
        Maximum time to wait (seconds)
    poll_interval : float
        Time between checks (seconds)
    """
    start_time = time_module.time()
    while time_module.time() - start_time < timeout:
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            return True
        time_module.sleep(poll_interval)
    return False


def get_refcat_gaia(
    output_path: str,
    ra: float,
    dec: float,
    naxis1: int,
    naxis2: int,
    pixscale: float,
):
    """
    Generate a GAIA reference catalog by:
    1. Finding relevant GAIA healpix catalogs based on center coordinates
    2. Loading and aggregating the catalogs
    3. Selecting sources within an elliptical region
    4. Saving the result as a FITS LDAC file

    This function is thread-safe and process-safe. If multiple processes try to generate
    the same catalog simultaneously, only one will do the work while others wait.

    Parameters:
    -----------
    output_path : str
        Full path where the reference catalog should be saved
    ra : float
        Right ascension of center point in degrees
    dec : float
        Declination of center point in degrees
    naxis1 : int
        Image width in pixels
    naxis2 : int
        Image height in pixels
    pixscale : float
        Pixel scale in arcseconds per pixel

    Returns:
    --------
    str
        Path to the generated reference catalog file
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    lock_path = f"{output_path}.lock"

    # Fast path: if file already exists, return immediately (before any expensive computation)
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path

    # Try to acquire lock to generate the catalog
    try:
        with _file_lock(lock_path, timeout=300):
            # Double-check: another process might have created it while we waited for lock
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return output_path

            # We have the lock, now do the expensive computation
            # Calculate center pixel coordinates
            xcent = (naxis1 + 1) / 2.0
            ycent = (naxis2 + 1) / 2.0

            # Find relevant healpix tiles (matching_r in degrees)
            matching_r = 2.0
            fpaths = find_healpix_tiles(ra, dec, matching_r)

            # Load all healpix catalogs
            dfs = []
            for f in fpaths:
                df = read_healpix_gaia(f)
                dfs.append(df)
            if not dfs:
                raise ValueError(f"No GAIA healpix catalogs found for RA={ra}, DEC={dec}")
            df = pd.concat(dfs, ignore_index=True)
            del dfs
            gc.collect()

            # Create SkyCoord center
            center = SkyCoord(ra=ra, dec=dec, unit="deg")

            # Select sources in ellipse (same parameters as ref_cat_generator2.py)
            semi_major = 0.98  # degrees
            semi_minor = 0.82  # degrees
            position_angle = 0  # degrees
            df_sel = select_sources_in_ellipse(df, center, semi_major, semi_minor, position_angle)
            del df
            gc.collect()

            # Convert errors from mas to deg
            df_sel.loc[:, "ra_error"] = df_sel["ra_error"] / (3600 * 10**3)
            df_sel.loc[:, "dec_error"] = df_sel["dec_error"] / (3600 * 10**3)

            # Clean the dataframe
            df_sel = clean_refcat_df(df_sel)

            # Convert to astropy Table
            outbl = Table.from_pandas(df_sel)
            del df_sel
            gc.collect()

            # Build WCS header
            wcs = build_wcs(ra, dec, xcent, ycent, pixscale, pa_deg=0, flip=True)
            wcs_header = wcs.to_header()

            # Create a minimal FITS header
            header_out = fits.Header()
            header_out["SIMPLE"] = (True, "conforms to FITS standard")
            header_out["BITPIX"] = (-32, "array data type")
            header_out["NAXIS"] = (2, "number of array dimensions")
            header_out["NAXIS1"] = naxis1
            header_out["NAXIS2"] = naxis2
            header_out.update(wcs_header)

            # Generate the catalog (we have the lock, so safe to write)
            write_ldac(header_out, outbl, output_path)
            return output_path

    except TimeoutError:
        # Lock acquisition timed out - another process is generating it
        # Wait for the file to appear
        if _wait_for_file(output_path, timeout=300):
            return output_path
        else:
            raise RuntimeError(
                f"Timeout waiting for reference catalog to be generated: {output_path}. "
                "Another process may be generating it, or the process may have crashed."
            )


def is_point_in_tile(point, tile_vertices):
    """
    Check if a point (SkyCoord) is inside a tile defined by its vertices.

    :param point: SkyCoord, the point to check.
    :param tile_vertices: List of SkyCoord, vertices of the tile in order.
    :return: Boolean, True if the point is inside the tile.
    """
    # Convert vertices to SphericalPolygon
    polygon = SphericalPolygon.from_lonlat([v.ra.deg for v in tile_vertices], [v.dec.deg for v in tile_vertices])

    # Check if the point is inside the polygon
    return polygon.contains_lonlat(point.ra.deg, point.dec.deg)


def find_tile_for_point(point, tiles):
    """
    Find which tile a point belongs to.

    :param point: SkyCoord, the point to check.
    :param tiles: List of dictionaries, where each dictionary contains:
                  - 'tile_id': int, identifier for the tile
                  - 'vertices': List of SkyCoord, vertices of the tile
    :return: The tile_id of the tile containing the point, or None if not found.
    """

    for tile in tiles:
        tile_id = tile["tile_id"]
        tile_vertices = tile["vertices"]

        if is_point_in_tile(point, tile_vertices):
            return tile_id
    return None


def is_points_in_ellipse_skycoord(points, ra_center, dec_center, a, b, theta):
    # Convert center and points to SkyCoord
    center = SkyCoord(ra=ra_center * u.deg, dec=dec_center * u.deg, frame="icrs")
    points = SkyCoord(ra=points[:, 0] * u.deg, dec=points[:, 1] * u.deg, frame="icrs")

    # Compute offsets in a rotated frame
    separation = points.separation(center).radian
    position_angle = points.position_angle(center).radian - np.radians(theta)

    # Convert to ellipse frame
    x = separation * np.cos(position_angle)
    y = separation * np.sin(position_angle)

    # Check ellipse condition
    return (x / a) ** 2 + (y / b) ** 2 <= 1


def read_healpix_gaia(fpath, extract=True):
    # load full column names
    colname_path = "/lyman/data1/factory/catalog/gaia_source_dr3/column_names.txt"
    with open(colname_path, "r") as f:
        colnames = [line.strip() for line in f]

    columns_to_extract = [
        "ra",
        "dec",
        "ra_error",
        "dec_error",
        "pmra",
        "pmdec",
        "pmra_error",
        "pmdec_error",
        "phot_g_mean_flux",
        "phot_g_mean_flux_error",
        "phot_g_mean_mag",  # CAUTION: its error should be derived later
        "ref_epoch",
    ]
    # df = pd.read_csv(fpath, header=None, dtype={151: 'str'}, names=colnames)  #, low_memory=False)
    # if not extract:
    #     return df
    # df = df[columns_to_extract]  # slow

    columns_to_extract_indices = [colnames.index(col) for col in columns_to_extract]

    # Load only required columns
    df = pd.read_csv(fpath, header=None, dtype={151: "str"}, names=colnames, usecols=columns_to_extract_indices)

    # cal mag err
    df["phot_g_mean_mag_error"] = 2.5 / np.log(10) * df["phot_g_mean_flux_error"] / df["phot_g_mean_flux"]
    # drop flux cols
    df = df.drop(columns=["phot_g_mean_flux_error", "phot_g_mean_flux"])
    # reorder columns
    cols = [col for col in df.columns if col != "ref_epoch"] + ["ref_epoch"]
    df = df[cols]
    return df


def find_healpix_tiles(ra_center, dec_center, matching_r):
    """
    Find healpix tile files that overlap with the search region using in-memory calculations.
    No I/O operations - constructs file paths directly from healpix pixel indices.

    Parameters:
    -----------
    ra_center : float
        Right ascension of center point in degrees
    dec_center : float
        Declination of center point in degrees
    matching_r : float
        Search radius in degrees

    Returns:
    --------
    list of str
        List of file paths to healpix tile CSV files
    """
    nside = 64  # healpix nside parameter

    # Convert RA/DEC to healpix pixel index
    # healpy uses (theta, phi) in radians, where theta is colatitude (0 at north pole)
    # and phi is longitude (0 at RA=0)
    theta = np.pi / 2 - np.deg2rad(dec_center)  # colatitude
    phi = np.deg2rad(ra_center)  # longitude (RA)

    # Convert matching radius from degrees to radians
    radius_rad = np.deg2rad(matching_r)

    # Find all healpix pixels within the disc using query_disc
    # query_disc returns pixel indices within the disc
    vec = hp.ang2vec(theta, phi)  # unit vector pointing to center
    ipix_list = hp.query_disc(nside, vec, radius_rad, nest=True)

    # Construct file paths directly from pixel indices (no globbing)
    file_paths = [os.path.join(GAIA_ROOT_DIR, f"tile_{ipix}.csv") for ipix in ipix_list]

    return file_paths


# def get_abpa_from_mvee(center, radii, rotation):
#     """unused"""

#     semi_major_idx = np.argmax(radii)  # Index of the semi-major axis
#     semi_minor_idx = 1 - semi_major_idx  # Index of the semi-minor axis

#     semi_major = radii[semi_major_idx]
#     semi_minor = radii[semi_minor_idx]

#     # Ensure the position angle corresponds to the semi-major axis
#     if semi_major_idx == 0:
#         position_angle = np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))
#     else:
#         position_angle = np.degrees(np.arctan2(rotation[1, 1], rotation[0, 1]))
#     return center, semi_major, semi_minor, position_angle


def select_sources_in_ellipse(sources_df, center, a, b, pa_deg):
    src = SkyCoord(ra=sources_df["ra"].values, dec=sources_df["dec"].values, unit="deg")

    sep = center.separation(src)  # angle to center
    pa = center.position_angle(src)  # E of N
    # dth = (pa - Angle(pa_deg, unit="deg")).radian  # angle from major axis, in rad
    # dth = pa  # angle from major axis, in rad
    dth = (pa - Angle(90 + pa_deg, unit="deg")).radian  # angle from major axis, in rad

    # correct polar radius of the ellipse
    rlim = (a * b) / np.sqrt((b * np.cos(dth)) ** 2 + (a * np.sin(dth)) ** 2)

    inside = sep.to(u.deg).value <= rlim
    return sources_df[inside]


# this works in the tangent plane
# def select_sources_in_ellipse(sources_df, center, a, b, pa_deg):
#     src = SkyCoord(ra=sources_df["ra"].values, dec=sources_df["dec"].values, unit="deg")
#     off = src.transform_to(center.skyoffset_frame())

#     # small-angle offsets in degrees (East, North)
#     x = off.lon.to(u.deg).value
#     y = off.lat.to(u.deg).value

#     th = np.deg2rad(pa_deg)
#     xp =  x*np.cos(th) + y*np.sin(th)   # rotate so xp aligns with major axis
#     yp = -x*np.sin(th) + y*np.cos(th)

#     inside = (xp/a)**2 + (yp/b)**2 <= 1.0
#     return sources_df[inside]


def inject_pmdec_error(mag_err):
    return 230.0 * mag_err + 0.1


def clean_refcat_df(df: pd.DataFrame) -> pd.DataFrame:
    pmcols = ("pmra", "pmdec", "pmra_error", "pmdec_error")

    # 1) Drop rows with any invalid values in non-PM numeric columns
    mask = np.ones(len(df), dtype=bool)
    for c in df.columns:
        if c.startswith("pm"):
            continue
        col = df[c]
        if pd.api.types.is_numeric_dtype(col):
            vals = col.to_numpy(copy=False)
            mask &= np.isfinite(vals)
    if not mask.all():
        df = df.loc[mask].copy()

    # 2) Ensure PM cols exist and are float (so NaNs are allowed)
    for c in pmcols:
        if c in df.columns:
            df[c] = df[c].astype("float64", copy=False)
        else:
            df[c] = np.nan

    # pmerr_injected flag
    col_flag = "arbitrary_pmerr"
    if col_flag not in df.columns:
        df[col_flag] = False

    # 3) Inject missing PM values/errors (vectorized, in-place)
    pmra = df["pmra"].to_numpy(copy=False)
    pmdec = df["pmdec"].to_numpy(copy=False)
    pmra_err = df["pmra_error"].to_numpy(copy=False)
    pmdec_err = df["pmdec_error"].to_numpy(copy=False)
    pinj = df[col_flag].to_numpy(copy=False)

    bad_pmra = ~np.isfinite(pmra)
    if bad_pmra.any():
        pmra[bad_pmra] = 0.0
        pinj[bad_pmra] = True

    bad_pmdec = ~np.isfinite(pmdec)
    if bad_pmdec.any():
        pmdec[bad_pmdec] = 0.0
        pinj[bad_pmdec] = True

    # pmdec_error from phot_g_mean_mag_error
    mag_err = df["phot_g_mean_mag_error"].to_numpy(copy=False)
    pmdec_err_candidate = 230.0 * mag_err + 0.1
    bad_pmdec_err = ~np.isfinite(pmdec_err)
    if bad_pmdec_err.any():
        pmdec_err[bad_pmdec_err] = pmdec_err_candidate[bad_pmdec_err]
        pinj[bad_pmdec_err] = True

    # pmra_error = pmdec_error * cos(dec)
    dec_rad = np.deg2rad(df["dec"].to_numpy(copy=False))
    pmra_err_candidate = pmdec_err * np.cos(dec_rad)
    bad_pmra_err = ~np.isfinite(pmra_err)
    if bad_pmra_err.any():
        pmra_err[bad_pmra_err] = pmra_err_candidate[bad_pmra_err]
        pinj[bad_pmra_err] = True

    return df


def build_7dt_wcs_header(center):
    header = fits.Header()
    header["SIMPLE"] = (True, "conforms to FITS standard")
    header["BITPIX"] = (-32, "array data type")
    header["NAXIS"] = (2, "number of array dimensions")
    header["NAXIS1"] = 9576
    header["NAXIS2"] = 6388
    coarse_wcs = build_wcs(center["ra"], center["dec"], 4788.5, 3194.5, 0.505, 0, flip=True)
    header.update(coarse_wcs.to_header())
    # print(header)
    return header


def run_single(tile_idx, matching_r=2, show=False, save=False):
    """Things hard-coded"""
    start = time()

    fpaths = find_healpix_tiles(tile_idx, matching_r)  # relevant hp tile filenames
    # print(len(fpaths), fpaths)

    # load all as a single df
    dfs = []
    for f in fpaths:
        df = read_healpix_gaia(f)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()  # Force garbage collection

    center = dh.get_tiles(tile_idx, center=True)
    header = build_7dt_wcs_header(center)

    center = SkyCoord(ra=center["ra"], dec=center["dec"], unit="deg")
    semi_major = 0.98
    semi_minor = 0.82
    position_angle = 0
    df_sel = select_sources_in_ellipse(df, center, semi_major, semi_minor, position_angle)

    # ra dec err mas to deg
    df_sel.loc[:, "ra_error"] = df_sel["ra_error"] / (3600 * 10**3)
    df_sel.loc[:, "dec_error"] = df_sel["dec_error"] / (3600 * 10**3)

    # clean NaN
    df_sel = clean_refcat_df(df_sel)

    if show:
        plt.figure(dpi=300)
        # ax = plt.axes(projection="astro zoom", center="9h -90d", radius="5 deg")
        ax = plt.axes(projection="astro zoom", center=center, radius="2 deg")

        ra, dec = (df_sel["ra"], df_sel["dec"])
        # ax.scatter(ra, dec, c="C0", s=0.001, alpha=0.5)

        ax.scatter(ra, dec, s=0.3, alpha=0.5, transform=ax.get_transform("world"))
        ax.grid()
        ax.coords[0].set_format_unit(u.deg)  # ra axis hour to deg

        # draw ellipse
        th = np.linspace(0, 2 * np.pi, 720)
        pa = np.deg2rad(position_angle)  # 0°=North, +E
        x = semi_major * np.cos(th)
        y = semi_minor * np.sin(th)
        xp = x * np.cos(pa) + y * np.sin(pa)
        yp = -x * np.sin(pa) + y * np.cos(pa)
        ell = SkyCoord(lon=xp * u.deg, lat=yp * u.deg, frame=center.skyoffset_frame()).icrs
        ax.plot(ell.ra.deg, ell.dec.deg, transform=ax.get_transform("world"), ls="--", lw=1, color="k")

        # dh.set_xylim(ax, 129, 141, -4.5, 5)
        dh.overlay_tiles(fontsize=6, color="k", fontweight="bold")
        plt.xlabel("RA (deg)")
        plt.ylabel("Dec (deg)")
        # plt.savefig('test.png')
        plt.show()

    outbl = Table.from_pandas(df_sel)  # to astropy Table
    del df_sel
    gc.collect()  # Force garbage collection
    # tablename = f"/lyman/data1/factory/catalog/gaia_dr3_7DT/T{tile_idx:05}.fits"
    if save:
        tablename = f"/lyman/data2/factory/ref_scamp/gaia_dr3_7DT/T{tile_idx:05}.fits"
        write_ldac(header, outbl, tablename)
        print(f"saved to {tablename}")
    if show:
        print(f"took {time() - start:.2f} s")


# %%

if __name__ == "__main__":
    import multiprocessing
    from tqdm.contrib.concurrent import process_map
    import multiprocessing as mp
    import signal
    from functools import partial
    from tqdm import tqdm

    def _init_worker():
        # Let the parent handle Ctrl+C; workers will be terminated/closed by parent.
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    start = time()

    nthread = 16  # I/O bound. 64
    tiles = list(dh.get_tiles(center=True)["id"])
    globbed = glob(f"/lyman/data2/factory/ref_scamp/gaia_dr3_7DT/T*.fits")
    existing = [int(Path(s).stem[1:]) for s in globbed]
    tiles = [item for item in tiles if item not in existing]

    with multiprocessing.Pool(processes=nthread) as pool:
        # results = pool.map(run_single, tiles)
        # results = list(tqdm(pool.imap(run_single, tiles), total=len(tiles)))
        # for _ in tqdm(pool.imap(run_single, tiles), total=len(tiles)):
        #     pass
        # for _ in tqdm(pool.imap_unordered(run_single, tiles), total=len(tiles)):
        #     pass

        ctx = mp.get_context("fork")  # on macOS/Windows, use "spawn"
        pool = ctx.Pool(processes=nthread, initializer=_init_worker)

        func = partial(run_single, save=True)  # pass save=True to every task
        chunksize = 4  # optional but helps with scheduling overhead

        try:
            for _ in tqdm(pool.imap_unordered(func, tiles, chunksize=chunksize), total=len(tiles)):
                pass
        except KeyboardInterrupt:
            print("\n^C received, terminating workers…", flush=True)
            pool.terminate()
        else:
            pool.close()
        finally:
            pool.join()
            print("Done.")

    print((time() - start) / 3600, "h elapsed")


# ulimit -v 314572800
# nice -n 10 python ref_cat_generator2.py
# pkill -u snu -f python
