import os
import gc
from time import time
from tqdm import tqdm
from glob import glob
from pathlib import Path
from functools import lru_cache


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
from ..const import GAIA_ROOT_DIR

# import dhutil as dh

# only for astro zoom
# import ligo.skymap.plot


@lru_cache(maxsize=1)
def _get_gaia_healpix_files():
    return sorted(glob(f"{GAIA_ROOT_DIR}/tile_*.csv"))


# GAIA_HEALPIX_FILES = sorted(glob(f"{GAIA_ROOT_DIR}/tile_*.csv"))  # avoid multiple glob calls


def get_refcat_gaia(image: str):
    header = fits.getheader(image)
    ra, dec = header["RA"], header["DEC"]
    return


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

    # load healpix ids
    # GAIA_ROOT_DIR = "/lyman/data1/factory/catalog/gaia_source_dr3/healpix_nside64"
    # files = sorted(glob(f"{GAIA_ROOT_DIR}/tile_*.csv"))
    files = _get_gaia_healpix_files()  #GAIA_HEALPIX_FILES
    # ipix_list = [s.split('_')[-1].replace('.csv', '') for s in files]
    ipix_list = [int(Path(s).stem.split("_")[1]) for s in files]
    radec = np.array([hp.pix2ang(64, ipix, nest=True, lonlat=True) for ipix in ipix_list])
    heal_ra = radec[:, 0]
    heal_dec = radec[:, 1]

    # Get center of 7DT tile and set matching radius

    # find healpix tiles
    reference_coord = SkyCoord(ra=ra_center * u.deg, dec=dec_center * u.deg, frame="icrs")
    heal_coords = SkyCoord(ra=heal_ra * u.deg, dec=heal_dec * u.deg, frame="icrs")
    distances = reference_coord.separation(heal_coords)
    matched_files = np.array(files)[distances < matching_r * u.deg]

    return matched_files


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
