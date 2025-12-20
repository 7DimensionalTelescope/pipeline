from collections import Counter
import json
import os
import unicodedata
import re
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, SkyOffsetFrame

from ..path import NameHandler
from ..const import REF_DIR


def read_text_header(file: str, sep="\n"):
    with open(file, "r", encoding="utf-8") as f:
        content = f.read()

    # Clean non-ASCII characters
    cleaned_string = unicodedata.normalize("NFKD", content).encode("ascii", "ignore").decode("ascii")
    hdr = fits.Header.fromstring(cleaned_string, sep=sep)
    return hdr


def read_scamp_header(file: str, return_wcs=False):
    """
    Read a SCAMP output HEAD file, normalizing unicode and correcting WCS types.

    Args:
        file (str): Path to the header file

    Returns:
        fits.Header: Processed and cleaned FITS header with corrected WCS types

    Note:
        - Removes non-ASCII characters
        - Converts WCS projection type from TAN to TPV
    """

    hdr = read_text_header(file)
    # Correct CTYPE (TAN --> TPV)
    hdr["CTYPE1"] = ("RA---TPV", "WCS projection type for this axis")
    hdr["CTYPE2"] = ("DEC--TPV", "WCS projection type for this axis")

    if return_wcs:
        return WCS(hdr)

    return hdr


def extract_from_scamp_log(filename: str) -> None:
    """
    Open a text file, read it line by line,
    and print all lines containing 'removed'.
    """
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if "removed" in line:
                print(line.rstrip())


def build_wcs(ra_deg, dec_deg, crpix1, crpix2, pixscale_arcsec, pa_deg, flip=True):
    """
    Build an Astropy WCS with TAN projection and a CD matrix from pixel scale + PA.
    FITS convention assumed:
    - +X increases to the right; +Y increases up
    - For PA = 0 deg (north up, east left), CD becomes diag([-s, +s]) in deg/pix
    - PA measured from +Y (north) toward +X (east)
    - If flip is True, the image is flipped along the RA-axis. 7DT images are flipped.
    """
    s = pixscale_arcsec / 3600.0  # deg/pix
    pa = np.deg2rad(pa_deg)

    # CD matrix per FITS standard for a PA about the reference point:
    #   [CD1_1  CD1_2] = s * [ -cos(PA)   sin(PA) ]
    #   [CD2_1  CD2_2]       [  sin(PA)   cos(PA) ]
    cd11 = -s * np.cos(pa)
    cd12 = s * np.sin(pa)
    cd21 = s * np.sin(pa)
    cd22 = s * np.cos(pa)
    if flip:
        cd11, cd12 = -cd11, -cd12

    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cunit = ["deg", "deg"]
    w.wcs.crval = [ra_deg, dec_deg]  # sky at reference pixel
    w.wcs.crpix = [crpix1, crpix2]  # reference pixel (in SEXtractor pixel convention; origin handled later)
    w.wcs.cd = np.array([[cd11, cd12], [cd21, cd22]], dtype=float)
    return w


def read_TPV_wcs(image: str | fits.Header) -> WCS:
    """Read a FITS header with TPV WCS and return a WCS object. If malformed, inject null PV values."""
    if isinstance(image, str):
        hdr = fits.getheader(image)
    elif isinstance(image, fits.Header):
        hdr = image
    else:
        raise ValueError(f"Invalid input type: {type(image)}")

    formatted_pv = hdr.get("CTYPE1", "").endswith("TPV") or hdr.get("CTYPE2", "").endswith("TPV")
    has_pv = any(k.startswith("PV") for k in hdr.keys())

    # print(formatted_pv, has_pv)

    if formatted_pv and not has_pv:
        # inject null PV values
        for ax in (1, 2):
            hdr[f"PV{ax}_0"] = 0.0
            hdr[f"PV{ax}_1"] = 1.0
            hdr[f"PV{ax}_2"] = 0.0
    #     hdr['CTYPE1'] = 'RA---TAN'
    #     hdr['CTYPE2'] = 'DEC--TAN'
    wcs = WCS(hdr)
    return wcs


def strip_wcs(header: fits.Header, *, extra_keys=None, keep_keys=None) -> fits.Header:
    """
    Remove WCS-/SIP-related cards from a FITS Header by iterating backwards.

    Parameters
    ----------
    header : astropy.io.fits.Header
        The header to modify in place.
    extra_keys : Iterable[str], optional
        Additional exact keyword names to remove (e.g., {"CD3_3"}).
    keep_keys : Iterable[str], optional
        Exact keyword names to preserve even if they match patterns.

    Returns
    -------
    astropy.io.fits.Header
        The same header object, for convenience.

    Notes
    -----
    - Iterates indices from end to start to safely remove duplicates.
    - Supports alternate WCS versions via optional trailing letter (e.g., CTYPE1A).
    - Skips non-key cards like COMMENT/HISTORY/CONTINUE/blank.
    """

    _WCS_PATTERNS = [
        # Matrix and parameterized transforms
        r"CD\d_\d[A-Z]?", r"PC\d_\d[A-Z]?",
        r"PV\d+_\d+[A-Z]?", r"PS\d+_\d+[A-Z]?",
        # Core WCS axis definitions
        r"WCSAXES[A-Z]?", r"WCSNAME\d*[A-Z]?",
        r"CTYPE\d+[A-Z]?", r"CUNIT\d+[A-Z]?",
        r"CRPIX\d+[A-Z]?", r"CRVAL\d+[A-Z]?", r"CDELT\d+[A-Z]?",
        r"CROTA\d+[A-Z]?",
        # Reference system / epoch bits commonly tied to WCS
        r"RADESYS[A-Z]?", r"EQUINOX[A-Z]?",
        r"LONPOLE[A-Z]?", r"LATPOLE[A-Z]?",
        # Observing time sometimes bundled with WCS definitions
        # r"MJD-OBS[A-Z]?", r"DATE-OBS[A-Z]?",
        r"MJDREF[A-Z]?", # this shows up in our coarse wcs
        # Uncertainty keywords used with WCS in some headers
        r"CRDER\d*[A-Z]?", r"CSYER\d*[A-Z]?",
        # SIP distortion (TAN-SIP) keywords
        r"A_ORDER[A-Z]?", r"B_ORDER[A-Z]?",
        r"A_DMAX[A-Z]?", r"B_DMAX[A-Z]?",
        r"A_\d+_\d+[A-Z]?", r"B_\d+_\d+[A-Z]?",
        r"AP_\d+_\d+[A-Z]?", r"BP_\d+_\d+[A-Z]?",
    ]  # fmt: skip

    _WCS_REGEXES = [re.compile(p + r"$") for p in _WCS_PATTERNS]
    preserve = set(k.upper() for k in (keep_keys or ()))
    extra = set(k.upper() for k in (extra_keys or ()))

    def _is_wcs_key(k: str) -> bool:
        if not k or k in ("COMMENT", "HISTORY", "CONTINUE"):
            return False
        ku = k.upper()
        if ku in preserve:
            return False
        if ku in extra:
            return True
        return any(rx.match(ku) for rx in _WCS_REGEXES)

    # Walk backwards and pop by index to handle duplicates reliably
    for idx in range(len(header) - 1, -1, -1):
        card = header.cards[idx]
        # Some cards can have blank keywords; guard for that
        key = getattr(card, "keyword", None)
        if key and _is_wcs_key(key):
            header.pop(idx)

    return header


def get_fov_center(wcs: WCS, naxis1: int, naxis2: int):
    center_pix = (naxis1 - 1) / 2, (naxis2 - 1) / 2  # 0-indexed
    center_world = wcs.pixel_to_world(*center_pix)
    return center_world.ra.deg, center_world.dec.deg


def get_fov_quad(wcs, x, y, edge=True):
    if is_flipped(wcs):
        vertices = np.array([[x, y], [x, 1], [0, 1], [0, y]], dtype="float")
        if edge:
            vertices[0] += 0.5
            vertices[1, 0] += 0.5
            vertices[1, 1] -= 0.5
            vertices[2] -= 0.5
            vertices[3, 0] -= 0.5
            vertices[3, 1] += 0.5
    else:
        vertices = np.array([[0, y], [0, 0], [x, 0], [x, y]], dtype="float")
        if edge:
            vertices[0, 0] -= 0.5
            vertices[0, 1] += 0.5
            vertices[1] -= 0.5
            vertices[2, 0] += 0.5
            vertices[2, 1] -= 0.5
            vertices[3] += 0.5
    ra, dec = wcs.all_pix2world(vertices[:, 0], vertices[:, 1], 1)
    return ra, dec


def polygon_info_header(image_info: "ImageInfo") -> fits.Header:
    fov_ra = image_info.get("fov_ra", None)
    fov_dec = image_info.get("fov_dec", None)
    return make_polygon_header(image_info.wcs, image_info.naxis1, image_info.naxis2, fov_ra=fov_ra, fov_dec=fov_dec)


def make_polygon_header(
    wcs: WCS, naxis1: int, naxis2: int, fov_ra: np.array = None, fov_dec: np.array = None
) -> fits.Header:
    """Returns header with polygon info & field rotation"""

    ra, dec = get_fov_center(wcs, naxis1, naxis2)
    # FOV Center
    cards = [
        ("RACENT", round(ra, 3), "RA CENTER [deg]"),
        ("DECCENT", round(dec, 3), "DEC CENTER [deg]"),
    ]

    # FOV Polygon
    if fov_ra is None or fov_dec is None:
        fov_ra, fov_dec = get_fov_quad(wcs, naxis1, naxis2)

    for i in range(len(fov_ra)):
        cards.append((f"RAPOLY{i}", round(fov_ra[i], 3), f"RA POLYGON {i} [deg]"))
        cards.append((f"DEPOLY{i}", round(fov_dec[i], 3), f"DEC POLYGON {i} [deg]"))

    # Field Rotation
    rotation_angle_1 = field_rotation(wcs)
    cards.append(("ROTANG", rotation_angle_1, "Counterclockwise from North [deg]"))
    # updates.append(("ROTANG2", rotation_angle_2, "Rotation angle from East [deg]"))

    # return fits.Header(cards)
    return cards


def field_rotation(wcs: WCS) -> float:
    """
    From the WCS standard, the field rotation is measured counterclockwise from the North.
    Formally, arctan of CD2_1 / CD2_2.

    Note:
    - cd1_1: CD1_1 value from the FITS header. delta RA / delta X_IMAGE
    - cd1_2: CD1_2 value from the FITS header. delta Dec / delta X_IMAGE
    - cd2_1: CD2_1 value from the FITS header. delta RA / delta Y_IMAGE
    - cd2_2: CD2_2 value from the FITS header. delta Dec / delta Y_IMAGE


    Returns:
        - counterclockwise rotation angle from the North in degrees
    """
    ((cd1_1, cd1_2), (cd2_1, cd2_2)) = wcs.wcs.cd
    return np.degrees(np.arctan2(cd2_1, cd2_2))


def is_flipped(wcs: WCS) -> bool:
    """
    Check if the image is flipped.
    7DT raw single images are mirrored along the RA-axis (RA increases with x),
    but the stacked images are not.
    """

    ((cd1_1, cd1_2), (cd2_1, cd2_2)) = wcs.wcs.cd
    determinant = cd1_1 * cd2_2 - cd1_2 * cd2_1
    return np.sign(determinant) > 0


def _order_ccw(xy):
    """Return points ordered CCW around their centroid."""
    c = xy.mean(axis=0)
    ang = np.arctan2(xy[:, 1] - c[1], xy[:, 0] - c[0])
    return xy[np.argsort(ang)]


# quick but fails at poles
def inside_quad_radec(points_ra, points_dec, quad_ra, quad_dec, eps=1e-12):
    """
    Vectorized inclusion test of points in a convex quad on the sky.
    points_*: arrays of candidate points (deg)
    quad_*:   length-4 arrays/lists of quad vertices (deg), any order
    eps:      numerical tolerance in projected plane units (deg)
    Returns:  boolean array of length len(points_ra)
    """
    quad_ra = np.asarray(quad_ra, dtype=float)
    quad_dec = np.asarray(quad_dec, dtype=float)

    # 1) Choose a projection center near the quad, handling RA wrap
    ra0 = np.mean(quad_ra)
    dra = ((quad_ra - ra0 + 180) % 360) - 180
    ra0 = ra0 + np.mean(dra)
    dec0 = np.mean(quad_dec)
    center = SkyCoord(ra0 * u.deg, dec0 * u.deg, frame="icrs")
    off = SkyOffsetFrame(origin=center)

    # 2) Project vertices and points to the tangent plane
    verts = SkyCoord(quad_ra * u.deg, quad_dec * u.deg, frame="icrs").transform_to(off)
    V = np.column_stack([verts.lon.to_value(u.deg), verts.lat.to_value(u.deg)])
    V = _order_ccw(V)  # enforce CCW

    pts = SkyCoord(points_ra * u.deg, points_dec * u.deg, frame="icrs").transform_to(off)
    P = np.column_stack([pts.lon.to_value(u.deg), pts.lat.to_value(u.deg)])

    # 3) Half-space (cross-product) test
    # For CCW vertices, (E_i x (P - V_i)) >= 0 for all i means inside.
    E = np.roll(V, -1, axis=0) - V  # edges (4,2)
    # 2D "cross" z-component for all points vs each edge
    # cross((dx,dy), (ux,uy)) = dx*uy - dy*ux
    U = P[:, None, :] - V[None, :, :]  # (N,4,2)
    cross = E[None, :, 0] * U[:, :, 1] - E[None, :, 1] * U[:, :, 0]  # (N,4)

    # 4) Accept points on or inside (tolerance eps)
    return np.all(cross >= -eps, axis=1)


# precise but slower
def inside_quad_spherical(points_ra, points_dec, quad_ra, quad_dec):
    """
    points_*: array-like (deg) — can be scalars or numpy arrays
    quad_*:   4 vertices (deg), ordered around the boundary (CW or CCW)
    returns:  boolean array of shape broadcast(points_ra, points_dec)
    """
    from spherical_geometry.polygon import SphericalPolygon

    # build polygon in degrees
    poly = SphericalPolygon.from_radec(quad_ra, quad_dec, degrees=True)

    # ensure arrays and broadcast to a common shape
    ra = np.atleast_1d(points_ra)
    dec = np.atleast_1d(points_dec)
    ra_b, dec_b = np.broadcast_arrays(ra, dec)

    # evaluate point-by-point (contains_lonlat expects scalars)
    out = np.empty(ra_b.shape, dtype=bool)
    it = np.nditer(
        [ra_b, dec_b, out], flags=["multi_index", "refs_ok"], op_flags=[["readonly"], ["readonly"], ["writeonly"]]
    )
    for lon, lat, o in it:
        o[...] = bool(poly.contains_lonlat(float(lon), float(lat), degrees=True))

    # return a scalar bool if inputs were scalars
    return out if out.shape else bool(out)


def get_3x3_stars(
    matched_catalog: Table, H: int, W: int, cutout_size=30, return_id: bool = True, id_col: str = "id"
) -> list[int]:
    # H, W = data.shape  # numpy: (y, x)

    # remove ref-only rows
    matched_catalog = matched_catalog[~matched_catalog["separation"].mask]
    if len(matched_catalog) == 0:
        return []

    # centers in 0-based pixel coords (SExtractor is 1-based)
    x_img = matched_catalog["X_IMAGE"].astype(float) - 1.0
    y_img = matched_catalog["Y_IMAGE"].astype(float) - 1.0

    # 3x3 target points (corners, edge centers, center), clamped so cutouts fit
    margin = int(np.ceil(cutout_size / 2)) + 1
    xs = np.array([0, W / 2, W - 1], dtype=float)
    ys = np.array([0, H / 2, H - 1], dtype=float)
    targets = np.array([(x, y) for y in ys for x in xs], dtype=float)
    targets[:, 0] = np.clip(targets[:, 0], margin, W - 1 - margin)
    targets[:, 1] = np.clip(targets[:, 1], margin, H - 1 - margin)

    # Precompute candidate matrix
    cand_xy = np.c_[x_img, y_img]

    # For each target, pick nearest candidate; if already taken, use next nearest
    selected_idx = _select_3x3_by_nearest(cand_xy, targets)
    if not return_id:
        return selected_idx

    # unique ids of sources in matched catalog
    matched_ids = [matched_catalog[i][id_col] if i is not None else None for i in selected_idx]
    return matched_ids  # None if no candidate available


def find_id_rows(matched_catalog: Table, matched_ids: list[int], id_col: str = "id") -> Table:
    """This can't handle None in matched_ids"""
    order = np.array(matched_ids)  # Make an indexer array that preserves order
    mask = np.in1d(order, matched_catalog[id_col])
    order = order[mask]
    id_to_idx = {id_: i for i, id_ in enumerate(matched_catalog[id_col])}
    return matched_catalog[[id_to_idx[_id] for _id in order]]


def resolve_rows_by_id(matched_catalog: Table, matched_ids: list[int], id_col: str = "id") -> list[Table | None]:
    """This can handle None in matched_ids"""
    # map id -> row once
    id_to_row = {int(row[id_col]): row for row in matched_catalog}  # cast if needed
    resolved = []
    for _id in matched_ids:
        if _id is None:
            resolved.append(None)
        else:
            resolved.append(id_to_row.get(int(_id)))
    return resolved


def _select_3x3_by_nearest(cand_xy, targets):
    """
    For each target, pick the nearest candidate (by Euclidean distance).
    Deduplicate: if the same source would be used twice, assign to the target
    where it is closer; for the other target, pick next-nearest.

    Returns a list of length 9 with catalog indices (into the original table)
    or None if no candidate available (should be rare given the logic).
    """
    try:
        from scipy.spatial import cKDTree as KDTree

        _HAVE_KDTREE = True
    except Exception:
        _HAVE_KDTREE = False

    selected = [None] * len(targets)
    taken = set()

    if _HAVE_KDTREE and len(cand_xy) > 0:
        tree = KDTree(cand_xy)
        # Query more than 1 in case of duplicates; cap at len(cand_xy)
        k_step = 5
        max_k = min(200, len(cand_xy))
        for ti, t in enumerate(targets):
            k = k_step
            while selected[ti] is None and k <= max_k:
                dists, nn = tree.query(t, k=k)
                if np.isscalar(nn):
                    nn = np.array([nn])
                    dists = np.array([dists])
                # Sort by distance explicitly (tree.query already returns ordered, but be safe)
                order = np.argsort(dists)
                for oi in order:
                    ci = int(nn[oi])
                    gi = int(ci)
                    if gi not in taken:
                        selected[ti] = gi
                        taken.add(gi)
                        break
                k += k_step

    else:
        # NumPy fallback: brute-force distances
        for ti, t in enumerate(targets):
            if len(cand_xy) == 0:
                break
            d = np.hypot(cand_xy[:, 0] - t[0], cand_xy[:, 1] - t[1])
            order = np.argsort(d)
            for ci in order:
                gi = int(ci)
                if gi not in taken:
                    selected[ti] = gi
                    taken.add(gi)
                    break

    return selected


def get_num_sources(catalog: str | Table, depth: float = 17.0, zp: float = 0, mag_key: str = "phot_g_mean_mag") -> int:
    """
    Get the number of sources in a catalog.
    Use any depth shallower than the image's expected depth.
    """
    if isinstance(catalog, str):
        catalog = Table.read(catalog, hdu=2)
    catalog = catalog[catalog[mag_key] + zp < depth]
    return len(catalog)


def get_source_num_frac(sci_cat: str, local_astref: str, ref_depth: float = 17.0):
    """hard-coded"""
    SCI_TO_REF_AREA_RATIO = 0.5  # Science image area / Reference image area

    sci_filter = NameHandler(sci_cat).filter
    sci_depth = json.load(open(os.path.join(REF_DIR, "depths.json")))[sci_filter]
    sci_zp = json.load(open(os.path.join(REF_DIR, "zeropoints.json")))[sci_filter]

    sci_num_sources = get_num_sources(sci_cat, zp=sci_zp, depth=sci_depth, mag_key="MAG_AUTO")
    ref_num_sources = get_num_sources(local_astref, depth=ref_depth, mag_key="phot_g_mean_mag")

    return sci_num_sources / (SCI_TO_REF_AREA_RATIO * ref_num_sources)
