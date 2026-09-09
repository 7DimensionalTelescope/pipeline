import os, sys
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from ..const.environ import REF_DIR

from .header_set import InputHeaderSet
from ..io.ldac import read_catalog


def extract_date_and_time(date_obs_str, round_seconds=False):
    """
    Extract date and time from the 'DATE-OBS' FITS header keyword value.

    Parameters:
    date_obs_str (str): The DATE-OBS string, usually in the format 'YYYY-MM-DDTHH:MM:SS.sss'
    round_seconds (bool): Whether to round the seconds to the nearest whole number

    Returns:
    str, str: Extracted date and time strings in 'YYYYMMDD' and 'HHMMSS' formats
    """
    from astropy.time import Time

    # Convert the DATE-OBS string to an Astropy Time object
    time_obj = Time(date_obs_str)

    # Extract the date and time components
    date_str = time_obj.strftime("%Y%m%d")
    if round_seconds:
        time_str = time_obj.strftime("%H%M%S")
    else:
        time_str = f"{time_obj.datetime.hour:02}{time_obj.datetime.minute:02}{int(time_obj.datetime.second):02}"

    return date_str, time_str


def calc_mean_dateloc(dateloclist):
    from datetime import datetime

    datetime_objects = [datetime.fromisoformat(t) for t in dateloclist]
    posix_times = [dt.timestamp() for dt in datetime_objects]
    mean_posix_time = np.mean(posix_times)
    mean_datetime = datetime.fromtimestamp(mean_posix_time)
    mean_isot_time = mean_datetime.isoformat()
    return mean_isot_time


def unpack(packed, type, ex=None):
    if len(packed) != 1:
        print(f"There are more than one ({len(packed)}) {type}s")
        unpacked = input(f"Type {type.upper()} name (e.g. {packed if ex is None else ex}):")
    else:
        unpacked = packed[0]
    return unpacked
    # return float(unpacked)


def move_file(src, dst):
    """For lazy import"""
    import shutil

    shutil.move(src, dst)


def determine_size(
    input_images: list[str], match_swarp_size: bool, frame_cache: dict | None = None
) -> tuple[int, int, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the coadd target grid and per-frame numpy offsets.
    Returns ``(target_w, target_h, target_cx, target_cy, x0, y0, shapes)``."""
    # First pass: CRPIX and shape of each frame.
    crpix = []
    shapes = []
    for f in input_images:
        cached = frame_cache.get(f) if frame_cache is not None else None
        if cached is not None:
            data, hdr = cached
            crpix.append((hdr["CRPIX1"], hdr["CRPIX2"]))
            shapes.append(data.shape)
        else:
            with fits.open(f, memmap=True) as hdul:
                hdr = hdul[0].header
                crpix.append((hdr["CRPIX1"], hdr["CRPIX2"]))
                shapes.append(hdul[0].data.shape)
    crpix = np.array(crpix, dtype=float)
    shapes = np.array(shapes, dtype=int)

    # Target grid: SWarp's IMAGE_SIZE centered at N/2+0.5 (its CRPIX convention,
    # verified to match the SWarp coadd bit-for-bit when the resamp WCS is
    # reused with CRPIX overridden), or a tight bbox spanning the inputs.
    if match_swarp_size:
        target_w, target_h = _parse_swarp_image_size(os.path.join(REF_DIR, "7dt.swarp"))
        target_cx = target_w / 2 + 0.5
        target_cy = target_h / 2 + 0.5
        x0 = np.rint(target_cx - crpix[:, 0]).astype(int)
        y0 = np.rint(target_cy - crpix[:, 1]).astype(int)
    else:
        target_cx, target_cy = float(crpix[:, 0].max()), float(crpix[:, 1].max())
        x0 = np.rint(target_cx - crpix[:, 0]).astype(int)  # column offset of each frame in target
        y0 = np.rint(target_cy - crpix[:, 1]).astype(int)
        target_w = int((x0 + shapes[:, 1]).max())
        target_h = int((y0 + shapes[:, 0]).max())
    # self.logger.debug(f"Target shape ({target_h}, {target_w}) with CRPIX ({target_cx}, {target_cy})")
    return target_w, target_h, target_cx, target_cy, x0, y0, shapes


def build_coadd_wcs_header(
    wcs_source: str,
    target_cx: float,
    target_cy: float,
    coadd_header: InputHeaderSet,
    frame_cache: dict | None = None,
) -> fits.Header:
    """Build the coadd output header from a reference frame's WCS + coadd_header.

    Clean WCS via astropy.wcs drops per-frame keys like FLXSCALE/SKYVAL/BACKTYPE
    that would otherwise leak from any single input; ``self.input_headers.coadd_header``
    carries the aggregated coadd metadata on top."""
    cached = frame_cache.get(wcs_source) if frame_cache is not None else None
    wcs = WCS(cached[1] if cached is not None else fits.getheader(wcs_source))
    wcs.wcs.crpix = [target_cx, target_cy]
    out_header = wcs.to_header(relax=True)
    for card in coadd_header.cards:
        out_header[card.keyword] = (card.value, card.comment)
    return out_header


def write_mask_plio(path, mask, header=None) -> str:
    """Binary mask as PLIO_1-compressed FITS (~1-3 MB for a 61 Mpx frame).

    PLIO is FITS-standard tile compression designed for pixel masks; any compliant
    reader sees an ordinary integer image. Plain uint8 working copies are still
    written wherever SExtractor must read the mask (it cannot read tile-compressed)."""
    hdu = fits.CompImageHDU(
        data=np.asarray(mask, dtype=np.uint8),
        header=header,
        compression_type="PLIO_1",
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fits.HDUList([fits.PrimaryHDU(header=header), hdu]).writeto(path, overwrite=True)
    return path


def count_dtype(n_inputs: int):
    """Smallest exact unsigned integer type that holds a count of *n_inputs* frames."""
    for dtype in (np.uint8, np.uint16, np.uint32):
        if n_inputs <= np.iinfo(dtype).max:
            return dtype
    raise ValueError(f"{n_inputs} inputs exceed the largest supported count plane (uint32)")


def write_count_planes(path, planes: dict[str, np.ndarray], header=None, legend=None) -> str:
    """Integer count planes as one losslessly tile-compressed multi-extension FITS, one plane per EXTNAME."""
    from .counts import LEGEND

    legend = LEGEND if legend is None else legend
    shapes = {name: tuple(np.shape(plane)) for name, plane in planes.items()}
    if len(set(shapes.values())) > 1:
        raise ValueError(f"count planes must share one shape: {shapes}")
    primary = fits.PrimaryHDU(header=header)
    primary.header["NCOUNTPL"] = (len(planes), "Number of count planes in this file")
    for i, name in enumerate(planes, 1):
        primary.header[f"COUNTPL{i}"] = (name, legend.get(name, "count plane"))
    # the planes carry the WCS only; the full coadd provenance stays on the primary
    plane_header = WCS(header).to_header(relax=True) if header is not None else None
    hdus = [primary]
    for name, plane in planes.items():
        hdu = fits.CompImageHDU(
            data=np.ascontiguousarray(plane),
            header=plane_header.copy() if plane_header is not None else None,
            compression_type="RICE_1",
            name=name,
        )
        hdu.header["BUNIT"] = ("count", legend.get(name, "count plane"))
        hdus.append(hdu)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


def read_count_planes(path) -> dict[str, np.ndarray]:
    """Count planes of a coverage product, keyed by EXTNAME."""
    with fits.open(path, memmap=False) as hdul:
        return {hdu.name: hdu.data for hdu in hdul[1:]}


def read_mask_plio(path):
    """Persisted mask -> bool array, or None when absent/unreadable."""
    if not os.path.exists(path):
        return None
    try:
        return fits.getdata(path, ext=1).astype(bool)
    except OSError:
        return None


def build_source_mask(
    catalog: str,
    shape: tuple[int, int],
    star_scale: float,
    galaxy_scale: float,
    class_star_cut: float,
    min_radius: float,
    bright_flux_adu: float = 3.0e5,
    bright_radius_per_dex: float = 90.0,
    logger=None,
) -> np.ndarray:
    """Elliptical source mask from a SExtractor catalog; extended sources get a wider ellipse.

    Each source is masked out to its Kron ellipse (``A_IMAGE``/``B_IMAGE`` scaled by
    ``KRON_RADIUS``) times a class-dependent factor, so the low-surface-brightness wings
    a segmentation map would miss are covered too."""
    mask = np.zeros(shape, dtype=bool)
    try:
        cat = catalog if isinstance(catalog, Table) else Table.read(catalog, format="ascii.sextractor")
    except Exception as e:
        # transient SExtractor failures ("no key to print in table OBJECTS") leave a
        # 0-byte catalog; one bad frame must not kill a 1000-frame run
        if logger is not None:
            logger.warning(f"Unreadable detection catalog {os.path.basename(catalog)} ({e}); source mask empty")
        return mask
    if not len(cat):
        if logger is not None:
            name = "the supplied table" if isinstance(catalog, Table) else os.path.basename(catalog)
            logger.warning(f"No source detected in {name}; source mask is empty")
        return mask

    # KRON_RADIUS is 0 when SExtractor's Kron measurement failed; 1 keeps the isophotal ellipse
    kron = np.clip(np.asarray(cat["KRON_RADIUS"], dtype=float), 1.0, None)
    is_extended = np.asarray(cat["CLASS_STAR"], dtype=float) < class_star_cut
    scale = np.where(is_extended, galaxy_scale, star_scale)
    a = np.maximum(np.asarray(cat["A_IMAGE"], dtype=float) * kron * scale, min_radius)
    b = np.maximum(np.asarray(cat["B_IMAGE"], dtype=float) * kron * scale, min_radius)
    theta = np.radians(np.asarray(cat["THETA_IMAGE"], dtype=float))
    # SExtractor image coordinates are 1-indexed
    xc = np.asarray(cat["X_IMAGE"], dtype=float) - 1.0
    yc = np.asarray(cat["Y_IMAGE"], dtype=float) - 1.0

    # bright tier by INSTRUMENTAL flux (ADU): saturated/near-saturated stars break
    # KRON/CLASS_STAR, and their wings/spikes escape any Kron ellipse. Circular on
    # purpose: single-exposure PAs differ, so spikes cannot line up on the reprojected
    # plane. r grows per decade of flux; constants are first-guess pending wing-profile
    # calibration on a deep coadd.
    if "FLUX_AUTO" in cat.colnames:
        flux = np.asarray(cat["FLUX_AUTO"], dtype=float)
        bright = np.isfinite(flux) & (flux > bright_flux_adu)
        if bright.any():
            r_bright = min_radius + bright_radius_per_dex * np.log10(
                np.maximum(flux, bright_flux_adu) / bright_flux_adu
            )
            a = np.where(bright, np.maximum(a, r_bright), a)
            b = np.where(bright, np.maximum(b, r_bright), b)
            if logger is not None:
                logger.debug(f"{int(bright.sum())} bright stars got circular wing masks "
                             f"(max r {float(r_bright.max()):.0f} px)")
    elif logger is not None:
        logger.debug("catalog has no FLUX_AUTO (pre-2026-08-11 bkgdet cat); bright-star tier skipped")

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # half-sizes of each ellipse's axis-aligned bounding box
    dx = np.sqrt((a * cos_t) ** 2 + (b * sin_t) ** 2)
    dy = np.sqrt((a * sin_t) ** 2 + (b * cos_t) ** 2)

    h, w = shape
    for i in range(len(cat)):
        x0 = max(0, int(np.floor(xc[i] - dx[i]))); x1 = min(w, int(np.ceil(xc[i] + dx[i])) + 1)  # fmt: skip
        y0 = max(0, int(np.floor(yc[i] - dy[i]))); y1 = min(h, int(np.ceil(yc[i] + dy[i])) + 1)  # fmt: skip
        if x1 <= x0 or y1 <= y0:
            continue
        yy = np.arange(y0, y1, dtype=float)[:, None] - yc[i]
        xx = np.arange(x0, x1, dtype=float)[None, :] - xc[i]
        u = xx * cos_t[i] + yy * sin_t[i]
        v = -xx * sin_t[i] + yy * cos_t[i]
        mask[y0:y1, x0:x1] |= (u / a[i]) ** 2 + (v / b[i]) ** 2 <= 1.0

    if logger is not None:
        logger.debug(
            f"Source mask from {len(cat)} sources ({int(is_extended.sum())} extended, "
            f"x{galaxy_scale} vs x{star_scale}): {100 * mask.mean():.1f}% of pixels masked"
        )
    return mask


# B_IMAGE is derivable; the rest must be in the catalog
_ELLIPSE_KEYS = ["ALPHA_J2000", "DELTA_J2000", "A_IMAGE", "THETA_IMAGE", "KRON_RADIUS", "CLASS_STAR", "FLUX_AUTO"]


def source_ellipses_on_frame(catalog, source_header, target_header, logger=None):
    """Catalog ellipses on another frame's pixels; None if the catalog lacks a column."""
    from astropy.wcs import WCS

    try:
        cat = catalog if isinstance(catalog, Table) else read_catalog(catalog)
    except Exception as e:
        if logger is not None:
            logger.debug(f"Cannot read {os.path.basename(str(catalog))} ({e})")
        return None
    if not len(cat):
        return None

    missing = [c for c in _ELLIPSE_KEYS if c not in cat.colnames]
    # main.param comments B_IMAGE out; ELLIPTICITY = 1 - B/A and ELONGATION = A/B recover it
    axis_ratio = None
    if "B_IMAGE" in cat.colnames:
        axis_ratio = np.asarray(cat["B_IMAGE"], float) / np.asarray(cat["A_IMAGE"], float)
    elif "ELLIPTICITY" in cat.colnames:
        axis_ratio = 1.0 - np.asarray(cat["ELLIPTICITY"], float)
    elif "ELONGATION" in cat.colnames:
        axis_ratio = 1.0 / np.asarray(cat["ELONGATION"], float)
    else:
        missing.append("B_IMAGE/ELLIPTICITY/ELONGATION")
    if missing:
        if logger is not None:
            logger.info(f"Photometry catalog lacks {missing}; running a detection pass instead")
        return None

    wcs_s, wcs_t = WCS(source_header), WCS(target_header)
    ra = np.asarray(cat["ALPHA_J2000"], float)
    dec = np.asarray(cat["DELTA_J2000"], float)
    x_s, y_s = wcs_s.all_world2pix(ra, dec, 0)
    x_t, y_t = wcs_t.all_world2pix(ra, dec, 0)

    def step(dx, dy):
        """Where a unit step in the source frame lands in the target frame."""
        r1, d1 = wcs_s.all_pix2world(x_s + dx, y_s + dy, 0)
        x1, y1 = wcs_t.all_world2pix(r1, d1, 0)
        return x1 - x_t, y1 - y_t

    theta = np.radians(np.asarray(cat["THETA_IMAGE"], float))
    ux, uy = step(np.cos(theta), np.sin(theta))  # along the major axis
    vx, vy = step(-np.sin(theta), np.cos(theta))  # along the minor axis
    a_src = np.asarray(cat["A_IMAGE"], float)

    return Table({
        "X_IMAGE": x_t + 1.0,  # build_source_mask subtracts 1: SExtractor is 1-indexed
        "Y_IMAGE": y_t + 1.0,
        "A_IMAGE": a_src * np.hypot(ux, uy),
        "B_IMAGE": a_src * axis_ratio * np.hypot(vx, vy),
        "THETA_IMAGE": np.degrees(np.arctan2(uy, ux)),
        "KRON_RADIUS": np.asarray(cat["KRON_RADIUS"], float),
        "CLASS_STAR": np.asarray(cat["CLASS_STAR"], float),
        "FLUX_AUTO": np.asarray(cat["FLUX_AUTO"], float),
    })  # fmt: skip


def parse_sex_config(config_path: str, keys) -> dict:
    """Named settings from a SExtractor config, as strings."""
    values = {}
    with open(config_path) as fp:
        for line in fp:
            tokens = line.split("#", 1)[0].split()
            if tokens and tokens[0] in keys:
                values[tokens[0]] = " ".join(tokens[1:])
    missing = set(keys) - set(values)
    if missing:
        raise ValueError(f"{sorted(missing)} not found in {config_path}")
    return values


def estimate_background(data, mask=None, back_size: int = 64, filter_size: int = 3):
    """Mesh background + RMS via sep; ``mask`` marks pixels to EXCLUDE."""
    import sep

    # sep rejects the big-endian arrays FITS hands back; this also gives it C-contiguity.
    # ascontiguousarray, not astype: a caller that already converted pays no second copy
    arr = np.ascontiguousarray(data, dtype=np.float32)
    bkg = sep.Background(arr, mask=mask, bw=back_size, bh=back_size,
                         fw=filter_size, fh=filter_size, fthresh=0.0)  # fmt: skip
    return bkg.back(), bkg.rms()


def _parse_swarp_image_size(config_path: str) -> tuple[int, int]:
    """Return (NX, NY) from a SWarp config's ``IMAGE_SIZE NX,NY`` line."""
    with open(config_path) as fp:
        for line in fp:
            tokens = line.split("#", 1)[0].split()
            if tokens and tokens[0] == "IMAGE_SIZE":
                nx, ny = tokens[1].split(",")
                return int(nx), int(ny)
    raise ValueError(f"IMAGE_SIZE not found in {config_path}")


def fill_masked_tiles(data, holes, targets, tile=512, max_margin=4096, logger=None):
    """Fill *targets* tile by tile with maskfill's own kernel, never over the whole grid.

    `maskfill.maskfill` is unusable at coadd scale here for two reasons: every pass reallocates and
    reconvolves the WHOLE array it is handed, and its loop fills every NaN it can see rather than the mask it
    was given, so it also pays for the uncovered exterior, which is thousands of pixels deep. Measured on a
    10200x6800 coadd, 1818 hole pixels cost 68 s. This drives `process_masked_pixels` -- maskfill's own
    3x3 nanmedian step, and its nanmean smoothing pass -- over a padded tile, filling only what it was asked
    to. A pixel at depth d is decided on pass d by the good pixels within d of it, so padding past the deepest
    target reproduces the whole-grid values exactly wherever no exterior lies inside that radius.

    Where one does -- a hole at the very rim of the coverage -- the two differ ON PURPOSE: the exterior stays
    NaN and `nanmedian` ignores it, so the hole is filled from covered data alone, while the whole-grid call
    propagated invented exterior values inward. Returns (tiles, pixels written)."""
    from maskfill.maskfill import process_masked_pixels
    from scipy.ndimage import binary_dilation, distance_transform_edt

    neighbourhood = np.ones((3, 3), dtype=bool)
    height, width = data.shape
    rows, cols = np.flatnonzero(targets.any(axis=1)), np.flatnonzero(targets.any(axis=0))
    if not rows.size:
        return 0, 0
    tiles = written = stalled = 0
    for y0 in range(rows[0] // tile * tile, rows[-1] + 1, tile):
        for x0 in range(cols[0] // tile * tile, cols[-1] + 1, tile):
            y1, x1 = min(y0 + tile, height), min(x0 + tile, width)
            core = targets[y0:y1, x0:x1]
            if not core.any():
                continue
            margin = 8
            while True:
                by0, bx0 = max(y0 - margin, 0), max(x0 - margin, 0)
                by1, bx1 = min(y1 + margin, height), min(x1 + margin, width)
                mask = targets[by0:by1, bx0:bx1]
                depth = distance_transform_edt(mask)
                need = int(np.ceil(depth[y0 - by0 : y1 - by0, x0 - bx0 : x1 - bx0][core].max())) + 2
                if need <= margin or margin >= max_margin:
                    break
                margin = min(max(need, 2 * margin), max_margin)
            block = data[by0:by1, bx0:bx1].copy()
            block[holes[by0:by1, bx0:bx1]] = np.nan  # every hole is NaN; only `mask` will be written
            remaining = mask.copy()
            while remaining.any():
                frontier = remaining & binary_dilation(~np.isnan(block), neighbourhood)
                if not frontier.any():
                    # nothing covered is reachable; the caller promised a NaN-free image, so settle for a level
                    stalled += int(remaining.sum())
                    finite = block[~np.isnan(block)]
                    block[remaining] = np.median(finite) if finite.size else 0.0
                    break
                process_masked_pixels(block, 1, mask=frontier, operator_func=np.nanmedian)
                remaining &= ~frontier
            process_masked_pixels(block, 1, mask=mask, operator_func=np.nanmean)  # maskfill's smoothing pass
            patch = data[y0:y1, x0:x1]
            patch[core] = filled_core = block[y0 - by0 : y1 - by0, x0 - bx0 : x1 - bx0][core]
            tiles += 1
            written += int(filled_core.size)
    if stalled and logger is not None:
        logger.warning(
            f"NaN fill: {stalled} hole pixels had no covered pixel reachable inside their tile and took the "
            f"tile median instead"
        )
    return tiles, written
