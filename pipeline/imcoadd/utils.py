import os, sys
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from ..const.environ import REF_DIR

from .header_set import InputHeaderSet


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
    input_images: list[str], match_swarp_size: bool
) -> tuple[int, int, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the coadd target grid and per-frame numpy offsets.
    Returns ``(target_w, target_h, target_cx, target_cy, x0, y0, shapes)``."""
    # First pass: CRPIX and shape of each frame.
    crpix = []
    shapes = []
    for f in input_images:
        with fits.open(f, memmap=True) as hdul:
            hdr = hdul[0].header
            crpix.append((hdr["CRPIX1"], hdr["CRPIX2"]))
            shapes.append(hdul[0].data.shape)  # (h, w)
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
    wcs_source: str, target_cx: float, target_cy: float, coadd_header: InputHeaderSet
) -> fits.Header:
    """Build the coadd output header from a reference frame's WCS + coadd_header.

    Clean WCS via astropy.wcs drops per-frame keys like FLXSCALE/SKYVAL/BACKTYPE
    that would otherwise leak from any single input; ``self.input_headers.coadd_header``
    carries the aggregated coadd metadata on top."""
    wcs = WCS(fits.getheader(wcs_source))
    wcs.wcs.crpix = [target_cx, target_cy]
    out_header = wcs.to_header(relax=True)
    for card in coadd_header.cards:
        out_header[card.keyword] = (card.value, card.comment)
    return out_header


def build_source_mask(
    catalog: str,
    shape: tuple[int, int],
    star_scale: float,
    galaxy_scale: float,
    class_star_cut: float,
    min_radius: float,
    logger=None,
) -> np.ndarray:
    """Elliptical source mask from a SExtractor catalog; extended sources get a wider ellipse.

    Each source is masked out to its Kron ellipse (``A_IMAGE``/``B_IMAGE`` scaled by
    ``KRON_RADIUS``) times a class-dependent factor, so the low-surface-brightness wings
    a segmentation map would miss are covered too."""
    mask = np.zeros(shape, dtype=bool)
    try:
        cat = Table.read(catalog, format="ascii.sextractor")
    except Exception as e:
        # transient SExtractor failures ("no key to print in table OBJECTS") leave a
        # 0-byte catalog; one bad frame must not kill a 1000-frame run
        if logger is not None:
            logger.warning(f"Unreadable detection catalog {os.path.basename(catalog)} ({e}); source mask empty")
        return mask
    if not len(cat):
        if logger is not None:
            logger.warning(f"No source detected in {os.path.basename(catalog)}; source mask is empty")
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


def _parse_swarp_image_size(config_path: str) -> tuple[int, int]:
    """Return (NX, NY) from a SWarp config's ``IMAGE_SIZE NX,NY`` line."""
    with open(config_path) as fp:
        for line in fp:
            tokens = line.split("#", 1)[0].split()
            if tokens and tokens[0] == "IMAGE_SIZE":
                nx, ny = tokens[1].split(",")
                return int(nx), int(ny)
    raise ValueError(f"IMAGE_SIZE not found in {config_path}")
