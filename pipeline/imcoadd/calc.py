"""Pixel-level arithmetic on SWarp-resampled FITS images.

All functions here assume the input frames share the same pixel scale and
WCS projection (same CRVAL / CD matrix), differing only in CRPIX.  Alignment
is therefore a pure integer pixel shift derived from the CRPIX difference —
no reprojection is performed.
"""

import os
import time
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from ..utils import time_diff_in_seconds, add_suffix
from ..services.logger import Logger
from .utils import determine_size, build_coadd_wcs_header


def mean_coadd_numpy(
    input_images: list[str],
    output_path: str,
    coadd_header: fits.Header,
    weights: list[str] | None = None,
    weight_output: str | None = None,
    flxscales: list[float] | bool | None = None,
    match_swarp_size: bool = True,
    logger: Logger | None = None,
) -> str:
    """Per-pixel flux-scaled mean coadd (simple or inverse-variance weighted).

    Works on SWarp-resampled images that are centered differently.
    """
    st = time.time()
    backend = "weighted" if weights is not None else "simple"
    grid = "swarp grid" if match_swarp_size else "tight bbox"
    if logger is not None:
        logger.info(f"Start in-memory numpy coaddition ({backend} mean, {grid})")
    if weights is not None and len(weights) != len(input_images):
        raise ValueError(f"weights ({len(weights)}) and input_images ({len(input_images)}) length mismatch")
    if isinstance(flxscales, list) and len(flxscales) != len(input_images):
        raise ValueError(f"flxscales ({len(flxscales)}) and input_images ({len(input_images)}) length mismatch")

    # Flux-scaling source (logged once): False disables; an explicit list is the
    # snapshot source of truth; None falls back to each file's FLXSCALE header.
    scale_mode = "disabled" if flxscales is False else ("explicit list" if flxscales is not None else "FLXSCALE header")
    if logger is not None:
        logger.info(f"Flux scaling: {scale_mode}")

    target_w, target_h, target_cx, target_cy, x0, y0, shapes = determine_size(input_images, match_swarp_size)

    sum_arr = np.zeros((target_h, target_w), dtype=np.float64)
    norm_arr = np.zeros((target_h, target_w), dtype=np.float64 if weights is not None else np.int32)
    for i, f in enumerate(input_images):
        with fits.open(f) as hdul:
            a = hdul[0].data.astype(np.float64)
            # False disables; explicit list = snapshot source of truth; None = file FLXSCALE.
            if flxscales is False:
                flxscale = 1.0
            elif flxscales is not None:
                flxscale = flxscales[i]
            else:
                flxscale = hdul[0].header.get("FLXSCALE", 1.0)
        flxscale = 1.0 if flxscale is None else flxscale
        h, w = a.shape

        tx0 = max(0, x0[i]); tx1 = min(target_w, x0[i] + w)  # fmt: skip
        ty0 = max(0, y0[i]); ty1 = min(target_h, y0[i] + h)  # fmt: skip
        if tx1 <= tx0 or ty1 <= ty0:
            if logger is not None:
                logger.debug(f"{os.path.basename(f)}: no overlap with target grid; skipped")
            continue
        sx0 = tx0 - x0[i]; sx1 = tx1 - x0[i]  # fmt: skip
        sy0 = ty0 - y0[i]; sy1 = ty1 - y0[i]  # fmt: skip
        src = a[sy0:sy1, sx0:sx1]
        valid = np.isfinite(src) & (src != 0.0)

        if weights is None:
            sum_arr[ty0:ty1, tx0:tx1] += np.where(valid, src * flxscale, 0.0)
            norm_arr[ty0:ty1, tx0:tx1] += valid
        else:
            # SWarp's MAP_WEIGHT = 1/variance of the raw resampled data
            # (RESCALE_WEIGHTS = N). After we multiply data by FLXSCALE its
            # variance scales by FLXSCALE^2, so use w/FLXSCALE^2 as the
            # inverse-variance weight of the flux-normalised image.
            with fits.open(weights[i]) as hdul:
                w_full = hdul[0].data.astype(np.float64)
            w_eff = w_full[sy0:sy1, sx0:sx1] / (flxscale * flxscale)
            valid &= w_eff > 0
            sum_arr[ty0:ty1, tx0:tx1] += np.where(valid, w_eff * src * flxscale, 0.0)
            norm_arr[ty0:ty1, tx0:tx1] += np.where(valid, w_eff, 0.0)

    coadd = np.where(norm_arr > 0, sum_arr / np.where(norm_arr > 0, norm_arr, 1), np.nan).astype(np.float32)

    out_header = build_coadd_wcs_header(input_images[0], target_cx, target_cy, coadd_header)
    fits.writeto(output_path, coadd, header=out_header, overwrite=True)

    if weights is not None:
        weight_out = weight_output or add_suffix(output_path, "weight")
        fits.writeto(weight_out, norm_arr.astype(np.float32), header=out_header, overwrite=True)
        if logger is not None:
            logger.debug(f"Wrote coadd weight map: {weight_out}")

    if logger is not None:
        logger.info(f"Numpy coaddition completed in {time_diff_in_seconds(st)} seconds")
    return output_path


def median_coadd_numpy(
    input_images: list[str],
    output_path: str,
    coadd_header: fits.Header,
    flxscales: list[float] | bool | None = None,
    match_swarp_size: bool = True,
    chunk_h: int = 128,
    logger: Logger | None = None,
) -> str:
    """Per-pixel flux-scaled median coadd.

    Works on SWarp-resampled images that are centered differently.

    Peak memory bounded by chunk_h.
    """
    st = time.time()
    size = "swarp FOV size" if match_swarp_size else "tight bbox"
    if logger is not None:
        logger.info(f"Start in-memory numpy coaddition (median, {size})")
    if isinstance(flxscales, list) and len(flxscales) != len(input_images):
        raise ValueError(f"flxscales ({len(flxscales)}) and input_images ({len(input_images)}) length mismatch")

    target_w, target_h, target_cx, target_cy, x0, y0, shapes = determine_size(input_images, match_swarp_size)

    handles = [fits.open(f, memmap=True) for f in input_images]
    # Flux-scaling source (logged once): False disables; explicit list = snapshot
    # source of truth; None falls back to each file's FLXSCALE header.
    if flxscales is False:
        scale_mode = "disabled"
        flxscales = np.ones(len(input_images), dtype=np.float32)
    else:
        if flxscales is None:
            scale_mode = "FLXSCALE header"
            flxscales = [h[0].header.get("FLXSCALE", 1.0) for h in handles]
        else:
            scale_mode = "explicit list"
        flxscales = np.array([1.0 if f is None else f for f in flxscales], dtype=np.float32)
    if logger is not None:
        logger.info(f"Flux scaling: {scale_mode}")

    coadd = np.full((target_h, target_w), np.nan, dtype=np.float32)
    try:
        for ys in range(0, target_h, chunk_h):
            ye = min(target_h, ys + chunk_h)
            # (N, strip_h, target_w) stack; NaN-init so np.nanmedian ignores unfilled cells.
            stack = np.full((len(input_images), ye - ys, target_w), np.nan, dtype=np.float32)
            for i, hdul in enumerate(handles):
                h, w = shapes[i]
                tx0 = max(0, x0[i]); tx1 = min(target_w, x0[i] + w)  # fmt: skip
                ty0 = max(ys, max(0, y0[i])); ty1 = min(ye, min(target_h, y0[i] + h))  # fmt: skip
                if tx1 <= tx0 or ty1 <= ty0:
                    continue
                sx0 = tx0 - x0[i]; sx1 = tx1 - x0[i]  # fmt: skip
                sy0 = ty0 - y0[i]; sy1 = ty1 - y0[i]  # fmt: skip
                src = hdul[0].data[sy0:sy1, sx0:sx1].astype(np.float32) * flxscales[i]
                src[(src == 0.0) | ~np.isfinite(src)] = np.nan
                stack[i, ty0 - ys : ty1 - ys, tx0:tx1] = src
            coadd[ys:ye, :] = np.nanmedian(stack, axis=0)
    finally:
        for hdul in handles:
            hdul.close()

    out_header = build_coadd_wcs_header(input_images[0], target_cx, target_cy, coadd_header)
    fits.writeto(output_path, coadd, header=out_header, overwrite=True)
    if logger is not None:
        logger.info(f"Numpy median coaddition completed in {time_diff_in_seconds(st)} seconds")
    return output_path


def subtract_images(
    image_a: str,
    image_b: str,
    output_path: str,
    flxscale: bool = False,
    zp_key: str = "ZP_AUTO",
    logger: Logger | None = None,
    overwrite=True,
) -> str:
    """Subtract *image_b* from *image_a*, aligned by their CRPIX offsets.

    image_a is the reference for flux scaling and registration.
    """

    def chatter(msg: str, level: str = "debug"):
        if logger is not None:
            return getattr(logger, level)(msg)
        else:
            print(f"[subtract_images:{level.upper()}] {msg}")

    chatter("Start image subtraction")

    with fits.open(image_a) as ha:
        data_a = ha[0].data.astype(np.float32)
        hdr_a = ha[0].header.copy()
    with fits.open(image_b) as hb:
        data_b = hb[0].data.astype(np.float32)
        hdr_b = hb[0].header.copy()

    # Match image_b to image_a's level (image_a is the reference, fa=1).
    # Precedence: disabled -> ZP (zp_key) -> FLXSCALE -> none.
    if flxscale is False:
        flxscale_factor, scale_mode = 1.0, "disabled"
    elif hdr_a.get(zp_key) is not None and hdr_b.get(zp_key) is not None:
        flxscale_factor = 10 ** (0.4 * (float(hdr_a[zp_key]) - float(hdr_b[zp_key])))
        scale_mode = f"ZP ({zp_key})"
    elif (flxscale_a := hdr_a.get("FLXSCALE")) is not None:
        flxscale_factor = float(hdr_b.get("FLXSCALE", 1.0)) / float(flxscale_a)
        scale_mode = "FLXSCALE"
    else:
        flxscale_factor, scale_mode = 1.0, "none"
        chatter(f"Cannot determine flux scaling: neither {zp_key} nor FLXSCALE", level="warning")
    chatter(f"Flux scaling ({scale_mode}): image_b x {flxscale_factor:.4f} -> image_a level")

    crpix_a = np.array([hdr_a["CRPIX1"], hdr_a["CRPIX2"]], dtype=float)
    crpix_b = np.array([hdr_b["CRPIX1"], hdr_b["CRPIX2"]], dtype=float)

    h_a, w_a = data_a.shape
    h_b, w_b = data_b.shape

    # Integer pixel offset: image_b's pixel (0,0) [0-indexed] sits at column
    # dx, row dy in image_a's coordinate system.
    dx = int(np.rint(crpix_a[0] - crpix_b[0]))
    dy = int(np.rint(crpix_a[1] - crpix_b[1]))

    # image_a is the reference grid; no a-pixel is cropped. image_b is shifted
    # onto image_a's grid over their overlap; non-overlap stays NaN.
    diff = np.full((h_a, w_a), np.nan, dtype=np.float32)

    ax0 = max(0, dx);      ax1 = min(w_a, dx + w_b)  # fmt: skip
    ay0 = max(0, dy);      ay1 = min(h_a, dy + h_b)  # fmt: skip
    if ax1 > ax0 and ay1 > ay0:
        bx0 = ax0 - dx;  bx1 = ax1 - dx  # fmt: skip
        by0 = ay0 - dy;  by1 = ay1 - dy  # fmt: skip
        patch_a = data_a[ay0:ay1, ax0:ax1]
        patch_b = data_b[by0:by1, bx0:bx1]
        # NaN propagates through the subtraction; only SWarp 0.0 zero-padding
        # needs explicit masking.
        out = patch_a - flxscale_factor * patch_b
        out[(patch_a == 0.0) | (patch_b == 0.0)] = np.nan
        diff[ay0:ay1, ax0:ax1] = out

    out_header = WCS(hdr_a).to_header(relax=True)
    fits.writeto(output_path, diff, header=out_header, overwrite=overwrite)
    return output_path
