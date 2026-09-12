"""Sparse output-grid positions of detector bad pixels, shared by the quality masks and the combine.

One structure per input frame replaces the read-modify-write of the SWarp resampled weight map: the
nearest output pixel of every detector bad pixel is carried as a row-sorted linear index and applied
to whatever validity array a coadd backend already holds. Coordinates are in the resampled frame's
own pixel grid, which is what every backend slices with x0/y0.
"""

import os
from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from ..const import REF_DIR
from ..errors.definition import CoaddError
from .const import LANCZOS3_HALFWIDTH


def swarp_resampling_type(config=os.path.join(REF_DIR, "7dt.swarp")) -> str:
    """RESAMPLING_TYPE of the SWarp configuration the reprojection runs with."""
    with open(config) as fp:
        for line in fp:
            fields = line.split("#", 1)[0].split()
            if fields and fields[0] == "RESAMPLING_TYPE":
                return fields[1].upper() if len(fields) > 1 else ""
    return ""


def nearest_output_pixels(ra, dec, output_header, shape) -> tuple[np.ndarray, np.ndarray]:
    """Nearest output pixel (y, x) of each sky position, dropped where it falls off the grid."""
    ra = np.asarray(ra, dtype=np.float64)
    if not ra.size:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    x, y = WCS(output_header).all_world2pix(ra, np.asarray(dec, dtype=np.float64), 0)
    finite = np.isfinite(x) & np.isfinite(y)
    xi = np.zeros(x.shape, dtype=np.int64)
    yi = np.zeros(y.shape, dtype=np.int64)
    xi[finite] = np.rint(x[finite]).astype(np.int64)
    yi[finite] = np.rint(y[finite]).astype(np.int64)
    inside = finite & (xi >= 0) & (xi < shape[1]) & (yi >= 0) & (yi < shape[0])
    return yi[inside], xi[inside]


@dataclass(frozen=True, slots=True)
class ProjectedBadPixels:
    """One frame's detector bad pixels on its output grid, as a sorted row-major linear index."""

    index: np.ndarray
    shape: tuple[int, int]

    @property
    def size(self) -> int:
        return int(self.index.size)

    @property
    def nbytes(self) -> int:
        return int(self.index.nbytes)

    def rows(self, y0: int, y1: int) -> tuple[np.ndarray, np.ndarray]:
        """(y, x) of the positions whose row falls in [y0, y1)."""
        width = self.shape[1]
        lo, hi = np.searchsorted(self.index, (y0 * width, y1 * width))
        chunk = self.index[lo:hi]
        return chunk // width, chunk % width

    def block_mask(self, sy0: int, sy1: int, sx0: int, sx1: int) -> np.ndarray:
        """Boolean of the [sy0:sy1, sx0:sx1] block, True where a detector bad pixel projects."""
        out = np.zeros((sy1 - sy0, sx1 - sx0), dtype=bool)
        self.apply(out, sy0, sy1, sx0, sx1, True)
        return out

    def apply(self, target: np.ndarray, sy0: int, sy1: int, sx0: int, sx1: int, value) -> int:
        """Write *value* into *target* (shaped like the [sy0:sy1, sx0:sx1] block) at each position."""
        if target.shape != (sy1 - sy0, sx1 - sx0):
            raise ValueError(f"block {target.shape} does not match the requested window {(sy1 - sy0, sx1 - sx0)}")
        yy, xx = self.rows(sy0, sy1)
        if not yy.size:
            return 0
        inside = (xx >= sx0) & (xx < sx1)
        target[yy[inside] - sy0, xx[inside] - sx0] = value
        return int(inside.sum())


def detector_badpixels(mask_file: str, badpix: int) -> tuple[np.ndarray, np.ndarray]:
    """(y, x) of every bad pixel in a bad-pixel mask."""
    return np.nonzero(fits.getdata(mask_file, memmap=False) == badpix)


def kernel_support_positions(ys, xs, input_wcs, output_wcs, shape, halfwidth=LANCZOS3_HALFWIDTH):
    """Output pixels whose resampling kernel reaches a detector bad pixel: SWarp zeroes exactly these.

    An output pixel is affected when its centre, mapped back onto the detector, lies within `halfwidth` of the
    bad pixel along both axes. Candidates are the (2*halfwidth + 1)^2 output pixels around each projection;
    their centres are inverse-mapped once each (they overlap heavily where bad pixels cluster)."""
    resampling = swarp_resampling_type()
    if resampling != "LANCZOS3":
        raise CoaddError.AssumptionFailedError(
            f"the kernel support of {halfwidth} px assumes RESAMPLING_TYPE LANCZOS3; ref/7dt.swarp says {resampling!r}"
        )
    xs = np.asarray(xs, np.float64)
    ys = np.asarray(ys, np.float64)
    if not xs.size:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    ra, dec = input_wcs.all_pix2world(xs, ys, 0)
    xo, yo = output_wcs.all_world2pix(ra, dec, 0)
    finite = np.isfinite(xo) & np.isfinite(yo)
    xs, ys, xo, yo = xs[finite], ys[finite], xo[finite], yo[finite]
    reach = int(np.ceil(halfwidth))
    offsets = np.arange(-reach, reach + 1)
    di, dj = (a.ravel() for a in np.meshgrid(offsets, offsets))
    cx = (np.rint(xo)[:, None] + di).astype(np.int64)
    cy = (np.rint(yo)[:, None] + dj).astype(np.int64)
    inside = (cx >= 0) & (cx < shape[1]) & (cy >= 0) & (cy < shape[0])
    width = int(shape[1])
    linear = cy * width + cx
    unique, inverse = np.unique(linear[inside], return_inverse=True)
    r, d = output_wcs.all_pix2world((unique % width).astype(np.float64), (unique // width).astype(np.float64), 0)
    px, py = input_wcs.all_world2pix(r, d, 0)
    dx = np.full(cx.shape, np.inf)
    dy = np.full(cy.shape, np.inf)
    dx[inside] = px[inverse] - np.repeat(xs, cx.shape[1]).reshape(cx.shape)[inside]
    dy[inside] = py[inverse] - np.repeat(ys, cy.shape[1]).reshape(cy.shape)[inside]
    keep = (np.abs(dx) < halfwidth) & (np.abs(dy) < halfwidth)
    hit = np.unique(linear[keep])
    return hit // width, hit % width


def project_badpixels(ys, xs, input_header, output_header, shape, footprint: str = "1px") -> ProjectedBadPixels:
    """Project detector bad pixels onto an output grid: the nearest output pixel ('1px'), or every output pixel
    whose LANCZOS3 resampling kernel touches the bad pixel ('conservative')."""
    if footprint == "conservative":
        yi, xi = kernel_support_positions(ys, xs, WCS(input_header), WCS(output_header), shape)
    else:
        ra, dec = WCS(input_header).all_pix2world(np.asarray(xs, np.float64), np.asarray(ys, np.float64), 0)
        yi, xi = nearest_output_pixels(ra, dec, output_header, shape)
    index = np.unique(yi * int(shape[1]) + xi).astype(np.int32, copy=False)
    return ProjectedBadPixels(index=index, shape=(int(shape[0]), int(shape[1])))
