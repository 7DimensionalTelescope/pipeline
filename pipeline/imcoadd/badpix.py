"""Sparse output-grid positions of detector bad pixels, shared by the quality masks and the combine.

One structure per input frame replaces the read-modify-write of the SWarp resampled weight map: the
nearest output pixel of every detector bad pixel is carried as a row-sorted linear index and applied
to whatever validity array a coadd backend already holds. Coordinates are in the resampled frame's
own pixel grid, which is what every backend slices with x0/y0.
"""

from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


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


def project_badpixels(ys, xs, input_header, output_header, shape) -> ProjectedBadPixels:
    """Project detector bad pixels onto an output grid, one output pixel per detector bad pixel."""
    ra, dec = WCS(input_header).all_pix2world(np.asarray(xs, np.float64), np.asarray(ys, np.float64), 0)
    yi, xi = nearest_output_pixels(ra, dec, output_header, shape)
    index = np.unique(yi * int(shape[1]) + xi).astype(np.int32, copy=False)
    return ProjectedBadPixels(index=index, shape=(int(shape[0]), int(shape[1])))
