"""Persistent smooth weight of a single, saved beside it."""

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from ..path.path import PathHandler

single_weight_path = PathHandler.weight_map


def persist_single_weight(out_path: str, weight: np.ndarray, header: fits.Header, cards: dict) -> str:
    """Smooth weight in a GZIP_2 extension WEIGHT, absolute step 1e-3 of its smallest value, NO_DITHER."""
    hdr = header.copy()
    hdr.strip()
    for key, value in cards.items():
        hdr[key] = value
    weight = np.asarray(weight, dtype=np.float32)
    hdu = fits.CompImageHDU(
        data=weight,
        header=WCS(hdr, fix=False).to_header(relax=True),
        compression_type="GZIP_2",
        tile_shape=(64, weight.shape[1]),
        quantize_level=-1e-3 * float(weight[weight > 0].min()),
        quantize_method=-1,
        name="WEIGHT",
    )
    hdu.header["BUNIT"] = ("ADU**-2", "single-frame inverse variance")
    fits.HDUList([fits.PrimaryHDU(header=hdr), hdu]).writeto(out_path, overwrite=True)
    return out_path
