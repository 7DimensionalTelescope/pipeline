"""Durable per-single weight maps (user decision 2026-08-10).

The "equivalent things" contract:
- PRISTINE noise-model output: raw inverse variance (ADU^-2) of the single frame --
  no FLXSCALE (the combine divides by it downstream) and no interp-hole/bpmask zeroing
  (each coadd config applies its own policy later). It is exactly the weight kernel's
  output before interpolation touches it.
- Same uint16 quantization grid as the factory sidecars (write_weight_int16), stored as
  a RICE-compressed image: bit-identical values at ~32% of a sci single's size.
- Masters provenance (mdark/mflat/biassig/flatsig) stamped and verified on load; a map
  built from different masters is refused, forcing a recompute.
"""

import os

import numpy as np
from astropy.io import fits

from ..path.path import PathHandler
from ..version import __version__

single_weight_path = PathHandler.single_weight_map

_PROV_CARDS = {"d": "WMDARK", "f": "WMFLAT", "sz": "WMBSIG", "sf": "WMFSIG"}


def _masters_cards(masters: dict) -> dict:
    return {card: os.path.basename(str(masters[key])) for key, card in _PROV_CARDS.items()}


def persist_single_weight(out_path: str, weight: np.ndarray, masters: dict) -> str:
    """Write the pristine weight on the sidecar's uint16 grid, RICE-compressed."""
    weight = np.where(np.isfinite(weight) & (weight >= 0), weight, 0.0).astype(np.float32)
    wmax = float(weight.max()) if weight.size else 0.0
    bscale = wmax / 65535.0 if wmax > 0 else 1.0
    # unsigned-int16 convention (BZERO = 32768*BSCALE): quantize through astropy's own
    # scale() so the stored integers are bit-identical to write_weight_int16's
    tmp = fits.PrimaryHDU(weight)
    tmp.scale("int16", bscale=bscale, bzero=32768.0 * bscale)
    hdu = fits.CompImageHDU(data=tmp.data, compression_type="RICE_1")
    hdu.header["BSCALE"] = bscale
    hdu.header["BZERO"] = 32768.0 * bscale
    hdu.header["WMUNITS"] = ("ADU**-2 raw inv variance", "no FLXSCALE, no interp/bpmask zeroing")
    for card, val in _masters_cards(masters).items():
        hdu.header[card] = (val, "master frame this map was built from")
    hdu.header["WMPIPVER"] = (__version__, "pipeline version at build time")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(out_path, overwrite=True)
    return out_path


def check_single_weight(path: str, masters: dict) -> bool:
    """Header-only provenance check; no pixel I/O."""
    if not os.path.exists(path):
        return False
    try:
        hdr = fits.getheader(path, ext=1)
    except OSError:
        return False
    return all(str(hdr.get(card, "")) == val for card, val in _masters_cards(masters).items())


def load_single_weight(path: str, masters: dict) -> np.ndarray | None:
    """Provenance-verified load -> float32 raw inverse variance, else None."""
    if not check_single_weight(path, masters):
        return None
    try:
        return fits.getdata(path, ext=1).astype(np.float32)
    except OSError:
        return None
