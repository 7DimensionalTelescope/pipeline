from pathlib import Path
from astropy.io import fits

keys = ("EXPTIME", "GAIN", "XBINNING")
groups: dict[tuple, list[str]] = {}

for f in Path(".").glob("**/*.fits"):
    with fits.open(f, memmap=False) as hdul:
        hdr = hdul[0].header
    key = tuple(hdr.get(k) for k in keys)
    groups.setdefault(key, []).append(str(f))
