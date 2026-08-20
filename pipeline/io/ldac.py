from astropy.io import fits
from astropy.table import Table


def read_catalog(path: str, **kwargs) -> Table:
    """Source table of a photometry catalog, FITS_LDAC or legacy single-BINTABLE."""
    with fits.open(path, memmap=False) as hdul:
        index = next((i for i, hdu in enumerate(hdul) if hdu.name == "LDAC_OBJECTS"), None)
        if index is None:
            index = next(i for i, hdu in enumerate(hdul) if isinstance(hdu, fits.BinTableHDU))
        return Table.read(hdul[index], **kwargs)
