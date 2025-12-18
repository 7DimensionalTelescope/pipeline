import os
from io import BytesIO
import ctypes
from astropy.io import fits
from astropy.table import Table

# 1) Load the shared library
# Set library path to ensure cfitsio can be found
cfitsio_paths = [
    "/usr/local/cfitsio/lib",
    "/usr/lib/x86_64-linux-gnu",
]
for path in cfitsio_paths:
    if os.path.exists(path):
        os.environ.setdefault("LD_LIBRARY_PATH", "")
        if path not in os.environ["LD_LIBRARY_PATH"]:
            os.environ["LD_LIBRARY_PATH"] = f"{path}:{os.environ['LD_LIBRARY_PATH']}"

libname = os.path.join(os.path.dirname(__file__), "libwrite_ldac.so")
try:
    lib = ctypes.CDLL(libname)
except OSError as e:
    # If loading fails, try with explicit library path
    import ctypes.util
    # Try to find cfitsio library
    cfitsio_lib = None
    for path in cfitsio_paths:
        cfitsio_candidate = os.path.join(path, "libcfitsio.so.10")
        if os.path.exists(cfitsio_candidate):
            cfitsio_lib = cfitsio_candidate
            break
    
    if cfitsio_lib:
        # Preload cfitsio to satisfy dependencies
        ctypes.CDLL(cfitsio_lib, mode=ctypes.RTLD_GLOBAL)
        lib = ctypes.CDLL(libname)
    else:
        raise OSError(f"Failed to load {libname}: {e}. Also could not find libcfitsio.so.10")

# 2) Declare arg/return types
# int write_ldac_from_arrays(const char *cards80, long ncards,
#                            const uint8_t *table_buf, size_t table_len,
#                            uint8_t **ldac_out, size_t *ldac_len)
lib.write_ldac_from_arrays.argtypes = [
    ctypes.c_char_p,  # cards80
    ctypes.c_long,  # ncards
    ctypes.POINTER(ctypes.c_uint8),  # table_buf
    ctypes.c_size_t,  # table_len
    ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),  # ldac_out**
    ctypes.POINTER(ctypes.c_size_t),  # ldac_len*
]
lib.write_ldac_from_arrays.restype = ctypes.c_int

# void ldac_free(void *p)
lib.ldac_free.argtypes = [ctypes.c_void_p]
lib.ldac_free.restype = None


# 3) Helper: turn an astropy Header into one contiguous 80-char card blob
def header_to_cards80_bytes(header: fits.Header) -> bytes:
    # Ensure compliant 80-char cards. Astropy guarantees this for normal keys,
    # but LONGSTR or CONTINUE can expand; let Astropy serialize then split.
    # The PrimaryHDU writer pads to 80. We leverage Header.tostring(sep='\n') for cards.
    txt = header.tostring(sep="\n", endcard=True)  # includes END card
    lines = [ln.rstrip("\n") for ln in txt.splitlines()]
    # Pad or trim to exactly 80 bytes per card
    cards = [(ln[:80]).ljust(80) for ln in lines]
    blob = "".join(cards).encode("ascii", errors="strict")
    return blob


# 4) Helper: turn a Table (or FITS_rec) into a memory FITS with a BINTABLE HDU
def table_to_fits_bintable_bytes(tbl) -> bytes:
    # Accept either astropy.table.Table or already a BinTableHDU/FITS_rec
    if isinstance(tbl, Table):
        hdu = fits.BinTableHDU(data=tbl.as_array())
    elif isinstance(tbl, fits.BinTableHDU):
        hdu = tbl
    elif isinstance(tbl, fits.FITS_rec):
        hdu = fits.BinTableHDU(data=tbl)
    else:
        raise TypeError("Pass an astropy Table, BinTableHDU, or FITS_rec")

    # Write a minimal FITS with Primary + this BINTABLE as the *first* binary table
    hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
    buf = BytesIO()
    hdul.writeto(buf, overwrite=True, output_verify="exception")  # strict compliance
    return buf.getvalue()


# 5) Glue: call the C function and write the LDAC file
def write_ldac(header: fits.Header, table: Table | fits.BinTableHDU | fits.FITS_rec, out_path: str):
    cards_blob = header_to_cards80_bytes(header)
    ncards = len(cards_blob) // 80
    table_blob = table_to_fits_bintable_bytes(table)

    # Prepare ctypes inputs
    cards_ptr = ctypes.c_char_p(cards_blob)  # const char*
    table_arr = (ctypes.c_uint8 * len(table_blob)).from_buffer_copy(table_blob)
    table_ptr = ctypes.cast(table_arr, ctypes.POINTER(ctypes.c_uint8))
    table_len = ctypes.c_size_t(len(table_blob))

    # Outputs
    out_ptr = ctypes.POINTER(ctypes.c_uint8)()
    out_len = ctypes.c_size_t()

    status = lib.write_ldac_from_arrays(
        cards_ptr, ctypes.c_long(ncards), table_ptr, table_len, ctypes.byref(out_ptr), ctypes.byref(out_len)
    )
    if status != 0:
        raise RuntimeError(f"CFITSIO/LDAC pack failed with status={status}")

    try:
        # Copy out the bytes and write to disk
        ldac_bytes = ctypes.string_at(out_ptr, out_len.value)
        with open(out_path, "wb") as f:
            f.write(ldac_bytes)
    finally:
        # Always free the C buffer
        lib.ldac_free(out_ptr)


# === Example usage ===
if __name__ == "__main__":
    # Build an example header and table
    hdr = fits.Header()
    hdr["SIMPLE"] = (True, "conforms to FITS standard")
    hdr["BITPIX"] = (8, "array data type")
    hdr["NAXIS"] = (0, "number of array dimensions")
    hdr["MYKEY"] = "example"
    # END will be added automatically by header_to_cards80_bytes()

    t = Table({"ID": [1, 2, 3], "X": [10.0, 20.5, 30.1], "Y": [5.5, 6.1, 7.2]})

    write_ldac(hdr, t, "output_ldac.fits")
    print("Wrote output_ldac.fits")
