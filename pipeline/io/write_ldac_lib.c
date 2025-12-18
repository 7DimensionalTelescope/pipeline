// write_ldac_lib.c
// Build (Linux):
//   gcc -O2 -fPIC -shared -o libwrite_ldac.so write_ldac_lib.c -lcfitsio -lm
// Build (macOS):
//   clang -O2 -fPIC -dynamiclib -o libwrite_ldac.dylib write_ldac_lib.c -lcfitsio -lm

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "fitsio.h"

#define CHECK_OR_GOTO(msg) do { if (status) { fits_report_error(stderr, status); fprintf(stderr, "Error at %s\n", msg); goto cleanup; } } while (0)

typedef struct {
    const char *cards80; // contiguous 80-byte cards (no newlines)
    long  ncards;        // number of cards
} header_blob;

/* Create empty primary HDU in an already-created memfile */
static void create_primary_hdu_in_mem(fitsfile *outfptr, int *pstatus) {
    int status = *pstatus;
    fits_create_img(outfptr, 8, 0, NULL, &status);
    fits_write_chksum(outfptr, &status);
    *pstatus = status;
}

/* Append LDAC_IMHEAD: 1-row BINTABLE, one column "Field Header Card" with TDIM='(80,N)' */
static void append_ldac_imhead(fitsfile *outfptr, const header_blob *hb, int *pstatus) {
    int status = *pstatus;
    long nrows   = 1;
    int  tfields = 1;

    char *ttype[1]; char *tform[1]; char *tunit[1];
    ttype[0] = (char*)"Field Header Card";

    long width = 80L * hb->ncards;
    char tform_buf[64];
    snprintf(tform_buf, sizeof(tform_buf), "%ldA", width);
    tform[0] = tform_buf;
    tunit[0] = NULL;

    fits_create_tbl(outfptr, BINARY_TBL, nrows, tfields, ttype, tform, tunit, "LDAC_IMHEAD", &status);
    if (status) { *pstatus = status; return; }

    char tdim[64];
    snprintf(tdim, sizeof(tdim), "(80, %ld)", hb->ncards);
    fits_update_key(outfptr, TSTRING, "TDIM1", tdim, "LDAC header cards shape", &status);
    if (status) { *pstatus = status; return; }

    fits_write_col(outfptr, TBYTE, 1, 1, 1, width, (void*)hb->cards80, &status);
    fits_write_chksum(outfptr, &status);

    *pstatus = status;
}

/* Copy first BINARY_TBL HDU from a FITS file opened from memory, and rename to LDAC_OBJECTS */
static void append_ldac_objects_from_mem(fitsfile *outfptr,
                                         const uint8_t *table_buf, size_t table_len,
                                         int *pstatus) {
    int status = *pstatus;
    fitsfile *tfptr = NULL;
    int hdutype = 0;
    int nhdus = 0;

    // Open input table FITS from memory (read-only)
    // CFITSIO 4.3.0 signature: ffomem(fptr, name, mode, buffptr, buffsize, deltasize, mem_realloc, status)
    void *ro_buf = (void*)table_buf;           // CFITSIO needs void** to the buffer pointer
    size_t ro_len = table_len;
    if (fits_open_memfile(&tfptr, "mem://table",
                          READONLY,
                          &ro_buf, &ro_len,
                          0,                      /* deltasize (no growth in READONLY) */
                          NULL,                   /* mem_realloc */
                          &status)) { *pstatus = status; return; }

    fits_get_num_hdus(tfptr, &nhdus, &status);
    if (status) { fits_close_file(tfptr, &status); *pstatus = status; return; }

    // Find first BINARY_TBL HDU
    int found = 0;
    for (int i = 1; i <= nhdus; ++i) {
        fits_movabs_hdu(tfptr, i, &hdutype, &status);
        if (status) break;
        if (hdutype == BINARY_TBL) { found = 1; break; }
    }
    if (!found) {
        status = BAD_HDU_NUM; // no BINARY_TBL found
        fits_close_file(tfptr, &status);
        *pstatus = status;
        return;
    }

    fits_copy_hdu(tfptr, outfptr, 0, &status);
    if (!status) {
        fits_update_key(outfptr, TSTRING, "EXTNAME", "LDAC_OBJECTS", NULL, &status);
        fits_write_chksum(outfptr, &status);
    }

    fits_close_file(tfptr, &status);
    *pstatus = status;
}

/* === Public API (ctypes) ===
   Packs: Primary + LDAC_IMHEAD(cards80) + LDAC_OBJECTS(copied from table_buf)
   Returns: *ldac_out/*ldac_len (owned by caller; free with ldac_free()) */
int write_ldac_from_arrays(const char *cards80, long ncards,
                           const uint8_t *table_buf, size_t table_len,
                           uint8_t **ldac_out, size_t *ldac_len)
{
    int status = 0;
    fitsfile *outfptr = NULL;
    void *outbuf = NULL; size_t outsize = 0;
    // Choose a reasonable growth quantum for the memfile buffer (multiple of FITS record = 2880)
    size_t deltasize = 2880;

    // Create output FITS in memory
    // CFITSIO 4.3.0 signature: ffimem(fptr, buffptr, buffsize, deltasize, mem_realloc, status)
    if (fits_create_memfile(&outfptr, &outbuf, &outsize, deltasize, realloc, &status))
        return status;

    create_primary_hdu_in_mem(outfptr, &status);
    if (status) goto cleanup;

    header_blob hb = { .cards80 = cards80, .ncards = ncards };
    append_ldac_imhead(outfptr, &hb, &status);
    if (status) goto cleanup;

    append_ldac_objects_from_mem(outfptr, table_buf, table_len, &status);
    if (status) goto cleanup;

    // Close to flush headers/data into the memory buffer, but DO NOT free it.
    // Because we supplied realloc/free callbacks, CFITSIO allocates via those;
    // after close, the buffer remains valid and is owned by the caller to free().
    fits_close_file(outfptr, &status);
    outfptr = NULL;
    if (status) goto cleanup;

    // Hand buffer ownership to caller
    *ldac_out = (uint8_t*)outbuf;
    *ldac_len = outsize;
    return 0;

cleanup:
    // On error, close the memfile; CFITSIO will free any internal buffers as needed.
    if (outfptr) fits_close_file(outfptr, &status);
    return status;
}

/* Caller frees the buffer via ctypes */
void ldac_free(void *p) { free(p); }