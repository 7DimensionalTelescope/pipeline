/*
* write_ldac.c
*
* Create a SCAMP-compatible FITS_LDAC catalog from:
*   (1) an image header (WCS etc.) read from a FITS image HDU or a .hdr text file
*   (2) a FITS binary table of sources (columns/rows)
*
* Output layout:
*   Primary HDU (empty)
*   EXTNAME='LDAC_IMHEAD'  (1-row BINTABLE, 1 column 'Field Header Card', TDIM1='(80, N)')
*   EXTNAME='LDAC_OBJECTS' (BINTABLE copied from your input table)
*
* Build: gcc -O2 -o write_ldac write_ldac.c -lcfitsio -lm
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <inttypes.h>
#include "fitsio.h"

#define CHECK_STATUS(msg) do { \
    if (status) { \
        fits_report_error(stderr, status); \
        fprintf(stderr, "Error at %s\n", msg); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

typedef struct {
    char *cards;     // contiguous 80-char records
    long  ncards;    // number of 80-char cards
} header_blob;

/* Pad or trim a string to exactly 80 chars (FITS card) */
static void pad_to_80(char *dst, const char *src) {
    size_t len = src ? strlen(src) : 0;
    size_t i;
    for (i = 0; i < 80; ++i) {
        dst[i] = (i < len) ? src[i] : ' ';
    }
}

/* Read header cards from a FITS HDU */
// static header_blob read_header_from_fits(const char *fname, int hdu_index) {
//    fitsfile *fptr = NULL;
//    int status = 0, hdutype = 0;

//    // NOTE: fits_get_hdrspace requires int*, not long*.
//    int nkeys_i = 0, morekeys_i = 0;

//    header_blob hb = {0};

//    fits_open_file(&fptr, fname, READONLY, &status);
//    CHECK_STATUS("fits_open_file(image)");
//    fits_movabs_hdu(fptr, hdu_index, &hdutype, &status);
//    CHECK_STATUS("fits_movabs_hdu(image)");

//    fits_get_hdrspace(fptr, &nkeys_i, &morekeys_i, &status);
//    CHECK_STATUS("fits_get_hdrspace(image)");

//    if (nkeys_i <= 0) {
//        fprintf(stderr, "No header cards found in %s[HDU=%d]\n", fname, hdu_index);
//        exit(EXIT_FAILURE);
//    }

//    hb.ncards = (long)nkeys_i;                 // store as long in our struct
//    hb.cards  = (char*)malloc(80 * (size_t)hb.ncards);
//    if (!hb.cards) { perror("malloc"); exit(EXIT_FAILURE); }

//    // fits_read_record's 'keynum' argument is 1-based int
//    for (int i = 1; i <= nkeys_i; ++i) {
//        char card[FLEN_CARD]; // 81 incl. null
//        fits_read_record(fptr, i, card, &status);
//        CHECK_STATUS("fits_read_record(image)");
//        pad_to_80(&hb.cards[(long)(i-1)*80], card);
//    }

//    fits_close_file(fptr, &status);
//    CHECK_STATUS("fits_close_file(image)");
//    return hb;
// }

// this one ensures the END card is read
/* Read header cards from a FITS HDU */
static header_blob read_header_from_fits(const char *fname, int hdu_index) {
    fitsfile *fptr = NULL;
    int status = 0, hdutype = 0;
    // NOTE: fits_get_hdrspace requires int*, not long*.
    int nkeys_i = 0, morekeys_i = 0;
 
    header_blob hb = (header_blob){0};
 
    fits_open_file(&fptr, fname, READONLY, &status);
    CHECK_STATUS("fits_open_file(image)");
    fits_movabs_hdu(fptr, hdu_index, &hdutype, &status);
    CHECK_STATUS("fits_movabs_hdu(image)");
 
    fits_get_hdrspace(fptr, &nkeys_i, &morekeys_i, &status);
    CHECK_STATUS("fits_get_hdrspace(image)");
 
    if (nkeys_i <= 0) {
        fprintf(stderr, "No header cards found in %s[HDU=%d]\n", fname, hdu_index);
        exit(EXIT_FAILURE);
    }
 
    /* +1 to store the END card */
    hb.ncards = (long)nkeys_i + 1;                 // store as long in our struct
    hb.cards  = (char*)malloc(80 * (size_t)hb.ncards);
    if (!hb.cards) { perror("malloc"); exit(EXIT_FAILURE); }
 
    for (int i = 1; i <= nkeys_i; ++i) {
        char card[FLEN_CARD]; // 81 incl. null
        fits_read_record(fptr, i, card, &status);
        CHECK_STATUS("fits_read_record(image)");
        pad_to_80(&hb.cards[(long)(i-1)*80], card);
    }
 
    /* Read or synthesize the END card */
    char endcard[FLEN_CARD] = "END";
    int loc_status = 0;
    fits_read_record(fptr, nkeys_i+1, endcard, &loc_status); /* CFITSIO returns the END record here */
    pad_to_80(&hb.cards[(long)nkeys_i*80], endcard);
 
    fits_close_file(fptr, &status);
    CHECK_STATUS("fits_close_file(image)");
    return hb;
 }

/* Read header cards from a text .hdr file: one 80-char card per line (longer lines are trimmed) */
static header_blob read_header_from_text(const char *hdr_path) {
    FILE *fp = fopen(hdr_path, "r");
    if (!fp) { perror("fopen(hdr)"); exit(EXIT_FAILURE); }

    header_blob hb = {0};
    size_t cap = 256;
    hb.cards = (char*)malloc(80 * cap);
    if (!hb.cards) { perror("malloc"); exit(EXIT_FAILURE); }

    char line[4096];
    int saw_end = 0;
    while (fgets(line, sizeof(line), fp)) {
        size_t L = strlen(line);
        if (L && (line[L-1] == '\n' || line[L-1] == '\r')) line[--L] = '\0';
        if (strncmp(line, "END", 3) == 0) saw_end = 1;

        if (hb.ncards >= (long)cap) {
            cap *= 2;
            hb.cards = (char*)realloc(hb.cards, 80 * cap);
            if (!hb.cards) { perror("realloc"); exit(EXIT_FAILURE); }
        }
        pad_to_80(&hb.cards[hb.ncards * 80], line);
        hb.ncards++;
    }
    fclose(fp);

    if (hb.ncards == 0) {
        fprintf(stderr, "No lines found in header text file %s\n", hdr_path);
        exit(EXIT_FAILURE);
    }

    if (!saw_end) {
        if (hb.ncards >= (long)cap) {
            cap += 1;
            hb.cards = (char*)realloc(hb.cards, 80 * cap);
            if (!hb.cards) { perror("realloc"); exit(EXIT_FAILURE); }
        }
        char endline[4] = "END";
        pad_to_80(&hb.cards[hb.ncards * 80], endline);
        hb.ncards++;
    }

    return hb;
}
// static header_blob read_header_from_text(const char *hdr_path) {
//     FILE *fp = fopen(hdr_path, "r");
//     if (!fp) { perror("fopen(hdr)"); exit(EXIT_FAILURE); }

//     header_blob hb = {0};
//     size_t cap = 256; // initial guess
//     hb.cards = (char*)malloc(80 * cap);
//     if (!hb.cards) { perror("malloc"); exit(EXIT_FAILURE); }

//     char line[4096];
//     while (fgets(line, sizeof(line), fp)) {
//         // strip trailing newline
//         size_t L = strlen(line);
//         if (L && (line[L-1] == '\n' || line[L-1] == '\r')) line[--L] = '\0';

//         if (hb.ncards >= (long)cap) {
//             cap *= 2;
//             hb.cards = (char*)realloc(hb.cards, 80 * cap);
//             if (!hb.cards) { perror("realloc"); exit(EXIT_FAILURE); }
//         }
//         pad_to_80(&hb.cards[hb.ncards * 80], line);
//         hb.ncards++;
//     }
//     fclose(fp);

//     if (hb.ncards == 0) {
//         fprintf(stderr, "No lines found in header text file %s\n", hdr_path);
//         exit(EXIT_FAILURE);
//     }
//     return hb;
// }

/* Create a Primary HDU (empty) in output */
static void create_primary_hdu(fitsfile **outfptr, const char *outname) {
    int status = 0;
    fits_create_file(outfptr, outname, &status);
    CHECK_STATUS("fits_create_file(output)");
    // empty primary HDU
    fits_create_img(*outfptr, 8, 0, NULL, &status);
    CHECK_STATUS("fits_create_img(primary)");
    fits_write_chksum(*outfptr, &status); // checksum primary
    CHECK_STATUS("fits_write_chksum(primary)");
}

/* Append LDAC_IMHEAD BINTABLE with TDIM1='(80, N)' and data = packed header cards */
static void append_ldac_imhead(fitsfile *outfptr, const header_blob *hb) {
    int status = 0;
    long nrows   = 1;
    int  tfields = 1;

    char *ttype[1]; char *tform[1]; char *tunit[1];
    ttype[0] = "Field Header Card";

    // column width in bytes = 80 * ncards
    long width = 80L * hb->ncards;
    char tform_buf[64];
    snprintf(tform_buf, sizeof(tform_buf), "%ldA", width);
    tform[0] = tform_buf;
    tunit[0] = NULL;

    fits_create_tbl(outfptr, BINARY_TBL, nrows, tfields, ttype, tform, tunit, "LDAC_IMHEAD", &status);
    CHECK_STATUS("fits_create_tbl(LDAC_IMHEAD)");

    // TDIM1 = "(80, Ncards)"
    char tdim[64];
    snprintf(tdim, sizeof(tdim), "(80, %ld)", hb->ncards);
    fits_update_key(outfptr, TSTRING, "TDIM1", tdim, "LDAC header cards shape", &status);
    CHECK_STATUS("fits_update_key(TDIM1)");

    // Write the one row (all bytes of the single element)
    long firstrow   = 1;
    long firstelem  = 1;
    long nelements  = width; // number of bytes to write
    fits_write_col(outfptr, TBYTE, 1, firstrow, firstelem, nelements, (void*)hb->cards, &status);
    CHECK_STATUS("fits_write_col(LDAC_IMHEAD)");

    fits_write_chksum(outfptr, &status);
    CHECK_STATUS("fits_write_chksum(LDAC_IMHEAD)");
}

// /* Append LDAC_IMHEAD with N rows of 80A (no TDIM) */
// static void append_ldac_imhead(fitsfile *outfptr, const header_blob *hb) {
//     int status = 0;

//     long nrows   = hb->ncards;  // one row per 80-char card
//     int  tfields = 1;

//     char *ttype[1]; char *tform[1]; char *tunit[1];
//     ttype[0] = "Field Header Card";
//     tform[0] = "80A";           // each row is a single 80-char string
//     tunit[0] = NULL;

//     fits_create_tbl(outfptr, BINARY_TBL, nrows, tfields, ttype, tform, tunit, "LDAC_IMHEAD", &status);
//     CHECK_STATUS("fits_create_tbl(LDAC_IMHEAD)");

//     // Optional but harmless: set EXTVER=1
//     long one = 1;
//     fits_update_key(outfptr, TLONG, "EXTVER", &one, NULL, &status);
//     CHECK_STATUS("fits_update_key(EXTVER)");

//     // Write each 80-char card into its own row
//     for (long r = 1; r <= nrows; ++r) {
//         // CFITSIO expects a C string for 80A; ensure we pass 80 bytes (no embedded NULs).
//         // We'll write as raw bytes to avoid premature NUL termination issues.
//         fits_write_col(outfptr, TBYTE, 1, r, 1, 80, (void*)&hb->cards[(r-1)*80], &status);
//         CHECK_STATUS("fits_write_col(LDAC_IMHEAD row)");
//     }

//     fits_write_chksum(outfptr, &status);
//     CHECK_STATUS("fits_write_chksum(LDAC_IMHEAD)");
// }

/* Copy the source BINTABLE from input and rename EXTNAME to LDAC_OBJECTS */
static void append_ldac_objects(fitsfile *outfptr, const char *table_path, int table_hdu) {
    fitsfile *tfptr = NULL;
    int status = 0, hdutype = 0;

    fits_open_file(&tfptr, table_path, READONLY, &status);
    CHECK_STATUS("fits_open_file(table)");
    fits_movabs_hdu(tfptr, table_hdu, &hdutype, &status);
    CHECK_STATUS("fits_movabs_hdu(table)");

    if (hdutype != BINARY_TBL) {
        fprintf(stderr, "Selected HDU in %s is not a BINARY table (HDU=%d)\n", table_path, table_hdu);
        exit(EXIT_FAILURE);
    }

    // Copy the entire HDU (header + data)
    fits_copy_hdu(tfptr, outfptr, 0, &status);
    CHECK_STATUS("fits_copy_hdu(objects)");

    // Ensure EXTNAME is LDAC_OBJECTS
    fits_update_key(outfptr, TSTRING, "EXTNAME", "LDAC_OBJECTS", NULL, &status);
    CHECK_STATUS("fits_update_key(EXTNAME LDAC_OBJECTS)");

    fits_write_chksum(outfptr, &status);
    CHECK_STATUS("fits_write_chksum(LDAC_OBJECTS)");

    fits_close_file(tfptr, &status);
    CHECK_STATUS("fits_close_file(table)");
}

/* Print CLI help */
static void usage(const char *prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s -i image.fits [-I hdu] -t table.fits [-T hdu] -o out.ldac.fits\n"
        "  %s -H header.hdr      -t table.fits [-T hdu] -o out.ldac.fits\n"
        "\n"
        "Options:\n"
        "  -i  FITS image file to harvest header cards from\n"
        "  -I  Image HDU index (1-based). Default: 1\n"
        "  -H  Header text file (80-char FITS cards per line). If set, overrides -i/-I\n"
        "  -t  FITS binary table with source catalog (columns/rows)\n"
        "  -T  Table HDU index (1-based). Default: 1\n"
        "  -o  Output LDAC FITS file (prefix with '!' to overwrite)\n"
        "\n"
        "Notes:\n"
        "  * The output contains Primary + LDAC_IMHEAD + LDAC_OBJECTS.\n"
        "  * LDAC_IMHEAD packs exactly N 80-char cards and sets TDIM1='(80, N)'.\n"
        "  * The input table HDU must be a BINARY table; it is copied verbatim and\n"
        "    renamed to EXTNAME='LDAC_OBJECTS'.\n"
        , prog, prog);
}

/* Main */
int main(int argc, char **argv) {
    const char *image_path = NULL;
    const char *hdr_text   = NULL;
    const char *table_path = NULL;
    const char *out_path   = NULL;
    int image_hdu = 1;
    int table_hdu = 1;

    // Parse args
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-i") && i+1 < argc) { image_path = argv[++i]; continue; }
        if (!strcmp(argv[i], "-I") && i+1 < argc) { image_hdu  = atoi(argv[++i]); continue; }
        if (!strcmp(argv[i], "-H") && i+1 < argc) { hdr_text   = argv[++i]; continue; }
        if (!strcmp(argv[i], "-t") && i+1 < argc) { table_path = argv[++i]; continue; }
        if (!strcmp(argv[i], "-T") && i+1 < argc) { table_hdu  = atoi(argv[++i]); continue; }
        if (!strcmp(argv[i], "-o") && i+1 < argc) { out_path   = argv[++i]; continue; }
        if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
        fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
        usage(argv[0]); return 1;
    }

    if ((!hdr_text && !image_path) || !table_path || !out_path) {
        usage(argv[0]); return 1;
    }

    header_blob hb = {0};
    if (hdr_text) hb = read_header_from_text(hdr_text);
    else          hb = read_header_from_fits(image_path, image_hdu);

    // Create output and write
    fitsfile *outfptr = NULL;
    create_primary_hdu(&outfptr, out_path);
    append_ldac_imhead(outfptr, &hb);
    append_ldac_objects(outfptr, table_path, table_hdu);

    // Done
    int status = 0;
    fits_close_file(outfptr, &status);
    CHECK_STATUS("fits_close_file(output)");

    free(hb.cards);
    return 0;
}