# Pipeline Dir Tree
# output_dir  (under PROCESSED_DIR)
#  └── difference
#  └── figures
#  └── singles
#      └── *.fits
#      └── *_cat.fits
#  └── coadd
#      └── *_coadd.fits
#      └── *_coadd_cat.fits
#  └── *.log
#  └── *.yml

# User-input Dir Tree
# output_dir = working_dir = cwd
#  └── difference
#      └── *._diff.fits
#      └── *._diff_cat.fits
#  └── figures
#  └── tmp
#      └── astrometry
#      └── imcoadd
#      └── imsubtract
#      └── photometry
#  └── *.fits
#  └── *_cat.fits
#  └── *_coadd.fits
#  └── *_coadd_cat.fits
#  └── *.log
#  └── *.yml


# DIRNAME for the directory names, not the full paths
FIGURES_DIRNAME = "figures"
ASTRM_DIRNAME = "astrometry"
PHOTOMETRY_DIRNAME = "photometry"
SINGLES_DIRNAME = "singles"
DAILY_COADD_DIRNAME = "coadd"
DIFFIM_DIRNAME = "difference"
TMP_DIRNAME = "tmp"
IMCOADD_TMP_DIRNAME = "imcoadd"
IMSUBTRACT_TMP_DIRNAME = "imsubtract"
