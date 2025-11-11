# Pipeline Dir Tree
# output_dir  (under PROCESSED_DIR)
#  └── difference
#  └── figures
#  └── singles
#      └── *.fits
#      └── *_cat.fits
#  └── stacked
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
#      └── imstack
#      └── imsubtract
#      └── photometry
#  └── *.fits
#  └── *_cat.fits
#  └── *_coadd.fits
#  └── *_coadd_cat.fits
#  └── *.log
#  └── *.yml


# DIRNAME for the directory names, not the full paths
ASTRM_DIRNAME = "astrometry"
DIFFIM_DIRNAME = "difference"
FIGURES_DIRNAME = "figures"
IMSTACK_DIRNAME = "imstack"
IMSUBTRACT_DIRNAME = "subtracted"  # TODO: no longer in use. consider removing
PHOTOMETRY_DIRNAME = "photometry"
SINGLES_DIRNAME = "singles"
STACKED_DIRNAME = "stacked"
TMP_DIRNAME = "tmp"
