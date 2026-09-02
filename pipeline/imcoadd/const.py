from enum import IntFlag


ZP_KEY = "ZP_AUTO"


class MaskBit(IntFlag):
    BAD = 1
    SATURATED = 2
    SATELLITE = 4
    OUTLIER = 8
    HOT = 16
    DEAD = 32


MASK_HEADER_CARDS = {
    "MASKBAD": (int(MaskBit.BAD), "BAD: generic detector bad pixel"),
    "MASKSAT": (int(MaskBit.SATURATED), "SATURATED: detector saturation"),
    "MASKTRAI": (int(MaskBit.SATELLITE), "SATELLITE: Hough trail mask"),
    "MASKOUT": (int(MaskBit.OUTLIER), "OUTLIER: clipped-mean rejection"),
    "MASKHOT": (int(MaskBit.HOT), "HOT: reserved"),
    "MASKDEAD": (int(MaskBit.DEAD), "DEAD: reserved"),
}

IC_KEYS = [
    "EGAIN",
    "TELESCOP",
    "EGAIN",
    "FILTER",
    "OBJECT",
    "OBJCTRA",
    "OBJCTDEC",
    "JD",
    "MJD",
    "SKYVAL",
    "EXPTIME",
    ZP_KEY,
]

# self.keys = [
#     "imagetyp",
#     "telescop",
#     "object",
#     "filter",
#     "exptime",
#     "ul5_1",
#     "seeing",
#     "elong",
#     "ellip",
# ]

# Keys expected to be uniform across all inputs of a coadd; aggregate just
# picks one (the first unmasked value).
HOMOGENEOUS_KEYS = [
    "IMAGETYP",
    "OBJECT",
    "OBJTYPE",
    "OBJCTID",
    "FILTER",
    "INSTRUME",
    "TELESCOP",
    "OBSMODE",
    "SPECMODE",
    "OBSERVER",
    "NTELSCOP",
    "IS_TOO",
    "XBINNING",
    "YBINNING",
    "XPIXSZ",
    "YPIXSZ",
    "FOCALLEN",
    "FOCALRAT",
    "APTDIA",
    "APTAREA",
    "SITELAT",
    "SITELONG",
    "SITEELEV",
    "OBJCTRA",
    "OBJCTDEC",
    "EQUINOX",
    "RADESYS",
    # unsure: uncomment when confirmed homogeneous in current headers
    # "PIERSIDE",
    # "FWHEEL",
    # "OBJCTROT",
    # "FOCNAME",
    # "FOCPOS",
    # "FOCUSPOS",
    # "FOCUSSZ",
    # "ROWORDER",
    # "SWCREATE",
    # "FOCRATIO",  # superseded by FOCALRAT
]

# Keys expected to vary across inputs; aggregate returns a representative
# (numeric mean / "MIXED" for strings).
INHOMOGENEOUS_KEYS = [
    "ALTITUDE",
    "AZIMUTH",
    "CENTALT",
    "CENTAZ",
    "RA",
    "DEC",
    "AIRMASS",
    "CCD-TEMP",
    "SET-TEMP",
    "MOONSEP",
    "MOONPHAS",
    # DATE-LOC, DATE-OBS handled separately in coadd_header
    # EGAIN handled separately (coadd-effective EGAIN with FLXSCALE)
]

CORE_KEYS = HOMOGENEOUS_KEYS + INHOMOGENEOUS_KEYS
