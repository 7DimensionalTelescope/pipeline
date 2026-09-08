from .sciproc import ProcessSpec, SciProcessRegistry


WHITE_COADD_SPEC = ProcessSpec(
    name="white_coadd",
    config_section="imcoadd",
    error_code=8,
    progress_start=0,
    progress_end=80,
    milestones=(
        ("bkgsub", 10),
        ("zpscale", 20),
        ("calculate_weight_map", 30),
        ("apply_bpmask", 40),
        ("joint_registration", 50),
        ("run_convolution", 60),
        ("coadd_with_swarp", 70),
        ("plot_coadd_image", 78),
    ),
)

PHOT7DS_SPEC = ProcessSpec(
    name="phot7ds",
    config_section="phot7ds",
    error_code=14,
    progress_start=80,
    progress_end=100,
)

# user-input alternative to phot7ds; progress_start 81 only for registry uniqueness
WHITE_PHOTOMETRY_SPEC = ProcessSpec(
    name="white_photometry",
    config_section="photometry",
    error_code=10,
    progress_start=81,
    progress_end=100,
    photometry_mode="white_photometry",
    input_key="white_image",
)

CROSSFILTERPROCESS_REGISTRY = SciProcessRegistry([WHITE_COADD_SPEC, PHOT7DS_SPEC, WHITE_PHOTOMETRY_SPEC])
