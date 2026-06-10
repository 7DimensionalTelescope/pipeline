# AUTO-GENERATED — do not edit manually.
# Source: sciproc_base.yml
# Run update_config_artifacts() to regenerate.
from __future__ import annotations
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.config.base import ConfigNode

    class FlagNode(ConfigNode):
        astrometry: bool
        single_photometry: bool
        coadd: bool
        coadd_photometry: bool
        subtraction: bool
        difference_photometry: bool

    class InfoNode(ConfigNode):
        file: Any
        project: str
        creation_version: Any
        runtime_version: Any
        creation_datetime: Any
        last_update_datetime: Any

    class SettingsNode(ConfigNode):
        is_too: bool
        is_pipeline: bool
        is_multi_epoch: bool
        obsmode: Any
        coadd: bool
        gpu_enabled: bool
        verbose_gpu: bool

    class LoggingNode(ConfigNode):
        level: str
        file: Any
        format: str
        handlers: list

    class InputNode(ConfigNode):
        calibrated_images: Any
        processed_dir: Any
        coadd_image: Any
        difference_image: Any

    class AstrometryNode(ConfigNode):
        input_images: Any
        runtime_version: Any
        eval_match_radius: float
        path: dict
        ahead_file: str
        local_astref: Any
        scamp_timeout: int
        solvefield_timeout: int

    class PhotometryNode(ConfigNode):
        input_images: Any
        runtime_version: Any
        ref_cat: Any
        query_radius: float
        match_radius: float
        photfraction: float
        refcatname: str
        refqueryradius: float
        ref_mag_lower: int
        ref_mag_upper: float
        ref_mag_err_upper: float
        flagcut: int
        check: bool
        filters_to_check: Any
        satur_margin: float
        sex_vars: dict

    class ImcoaddNode(ConfigNode):
        input_images: Any
        runtime_version: Any
        coadd_routine: str
        coadd_mode: str
        coadd_image: Any
        gpu: bool
        device: int
        zp_key: str
        bkgsub_type: Any
        weight_map: bool
        joint_wcs: bool
        apply_bpmask: bool
        interp_type: str
        convolve: Any
        target_seeing: Any
        propagate_mask: bool

    class ImsubtractNode(ConfigNode):
        input_image: Any
        runtime_version: Any
        reference_image: Any

    class SciProcNode(ConfigNode):
        name: Any
        process_id: Any
        flag: FlagNode
        info: InfoNode
        settings: SettingsNode
        logging: LoggingNode
        input: InputNode
        astrometry: AstrometryNode
        photometry: PhotometryNode
        imcoadd: ImcoaddNode
        imsubtract: ImsubtractNode

