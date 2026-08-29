# AUTO-GENERATED — do not edit manually.
# Source: crossfilter_base.yml
# Run update_config_artifacts() to regenerate.
from __future__ import annotations
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.config.base import ConfigNode

    class FlagNode(ConfigNode):
        white_coadd: bool
        phot7ds: bool
        white_photometry: bool

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
        sanity_from_db: bool
        config_suffix: Any
        factory_scratch: Any

    class LoggingNode(ConfigNode):
        level: str
        file: Any
        format: str
        handlers: list

    class InputNode(ConfigNode):
        science_configs: Any
        expected_coadd_images: Any
        coadd_images: Any
        filters: Any
        used_filters: Any
        sanity_rejected_science_configs: Any
        missing_coadd_images: Any
        source_raw_images: Any
        discovery_method: Any
        discovery_datetime: Any
        parents_changed: bool
        minimum_filters: int
        output_dir: Any
        white_image: Any
        white_catalog: Any

    class ImcoaddNode(ConfigNode):
        input_images: Any
        runtime_version: Any
        coadd_routine: str
        coadd_mode: str
        coadd_weighting: str
        proper_coadd_weight_map_policy: str
        match_swarp_size: bool
        image_selection: bool
        image_selection_cuts: Any
        image_selection_source: str
        ppflag_bitmask: str
        coadd_image: Any
        gpu: bool
        device: int
        zp_key: str
        zpscale: bool
        bkgsub_type: str
        sex_vars: dict
        source_mask: bool
        output_weight_map: bool
        output_single_weight_map: bool
        output_footprint: bool
        output_sky_rms_map: bool
        output_bkg_map: bool
        persist_weight_maps: bool
        joint_wcs: bool
        interpolate_badpix: bool
        interp_type: str
        zero_badpix_weight: bool
        badpix_reprojection_policy: str
        streamline_reprojection: bool
        discard_interp: bool
        lean_factory: bool
        combine_lock_threshold: int
        combine_scratch: str
        convolve: bool
        target_seeing: Any

    class Phot7dsNode(ConfigNode):
        runtime_version: Any
        catalog: Any

    class PhotometryNode(ConfigNode):
        input_images: Any
        runtime_version: Any
        use_weight_map: bool
        refcatname: str
        check: bool
        satur_margin: float
        sex_vars: dict

    class CrossFilterNode(ConfigNode):
        name: Any
        process_id: Any
        sanity: Any
        flag: FlagNode
        info: InfoNode
        settings: SettingsNode
        logging: LoggingNode
        input: InputNode
        imcoadd: ImcoaddNode
        phot7ds: Phot7dsNode
        photometry: PhotometryNode
