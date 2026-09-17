# AUTO-GENERATED — do not edit manually.
# Source: crossfilter_base.yml composed on sciproc_base.yml (crossfilter_template)
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
        factory_scratch_cap_gb: int

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
        coadd_mode_options: dict
        coverage_policy: str
        match_swarp_size: bool
        image_selection: bool
        image_selection_cuts: Any
        image_selection_source: str
        ppflag_bitmask: str
        coadd_image: Any
        coadd_mask_image: Any
        coadd_counts_image: Any
        gpu: bool
        device: int
        zp_key: str
        zpscale: bool
        bkgsub_type: str
        background: dict
        source_mask: bool
        dump_source_masks: bool
        output_weight_map: bool
        output_single_weight_map: bool
        output_footprint: bool
        output_mask_map: bool
        output_counts_map: bool
        fill_nan: bool
        dump_reprojected_masks: bool
        satellite_mask: dict
        output_sky_rms_map: bool
        output_bkg_map: bool
        persist_weight_maps: bool
        joint_wcs: bool
        joint_wcs_catalog: str
        interpolate_badpix: bool
        interp_type: str
        zero_badpix_coadd_weight: bool
        badpix_reprojection_policy: str
        saturation_reprojection_policy: str
        dump_unreprojected_interp: bool
        dump_unreprojected_weight: bool
        dump_unsmoothed_single_weight_map: bool
        intermediate_policy: str
        memory_image_limit: int
        dump_bkgsub: bool
        lean_factory: bool
        combine_lock_threshold: int
        coadd_scratch: str
        swarp_options_override: Any
        convolve: bool
        target_seeing: Any

    class Phot7dsNode(ConfigNode):
        runtime_version: Any
        catalog: Any

    class PhotometryNode(ConfigNode):
        input_images: Any
        runtime_version: Any
        use_weight_map: bool
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
        satellite_mask: dict

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
