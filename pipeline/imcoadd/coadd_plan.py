import builtins
import math
from dataclasses import dataclass

from ..config.utils import get_key


@dataclass(frozen=True, slots=True)
class CoaddPlan:
    """Config options in, one attribute per decision point of the imcoadd code out.

    Fields carry their config key's name (values normalised); properties are the derived decisions. Where a
    config key does not map one-to-one onto a decision, this is the structure:

        coadd_routine ──────────────┬─► reproject_with_swarp ('reproject-first'; 'direct' is the same routine
                                    │        with the reprojection skipped)  ─┬─► resample_weight_in_sci_pass
                                    ├─► inputs_are_reprojected (not legacy)    │     = and use_smooth_weight_during_coaddition
                                    └─► reject_saturated_pixels (not legacy)   │
        coadd_weighting ────────────┬──────────────────────────────────────────┼─► use_smooth_weight_during_coaddition
        output_weight_map ──────────┼─► compute_single_weight_maps ───────────┘     = weights computed, not legacy
        coadd_mode ─────────────────┤     = output map (not proper), pixel-wise,     └─► output_smooth_weight_map_for_coadd_image
        saturation_reprojection_ ───┤       or the saturation sidecar below                 = and output_weight_map and not proper
          policy                    │     └─► sidecar_only_for_saturation (the sidecar exists for nothing else: logged)
                                    └─► zero_saturated_in_weight_before_reprojection
                                             = reproject_with_swarp and 'conservative' (needs resample_weight_in_sci_pass)
                                             ├─► nsat_from_resampled_weight
                                             ├─► read_resampled_weight_as_exclusion_mask
                                             └─► exclude_saturated_by_projected_index = reject_saturated_pixels and not the sidecar
        zero_badpix_coadd_weight ───┬─► badpix_propagation_policy_across_astrometric_reprojection
        badpix_reprojection_policy ─┘        = the configured footprint ('1px' | 'conservative'), 'off' unless zeroing
                                             ├─► exclude_badpix_by_projected_index = not legacy and not 'off'
                                             │        (sparse in-memory index; 'conservative' dilates it to the LANCZOS3 support)
                                             └─► zero_badpix_in_single_weight_map = legacy (its NEAREST wht pass carries the zeros)
        sci_pass_type / weight_pass_type: the SWarp pass whose products the science / weight resamples are read from.
    """

    coadd_routine: str
    coadd_mode: str
    interpolate_badpix: bool
    zero_badpix_coadd_weight: bool
    badpix_propagation_policy_across_astrometric_reprojection: str
    saturation_reprojection_policy: str
    coadd_weighting: str
    output_weight_map: bool
    output_footprint: bool
    combine_lock_threshold: int
    coverage_policy: str
    clip_sigma: float
    clip_ampfrac: float
    clip_two_sample_fallback: str
    proper_weight_map_policy: str
    output_mask_map: bool
    output_counts_map: bool
    output_egain_map: bool
    fill_nan: bool
    dump_reprojected_masks: bool
    satellite_mask_enabled: bool
    intermediate_policy: str
    memory_image_limit: int
    dump_bkgsub: bool
    output_bkg_map: bool
    output_sky_rms_map: bool
    zpscale: bool
    joint_wcs: bool
    joint_wcs_catalog: str
    convolve: bool | str | None
    source_mask: bool | str
    interp_type: str
    match_swarp_size: bool
    dump_unreprojected_interp: bool
    dump_unreprojected_weight: bool
    dump_unsmoothed_single_weight_map: bool
    lean_factory: bool
    coadd_scratch: str | None
    persist_weight_maps: bool
    output_single_weight_map: bool
    background_box_size: int
    background_filter_size: int
    background_exclude_percentile: float
    background_min_usable: float
    background_max_dropped_boxes: float
    dequantize_background_below: float

    # ---------------------------------------------------------------- routine
    @property
    def reproject_with_swarp(self) -> bool:
        return self.coadd_routine == "reproject-first"

    @property
    def background_before_reprojection(self) -> bool:
        """The sky model comes off each detector-grid frame in the fused loop; the resample stage only masks and measures."""
        return self.reproject_with_swarp

    @property
    def inputs_are_reprojected(self) -> bool:
        """The frames bkgsub and the check plots handle sit on the sky grid; only legacy works on detector frames."""
        return self.coadd_routine != "legacy"

    @property
    def reject_saturated_pixels(self) -> bool:
        """Legacy coadds in SWarp and has no estimator to drop a saturated sample from."""
        return self.coadd_routine != "legacy"

    # ---------------------------------------------------------------- weight maps
    @property
    def compute_single_weight_maps(self) -> bool:
        output_needs_maps = self.output_weight_map and self.coadd_mode != "proper"
        return output_needs_maps or self.coadd_weighting == "pixelwise" or self.zero_saturated_in_weight_before_reprojection

    @property
    def sidecar_only_for_saturation(self) -> bool:
        """The weight sidecar is generated for nothing but the saturation footprint; worth a log line."""
        return self.zero_saturated_in_weight_before_reprojection and not (
            (self.output_weight_map and self.coadd_mode != "proper") or self.coadd_weighting == "pixelwise"
        )

    @property
    def use_smooth_weight_during_coaddition(self) -> bool:
        """The per-frame weight is the fitted vignetting surface, not the per-pixel noise model."""
        return (
            self.compute_single_weight_maps and self.coadd_routine != "legacy"
        )

    @property
    def output_smooth_weight_map_for_coadd_image(self) -> bool:
        """The coadd weight product is the propagated smooth surface (proper writes its own weight)."""
        return self.output_weight_map and self.use_smooth_weight_during_coaddition and self.coadd_mode != "proper"

    @property
    def resample_weight_in_sci_pass(self) -> bool:
        """The smooth sidecar rides the LANCZOS3 sci pass as -WEIGHT_IMAGE; no NEAREST wht pass runs."""
        return self.reproject_with_swarp and self.use_smooth_weight_during_coaddition

    @property
    def sci_pass_type(self) -> str:
        return "sci" if self.compute_single_weight_maps else ""

    @property
    def weight_pass_type(self) -> str:
        if not self.compute_single_weight_maps:
            return ""
        return "sci" if self.resample_weight_in_sci_pass else "wht"

    # ---------------------------------------------------------------- bad pixels
    @property
    def exclude_badpix_by_projected_index(self) -> bool:
        """Bad pixels excluded through the sparse in-memory index ('1px' nearest pixel, 'conservative' kernel support)."""
        return self.coadd_routine != "legacy" and self.badpix_propagation_policy_across_astrometric_reprojection != "off"

    @property
    def zero_badpix_in_single_weight_map(self) -> bool:
        """Legacy alone carries the zeros in the weight file: SWarp's NEAREST wht pass is its only channel."""
        return self.coadd_routine == "legacy" and self.zero_badpix_coadd_weight

    # ---------------------------------------------------------------- saturation
    @property
    def zero_saturated_in_weight_before_reprojection(self) -> bool:
        """Saturation rides the weight sidecar through the LANCZOS3 sci pass, so SWarp zeroes the kernel support."""
        return self.reproject_with_swarp and self.saturation_reprojection_policy == "conservative"

    @property
    def nsat_from_resampled_weight(self) -> bool:
        """NSAT is read back from SWarp's own resampled weight, the kernel-dilated footprint the estimator rejected."""
        return self.zero_saturated_in_weight_before_reprojection

    @property
    def read_resampled_weight_as_exclusion_mask(self) -> bool:
        """The in-memory backends read each frame's resampled weight as its validity mask (its zeros are saturation)."""
        return self.zero_saturated_in_weight_before_reprojection

    @property
    def exclude_saturated_by_projected_index(self) -> bool:
        return self.reject_saturated_pixels and not self.zero_saturated_in_weight_before_reprojection

    # ---------------------------------------------------------------- quality masks
    @property
    def build_per_frame_quality_masks(self) -> bool:
        """Config-only reasons to build the per-frame bit masks; MaskMixin adds the count-plane reason."""
        return self.output_mask_map or self.dump_reprojected_masks or self.satellite_mask_enabled


def resolve_coadd_plan(node, errors=builtins) -> CoaddPlan:
    """`errors` is the stage's composite error family (CoaddError), so a refused plan carries its error code."""
    routine = str(get_key(node, "coadd_routine") or "").strip().lower().replace("_", "-")
    if routine not in ("legacy", "reproject-first", "direct"):
        raise errors.ValueError(
            f"Invalid coadd routine: {routine!r} (expected 'legacy', 'reproject-first', or 'direct')"
        )

    mode = str(get_key(node, "coadd_mode") or "").strip().lower()
    if mode not in ("mean", "median", "clipped", "proper"):
        raise errors.ValueError(f"Invalid coadd mode: {mode!r} (expected 'mean', 'median', 'clipped', or 'proper')")
    if routine == "legacy" and mode != "median":
        raise errors.ValueError("The legacy routine uses SWarp's median coadd; set imcoadd.coadd_mode: median")

    mode_options = node.coadd_mode_options
    clip_sigma = float(mode_options["clipped"]["clip_sigma"])
    clip_ampfrac = float(mode_options["clipped"]["clip_ampfrac"])
    if not math.isfinite(clip_sigma) or clip_sigma <= 0:
        raise errors.ValueError("imcoadd.coadd_mode_options.clipped.clip_sigma must be positive")
    if not math.isfinite(clip_ampfrac) or clip_ampfrac < 0:
        raise errors.ValueError("imcoadd.coadd_mode_options.clipped.clip_ampfrac must be non-negative")
    clip_two_sample_fallback = str(mode_options["clipped"]["two_sample_fallback"]).strip().lower()
    if clip_two_sample_fallback not in ("mean", "min"):
        raise errors.ValueError(
            "Invalid imcoadd.coadd_mode_options.clipped.two_sample_fallback: "
            f"{clip_two_sample_fallback!r} (expected 'mean' or 'min')"
        )

    proper_raw = mode_options["proper"]["weight_map_policy"]
    proper_weight_map_policy = str(proper_raw or "off").lower().replace("_", "-")
    proper_policies = ("off", "weighted-mean", "white-noise", "colored-noise")
    if proper_weight_map_policy not in proper_policies:
        raise errors.ValueError(
            "Invalid imcoadd.coadd_mode_options.proper.weight_map_policy: "
            f"{proper_raw!r} (expected one of {proper_policies})"
        )

    coverage_policy = str(node.coverage_policy).strip().lower()
    if coverage_policy not in ("union", "intersection"):
        raise errors.ValueError(
            f"Invalid imcoadd.coverage_policy: {coverage_policy!r} " "(expected 'union' or 'intersection')"
        )

    satellite_mask_enabled = bool(node.satellite_mask["enabled"])
    if satellite_mask_enabled and routine != "reproject-first":
        raise errors.ValueError("imcoadd.satellite_mask.enabled requires coadd_routine: reproject-first")

    background = node.background
    background_box_size = int(background["box_size"])
    background_filter_size = int(background["filter_size"])
    background_exclude_percentile = float(background["exclude_percentile"])
    background_min_usable = float(background["min_usable"])
    background_max_dropped_boxes = float(background["max_dropped_boxes"])
    dequantize_background_below = float(background["dequantize_background_below"])  # SKYVAL gate shared with Photometry's measure_sky
    if background_box_size < 8:
        raise errors.ValueError("imcoadd.background.box_size must be at least 8 pixels")
    if background_filter_size < 1 or background_filter_size % 2 == 0:
        raise errors.ValueError("imcoadd.background.filter_size must be a positive odd number of mesh nodes")
    if not 0 <= background_exclude_percentile <= 100:
        raise errors.ValueError("imcoadd.background.exclude_percentile must be between 0 and 100")
    if not 0 <= background_min_usable <= 100:
        raise errors.ValueError("imcoadd.background.min_usable must be between 0 and 100")
    if not 0 <= background_max_dropped_boxes <= 100:
        raise errors.ValueError("imcoadd.background.max_dropped_boxes must be between 0 and 100")

    intermediate_policy = str(node.intermediate_policy).strip().lower()
    if intermediate_policy not in ("auto", "memory", "disk"):
        raise errors.ValueError(
            f"Invalid imcoadd.intermediate_policy: {intermediate_policy!r} " "(expected 'auto', 'memory', or 'disk')"
        )
    memory_image_limit = int(node.memory_image_limit)
    if memory_image_limit < 1:
        raise errors.ValueError("imcoadd.memory_image_limit must be at least 1")

    zero_badpix_coadd_weight = bool(node.zero_badpix_coadd_weight)
    policy = str(node.badpix_reprojection_policy).lower()
    policy = {"false": "off", "none": "off", "no": "off"}.get(policy, policy)
    if policy not in ("off", "1px", "conservative"):
        raise errors.ValueError(f"Invalid badpix_reprojection_policy: {policy!r} ('off', '1px' or 'conservative')")
    if zero_badpix_coadd_weight and policy == "off":
        raise errors.ValueError(
            "zero_badpix_coadd_weight needs an exclusion footprint: badpix_reprojection_policy '1px' or 'conservative'"
        )
    # the plan carries the EFFECTIVE footprint: 'off' whenever bad pixels keep their weight in the coadd
    policy = policy if zero_badpix_coadd_weight else "off"
    if routine == "direct" and policy == "conservative":
        raise errors.ValueError("The direct routine has no resampling kernel; use badpix_reprojection_policy: 1px")

    saturation_policy = str(node.saturation_reprojection_policy).lower()
    if saturation_policy not in ("1px", "conservative"):
        raise errors.ValueError(
            f"Invalid saturation_reprojection_policy: {saturation_policy!r} ('1px' or 'conservative')"
        )
    if routine == "direct" and saturation_policy == "conservative":
        raise errors.ValueError("The direct routine has no resampling kernel; use saturation_reprojection_policy: 1px")

    joint_wcs_catalog = str(node.joint_wcs_catalog).strip().lower()
    if joint_wcs_catalog not in ("auto", "prep", "main"):
        raise errors.ValueError(
            f"Invalid imcoadd.joint_wcs_catalog: {joint_wcs_catalog!r} (expected 'auto', 'prep' or 'main')"
        )

    weighting = str(node.coadd_weighting).lower()
    weighting = weighting.replace("-", "").replace("_", "")
    weighting = {"false": "off", "none": "off", "no": "off"}.get(weighting, weighting)
    if weighting not in ("off", "global", "pixelwise"):
        raise errors.ValueError(f"Invalid imcoadd.coadd_weighting: {weighting!r} (False, 'global' or 'pixel-wise')")
    if mode == "clipped" and weighting == "off":
        raise errors.ValueError("coadd_mode 'clipped' requires coadd_weighting 'global' or 'pixel-wise'")
    if mode == "proper" and weighting == "pixelwise":
        raise errors.ValueError("coadd_mode 'proper' is incompatible with pixel-wise weighting")

    plan = CoaddPlan(
        coadd_routine=routine,
        coadd_mode=mode,
        interpolate_badpix=bool(node.interpolate_badpix),
        zero_badpix_coadd_weight=zero_badpix_coadd_weight,
        badpix_propagation_policy_across_astrometric_reprojection=policy,
        saturation_reprojection_policy=saturation_policy,
        coadd_weighting=weighting,
        output_weight_map=bool(node.output_weight_map),
        output_footprint=bool(node.output_footprint),
        combine_lock_threshold=int(node.combine_lock_threshold),
        coverage_policy=coverage_policy,
        clip_sigma=clip_sigma,
        clip_ampfrac=clip_ampfrac,
        clip_two_sample_fallback=clip_two_sample_fallback,
        proper_weight_map_policy=proper_weight_map_policy,
        output_mask_map=bool(node.output_mask_map),
        output_counts_map=bool(node.output_counts_map),
        output_egain_map=bool(node.output_egain_map),
        fill_nan=bool(node.fill_nan),
        dump_reprojected_masks=bool(node.dump_reprojected_masks),
        satellite_mask_enabled=satellite_mask_enabled,
        intermediate_policy=intermediate_policy,
        memory_image_limit=memory_image_limit,
        dump_bkgsub=bool(node.dump_bkgsub),
        output_bkg_map=bool(node.output_bkg_map),
        output_sky_rms_map=bool(node.output_sky_rms_map),
        zpscale=bool(node.zpscale),
        joint_wcs=bool(node.joint_wcs),
        joint_wcs_catalog=joint_wcs_catalog,
        convolve=node.convolve,
        source_mask=node.source_mask,
        interp_type=str(node.interp_type),
        match_swarp_size=bool(node.match_swarp_size),
        dump_unreprojected_interp=bool(node.dump_unreprojected_interp),
        dump_unreprojected_weight=bool(node.dump_unreprojected_weight),
        dump_unsmoothed_single_weight_map=bool(node.dump_unsmoothed_single_weight_map),
        lean_factory=bool(node.lean_factory),
        coadd_scratch=node.coadd_scratch,
        persist_weight_maps=bool(node.persist_weight_maps),
        output_single_weight_map=bool(node.output_single_weight_map),
        background_box_size=background_box_size,
        background_filter_size=background_filter_size,
        background_exclude_percentile=background_exclude_percentile,
        background_min_usable=background_min_usable,
        background_max_dropped_boxes=background_max_dropped_boxes,
        dequantize_background_below=dequantize_background_below,
    )
    if plan.zero_saturated_in_weight_before_reprojection and not plan.resample_weight_in_sci_pass:
        raise errors.NotImplementedError(
            "saturation_reprojection_policy 'conservative' propagates the saturation zeros through the LANCZOS3 sci "
            "pass and requires a smooth weight sidecar; "
            "use a modern coadd routine or saturation_reprojection_policy: '1px'"
        )
    if routine == "legacy" and not (plan.interpolate_badpix and zero_badpix_coadd_weight and policy == "1px"):
        raise errors.ValueError(
            "coadd_routine 'legacy' offers only its sci/wht double SWarp pass, the nearest equivalent of '1px'; "
            "set interpolate_badpix: True, zero_badpix_coadd_weight: True and badpix_reprojection_policy: '1px', "
            "or use coadd_routine: reproject-first"
        )
    if plan.dump_reprojected_masks and not plan.reproject_with_swarp:
        raise errors.ValueError("imcoadd.dump_reprojected_masks requires coadd_routine: reproject-first")
    if plan.reproject_with_swarp and not plan.interpolate_badpix and policy == "1px":
        raise errors.ValueError(
            "interpolate_badpix: False leaves the bad pixel's value in the frame, and LANCZOS3 spreads it over its "
            "kernel support, which a '1px' exclusion does not cover; set interpolate_badpix: True, "
            "badpix_reprojection_policy: conservative, or zero_badpix_coadd_weight: False (bad pixels vote)"
        )
    if plan.reproject_with_swarp and not plan.interpolate_badpix and policy == "conservative" and plan.convolve:
        raise errors.ValueError(
            "interpolate_badpix: False with badpix_reprojection_policy: conservative excludes the LANCZOS3 support only, "
            "and the seeing-match convolution spreads the bad pixel's value beyond it; set interpolate_badpix: True "
            "or convolve: False"
        )
    if plan.joint_wcs and routine == "direct":
        raise errors.ValueError("imcoadd.joint_wcs registers frames for reprojection; the direct routine has none")
    if routine == "direct" and plan.convolve:
        raise errors.ValueError("The direct routine does not convolve; set imcoadd.convolve: False")
    if plan.output_single_weight_map and not plan.compute_single_weight_maps:
        raise errors.ValueError(
            "imcoadd.output_single_weight_map has no weight map to save; enable output_weight_map, pixel-wise "
            "weighting, or saturation_reprojection_policy: conservative"
        )
    if plan.output_egain_map and mode not in ("mean", "clipped"):
        raise errors.ValueError(
            f"imcoadd.output_egain_map is not supported for coadd_mode {mode!r}; use 'mean' or 'clipped', "
            "or set output_egain_map: False"
        )
    return plan
