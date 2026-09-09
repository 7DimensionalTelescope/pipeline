import math
from dataclasses import dataclass

from ..config.utils import get_key


@dataclass(frozen=True, slots=True)
class CoaddPlan:
    routine: str
    mode: str
    interpolate: bool
    zero: bool
    policy: str
    weighting: str
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
    lean_factory: bool
    coadd_scratch: str | None
    persist_weight_maps: bool
    output_single_weight_map: bool

    @property
    def need_weights(self) -> bool:
        output_needs_maps = self.output_weight_map and self.mode != "proper"
        return output_needs_maps or self.weighting == "pixelwise" or self.policy in ("1px", "conservative")

    @property
    def smooth_weight(self) -> bool:
        return self.need_weights and self.weighting != "pixelwise" and self.routine != "legacy"

    @property
    def reproject(self) -> bool:
        return self.routine == "reproject-first"

    @property
    def weight_on_sci_pass(self) -> bool:
        return self.reproject and self.smooth_weight

    @property
    def catalog_badpix_zeros(self) -> bool:
        return self.weight_on_sci_pass and self.zero and self.policy == "1px"

    @property
    def need_quality_masks(self) -> bool:
        """Config-only reasons to build the per-frame bit masks; MaskMixin adds the count-plane reason."""
        return self.output_mask_map or self.dump_reprojected_masks or self.satellite_mask_enabled

    @property
    def propagate_mask_on_sci_pass(self) -> bool:
        """Conservative holes come from the sci pass: SWarp zeroes the kernel support of a zero-weight input."""
        return self.reproject and self.policy == "conservative" and self.need_weights

    @property
    def zero_before_reprojection(self) -> bool:
        return self.propagate_mask_on_sci_pass or (self.zero and not self.catalog_badpix_zeros)

    @property
    def zero_saturated_before_reprojection(self) -> bool:
        """Saturation rides the weight sidecar so SWarp spreads it over the resampling kernel, never 1 pixel.

        Only where a sidecar exists and is otherwise zero-free: a zero there would be unattributable, and a
        weight nothing reads would carry the holes nowhere."""
        return self.reproject and self.need_weights and not self.zero_before_reprojection

    @property
    def saturation_from_resampled_weight(self) -> bool:
        """NSAT is read back from SWarp's own resampled weight, so the plane carries the kernel-dilated
        footprint the estimator actually rejected instead of the 1-pixel projection it used to carry."""
        return self.zero_saturated_before_reprojection

    @property
    def sci_pass(self) -> str:
        return "sci" if self.need_weights else ""

    @property
    def weight_pass(self) -> str:
        if not self.need_weights:
            return ""
        return "sci" if self.weight_on_sci_pass else "wht"



def resolve_coadd_plan(node) -> CoaddPlan:
    routine = str(get_key(node, "coadd_routine") or "").strip().lower().replace("_", "-")
    if routine not in ("legacy", "reproject-first", "direct"):
        raise ValueError(f"Invalid coadd routine: {routine!r} (expected 'legacy', 'reproject-first', or 'direct')")

    mode = str(get_key(node, "coadd_mode") or "").strip().lower()
    if mode not in ("mean", "median", "clipped", "proper"):
        raise ValueError(f"Invalid coadd mode: {mode!r} (expected 'mean', 'median', 'clipped', or 'proper')")
    if routine == "legacy" and mode != "median":
        raise ValueError("The legacy routine uses SWarp's median coadd; set imcoadd.coadd_mode: median")

    mode_options = node.coadd_mode_options
    clip_sigma = float(mode_options["clipped"]["clip_sigma"])
    clip_ampfrac = float(mode_options["clipped"]["clip_ampfrac"])
    if not math.isfinite(clip_sigma) or clip_sigma <= 0:
        raise ValueError("imcoadd.coadd_mode_options.clipped.clip_sigma must be positive")
    if not math.isfinite(clip_ampfrac) or clip_ampfrac < 0:
        raise ValueError("imcoadd.coadd_mode_options.clipped.clip_ampfrac must be non-negative")
    clip_two_sample_fallback = str(mode_options["clipped"]["two_sample_fallback"]).strip().lower()
    if clip_two_sample_fallback not in ("mean", "min"):
        raise ValueError(
            "Invalid imcoadd.coadd_mode_options.clipped.two_sample_fallback: "
            f"{clip_two_sample_fallback!r} (expected 'mean' or 'min')"
        )

    proper_raw = mode_options["proper"]["weight_map_policy"]
    proper_weight_map_policy = str(proper_raw or "off").lower().replace("_", "-")
    proper_policies = ("off", "weighted-mean", "white-noise", "colored-noise")
    if proper_weight_map_policy not in proper_policies:
        raise ValueError(
            "Invalid imcoadd.coadd_mode_options.proper.weight_map_policy: "
            f"{proper_raw!r} (expected one of {proper_policies})"
        )

    coverage_policy = str(node.coverage_policy).strip().lower()
    if coverage_policy not in ("union", "intersection"):
        raise ValueError(
            f"Invalid imcoadd.coverage_policy: {coverage_policy!r} " "(expected 'union' or 'intersection')"
        )

    satellite_mask_enabled = bool(node.satellite_mask["enabled"])
    if satellite_mask_enabled and routine != "reproject-first":
        raise ValueError("imcoadd.satellite_mask.enabled requires coadd_routine: reproject-first")

    intermediate_policy = str(node.intermediate_policy).strip().lower()
    if intermediate_policy not in ("auto", "memory", "disk"):
        raise ValueError(
            f"Invalid imcoadd.intermediate_policy: {intermediate_policy!r} " "(expected 'auto', 'memory', or 'disk')"
        )
    memory_image_limit = int(node.memory_image_limit)
    if memory_image_limit < 1:
        raise ValueError("imcoadd.memory_image_limit must be at least 1")

    policy = str(node.badpix_reprojection_policy).lower()
    policy = {"false": "off", "none": "off", "no": "off"}.get(policy, policy)
    if policy not in ("off", "1px", "conservative"):
        raise ValueError(f"Invalid badpix_reprojection_policy: {policy!r} ('off', '1px' or 'conservative')")
    if routine == "direct" and policy == "conservative":
        raise ValueError("The direct routine has no resampling kernel; use badpix_reprojection_policy: 1px")

    joint_wcs_catalog = str(node.joint_wcs_catalog).strip().lower()
    if joint_wcs_catalog not in ("auto", "prep", "main"):
        raise ValueError(
            f"Invalid imcoadd.joint_wcs_catalog: {joint_wcs_catalog!r} (expected 'auto', 'prep' or 'main')"
        )

    weighting = str(node.coadd_weighting).lower()
    weighting = weighting.replace("-", "").replace("_", "")
    weighting = {"false": "off", "none": "off", "no": "off"}.get(weighting, weighting)
    if weighting not in ("off", "global", "pixelwise"):
        raise ValueError(f"Invalid imcoadd.coadd_weighting: {weighting!r} (False, 'global' or 'pixel-wise')")
    if mode == "clipped" and weighting == "off":
        raise ValueError("coadd_mode 'clipped' requires coadd_weighting 'global' or 'pixel-wise'")
    if mode == "proper" and weighting == "pixelwise":
        raise ValueError("coadd_mode 'proper' is incompatible with pixel-wise weighting")

    plan = CoaddPlan(
        routine=routine,
        mode=mode,
        interpolate=bool(node.interpolate_badpix),
        zero=bool(node.zero_badpix_weight),
        policy=policy,
        weighting=weighting,
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
        lean_factory=bool(node.lean_factory),
        coadd_scratch=node.coadd_scratch,
        persist_weight_maps=bool(node.persist_weight_maps),
        output_single_weight_map=bool(node.output_single_weight_map),
    )
    if plan.reproject and plan.zero_before_reprojection:
        raise NotImplementedError(
            "saturation is propagated conservatively by zeroing the pre-reprojection weight sidecar, and this "
            f"combination (badpix_reprojection_policy: {plan.policy!r}, coadd_weighting: {plan.weighting!r}) "
            "already puts the bad-pixel zeros there, so a zero weight would not say which of the two it is; "
            "use badpix_reprojection_policy: '1px' with coadd_weighting: 'global'"
        )
    if plan.routine == "legacy" and not (plan.interpolate and plan.zero and plan.policy == "1px"):
        raise ValueError(
            "coadd_routine 'legacy' offers only its sci/wht double SWarp pass, the nearest equivalent of '1px'; "
            "set interpolate_badpix: True, zero_badpix_weight: True and badpix_reprojection_policy: '1px', "
            "or use coadd_routine: reproject-first"
        )
    if plan.dump_reprojected_masks and not plan.reproject:
        raise ValueError("imcoadd.dump_reprojected_masks requires coadd_routine: reproject-first")
    if plan.reproject and not plan.interpolate:
        raise ValueError(
            "coadd_routine 'reproject-first' interpolates bad pixels in every frame; "
            "set interpolate_badpix: True (or coadd_routine: direct for pre-aligned inputs)"
        )
    if plan.joint_wcs and plan.routine == "direct":
        raise ValueError("imcoadd.joint_wcs registers frames for reprojection; the direct routine has none")
    if plan.routine == "direct" and plan.convolve:
        raise ValueError("The direct routine does not convolve; set imcoadd.convolve: False")
    if plan.policy == "1px" and not plan.zero:
        raise NotImplementedError(
            "badpix_reprojection_policy '1px' requires zero_badpix_weight: True; "
            "use 'off' or 'conservative' when weights stay nonzero"
        )
    if plan.zero and not plan.need_weights:
        raise ValueError(
            "zero_badpix_weight has no weight map to modify; enable output_weight_map, "
            "pixel-wise weighting, or badpix_reprojection_policy: 1px"
        )
    if plan.zero and plan.policy == "off" and plan.weighting != "pixelwise":
        raise ValueError(
            "zero_badpix_weight with policy 'off' leaves zero-variance pixels in the science vote; "
            "use policy '1px', pixel-wise weighting, or keep the weights nonzero"
        )
    return plan
