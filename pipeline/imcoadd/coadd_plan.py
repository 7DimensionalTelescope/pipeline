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
    proper_weight_map_policy: str
    output_mask_map: bool
    dump_reprojected_masks: bool
    satellite_mask_enabled: bool
    intermediate_policy: str
    memory_image_limit: int
    dump_bkgsub: bool

    @property
    def need_weights(self) -> bool:
        output_needs_maps = self.output_weight_map and self.mode != "proper"
        return output_needs_maps or self.weighting == "pixelwise" or self.policy == "1px"

    @property
    def smooth_weight(self) -> bool:
        return self.need_weights and self.weighting != "pixelwise" and self.routine != "legacy"

    @property
    def weight_on_sci_pass(self) -> bool:
        return self.routine == "reproject-first" and self.smooth_weight

    @property
    def catalog_badpix_zeros(self) -> bool:
        return self.weight_on_sci_pass and self.zero and self.policy == "1px"

    @property
    def propagate_mask_on_sci_pass(self) -> bool:
        """Conservative holes come from the sci pass: SWarp zeroes the kernel support of a zero-weight input."""
        return self.routine == "reproject-first" and self.policy == "conservative" and self.need_weights

    @property
    def zero_before_reprojection(self) -> bool:
        return self.propagate_mask_on_sci_pass or (self.zero and not self.catalog_badpix_zeros)

    @property
    def weight_pass(self) -> str:
        if not self.need_weights:
            return ""
        return "sci" if self.weight_on_sci_pass else "wht"


def resolve_coadd_plan(node) -> CoaddPlan:
    def opt(new, old, default):
        value = get_key(node, new)
        return get_key(node, old, default=default) if value is None else value

    routine = str(get_key(node, "coadd_routine") or "").strip().lower().replace("_", "-")
    if routine not in ("legacy", "reproject-first", "direct"):
        raise ValueError(f"Invalid coadd routine: {routine!r} (expected 'legacy', 'reproject-first', or 'direct')")

    mode = str(get_key(node, "coadd_mode") or "").strip().lower()
    if mode not in ("mean", "median", "clipped", "proper"):
        raise ValueError(f"Invalid coadd mode: {mode!r} (expected 'mean', 'median', 'clipped', or 'proper')")
    if routine == "legacy" and mode != "median":
        raise ValueError("The legacy routine uses SWarp's median coadd; set imcoadd.coadd_mode: median")

    options = get_key(node, "coadd_options", default={}) or {}
    if not isinstance(options, dict):
        raise ValueError("imcoadd.coadd_options must be a mapping")

    def mode_options(name):
        value = options.get(name, {}) or {}
        if not isinstance(value, dict):
            raise ValueError(f"imcoadd.coadd_options.{name} must be a mapping")
        return value

    clipped_options = mode_options("clipped")
    proper_options = mode_options("proper")
    clip_sigma = float(clipped_options.get("clip_sigma", 5.0))
    clip_ampfrac = float(clipped_options.get("clip_ampfrac", 0.3))
    if not math.isfinite(clip_sigma) or clip_sigma <= 0:
        raise ValueError("imcoadd.coadd_options.clipped.clip_sigma must be positive")
    if not math.isfinite(clip_ampfrac) or clip_ampfrac < 0:
        raise ValueError("imcoadd.coadd_options.clipped.clip_ampfrac must be non-negative")

    proper_raw = proper_options.get(
        "weight_map_policy",
        get_key(node, "proper_coadd_weight_map_policy", default="white-noise"),
    )
    proper_weight_map_policy = str(proper_raw or "off").lower().replace("_", "-")
    proper_policies = ("off", "weighted-mean", "white-noise", "colored-noise")
    if proper_weight_map_policy not in proper_policies:
        raise ValueError(
            "Invalid imcoadd.coadd_options.proper.weight_map_policy: "
            f"{proper_raw!r} (expected one of {proper_policies})"
        )

    coverage_policy = str(get_key(node, "coverage_policy", default="union") or "union").strip().lower()
    if coverage_policy not in ("union", "intersection"):
        raise ValueError(
            f"Invalid imcoadd.coverage_policy: {coverage_policy!r} " "(expected 'union' or 'intersection')"
        )

    satellite_options = get_key(node, "satellite_mask", default={}) or {}
    if not isinstance(satellite_options, dict):
        raise ValueError("imcoadd.satellite_mask must be a mapping")
    satellite_mask_enabled = bool(satellite_options.get("enabled", False))
    if satellite_mask_enabled and routine != "reproject-first":
        raise ValueError("imcoadd.satellite_mask.enabled requires coadd_routine: reproject-first")

    intermediate_policy = str(get_key(node, "intermediate_policy", default="auto") or "auto").strip().lower()
    if intermediate_policy not in ("auto", "memory", "disk"):
        raise ValueError(
            f"Invalid imcoadd.intermediate_policy: {intermediate_policy!r} " "(expected 'auto', 'memory', or 'disk')"
        )
    memory_image_limit = int(get_key(node, "memory_image_limit", default=6))
    if memory_image_limit < 1:
        raise ValueError("imcoadd.memory_image_limit must be at least 1")

    policy = str(opt("badpix_reprojection_policy", "bpmask_policy", "off") or "off").lower()
    policy = {"false": "off", "none": "off", "no": "off"}.get(policy, policy)
    if policy not in ("off", "1px", "conservative"):
        raise ValueError(f"Invalid badpix_reprojection_policy: {policy!r} ('off', '1px' or 'conservative')")
    if routine == "direct" and policy == "conservative":
        raise ValueError("The direct routine has no resampling kernel; use badpix_reprojection_policy: 1px")

    weighting = str(get_key(node, "coadd_weighting", default="global") or "off").lower()
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
        interpolate=bool(opt("interpolate_badpix", "apply_bpmask", True)),
        zero=bool(opt("zero_badpix_weight", "zero_interp_weight", False)),
        policy=policy,
        weighting=weighting,
        output_weight_map=bool(opt("output_weight_map", "weight_map", True)),
        output_footprint=bool(get_key(node, "output_footprint", default=True)),
        combine_lock_threshold=int(get_key(node, "combine_lock_threshold", default=20)),
        coverage_policy=coverage_policy,
        clip_sigma=clip_sigma,
        clip_ampfrac=clip_ampfrac,
        proper_weight_map_policy=proper_weight_map_policy,
        output_mask_map=bool(get_key(node, "output_mask_map", default=True)),
        dump_reprojected_masks=bool(get_key(node, "dump_reprojected_masks", default=False)),
        satellite_mask_enabled=satellite_mask_enabled,
        intermediate_policy=intermediate_policy,
        memory_image_limit=memory_image_limit,
        dump_bkgsub=bool(get_key(node, "dump_bkgsub", default=False)),
    )
    if plan.dump_reprojected_masks and plan.routine != "reproject-first":
        raise ValueError("imcoadd.dump_reprojected_masks requires coadd_routine: reproject-first")
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
