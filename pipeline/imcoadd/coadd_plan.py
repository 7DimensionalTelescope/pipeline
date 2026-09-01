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

    @property
    def need_weights(self) -> bool:
        output_needs_maps = self.output_weight_map and self.mode != "proper"
        return (
            output_needs_maps or self.weighting == "pixelwise" or self.policy == "1px"
        )

    @property
    def smooth_weight(self) -> bool:
        return (
            self.need_weights
            and self.weighting != "pixelwise"
            and self.routine != "legacy"
        )

    @property
    def weight_on_sci_pass(self) -> bool:
        return self.routine == "reproject-first" and self.smooth_weight

    @property
    def catalog_badpix_zeros(self) -> bool:
        return self.weight_on_sci_pass and self.zero

    @property
    def weight_pass(self) -> str:
        if not self.need_weights:
            return ""
        return "sci" if self.weight_on_sci_pass else "wht"


def resolve_coadd_plan(node) -> CoaddPlan:
    def opt(new, old, default):
        value = get_key(node, new)
        return get_key(node, old, default=default) if value is None else value

    routine = (
        str(get_key(node, "coadd_routine") or "").strip().lower().replace("_", "-")
    )
    if routine not in ("legacy", "reproject-first", "direct"):
        raise ValueError(
            f"Invalid coadd routine: {routine!r} (expected 'legacy', 'reproject-first', or 'direct')"
        )

    mode = str(get_key(node, "coadd_mode") or "").strip().lower()
    if mode not in ("mean", "median", "clipped", "proper"):
        raise ValueError(
            f"Invalid coadd mode: {mode!r} (expected 'mean', 'median', 'clipped', or 'proper')"
        )
    if routine == "legacy" and mode != "median":
        raise ValueError(
            "The legacy routine uses SWarp's median coadd; set imcoadd.coadd_mode: median"
        )

    policy = str(
        opt("badpix_reprojection_policy", "bpmask_policy", "1px") or "off"
    ).lower()
    policy = {"false": "off", "none": "off", "no": "off"}.get(policy, policy)
    if policy not in ("off", "1px", "conservative"):
        raise ValueError(
            f"Invalid badpix_reprojection_policy: {policy!r} ('off', '1px' or 'conservative')"
        )
    if routine == "direct" and policy == "conservative":
        raise ValueError(
            "The direct routine has no resampling kernel; use badpix_reprojection_policy: 1px"
        )

    weighting = str(get_key(node, "coadd_weighting", default="global") or "off").lower()
    weighting = weighting.replace("-", "").replace("_", "")
    weighting = {"false": "off", "none": "off", "no": "off"}.get(weighting, weighting)
    if weighting not in ("off", "global", "pixelwise"):
        raise ValueError(
            f"Invalid imcoadd.coadd_weighting: {weighting!r} (False, 'global' or 'pixel-wise')"
        )
    if mode == "clipped" and weighting == "off":
        raise ValueError(
            "coadd_mode 'clipped' requires coadd_weighting 'global' or 'pixel-wise'"
        )
    if mode == "proper" and weighting == "pixelwise":
        raise ValueError(
            "coadd_mode 'proper' is incompatible with pixel-wise weighting"
        )

    plan = CoaddPlan(
        routine=routine,
        mode=mode,
        interpolate=bool(opt("interpolate_badpix", "apply_bpmask", False)),
        zero=bool(opt("zero_badpix_weight", "zero_interp_weight", True)),
        policy=policy,
        weighting=weighting,
        output_weight_map=bool(opt("output_weight_map", "weight_map", True)),
        output_footprint=bool(get_key(node, "output_footprint", default=True)),
        combine_lock_threshold=int(get_key(node, "combine_lock_threshold", default=50)),
    )
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
