import os
import shutil
import time

import numpy as np

from ..config.utils import get_key
from ..const import REF_DIR
from ..utils import add_suffix, atleast_1d, collapse, get_basename, time_diff_in_seconds
from .calc import clipped_mean_coadd_numpy, mean_coadd_numpy, median_coadd_numpy
from .coadd_plan import CoaddPlan, resolve_coadd_plan


class InMemoryCoaddMixin:
    _LOCAL_FSTYPES = {
        "ext2",
        "ext3",
        "ext4",
        "xfs",
        "btrfs",
        "zfs",
        "f2fs",
        "reiserfs",
    }

    @staticmethod
    def _fstype_of(path: str) -> tuple[str, str]:
        """(mount point, fstype) of the filesystem holding *path*, longest prefix wins."""
        best = ("", "")
        real = os.path.realpath(path)
        try:
            with open("/proc/mounts") as fp:
                entries = [ln.split()[:3] for ln in fp if len(ln.split()) >= 3]
        except OSError:
            return best
        for _dev, mnt, fstype in entries:
            if (real == mnt or real.startswith(mnt.rstrip("/") + "/")) and len(
                mnt
            ) > len(best[0]):
                best = (mnt, fstype)
        return best

    def _pick_combine_scratch(self, files) -> str | None:
        """Choose local scratch for a large network-backed combine."""
        n_frames = len({f for _g, f in files})
        if n_frames < int(self.plan.combine_lock_threshold):
            return None
        src_mnt, src_fstype = self._fstype_of(os.path.dirname(files[0][1]))
        if src_fstype in self._LOCAL_FSTYPES:
            self.logger.debug(
                f"combine_scratch auto: inputs already local on {src_mnt} ({src_fstype})"
            )
            return None

        need = sum(os.path.getsize(f) for _g, f in files if os.path.exists(f)) * 1.1
        best = None
        seen_dev = set()
        try:
            with open("/proc/mounts") as fp:
                entries = [ln.split()[:3] for ln in fp if len(ln.split()) >= 3]
        except OSError:
            return None
        for _dev, mnt, fstype in entries:
            # never the system disk, never a home directory: /home is often the same
            # physical disk as a scratch mount anyway, and filling either is an outage
            if (
                fstype not in self._LOCAL_FSTYPES
                or mnt == "/"
                or mnt.startswith(("/home", "/root", "/boot"))
            ):
                continue
            try:
                dev = os.stat(
                    mnt
                ).st_dev  # one entry per physical filesystem, not per mount
                if dev in seen_dev:
                    continue
                seen_dev.add(dev)
                root = os.path.join(mnt, "pipeline_coadd_scratch")
                os.makedirs(root, exist_ok=True)
                free = shutil.disk_usage(mnt).free
            except OSError as e:
                self.logger.debug(
                    f"combine_scratch auto: {mnt} ({fstype}) unusable -- {e}"
                )
                continue
            if free > need and (best is None or free > best[1]):
                best = (root, free)
        if best is None:
            self.logger.warning(
                f"combine_scratch auto: inputs are on {src_fstype} ({src_mnt}) and no local "
                f"filesystem has the {need/1e9:.0f} GB needed; combining over the network"
            )
            return None
        self.logger.info(
            f"combine_scratch auto: inputs on {src_fstype} ({src_mnt}); staging "
            f"{need/1e9:.0f} GB to {best[0]} ({best[1]/1e12:.1f} TB free)"
        )
        return best[0]

    def _stage_for_combine(self, groups: dict[str, list[str] | None]):
        """Stage combine inputs locally and return remapped groups plus cleanup."""
        if getattr(self, "_intermediate_policy", "disk") == "memory":
            return groups, lambda: None
        scratch = get_key(self.config_node.imcoadd, "combine_scratch")
        files = [(g, f) for g, lst in groups.items() if lst for f in lst]
        if not files:
            return groups, lambda: None
        if str(scratch).lower() == "auto":
            scratch = self._pick_combine_scratch(files)
        if not scratch:
            return groups, lambda: None

        need = sum(os.path.getsize(f) for _, f in files)
        free = shutil.disk_usage(scratch).free
        if need * 1.05 > free:
            self.logger.warning(
                f"combine_scratch {scratch}: need {need/1e9:.0f} GB, only {free/1e9:.0f} GB free; combining from NFS"
            )
            return groups, lambda: None

        stem = os.path.splitext(get_basename(self.config_node.info.file))[0]
        base = os.path.join(scratch, "imcoadd_staged", stem)
        st = time.time()
        self.logger.info(f"Staging {len(files)} files ({need/1e9:.0f} GB) to {base}")
        from concurrent.futures import ThreadPoolExecutor

        def _copy(item):
            group, src = item
            dst = os.path.join(base, group, os.path.basename(src))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
            return src, dst

        with ThreadPoolExecutor(max_workers=2) as pool:
            mapping = dict(pool.map(_copy, files))
        self.logger.info(
            f"Staged in {time_diff_in_seconds(st)} seconds "
            f"({need/1e9/max(time.time()-st, 1):.2f} GB/s sequential from NFS)"
        )
        remapped = {
            g: ([mapping[f] for f in lst] if lst else lst) for g, lst in groups.items()
        }
        return remapped, lambda: shutil.rmtree(base, ignore_errors=True)

    def coadd_in_memory(
        self,
        input_images: list[str] | None = None,
        device_id=None,
        weight_images: list[str] | None = None,
    ) -> str:
        """Dispatch the selected in-memory coadd backend."""
        if input_images is None:
            input_images = self.images_to_coadd
        self._guard_sky_rms_propagation()

        if device_id is not None:
            self.coadd_with_cupy(input_images, device_id=device_id)
            return self.config_node.imcoadd.coadd_image

        plan = self.plan
        weighting = plan.weighting
        policy = plan.policy
        smoothed = "smoothed" if plan.smooth_weight else "per-pixel"
        self.logger.info(
            f"Coadd weighting: {weighting}; badpix policy: {policy}; weight maps: {smoothed}; "
            f"coverage: {plan.coverage_policy}"
        )
        if plan.smooth_weight and not (
            plan.interpolate or plan.zero or policy == "conservative"
        ):
            # the fitted surface has no bad pixels, and nothing else is marking them either
            self.logger.warning(
                "Smoothed weight maps with no bad-pixel channel: set interpolate_badpix, "
                "zero_badpix_weight, or badpix_reprojection_policy: conservative; bad pixels vote"
            )
        if weighting == "pixelwise" and policy == "off" and self.plan.zero:
            self.logger.info(
                "pixel-wise weighting with zeroed bad-pixel weights: bad pixels cannot vote "
                "regardless of policy 'off' (a zero-weight vote is no vote); set "
                "zero_badpix_weight: False to let interpolated pixels vote"
            )

        wht_maps = None
        if self.plan.need_weights:
            # NEAREST-resampled weights live next to the wht pass output; the
            # LANCZOS3 companions next to the sci resamp ring to ~0 almost
            # everywhere (99%+ zeros) and must NOT be used.
            # (that was the raw-weight era: under smooth_weight the sci-pass companion is a
            # smooth zero-free surface and _weight_pass_type() selects it on purpose)
            wht_dir = self.path.imcoadd.factory.swarp_resample_dir(
                self._weight_pass_type()
            )
            if weight_images is not None:
                candidates = atleast_1d(weight_images)
            else:
                # named after what SWarp resampled, not after the later bkgsub products
                resampled = (
                    get_key(self.config_node.imcoadd, "resampled_images")
                    or input_images
                )
                candidates = atleast_1d(
                    self.path.imcoadd.factory.resampled_weight_images(
                        resampled, pass_type=self._weight_pass_type()
                    )
                )
            if all(os.path.exists(w) for w in candidates):
                wht_maps = candidates
            else:
                missing = [w for w in candidates if not os.path.exists(w)][:3]
                raise self._process_error.FileNotFoundError(
                    f"Required resampled weight maps not found in {wht_dir} (e.g. {missing})"
                )

        weights = None
        if weighting == "pixelwise":
            weights = wht_maps
        elif weighting == "global":
            skysigs = self.input_headers.values("SKYSIG")
            missing = [i for i, s in enumerate(skysigs) if not s]
            if missing:
                # 1.0 ADU^-2 against a typical 0.0086 is ~100x a normal frame: that frame would
                # own the coadd. The card comes from single photometry, so this is its failure.
                names = [get_basename(f) for f in atleast_1d(input_images)]
                raise self._process_error.PreviousStageError(
                    f"SKYSIG missing on {len(missing)}/{len(skysigs)} frames "
                    f"(e.g. {[names[i] for i in missing[:3]]}); rerun single photometry"
                )
            weights = [1.0 / float(s) ** 2 for s in skysigs]

        if policy == "conservative":
            masks = self._propagated_bpmasks()
        elif policy == "1px" and weighting != "pixelwise":
            masks = wht_maps
        else:
            masks = None

        if plan.mode == "proper":
            return self.coadd_proper_with_numpy(input_images, holes=masks)

        var_maps = wht_maps if weighting != "pixelwise" else None
        stage_wht = weights if weighting == "pixelwise" else None
        var_is_mask = (
            var_maps is not None and masks is not None and list(var_maps) == list(masks)
        )
        if getattr(self, "_intermediate_policy", "disk") == "memory":
            cached_paths = set(wht_maps or [])
            cached_paths.update(masks or [])
            for path in cached_paths:
                self._read_stage_frame(path)
        staged, cleanup = self._stage_for_combine(
            {
                "sci": input_images,
                "wht": stage_wht,
                "bpm": masks,
                "var": None if var_is_mask else var_maps,
            }
        )
        input_images, masks = staged["sci"], staged["bpm"]
        if weighting == "pixelwise":
            weights = staged["wht"]
        var_maps = masks if var_is_mask else staged["var"]

        mode = plan.mode
        match_swarp_size = bool(
            get_key(self.config_node.imcoadd, "match_swarp_size", default=True)
        )
        # Serialize high-demand combines per filesystem and lease their planned memory.
        from ..services.combine_lock import CombineSlot, NullSlot

        anchor = os.path.dirname(collapse(atleast_1d(input_images)[0], force=True))
        slot_ctx = (
            CombineSlot(anchor, logger=self.logger)
            if len(atleast_1d(input_images)) >= plan.combine_lock_threshold
            else NullSlot()
        )
        try:
            with slot_ctx as slot:
                if mode == "mean":
                    slot.lease(
                        4 * 110_000_000 * 8
                    )  # sum/norm/count/gain accumulators, ~3.5 GB
                    self.coadd_with_numpy(
                        input_images,
                        weights=weights,
                        masks=masks,
                        var_maps=var_maps,
                        match_swarp_size=match_swarp_size,
                        write_weight=plan.output_weight_map,
                        write_footprint=plan.output_footprint,
                    )
                elif mode == "clipped":
                    slot.lease(6 * 110_000_000 * 8)  # two-pass accumulators, ~5 GB
                    self.coadd_clipped_with_numpy(
                        input_images,
                        weights=weights,
                        masks=masks,
                        var_maps=var_maps,
                        match_swarp_size=match_swarp_size,
                        write_weight=plan.output_weight_map,
                        write_footprint=plan.output_footprint,
                        outlier_callback=(
                            self._coadd_mask_builder.mark_outliers
                            if getattr(self, "_coadd_mask_builder", None) is not None
                            else None
                        ),
                    )
                elif mode == "median":
                    from ..services.combine_lock import memory_headroom_bytes
                    from .calc import plan_median_memory
                    from .utils import _parse_swarp_image_size

                    reserved = slot.reserved_bytes
                    grid_w, grid_h = _parse_swarp_image_size(
                        os.path.join(REF_DIR, "7dt.swarp")
                    )
                    budget = int(0.3 * memory_headroom_bytes(reserved))
                    _, planned = plan_median_memory(
                        len(atleast_1d(input_images)), grid_w, grid_h, budget
                    )
                    slot.lease(planned)
                    self.coadd_median_with_numpy(
                        input_images,
                        weights=weights,
                        masks=masks,
                        match_swarp_size=match_swarp_size,
                        reserved_bytes=reserved,
                        var_maps=var_maps,
                        write_weight=plan.output_weight_map,
                        write_footprint=plan.output_footprint,
                    )
                else:
                    raise ValueError(
                        f"Invalid coadd mode: {mode!r} (expected 'mean', 'median' or 'clipped')"
                    )
        finally:
            cleanup()
        return self.config_node.imcoadd.coadd_image

    def coadd_proper_with_numpy(
        self, input_images: list[str], holes: list[str] | None = None
    ) -> str:
        """Run proper coaddition with its mode-specific options."""
        from ..services.combine_lock import CombineSlot, NullSlot
        from .proper import proper_coadd_numpy

        plan = self.plan
        policy = self._proper_weight_policy()
        coadd_image = self.config_node.imcoadd.coadd_image
        anchor = os.path.dirname(collapse(atleast_1d(input_images)[0], force=True))
        slot_ctx = (
            CombineSlot(anchor, logger=self.logger)
            if len(atleast_1d(input_images)) >= plan.combine_lock_threshold
            else NullSlot()
        )
        with slot_ctx as slot:
            slot.lease(
                5 * 110_000_000 * 8
            )  # numerator + share accumulators + final FFT pair, ~4.4 GB
            return proper_coadd_numpy(
                input_images,
                output_path=coadd_image,
                coadd_header=self.input_headers.coadd_header,
                peeings=self._proper_peeings(input_images),
                skysigs=self.input_headers.values("SKYSIG"),
                flxscales=self._combine_flxscales(),
                weight_map_policy=policy,
                weight_output=(
                    add_suffix(coadd_image, "weight") if policy != "off" else False
                ),
                footprint_output=(
                    add_suffix(coadd_image, "footprint")
                    if plan.output_footprint
                    else False
                ),
                psf_output=add_suffix(coadd_image, "psf"),
                holes=holes,
                match_swarp_size=bool(
                    get_key(self.config_node.imcoadd, "match_swarp_size", default=True)
                ),
                coverage_policy=plan.coverage_policy,
                logger=self.logger,
            )

    def _proper_weight_policy(self) -> str:
        """Validated proper-coadd weight-map policy."""
        return self.plan.proper_weight_map_policy

    def _proper_peeings(self, input_images: list[str]) -> list[float]:
        """Per-frame PSF FWHM in pixels; the homogenized target when convolution ran."""
        n = len(atleast_1d(input_images))
        if self.config_node.imcoadd.convolve and getattr(self, "_max_peeing", None):
            return [float(self._max_peeing)] * n
        peeings = self.input_headers.values("PEEING")
        if len(peeings) != n or any(p is None for p in peeings):
            missing = [
                name for name, p in zip(self.input_headers.names, peeings) if p is None
            ]
            self.logger.error(
                f"No PEEING for {missing[:3]}; proper coadd needs a per-frame PSF",
                self._process_error.KeyError,
            )
            raise self._process_error.KeyError(
                f"No PEEING for {len(missing)} input(s); proper coadd needs a per-frame PSF"
            )
        return [float(p) for p in peeings]

    def _validate_proper_mode(self):
        """Fail fast on option combinations the Fourier-domain combine cannot honor."""
        if self.plan.routine not in ("reproject-first", "direct"):
            raise self._process_error.ValueError(
                "coadd_mode 'proper' requires coadd_routine 'reproject-first' or 'direct'"
            )
        plan = self.plan
        if not plan.interpolate and self._proper_requires_interpolation:
            raise self._process_error.ValueError(
                "coadd_mode 'proper' requires interpolate_badpix: True (a Fourier-domain vote cannot skip pixels)"
            )
        self._proper_weight_policy()
        if plan.weighting == "off":
            self.logger.info("coadd_weighting has no effect under 'proper': frames are inverse-variance weighted by construction")  # fmt: skip

    def _coadd_plan(self) -> CoaddPlan:
        return resolve_coadd_plan(self.config_node.imcoadd)

    def _combine_flxscales(self):
        """Return snapshot flux scales, or False when scaling is disabled."""
        if get_key(self.config_node.imcoadd, "zpscale", default=True):
            return self.input_headers.values("FLXSCALE")
        return False

    def coadd_with_numpy(
        self,
        input_images: list[str],
        weights: list[str] | None = None,
        masks: list[str] | None = None,
        match_swarp_size: bool = True,
        var_maps: list[str] | None = None,
        write_weight: bool = True,
        write_footprint: bool = True,
    ) -> str:
        return mean_coadd_numpy(
            input_images,
            output_path=self.config_node.imcoadd.coadd_image,
            coadd_header=self.input_headers.coadd_header,
            weights=weights,
            weight_output=(
                add_suffix(self.config_node.imcoadd.coadd_image, "weight")
                if write_weight
                else False
            ),
            footprint_output=(
                add_suffix(self.config_node.imcoadd.coadd_image, "footprint")
                if write_footprint
                else False
            ),
            masks=masks,
            flxscales=self._combine_flxscales(),
            match_swarp_size=match_swarp_size,
            var_maps=var_maps,
            coverage_policy=self.plan.coverage_policy,
            frame_cache=getattr(self, "_frame_cache", None),
            logger=self.logger,
        )

    def coadd_clipped_with_numpy(
        self,
        input_images: list[str],
        weights: list[str] | None = None,
        masks: list[str] | None = None,
        match_swarp_size: bool = True,
        var_maps: list[str] | None = None,
        write_weight: bool = True,
        write_footprint: bool = True,
        outlier_callback=None,
    ) -> str:
        return clipped_mean_coadd_numpy(
            input_images,
            output_path=self.config_node.imcoadd.coadd_image,
            coadd_header=self.input_headers.coadd_header,
            weights=weights,
            weight_output=(
                add_suffix(self.config_node.imcoadd.coadd_image, "weight")
                if write_weight
                else False
            ),
            footprint_output=(
                add_suffix(self.config_node.imcoadd.coadd_image, "footprint")
                if write_footprint
                else False
            ),
            masks=masks,
            flxscales=self._combine_flxscales(),
            match_swarp_size=match_swarp_size,
            clip_sigma=self.plan.clip_sigma,
            clip_ampfrac=self.plan.clip_ampfrac,
            var_maps=var_maps,
            coverage_policy=self.plan.coverage_policy,
            outlier_callback=outlier_callback,
            frame_cache=getattr(self, "_frame_cache", None),
            logger=self.logger,
        )

    def coadd_median_with_numpy(
        self,
        input_images: list[str],
        weights: list[str] | None = None,
        masks: list[str] | None = None,
        match_swarp_size: bool = True,
        chunk_h: (
            int | None
        ) = None,  # None: auto-sized from idle memory (see calc._auto_chunk_h)
        reserved_bytes: int = 0,
        var_maps: list[str] | None = None,
        write_weight: bool = True,
        write_footprint: bool = True,
    ) -> str:
        return median_coadd_numpy(
            input_images,
            output_path=self.config_node.imcoadd.coadd_image,
            coadd_header=self.input_headers.coadd_header,
            weights=weights,
            weight_output=(
                add_suffix(self.config_node.imcoadd.coadd_image, "weight")
                if write_weight
                else False
            ),
            footprint_output=(
                add_suffix(self.config_node.imcoadd.coadd_image, "footprint")
                if write_footprint
                else False
            ),
            masks=masks,
            flxscales=self._combine_flxscales(),
            match_swarp_size=match_swarp_size,
            chunk_h=chunk_h,
            reserved_bytes=reserved_bytes,
            var_maps=var_maps,
            coverage_policy=self.plan.coverage_policy,
            frame_cache=getattr(self, "_frame_cache", None),
            logger=self.logger,
        )

    def coadd_with_cupy(self, input_images: list[str], device_id) -> str:
        raise NotImplementedError("GPU coadd_with_cupy is not implemented yet")
