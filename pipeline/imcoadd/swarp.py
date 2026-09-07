import json
import os
import shutil
import threading
import time
from typing import Literal

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from .. import external
from ..config.utils import get_key
from ..path.path import PathHandler
from ..services.utils import conservative_worker_count
from ..utils import (
    add_suffix,
    atleast_1d,
    collapse,
    get_basename,
    swap_ext,
    time_diff_in_seconds,
)
from .coadd_plan import CoaddPlan


class SwarpMixin:
    _swarp_launch_lock = threading.Lock()
    _swarp_last_launch = 0.0

    plan: CoaddPlan
    path: PathHandler
    _bpm_resampled_masks: list[str]
    _manifest: dict | None

    def apply_legacy_coverage_policy(self, swarp_inputs: list[str]) -> None:
        """Apply intersection coverage to a legacy SWarp coadd."""
        if self.plan.coverage_policy == "union":
            return
        pass_type = "sci" if self.plan.need_weights else ""
        resampled = atleast_1d(
            self.path.imcoadd.factory.resampled_images(
                swarp_inputs,
                pass_type=pass_type,
            )
        )
        coadd_path = self.config_node.imcoadd.coadd_image
        coadd, header = fits.getdata(coadd_path, header=True, memmap=False)
        geometric_count = np.zeros(coadd.shape, dtype=np.uint16)
        for path in resampled:
            data = fits.getdata(path, memmap=False)
            if data.shape != coadd.shape:
                raise ValueError(f"Legacy resample shape {data.shape} differs from coadd {coadd.shape}: {path}")
            geometric_count += np.isfinite(data) & (data != 0)
        keep = geometric_count == len(resampled)
        coadd[~keep] = np.nan
        fits.writeto(coadd_path, coadd, header=header, overwrite=True)

        weight_path = add_suffix(coadd_path, "weight")
        if os.path.exists(weight_path):
            weight, weight_header = fits.getdata(weight_path, header=True, memmap=False)
            weight[~keep] = 0
            fits.writeto(weight_path, weight, header=weight_header, overwrite=True)
        if self.plan.output_footprint:
            footprint_path = add_suffix(coadd_path, "footprint")
            if os.path.exists(footprint_path):
                footprint, footprint_header = fits.getdata(
                    footprint_path,
                    header=True,
                    memmap=False,
                )
                if footprint.shape != coadd.shape:
                    raise ValueError(
                        f"Legacy footprint shape {footprint.shape} differs from coadd {coadd.shape}: "
                        f"{footprint_path}"
                    )
                footprint[~keep] = 0
            else:
                footprint = geometric_count.astype(np.int16)
                footprint_header = header
            fits.writeto(
                footprint_path,
                footprint,
                header=footprint_header,
                overwrite=True,
            )
        self.logger.info(
            f"Intersection coverage retained {int(keep.sum())}/{keep.size} pixels " f"({100 * keep.mean():.2f}%)"
        )

    def _remove_reprojection_intermediates(self):
        """Remove unreprojected products unless their dump options are enabled."""
        dump_interp = self.plan.dump_unreprojected_interp
        dump_weight = self.plan.dump_unreprojected_weight
        if dump_interp and dump_weight:
            return
        interp_images = atleast_1d(get_key(self.config_node.imcoadd, "interp_images") or [])
        method = self.config_node.imcoadd.interp_type
        freed = n_interp = n_weight = 0
        for outim in interp_images:
            sidecar = add_suffix(outim, "weight")
            if self._lookahead_done(outim, method):
                if not dump_interp and os.path.exists(outim):
                    freed += os.path.getsize(outim)
                    os.remove(outim)
                    n_interp += 1
                if not dump_weight and os.path.exists(sidecar):
                    freed += os.path.getsize(sidecar)
                    os.remove(sidecar)
                    n_weight += 1
        if n_interp or n_weight:
            self.logger.info(
                f"Discarded {n_interp} unreprojected interp image(s) and "
                f"{n_weight} weight map(s) ({freed/1e9:.0f} GB freed)"
            )

    def _discard_consumed_bkgsub_inputs(self):
        """Delete reconstructible inputs after background subtraction."""
        if not self.plan.lean_factory:
            return
        bkgsub_images = atleast_1d(get_key(self.config_node.imcoadd, "bkgsub_images") or [])
        resampled = atleast_1d(get_key(self.config_node.imcoadd, "resampled_images") or [])
        keep_models = [self.plan.output_bkg_map, self.plan.output_sky_rms_map]
        freed = n = 0
        for resamp, bkgsub in zip(resampled, bkgsub_images):
            if not self._stage_frame_exists(bkgsub):
                continue
            # bkg/bkgrms are staged off the resamp name (stage_images suffix convention)
            # into the bkgsub dir (background.py stages them beside the bkgsub product)
            model_stem = bkgsub[: -len("_bkgsub.fits")]
            models = [f"{model_stem}_bkg.fits", f"{model_stem}_bkgrms.fits"]
            doomed = [resamp] + [m for m, keep in zip(models, keep_models) if not keep]
            for f in doomed:
                if os.path.exists(f):
                    freed += os.path.getsize(f)
                    os.remove(f)
                    n += 1
        if n:
            self.logger.info(f"Lean factory: discarded {n} consumed bkgsub inputs/models ({freed/1e9:.0f} GB freed)")

    def _reproject_single(self, interp_im: str, sidecar: str) -> None:
        """Reproject one science image and its weight map onto the fixed output WCS."""
        factory = self.path.imcoadd.factory
        base = os.path.splitext(get_basename(interp_im))[0]
        self._stagger_swarp()
        if self.plan.weight_on_sci_pass:
            passes = (
                (
                    "sci",
                    ["-RESAMPLING_TYPE", "LANCZOS3", "-WEIGHT_IMAGE", sidecar],
                    True,
                ),
            )
        else:
            sci_args = ["-RESAMPLING_TYPE", "LANCZOS3"]
            if self.plan.propagate_mask_on_sci_pass:
                sci_args += ["-WEIGHT_IMAGE", sidecar]
            passes = (
                ("sci", sci_args, self.plan.propagate_mask_on_sci_pass),
                (
                    "wht",
                    ["-RESAMPLING_TYPE", "NEAREST", "-WEIGHT_IMAGE", sidecar],
                    True,
                ),
            )
        for pass_type, args, use_w in passes:
            rdir = factory.swarp_resample_dir(pass_type)
            external.swarp(
                input=[interp_im],
                output=os.path.join(os.path.dirname(rdir), f"{base}_single_coadd.fits"),
                overwrite=self.overwrite,
                center=self.center,
                resample_dir=rdir,
                coadd=False,
                log_file=os.path.join(os.path.dirname(rdir), f"{base}_swarp.log"),
                use_weight_map=use_w,
                logger=self.logger,
                swarp_args=args,
            )
            self._drop_swarp_byproduct([interp_im], pass_type)  # as it appears, not in a storm at the end
        sci = collapse(factory.resampled_images([interp_im], pass_type="sci"), force=True)
        if self.plan.catalog_badpix_zeros:
            self._zero_badpix_in_resampled_weight(
                [interp_im],
                atleast_1d(factory.resampled_weight_images([sci], pass_type="sci")),
            )
        self._manifest_note(
            sci,
            interp=str(self.config_node.imcoadd.interp_type).upper(),
            badpix=self.plan.policy,
        )

    def _drop_swarp_byproduct(self, swarp_inputs, pass_type: str) -> None:
        """Remove the unused image or weight emitted by a reproject-only SWarp pass."""
        if pass_type not in ("sci", "wht") or self.plan.weight_on_sci_pass:
            return
        if pass_type == "sci" and self.plan.propagate_mask_on_sci_pass:
            return
        images = atleast_1d(self.path.imcoadd.factory.resampled_images(swarp_inputs, pass_type=pass_type))
        doomed = images if pass_type == "wht" else [swap_ext(f, "weight.fits") for f in images]
        for f in doomed:
            if os.path.exists(f):
                os.remove(f)

    def _manifest_load(self) -> dict:
        if self._manifest is None:
            try:
                with open(self.path.imcoadd.factory.manifest_file) as fp:
                    self._manifest = json.load(fp)
            except (OSError, ValueError):
                self._manifest = {}
        return self._manifest

    def _manifest_flush(self) -> None:
        if self._manifest is None:
            return
        manifest_file = self.path.imcoadd.factory.manifest_file
        os.makedirs(os.path.dirname(manifest_file), exist_ok=True)
        tmp = manifest_file + ".tmp"
        with open(tmp, "w") as fp:
            json.dump(self._manifest, fp)
        os.replace(tmp, manifest_file)

    def _manifest_key(self, path: str) -> str:
        return os.path.relpath(path, self.path.imcoadd.tmp_dir)

    def _manifest_note(self, path: str, **options) -> None:
        try:
            st = os.stat(path)
        except OSError:
            return
        self._manifest_load()[self._manifest_key(path)] = {
            "mtime_ns": st.st_mtime_ns, "size": st.st_size, **options
        }  # fmt: skip

    def _manifest_options(self, path: str) -> dict | None:
        """Recorded options for path, or None when absent or stale (stat mismatch)."""
        entry = self._manifest_load().get(self._manifest_key(path))
        if not entry:
            return None
        try:
            st = os.stat(path)
        except OSError:
            return None
        if st.st_mtime_ns != entry.get("mtime_ns") or st.st_size != entry.get("size"):
            return None
        return entry

    def _stagger_swarp(self, gap: float = 0.5) -> None:
        """Space out per-image SWarp launches so parallel singles don't hit the disk at once."""
        cls = type(self)
        with cls._swarp_launch_lock:
            now = time.time()
            wait = max(0.0, cls._swarp_last_launch + gap - now)
            cls._swarp_last_launch = now + wait
        if wait:
            time.sleep(wait)

    def _lookahead_done(self, interp_im: str, method: str) -> bool:
        """Return whether reusable resamples already represent this interpolation."""
        factory = self.path.imcoadd.factory
        sci = collapse(factory.resampled_images([interp_im], pass_type="sci"), force=True)
        wht = collapse(
            factory.resampled_weight_images([sci], pass_type=self._weight_pass_type()),
            force=True,
        )
        if not (os.path.exists(sci) and os.path.exists(wht)):
            return False
        entry = self._manifest_options(sci)
        if entry is not None:
            return (
                str(entry.get("interp", "")).upper() == str(method).upper() and entry.get("badpix") == self.plan.policy
            )
        if self.plan.policy != "off":
            return False  # a header can vouch for INTERP but not for the weight's badpix zeros
        try:
            ok = str(fits.getheader(sci).get("INTERP", "")).upper() == str(method).upper()
        except OSError:
            return False
        if ok:
            self._manifest_note(sci, interp=str(method).upper(), badpix=self.plan.policy)
        return ok

    def weight_and_interpolate(self, input_images: list[str] | None = None) -> list[str]:
        """Calculate weights, interpolate bad pixels, and return their resampled products."""
        if input_images is None:
            input_images = self.input_images
        self._manifest = None
        st = time.time()
        self.logger.info("Start fused weight-map calculation + bad-pixel interpolation")

        factory = self.path.imcoadd.factory
        interp_images = factory.stage_images(input_images, "interp", factory.interp_dir)
        self.config_node.imcoadd.interp_images = interp_images
        single_of = dict(zip(interp_images, input_images))

        method = self.config_node.imcoadd.interp_type
        zero_interp = self.plan.zero_before_reprojection

        from collections import deque
        from concurrent.futures import ThreadPoolExecutor

        n_tail = conservative_worker_count(len(input_images))
        self._manifest_load()  # before the tail threads: two first misses would both start from {}
        tail_pool = ThreadPoolExecutor(max_workers=n_tail)
        tail_futures = deque()
        dump_interp = self.plan.dump_unreprojected_interp
        dump_weight = self.plan.dump_unreprojected_weight
        self.logger.info(f"Reprojection tail on {n_tail} workers")

        def _reproject_frame(sci_out):
            sidecar = add_suffix(sci_out, "weight")
            self._reproject_single(sci_out, sidecar)
            if not dump_interp:
                os.remove(sci_out)
            if not dump_weight:
                os.remove(sidecar)

        def _drain_tail(keep: int = 0):
            while tail_futures and len(tail_futures) > keep:
                tail_futures.popleft().result()

        # resamp-first: a frame whose reprojected products exist needs nothing here,
        # whatever the state of its interp/weight intermediates
        todo_in, todo_out, n_lookahead, reproject_only = [], [], 0, []
        for inim, outim in zip(input_images, interp_images):
            if not self.overwrite and self._lookahead_done(outim, method):
                n_lookahead += 1
                self.logger.debug(f"Resamps exist with matching options; nothing to do for {outim}")
            elif os.path.exists(outim) and os.path.exists(add_suffix(outim, "weight")) and not self.overwrite:
                reproject_only.append(outim)
            else:
                todo_in.append(inim)
                todo_out.append(outim)
        if n_lookahead:
            self.logger.info(f"{n_lookahead} frames skipped via their reprojected products")
        if reproject_only:
            self.logger.info(f"{len(reproject_only)} frames reproject-only (interp exists, resamps missing)")
            with ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(_reproject_frame, reproject_only))

        if len(todo_in) < len(input_images):
            self.logger.info(
                f"{len(input_images) - len(todo_in)} existing fused products skipped, {len(todo_in)} to compute"
            )
        if todo_in:
            from .interpolate import weight_and_interpolate_cpu
            from .weight import _load_calibration_data

            groups = self._group_IMCMB(todo_in, todo_out)
            self.logger.info(f"{len(groups)} groups for fused weight+interpolation.")
            persist = self.plan.persist_weight_maps
            for group_id, ((z, d, f), [group_in, group_out]) in enumerate(groups.items()):
                st_group = time.time()
                mask_file, badpix = self._get_bpmask(group_in[0])
                d_m_file, f_m_file, sig_z_file, sig_f_file = PathHandler.resolve_weight_map_input_abspath([z, d, f])
                weight_store = None
                calib = None
                if persist:
                    from .weight_store import check_single_weight

                    masters = {
                        "d": d_m_file,
                        "f": f_m_file,
                        "sz": sig_z_file,
                        "sf": sig_f_file,
                    }
                    store_paths = [PathHandler.single_weight_map(im) for im in group_in]
                    weight_store = (store_paths, masters)
                    n_reusable = sum(check_single_weight(p, masters) for p in store_paths)
                    if n_reusable:
                        self.logger.info(
                            f"Group {group_id + 1}: {n_reusable}/{len(group_in)} single weight maps reusable"
                        )
                    if n_reusable == len(group_in):
                        calib = "skip"  # every frame verified: masters never touched
                if calib != "skip":
                    calib = _load_calibration_data(d_m_file, f_m_file, sig_z_file, sig_f_file)
                else:
                    calib = None

                def post_frame(sci_out, sci=None, header=None):
                    if sci is not None:
                        # saturation is judged on the single, before LANCZOS3 smooths the core edge
                        self._record_saturated_pixels(single_of[sci_out], sci, header)
                    _drain_tail(keep=2 * n_tail)
                    tail_futures.append(tail_pool.submit(_reproject_frame, sci_out))

                weight_and_interpolate_cpu(
                    group_in,
                    mask_file,
                    group_out,
                    calib,
                    weight_store=weight_store,
                    method=method,
                    badpix=badpix,
                    zero_interp_weight=zero_interp,
                    logger=self.logger,
                    post_frame=post_frame,
                    source_catalogs=self._source_catalogs(group_in),
                )
                self.logger.info(
                    f"Weight+interp completed for group {group_id + 1}/{len(groups)} in "
                    f"{time_diff_in_seconds(st_group)} seconds "
                    f"({time_diff_in_seconds(st_group, return_float=True) / len(group_in):.1f} s/image)"
                )
        else:
            self.logger.info("All fused weight+interp products already exist. Skipping")

        try:
            _drain_tail()
        finally:
            tail_pool.shutdown()

        self._manifest_flush()
        self.logger.info(f"Fused weight+interp completed in {time_diff_in_seconds(st)} seconds")
        return self._record_resampled_products(interp_images)

    def reproject_and_coadd_with_swarp(
        self,
        input_images: list[str] | None = None,
        coadd: bool = True,
        swarp_options_override: list[str] = [],
        weight_images: list[str] | None = None,
    ) -> str | list[str]:
        """Run SWarp for either coaddition or per-input reprojection."""
        st = time.time()
        action = "coadding" if coadd else "reprojecting"
        self.logger.info(f"Start to run swarp for {action} images")

        if input_images is None:
            input_images = self.images_to_coadd
        self.logger.debug(f"input_images: {input_images}")

        swarp_options_override_from_config = get_key(self.config_node.imcoadd, "swarp_options_override", default=[])
        swarp_options_override = swarp_options_override_from_config + swarp_options_override
        if not self.plan.zpscale:
            # zpscale off: stale FLXSCALE cards on the files must not flux-scale the combine
            swarp_options_override = swarp_options_override + [
                "-FSCALE_KEYWORD",
                "NOFSCALE",
            ]
        if swarp_options_override:
            self.logger.warning(f"SWarp options override: {swarp_options_override}")

        self.path_imagelist = os.path.join(self.path.imcoadd.tmp_dir, "images_to_coadd.txt")
        with open(self.path_imagelist, "w") as f:
            for inim in input_images:
                f.write(f"{inim}\n")

        self.logger.debug(f"Total Exptime: {self.input_headers.total_exptime}")

        sci_resampling = ["-RESAMPLING_TYPE", "LANCZOS3"]
        if not self.plan.need_weights:
            self._run_swarp(
                "",
                coadd=coadd,
                swarp_args=sci_resampling + swarp_options_override,
                use_weight_map=False,
            )
        elif self.plan.weight_on_sci_pass:
            # a zero-free smooth surface survives LANCZOS3 intact (measured: no interior zeros, no
            # dust), so it rides the sci pass and its companion IS the resampled weight -- one pass,
            # and the science pixels come out bit-identical to the unweighted pass
            self._run_swarp(
                "sci",
                coadd=coadd,
                swarp_args=sci_resampling + swarp_options_override,
                weight_images=weight_images,
            )
            if not coadd and self.plan.catalog_badpix_zeros:
                factory = self.path.imcoadd.factory
                resampled = atleast_1d(factory.resampled_images(input_images, pass_type="sci"))
                self._zero_badpix_in_resampled_weight(
                    self.input_images,
                    atleast_1d(factory.resampled_weight_images(resampled, pass_type="sci")),
                )
        else:
            self._run_swarp(
                "sci",
                coadd=coadd,
                swarp_args=sci_resampling + swarp_options_override,
                use_weight_map=self.plan.propagate_mask_on_sci_pass,
                weight_images=weight_images if self.plan.propagate_mask_on_sci_pass else None,
            )  # Disable weight in the sci pass
            if not coadd:
                self._drop_swarp_byproduct(input_images, "sci")
            # The wht pass is skipped on its weights alone, the same shape as the bpm guard
            # below: `external.swarp` would also demand the NEAREST-resampled science that
            # `_drop_swarp_byproduct` deletes, and re-resample every frame on every rerun.
            factory = self.path.imcoadd.factory
            wht_predicted = atleast_1d(
                factory.resampled_weight_images(
                    atleast_1d(factory.resampled_images(input_images, pass_type="sci")),
                    pass_type="wht",
                )
            )
            if not coadd and not self.overwrite and all(os.path.exists(w) for w in wht_predicted):
                self.logger.info(f"wht pass outputs already exist ({len(wht_predicted)} weights), skipping")
            else:
                self._run_swarp(
                    "wht",
                    coadd=coadd,
                    swarp_args=["-RESAMPLING_TYPE", "NEAREST"] + swarp_options_override,
                    weight_images=weight_images,
                )
                if not coadd:
                    self._drop_swarp_byproduct(input_images, "wht")

        factory = self.path.imcoadd.factory
        bpm_inputs = self.input_images if self.plan.routine == "reproject-first" else input_images
        masks_predicted = atleast_1d(
            factory.resampled_weight_images(
                atleast_1d(factory.resampled_images(bpm_inputs, pass_type="bpm")),
                pass_type="bpm",
            )
        )
        bp_policy = self.plan.policy
        bpm_pass = bp_policy == "conservative" and not self.plan.propagate_mask_on_sci_pass
        if bpm_pass and not self.overwrite and all(os.path.exists(m) for m in masks_predicted):
            # checked before get_bpmask: resolving 1000 bpmasks costs ~20 min
            self.logger.info(f"bpm pass outputs already exist ({len(masks_predicted)} masks), skipping")
        elif bpm_pass:
            # bpmask_file = self.config.preprocess.bpmask_file
            per_image = atleast_1d(PathHandler.get_bpmask(bpm_inputs))
            if len(per_image) != len(bpm_inputs):
                per_image = per_image * len(bpm_inputs)
            by_mask: dict[str, list[str]] = {}
            for inim, mfile in zip(bpm_inputs, per_image):
                by_mask.setdefault(mfile, []).append(inim)
            self.logger.info(f"bpm pass over {len(by_mask)} distinct bpmask(s)")
            for k, (bpmask_file, group_frames) in enumerate(by_mask.items()):
                bpmask_inverted = 1 - fits.getdata(bpmask_file)
                bpmask_inverted_file = self.path.imcoadd.factory.bpmask_inverted(bpmask_file)
                fits.writeto(bpmask_inverted_file, bpmask_inverted, overwrite=True)
                self.logger.debug(f"Inverted bpmask saved as {bpmask_inverted_file}")
                group_list = os.path.join(self.path.imcoadd.tmp_dir, f"images_bpm_{k}.txt")
                with open(group_list, "w") as fp:
                    fp.write("\n".join(group_frames) + "\n")
                # BADPIX=1 means bad, MAP_WEIGHT means >0 is good: SWarp needs the inverse.
                # The resampling follows the sci pass rather than being pinned here: a mask
                # resampled differently from the image it describes would not line up with it.
                args = ["-WEIGHT_IMAGE", bpmask_inverted_file] + sci_resampling
                self._run_swarp("bpm", coadd=coadd, swarp_args=args + swarp_options_override,
                                input_list=group_list)  # fmt: skip
        if bpm_pass:
            self._bpm_resampled_masks = masks_predicted

        if coadd:
            self._guard_sky_rms_propagation()
            self._update_header()
            self.logger.info(f"Running swarp is completed in {time_diff_in_seconds(st)} seconds")
            return self.config_node.imcoadd.coadd_image

        resampled = self._record_resampled_products(input_images)
        self.logger.info(f"SWarp reprojection completed in {time_diff_in_seconds(st)} seconds")
        return resampled

    def _record_resampled_products(self, swarp_inputs: list[str]) -> list[str]:
        """Register and return the science and weight products of reprojection."""
        factory = self.path.imcoadd.factory
        resampled = atleast_1d(
            factory.resampled_images(swarp_inputs, pass_type="sci" if self.plan.need_weights else "")
        )
        self.config_node.imcoadd.resampled_images = resampled
        if self.plan.need_weights:
            self.config_node.imcoadd.bkgsub_weight_images = atleast_1d(
                factory.resampled_weight_images(resampled, pass_type=self._weight_pass_type())
            )
        if self.plan.propagate_mask_on_sci_pass:
            self._bpm_resampled_masks = atleast_1d(factory.resampled_weight_images(resampled, pass_type="sci"))
        self._save_single_weight_products(resampled)
        self.images_to_coadd = resampled
        return resampled

    def _save_single_weight_products(self, resampled: list[str]) -> None:
        """Keep each frame's resampled weight beside its single."""
        if not self.plan.output_single_weight_map:
            return
        from .interpolate import write_weight_int16

        sources = atleast_1d(
            self.path.imcoadd.factory.resampled_weight_images(resampled, pass_type=self._weight_pass_type())
        )
        targets = atleast_1d(self.path.weight)
        if not (len(sources) == len(targets) == len(atleast_1d(resampled))):
            self.logger.warning("Resampled weights do not map 1:1 onto the inputs; not saved as products")
            return
        n = 0
        for src, dst in zip(sources, targets):
            if not os.path.exists(src):
                continue
            with fits.open(src, memmap=True) as hdul:
                write_weight_int16(dst, hdul[0].data, hdul[0].header)
            n += 1
        self.logger.info(f"Saved {n} resampled weight maps beside their singles")

    def _source_catalogs(self, images) -> list[str | None] | None:
        """Photometry catalogs aligned with *images*, or None when the weight is not smoothed."""
        if not self.plan.smooth_weight:
            return None
        singles = list(atleast_1d(self.input_images))
        try:
            catalogs = list(atleast_1d(self.path.photometry.final_catalog))
        except Exception as e:
            catalogs = []
            self.logger.warning(f"No photometry catalogs resolvable ({e})")
        by_single = dict(zip(singles, catalogs)) if len(catalogs) == len(singles) else {}
        resolved = [by_single.get(im) for im in atleast_1d(images)]
        if not all(c and os.path.exists(c) for c in resolved):
            self.logger.warning(
                "Smoothing the weight map without source masks for "
                f"{sum(1 for c in resolved if not (c and os.path.exists(c)))}/{len(resolved)} frames; "
                "bright sources will pull their own block medians"
            )
        return resolved

    def _weight_pass_type(self) -> str:
        return self.plan.weight_pass

    def _zero_badpix_in_resampled_weight(self, input_images, resampled_weights) -> int:
        """Set the nearest resampled weight pixel to zero for each detector bad pixel."""
        _BPX_CARD = "BPXZERO"
        st = time.time()
        detector: dict[str, tuple] = {}
        n_done = n_holes = 0
        images = list(atleast_1d(input_images))
        weights = list(atleast_1d(resampled_weights))
        if len(images) != len(weights):
            raise ValueError(f"input images ({len(images)}) and resampled weights ({len(weights)}) differ")
        for image, weight in zip(images, weights):
            if not os.path.exists(weight):
                self.logger.warning(f"No resampled weight to zero bad pixels in: {get_basename(weight)}")
                continue
            with fits.open(weight, memmap=False) as hdul:
                data, out_header = hdul[0].data, hdul[0].header.copy()
            if out_header.get(_BPX_CARD) and not self.overwrite:
                continue
            mask_file, badpix = self._get_bpmask(image)
            if mask_file not in detector:
                detector[mask_file] = np.nonzero(fits.getdata(mask_file) == badpix)
            ys, xs = detector[mask_file]
            ra, dec = WCS(fits.getheader(image)).all_pix2world(xs.astype(np.float64), ys.astype(np.float64), 0)
            x, y = WCS(out_header).all_world2pix(ra, dec, 0)
            h, w = data.shape
            finite = np.isfinite(x) & np.isfinite(y)
            xi = np.zeros(x.shape, dtype=np.int64)
            yi = np.zeros(y.shape, dtype=np.int64)
            xi[finite] = np.rint(x[finite]).astype(np.int64)
            yi[finite] = np.rint(y[finite]).astype(np.int64)
            inside = finite & (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            data[yi[inside], xi[inside]] = 0.0
            out_header[_BPX_CARD] = (True, "bad pixels zeroed from their sky positions")
            fits.writeto(weight, data, header=out_header, overwrite=True)
            n_done += 1
            n_holes += int(inside.sum())
        if n_done:
            self.logger.info(
                f"Zeroed bad pixels in {n_done} resampled weight map(s) from their sky positions "
                f"({n_holes // max(n_done, 1)} px/frame) in {time_diff_in_seconds(st)} seconds"
            )
        return n_done

    def _propagated_bpmasks(self) -> list[str] | None:
        """Return per-frame resampled masks for conservative rejection."""
        if self.plan.policy != "conservative":
            return None
        masks = atleast_1d(self._bpm_resampled_masks)
        missing = [m for m in masks if not os.path.exists(m)]
        if not masks or missing:
            raise self._process_error.FileNotFoundError(
                f"Conservative bad-pixel policy requires every bpm resample (e.g. {missing[:2]})"
            )
        return masks

    def _run_swarp(
        self,
        type: Literal["sci", "wht", "bpm"] = "",
        coadd=True,
        swarp_args=None,
        use_weight_map: bool = True,
        weight_images: list[str] | None = None,
        input_list: str | None = None,
    ) -> str:
        """Pass type='' for no weight. Returns the SWarp resample directory."""
        if weight_images:
            # @file, never a comma-joined argument: ~1000 paths overflow SWarp's option
            # buffer (SIGABRT). Order must match the input imagelist.
            weight_list = os.path.join(
                os.path.dirname(self.path_imagelist),
                f"weights_to_coadd_{type or 'all'}.txt",
            )
            with open(weight_list, "w") as fp:
                fp.write("\n".join(atleast_1d(weight_images)) + "\n")
            swarp_args = (swarp_args or []) + ["-WEIGHT_IMAGE", f"@{weight_list}"]

        factory = self.path.imcoadd.factory
        resample_dir = factory.swarp_resample_dir(type)
        working_dir = os.path.dirname(resample_dir)  # created by external.swarp's makedirs(resample_dir)

        log_file = os.path.join(working_dir, "_".join([self.config_node.name, type, "swarp.log"]))

        if type == "":
            output_file = self.config_node.imcoadd.coadd_image  # output to output_dir directly
        else:
            output_file = os.path.join(working_dir, get_basename(self.config_node.imcoadd.coadd_image))

        external.swarp(
            input=input_list or self.path_imagelist,
            output=output_file,
            overwrite=self.overwrite,
            center=self.center,
            resample_dir=resample_dir,
            coadd=coadd,
            log_file=log_file,
            logger=self.logger,
            use_weight_map=use_weight_map,
            swarp_args=swarp_args,
        )

        if coadd:
            if type == "sci":
                shutil.move(output_file, self.config_node.imcoadd.coadd_image)
            elif type == "wht":
                shutil.move(
                    add_suffix(output_file, "weight"),
                    add_suffix(self.config_node.imcoadd.coadd_image, "weight"),
                )
            elif type == "bpm":
                # legacy: SWarp's own combine produced the summed good-pixel coverage
                shutil.move(
                    add_suffix(output_file, "weight"),
                    add_suffix(self.config_node.imcoadd.coadd_image, "footprint"),
                )

        return resample_dir
