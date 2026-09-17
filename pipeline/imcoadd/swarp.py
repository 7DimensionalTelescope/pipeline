import hashlib
import json
import os
import shutil
import threading
import time
from typing import TYPE_CHECKING, Literal

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from .. import external
from ..const import REF_DIR
from ..errors import AstrometryError, ScampError
from ..services.logger import Logger
from ..config.utils import get_key
from ..path.path import PathHandler
from ..services.utils import conservative_worker_count
from ..utils import (
    atleast_1d,
    collapse,
    force_symlink,
    get_basename,
    swap_ext,
    time_diff_in_seconds,
)
from .coadd_plan import CoaddPlan
from .header_set import InputHeaderSet
from .storage import IntermediateStorage
from .flat_weight import WEIGHT_MODEL, WEIGHT_MODEL_SKY, copy_weight_fit_header


if TYPE_CHECKING:
    from ..config._crossfilter_stubs import CrossFilterNode
    from ..config._sciproc_stubs import SciProcNode

    ConfigNodeT = SciProcNode | CrossFilterNode  # ImCoadd runs on the first, WhiteImage on the second


class SwarpMixin:
    _swarp_launch_lock = threading.Lock()
    _swarp_last_launch = 0.0

    config_node: "ConfigNodeT"
    logger: Logger
    path: PathHandler
    plan: CoaddPlan
    storage: IntermediateStorage
    input_images: list[str]
    input_headers: InputHeaderSet
    images_to_coadd: list[str] | None
    overwrite: bool | None
    center: str | None
    _use_gpu: bool
    _output_wcs_id: str | None
    _bpm_resampled_masks: list[str]
    _manifest: dict | None
    _joint_wcs_head_of: dict[str, str]
    _single_of: dict[str, str]
    _bpmid_of: dict[str, str]

    def legacy_coverage_counts(self, swarp_inputs: list[str]) -> tuple[np.ndarray, np.ndarray, int]:
        """One pass over the legacy resamples: (NGEOM, NUSED, n_resamples) on the coadd grid."""
        st = time.time()
        resampled = atleast_1d(
            self.path.imcoadd.factory.resampled_images(
                swarp_inputs,
                pass_type=self.plan.sci_pass_type,
            )
        )
        header = fits.getheader(self.config_node.imcoadd.coadd_image)
        shape = (int(header["NAXIS2"]), int(header["NAXIS1"]))
        geometric = np.zeros(shape, dtype=np.uint16)
        used = np.zeros(shape, dtype=np.uint16)
        for path in resampled:
            data = fits.getdata(path, memmap=False)
            if data.shape != shape:
                raise ValueError(f"Legacy resample shape {data.shape} differs from coadd {shape}: {path}")
            geometric += data != 0
            used += np.isfinite(data) & (data != 0)
        self.logger.info(
            f"Legacy coverage counts over {len(resampled)} resamples in {time_diff_in_seconds(st)} seconds"
        )
        return geometric, used, len(resampled)

    def apply_legacy_coverage_policy(self, swarp_inputs: list[str]) -> None:
        """Apply intersection coverage to a legacy SWarp coadd and record its coverage counts."""
        intersection = self.plan.coverage_policy != "union"
        if not (intersection or self.plan.output_counts_map or self.plan.output_footprint):
            return
        resampled = atleast_1d(
            self.path.imcoadd.factory.resampled_images(swarp_inputs, pass_type=self.plan.sci_pass_type)
        )
        missing = [path for path in resampled if not os.path.exists(path)]
        if missing:
            if intersection:
                raise self._process_error.FileNotFoundError(
                    f"Intersection coverage needs every legacy resample (e.g. {missing[:2]})"
                )
            # counts alone are not worth failing a finished coadd for
            self.logger.warning(
                f"{len(missing)} legacy resample(s) missing (e.g. {missing[:2]}); coverage counts not computed"
            )
            return
        geometric, geometric_count, n_resamples = self.legacy_coverage_counts(swarp_inputs)
        coadd_path = self.config_node.imcoadd.coadd_image
        header = fits.getheader(coadd_path)
        keep = None
        if intersection:
            coadd, header = fits.getdata(coadd_path, header=True, memmap=False)
            keep = geometric_count == n_resamples
            coadd[~keep] = np.nan
            fits.writeto(coadd_path, coadd, header=header, overwrite=True)

            weight_path = PathHandler.weight_map(coadd_path)
            if os.path.exists(weight_path):
                weight, weight_header = fits.getdata(weight_path, header=True, memmap=False)
                weight[~keep] = 0
                fits.writeto(weight_path, weight, header=weight_header, overwrite=True)
        if self.plan.output_counts_map:
            # NUSED follows the coadd, the way apply_coverage_policy zeroes the numpy backends' count
            used = geometric_count if keep is None else np.where(keep, geometric_count, 0)
            self._coadd_counts.geometric, self._coadd_counts.used = geometric, used
        if self.plan.output_footprint:
            footprint_path = PathHandler.footprint(coadd_path)
            if os.path.exists(footprint_path):
                footprint, footprint_header = fits.getdata(
                    footprint_path,
                    header=True,
                    memmap=False,
                )
                if footprint.shape != geometric_count.shape:
                    raise ValueError(
                        f"Legacy footprint shape {footprint.shape} differs from coadd {geometric_count.shape}: "
                        f"{footprint_path}"
                    )
                if keep is not None:
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
        if keep is not None:
            self.logger.info(
                f"Intersection coverage retained {int(keep.sum())}/{keep.size} pixels " f"({100 * keep.mean():.2f}%)"
            )

    def _joint_wcs_catalogs(self) -> tuple[list[str], str]:
        """SCAMP input per single: the astrometry factory catalogs when complete, else the main catalogs beside the singles."""
        choice = self.plan.joint_wcs_catalog
        prep = list(atleast_1d(self.path.astrometry.factory.catalog))
        main = list(atleast_1d(self.path.photometry.final_catalog))
        missing_prep = [c for c in prep if not os.path.exists(c)]
        if choice == "prep" or (choice == "auto" and not missing_prep):
            if missing_prep:
                raise self._process_error.FileNotFoundError(
                    f"joint_wcs_catalog 'prep': {len(missing_prep)} astrometry catalog(s) missing (e.g. {missing_prep[:2]})"
                )
            return prep, "prep"
        missing_main = [c for c in main if not os.path.exists(c)]
        if missing_main:
            raise self._process_error.FileNotFoundError(
                f"joint_wcs: {len(missing_main)} main catalog(s) missing (e.g. {missing_main[:2]})"
            )
        if choice == "auto":
            self.logger.info(
                f"joint_wcs_catalog auto: {len(missing_prep)} astrometry catalog(s) missing; using the main catalogs"
            )
        return main, "main"

    def _joint_wcs_astrefcat(self) -> str:
        """This config's astrometry reference catalog, generated around the first single when missing."""
        from ..astrometry.astrometry import Astrometry

        try:
            astrefcat = self.config_node.astrometry.local_astref or self.path.astrometry.astrefcat
            Astrometry.ensure_astrefcat(astrefcat, self.input_images[0], self.path, self.logger)
        except Exception as e:
            raise AstrometryError.JointWcsError(
                f"No astrometry reference catalog for the joint WCS (set astrometry.local_astref): {e}"
            ) from e
        self.config_node.astrometry.local_astref = astrefcat
        return astrefcat

    def joint_registration(self, swarp_inputs: list[str]) -> list[str]:
        """Joint SCAMP over the inputs' catalogs: one .head per single in the joint_wcs factory dir, evaluated and summarised."""
        from ..astrometry.astrometry import Astrometry
        from ..astrometry.evaluation_helpers import bad_wcs_cards
        from ..astrometry.utils import get_adaptive_scamp_timeout

        swarp_inputs = list(atleast_1d(swarp_inputs))
        singles = list(atleast_1d(self.input_images))
        if len(swarp_inputs) != len(singles):
            raise ValueError(f"joint_wcs: {len(swarp_inputs)} SWarp inputs for {len(singles)} singles")
        factory = self.path.imcoadd.factory
        heads = list(atleast_1d(factory.joint_wcs_heads))
        cards_files = list(atleast_1d(factory.joint_wcs_eval_cards))
        self._joint_wcs_head_of = {**dict(zip(singles, heads)), **dict(zip(swarp_inputs, heads))}
        catalogs, source = self._joint_wcs_catalogs()
        st = time.time()
        if not self.overwrite and all(os.path.exists(f) for f in heads + cards_files):
            self.logger.info(f"Joint WCS: .head and evaluation cards exist for all {len(heads)} frames; reused")
            cards = [[(c.keyword, c.value, c.comment) for c in fits.Header.fromtextfile(f).cards] for f in cards_files]
        else:
            # SCAMP writes <input>.head beside the catalog path it is given: link each catalog under the single's stem
            links = [swap_ext(head, "cat") for head in heads]
            try:
                for catalog, link in zip(catalogs, links):
                    force_symlink(catalog, link)
                n_det = sum(int(fits.getheader(catalog, 2).get("NAXIS2", 0)) for catalog in catalogs)
                timeout = get_adaptive_scamp_timeout(
                    self.config_node.astrometry.scamp_timeout, n_det, n_catalogs=len(catalogs)
                )
                astrefcat = self._joint_wcs_astrefcat()
                self.logger.info(
                    f"Start joint SCAMP over {len(catalogs)} {source} catalogs ({n_det} detections, timeout {timeout} s)"
                )
                Astrometry.scamp_catalogs(
                    links,
                    factory.joint_wcs_manifest,
                    timeout=timeout,
                    path_ref_scamp=self.path.astrometry.ref_query_dir,
                    astrefcat=astrefcat,
                    scamp_preset="main",
                    logger=self.logger,
                    overwrite=True,
                )
            except ScampError as e:
                raise AstrometryError.JointWcsError(f"Joint SCAMP over {len(catalogs)} catalogs failed: {e}") from e
            finally:
                for link in links:
                    if os.path.islink(link):
                        os.remove(link)
            headers = [self._single_wcs_header(single) for single in singles]
            refcat = Table.read(astrefcat, hdu=2)
            cards = Astrometry.evaluate_wcs_headers(
                singles,
                catalogs,
                headers,
                refcat,
                matched_catalog_paths=list(atleast_1d(factory.joint_wcs_matched)),
                match_radius=self.config_node.astrometry.eval_match_radius,
                logger=self.logger,
            )
            for frame_cards, path in zip(cards, cards_files):
                fits.Header(frame_cards).totextfile(path, overwrite=True)
        rejected = [get_basename(single) for single, c in zip(singles, cards) if bad_wcs_cards(c)]
        if rejected:
            raise AstrometryError.JointWcsError(
                f"Joint WCS rejected for {len(rejected)}/{len(singles)} frames "
                f"(UNMATCH > 0.9 or RSEP_Q2 > 2 * PIXSCALE arcsec), e.g. {rejected[:3]}"
            )
        self._summarize_joint_wcs(cards)
        self.logger.info(f"Joint WCS for {len(heads)} frames completed in {time_diff_in_seconds(st)} seconds")
        return swarp_inputs

    def _summarize_joint_wcs(self, cards_per_frame) -> None:
        """Medians over the inputs of every numeric per-frame card, under the singles' own key names."""
        values, comments = {}, {}
        for frame_cards in cards_per_frame:
            for key, value, comment in frame_cards:
                if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
                    continue
                values.setdefault(key, []).append(float(value))
                comments.setdefault(key, comment)
        for key, series in values.items():
            self.input_headers.run_cards[key] = (float(np.median(series)), ("input median: " + comments[key])[:47])
        for key in ("RSEP_RMS", "ISEP_RMS", "I_RECALL", "UNMATCH"):
            prior = [
                v for v in self.input_headers.values(key) if isinstance(v, (int, float)) and not isinstance(v, bool)
            ]
            if prior and key in values:
                self.logger.info(
                    f"Joint WCS {key}: median of the input headers {np.median(prior):.4g} -> joint {np.median(values[key]):.4g}"
                )

    def _joint_wcs_head_args(self, images: list[str], list_name: str) -> list[str]:
        """SWarp's -HEADER_NAME @list of the images' joint .head files in their order; empty without joint_wcs."""
        if not self.plan.joint_wcs:
            return []
        head_list = os.path.join(self.path.imcoadd.tmp_dir, list_name)
        with open(head_list, "w") as fp:
            fp.write("\n".join(self._joint_wcs_head_of[image] for image in images) + "\n")
        return ["-HEADER_NAME", f"@{head_list}"]

    def _single_wcs_header(self, single: str, header: fits.Header | None = None) -> fits.Header:
        """The single's header, carrying the joint WCS when one was solved for it."""
        header = fits.getheader(single) if header is None else header
        head = self._joint_wcs_head_of.get(single)
        if head is None or not os.path.exists(head):
            return header
        from ..astrometry.utils import read_scamp_header, strip_wcs

        merged = strip_wcs(header.copy())
        merged.update(read_scamp_header(head))
        return merged

    def _remove_reprojection_intermediates(self):
        """Remove unreprojected products unless their dump options are enabled."""
        dump_interp = self.plan.dump_unreprojected_interp
        dump_weight = self.plan.dump_unreprojected_weight
        if dump_interp and dump_weight:
            return
        interp_images = atleast_1d(get_key(self.config_node.imcoadd, "interp_images") or [])
        method = self._interp_method()
        freed = n_interp = n_weight = 0
        for outim in interp_images:
            sidecar = PathHandler.weight_map(outim)
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
        if self.plan.resample_weight_in_sci_pass:
            passes = (
                (
                    "sci",
                    ["-RESAMPLING_TYPE", "LANCZOS3", "-WEIGHT_IMAGE", sidecar],
                    True,
                ),
            )
        else:
            passes = [(self.plan.sci_pass_type, ["-RESAMPLING_TYPE", "LANCZOS3"], False)]
            if self.plan.compute_single_weight_maps:
                passes.append(("wht", ["-RESAMPLING_TYPE", "NEAREST", "-WEIGHT_IMAGE", sidecar], True))
        head = self._joint_wcs_head_of.get(interp_im)
        for pass_type, args, use_w in passes:
            if head is not None:
                args = args + ["-HEADER_NAME", head]
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
        sci = collapse(factory.resampled_images([interp_im], pass_type=self.plan.sci_pass_type), force=True)
        options = self._resample_options(interp_im)
        self._manifest_note(sci, **options)
        if self.plan.compute_single_weight_maps:
            wht = collapse(
                factory.resampled_weight_images([sci], pass_type=self._weight_pass_type()), force=True
            )
            if self.plan.use_smooth_weight_during_coaddition:
                copy_weight_fit_header(sidecar, wht)
            self._manifest_note(wht, **options)

    def _drop_swarp_byproduct(self, swarp_inputs, pass_type: str) -> None:
        """Remove the unused image or weight emitted by a reproject-only SWarp pass."""
        if pass_type not in ("sci", "wht") or self.plan.resample_weight_in_sci_pass:
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
        sci = collapse(factory.resampled_images([interp_im], pass_type=self.plan.sci_pass_type), force=True)
        if not os.path.exists(sci):
            return False
        if self.plan.compute_single_weight_maps:
            wht = collapse(factory.resampled_weight_images([sci], pass_type=self._weight_pass_type()), force=True)
            if not os.path.exists(wht):
                return False
        entry = self._manifest_options(sci)
        if entry is not None:
            wanted = self._resample_options(interp_im)
            if any(entry.get(key) != value for key, value in wanted.items()):
                return False
            if not self.plan.compute_single_weight_maps:
                return True
            # the weight is a product in its own right: validate its stat and identity too
            weight_entry = self._manifest_options(wht)
            return weight_entry is not None and not any(weight_entry.get(k) != v for k, v in wanted.items())
        return False  # a header can vouch for INTERP but not for the sky model that came off the frame before SWarp

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
        self._single_of = single_of
        self._record_bpmids()

        method = self._interp_method()
        zero_interp = self.plan.zero_badpix_in_single_weight_map

        from collections import deque
        from concurrent.futures import ThreadPoolExecutor

        n_tail = conservative_worker_count(len(input_images))
        self._manifest_load()  # before the tail threads: two first misses would both start from {}
        tail_pool = ThreadPoolExecutor(max_workers=n_tail)
        tail_futures = deque()
        dump_interp = self.plan.dump_unreprojected_interp
        dump_weight = self.plan.dump_unreprojected_weight
        if self.plan.background_before_reprojection:
            self.prepare_background_before_reprojection()
        self.logger.info(f"Reprojection tail on {n_tail} workers")

        def _reproject_frame(sci_out):
            sidecar = PathHandler.weight_map(sci_out)
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
            elif (
                os.path.exists(outim)
                and os.path.exists(PathHandler.weight_map(outim))
                and not self.overwrite
                and self._interp_current(outim)
            ):
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
                mask_file, badpix, bpmid = self._bpmask_info(group_in[0])
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
                    store_paths = [PathHandler.weight_map(im) for im in group_in]
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

                background_model = background = None
                if self.plan.background_before_reprojection:
                    catalogs = self._photometry_catalogs(group_in)

                    def background_model(idx, header, data, hole, src):
                        return self.fit_background_before_reprojection(
                            group_in[idx], header, data, hole=hole, sources=src, catalog=catalogs[idx]
                        )

                    def background(idx, header, data, model, exclude):
                        return self.subtract_background_before_reprojection(group_in[idx], header, data, model, exclude)

                weight_and_interpolate_cpu(
                    group_in,
                    mask_file,
                    group_out,
                    calib,
                    weight_store=weight_store,
                    flat_file=f_m_file,
                    method=method,
                    badpix=badpix,
                    zero_interp_weight=zero_interp,
                    logger=self.logger,
                    post_frame=post_frame,
                    source_catalogs=self._source_catalogs(group_in),
                    bpmid=bpmid,
                    saturated_mask=(
                        self._saturated_detector_mask if self.plan.zero_saturated_in_weight_before_reprojection else None
                    ),
                    interpolate=self.plan.interpolate_badpix,
                    ivar_out=PathHandler.ivar_map(group_in) if self.plan.dump_unsmoothed_single_weight_map else None,
                    background_model=background_model,
                    background=background,
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

        if self.plan.background_before_reprojection:
            self.config_node.imcoadd.bkg_images = self._prereprojection_models or None
            self.config_node.imcoadd.bkg_rms_images = None
        self._manifest_flush()
        self.logger.info(f"Fused weight+interp completed in {time_diff_in_seconds(st)} seconds")
        return self._record_resampled_products(interp_images)

    def coadd_with_swarp(self, input_images: list[str] | None = None, swarp_options_override: list[str] = []) -> str:
        """Legacy routine: SWarp reprojects and coadds in one run."""
        st = time.time()
        self.logger.info("Start to run swarp for coadding images")

        if input_images is None:
            input_images = self.images_to_coadd
        self.logger.debug(f"input_images: {input_images}")

        swarp_options_override = list(self.config_node.imcoadd.swarp_options_override or []) + swarp_options_override
        if not self.plan.zpscale:
            # zpscale off: stale FLXSCALE cards on the files must not flux-scale the coadd
            swarp_options_override = swarp_options_override + [
                "-FSCALE_KEYWORD",
                "NOFSCALE",
            ]
        if swarp_options_override:
            self.logger.warning(f"SWarp options override: {swarp_options_override}")

        head_args = self._joint_wcs_head_args(input_images, "joint_wcs_heads.txt")

        self.path_imagelist = os.path.join(self.path.imcoadd.tmp_dir, "images_to_coadd.txt")
        with open(self.path_imagelist, "w") as f:
            for inim in input_images:
                f.write(f"{inim}\n")

        self.logger.debug(f"Total Exptime: {self.input_headers.total_exptime}")

        sci_resampling = ["-RESAMPLING_TYPE", "LANCZOS3"]
        if not self.plan.compute_single_weight_maps:
            self._run_swarp("", swarp_args=sci_resampling + swarp_options_override + head_args, use_weight_map=False)
        else:
            self._run_swarp(
                "sci", swarp_args=sci_resampling + swarp_options_override + head_args, use_weight_map=False
            )  # Disable weight in the sci pass
            self._run_swarp("wht", swarp_args=["-RESAMPLING_TYPE", "NEAREST"] + swarp_options_override + head_args)

        factory = self.path.imcoadd.factory
        masks_predicted = atleast_1d(
            factory.resampled_weight_images(
                atleast_1d(factory.resampled_images(input_images, pass_type="bpm")),
                pass_type="bpm",
            )
        )
        bpm_pass = self.plan.badpix_propagation_policy_across_astrometric_reprojection == "conservative"
        if (
            bpm_pass
            and not self.overwrite
            and all(
                os.path.exists(m) and (self._manifest_options(m) or {}).get("joint_wcs") == self.plan.joint_wcs
                for m in masks_predicted
            )
        ):
            # checked before get_bpmask: resolving 1000 bpmasks costs ~20 min
            self.logger.info(f"bpm pass outputs already exist ({len(masks_predicted)} masks), skipping")
        elif bpm_pass:
            # bpmask_file = self.config.preprocess.bpmask_file
            per_image = atleast_1d(PathHandler.get_bpmask(input_images))
            if len(per_image) != len(input_images):
                per_image = per_image * len(input_images)
            by_mask: dict[str, list[str]] = {}
            for inim, mfile in zip(input_images, per_image):
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
                head_args = self._joint_wcs_head_args(group_frames, f"joint_wcs_heads_bpm_{k}.txt")
                self._run_swarp("bpm", swarp_args=args + swarp_options_override + head_args, input_list=group_list)
        if bpm_pass:
            for mask in masks_predicted:
                self._manifest_note(mask, joint_wcs=self.plan.joint_wcs)
            self._manifest_flush()
            self._bpm_resampled_masks = masks_predicted

        self._guard_sky_rms_propagation()
        self._update_header()
        self.logger.info(f"Running swarp is completed in {time_diff_in_seconds(st)} seconds")
        return self.config_node.imcoadd.coadd_image

    def _record_resampled_products(self, swarp_inputs: list[str]) -> list[str]:
        """Register and return the science and weight products of reprojection."""
        factory = self.path.imcoadd.factory
        resampled = atleast_1d(factory.resampled_images(swarp_inputs, pass_type=self.plan.sci_pass_type))
        self.config_node.imcoadd.resampled_images = resampled
        if self.plan.compute_single_weight_maps:
            self.config_node.imcoadd.bkgsub_weight_images = atleast_1d(
                factory.resampled_weight_images(resampled, pass_type=self._weight_pass_type())
            )
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
        badpix = self.badpix_positions(atleast_1d(resampled))
        for i, (src, dst) in enumerate(zip(sources, targets)):
            if not os.path.exists(src):
                continue
            with fits.open(src, memmap=True) as hdul:
                data, header = np.array(hdul[0].data, dtype=np.float32), hdul[0].header
            # the factory copy stays the pristine SWarp result; this product keeps the 1px holes
            if badpix is not None and badpix[i] is not None:
                badpix[i].apply(data, 0, data.shape[0], 0, data.shape[1], 0.0)
            write_weight_int16(dst, data, header)
            n += 1
        self.logger.info(f"Saved {n} resampled weight maps beside their singles")

    def _photometry_catalogs(self, images) -> list[str | None]:
        """Photometry catalogs aligned with *images*; None where a single has none."""
        singles = list(atleast_1d(self.input_images))
        try:
            catalogs = list(atleast_1d(self.path.photometry.final_catalog))
        except Exception as e:
            catalogs = []
            self.logger.warning(f"No photometry catalogs resolvable ({e})")
        by_single = dict(zip(singles, catalogs)) if len(catalogs) == len(singles) else {}
        return [by_single.get(im) for im in atleast_1d(images)]

    def _source_catalogs(self, images) -> list[str | None] | None:
        """Photometry catalogs aligned with *images*, or None when the weight is not smoothed."""
        if not self.plan.use_smooth_weight_during_coaddition:
            return None
        resolved = self._photometry_catalogs(images)
        if not all(c and os.path.exists(c) for c in resolved):
            self.logger.warning(
                "Smoothing the weight map without source masks for "
                f"{sum(1 for c in resolved if not (c and os.path.exists(c)))}/{len(resolved)} frames; "
                "bright sources will pull their own block medians"
            )
        return resolved

    def _weight_pass_type(self) -> str:
        return self.plan.weight_pass_type

    def _swarp_output_wcs_id(self) -> str:
        """Identity of the reprojection output grid: the SWarp config bytes plus this run's center."""
        if self._output_wcs_id is None:
            with open(os.path.join(REF_DIR, "7dt.swarp"), "rb") as fp:
                digest = hashlib.sha1(fp.read())
            digest.update(str(self.center).encode())
            self._output_wcs_id = digest.hexdigest()[:12]
        return self._output_wcs_id

    def _interp_current(self, interp_im: str) -> bool:
        """Whether the interpolated frame on disk came from this run's method, mask, input and weight policy."""
        wanted = self._resample_options(interp_im)
        try:
            header = fits.getheader(interp_im)
            sidecar = fits.getheader(PathHandler.weight_map(interp_im))
        except OSError:
            return False
        if str(header.get("INTERP", "") or "").upper() != wanted["interp"]:
            return False
        # interpolation happens before SWarp: the mask, the input frame and the sidecar's zero policy
        # all decide the interp products, so a reproject-only reuse must check them, not just the resamp
        for card, value in (("BPMID", wanted["bpmid"]), ("IMAGEID", wanted["imageid"])):
            if value and str(header.get(card, "") or "").strip() != str(value):
                return False
        if ("SATZERO" in sidecar) != bool(self.plan.zero_saturated_in_weight_before_reprojection):
            return False
        if sidecar.get("WGTMODEL") != wanted["weight_model"]:
            return False
        if header.get("BKGMESH") != wanted["bkgmesh"]:
            return False
        holes = sidecar.get("WGTHOLES")
        return holes is None or bool(holes) == bool(self.plan.zero_badpix_in_single_weight_map)

    def weight_model(self) -> str:
        """WGTMODEL of this run's single weights: the sky-template fit wherever a sky model comes off the frame before SWarp."""
        if not self.plan.use_smooth_weight_during_coaddition:
            return "PIXEL"
        requested = get_key(self.config_node.imcoadd, "bkgsub_type")
        if self.plan.background_before_reprojection and requested is not False and str(requested or "").lower() != "none":
            return WEIGHT_MODEL_SKY
        return WEIGHT_MODEL

    def _interp_method(self) -> str:
        """INTERP card value: imcoadd.interp_type, or 'none' when interpolate_badpix is False."""
        return str(self.config_node.imcoadd.interp_type) if self.plan.interpolate_badpix else "none"

    def _resample_options(self, interp_im: str) -> dict:
        """Everything one frame's interpolated and resampled products depend on."""
        single = self._single_of.get(interp_im, interp_im)
        return {
            "interp": self._interp_method().upper(),
            "badpix": self.plan.badpix_propagation_policy_across_astrometric_reprojection,
            "zero": bool(self.plan.zero_badpix_coadd_weight),
            "joint_wcs": bool(self.plan.joint_wcs),
            "satzero": bool(self.plan.zero_saturated_in_weight_before_reprojection),
            "satpol": self.plan.saturation_reprojection_policy,
            "weight_model": self.weight_model(),
            "bkgmesh": self._prereprojection_fingerprint(single),
            "wcsid": self._swarp_output_wcs_id(),
            "imageid": self._imageid_of.get(single),
            "bpmid": self._bpmid_of.get(single),
        }

    def _bpmask_ids(self, input_images) -> dict[str, str]:
        """BPMID per input, resolved once per IMCMB group: get_bpmask costs ~1 s a frame."""
        if not self._has_detector_bpm:
            return {}
        mapping = {}
        for group in self._group_IMCMB(list(atleast_1d(input_images))).values():
            bpmid = self._bpmask_info(group[0])[2]
            mapping.update({image: bpmid for image in group})
        return mapping

    def _record_bpmids(self) -> dict[str, str]:
        """Resolve every input's BPMID and stamp it on the header snapshot for coadd provenance."""
        if self._bpmid_of:
            return self._bpmid_of
        self._bpmid_of = self._bpmask_ids(self.input_images)
        names = set(self.input_headers.names)
        for image, bpmid in self._bpmid_of.items():
            name = get_basename(image)
            if name in names:
                self.input_headers[name]["BPMID"] = (bpmid, "IMAGEID of the bad-pixel mask")
        return self._bpmid_of

    def _propagated_bpmasks(self) -> list[str] | None:
        """Return per-frame resampled masks for conservative rejection."""
        if self.plan.badpix_propagation_policy_across_astrometric_reprojection != "conservative":
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
                    PathHandler.weight_map(output_file),
                    PathHandler.weight_map(self.config_node.imcoadd.coadd_image),
                )
            elif type == "bpm":
                # legacy: SWarp's own combine produced the summed good-pixel coverage
                shutil.move(
                    PathHandler.weight_map(output_file),
                    PathHandler.footprint(self.config_node.imcoadd.coadd_image),
                )

        return resample_dir
