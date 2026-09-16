import os
import time
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
from astropy.io import fits

from ..config.utils import get_key
from ..path.path import PathHandler
from ..services.logger import Logger
from ..services.utils import conservative_worker_count
from ..utils import add_suffix, atleast_1d, get_basename, time_diff_in_seconds
from ..utils.header import update_padded_header
from .const import MaskBit
from .coadd_plan import CoaddPlan
from .plotting import plot_background, plot_source_mask
from .header_set import InputHeaderSet
from .storage import IntermediateStorage


if TYPE_CHECKING:
    from ..config._crossfilter_stubs import CrossFilterNode
    from ..config._sciproc_stubs import SciProcNode

    ConfigNodeT = SciProcNode | CrossFilterNode  # ImCoadd runs on the first, WhiteImage on the second


def _key_tally(chosen: list[tuple]) -> str:
    """How many frames took each header key, for the debug line."""
    counts = Counter(key or "no card" for _, key in chosen)
    return ", ".join(f"{key} x{n}" for key, n in counts.items())


class BackgroundMixin:
    config_node: "ConfigNodeT"
    logger: Logger
    path: PathHandler
    plan: CoaddPlan
    storage: IntermediateStorage
    input_images: list[str]
    input_headers: InputHeaderSet
    images_to_coadd: list[str] | None
    overwrite: bool | None
    path_bkgsub: str
    _fov_masks: list[str | None] | None
    _quality_masks: list[np.ndarray | str] | None

    def _background_output_exists(self, image):
        return self._stage_frame_exists(image)

    def _write_background_output(self, image, data, header):
        self._store_stage_frame(image, data, header)

    def bkgsub(
        self,
        input_images: list[str] | None = None,
        ignore_steppy_flag: bool = False,
        skyval_cut: float = 40,
        mask_out_of_fov: bool = False,
        mask_sources: bool | str = False,
        fov_masks: list | None = None,
    ) -> list[str]:
        """Subtract the configured scalar or mesh background from each image."""
        if input_images is None:
            input_images = self.input_images
        st = time.time()

        factory = self.path.imcoadd.factory
        self.path_bkgsub = self.storage.bkgsub_dir

        bkgsub_images = factory.stage_images(input_images, "bkgsub", self.path_bkgsub)
        self.config_node.imcoadd.bkgsub_images = bkgsub_images
        if self.storage.policy == "memory":
            durable = factory.stage_images(input_images, "bkgsub", factory.bkgsub_dir)
            self.storage.bkgsub_dump_pairs = list(zip(bkgsub_images, durable))

        bkg_images = factory.stage_images(input_images, "bkg", self.path_bkgsub)
        bkg_rms_images = factory.stage_images(input_images, "bkgrms", self.path_bkgsub)

        # Convolution destroys SWarp's exactly-zero padding, so a prebuilt mask wins.
        if fov_masks is not None:
            fov_mask_images = list(fov_masks)
        elif mask_out_of_fov:
            fov_mask_images = factory.stage_images(input_images, "fovmask", self.path_bkgsub)
        else:
            fov_mask_images = [None] * len(input_images)

        # the off-source pair when photometry measured it, the SExtractor pair otherwise
        sky_levels = self.input_headers.values_any_with_key("BACKVAL", "SKYVAL")
        sky_sigmas = self.input_headers.values_any_with_key("BACKSIG", "SKYSIG")  # the mask law's threshold
        skyvalues = [value for value, _ in sky_levels]
        skysigmas = [value for value, _ in sky_sigmas]
        self.logger.debug(f"Sky level from {_key_tally(sky_levels)}; sky noise from {_key_tally(sky_sigmas)}")
        methods = self.bkgsub_methods()
        requested = get_key(self.config_node.imcoadd, "bkgsub_type")
        if requested is False:
            requested = "none"  # YAML `false` spells the same switch as 'none'; empty stays auto
        elif requested:
            requested = str(requested).lower()
        else:
            requested = self._default_bkgsub_type(skyvalues, skyval_cut)
            self.logger.debug(f"bkgsub_type unset; filled in as {requested!r} for the group")
        if requested != "individual" and requested not in methods:
            raise ValueError(
                f"bkgsub_type: {requested!r} is invalid (expected 'individual' or one of {sorted(methods)})"
            )
        types = [self._resolve_bkgsub_type(requested, sv, skyval_cut) for sv in skyvalues]
        self.config_node.imcoadd.bkgsub_type = requested
        if requested == "none":
            mask_sources = False  # no mesh to protect from sources, and no fallback may reinstate one
            self.logger.info("bkgsub_type 'none': staging the frames unchanged, no sky model subtracted")

        # The header snapshot aggregates a mixed group to BACKTYPE=MIXED.
        for hdr, btype in zip(self.input_headers, types):
            hdr["BACKTYPE"] = (btype.upper(), "Background subtraction type")

        any_dynamic = "dynamic" in types
        source_mask_images = (
            factory.stage_images(input_images, "srcmask", self.storage.source_mask_dir)
            if mask_sources
            else [None] * len(input_images)
        )
        if mask_sources and not any_dynamic:
            self.logger.info("Constant background: source_mask is used for residual QA only")

        # the source mask comes from the singles' own catalogs, so the inputs must be their derivatives 1:1
        singles = atleast_1d(self.input_images)
        try:
            catalogs = atleast_1d(self.path.photometry.final_catalog)
        except Exception as e:
            self.logger.warning(f"No photometry catalogs resolvable ({e}); no frame will get a source mask")
            catalogs = []
        if not (len(singles) == len(catalogs) == len(input_images)):
            singles = catalogs = [None] * len(input_images)

        counts = {name: types.count(name) for name in sorted(set(types))}
        self.logger.info(f"Start background subtraction (bkgsub_type={requested!r}): {counts}")
        if any_dynamic:
            self.config_node.imcoadd.bkg_images = bkg_images if self.plan.output_bkg_map else None
            self.config_node.imcoadd.bkg_rms_images = bkg_rms_images if self.plan.output_sky_rms_map else None
        else:
            if get_key(self.config_node.imcoadd, "bkg_images"):
                self.config_node.imcoadd.bkg_images = None
            if get_key(self.config_node.imcoadd, "bkg_rms_images"):
                self.config_node.imcoadd.bkg_rms_images = None

        def _bkgsub_one(
            i,
            inim,
            outim,
            bkg,
            bkg_rms,
            skyvalue,
            skysigma,
            fov_mask,
            src_mask,
            btype,
            single,
            phot_cat,
        ):
            st_loop = time.time()
            cached = self.storage.frame_cache.pop(inim, None)
            if cached is None:
                data, header = fits.getdata(inim, header=True, memmap=False)
                data = np.ascontiguousarray(data, dtype=np.float32)
            else:
                data, header = cached

            quality_mask = None
            if self._quality_masks is not None:
                quality_mask = self.quality_mask(i)

            if fov_mask is None:
                fov_valid = None
            elif fov_masks is not None and os.path.exists(fov_mask):
                fov_valid = fits.getdata(fov_mask, memmap=False).astype(bool)
            else:
                fov_valid = self._fov_valid(data, get_basename(inim))
            exclude = None
            qa_mask = None
            if src_mask is not None:
                sources, valid, usable = self._source_mask(
                    inim, header, fov_valid, src_mask, skysig=skysigma,
                    photometry_catalog=phot_cat, source_image=single,
                )  # fmt: skip
                if sources is None:
                    btype = self._fall_back_to_constant(i, inim, btype, skyvalue, "no source mask")
                else:
                    exclude = sources
                    qa_mask = sources
                    plot_source_mask(
                        data,
                        ~valid,
                        factory.source_mask_figures(inim)[0],
                        os.path.splitext(get_basename(inim))[0],
                        subtitle=f"{100 - usable:.2f}% excluded by the source and FOV masks, "
                        f"{usable:.2f}% usable for the background mesh",
                        header=header,
                        reprojected=self.plan.inputs_are_reprojected,
                    )
            if quality_mask is not None:
                trail = (quality_mask & int(MaskBit.SATELLITE)) != 0
                exclude = trail if exclude is None else (exclude | trail)
                if self.plan.satellite_mask_enabled:  # absent card = never evaluated, 0 = evaluated and clear
                    card = (int(trail.sum()), "Pixels masked as satellite trail")
                    header["NTRAILPX"], self.input_headers[i]["NTRAILPX"] = card, card
            if btype == "dynamic" and exclude is not None:
                btype = self._crowding_fallback(i, inim, exclude, fov_valid, btype, skyvalue)
            frac_card = (self._usable_fraction(exclude, fov_valid), "Fraction of pixels used for the sky estimate")
            header["BACKFRAC"], self.input_headers[i]["BACKFRAC"] = frac_card, frac_card
            is_steppy = methods[btype](
                inim,
                outim,
                data=data,
                header=header,
                bkg=bkg,
                bkg_rms=bkg_rms,
                skyval=skyvalue,
                ignore_steppy_flag=ignore_steppy_flag,
                exclude=exclude,
                fov_valid=fov_valid,
                quality_mask=quality_mask,
                index=i,
                qa_mask=qa_mask,
                qa_coverage=fov_valid if fov_valid is not None else (np.isfinite(data) & (data != 0)),
            )

            # if is_steppy and not ignore_steppy_flag:
            #     self.logger.warning(f"Background subtraction failed for {get_basename(outim)}")
            #     self.logger.warning(f"Re-running background subtraction with constant value")
            #     self._const_bkgsub(inim, outim, skyval=skyvalue)

            self.logger.info(
                f"Background subtraction ({btype}) completed for {get_basename(outim)} [image {i+1}/{len(input_images)}] in {time_diff_in_seconds(st_loop)} seconds"
            )

        jobs = list(enumerate(zip(input_images, bkgsub_images, bkg_images, bkg_rms_images, skyvalues, skysigmas,
                                  fov_mask_images, source_mask_images, types, singles, catalogs)))  # fmt: skip
        if not self.overwrite:
            n_all = len(jobs)
            pending = []
            for i, job in jobs:
                if not self._background_output_exists(job[1]):
                    pending.append((i, job))
                    continue
                # the header snapshot must follow what the kept product did (fallback CONSTANT)
                cached = self.storage.frame_cache.get(job[1])
                try:
                    kept = cached[1] if cached else fits.getheader(job[1])
                except OSError:
                    kept = {}
                # a product subtracted the other way cannot stand in, or toggling bkgsub would be a no-op
                kept_type = str(kept.get("BACKTYPE") or "").upper()
                if kept_type and kept_type != types[i].upper() and not (types[i] == "dynamic" and kept_type == "CONSTANT"):
                    self.logger.info(
                        f"{get_basename(job[1])} was made with BACKTYPE={kept_type}, this run wants "
                        f"{types[i].upper()}; recomputing"
                    )
                    pending.append((i, job))
                    continue
                if kept.get("BACKTYPE"):
                    self.input_headers[i]["BACKTYPE"] = (str(kept["BACKTYPE"]).upper(), "Background subtraction type")
                for key, comment in (("BACKFRAC", "Fraction of pixels used for the sky estimate"),
                                     ("NTRAILPX", "Pixels masked as satellite trail")):  # fmt: skip
                    if kept.get(key) is not None:
                        self.input_headers[i][key] = (kept[key], comment)
            jobs = pending
            if len(jobs) < n_all:
                self.logger.info(f"{n_all - len(jobs)} existing bkgsub products skipped, {len(jobs)} to compute")
        n_workers = conservative_worker_count(len(jobs))
        if n_workers <= 1:
            for i, job in jobs:
                _bkgsub_one(i, *job)
        else:
            self.logger.info(f"Background subtraction with {n_workers} workers")
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = [pool.submit(_bkgsub_one, i, *job) for i, job in jobs]
                for f in futures:
                    f.result()

        self.logger.info(
            f"Background subtraction is completed in {time_diff_in_seconds(st)} ({time_diff_in_seconds(st, return_float=True)/len(input_images):.1f} s/image)"
        )

        fractions = [v for v in self.input_headers.values("BACKFRAC") if v is not None]
        if fractions:
            card = (round(float(np.mean(fractions)), 4), "Input mean fraction of pixels used for sky")
            self.input_headers.run_cards["BACKFRAC"] = card

        self.images_to_coadd = bkgsub_images
        return bkgsub_images

    def bkgsub_methods(self) -> dict:
        """Map configured background names to per-image routines."""
        return {"none": self._no_bkgsub, "constant": self._const_bkgsub, "dynamic": self._dynamic_bkgsub}

    def _no_bkgsub(self, inim, outim, data=None, header=None, fov_valid=None, quality_mask=None, **kwargs):
        """Stage the frame with no sky model removed; mask exactly as the subtracting routines do."""
        _data, _hdr = self._read_frame(inim, data, header)
        _hdr["BACKTYPE"] = ("NONE", "Background subtraction type")
        if fov_valid is not None:
            _data[~fov_valid] = 0.0  # keep out-of-FOV at 0: the coadd's validity marker
        if quality_mask is not None:
            # NaN, not 0: the pixel stays inside the geometric footprint for coverage_policy
            trail = (quality_mask & int(MaskBit.SATELLITE)) != 0
            _data[trail if fov_valid is None else (trail & fov_valid)] = np.nan
        self._record_background_residuals(_data, _hdr, kwargs.get("qa_mask"), kwargs.get("qa_coverage", fov_valid), quality_mask)
        self._write_background_output(outim, _data, _hdr)
        return False

    def _default_bkgsub_type(self, skyvalues, skyval_cut: float) -> str:
        """Choose one background routine for a group without an explicit setting."""
        return "constant" if any(sv is not None and sv < skyval_cut for sv in skyvalues) else "dynamic"

    def _resolve_bkgsub_type(self, requested: str, skyval, skyval_cut: float) -> str:
        """Which routine one image gets. Only 'individual' decides per image."""
        if requested != "individual":
            return requested
        if skyval is None:
            return "dynamic"
        # a nearly-empty sky quantises into a step-like mesh background, so prefer the
        # scalar SKYVAL there
        return "constant" if skyval < skyval_cut else "dynamic"

    @staticmethod
    def _usable_fraction(exclude: np.ndarray | None, fov_valid: np.ndarray | None) -> float:
        """Fraction of the frame left for the sky estimate after the source, trail and FOV masks."""
        if exclude is None:
            return 1.0 if fov_valid is None else float(fov_valid.mean())
        valid = ~exclude if fov_valid is None else (fov_valid & ~exclude)
        return float(valid.mean())

    def _fall_back_to_constant(self, index: int, inim: str, btype: str, skyvalue, reason: str) -> str:
        """Take the scalar sky for one frame, unless it carries no sky level to take."""
        if skyvalue is None:
            self.logger.warning(
                f"{get_basename(inim)}: {reason}, and no sky level to subtract instead; keeping the mesh background"
            )
            return btype
        self.logger.warning(f"{get_basename(inim)}: {reason}; falling back to constant background subtraction")
        self.input_headers[index]["BACKTYPE"] = ("CONSTANT", "Background subtraction type")
        return "constant"

    def _crowding_fallback(self, index: int, inim: str, exclude, fov_valid, btype: str, skyvalue=None) -> str:
        """Constant sky when the mesh would be mostly interpolated: too little frame left, or too many boxes dropped."""
        from .utils import boxes_dropped

        plan = self.plan
        usable = 100 * self._usable_fraction(exclude, fov_valid)
        excluded = exclude if fov_valid is None else (exclude | ~fov_valid)
        dropped = boxes_dropped(excluded, plan.background_box_size, plan.background_exclude_percentile)
        if usable >= plan.background_min_usable and dropped <= plan.background_max_dropped_boxes:
            return btype
        return self._fall_back_to_constant(
            index,
            inim,
            btype,
            skyvalue,
            f"{usable:.0f}% of the frame usable and {dropped:.0f}% of the mesh boxes dropped "
            f"(limits {plan.background_min_usable:g}% and {plan.background_max_dropped_boxes:g}%)",
        )

    def _source_mask(
        self,
        inim: str,
        header,
        fov_valid: np.ndarray | None,
        outmask: str,
        skysig: float | None = None,
        star_scale: float = 2.0,
        galaxy_scale: float = 2.5,
        class_star_cut: float = 0.5,
        min_radius: float = 3.0,
        photometry_catalog: str | None = None,
        source_image: str | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None, float]:
        """Source mask and the in-FOV, off-source mask, from the single's own photometry catalog.

        The catalog beside the single is the only source of detections: there is no second SExtractor
        pass, so a frame whose catalog is missing or unusable gets no mask and the caller decides."""
        from .utils import build_source_mask, source_ellipses_on_frame, write_mask_plio

        ellipses = None
        if photometry_catalog and os.path.exists(photometry_catalog):
            ellipses = source_ellipses_on_frame(
                photometry_catalog,
                fits.getheader(source_image or inim),
                header,
                logger=self.logger,
            )
        if ellipses is None:
            self.logger.warning(
                f"No usable photometry catalog for {get_basename(inim)}"
                f"{'' if photometry_catalog else ' (none resolvable)'}; no source mask"
            )
            return None, None, 0.0
        self.logger.debug(f"{len(ellipses)} source ellipses from {get_basename(photometry_catalog)}")

        sources = build_source_mask(
            ellipses,
            (header["NAXIS2"], header["NAXIS1"]),
            skysig=skysig,
            star_scale=star_scale,
            galaxy_scale=galaxy_scale,
            class_star_cut=class_star_cut,
            min_radius=min_radius,
            logger=self.logger,
        )

        valid = ~sources if fov_valid is None else (fov_valid & ~sources)
        saved = []
        if self.storage.policy == "disk":
            write_mask_plio(outmask, valid)
            saved.append(outmask)
        if self.config_node.imcoadd.dump_source_masks:
            dump = os.path.join(self.path.imcoadd.factory.source_mask_dump_dir, get_basename(outmask))
            write_mask_plio(dump, valid)
            saved.append(dump)
        usable = float(100 * valid.mean())
        destination = ", ".join(saved) if saved else "memory"
        self.logger.debug(f"Source mask ({usable:.1f}% usable): {destination}")
        return sources, valid, usable

    def build_fov_masks(self, resampled_images, erode_iter: int = 3) -> list[str | None]:
        """Build background masks from pristine resampled footprints."""
        factory = self.path.imcoadd.factory
        outputs = factory.stage_images(resampled_images, "fovmask", self.storage.bkgsub_dir)
        self._fov_masks = []
        for inim, outmask in zip(atleast_1d(resampled_images), outputs):
            if os.path.exists(outmask) and not self.overwrite:
                # a written mask means the frame needed one; frames that needed none wrote
                # nothing and re-derive below (they pay one resamp read, nothing else)
                self._fov_masks.append(outmask)
            else:
                self._fov_masks.append(
                    outmask if self._write_fov_mask(inim, outmask, erode_iter=erode_iter) is not None else None
                )
        return self._fov_masks

    def shrink_fov_masks(self, delta_peeings, kernel_extent: float = 4.0) -> list[str | None]:
        """Shrink stored footprints by the convolution kernel's reach."""
        from scipy.ndimage import binary_erosion

        for i, (mask, delta) in enumerate(zip(self._fov_masks, atleast_1d(delta_peeings))):
            if mask is None or not delta:
                continue
            extra = int(np.ceil(kernel_extent * float(delta) / np.sqrt(8 * np.log(2))))
            if extra < 1:
                continue
            valid = fits.getdata(mask).astype(bool)
            valid = binary_erosion(valid, np.ones((3, 3), dtype=bool), iterations=extra, border_value=0)
            shrunk = add_suffix(mask, "shrunk")  # the pristine mask is what a resume reuses
            fits.writeto(shrunk, valid.astype(np.uint8), overwrite=True)
            self._fov_masks[i] = shrunk
            self.storage.working_mask_paths.append(shrunk)  # removed with the run's masks
            self.logger.debug(f"Shrank {get_basename(mask)} by {extra} px for a {delta:.2f} px kernel")
        return self._fov_masks

    def _fov_valid(self, data: np.ndarray, name: str, erode_iter: int = 3) -> np.ndarray | None:
        """Return the eroded valid-pixel mask of a reprojected frame."""
        from scipy.ndimage import binary_erosion

        valid = np.isfinite(data) & (data != 0)  # NaN pads a coadd used as direct input
        if valid.all():
            self.logger.debug(f"No out-of-FOV pixels in {name}; skipping FOV mask")
            return None

        # border_value=1: the array bound is not an FOV edge, only the zero padding is
        valid = binary_erosion(valid, np.ones((3, 3), dtype=bool), iterations=erode_iter, border_value=1)
        self.logger.debug(f"FOV mask ({100 * valid.mean():.1f}% valid) for {name}")
        return valid

    def _write_fov_mask(self, inim: str, outmask: str, erode_iter: int = 3) -> np.ndarray | None:
        """`_fov_valid` on a frame read from disk, persisted for a later stage to reuse."""
        valid = self._fov_valid(fits.getdata(inim, memmap=False), get_basename(inim), erode_iter=erode_iter)
        if valid is None:
            return None
        fits.writeto(outmask, valid.astype(np.uint8), overwrite=True)
        self.logger.debug(f"FOV mask saved as {get_basename(outmask)}")
        return valid

    def _guard_sky_rms_propagation(self):
        """Raise if a coadd sky-noise map is asked for; propagation is unimplemented."""
        if self.plan.output_sky_rms_map:
            raise NotImplementedError(
                "imcoadd.output_sky_rms_map: the per-frame sky-RMS models are written, but "
                "propagating them into a coadd sky-noise map (the source-free counterpart of "
                "the current weight map) is not implemented yet"
            )

    def _const_bkgsub(
        self,
        inim,
        outim,
        skyval,
        data=None,
        header=None,
        skyval_cut=40,
        fov_valid=None,
        quality_mask=None,
        **kwargs,
    ):

        if self._background_output_exists(outim):
            try:
                cached = self.storage.frame_cache.get(outim)
                _backtype = cached[1].get("BACKTYPE") if cached else fits.getval(outim, "BACKTYPE")
            except KeyError:
                _backtype = ""
            if _backtype.upper() == "CONSTANT":
                if not self.overwrite:
                    self.logger.info(f"Background subtraction result exists; skipping: {get_basename(outim)}")
                    return

        is_steppy = skyval < skyval_cut

        _data, _hdr = self._read_frame(inim, data, header)
        _hdr["BACKTYPE"] = ("CONSTANT", "Background subtraction type")
        # _hdr["BKG_STEP"] = (is_steppy, "SE Background can be step-like")
        _data -= skyval
        if fov_valid is not None:
            _data[~fov_valid] = 0.0  # keep out-of-FOV at 0: the coadd's validity marker
        if quality_mask is not None:
            # NaN, not 0: the pixel stays inside the geometric footprint for coverage_policy
            trail = (quality_mask & int(MaskBit.SATELLITE)) != 0
            _data[trail if fov_valid is None else (trail & fov_valid)] = np.nan
        self.logger.debug(f"Using SKYVAL: {skyval:.3f}")
        self._record_background_residuals(_data, _hdr, kwargs.get("qa_mask"), kwargs.get("qa_coverage", fov_valid), quality_mask)
        self._write_background_output(outim, _data, _hdr)

        return False  # is_steppy is False by definition for constant background subtraction

    def _dynamic_bkgsub(self, inim, outim, bkg, bkg_rms, skyval=None, data=None, header=None, ignore_steppy_flag=False, exclude=None, fov_valid=None, quality_mask=None, index=None, **kwargs):  # fmt: skip
        from .utils import estimate_background

        # from .bkg_step import step_background_check

        plan = self.plan
        _data, _hdr = self._read_frame(inim, data, header)
        try:
            bkg_data, bkg_rms_data = estimate_background(
                _data,
                mask=exclude,
                coverage_mask=None if fov_valid is None else ~fov_valid,
                box_size=plan.background_box_size,
                filter_size=plan.background_filter_size,
                exclude_percentile=plan.background_exclude_percentile,
                with_rms=plan.output_sky_rms_map,
            )
        except ValueError as e:
            # Background2D raises when every box is below the good-pixel threshold; sep used to return zeros
            if skyval is None:
                raise
            self.logger.warning(f"{get_basename(inim)}: no mesh box survived ({e}); constant background instead")
            if index is not None:
                self.input_headers[index]["BACKTYPE"] = ("CONSTANT", "Background subtraction type")
            return self._const_bkgsub(inim, outim, skyval=skyval, data=_data, header=_hdr,
                                      fov_valid=fov_valid, quality_mask=quality_mask, qa_mask=kwargs.get("qa_mask"),
                                      qa_coverage=kwargs.get("qa_coverage", fov_valid))  # fmt: skip
        if self.plan.output_sky_rms_map:
            fits.writeto(bkg_rms, bkg_rms_data, overwrite=True)
        del bkg_rms_data  # do not hold a second full frame past its write
        if self.plan.output_bkg_map:
            fits.writeto(bkg, bkg_data, overwrite=True)
        inside = bkg_data if fov_valid is None else bkg_data[fov_valid]
        plot_background(
            bkg_data,
            self.path.imcoadd.factory.background_figures(inim)[0],
            os.path.splitext(get_basename(inim))[0],
            subtitle=f"box {plan.background_box_size} x filter {plan.background_filter_size}, "
            f"exclude_percentile {plan.background_exclude_percentile:g}%; "
            f"median {np.median(inside):.2f}, peak to peak {np.ptp(inside):.2f} ADU/pixel",
            header=_hdr,
            reprojected=self.plan.inputs_are_reprojected,
        )

        # if ignore_steppy_flag:
        #     is_steppy = False
        # else:
        #     h, w = bkg_data.shape
        #     stripe = np.mean(bkg_data[h // 2 - 100 : h // 2 + 100, :], axis=0)  # already smooth bkg: mean is okay?
        #     is_steppy, info = step_background_check(stripe)
        #     if is_steppy:
        #         self.logger.warning(f"Background is steppy in {get_basename(outim)}")
        #         self.logger.debug(f"Background is steppy: {info}")
        #         return True
        #     else:
        #         self.logger.debug(f"Background is not steppy in {get_basename(outim)}: {info}")

        _hdr["BACKTYPE"] = ("DYNAMIC", "Background subtraction type")
        # _hdr["BKG_STEP"] = (is_steppy, "Background is step-like; likely quantization artifact")
        _data -= bkg_data
        if fov_valid is not None:
            _data[~fov_valid] = 0.0  # keep out-of-FOV at 0: the coadd's validity marker
        if quality_mask is not None:
            # NaN, not 0: the pixel stays inside the geometric footprint for coverage_policy
            trail = (quality_mask & int(MaskBit.SATELLITE)) != 0
            _data[trail if fov_valid is None else (trail & fov_valid)] = np.nan
        self._record_background_residuals(_data, _hdr, kwargs.get("qa_mask"), kwargs.get("qa_coverage", fov_valid), quality_mask)
        self._write_background_output(outim, _data, _hdr)

        # return is_steppy

    @staticmethod
    def _read_frame(inim, data, header):
        """The frame the caller already read, or read it now for a direct routine call."""
        if data is not None and header is not None:
            return data, header
        data, header = fits.getdata(inim, header=True, memmap=False)
        return np.ascontiguousarray(data, dtype=np.float32), header

    # # TODO:
    # def _bkg_qa(self, bkgsub_type: str = "dynamic"):
    #     if bkgsub_type == "dynamic":
    #         # do assessment below
    #         for f in self.config_node.imcoadd.bkg_images:
    #             data = fits.getdata(f)
    #             H, W = data.shape
    #             stripe = np.mean(data[H // 2 - 100 : H // 2 + 100, :], axis=0)

    #         pass
    #     elif bkgsub_type == "constant":
    #         # add dummy key
    #         for f in self.input_images:
    #             update_padded_header(f, {"BACKARTF": (False, "Dynamic bkgsub will cause artifacts")})
    #     else:
    #         raise ValueError(f"_bkg_qa - Invalid bkgsub_type: {bkgsub_type}")

    #     update_padded_header(f, {"BACKARTF": (False, "Dynamic bkgsub will cause artifacts")})

    #     recommenced_bkgsub_type = "constant"  # BACKTYPE "Recommended bkgsub type"
    #     return recommenced_bkgsub_type

    def _record_background_residuals(self, data, header, sources, coverage, quality=None):
        from .background_qa import clear_residual_cards, measure_background_residuals

        clear_residual_cards(header)
        if sources is None:
            self.logger.debug("No source mask; residual sky QA not measured")
            return
        exclude = sources if quality is None else (sources | (quality != 0))
        result = measure_background_residuals(
            data, exclude=exclude, coverage=coverage,
        )
        header.update(result.cards())
        self.logger.debug(f"Residual sky: {result}")
