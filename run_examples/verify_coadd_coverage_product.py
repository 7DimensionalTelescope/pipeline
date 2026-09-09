"""Regression check for the coadd coverage/count product and its independence from the OR bitmask.

    python run_examples/verify_coadd_coverage_product.py

Synthetic data only: tiny FITS frames in a temporary directory, no PathHandler product path, no
database, no production tree. Named outside test/ because .gitignore drops test*.py and test/.

Covers the four output_counts_map x output_mask_map combinations, a coadd with no detector
bad-pixel mask (the cross-filter shape), and the legacy routine's NGEOM/NUSED pass.
"""

import copy
import os
import shutil
import sys
import tempfile
import types

import numpy as np
import yaml
from astropy.io import fits

import pipeline  # noqa: F401  (config-hash gate)
import pipeline.imcoadd.imcoadd as imcoadd_module
import pipeline.path.path as path_module
from pipeline.const import REF_DIR
from pipeline.imcoadd.coadd_plan import resolve_coadd_plan
from pipeline.imcoadd.const import MaskBit
from pipeline.imcoadd.counts import COUNT_PLANES
from pipeline.imcoadd.masks import MaskMixin
from pipeline.imcoadd.reproject_first import ReprojectFirstCoaddMixin
from pipeline.imcoadd.swarp import SwarpMixin
from pipeline.imcoadd.utils import read_count_planes

H, W, N_FRAMES = 48, 60, 5
BAD_Y, BAD_X = 20, 30
OUTLIER_Y, OUTLIER_X = 10, 12


class Node:
    def __init__(self, mapping):
        self.__dict__.update(mapping)


def frame_header(saturate=None):
    header = fits.Header()
    header["CTYPE1"], header["CTYPE2"] = "RA---TAN", "DEC--TAN"
    header["CRVAL1"], header["CRVAL2"] = 150.0, 2.0
    header["CRPIX1"], header["CRPIX2"] = 30.0, 24.0
    header["CD1_1"], header["CD2_2"] = -1.4e-4, 1.4e-4
    header["EGAIN"], header["SKYSIG"], header["PEEING"] = 2.0, 1.0, 3.0
    if saturate is not None:
        header["SATURATE"] = saturate
    return header


def build_inputs(work, with_saturation=True):
    """N_FRAMES direct-grid frames, their weight maps, and a bad-pixel mask covering one pixel."""
    rng = np.random.default_rng(17)
    images, weights = [], []
    for i in range(N_FRAMES):
        data = (100.0 + rng.normal(0, 1.0, (H, W))).astype(np.float32)
        data[0, :] = 0.0  # a no-data border, so NGEOM and NUSED are not trivially uniform
        if i == 0:
            data[OUTLIER_Y, OUTLIER_X] = 9000.0  # a genuine clipped outlier
        if with_saturation and i < 3:
            data[5, 5] = 70000.0
        image = os.path.join(work, f"f{i}.fits")
        weight = os.path.join(work, f"f{i}_weight.fits")
        fits.writeto(image, data, header=frame_header(65535.0 if with_saturation else None), overwrite=True)
        weight_data = np.full((H, W), 0.25, dtype=np.float32)
        weight_data[0, :] = 0.0
        fits.writeto(weight, weight_data, header=frame_header(), overwrite=True)
        images.append(image)
        weights.append(weight)

    mask = np.zeros((H, W), dtype=np.uint8)
    mask[BAD_Y, BAD_X] = 1
    bpmask = os.path.join(work, "bpmask.fits")
    hdu = fits.CompImageHDU(data=mask, compression_type="RICE_1")
    hdu.header["BADPIX"] = 1
    hdu.header["IMAGEID"] = "bpmid00000000000000000000000000"
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(bpmask, overwrite=True)
    return images, weights, bpmask


class Stub(ReprojectFirstCoaddMixin, MaskMixin, SwarpMixin):
    """The real coadd/mask/legacy methods over synthetic paths."""

    # resolved on ImCoadd itself, not on a mixin; bound here so the test exercises the real body
    _bpmask_info = imcoadd_module.ImCoadd._bpmask_info
    _get_bpmask = imcoadd_module.ImCoadd._get_bpmask
    _process_error = imcoadd_module.CoaddError

    def __init__(self, work, plan, images, has_detector_bpm=True):
        imcoadd_module.ImCoadd._reset_run_state(self)  # the one definition of per-run state, never a copy of it
        self.plan = plan
        self.input_images = images
        self.images_to_coadd = images
        self._has_detector_bpm = has_detector_bpm
        self._use_gpu = False
        self._max_peeing = 3.0
        self.overwrite = True
        self.logger = types.SimpleNamespace(
            info=lambda *a: None, debug=lambda *a: None, warning=lambda *a: None, error=lambda *a: None
        )
        coadd_image = os.path.join(work, "obj_coadd.fits")
        node = copy.deepcopy(BASE_NODE)
        node["coadd_image"] = coadd_image
        self.config_node = types.SimpleNamespace(imcoadd=types.SimpleNamespace(**node))
        mask_dir = os.path.join(work, "factory", "mask")
        os.makedirs(mask_dir, exist_ok=True)
        factory = types.SimpleNamespace(
            mask_dir=mask_dir,
            stage_images=lambda imgs, suffix, subdir: [
                os.path.join(subdir, os.path.basename(f).replace(".fits", f"_{suffix}.fits")) for f in imgs
            ],
            coadd_counts_image=coadd_image.replace(".fits", "_counts.fits"),
            resampled_images=lambda imgs, pass_type="": list(imgs),
            swarp_resample_dir=lambda pass_type="": work,
            resampled_weight_images=lambda imgs, pass_type="": [
                f.replace(".fits", "_weight.fits") for f in imgs
            ],
        )
        self.path = types.SimpleNamespace(imcoadd=types.SimpleNamespace(factory=factory))
        self.input_headers = types.SimpleNamespace(
            values=lambda key: [fits.getheader(f).get(key) for f in images],
            coadd_header=fits.Header(),
            names=[os.path.basename(f) for f in images],
        )
        self.intermediate_storage = types.SimpleNamespace(policy="disk", frame_cache={}, working_mask_paths=[])

    @property
    def storage(self):
        return self.intermediate_storage

    def _read_stage_frame(self, image):
        data, header = fits.getdata(image, header=True, memmap=False)
        return np.ascontiguousarray(data, dtype=np.float32), header

    def _single_wcs_header(self, single, header=None):
        return fits.getheader(single) if header is None else header

    def _guard_sky_rms_propagation(self):
        pass

    def _coadd_flxscales(self):
        return False


BASE_NODE = yaml.safe_load(open(os.path.join(REF_DIR, "sciproc_base.yml")))["imcoadd"]


def make_plan(**overrides):
    node = copy.deepcopy(BASE_NODE)
    # match_swarp_size False keeps the target grid at the frames' bounding box, not the 10200x6800 SWarp grid
    node.update(
        dict(coadd_routine="direct", coadd_mode="clipped", joint_wcs=False, convolve=False, match_swarp_size=False)
    )
    node.update(overrides)
    return resolve_coadd_plan(Node(node))


def run_coadd(work, plan, images, weights, has_detector_bpm=True):
    """The reproject-first tail, in the order the routine runs it."""
    stub = Stub(work, plan, images, has_detector_bpm=has_detector_bpm)
    if stub._need_quality_masks:
        stub.prepare_quality_masks(images, detector_images=images)
    stub.coadd_in_memory(images, weight_images=weights)
    stub.finalize_quality_masks()
    return stub


def check(label, condition, detail=""):
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    return bool(condition)


def main():
    root = tempfile.mkdtemp(prefix="verify_coverage_")
    ok = True
    try:
        work = os.path.join(root, "coadd")
        os.makedirs(work)
        images, weights, bpmask = build_inputs(work)
        path_module.PathHandler.get_bpmask = classmethod(lambda cls, image: bpmask)
        imcoadd_module.PathHandler.get_bpmask = path_module.PathHandler.get_bpmask

        print("Four output_counts_map x output_mask_map combinations (direct + clipped, detector bpmask):")
        for counts_on in (True, False):
            for mask_on in (True, False):
                case = os.path.join(root, f"c{int(counts_on)}m{int(mask_on)}")
                shutil.copytree(work, case)
                case_images = [os.path.join(case, os.path.basename(f)) for f in images]
                case_weights = [os.path.join(case, os.path.basename(f)) for f in weights]
                plan = make_plan(output_counts_map=counts_on, output_mask_map=mask_on)
                stub = run_coadd(case, plan, case_images, case_weights)
                coadd = stub.config_node.imcoadd.coadd_image
                counts_path, mask_path = coadd.replace(".fits", "_counts.fits"), coadd.replace(".fits", "_mask.fits")
                label = f"counts={counts_on!s:5s} mask={mask_on!s:5s}"
                ok &= check(
                    f"{label} -> products",
                    os.path.exists(counts_path) == counts_on and os.path.exists(mask_path) == mask_on,
                    f"counts_file={os.path.exists(counts_path)} mask_file={os.path.exists(mask_path)}",
                )
                if counts_on:
                    planes = read_count_planes(counts_path)
                    ok &= check(
                        f"{label} -> all six planes (the bit masks are built for the counts too)",
                        list(planes) == [p.name for p in COUNT_PLANES],
                        f"planes={list(planes)}",
                    )
                    ok &= check(
                        f"{label} -> NGEOM/NUSED/NBAD/NSAT/NOUTLIER values",
                        (
                            int(planes["NGEOM"][BAD_Y, BAD_X]) == N_FRAMES
                            and int(planes["NUSED"][30, 40]) == N_FRAMES
                            and int(planes["NBAD"][BAD_Y, BAD_X]) == N_FRAMES
                            and int(planes["NSAT"][5, 5]) == 3
                            and int(planes["NOUTLIER"][OUTLIER_Y, OUTLIER_X]) == 1
                            and int(planes["NGEOM"][0, 0]) == 0
                        ),
                        f"NGEOM@bad={int(planes['NGEOM'][BAD_Y, BAD_X])} NBAD@bad={int(planes['NBAD'][BAD_Y, BAD_X])} "
                        f"NSAT={int(planes['NSAT'][5, 5])} NOUTLIER={int(planes['NOUTLIER'][OUTLIER_Y, OUTLIER_X])}",
                    )
                    ok &= check(
                        f"{label} -> saturated samples do not contribute (NSAT reduces NUSED)",
                        int(planes["NSAT"][5, 5]) == 3 and int(planes["NUSED"][5, 5]) == N_FRAMES - 3,
                        f"NSAT={int(planes['NSAT'][5, 5])} NUSED={int(planes['NUSED'][5, 5])} "
                        f"NGEOM={int(planes['NGEOM'][5, 5])}",
                    )
                    coadd_data = fits.getdata(coadd)
                    ok &= check(
                        f"{label} -> the coadd at that pixel is the two unsaturated frames, not 70000",
                        np.isfinite(coadd_data[5, 5]) and abs(float(coadd_data[5, 5]) - 100.0) < 5.0,
                        f"coadd[5,5]={float(coadd_data[5, 5]):.3f} (background is ~100, saturated value 70000)",
                    )
                    weight_data = fits.getdata(coadd.replace(".fits", "_weight.fits"))
                    ratio = float(weight_data[5, 5]) / float(weight_data[30, 40])
                    ok &= check(
                        f"{label} -> the weight there is the two-contributor weight",
                        abs(ratio - (N_FRAMES - 3) / N_FRAMES) < 1e-3,
                        f"weight[5,5]/weight[clean]={ratio:.4f}, expected {(N_FRAMES - 3) / N_FRAMES:.4f}",
                    )
                    if mask_on:
                        or_mask = fits.getdata(mask_path, ext=1).astype(np.uint8)
                        derived = np.zeros_like(or_mask)
                        for bit, plane in (
                            (MaskBit.BADPIX, "NBAD"),
                            (MaskBit.SATURATED, "NSAT"),
                            (MaskBit.SATELLITE, "NTRAIL"),
                            (MaskBit.OUTLIER, "NOUTLIER"),
                        ):
                            derived[planes[plane] > 0] |= int(bit)
                        ok &= check(f"{label} -> OR mask derivable from the counts", np.array_equal(derived, or_mask))

        print("Coverage without a detector bad-pixel mask (the cross-filter shape: direct + proper):")
        case = os.path.join(root, "nobpm")
        shutil.copytree(work, case)
        case_images = [os.path.join(case, os.path.basename(f)) for f in images]
        case_weights = [os.path.join(case, os.path.basename(f)) for f in weights]
        path_module.PathHandler.get_bpmask = classmethod(
            lambda cls, image: (_ for _ in ()).throw(AssertionError("no detector bpmask on this path"))
        )
        imcoadd_module.PathHandler.get_bpmask = path_module.PathHandler.get_bpmask
        plan = make_plan(coadd_mode="proper", output_counts_map=True, output_mask_map=False)
        stub = run_coadd(case, plan, case_images, case_weights, has_detector_bpm=False)
        counts_path = stub.config_node.imcoadd.coadd_image.replace(".fits", "_counts.fits")
        planes = read_count_planes(counts_path)
        ok &= check(
            "counts written with no bpmask resolved",
            os.path.exists(counts_path) and list(planes) == ["NGEOM", "NUSED"],
            f"planes={list(planes)}",
        )
        ok &= check(
            "NGEOM/NUSED are real counts there",
            int(planes["NGEOM"][30, 40]) == N_FRAMES and int(planes["NGEOM"][0, 0]) == 0,
            f"NGEOM interior={int(planes['NGEOM'][30, 40])} border={int(planes['NGEOM'][0, 0])}",
        )

        print("Per-pixel saturation ceiling from the master flat:")
        case = os.path.join(root, "saturmap")
        shutil.copytree(work, case)
        case_images = [os.path.join(case, os.path.basename(f)) for f in images]
        case_weights = [os.path.join(case, os.path.basename(f)) for f in weights]
        # a vignetted master flat: 1.0 at the centre, 0.7 at the corners, CENCLPMD taken over the centre
        yy, xx = np.mgrid[0:H, 0:W]
        radius = np.hypot(yy - H / 2, xx - W / 2) / np.hypot(H / 2, W / 2)
        flat = (1.0 - 0.3 * radius**2).astype(np.float32)
        flat_path = os.path.join(case, "flat.fits")
        fits.writeto(flat_path, flat, header=fits.Header({"CENCLPMD": float(np.median(flat[20:28, 26:34]))}),
                     overwrite=True)
        path_module.PathHandler.resolve_weight_map_input_abspath = classmethod(
            lambda cls, zdf: ("dark", flat_path, "sigz", "sigf")
        )
        plan = make_plan(output_counts_map=True, output_mask_map=False)
        stub = Stub(case, plan, case_images, True)
        stub._zdf_cache = {f: ("z", "d", "f") for f in case_images}
        level, flat_id = stub._saturation_map(case_images[0], 65535.0, (H, W))
        centre, corner = float(level[H // 2, W // 2]), float(level[1, 1])
        ok &= check(
            "the ceiling follows the flat: highest where the flat is lowest",
            hasattr(level, "shape") and corner > centre and abs(centre / 65535.0 - 1.0) < 0.02,
            f"centre={centre:.0f} corner={corner:.0f} ratio={corner / centre:.3f} (flat 1.00 -> 0.70)",
        )
        # a corner value ABOVE the scalar level but BELOW that corner's own ceiling: the scalar test
        # calls it saturated, the per-pixel one does not, because its raw value never reached full well
        probe_value = 70000.0
        probe = fits.getdata(case_images[0])
        probe[1, 1] = probe_value
        probe[H // 2, W // 2] = probe_value  # the same value at the centre IS saturated there
        fits.writeto(case_images[0], probe, header=fits.getheader(case_images[0]), overwrite=True)
        stub2 = Stub(case, plan, case_images, True)
        stub2._zdf_cache = {f: ("z", "d", "f") for f in case_images}
        positions = stub2._saturated_positions(case_images[0], fits.getheader(case_images[0]), (H, W))
        flagged = positions.block_mask(0, H, 0, W)
        ok &= check(
            "the same value is saturated at the centre and not in the corner",
            bool(flagged[H // 2, W // 2]) and not bool(flagged[1, 1]),
            f"{probe_value:.0f} >= scalar 65535 everywhere, but ceilings are {centre:.0f} (centre, flagged="
            f"{bool(flagged[H // 2, W // 2])}) and {corner:.0f} (corner, flagged={bool(flagged[1, 1])})",
        )

        print("Saturation reaches the coadd on exactly one channel, and the channel is read:")
        for label, over in (
            ("reproject-first + 1px", dict(coadd_routine="reproject-first")),
            ("reproject-first + policy off", dict(coadd_routine="reproject-first",
                                                  badpix_reprojection_policy="off", zero_badpix_weight=False)),
            ("reproject-first + off, no weight", dict(coadd_routine="reproject-first", coadd_mode="mean",
                                                      badpix_reprojection_policy="off",
                                                      zero_badpix_weight=False, output_weight_map=False)),
            ("direct + 1px", dict(coadd_routine="direct")),
        ):
            p = make_plan(**over)
            via_weight = p.zero_saturated_before_reprojection
            # the same expression coadd_in_memory uses to decide whether the resampled weight becomes `masks`
            weight_is_read = p.weighting != "pixelwise" and (p.policy == "1px" or via_weight)
            ok &= check(
                f"{label}: zeros in the weight are always read back",
                (not via_weight) or weight_is_read,
                f"zero_saturated_before_reprojection={via_weight} weight_read_as_masks={weight_is_read} "
                f"(sparse channel used when not via_weight)",
            )
        for blocked, over in (
            ("reproject-first + conservative", dict(coadd_routine="reproject-first",
                                                    badpix_reprojection_policy="conservative")),
            ("reproject-first + pixel-wise + 1px", dict(coadd_routine="reproject-first",
                                                        coadd_weighting="pixel-wise")),
        ):
            try:
                make_plan(**over)
                raised = ""
            except NotImplementedError as e:
                raised = str(e)
            ok &= check(f"{blocked}: refused at plan stage", "would not say which of the two" in raised,
                        raised[:110] or "no exception raised")

        print("Saturation reaches SWarp through the weight sidecar (conservative propagation):")
        import pipeline.imcoadd.weight as weight_module
        from pipeline.imcoadd.interpolate import weight_and_interpolate_cpu
        from pipeline.utils import add_suffix

        weight_module.optimized_parallel = lambda sci, *calib: np.ones_like(sci, dtype=np.float32)
        sat_case = os.path.join(root, "sidecar")
        os.makedirs(sat_case)
        single = os.path.join(sat_case, "single.fits")
        frame = np.full((H, W), 100.0, dtype=np.float32)
        frame[5, 5] = 70000.0  # saturated at the centre-ish; the ceiling there is ~the scalar
        fits.writeto(single, frame, header=frame_header(65535.0), overwrite=True)
        bpm = np.zeros((H, W), dtype=np.uint8)
        bpm[BAD_Y, BAD_X] = 1
        bpm_path = os.path.join(sat_case, "bpmask.fits")
        hdu = fits.CompImageHDU(data=bpm, compression_type="RICE_1")
        hdu.header["BADPIX"] = 1
        fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(bpm_path, overwrite=True)
        path_module.PathHandler.get_bpmask = classmethod(lambda cls, image: bpm_path)
        imcoadd_module.PathHandler.get_bpmask = path_module.PathHandler.get_bpmask
        sat_stub = Stub(sat_case, make_plan(), [single], True)
        for zeroing, tag in ((sat_stub._saturated_detector_mask, "on"), (None, "off")):
            out = os.path.join(sat_case, f"interp_{tag}.fits")
            weight_and_interpolate_cpu(
                [single], bpm_path, [out], calib=(0, 0, 0, 0), method="median", badpix=1,
                zero_interp_weight=False, saturated_mask=zeroing,
            )
            sidecar = fits.getdata(add_suffix(out, "weight"))
            ok &= check(
                f"saturated_mask {tag}: the pre-reprojection sidecar is {'zeroed' if zeroing else 'untouched'}"
                " at the saturated pixel",
                (float(sidecar[5, 5]) == 0.0) == bool(zeroing) and float(sidecar[20, 20]) > 0,
                f"sidecar[sat]={float(sidecar[5, 5]):.3f} sidecar[clean]={float(sidecar[20, 20]):.3f}",
            )

        print("Legacy routine NGEOM/NUSED from its resamples:")
        case = os.path.join(root, "legacy")
        shutil.copytree(work, case)
        case_images = [os.path.join(case, os.path.basename(f)) for f in images]
        plan = make_plan(
            coadd_routine="legacy", coadd_mode="median", output_counts_map=True, output_mask_map=False
        )
        stub = Stub(case, plan, case_images, plan.output_counts_map)
        # a legacy coadd is written by SWarp; stand in for it with a frame-shaped product
        fits.writeto(
            stub.config_node.imcoadd.coadd_image,
            np.full((H, W), 100.0, dtype=np.float32),
            header=frame_header(),
            overwrite=True,
        )
        one = fits.getdata(case_images[1])
        one[12, 13] = np.nan  # finite-but-present vs geometrically-present differ at this pixel
        fits.writeto(case_images[1], one, header=fits.getheader(case_images[1]), overwrite=True)
        stub.apply_legacy_coverage_policy(case_images)
        stub._coadd_mask_builder = None
        stub.write_coadd_counts()
        planes = read_count_planes(stub.config_node.imcoadd.coadd_image.replace(".fits", "_counts.fits"))
        ok &= check(
            "legacy produces real NGEOM and NUSED",
            list(planes) == ["NGEOM", "NUSED"]
            and int(planes["NGEOM"][30, 40]) == N_FRAMES
            and int(planes["NUSED"][30, 40]) == N_FRAMES
            and int(planes["NGEOM"][12, 13]) == N_FRAMES
            and int(planes["NUSED"][12, 13]) == N_FRAMES - 1
            and int(planes["NGEOM"][0, 0]) == 0,
            f"planes={list(planes)} NGEOM@nan={int(planes['NGEOM'][12, 13])} NUSED@nan={int(planes['NUSED'][12, 13])}",
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print("COVERAGE-PRODUCT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
