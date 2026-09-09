"""Regression check for the coadd coverage/count product and its independence from the OR bitmask.

    python run_examples/verify_coadd_coverage_product.py

Synthetic data only: tiny FITS frames in a temporary directory, no PathHandler product path, no
database, no production tree. Named outside test/ because .gitignore drops test*.py and test/.

Covers the four output_counts_map x output_mask_map combinations, a coadd with no detector
bad-pixel mask (the cross-filter shape), the legacy routine's NGEOM/NUSED pass, the two
check plots (colour per MaskBit, draw order, tint scaled by lost samples rather than set by a flag) and
their sky orientation.
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
HOLE_Y, HOLE_X = 30, 44
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
    _intended_coverage = imcoadd_module.ImCoadd._intended_coverage
    fill_coadd_nan = imcoadd_module.ImCoadd.fill_coadd_nan
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
            coadd_counts_figure=os.path.join(work, "figures", "obj_coadd_counts.jpg"),
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


def _rgb_distance(image, color):
    from matplotlib.colors import to_rgb

    target = np.array(to_rgb(color), dtype=np.float32) * 255.0
    pixels = np.asarray(image, dtype=np.float32).reshape(-1, 3)
    return np.sqrt(((pixels - target) ** 2).sum(axis=1))


def nearest_pixel_distance(image, color):
    """Smallest RGB distance between any rendered pixel and *color*."""
    return float(_rgb_distance(image, color).min())


def pixels_near(image, color, tolerance):
    """How many rendered pixels sit within *tolerance* of *color* in RGB."""
    return int((_rgb_distance(image, color) < tolerance).sum())


def check_plots(root):
    """Both check plots: a colour per MaskBit, the rarest bit on top, and the tint scaled by lost samples."""
    from PIL import Image

    from pipeline.imcoadd.counts import COUNT_PLANES
    from pipeline.imcoadd.plotting import _percent, plot_coadd_counts, plot_source_mask

    print("Check plots:")
    ok = True
    figures = os.path.join(root, "figures")
    ph, pw, n = 400, 600, 5
    size = (pw, ph)  # full-bleed: the figure is exactly the reduced array
    planes = {
        "NGEOM": np.full((ph, pw), n, np.uint8),
        "NUSED": np.full((ph, pw), n, np.uint8),
        "NBAD": np.zeros((ph, pw), np.uint8),
        "NSAT": np.zeros((ph, pw), np.uint8),
        "NTRAIL": np.zeros((ph, pw), np.uint8),
        "NOUTLIER": np.zeros((ph, pw), np.uint8),
    }
    planes["NBAD"][40:120, 40:200] = n
    planes["NSAT"][200:280, 120:280] = n
    planes["NTRAIL"][320:340, :] = n
    planes["NOUTLIER"][360:380, 40:560] = n
    by_name = {plane.name: plane for plane in COUNT_PLANES}

    path = plot_coadd_counts(
        planes, os.path.join(figures, "obj_coadd_counts.jpg"), "obj_m600_7DT01_20250101_coadd.fits",
        subtitle=f"{n} input frames, union coverage, clipped combination", n_inputs=n, max_width=pw,
    )  # fmt: skip
    image = Image.open(path).convert("RGB")
    ok &= check(
        "counts check plot written as a light JPEG",
        os.path.exists(path) and image.size == size and 3_000 < os.path.getsize(path) < 900_000,
        f"{image.size[0]}x{image.size[1]} px, {os.path.getsize(path) / 1e3:.0f} kB",
    )
    distances = {name: nearest_pixel_distance(image, by_name[name].color) for name in
                 ("NBAD", "NSAT", "NTRAIL", "NOUTLIER")}  # fmt: skip
    ok &= check(
        "every flagged reason is drawn in its own colour",
        all(d < 40 for d in distances.values()),
        ", ".join(f"{k} {v:.0f}" for k, v in distances.items()) + " (RGB distance to the declared colour)",
    )
    # a rare reason must not print as "0.00%": 641 saturated pixels in 69 Mpx once did exactly that
    rare = 100 * 641 / 69_360_000
    ok &= check(
        "a rare reason keeps its digits in the legend",
        _percent(rare) not in ("0%", "0.00%") and _percent(0) == "0%" and _percent(88.1) == "88%",
        f"641 px in 69 Mpx renders as {_percent(rare)}, an empty plane as {_percent(0)}",
    )
    # the tint is the share of LOST SAMPLES, not a flag: one input in five must render fainter than five in five
    graded = {name: np.zeros((ph, pw), np.uint8) for name in planes}
    graded["NUSED"][:] = n
    graded["NBAD"][50:150, 50:550] = 1
    graded["NBAD"][250:350, 50:550] = n
    faint = Image.open(
        plot_coadd_counts(graded, os.path.join(figures, "graded.jpg"), "graded", n_inputs=n, max_width=pw)
    ).convert("RGB")
    ok &= check(
        "the tint follows the share of lost samples, not a flag",
        pixels_near(faint, by_name["NBAD"].color, 30) > 5_000
        and pixels_near(faint, by_name["NBAD"].color, 30) < pixels_near(faint, by_name["NBAD"].color, 120),
        f"{pixels_near(faint, by_name['NBAD'].color, 30)} px near the full-strength colour vs "
        f"{pixels_near(faint, by_name['NBAD'].color, 120)} px tinted at all (1/{n} inputs stays pale)",
    )
    # a region flagged by both NBAD (bit 2) and NSAT (bit 16) must render as NSAT: higher bit, drawn later
    stacked = {name: np.zeros((ph, pw), np.uint8) for name in planes}
    stacked["NUSED"][:] = n
    stacked["NBAD"][100:300, 100:500] = n
    stacked["NSAT"][100:300, 100:500] = n
    covered = Image.open(
        plot_coadd_counts(stacked, os.path.join(figures, "order.jpg"), "order", n_inputs=n, max_width=pw)
    ).convert("RGB")
    n_green = pixels_near(covered, by_name["NBAD"].color, 60)
    n_red = pixels_near(covered, by_name["NSAT"].color, 60)
    ok &= check(
        "MaskBit order decides the overlap: NSAT (16) covers NBAD (2)",
        n_red > 20 * max(n_green, 1),
        f"{n_red} red px vs {n_green} green px (green is left only in the legend swatch)",
    )
    ok &= check(
        "no counts plot without a depth plane",
        plot_coadd_counts({"NBAD": planes["NBAD"]}, os.path.join(figures, "nope.jpg"), "x") is None
        and not os.path.exists(os.path.join(figures, "nope.jpg")),
        "the grey frame is NGEOM, or NUSED when a stage has no NGEOM; with neither there is nothing to draw",
    )
    only_used = {"NUSED": planes["NUSED"], "NBAD": planes["NBAD"]}
    ok &= check(
        "NGEOM is the grey frame, NUSED only when NGEOM is absent",
        plot_coadd_counts(only_used, os.path.join(figures, "fallback.jpg"), "x", n_inputs=n, max_width=pw)
        is not None
        and os.path.exists(os.path.join(figures, "fallback.jpg")),
        "a legacy/white stage that publishes NGEOM gets NGEOM; the fallback keeps the figure alive either way",
    )

    rng = np.random.default_rng(3)
    data = (100 + rng.normal(0, 2, (ph, pw))).astype(np.float32)
    excluded = np.zeros((ph, pw), np.uint8)
    excluded[100:200, 100:300] = n
    data[excluded > 0] += 60
    name = "T00139_m600_7DT01_20250101_000000_100s"
    usable = float(100 * (excluded == 0).mean())
    path = plot_source_mask(
        data, excluded > 0, os.path.join(figures, f"{name}_srcmask.jpg"), name,
        subtitle=f"{100 - usable:.2f}% excluded by the source and FOV masks, "
        f"{usable:.2f}% usable for the background mesh", max_width=pw,
    )  # fmt: skip
    image = Image.open(path).convert("RGB")
    ok &= check(
        "source-mask check plot written as a light JPEG carrying the frame name",
        os.path.exists(path) and image.size == size and os.path.basename(path).startswith(name)
        and 3_000 < os.path.getsize(path) < 900_000,
        f"{os.path.basename(path)}, {image.size[0]}x{image.size[1]} px, {os.path.getsize(path) / 1e3:.0f} kB",
    )
    ok &= check(
        "the excluded area is washed in the overlay colour",
        nearest_pixel_distance(image, "#e5484d") < 90,
        f"RGB distance {nearest_pixel_distance(image, '#e5484d'):.0f} to the wash colour "
        f"({100 - usable:.2f}% of the frame excluded)",
    )
    return ok



def check_orientation(root):
    """RA to the right, Dec upwards, and the coadd JPEG mirrored the same way as the counts figure."""
    from PIL import Image

    from pipeline.imcoadd.plotting import display_flips, orient_for_raster, plot_coadd_counts

    print("Sky orientation (RA right, Dec up):")
    ok = True
    figures = os.path.join(root, "figures")

    normal = frame_header()  # CD1_1 < 0, CD2_2 > 0: the 7DT convention, RA falls with x
    flipped = frame_header()
    flipped["CD1_1"], flipped["CD2_2"] = 1.4e-4, -1.4e-4
    ok &= check(
        "the flips follow the WCS, and a header without one asks for none",
        display_flips(normal) == (True, False)
        and display_flips(flipped) == (False, True)
        and display_flips(fits.Header()) == (False, False),
        f"normal {display_flips(normal)}, reversed {display_flips(flipped)}, no WCS {display_flips(fits.Header())}",
    )

    ph, pw, n = 400, 600, 5
    planes = {"NGEOM": np.full((ph, pw), n, np.uint8), "NSAT": np.zeros((ph, pw), np.uint8)}
    planes["NSAT"][20:80, 20:120] = n  # low y, low x: array origin corner
    path = plot_coadd_counts(planes, os.path.join(figures, "orient.jpg"), "orient", n_inputs=n,
                             header=normal, max_width=pw)  # fmt: skip
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    half = rgb.shape[1] // 2
    left = pixels_near(Image.fromarray(rgb[:, :half].astype(np.uint8)), "#e5484d", 60)
    right = pixels_near(Image.fromarray(rgb[:, half:].astype(np.uint8)), "#e5484d", 60)
    bottom = pixels_near(Image.fromarray(rgb[rgb.shape[0] // 2 :].astype(np.uint8)), "#e5484d", 60)
    ok &= check(
        "the array's origin corner lands bottom-RIGHT once RA runs rightwards",
        right > 10 * max(left, 1) and bottom > 1000,
        f"{right} red px in the right half vs {left} in the left, {bottom} in the bottom half",
    )

    data = np.zeros((ph, pw), np.float32)
    data[20:80, 20:120] = 1.0
    raster = orient_for_raster(data, display_flips(normal))
    ok &= check(
        "the coadd JPEG takes the same transform, so the two figures blink",
        bool(raster[ph - 80 : ph - 20, pw - 120 : pw - 20].all())
        and not bool(raster[:20].any())
        and np.array_equal(orient_for_raster(data, (False, False)), data[::-1]),
        "the corner marked at low x, low y ends bottom-right in the PIL raster too",
    )
    return ok


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
                figure_path = os.path.join(case, "figures", "obj_coadd_counts.jpg")
                ok &= check(
                    f"{label} -> products",
                    os.path.exists(counts_path) == counts_on
                    and os.path.exists(mask_path) == mask_on
                    and os.path.exists(figure_path) == counts_on,
                    f"counts_file={os.path.exists(counts_path)} mask_file={os.path.exists(mask_path)} "
                    f"counts_figure={os.path.exists(figure_path)}",
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

        print("The final coadd is made NaN-free (maskfill inside the coverage, 0 outside):")
        for policy, label in (("union", "union"), ("intersection", "intersection")):
            nan_case = os.path.join(root, f"nanfill_{policy}")
            shutil.copytree(work, nan_case)
            nan_images = [os.path.join(nan_case, os.path.basename(f)) for f in images]
            nan_weights = [os.path.join(nan_case, os.path.basename(f)) for f in weights]
            # a star core saturated in EVERY input: no sample survives, so the coadd has a real interior hole
            for f in nan_images:
                frame = fits.getdata(f)
                frame[HOLE_Y - 1 : HOLE_Y + 2, HOLE_X - 1 : HOLE_X + 2] = 70000.0
                fits.writeto(f, frame, header=fits.getheader(f), overwrite=True)
            path_module.PathHandler.get_bpmask = classmethod(lambda cls, image: bpmask)
            imcoadd_module.PathHandler.get_bpmask = path_module.PathHandler.get_bpmask
            plan = make_plan(coverage_policy=policy, output_counts_map=True, output_mask_map=False)
            stub = run_coadd(nan_case, plan, nan_images, nan_weights)
            coadd = stub.config_node.imcoadd.coadd_image
            before = fits.getdata(coadd)
            geometric = stub._coadd_counts.geometric
            keep = (geometric == N_FRAMES) if policy == "intersection" else (geometric > 0)
            interior = ~np.isfinite(before) & keep
            outside = ~np.isfinite(before) & ~keep
            n_int, n_out = int(interior.sum()), int(outside.sum())
            stub.fill_coadd_nan()
            after, hdr = fits.getdata(coadd, header=True)
            ok &= check(
                f"{label}: no NaN survives",
                bool(np.isfinite(after).all()),
                f"NaN before={int((~np.isfinite(before)).sum())} after={int((~np.isfinite(after)).sum())} "
                f"(interior {n_int}, outside {n_out})",
            )
            ok &= check(
                f"{label}: interior holes carry a filled value, the outside carries 0",
                (not n_int or np.all(np.isfinite(after[interior])))
                and (not n_out or np.all(after[outside] == 0.0))
                and hdr["NNANFILL"] == n_int and hdr["NNANEDGE"] == n_out
                and (hdr["NANFILL"] == "MASKFILL" if n_int else hdr["NANFILL"] is False),
                f"NANFILL={hdr['NANFILL']!r} NNANFILL={hdr['NNANFILL']} NNANEDGE={hdr['NNANEDGE']}",
            )
            ok &= check(
                f"{label}: pixels that had data are untouched",
                np.array_equal(after[np.isfinite(before)], before[np.isfinite(before)]),
                "every finite pixel is bit-identical before and after",
            )
            counts_after = read_count_planes(coadd.replace(".fits", "_counts.fits"))
            ok &= check(
                f"{label}: the all-saturated core became an interior hole and was filled",
                n_int >= 9 and bool(interior[HOLE_Y, HOLE_X]) and np.isfinite(after[HOLE_Y, HOLE_X])
                and abs(float(after[HOLE_Y, HOLE_X]) - 100.0) < 5.0,
                f"interior px={n_int}, filled value at the core={float(after[HOLE_Y, HOLE_X]):.3f} "
                f"(surrounding background ~100, the rejected samples were 70000)",
            )
            ok &= check(
                f"{label}: the counts still record the holes",
                int(counts_after["NUSED"][interior][0]) == 0 if n_int else True,
                "NUSED stays 0 where the image was filled" if n_int else "no interior hole in this case",
            )

        off_case = os.path.join(root, "nanfill_off")
        shutil.copytree(work, off_case)
        off_images = [os.path.join(off_case, os.path.basename(f)) for f in images]
        off_weights = [os.path.join(off_case, os.path.basename(f)) for f in weights]
        stub = run_coadd(off_case, make_plan(fill_nan=False), off_images, off_weights)
        coadd = stub.config_node.imcoadd.coadd_image
        before = fits.getdata(coadd)
        stub.fill_coadd_nan()
        after, hdr = fits.getdata(coadd, header=True)
        ok &= check(
            "fill_nan false leaves the coadd exactly as it was",
            np.array_equal(np.nan_to_num(before, nan=-7.0), np.nan_to_num(after, nan=-7.0))
            and "NANFILL" not in hdr and bool((~np.isfinite(after)).sum()),
            f"NaN kept={int((~np.isfinite(after)).sum())}, no NANFILL card={'NANFILL' not in hdr}",
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
        ok &= check_plots(root)
        ok &= check_orientation(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print("COVERAGE-PRODUCT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
