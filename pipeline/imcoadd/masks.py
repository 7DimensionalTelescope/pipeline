from __future__ import annotations
import os
import threading
from typing import TYPE_CHECKING

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from ..config.utils import get_key
from ..path.path import PathHandler
from ..services.logger import Logger
from ..utils import atleast_1d, get_basename
from .badpix import ProjectedBadPixels, detector_badpixels, nearest_output_pixels, project_badpixels
from .coadd_plan import CoaddPlan
from .storage import IntermediateStorage
from .const import MASK_HEADER_CARDS, MaskBit
from .counts import CoaddCounts
from .plotting import plot_coadd_counts
from .source_mask import line_profile, core_width, optimized_half_width_for_trails_with_known_profile
from .utils import count_dtype, determine_size, write_count_planes, write_mask_plio
from .header_set import InputHeaderSet


if TYPE_CHECKING:
    from ..config._crossfilter_stubs import CrossFilterNode
    from ..config._sciproc_stubs import SciProcNode

    ConfigNodeT = SciProcNode | CrossFilterNode  # ImCoadd runs on the first, WhiteImage on the second


def _robust_stats(image: np.ndarray, stride: int = 8) -> tuple[float, float]:
    sample = np.asarray(image[::stride, ::stride], dtype=np.float32).ravel()
    sample = sample[np.isfinite(sample)]
    if not sample.size:
        return 0.0, 1.0
    median = float(np.median(sample))
    sigma = float(1.4826 * np.median(np.abs(sample - median)))
    return median, max(sigma, 1e-6)


def _block_mean(image: np.ndarray, factor: int) -> np.ndarray:
    h = image.shape[0] // factor * factor
    w = image.shape[1] // factor * factor
    view = image[:h, :w].reshape(h // factor, factor, w // factor, factor)
    count = np.isfinite(view).sum(axis=(1, 3))
    total = np.nansum(view, axis=(1, 3), dtype=np.float32)
    return np.divide(
        total,
        count,
        out=np.full(total.shape, np.nan, dtype=np.float32),
        where=count > 0,
    )


def _line_polar(segment) -> tuple[float, float]:
    x1, y1, x2, y2 = segment
    theta = np.arctan2(y2 - y1, x2 - x1) % np.pi
    return theta, x1 * np.sin(theta) - y1 * np.cos(theta)


def _merge_segments(segments, angle_limit: float, distance_limit: float):
    lengths = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])
    clusters = []
    for i in np.argsort(-lengths):
        segment = segments[i]
        theta0, rho0 = _line_polar(segment)
        weight = float(lengths[i])
        for cluster in clusters:
            theta, rho = theta0, rho0
            delta = theta - cluster[0] / cluster[2]
            if abs(delta) > np.pi / 2:
                sign = np.sign(delta)
                theta -= sign * np.pi
                rho = -rho
                delta -= sign * np.pi
            if abs(delta) <= angle_limit and abs(rho - cluster[1] / cluster[2]) <= distance_limit:
                cluster[0] += weight * theta
                cluster[1] += weight * rho
                cluster[2] += weight
                cluster[3].append(segment)
                break
        else:
            clusters.append([weight * theta0, weight * rho0, weight, [segment]])
    return clusters


def _measure_trail_width(
    image,
    segment,
    noise,
    factor,
    members=None,
    distance_limit=0.0,
    padding=0.0,
    skysig=None,
    psf_fwhm=None,
    threshold_scale=1.0,
    aperture_sigma=1.0,
    profile_percentile=50.0,
    min_half_width=2.0,
    max_half_width=24.0,
):
    """Perpendicular-profile components of a merged cluster, one (segment, half_width, flux_per_length) per trail."""
    from scipy.ndimage import gaussian_filter1d, map_coordinates, uniform_filter1d

    minimum = float(min_half_width) / factor
    maximum = float(max_half_width) / factor
    if minimum < 0 or maximum < minimum:
        raise ValueError("satellite_mask half-width limits are invalid")
    if not 0 <= profile_percentile <= 100:
        raise ValueError("profile_percentile must be between 0 and 100")
    if aperture_sigma <= 0:
        raise ValueError("aperture_sigma must be positive")
    if threshold_scale <= 0:
        raise ValueError("satellite_mask.threshold_scale must be positive")
    p0, p1 = np.asarray(segment[:2]), np.asarray(segment[2:])
    direction = p1 - p0
    length = float(np.hypot(*direction))
    if length == 0:
        return [(np.asarray(segment), minimum, 0.0)]
    direction /= length
    normal = np.array([-direction[1], direction[0]])
    reach = float(distance_limit) + maximum
    radius = reach + max(3.0, 8.0 / factor)
    step = 0.25
    offsets = np.arange(-radius, radius + step / 2, step)
    samples = max(32, min(1024, int(np.ceil(length * 2))))
    along = np.linspace(0.05, 0.95, samples) * length
    points = p0[None, :] + along[:, None] * direction
    coordinates = points[None, :, :] + offsets[:, None, None] * normal
    values = map_coordinates(
        image,
        [coordinates[..., 1], coordinates[..., 0]],
        order=1,
        mode="constant",
        cval=np.nan,
    )
    outer = np.abs(offsets) >= reach + 1.0
    threshold = aperture_sigma * noise
    law = bool(skysig) and bool(psf_fwhm)
    if law:
        awin = float(psf_fwhm) / np.sqrt(8.0 * np.log(2.0))
        alpha = float(core_width(awin))

    def binned_model(center):
        """Unit-flux line-spread profile centred on the run, after the block mean, interpolation and smoothing."""
        model = line_profile(np.abs(offsets - center) * factor, alpha)
        model = gaussian_filter1d(uniform_filter1d(uniform_filter1d(model, 4), 4), 0.75)
        return model - float(np.median(model[outer]))  # the same baseline removal the data profile gets

    def component(columns, target):
        """Above-threshold component around target offset (None: the strongest within the cluster)."""
        profile = np.nanpercentile(values[:, columns], profile_percentile, axis=1)
        profile = gaussian_filter1d(profile, 0.75, mode="nearest")
        # the lower side is the baseline: a parallel trail can sit on the other side's annulus
        signal = profile - min(
            float(np.nanmedian(profile[outer & (offsets < 0)])), float(np.nanmedian(profile[outer & (offsets > 0)]))
        )
        window = np.abs(offsets) <= maximum if target is None else np.abs(offsets - target) <= max(1.0, minimum)
        peak = int(np.nanargmax(np.where(window, signal, np.nan)))
        above = signal >= threshold
        if not above[peak]:
            if target is not None:
                return None
            return (float(offsets[peak]) if signal[peak] > 0 else 0.0), minimum, 0.0
        left = right = peak
        while left > 0 and above[left - 1]:
            left -= 1
        while right + 1 < len(above) and above[right + 1]:
            right += 1
        center = 0.5 * float(offsets[left] + offsets[right])
        half_width = 0.5 * float(offsets[right] - offsets[left]) + step / 2
        if half_width > maximum:
            center = float(offsets[peak])  # capped mask stays on the ridge, not a lopsided run's midpoint
        half_width = min(maximum, max(minimum, half_width))
        if not law:
            return center, half_width, 0.0
        pad = int(round(8.0 / factor / step))  # 8 detector px beyond the run, where the smoothing spread the core flux
        inside = slice(max(0, left - pad), min(len(offsets), right + 1 + pad))
        flux_per_length = float(np.nansum(signal[inside])) * step * factor
        model_fraction = float(np.sum(binned_model(center)[inside])) * step * factor  # flux outside the aperture
        flux_per_length /= max(model_fraction, 1e-3)
        law_half_width = (
            float(optimized_half_width_for_trails_with_known_profile(flux_per_length, awin, skysig, threshold_scale))
            / factor
        )
        return center, max(half_width, law_half_width), flux_per_length

    found = [(*component(slice(None), None), 0.0, length)]
    for member in np.asarray(members if members is not None else []).reshape(-1, 2, 2):
        # a member is one Hough edge: its own along-track range keeps a shorter trail visible
        relative = member - p0
        track = relative @ direction
        t0, t1 = max(0.0, float(track.min()) - padding), min(length, float(track.max()) + padding)
        columns = (along >= t0) & (along <= t1)
        if columns.sum() < 8:
            continue
        hit = component(columns, float(np.mean(relative @ normal)))
        if hit is not None:
            found.append((*hit, t0, t1))
    merged = []
    for center, half_width, flux_per_length, t0, t1 in sorted(found):
        if merged and abs(center - merged[-1][0]) <= 0.5 and abs(half_width - merged[-1][1]) <= 0.5:
            last = merged[-1]
            merged[-1] = (
                last[0],
                max(last[1], half_width),
                max(last[2], flux_per_length),
                min(last[3], t0),
                max(last[4], t1),
            )
        else:
            merged.append((center, half_width, flux_per_length, t0, t1))
    return [
        (
            np.r_[p0 + t0 * direction + center * normal, p0 + t1 * direction + center * normal],
            half_width,
            flux_per_length,
        )
        for center, half_width, flux_per_length, t0, t1 in merged
    ]


def detect_satellite_trails(image: np.ndarray, skysig=None, psf_fwhm=None, **options) -> tuple[np.ndarray, np.ndarray]:
    """Trail mask and one (x1, y1, x2, y2, half_width_px, flux_per_length) row per masked component."""
    from PIL import Image, ImageDraw
    from scipy.ndimage import uniform_filter
    from skimage.feature import canny
    from skimage.transform import probabilistic_hough_line

    factor = int(options.get("bin_factor", 4))
    if factor not in (1, 2, 4, 8):
        raise ValueError("satellite_mask.bin_factor must be one of 1, 2, 4, or 8")
    image = np.asarray(image, dtype=np.float32)
    valid = np.isfinite(image) & (image != 0)
    small = _block_mean(np.where(valid, image, np.nan), factor)
    median, sigma = _robust_stats(small)
    filled = np.where(np.isfinite(small), small, median)
    clipped = np.minimum(filled, median + float(options.get("star_clip_sigma", 3.0)) * sigma)
    box = int(options.get("background_box", 65))
    background = uniform_filter(clipped, size=box, mode="nearest")
    flat = filled - background
    _, flat_sigma = _robust_stats(flat)
    low = float(options.get("low_sigma", 2.0)) * flat_sigma
    high = float(options.get("high_sigma", 8.0)) * flat_sigma
    stretched = np.clip((flat - low) / max(high - low, 1e-6), 0, 1)
    edges = canny(
        stretched,
        sigma=float(options.get("canny_sigma", 1.0)),
        low_threshold=float(options.get("canny_low_threshold", 40 / 255)),
        high_threshold=float(options.get("canny_high_threshold", 120 / 255)),
    )
    theta_step = float(options.get("theta_step", 0.25))
    if theta_step <= 0:
        raise ValueError("satellite_mask.theta_step must be positive")
    theta = np.deg2rad(np.arange(-90.0, 90.0, theta_step))
    diagonal = float(np.hypot(*small.shape))
    raw = probabilistic_hough_line(
        edges,
        threshold=int(options.get("hough_threshold", 50)),
        line_length=max(2, int(float(options.get("min_length_fraction", 0.08)) * diagonal)),
        line_gap=int(options.get("max_gap", 20)),
        theta=theta,
        rng=np.random.default_rng(0),
    )
    segments = np.asarray([(p0[0], p0[1], p1[0], p1[1]) for p0, p1 in raw], dtype=np.float64).reshape(-1, 4)
    accepted = []
    if segments.size:
        clusters = _merge_segments(
            segments,
            np.deg2rad(float(options.get("merge_angle", 1.0))),
            float(options.get("merge_distance", 12.0)),
        )
        support_limit = float(options.get("min_support_fraction", 0.08)) * diagonal
        padding = float(options.get("endpoint_padding", 8.0)) / factor
        for theta_sum, rho_sum, support, members in clusters:
            if support < support_limit:
                continue
            points = np.asarray(members).reshape(-1, 2)
            centered = points - points.mean(axis=0)
            _, _, axes = np.linalg.svd(centered, full_matrices=False)
            direction = axes[0]
            if direction[0] < 0:
                direction = -direction
            normal = np.array([direction[1], -direction[0]])
            point = normal * np.mean(points @ normal)
            projection = (points - point) @ direction
            t0 = projection.min() - padding
            t1 = projection.max() + padding
            segment = np.r_[point + t0 * direction, point + t1 * direction]
            accepted.extend(
                _measure_trail_width(
                    filled - median,  # the box background still holds the trail, by a chord that depends on its angle
                    segment,
                    flat_sigma,
                    factor,
                    members=members,
                    distance_limit=float(options.get("merge_distance", 12.0)),
                    padding=padding,
                    skysig=skysig,
                    psf_fwhm=psf_fwhm,
                    threshold_scale=float(options.get("threshold_scale", 1.0)),
                )
            )

    canvas = Image.new("1", (image.shape[1], image.shape[0]))
    draw = ImageDraw.Draw(canvas)
    center_offset = (factor - 1) / 2
    scaled = []
    for segment, half_width, flux_per_length in accepted:
        x1, y1, x2, y2 = segment * factor + center_offset
        radius = max(1, int(np.ceil(half_width * factor + 1.0)))
        draw.line((x1, y1, x2, y2), fill=1, width=2 * radius + 1)
        for x, y in ((x1, y1), (x2, y2)):
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=1)
        scaled.append((x1, y1, x2, y2, half_width * factor, flux_per_length))
    mask = np.asarray(canvas, dtype=bool).copy()
    mask &= valid
    lines = np.asarray(scaled, dtype=np.float64).reshape(-1, 6)
    return mask, lines


class CoaddMaskBuilder:
    def __init__(self, images, output_path, match_swarp_size, keep_frames, counts=None):
        self.images = list(images)
        target_w, target_h, _, _, self.x0, self.y0, self.shapes = determine_size(self.images, match_swarp_size)
        self.output_path = output_path
        self.output = np.zeros((target_h, target_w), dtype=np.uint8)
        self.frames = [None] * len(self.images)
        self.headers = [None] * len(self.images)
        self.keep_frames = keep_frames
        self.counts = CoaddCounts() if counts is None else counts
        self.counts.allocate((target_h, target_w), count_dtype(len(self.images)))

    def set_frame(self, index, mask, header=None):
        self.frames[index] = mask
        if header is not None:
            self.headers[index] = header.copy()
        h, w = mask.shape
        tx0 = max(0, self.x0[index])
        tx1 = min(self.output.shape[1], self.x0[index] + w)
        ty0 = max(0, self.y0[index])
        ty1 = min(self.output.shape[0], self.y0[index] + h)
        if tx1 <= tx0 or ty1 <= ty0:
            return
        sx0, sx1 = tx0 - self.x0[index], tx1 - self.x0[index]
        sy0, sy1 = ty0 - self.y0[index], ty1 - self.y0[index]
        block = mask[sy0:sy1, sx0:sx1]
        self.output[ty0:ty1, tx0:tx1] |= block
        for plane, bit in self.counts.bit_planes():
            plane[ty0:ty1, tx0:tx1] += (block & int(bit)) > 0

    def mark_outliers(self, index, geometry, rejected):
        tx0, tx1, ty0, ty1, sx0, sx1, sy0, sy1 = geometry
        if rejected.shape != (ty1 - ty0, tx1 - tx0) or tx1 > self.output.shape[1] or ty1 > self.output.shape[0]:
            # the coadd backend and the mask builder size their grids separately; they must agree
            raise ValueError(f"outlier geometry {geometry} does not fit the mask grid {self.output.shape}")
        target = self.output[ty0:ty1, tx0:tx1]
        target[rejected] |= int(MaskBit.OUTLIER)
        self.counts.outlier[ty0:ty1, tx0:tx1] += rejected
        if self.keep_frames:
            frame = self.frame(index)
            local = frame[sy0:sy1, sx0:sx1]
            local[rejected] |= int(MaskBit.OUTLIER)
            self.frames[index] = frame

    def frame(self, index):
        value = self.frames[index]
        if isinstance(value, str):
            return fits.getdata(value, memmap=False).astype(np.uint8, copy=False)
        return value

    def write_frames(self):
        for index, (image, stored) in enumerate(zip(self.images, self.frames)):
            frame_header = self.headers[index] if self.headers[index] is not None else fits.getheader(image)
            frame_header["BUNIT"] = "bitmask"
            for key, value in MASK_HEADER_CARDS.items():
                frame_header[key] = value
            write_mask_plio(
                PathHandler.mask(image),
                self.frame(index) if isinstance(stored, str) else stored,
                header=frame_header,
            )

    def write(self, dump_frames=False):
        header = fits.getheader(self.output_path)
        header["BUNIT"] = "bitmask"
        for key, value in MASK_HEADER_CARDS.items():
            header[key] = value
        mask_path = PathHandler.mask(self.output_path)
        write_mask_plio(mask_path, self.output, header=header)
        if dump_frames:
            self.write_frames()
        return mask_path


class MaskMixin:
    _saturation_map_lock = threading.Lock()  # the fused loop records saturation from its writer threads

    config_node: ConfigNodeT
    logger: Logger
    path: PathHandler
    plan: CoaddPlan
    storage: IntermediateStorage
    intermediate_storage: IntermediateStorage | None
    input_images: list[str]
    input_headers: InputHeaderSet
    _has_detector_bpm: bool
    _coadd_counts: CoaddCounts
    _zdf_cache: dict[str, tuple[str, str, str]]
    _coadd_mask_builder: CoaddMaskBuilder | None
    _quality_masks: list | None
    _badpix_positions_cache: dict[str, ProjectedBadPixels]
    _saturated_positions_cache: dict[str, ProjectedBadPixels]
    _saturation_map_cache: dict[tuple, tuple]
    _bpmask_coords_cache: dict[str, tuple]

    @staticmethod
    def _project_pixels(xs, ys, input_header, output_header, shape):
        if not len(xs):
            return np.zeros(shape, dtype=bool)
        ra, dec = WCS(input_header).all_pix2world(xs.astype(np.float64), ys.astype(np.float64), 0)
        return MaskMixin._project_sky(ra, dec, output_header, shape)

    @staticmethod
    def _project_sky(ra, dec, output_header, shape):
        """Nearest output pixel of each sky position (one-to-one, no kernel dilation)."""
        output = np.zeros(shape, dtype=bool)
        yi, xi = nearest_output_pixels(ra, dec, output_header, shape)
        output[yi, xi] = True
        return output

    def _badpix_positions(self, detector_image, output_header=None, output_shape=None) -> ProjectedBadPixels | None:
        """One frame's detector bad pixels projected onto its output grid, computed once per frame."""
        if not self._has_detector_bpm:
            return None
        cached = self._badpix_positions_cache.get(detector_image)
        if cached is not None:
            if output_shape is not None and tuple(cached.shape) != tuple(output_shape):
                raise ValueError(
                    f"Projected bad pixels of {get_basename(detector_image)} were built on a "
                    f"{cached.shape} grid but {tuple(output_shape)} was asked for"
                )
            return cached
        if output_header is None:
            raise ValueError(f"No output grid to project the bad pixels of {get_basename(detector_image)} onto")
        if output_shape is None:
            output_shape = (int(output_header["NAXIS2"]), int(output_header["NAXIS1"]))
        mask_file, badpix, _ = self._bpmask_info(detector_image)
        coords = self._bpmask_coords_cache.get(mask_file)
        if coords is None:
            coords = detector_badpixels(mask_file, badpix)
            self._bpmask_coords_cache[mask_file] = coords
        positions = project_badpixels(
            *coords,
            self._single_wcs_header(detector_image),
            output_header,
            output_shape,
            footprint=self.plan.badpix_propagation_policy_across_astrometric_reprojection,
        )
        self._badpix_positions_cache[detector_image] = positions
        return positions

    def badpix_positions(self, images, detector_images=None) -> list | None:
        """Sparse projected bad pixels aligned with *images*, for the policy that has no file channel.

        Only `exclude_badpix_by_projected_index` needs them; legacy alone carries its zeros in the weight file."""
        if not (self.plan.exclude_badpix_by_projected_index and self._has_detector_bpm):
            return None
        images = list(atleast_1d(images))
        detector_images = list(atleast_1d(detector_images if detector_images is not None else self.input_images))
        if len(images) != len(detector_images):
            raise ValueError("coadd images and detector images differ in length")
        positions = []
        for image, detector in zip(images, detector_images):
            cached = self._badpix_positions_cache.get(detector)
            header = None
            if cached is None:
                header = fits.getheader(image)
                cached = self._badpix_positions(detector, header)
            else:
                header = self._frame_shape_header(image)
            shape = (int(header["NAXIS2"]), int(header["NAXIS1"]))
            if cached.shape != shape:
                raise ValueError(
                    f"Projected bad pixels of {get_basename(detector)} were built on a "
                    f"{cached.shape} grid but {get_basename(image)} is {shape}"
                )
            positions.append(cached)
        self.logger.info(
            f"Projected bad pixels: {sum(p.size for p in positions)} output positions over "
            f"{len(positions)} frames ({sum(p.nbytes for p in positions) / 1e6:.0f} MB sparse, "
            "no resampled weight rewritten)"
        )
        return positions

    def saturated_positions(self, images, detector_images=None, compute: bool = True) -> list | None:
        """Sparse projected saturated pixels aligned with *images*: those samples do not enter the estimator.

        `compute` False returns them only when every frame is already cached, so a mode that cannot drop a
        sample (proper) never pays a full read pass just to discount its weight response."""
        images = list(atleast_1d(images))
        detector_images = list(atleast_1d(detector_images if detector_images is not None else self.input_images))
        if len(images) != len(detector_images):
            raise ValueError("coadd images and detector images differ in length")
        positions = []
        for image, detector in zip(images, detector_images):
            cached = self._saturated_positions_cache.get(detector)
            if cached is None:
                if not compute:
                    return None
                header = self._frame_shape_header(image)
                cached = self._saturated_positions(detector, header, (int(header["NAXIS2"]), int(header["NAXIS1"])))
            positions.append(cached)
        total = sum(p.size for p in positions)
        if not total:
            return None
        self.logger.info(
            f"Saturated pixels: {total} output positions over {len(positions)} frames "
            f"({sum(p.nbytes for p in positions) / 1e6:.0f} MB sparse), excluded from the estimator"
        )
        return positions

    def _frame_shape_header(self, image):
        storage = self.intermediate_storage  # may run before the storage is prepared
        cached = storage.frame_cache.get(image) if storage is not None else None
        return cached[1] if cached is not None else fits.getheader(image)

    @property
    def _need_quality_masks(self) -> bool:
        """Whether this run must build the per-frame bit masks, for the OR product or for a reason plane."""
        reason_planes = self._has_detector_bpm or self.plan.coadd_mode == "clipped"  # NBAD/NSAT, or NOUTLIER
        return self.plan.build_per_frame_quality_masks or (self.plan.output_counts_map and reason_planes)

    def _imcmb_key(self, detector_image):
        """This frame's (bias, dark, flat) master triple, or None when the header cannot give one."""
        from ..preprocess.utils import get_zdf_from_header_IMCMB

        cache = self._zdf_cache
        try:
            if detector_image not in cache:
                cache[detector_image] = get_zdf_from_header_IMCMB(detector_image)
            return tuple(cache[detector_image])
        except Exception:
            return None

    def _master_flat_of(self, detector_image, key) -> str | None:
        """The master flat of an IMCMB triple; the caller caches per triple, so this runs once per group."""
        if key is None:
            self.logger.warning(f"No IMCMB masters on {get_basename(detector_image)}; scalar SATURATE")
            return None
        try:
            return PathHandler.resolve_weight_map_input_abspath(list(key))[1]
        except Exception as e:
            self.logger.warning(f"No master flat for {get_basename(detector_image)} ({e}); scalar SATURATE")
            return None

    def _saturation_map(self, detector_image, saturation, shape) -> tuple:
        """Per-pixel saturation ceiling and the identity of the flat behind it, cached per IMCMB group.

        `SATURATE` is `(2**BITPIX - 1 - CLIPMED_bias - CLIPMED_dark) / CENCLPMD_flat`: a scalar whose only
        field-dependent term is the flat, taken at its CENTRAL clipped median. Dividing by the per-pixel flat
        instead restores that term, `satur(x, y) = SATURATE * CENCLPMD / flat(x, y)`, while bias and dark stay
        scalar because they are flat to a few ADU across the field."""
        key = (self._imcmb_key(detector_image), float(saturation), tuple(shape))
        cached = self._saturation_map_cache.get(key)
        if cached is not None:
            return cached
        with self._saturation_map_lock:
            cached = self._saturation_map_cache.get(key)
            if cached is not None:
                return cached
            from ..preprocess.utils import get_image_id

            flat_file = self._master_flat_of(detector_image, key[0])
            if flat_file is None:
                self._saturation_map_cache[key] = (float(saturation), "")
                return self._saturation_map_cache[key]
            try:
                flat, flat_header = fits.getdata(flat_file, header=True, memmap=False)
                central = float(flat_header["CENCLPMD"])
                flat = np.asarray(flat, dtype=np.float32)
                usable = np.isfinite(flat) & (flat > 0)
                level = np.full(flat.shape, np.inf, dtype=np.float32)
                np.divide(np.float32(float(saturation) * central), flat, out=level, where=usable)
            except (OSError, KeyError, TypeError, ValueError) as e:
                self.logger.warning(f"Unusable master flat {get_basename(flat_file)} ({e}); scalar SATURATE")
                value = (float(saturation), "")
            else:
                if level.shape != tuple(shape):
                    self.logger.warning(
                        f"Master flat {level.shape} does not match {get_basename(detector_image)} "
                        f"{tuple(shape)}; scalar SATURATE"
                    )
                    level = float(saturation)
                value = (level, str(get_image_id(flat_file) or get_basename(flat_file)))
                (
                    self.logger.info(
                        f"Saturation map from {get_basename(flat_file)}: {float(saturation):.0f} at the flat's "
                        f"central median, {float(np.nanmin(level[usable])):.0f} to "
                        f"{float(np.nanmax(level[usable])):.0f} across the field ({level.nbytes / 1e6:.0f} MB)"
                    )
                    if not np.isscalar(level)
                    else None
                )
            self._saturation_map_cache[key] = value
            return value

    def _saturated_detector_mask(self, detector_image, data, header):
        """Detector pixels at or above this frame's saturation ceiling, on the detector grid."""
        saturation = header.get("SATURATE")
        if saturation is None:
            return None
        level, _ = self._saturation_map(detector_image, saturation, data.shape)
        return np.isfinite(data) & (data >= level)

    def _saturated_catalog(self, detector_image) -> str:
        factory = self.path.imcoadd.factory
        return factory.stage_images([detector_image], "satpix", factory.mask_dir)[0]

    def _record_saturated_pixels(self, detector_image, data, header) -> None:
        """Sky positions of the pixels at or above SATURATE, from the frame already in memory."""
        saturation = header.get("SATURATE")
        if saturation is None:
            return
        level, flat_id = self._saturation_map(detector_image, saturation, data.shape)
        ys, xs = np.nonzero(np.isfinite(data) & (data >= level))
        wcs = WCS(self._single_wcs_header(detector_image, header))
        ra, dec = wcs.all_pix2world(xs.astype(np.float64), ys.astype(np.float64), 0)
        table = fits.BinTableHDU.from_columns(
            [fits.Column(name="RA", format="D", array=ra), fits.Column(name="DEC", format="D", array=dec)]
        )
        table.header["SATURATE"] = (float(saturation), "Saturation level of the source frame")
        table.header["SATURFLT"] = (flat_id, "Master flat shaping the per-pixel saturation ceiling")
        table.header["NSATPIX"] = (int(len(xs)), "Number of saturated pixels")
        table.header["JOINTWCS"] = (bool(self.plan.joint_wcs), "Sky positions from the joint WCS")
        path = self._saturated_catalog(detector_image)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        table.writeto(path, overwrite=True)

    @staticmethod
    def _saturated_catalog_current(catalog, detector_image, saturation, joint_wcs: bool, flat_id: str = "") -> bool:
        """A catalog no older than its frame, at the same SATURATE level and from the same master flat."""
        try:
            if os.path.getmtime(catalog) < os.path.getmtime(detector_image):
                return False
            header = fits.getheader(catalog, 1)
            return (
                header.get("SATURATE") == float(saturation)
                and bool(header.get("JOINTWCS", False)) == joint_wcs
                and str(header.get("SATURFLT", "") or "") == str(flat_id or "")
            )
        except OSError:
            return False

    def _saturated_positions(self, detector_image, output_header, output_shape, detector_data=None):
        """One frame's saturated detector pixels on its output grid, the single source of NSAT and its rejection."""
        cached = self._saturated_positions_cache.get(detector_image)
        if cached is not None:
            if tuple(cached.shape) != tuple(output_shape):
                raise ValueError(
                    f"Saturated pixels of {get_basename(detector_image)} were built on a "
                    f"{cached.shape} grid but {tuple(output_shape)} was asked for"
                )
            return cached
        shape = (int(output_shape[0]), int(output_shape[1]))
        input_header = self._single_wcs_header(detector_image)
        saturation = input_header.get("SATURATE")
        empty = ProjectedBadPixels(index=np.empty(0, dtype=np.int32), shape=shape)
        if saturation is None:
            return empty
        catalog = self._saturated_catalog(detector_image)
        # NAXIS survives _single_wcs_header (strip_wcs does not touch it), so the detector shape is free here
        detector_shape = (
            detector_data.shape
            if detector_data is not None
            else (int(input_header["NAXIS2"]), int(input_header["NAXIS1"]))
        )
        level, flat_id = self._saturation_map(detector_image, saturation, detector_shape)
        if detector_data is None and self._saturated_catalog_current(
            catalog, detector_image, saturation, self.plan.joint_wcs, flat_id
        ):
            table = fits.getdata(catalog)
            yi, xi = nearest_output_pixels(table["RA"], table["DEC"], output_header, shape)
        else:
            if detector_data is None:
                detector_data = fits.getdata(detector_image, memmap=False)
            ys, xs = np.nonzero(np.isfinite(detector_data) & (detector_data >= level))
            if detector_data.shape == shape:
                yi, xi = ys, xs  # already on the output grid: no projection to do
            else:
                positions = project_badpixels(ys, xs, input_header, output_header, shape)
                yi, xi = positions.rows(0, shape[0])
        positions = ProjectedBadPixels(
            index=np.unique(np.asarray(yi, np.int64) * shape[1] + np.asarray(xi, np.int64)).astype(np.int32),
            shape=shape,
        )
        self._saturated_positions_cache[detector_image] = positions
        return positions

    def _saturated_from_weight(self, weight_image, frame_data):
        """Where SWarp's resampled weight is zero inside the frame's own footprint.

        This is the kernel-dilated saturation the estimator rejects, taken from SWarp's output rather than
        redilated here, so the NSAT plane and the rejection cannot disagree."""
        weight, _ = self._read_stage_frame(weight_image)
        if weight.shape != frame_data.shape:
            self.logger.warning(
                f"Resampled weight {get_basename(weight_image)} does not match its frame; NSAT not set from it"
            )
            return None
        return (weight == 0) & (frame_data != 0)

    def _detector_mask_bits(self, detector_image, output_header, output_shape, detector_data=None,
                            weight_image=None, frame_data=None):  # fmt: skip
        output = np.zeros(output_shape, dtype=np.uint8)
        positions = self._badpix_positions(detector_image, output_header, output_shape)
        if positions is not None:
            output[positions.block_mask(0, output_shape[0], 0, output_shape[1])] |= int(MaskBit.BADPIX)
        saturated = None
        if weight_image is not None and self.plan.nsat_from_resampled_weight and frame_data is not None:
            saturated = self._saturated_from_weight(weight_image, frame_data)
        if saturated is not None:
            output[saturated] |= int(MaskBit.SATURATED)
            return output
        # Superseded on 2026-09-10 wherever the resampled weight carries saturation, and kept for the routines
        # that have no such weight (direct, legacy) and for reference: the 1-pixel nearest-output projection.
        projected = self._saturated_positions(detector_image, output_header, output_shape, detector_data)
        if projected.size:
            output[projected.block_mask(0, output_shape[0], 0, output_shape[1])] |= int(MaskBit.SATURATED)
        return output

    def prepare_quality_masks(self, images, detector_images=None, weight_images=None):
        images = list(atleast_1d(images))
        detector_images = list(atleast_1d(detector_images or self.input_images))
        if len(images) != len(detector_images):
            raise ValueError("quality-mask images and detector images differ in length")
        weights = list(atleast_1d(weight_images or get_key(self.config_node.imcoadd, "bkgsub_weight_images") or []))
        if len(weights) != len(images):
            weights = [None] * len(images)
        builder = CoaddMaskBuilder(
            images,
            self.config_node.imcoadd.coadd_image,
            self.plan.match_swarp_size,
            self.plan.dump_reprojected_masks,
            counts=self._coadd_counts,
        )
        satellite_options = self.config_node.imcoadd.satellite_mask
        skysigs = self.input_headers.values_any("BACKSIG", "SKYSIG")
        psf_fwhms = self.input_headers.values("PEEING")
        quality_masks = []
        trailed_frames = 0
        for index, (image, detector) in enumerate(zip(images, detector_images)):
            data, header = self._read_stage_frame(image)
            same_file = os.path.abspath(image) == os.path.abspath(detector)
            mask = self._detector_mask_bits(
                detector, header, data.shape, detector_data=data if same_file else None,
                weight_image=weights[index], frame_data=data,
            )  # fmt: skip
            if self.plan.satellite_mask_enabled:
                if not skysigs[index] or not psf_fwhms[index]:
                    self.logger.warning(
                        f"No sky noise or PEEING for {get_basename(image)}; trail width from the profile alone"
                    )
                trail, lines = detect_satellite_trails(
                    data, skysig=skysigs[index], psf_fwhm=psf_fwhms[index], **satellite_options
                )
                mask[trail] |= int(MaskBit.SATELLITE)
                trailed_frames += int(trail.any())
                widths = ", half-width " + "/".join(f"{w:.0f}" for w in lines[:, 4]) + " px" if len(lines) else ""
                self.logger.info(
                    f"Satellite mask: {len(lines)} line(s){widths}, {int(trail.sum())} pixels in {get_basename(image)}"
                )
            builder.set_frame(index, mask, header=header)
            if self.storage.policy == "memory":
                quality_masks.append(mask)
            else:
                path = self.path.imcoadd.factory.stage_images([image], "mask", self.path.imcoadd.factory.mask_dir)[0]
                write_mask_plio(path, mask, header=header)
                self.storage.working_mask_paths.append(path)
                quality_masks.append(path)
        builder.frames = quality_masks
        if self.plan.satellite_mask_enabled:  # absent card = never evaluated, 0 = evaluated and clear
            self.input_headers.run_cards["NTRAILIM"] = (
                trailed_frames,
                "Inputs with satellite-trail pixels masked",
            )
        self._coadd_mask_builder = builder
        self._quality_masks = quality_masks
        return quality_masks

    def quality_mask(self, index):
        value = self._quality_masks[index]
        if isinstance(value, str):
            return fits.getdata(value, memmap=False).astype(np.uint8, copy=False)
        return value

    def finalize_quality_masks(self):
        """Write the coadd companions the switches ask for: the count planes, the OR bitmask, the frame masks."""
        self.write_coadd_counts()
        builder = self._coadd_mask_builder
        if builder is None:
            return None
        if self.plan.output_mask_map:
            path = builder.write(dump_frames=self.plan.dump_reprojected_masks)
            self.config_node.imcoadd.coadd_mask_image = path
            self.logger.info(f"Coadd mask map saved as {path}")
            return path
        if self.plan.dump_reprojected_masks:
            builder.write_frames()
        return None

    def write_coadd_counts(self) -> str | None:
        """Canonical coverage/provenance product: one integer count plane per reason."""
        if not self.plan.output_counts_map:
            return None
        coadd_image = self.config_node.imcoadd.coadd_image
        if not (coadd_image and os.path.exists(coadd_image)):
            self.logger.warning("No coadd image to describe; the coverage count product is not written")
            return None
        n_inputs = len(atleast_1d(self.input_images))
        dtype = count_dtype(n_inputs)
        planes = {}
        for name, plane in self._coadd_counts.produced(self._has_detector_bpm).items():
            if int(plane.max(initial=0)) > np.iinfo(dtype).max:
                raise ValueError(f"count plane {name} exceeds {np.dtype(dtype).name} over {n_inputs} inputs")
            planes[name] = plane.astype(dtype, copy=False)
        if not planes:
            self.logger.warning("No count planes were accumulated; the coverage count product is not written")
            return None
        missing = CoaddCounts.missing(planes)
        if missing:
            self.logger.warning(
                f"Coverage counts written without {', '.join(missing)}: this run does not produce "
                f"{'them' if len(missing) > 1 else 'it'}; NCOUNTPL/COUNTPLn name the planes that are present"
            )
        factory = self.path.imcoadd.factory
        header = fits.getheader(coadd_image)
        path = write_count_planes(factory.coadd_counts_image, planes, header=header)
        self.config_node.imcoadd.coadd_counts_image = path
        self.logger.info(f"Coadd coverage counts ({', '.join(planes)}) saved as {path}")
        figure = plot_coadd_counts(
            planes,
            factory.coadd_counts_figure,
            os.path.basename(coadd_image),
            subtitle=f"{n_inputs} input frames, coverage_policy: {self.plan.coverage_policy}, "
            f"coadd_mode: {get_key(self.config_node.imcoadd, 'coadd_mode')}",
            n_inputs=n_inputs,
            header=header,
        )
        if figure:
            self.logger.info(f"Coadd coverage check plot saved as {figure}")
        return path
