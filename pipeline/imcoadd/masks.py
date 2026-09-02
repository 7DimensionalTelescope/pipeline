import os
import shutil
import tempfile

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from ..config.utils import get_key
from ..utils import add_suffix, atleast_1d, get_basename
from .const import MASK_HEADER_CARDS, MaskBit
from .utils import determine_size


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


def _measure_trail_width(image, segment, noise, factor, **options):
    from scipy.ndimage import gaussian_filter1d, map_coordinates

    minimum = float(options.get("min_half_width", 2.0)) / factor
    maximum = float(options.get("max_half_width", 24.0)) / factor
    if minimum < 0 or maximum < minimum:
        raise ValueError("satellite_mask half-width limits are invalid")
    p0, p1 = np.asarray(segment[:2]), np.asarray(segment[2:])
    direction = p1 - p0
    length = float(np.hypot(*direction))
    if length == 0:
        return np.asarray(segment), minimum
    direction /= length
    normal = np.array([-direction[1], direction[0]])
    radius = maximum + max(3.0, 8.0 / factor)
    offsets = np.arange(-radius, radius + 0.125, 0.25)
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
    percentile = float(options.get("width_profile_percentile", 50.0))
    if not 0 <= percentile <= 100:
        raise ValueError("satellite_mask.width_profile_percentile must be between 0 and 100")
    profile = np.nanpercentile(values, percentile, axis=1)
    profile = gaussian_filter1d(profile, 0.75, mode="nearest")
    outer = np.abs(offsets) >= maximum + 1.0
    baseline = float(np.nanmedian(profile[outer]))
    signal = profile - baseline
    search = np.abs(offsets) <= maximum
    peak = int(np.nanargmax(np.where(search, signal, np.nan)))
    width_sigma = float(options.get("width_sigma", 1.0))
    if width_sigma <= 0:
        raise ValueError("satellite_mask.width_sigma must be positive")
    above = signal >= width_sigma * noise
    if not above[peak]:
        center = float(offsets[peak]) if signal[peak] > 0 else 0.0
        half_width = minimum
    else:
        left = peak
        right = peak
        while left > 0 and above[left - 1]:
            left -= 1
        while right + 1 < len(above) and above[right + 1]:
            right += 1
        center = float(offsets[peak])
        half_width = float(
            max(center - offsets[left], offsets[right] - center) + 0.125
        )
        half_width = min(maximum, max(minimum, half_width))
    shifted = np.r_[p0 + center * normal, p1 + center * normal]
    return shifted, half_width


def detect_satellite_trails(image: np.ndarray, **options) -> tuple[np.ndarray, np.ndarray]:
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
    segments = np.asarray(
        [(p0[0], p0[1], p1[0], p1[1]) for p0, p1 in raw], dtype=np.float64
    ).reshape(-1, 4)
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
            accepted.append(
                _measure_trail_width(flat, segment, flat_sigma, factor, **options)
            )

    canvas = Image.new("1", (image.shape[1], image.shape[0]))
    draw = ImageDraw.Draw(canvas)
    padding = float(options.get("width_padding", 1.0))
    center_offset = (factor - 1) / 2
    scaled = []
    for segment, half_width in accepted:
        x1, y1, x2, y2 = segment * factor + center_offset
        radius = max(1, int(np.ceil(half_width * factor + padding)))
        draw.line((x1, y1, x2, y2), fill=1, width=2 * radius + 1)
        for x, y in ((x1, y1), (x2, y2)):
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=1)
        scaled.append((x1, y1, x2, y2))
    mask = np.asarray(canvas, dtype=bool).copy()
    mask &= valid
    lines = np.asarray(scaled, dtype=np.float64).reshape(-1, 4)
    return mask, lines


class CoaddMaskBuilder:
    def __init__(self, images, output_path, match_swarp_size, keep_frames):
        self.images = list(images)
        target_w, target_h, _, _, self.x0, self.y0, self.shapes = determine_size(
            self.images, match_swarp_size
        )
        self.output_path = output_path
        self.output = np.zeros((target_h, target_w), dtype=np.uint8)
        self.frames = [None] * len(self.images)
        self.headers = [None] * len(self.images)
        self.keep_frames = keep_frames

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
        self.output[ty0:ty1, tx0:tx1] |= mask[sy0:sy1, sx0:sx1]

    def mark_outliers(self, index, geometry, rejected):
        tx0, tx1, ty0, ty1, sx0, sx1, sy0, sy1 = geometry
        target = self.output[ty0:ty1, tx0:tx1]
        target[rejected] |= int(MaskBit.OUTLIER)
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
            frame_header = (
                self.headers[index]
                if self.headers[index] is not None
                else fits.getheader(image)
            )
            frame_header["BUNIT"] = "bitmask"
            for key, value in MASK_HEADER_CARDS.items():
                frame_header[key] = value
            fits.writeto(
                add_suffix(image, "mask"),
                self.frame(index) if isinstance(stored, str) else stored,
                header=frame_header,
                overwrite=True,
            )

    def write(self, dump_frames=False):
        header = fits.getheader(self.output_path)
        header["BUNIT"] = "bitmask"
        for key, value in MASK_HEADER_CARDS.items():
            header[key] = value
        mask_path = add_suffix(self.output_path, "mask")
        fits.writeto(mask_path, self.output, header=header, overwrite=True)
        if dump_frames:
            self.write_frames()
        return mask_path


class MaskMixin:
    def _prepare_intermediate_storage(self, images):
        if getattr(self, "_intermediate_policy_ready", False):
            return
        requested = self.plan.intermediate_policy
        use_memory = requested == "memory" or (
            requested == "auto" and len(atleast_1d(images)) <= self.plan.memory_image_limit
        )
        durable_models = bool(
            get_key(self.config_node.imcoadd, "output_bkg_map", default=False)
            or get_key(self.config_node.imcoadd, "output_sky_rms_map", default=False)
        )
        if use_memory and durable_models:
            if requested == "memory":
                raise ValueError(
                    "intermediate_policy 'memory' is incompatible with output_bkg_map or output_sky_rms_map"
                )
            use_memory = False
        if use_memory:
            root = "/dev/shm"
            need = int(2.5 * sum(os.path.getsize(path) for path in atleast_1d(images)))
            usable = os.path.isdir(root) and os.access(root, os.W_OK)
            free = shutil.disk_usage(root).free if usable else 0
            if not usable or free < need:
                if requested == "memory":
                    raise OSError(
                        f"intermediate_policy 'memory' needs {need / 1e9:.1f} GB in {root}; "
                        f"{free / 1e9:.1f} GB is available"
                    )
                use_memory = False
        self._intermediate_policy = "memory" if use_memory else "disk"
        self._frame_cache = {}
        self._working_mask_paths = []
        self._memory_bkgsub_dump_pairs = []
        if use_memory:
            self._memory_intermediate_dir = tempfile.mkdtemp(prefix="pipeline_imcoadd_", dir="/dev/shm")
            self._bkgsub_dir = os.path.join(self._memory_intermediate_dir, "bkgsub")
            os.makedirs(self._bkgsub_dir, exist_ok=True)
        else:
            self._memory_intermediate_dir = None
            self._bkgsub_dir = self.path.imcoadd.factory.bkgsub_dir
        self._intermediate_policy_ready = True
        self.logger.info(
            f"Intermediate policy: {self._intermediate_policy} "
            f"({len(atleast_1d(images))} images, memory limit {self.plan.memory_image_limit})"
        )

    def _read_stage_frame(self, image):
        cached = getattr(self, "_frame_cache", {}).get(image)
        if cached is not None:
            return cached
        data, header = fits.getdata(image, header=True, memmap=False)
        value = np.ascontiguousarray(data, dtype=np.float32), header
        if getattr(self, "_intermediate_policy", "disk") == "memory":
            self._frame_cache[image] = value
        return value

    @staticmethod
    def _project_pixels(xs, ys, input_header, output_header, shape):
        output = np.zeros(shape, dtype=bool)
        if not len(xs):
            return output
        ra, dec = WCS(input_header).all_pix2world(
            xs.astype(np.float64), ys.astype(np.float64), 0
        )
        x, y = WCS(output_header).all_world2pix(ra, dec, 0)
        finite = np.isfinite(x) & np.isfinite(y)
        xi = np.zeros(x.shape, dtype=np.int64)
        yi = np.zeros(y.shape, dtype=np.int64)
        xi[finite] = np.rint(x[finite]).astype(np.int64)
        yi[finite] = np.rint(y[finite]).astype(np.int64)
        inside = finite & (xi >= 0) & (xi < shape[1]) & (yi >= 0) & (yi < shape[0])
        output[yi[inside], xi[inside]] = True
        return output

    def _detector_mask_bits(self, detector_image, output_header, output_shape, detector_data=None):
        input_header = fits.getheader(detector_image)
        output = np.zeros(output_shape, dtype=np.uint8)
        mask_file, badpix = self._get_bpmask(detector_image)
        ys, xs = np.nonzero(fits.getdata(mask_file, memmap=False) == badpix)
        bad = self._project_pixels(xs, ys, input_header, output_header, output_shape)
        output[bad] |= int(MaskBit.BAD)
        saturation = input_header.get("SATURATE")
        if saturation is not None:
            if detector_data is not None:
                ys, xs = np.nonzero(
                    np.isfinite(detector_data) & (detector_data >= float(saturation))
                )
                if detector_data.shape == output_shape:
                    saturated = np.zeros(output_shape, dtype=bool)
                    saturated[ys, xs] = True
                else:
                    saturated = self._project_pixels(xs, ys, input_header, output_header, output_shape)
            else:
                detector_data = fits.getdata(detector_image, memmap=False)
                ys, xs = np.nonzero(np.isfinite(detector_data) & (detector_data >= float(saturation)))
                saturated = self._project_pixels(xs, ys, input_header, output_header, output_shape)
            output[saturated] |= int(MaskBit.SATURATED)
        return output

    def prepare_quality_masks(self, images, detector_images=None):
        images = list(atleast_1d(images))
        detector_images = list(atleast_1d(detector_images or self.input_images))
        if len(images) != len(detector_images):
            raise ValueError("quality-mask images and detector images differ in length")
        self._prepare_intermediate_storage(images)
        match_swarp_size = bool(get_key(self.config_node.imcoadd, "match_swarp_size", default=True))
        builder = CoaddMaskBuilder(
            images,
            self.config_node.imcoadd.coadd_image,
            match_swarp_size,
            self.plan.dump_reprojected_masks,
        )
        satellite_options = get_key(self.config_node.imcoadd, "satellite_mask", default={}) or {}
        quality_masks = []
        for index, (image, detector) in enumerate(zip(images, detector_images)):
            data, header = self._read_stage_frame(image)
            same_file = os.path.abspath(image) == os.path.abspath(detector)
            mask = self._detector_mask_bits(
                detector, header, data.shape, detector_data=data if same_file else None
            )
            if self.plan.satellite_mask_enabled:
                trail, lines = detect_satellite_trails(data, **satellite_options)
                mask[trail] |= int(MaskBit.SATELLITE)
                self.logger.info(
                    f"Satellite mask: {len(lines)} line(s), {int(trail.sum())} pixels in {get_basename(image)}"
                )
            builder.set_frame(index, mask, header=header)
            if self._intermediate_policy == "memory":
                quality_masks.append(mask)
            else:
                path = self.path.imcoadd.factory.stage_images(
                    [image], "mask", self.path.imcoadd.factory.mask_dir
                )[0]
                fits.writeto(path, mask, header=header, overwrite=True)
                self._working_mask_paths.append(path)
                quality_masks.append(path)
        builder.frames = quality_masks
        self._coadd_mask_builder = builder
        self._quality_masks = quality_masks
        return quality_masks

    def quality_mask(self, index):
        value = self._quality_masks[index]
        if isinstance(value, str):
            return fits.getdata(value, memmap=False).astype(np.uint8, copy=False)
        return value

    def discard_cached_frames(self):
        getattr(self, "_frame_cache", {}).clear()

    def finalize_quality_masks(self):
        builder = getattr(self, "_coadd_mask_builder", None)
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

    def _cleanup_imcoadd_intermediates(self):
        root = getattr(self, "_memory_intermediate_dir", None)
        plan = getattr(self, "_plan", None)
        if root and plan is not None and plan.dump_bkgsub and getattr(self, "_coadd_completed", False):
            for source, destination in getattr(self, "_memory_bkgsub_dump_pairs", []):
                if os.path.exists(source):
                    os.makedirs(os.path.dirname(destination), exist_ok=True)
                    shutil.copy2(source, destination)
        if root:
            shutil.rmtree(root, ignore_errors=True)
            destinations = [
                destination
                for _, destination in getattr(self, "_memory_bkgsub_dump_pairs", [])
            ]
            self.config_node.imcoadd.bkgsub_images = (
                destinations if plan is not None and plan.dump_bkgsub else None
            )
            self.images_to_coadd = None
        for path in getattr(self, "_working_mask_paths", []):
            try:
                os.remove(path)
            except OSError:
                pass
        self.discard_cached_frames()
        self._intermediate_policy_ready = False
