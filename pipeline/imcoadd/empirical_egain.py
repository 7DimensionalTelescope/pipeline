import numpy as np
from astropy.io import fits


def center_boxes(shape: tuple[int, int], n_boxes: int, size: int = 256, grid: int = 8) -> list[tuple[slice, slice]]:
    """Boxes of the central lattice, origin on a 128 px grid, evenly by lattice index."""
    y0, x0 = (round((n - grid * size) / size) * size // 2 for n in shape)
    rows, cols = np.divmod(np.round(np.linspace(0, grid * grid - 1, n_boxes)).astype(int), grid)
    return [
        (slice(y0 + r * size, y0 + (r + 1) * size), slice(x0 + c * size, x0 + (c + 1) * size))
        for r, c in zip(rows, cols)
    ]


def plan_pairs(
    images: list[str], headers: list[fits.Header], max_pairs: int = 12, budget: int = 192, max_boxes: int = 64
) -> tuple[list[tuple[str, str]], list[tuple[slice, slice]]]:
    """Consecutive frame pairs of the largest master key, thinned evenly, and their boxes."""
    members = {}
    for image, header in sorted(zip(images, headers), key=lambda item: item[1]["DATE-OBS"]):
        key = tuple(header.get(k) for k in ("IMCMB001", "IMCMB002", "IMCMB003", "IMCID001", "IMCID002", "IMCID003"))
        members.setdefault(key, []).append(image)
    frames = max(members.values(), key=len, default=[])
    pairs = list(zip(frames[:-1], frames[1:]))
    if len(pairs) > max_pairs:
        pairs = [pairs[k] for k in np.round(np.linspace(0, len(pairs) - 1, max_pairs)).astype(int)]
    if not pairs:
        return [], []
    shape = (headers[0]["NAXIS2"], headers[0]["NAXIS1"])
    return pairs, center_boxes(shape, min(max_boxes, budget // len(pairs)))


def pair_gain(
    pixels: list[np.ndarray],
    excluded: list[np.ndarray],
    bad: np.ndarray,
    flat: np.ndarray,
    bias_sigma: np.ndarray,
    dark: np.ndarray,
    dark_sigma: np.ndarray,
    pairs: list[tuple[int, int]],
    max_read_noise_fraction: float,
    window: float = 5.0,
    percentile: float = 95.0,
    min_pixels: int = 1000,
    min_boxes: int = 8,
) -> dict:
    """Pooled gain, standard deviation of the pair gains and diagnostics from frame-pair differences on box stacks."""
    shape = (len(pairs), len(flat))
    variance, count, signal, additive = (np.full(shape, np.nan) for _ in range(4))
    reference = dark_sigma[np.isfinite(dark_sigma) & ~bad]
    cut = np.percentile(reference, percentile) if reference.size else np.nan
    for t, f in enumerate(flat):
        usable = np.isfinite(f) & (f > 0) & (dark_sigma[t] <= cut)
        for p, (i, j) in enumerate(pairs):
            keep = usable & ~excluded[i][t] & ~excluded[j][t]
            if np.count_nonzero(keep) < min_pixels:
                continue
            fi, fj = f[keep] * pixels[i][t][keep], f[keep] * pixels[j][t][keep]
            d = np.rint(fi - fj).astype(np.float64)
            deviation = np.abs(d - np.median(d))
            inside = deviation <= window * 1.4826 * np.median(deviation)
            if np.count_nonzero(inside) < min_pixels:
                continue
            count[p, t] = np.count_nonzero(inside)
            variance[p, t] = d[inside].var(ddof=1)
            signal[p, t] = (
                fi[inside].mean(dtype=np.float64)
                + fj[inside].mean(dtype=np.float64)
                + 2 * dark[t][keep][inside].mean(dtype=np.float64)
            )
            additive[p, t] = 2 * np.square(bias_sigma[t][keep][inside]).mean(dtype=np.float64)
    valid = np.isfinite(variance) & np.isfinite(signal) & np.isfinite(additive)
    n = np.where(valid, count, 0.0)
    numerator, denominator = np.where(valid, n * signal, 0.0), np.where(valid, n * (variance - additive), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = numerator.sum() / denominator.sum()
        gains = numerator.sum(axis=1) / denominator.sum(axis=1)
        read_noise = np.where(valid, n * additive, 0.0).sum() / np.where(valid, n * variance, 0.0).sum()
        level = numerator.sum() / (2 * n.sum())
    gains = gains[denominator.sum(axis=1) > 0]
    boxes = int(valid.any(axis=0).sum())
    measured = bool(
        boxes >= min_boxes
        and denominator.sum() > 0
        and np.isfinite(gain)
        and gain > 0
        and read_noise <= max_read_noise_fraction
    )
    sd = float(gains.std(ddof=1)) if measured and gains.size > 1 else None
    return dict(
        gain=float(gain) if measured else None,
        sd=sd,
        read_noise=float(read_noise),
        level=float(level),
        pairs=int(gains.size),
        boxes=boxes,
    )


class EgainStash:
    """Boxes of the planned frames and of their masters, copied as the fused loop passes them."""

    def __init__(self, pairs: list[tuple[str, str]], boxes: list[tuple[slice, slice]]):
        self.pairs = pairs
        self.boxes = boxes
        self.planned = {image for pair in pairs for image in pair}
        self.frames: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.bad: np.ndarray | None = None
        self.masters: tuple[np.ndarray, ...] | None = None

    def _cut(self, array: np.ndarray, dtype=np.float32) -> np.ndarray:
        """Stack of the boxes of array."""
        return np.stack([array[box] for box in self.boxes]).astype(dtype, copy=False)

    def add_masters(self, images: list[str], calib: tuple, dark_sigma_file: str) -> None:
        """Flat, bias sigma, dark and dark sigma boxes of the planned frames' masters."""
        if self.planned.isdisjoint(images):
            return
        sig_z, d_m, f_m = calib[:3]
        with fits.open(dark_sigma_file, memmap=True) as hdul:
            dark_sigma = self._cut(hdul[0].data)
        self.masters = (self._cut(f_m), self._cut(sig_z), self._cut(d_m), dark_sigma)

    def add_frame(self, images: list[str], idx: int, sci: np.ndarray, hole, src, saturated) -> None:
        """Pixel and exclusion boxes of a planned frame that has a source mask."""
        if images[idx] not in self.planned or src is None:
            return
        excluded = np.stack(
            [
                hole[b] | src[b] | ~np.isfinite(sci[b]) | (False if saturated is None else saturated[b])
                for b in self.boxes
            ]
        )
        if self.bad is None:
            self.bad = self._cut(hole, bool)
        self.frames[images[idx]] = (self._cut(sci), excluded)

    def estimate(self, max_read_noise_fraction: float) -> dict:
        """pair_gain on the stashed boxes; blank without the masters or a stashed pair."""
        names = list(self.frames)
        pairs = [(names.index(a), names.index(b)) for a, b in self.pairs if a in self.frames and b in self.frames]
        result = dict(gain=None, sd=None, read_noise=np.nan, level=np.nan, pairs=0, boxes=0)
        if self.masters is not None and pairs:
            frames = [self.frames[name] for name in names]
            result = pair_gain(
                [pixels for pixels, _ in frames],
                [excluded for _, excluded in frames],
                self.bad,
                *self.masters,
                pairs,
                max_read_noise_fraction,
            )
        return result | dict(frames=len(names))
