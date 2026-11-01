# select which images to stack based on QA criteria
"""Quality selection of the singles going into a coadd.

Only multi-epoch coadds use this: a nightly stack takes whatever passed SANITY,
while a multi-epoch stack spans wildly different conditions and is worth culling.

Three metrics by default, each read from the first header key that exists (photometry
writes them; astrometry's variants are the fallback):

    seeing        SEEING / SEEINGMN            lower is better
    ellipticity   ELLIP / ELLIPMN              lower is better
    depth         UL5_5 / UL5_4 / ...          higher is better

Any of that is replaceable or extendable per call. ``metrics=`` swaps the whole set,
``extra=`` appends to the default one; both take ``{name: (key_or_keys, direction)}``
where direction is "lower" or "higher". ImCoadd's automatic selection uses the default
set unless told otherwise; an interactive session is where extra axes earn their keep,
and the plot lays out however many pairs result.

``auto`` applies the suggested cuts and never asks. ``interactive`` shows them on a
plot and takes your edits first; it only works inside a Jupyter kernel and falls back
to ``auto`` anywhere else, so a queued run can never block on stdin.

The same machinery is usable standalone, without ImCoadd, to look at a set of images
before deciding anything: build a metric table with ``metrics_from_files`` or
``metrics_from_image_qa``, then hand it to ``select_from_table``. The table carries its
own directions in ``table.meta["directions"]``, so nothing downstream needs telling
which metrics it is looking at.
"""

import json
import os
import sys
import textwrap
import time
import warnings
from typing import Literal

import numpy as np
from astropy.table import Table
from ..preprocess.ppflag import PPFLAG_INFORMATIONAL

# (metric, candidate keys in priority order, better direction)
#
# Depth: photometry names these UL5_<suffix> where get_aperture_suffix maps APER -> 0,
# APER_1 -> 1 ... APER_5 -> 5, so the real keys are UL5_0..UL5_5 (5 = fixed 10" aperture,
# the one image_qa stores as its depth column). UL5_AUTO is deliberately NOT a candidate:
# photometry hardcodes it to 0.0 for MAG_AUTO, so it exists on every frame and is never a
# depth.
QUALITY_METRICS = (
    ("seeing", ("SEEING", "SEEINGMN"), "lower"),
    ("ellipticity", ("ELLIP", "ELLIPMN", "ELLIPTICITY"), "lower"),
    ("depth", ("UL5_5", "UL5_4", "UL5_3", "UL5_2", "UL5_1", "UL5_0"), "higher"),
    ("ppflag", ("PPFLAG",), "lower"),
)

# same shape, but the keys are image_qa column names
IMAGE_QA_METRICS = (
    ("seeing", ("seeing", "seeingmn"), "lower"),
    ("ellipticity", ("ellipmn",), "lower"),
    ("depth", ("ul5_5",), "higher"),
    ("ppflag", ("ppflag",), "lower"),
)

# Metrics that cut but get no scatter panel: a preprocessing bitmask is a category, not a
# quantity, so it is drawn as the marker colour on every panel instead (see plot_selection).
CATEGORICAL_METRICS = ("ppflag",)

# PPFLAG is a bitmask, so it is selected on by an allow-list of bits, not a threshold: a
# frame is kept when every bit it sets is one you allowed, `PPFLAG & ~allowed == 0`.
#
# The allow-list is a string of 0/1 read as a binary number, most significant bit first,
# so with seven digits bit 64 is leftmost and bit 1 rightmost; 1 means that bit is tolerated.
# Write it full width — "11" (= 3) and the number eleven are easy to confuse at a glance,
# while "0000011" cannot be read as anything but a binary string:
#
#   "1000011"  tolerate a different-date calib and a manual master   (the default)
#   "1001011"  ...and lenient keys, while still rejecting SANITY=False masters
#   "0000000"  only PPFLAG == 0 (bit 64 stays allowed, it never rejects)
#
# This is what a numeric threshold could not express: "accept 8 but reject 4" is "0001011",
# whereas any `<= n` admitting 8 must also admit 4. Bit values from preprocess/ppflag.py.
PPFLAG_BITS = (
    (1, "different-date calib"),
    (2, "manual/auto-generated master"),
    (4, "master that failed SANITY"),
    (8, "lenient grouping keys ignored"),
    (16, "masterframe date wall ignored"),
    (32, "hard keys ignored"),
    (64, "bias master not shared with the dark master (informational, never rejects)"),
)

CATEGORICAL_DEFAULT_CUTS = {"ppflag": 67.0}  # "1000011"


def ppflag_mask(spec) -> int:
    """``"1000011"`` -> 67; binary number, bit 64 leftmost, ``"11"`` -> 3; quoted string only."""
    if spec is None:
        return 0
    if not isinstance(spec, str):
        raise ValueError(f"ppflag_allowed must be a quoted string of 0/1 such as '1000011', not {spec!r}")
    digits = "".join(spec.split())
    if any(ch not in "01" for ch in digits):
        raise ValueError(f"ppflag_allowed must contain only 0 and 1, got {spec!r}")
    if len(digits) > len(PPFLAG_BITS):
        raise ValueError(
            f"ppflag_allowed has {len(digits)} digits, at most {len(PPFLAG_BITS)} (bit 64 leftmost, bit 1 rightmost)"
        )
    return int(digits or "0", 2)


def ppflag_label(value) -> str:
    """``9`` -> ``"9 different-date calib + lenient grouping keys ignored"``."""
    names = [desc for bit, desc in PPFLAG_BITS if int(value) & bit]
    return f"{int(value)} = " + " + ".join(names) if names else "0 = clean"


def ppflag_spec(mask) -> str:
    """67 -> ``"1000011"``; the inverse of `ppflag_mask`, seven binary digits."""
    return format(int(mask), "07b")


# Key families where photometry writes 0 to mean "no value" instead of dropping the card.
# Only these treat 0 as missing: for an arbitrary key (SKYVAL on a subtracted frame,
# say) 0 is a real measurement and must not silently disappear.
ZERO_MEANS_MISSING = ("SEEING", "PEEING", "ELLIP", "ELONG", "UL3_", "UL5_", "ZP_", "EZP_")


def _robust_sigma(values: np.ndarray) -> float:
    """1.4826 x MAD; falls back to std when the MAD degenerates."""
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return 0.0
    sigma = 1.4826 * np.median(np.abs(finite - np.median(finite)))
    return float(sigma) if sigma > 0 else float(np.std(finite))


def normalize_metrics(metrics=None, extra=None, default=QUALITY_METRICS) -> tuple:
    """Resolve the caller's metric spec into ``((name, keys, direction), ...)``.

    ``metrics`` replaces the default set, ``extra`` appends to whatever it resolves to; an
    ``extra`` entry naming a metric already there replaces its keys and direction in place.
    Both accept ``{name: (key_or_keys, direction)}`` or an already-normalized sequence of
    triples; a lone string key is fine in place of a tuple of candidates.
    """

    def as_triples(spec):
        if spec is None:
            return ()
        if isinstance(spec, dict):
            items = spec.items()
        else:  # sequence of (name, keys, direction)
            return tuple((n, (k,) if isinstance(k, str) else tuple(k), d) for n, k, d in spec)
        out = []
        for name, value in items:
            keys, direction = value
            if direction not in ("lower", "higher"):
                raise ValueError(f"{name!r}: direction must be 'lower' or 'higher', not {direction!r}")
            out.append((name, (keys,) if isinstance(keys, str) else tuple(keys), direction))
        return tuple(out)

    base = as_triples(metrics) if metrics is not None else tuple(default)
    override = {name: (keys, direction) for name, keys, direction in as_triples(extra)}
    merged = tuple((name, *override.pop(name, (keys, direction))) for name, keys, direction in base)
    return merged + tuple((name, keys, direction) for name, (keys, direction) in override.items())


def _value(raw, key: str) -> float:
    """Header/column value as float, with the photometry 0-sentinel mapped to NaN."""
    if raw is None:
        return np.nan
    if raw == 0 and any(key.upper().startswith(p) for p in ZERO_MEANS_MISSING):
        return np.nan
    try:
        return float(raw)
    except (TypeError, ValueError):
        return np.nan


def directions_of(table: Table) -> dict[str, str]:
    """Per-metric direction the table was built with.

    Falls back to the default set for a hand-built table, so a Table someone assembled
    themselves still works as long as its columns use the standard metric names.
    """
    stored = table.meta.get("directions")
    if stored:
        return {k: v for k, v in stored.items() if k in table.colnames}
    return {n: d for n, _, d in QUALITY_METRICS if n in table.colnames}


def collect_metrics(names: list[str], headers: list, metrics=None, extra=None) -> tuple[Table, dict[str, str]]:
    """Table of the metrics per image, plus which header key each came from."""
    table = Table({"name": list(names)})
    used_keys: dict[str, str] = {}
    directions: dict[str, str] = {}
    for metric, keys, direction in normalize_metrics(metrics, extra):
        key = next((k for k in keys if any(h.get(k) is not None for h in headers)), None)
        if key is None:
            continue
        used_keys[metric] = key
        directions[metric] = direction
        table[metric] = [_value(h.get(key), key) for h in headers]
    table.meta["directions"] = directions
    table.meta["keys"] = used_keys
    return table, used_keys


def metrics_from_files(images: list[str], metrics=None, extra=None) -> Table:
    """Metric table read straight from a list of FITS files. Standalone entry point.

    >>> metrics_from_files(paths, extra={"airmass": ("AIRMASS", "lower")})
    """
    from astropy.io import fits

    headers = [fits.getheader(f) for f in images]
    table, _ = collect_metrics([os.path.basename(f) for f in images], headers, metrics=metrics, extra=extra)
    table["path"] = list(images)
    return table


def metrics_for_paths_from_image_qa(paths: list[str], metrics=None, extra=None) -> tuple[Table, int]:
    """Metric table aligned to ``paths`` from image_qa; rows absent from the DB come back NaN.

    Alignment is the contract: the caller zips the keep-mask against ``paths``. The
    ``path`` column stays on the table so the interactive prompt can still read headers
    on demand. Returns ``(table, n_found)``; header-key metric names are lower-cased to
    columns and rejected if image_qa has no such column.
    """
    from ..services.database.image_qa import ImageQATable
    from ..services.database.query import free_query

    known = set(ImageQATable.__annotations__)
    resolved = []
    for metric, candidates, direction in normalize_metrics(metrics, extra, default=IMAGE_QA_METRICS):
        col = next((c.lower() for c in candidates if c.lower() in known), None)
        if col is None:
            raise ValueError(f"{metric!r}: no image_qa column among {list(candidates)}")
        resolved.append((metric, col, direction))

    columns = ["image_path"] + [col for _, col, _ in resolved]
    rows = free_query(
        f"SELECT {', '.join(columns)} FROM image_qa WHERE image_path = ANY(%s)",
        (list(paths),),
        statement_timeout_ms=15000,
    )
    by_path = {r[0]: r[1:] for r in rows}

    table = Table({"name": [os.path.basename(p) for p in paths]})
    table["path"] = list(paths)
    directions, used = {}, {}
    for j, (metric, col, direction) in enumerate(resolved):
        table[metric] = [_value(by_path[p][j], col.upper()) if p in by_path else np.nan for p in paths]
        directions[metric] = direction
        used[metric] = col
    table.meta["directions"] = directions
    table.meta["keys"] = used
    return table, len(by_path)


def metrics_from_image_qa(
    where: str | None = None, params: list | None = None, metrics=None, extra=None, **equals
) -> Table:
    """Metric table from the ``image_qa`` table instead of from files. Standalone entry point.

    ``equals`` are ANDed equality filters on image_qa columns, e.g.
    ``metrics_from_image_qa(object="UDS", filter="m825", image_type="single")``.
    ``where`` is raw SQL for anything equality cannot express (``"nightdate >= %s"``),
    with its values in ``params``. ``metrics``/``extra`` name **image_qa columns** rather
    than header keys, e.g. ``extra={"sky": ("skyval", "lower")}``; they default to
    ``IMAGE_QA_METRICS``.
    """
    from ..services.database.image_qa import ImageQATable
    from ..services.database.query import free_query

    known = set(ImageQATable.__annotations__)
    unknown = set(equals) - known
    if unknown:
        raise ValueError(f"Not image_qa columns: {sorted(unknown)}")

    # first candidate that is a real column; unknown names are rejected rather than
    # concatenated into the SQL, which is also what keeps this free_query safe
    resolved = []
    for metric, candidates, direction in normalize_metrics(metrics, extra, default=IMAGE_QA_METRICS):
        col = next((c.lower() for c in candidates if c.lower() in known), None)
        if col is None:
            raise ValueError(f"{metric!r}: no image_qa column among {list(candidates)}")
        resolved.append((metric, col, direction))

    columns = ["image_name"] + [col for _, col, _ in resolved]
    clauses = [f"{col} = %s" for col in equals]  # names validated above; values bound
    values = list(equals.values())
    if where:
        clauses.append(f"({where})")
        values += list(params or [])
    sql = f"SELECT {', '.join(columns)} FROM image_qa"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)

    rows = free_query(sql, values)
    table = Table({"name": [r[0] for r in rows]})
    for i, (metric, col, _) in enumerate(resolved, start=1):
        table[metric] = [_value(r[i], col) for r in rows]
    table.meta["directions"] = {m: d for m, _, d in resolved}
    return table


def suggest_cuts(table: Table, nsigma: float = -1.0, fixed: dict[str, float] | None = None) -> dict[str, float]:
    """Threshold per metric at nsigma robust sigmas from the median toward the good side: -1 is one sigma lenient.

    ``fixed`` pins a metric to a given threshold instead; `CATEGORICAL_DEFAULT_CUTS`
    supplies one for the bitmask metrics, whose distribution has no useful median.
    """
    pinned = {**CATEGORICAL_DEFAULT_CUTS, **(fixed or {})}
    cuts = {}
    for metric, direction in directions_of(table).items():
        if metric in pinned:
            cuts[metric] = pinned[metric] if isinstance(pinned[metric], tuple) else float(pinned[metric])
            continue
        values = np.asarray(table[metric], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        center, sigma = float(np.median(finite)), _robust_sigma(values)
        cuts[metric] = center - nsigma * sigma if direction == "lower" else center + nsigma * sigma
    return cuts


def apply_cuts(table: Table, cuts: dict) -> np.ndarray:
    """Boolean keep-mask; a cut is a threshold or an inclusive ``(low, high)`` pair. Absent or NaN never rejects."""
    keep = np.ones(len(table), dtype=bool)
    for metric, direction in directions_of(table).items():
        if metric not in cuts:
            continue
        values = np.asarray(table[metric], dtype=float)
        if metric in CATEGORICAL_METRICS:
            # the "cut" is an allow-list of bits: keep a frame only if every bit it sets
            # was allowed
            allowed = int(cuts[metric])
            bits = np.where(np.isfinite(values), values, 0).astype(np.int64)
            allowed |= PPFLAG_INFORMATIONAL  # a fact about the reduction, not a compromise: no mask can reject on it
            passes = (bits & ~allowed) == 0
        elif isinstance(cuts[metric], tuple):
            passes = (values >= cuts[metric][0]) & (values <= cuts[metric][1])
        else:
            passes = values <= cuts[metric] if direction == "lower" else values >= cuts[metric]
        keep &= passes | ~np.isfinite(values)
    return keep


def rejection_counts(table: Table, cuts: dict[str, float]) -> dict[str, tuple[int, int]]:
    """Per cut, ``(rejected, alone)``: how many images it rejects, and how many of those
    nothing else would have rejected anyway.

    The first number overlaps with the other cuts and so is degenerate on its own — a
    frame failing three cuts is counted under all three, and a cut can read as expensive
    while costing nothing. The second is the answer to "what does dropping this cut give
    me back", which is the number worth acting on.
    """
    counts = {}
    for metric in directions_of(table):
        if metric not in cuts:
            continue
        fails = ~apply_cuts(table, {metric: cuts[metric]})
        others = apply_cuts(table, {m: v for m, v in cuts.items() if m != metric})
        counts[metric] = (int(fails.sum()), int((fails & others).sum()))
    return counts


PPFLAG_COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown")


def _pair_rows(metrics: list[str]) -> list[list[tuple[str, str]]]:
    """Metric pairs, one row per metric that introduced them.

    Row 0 is every pair among the metrics the table started with; each metric added later
    gets its own row of pairs against everything before it. So adding a key appends rows at
    the bottom instead of shuffling the existing panels around.
    """
    base = [(a, b) for i, a in enumerate(metrics[:3]) for b in metrics[i + 1 : 3]]
    rows = [base] if base else []
    for i in range(3, len(metrics)):
        rows.append([(m, metrics[i]) for m in metrics[:i]])
    return [r for r in rows if r]


def _edges(cut) -> tuple[float, ...]:
    """The threshold(s) a cut draws: one for a one-sided cut, both bounds for an interval."""
    return tuple(cut) if isinstance(cut, tuple) else (cut,)


def _rejected_spans(cut, direction, limits) -> list[tuple[float, float]]:
    """Axis ranges a cut rejects, for shading: the bad side, or both tails of an interval."""
    if isinstance(cut, tuple):
        return [(limits[0], cut[0]), (cut[1], limits[1])]
    return [(cut, limits[1]) if direction == "lower" else (limits[0], cut)]


def plot_selection(table: Table, cuts: dict, keep: np.ndarray, out_path: str | None = None):
    """One panel per metric pair with the cut lines drawn; kept in colour, rejected in grey.

    Pairwise rather than an N-D projection: the cuts are independent 1-D thresholds, and a
    projected cube makes it impossible to see which side of one a point is on. Metrics in
    `CATEGORICAL_METRICS` get no panel of their own — colour carries their value and a bold
    ``X`` marks the frames they rejected, since such a frame sits on the kept side of every
    axis drawn and would otherwise look cut for no reason.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex, to_rgb

    from ..path import NameHandler

    directions = directions_of(table)
    metrics = [m for m in directions if m not in CATEGORICAL_METRICS]
    rows = _pair_rows(metrics)
    if not rows:
        return None

    ncols = max(len(r) for r in rows)
    fig, axes = plt.subplots(len(rows), ncols, figsize=(4.6 * ncols, 4.4 * len(rows)), squeeze=False)

    def wrapped(items):
        """Greedy legend rows: 60 characters of 10-pt text per panel, 9 per entry's marker and padding."""
        packed = [[]]
        for item in items:
            if packed[-1] and sum(len(t) + 9 for t, _, _ in packed[-1] + [item]) > 60 * ncols:
                packed.append([])
            packed[-1].append(item)
        return packed

    status = np.asarray(table["status"] if "status" in table.colnames else np.full(len(table), "usable"))
    usable = status == "usable"

    # Colour carries the current bitmask's verdict, not the raw flag value: the question a
    # reader has is "did my mask throw this away", and that changes as the mask is edited.
    flag_metric = next((m for m in CATEGORICAL_METRICS if m in directions), None)
    flags = np.asarray(table[flag_metric], dtype=float) if flag_metric else None
    flag_rejected = (
        ~apply_cuts(table, {flag_metric: cuts[flag_metric]})
        if flag_metric and flag_metric in cuts
        else np.zeros(len(table), dtype=bool)
    ) & usable
    passed = usable & ~flag_rejected
    quality = apply_cuts(table, {m: v for m, v in cuts.items() if m not in CATEGORICAL_METRICS})
    groups, legend_rows = [], [[("kept", "o", "0.55"), ("rejected", "x", "0.55")]]
    if passed.any():
        survived = f"{int((keep & passed).sum())}/{int(passed.sum())} survived image quality rejection"
        groups.append((survived, passed, "tab:blue"))
        legend_rows.append([(survived, "o", "tab:blue")])
    beneath = []
    for name, cause, color, alpha in (
        ("sanity", "rejected by SANITY", "teal", 0.45),
        ("unprocessed", "not yet processed", "slategray", 0.6),
    ):
        mask = status == name
        if mask.any():
            beneath.append((mask, color, alpha))
            legend_rows.append([(f"{int(mask.sum())} {cause}", "x", color)])
    if flag_rejected.any():
        name, allowed = flag_metric.upper(), int(cuts[flag_metric]) | PPFLAG_INFORMATIONAL
        bits = np.where(np.isfinite(flags), flags, 0).astype(np.int64)
        values, counts = np.unique(bits[flag_rejected], return_counts=True)
        passing = int((flag_rejected & quality).sum())
        flag_line = [(f"{int(flag_rejected.sum())} rejected by {name} ({passing} pass quality cuts):", "", "none")]
        for shade, i in enumerate(np.argsort(-counts, kind="stable")):
            mask = flag_rejected & (bits == values[i])
            label = f"{counts[i]} ({int((mask & quality).sum())}) {name} {ppflag_spec(values[i])}"
            color = to_hex([c * (1 - 0.7 * shade / max(len(values) - 1, 1)) for c in to_rgb("peru")])
            groups.append((label, mask, color))
            flag_line.append((label + "," * (shade < len(values) - 1), "x", color))
        extra = int(np.bitwise_or.reduce(values)) & ~allowed
        notes = "; ".join(f"{name} {bit}: {desc}" for bit, desc in PPFLAG_BITS if extra & bit)
        legend_rows += wrapped(flag_line) + [[(note, "", "none")] for note in textwrap.wrap(notes, 60 * ncols - 9)]

    # Smallest group last so it is not buried by the larger one.
    groups.sort(key=lambda g: int(g[1].sum()), reverse=True)

    for r, row in enumerate(rows):
        for c in range(ncols):
            ax = axes[r][c]
            if c >= len(row):
                ax.set_visible(False)
                continue
            xk, yk = row[c]
            for mask, color, alpha in beneath:
                ax.scatter(table[xk][mask], table[yk][mask], s=12, marker="x", lw=1.1, c=color, alpha=alpha, zorder=1)
            for depth, (_, mask, color) in enumerate(groups):
                kept, cut = mask & keep, mask & ~keep
                if cut.any():
                    ax.scatter(table[xk][cut], table[yk][cut], s=12, marker="x", lw=1.1,
                               c=color, alpha=0.35, zorder=2 + depth)  # fmt: skip
                if kept.any():
                    ax.scatter(table[xk][kept], table[yk][kept], s=14, marker="o",
                               c=color, alpha=0.85, linewidths=0, zorder=2 + depth)  # fmt: skip
            if xk in cuts:
                xl = ax.get_xlim()
                for edge in _edges(cuts[xk]):
                    ax.axvline(edge, color="tab:red", ls="--", lw=1.2)
                    ax.annotate(f"{edge:g}", xy=(edge, 0.99), xycoords=("data", "axes fraction"),
                                color="tab:red", fontsize=10, ha="left", va="top")  # fmt: skip
                for lo, hi in _rejected_spans(cuts[xk], directions.get(xk), xl):
                    ax.axvspan(lo, hi, color="0.92", zorder=0)
                ax.set_xlim(xl)
            if yk in cuts:
                yl = ax.get_ylim()
                for edge in _edges(cuts[yk]):
                    ax.axhline(edge, color="tab:red", ls="--", lw=1.2)
                    ax.annotate(f"{edge:g}", xy=(0.99, edge), xycoords=("axes fraction", "data"),
                                color="tab:red", fontsize=10, ha="right", va="bottom")  # fmt: skip
                for lo, hi in _rejected_spans(cuts[yk], directions.get(yk), yl):
                    ax.axhspan(lo, hi, color="0.92", zorder=0)
                ax.set_ylim(yl)
            ax.set_xlabel(xk.upper(), fontsize=13)
            ax.set_ylabel(yk.upper(), fontsize=13)
            ax.tick_params(labelsize=11)

    # Legend at the figure level, not inside a panel: with several flag values an in-axes
    # legend covers the data, and the per-flag counts make the title line overflow.
    from matplotlib.lines import Line2D

    def extent(artist):
        return artist.get_window_extent().transformed(fig.transFigure.inverted())

    legend_title = None
    legends = [
        fig.legend(
            handles=[Line2D([], [], ls="", marker=m, color=c, label=t) for t, m, c in row],
            loc="lower center",
            ncol=len(row),
            fontsize=10,
            title=None if k else legend_title,
            frameon=False,
            borderaxespad=0,
        )
        for k, row in enumerate(legend_rows)
    ]
    bottom = 0.07 / (4.4 * len(rows))
    for leg, row in zip(legends[::-1], legend_rows[::-1]):
        blank = 0 if row[0][1] else (leg.handlelength + leg.handletextpad) * 10 / (72 * fig.get_figwidth())
        leg.set_bbox_to_anchor((0.5 - blank / 2, bottom))
        bottom = extent(leg).y1

    def fitted(text, cap):
        """``cap`` points, or smaller if that would run off the edge of the figure."""
        return min(cap, 72 * (4.6 * ncols) / (0.55 * max(len(text), 1)))

    target = ""
    if len(table):
        parsed = NameHandler(list(table["path" if "path" in table.colnames else "name"]))
        obj, filt = parsed.obj_collapse, parsed.filter_collapse
        target = f"{obj} {filt} " if isinstance(obj, str) and isinstance(filt, str) else ""
    title = f"{target}ImCoadd Input:  {int(keep.sum())}/{len(table)} survived"
    head = fig.text(0.5, 0.98, title, ha="center", va="top", fontsize=fitted(title, 17))

    # Second line, because a categorical cut has no panel and its cost is otherwise
    # invisible: with PPFLAG the points are coloured but nothing says how many it removed.
    rejected = rejection_counts(table[usable], cuts)
    if rejected:
        sign = {"lower": "≤", "higher": "≥"}

        def cut_label(m):
            if m in CATEGORICAL_METRICS:
                return f"{m.upper()} ({ppflag_spec(cuts[m])})"
            if isinstance(cuts[m], tuple):
                return f"{round(cuts[m][0], 2):g} ≤ {m.upper()} ≤ {round(cuts[m][1], 2):g}"
            return f"{m.upper()} {sign.get(directions.get(m), '')} {round(cuts[m], 2):g}"

        order = sorted(rejected, key=lambda m: m in CATEGORICAL_METRICS)
        counted = "; ".join(f"{cut_label(m)}: {rejected[m][1]}" for m in order)
        line = f"Independent rejection counts: {counted}"
        size = fitted(line, 12)
        head = fig.text(0.5, extent(head).y0 - 0.11 / (4.4 * len(rows)), line, ha="center", va="top", fontsize=size)
    fig.tight_layout(rect=(0, bottom, 1, extent(head).y0))
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=100)
    return fig


def in_notebook() -> bool:
    """True inside a Jupyter kernel; interactive selection is only usable there."""
    try:
        from IPython import get_ipython

        return type(get_ipython()).__module__.startswith("ipykernel")
    except Exception:
        return False


def cuts_json(cuts: dict) -> str:
    """The cuts as a one-line JSON object, ready to be edited and handed back."""

    def short(v):
        return [float(f"{b:.4g}") for b in v] if isinstance(v, tuple) else float(f"{v:.4g}")

    shown = {m: ppflag_spec(v) if m in CATEGORICAL_METRICS else short(v) for m, v in cuts.items()}
    return json.dumps(shown)


def cut_parts(table: Table, cuts: dict, suggested: dict | None = None) -> dict[str, str]:
    """Per metric ``seeing <= 3: -8(4)``: the cut, what it rejects, in parentheses what only it rejects.

    The first number overlaps between cuts, the second does not. A metric with no cut yet
    is listed with the value `suggest_cuts` would give it, so it can be adopted by typing.
    """
    rejected = rejection_counts(table, cuts)
    parts = {}
    for metric, direction in directions_of(table).items():
        if metric not in cuts:
            hint = (suggested or {}).get(metric)
            parts[metric] = f"{metric}: uncut (suggest {hint:.4g})" if hint is not None else f"{metric}: uncut"
            continue
        if metric in CATEGORICAL_METRICS:
            shown = f"{metric} = {ppflag_spec(cuts[metric])}"
        elif isinstance(cuts[metric], tuple):
            shown = f"{cuts[metric][0]:.4g} <= {metric} <= {cuts[metric][1]:.4g}"
        else:
            shown = f"{metric} {'<=' if direction == 'lower' else '>='} {cuts[metric]:.4g}"
        n, alone = rejected[metric]
        parts[metric] = f"{shown}: -{n}({alone})"
    return parts


def cuts_summary(table: Table, cuts: dict, keep, suggested: dict | None = None) -> str:
    """The whole state on one short line, so it fits inside an `input()` prompt."""
    return f"{int(np.asarray(keep).sum())}/{len(table)} kept | " + ", ".join(cut_parts(table, cuts, suggested).values())


def _settle(seconds: float = 0.3) -> None:
    """Best-effort pause so the figure reaches the frontend before the prompt opens.

    The stdin request and the iopub output travel on different channels with no
    ordering guarantee between them, so this can only improve the odds, never assure
    them. What actually makes the prompt answerable is that everything needed to
    answer it is in the prompt string, which is delivered with the request itself.
    """
    sys.stdout.flush()
    time.sleep(seconds)


def _as_number(value) -> float | None:
    """A JSON number, or a number someone left quoted; None for anything else."""
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_key(name: str, headers: list | None) -> str:
    """The header key a metric reads from — its own name, checked against the headers.

    Case-insensitive: upper-cased for the lookup because that is what a FITS card is,
    and lower case everywhere it is shown.
    """
    if headers is None:
        raise ValueError("metrics cannot be added here; no headers were given")
    key = name.upper()
    if not any(h.get(key) is not None for h in headers):
        raise ValueError(f"no {name} in any of these headers")
    return key


# "<=" before "=", or "airmass<=1.2" would split on the "=" and leave a stray "<"
_OPERATORS = ("<=", ">=", "=")


def _refuse_strict(item: str) -> None:
    """Every cut is inclusive, so a bare ``<`` or ``>`` is refused rather than read as ``<=`` / ``>=``."""
    if ("<" in item or ">" in item) and "<=" not in item and ">=" not in item:
        raise ValueError(f'{item!r}: cuts are inclusive; write "<=" or ">="')


def _interval(name: str, specs: list) -> tuple[float, float] | None:
    """``[(">=", "2"), ("<=", "3")]`` or ``[(None, [2, 3])]`` -> ``(2.0, 3.0)``; None when not a pair of bounds."""
    if len(specs) == 1 and specs[0][0] is None and isinstance(specs[0][1], (list, tuple)):
        values = [_as_number(v) for v in specs[0][1]]
        if len(values) != 2 or None in values:
            raise ValueError(f"{name}: a pair means [low, high], not {specs[0][1]!r}")
        lo, hi = values
    elif len(specs) == 2 and {op for op, _ in specs} == {"<=", ">="}:
        by_op = {op: _as_number(v) for op, v in specs}
        if None in by_op.values():
            raise ValueError(f"{name}: both bounds must be numbers, not {[v for _, v in specs]!r}")
        lo, hi = by_op[">="], by_op["<="]
    else:
        return None
    if name in CATEGORICAL_METRICS:
        raise ValueError(f'{name} takes a binary string, bit 64 leftmost, as in "{name}=1000011"')
    if lo > hi:
        raise ValueError(f"{name}: low bound {lo:g} is above high bound {hi:g}")
    return float(lo), float(hi)


def _parse_items(text: str) -> dict:
    """``seeing<=3.6, airmass lower`` -> ``{"seeing": [("<=", "3.6")], "airmass": [(None, "lower")]}``

    The operator form is exactly what `cuts_summary` prints, so what you are shown can be
    typed straight back with the number changed — one language for reading and writing.
    A metric named twice, ``seeing>=2, seeing<=3``, collects both bounds.
    """
    edits = {}
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        _refuse_strict(item)
        for op in _OPERATORS:
            name, sep, value = item.partition(op)
            if sep:
                if not name.strip() or not value.strip():
                    raise ValueError(f'{item!r} needs a name and a value, as in "airmass<=1.2"')
                edits.setdefault(name.strip(), []).append((op, value.strip()))
                break
        else:
            tokens = item.split()
            if len(tokens) != 2:
                raise ValueError(f'{item!r} should read "airmass<=1.2", "seeing 3.6" or "airmass lower"')
            edits.setdefault(tokens[0], []).append((None, tokens[1]))
    return edits


def apply_edits(table: Table, cuts: dict, text: str, headers: list | None = None) -> bool:
    """Fold one edit line into ``cuts``, returning whether anything changed.

    Three spellings, all giving everything at once. ``seeing<=3.6`` is the one the
    summary prints back at you; it carries the direction as well as the threshold, so
    it both **adds** a metric and cuts on it in one go (``airmass<=1.2``). ``airmass
    lower`` adds one *uncut*, to look before deciding. The JSON object `cuts_json`
    prints can also be pasted back and edited. ``ppflag=1000011`` sets a categorical
    allow-mask. Input is case-insensitive.

    Everything is validated before anything is applied, so a typo anywhere leaves both
    the cuts and the table exactly as they were and the whole line can be retyped.
    """
    text = text.strip()
    if text.startswith("{"):
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f'not valid JSON ({e.msg}); expected e.g. {{"seeing": 3.1}}')
        if not isinstance(loaded, dict):
            raise ValueError('expected an object such as {"seeing": 3.1, "depth": 16.5}')
        edits = {k: [(None, v)] for k, v in loaded.items()}
    else:
        edits = _parse_items(text)

    directions = directions_of(table)
    new_cuts, new_metrics, new_directions = {}, {}, {}
    for raw, specs in edits.items():
        name = raw.lower()
        bounds = _interval(name, specs)
        if bounds is not None:
            if name not in directions:
                new_metrics[name] = (_metric_key(name, headers), "lower")
            new_cuts[name] = bounds
            continue
        if len(specs) != 1:
            raise ValueError(f'{name}: one cut, or a ">=" and a "<=" pair for both bounds')
        ((op, value),) = specs
        if op in ("<=", ">="):
            if name in CATEGORICAL_METRICS:
                raise ValueError(f'{name} takes a binary string, bit 64 leftmost, as in "{name}=1000011"')
            number = _as_number(value)
            if number is None:
                raise ValueError(f"{name}: {value!r} is not a number")
            direction = "lower" if op == "<=" else "higher"
            if name not in directions:
                new_metrics[name] = (_metric_key(name, headers), direction)
            elif directions[name] != direction:
                new_directions[name] = direction
            new_cuts[name] = number
            continue
        if name in CATEGORICAL_METRICS:
            new_cuts[name] = ppflag_mask(value)
            continue
        number = _as_number(value)
        if number is not None:
            if name not in directions:
                raise ValueError(f'{name} is not a metric here; add it with "{name}<={value}"')
            new_cuts[name] = number
        elif isinstance(value, str) and value.lower() in ("lower", "higher"):
            new_metrics[name] = (_metric_key(name, headers), value.lower())
        elif isinstance(value, str):
            raise ValueError(f'{name}: {value!r} is not a number, nor "lower"/"higher" to add a metric')
        else:
            raise ValueError(f'{name}: {value!r} is neither a number nor a "lower"/"higher" spec')

    for name, (key, direction) in new_metrics.items():
        table[name] = [_value(h.get(key), key) for h in headers]
        table.meta.setdefault("directions", {})[name] = direction
        table.meta.setdefault("keys", {})[name] = key
        cuts.pop(name, None)  # a re-added metric takes a fresh suggestion
        print(f"    {name} ({direction} is better)")
    for name, direction in new_directions.items():
        table.meta.setdefault("directions", {})[name] = direction
        print(f"    {name} now reads {direction} is better")
    changed = bool(new_metrics) or bool(new_directions)
    for name, value in new_cuts.items():
        changed |= value != cuts.get(name)
        cuts[name] = value
    return changed


def prompt_cuts(
    table: Table,
    cuts: dict[str, float],
    headers: list | None = None,
    nsigma: float = -1.0,
    fixed: dict[str, float] | None = None,
) -> dict:
    """Show the suggestion, take edits, redraw, repeat until nothing changes.

    Every round asks exactly once, for the whole set of cuts at once, and **the prompt
    string carries the current state and the last error**. Both of those are forced by
    the notebook protocol: the prompt travels on the stdin channel while prints and
    figures travel on iopub, with no ordering guarantee between the two, so the input
    box routinely opens before the plot and the printed cuts arrive. Anything not in
    the prompt may simply not be on screen when it has to be answered, and several
    prompts per round could additionally be answered out of order.

    Blank accepts what is drawn and returns. Any edit redraws with its consequences
    before asking again, so you never commit to a cut you have not seen applied.
    ``headers`` lets the line bind new metrics to header keys; without it only the
    existing metrics can be re-cut.
    """
    import matplotlib.pyplot as plt

    chosen = dict(cuts)
    while True:
        # A metric added last round is drawn but NOT cut on: adding it is a request to
        # look at its distribution, and choosing where to cut is the next decision. The
        # value suggest_cuts would give it rides in the prompt, to adopt by typing it.
        suggested = suggest_cuts(table, nsigma=nsigma, fixed=fixed)

        keep = apply_cuts(table, chosen)
        fig = plot_selection(table, chosen, keep)
        plt.show()
        if fig is not None:
            plt.close(fig)
        summary = cuts_summary(table, chosen, keep, suggested)
        print(f"  {summary}")
        print(f"  {cuts_json(chosen)}")
        if any(m in chosen for m in CATEGORICAL_METRICS):
            print("  PPFLAG digits, left to right: " + ", ".join(f"{b}={desc}" for b, desc in reversed(PPFLAG_BITS)))
        if headers is not None:
            print('  a metric is added by its header key: "airmass<=1.2", or "airmass lower" to look first')

        changed, note = False, ""
        while True:
            _settle()
            answer = input(
                f"  {note}{summary} | -N(M): N fail that cut, M return if it is dropped "
                '| type "seeing<=3.6" to set or add, "seeing>=2, seeing<=3" for both bounds, Enter accepts: '
            ).strip()
            if not answer:
                break
            try:
                changed = apply_edits(table, chosen, answer, headers)
            except ValueError as e:
                print(f"    {e}")
                note = f"[{e}] "  # the prompt is the only output guaranteed to be on screen
                continue
            break

        if not changed:
            return chosen


def select_from_table(
    table: Table,
    mode: Literal["auto", "interactive"] = "auto",
    nsigma: float = -1.0,
    plot_path: str | None = None,
    logger=None,
    headers: list | None = None,
    min_keep: int = 3,
    fixed_cuts: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Suggest cuts over a metric table, optionally let the user edit them, apply.

    Works on any table `collect_metrics`, `metrics_from_files` or `metrics_from_image_qa`
    produced, so it serves both the ImCoadd step and standalone analysis.

    ``headers`` lets the interactive prompt add metrics from header keys; when it is not
    given and the table carries a ``path`` column (as `metrics_from_files` leaves), the
    headers are read from there on demand.
    """

    def say(level, msg):
        if logger is not None:
            getattr(logger, level)(msg)

    if not directions_of(table):
        say("warning", "No quality metrics available in the table; skipping selection")
        return np.ones(len(table), dtype=bool), {}

    cuts = suggest_cuts(table, nsigma=nsigma, fixed=fixed_cuts)
    if mode == "interactive":
        if in_notebook():
            if headers is None and "path" in table.colnames:
                from astropy.io import fits

                headers = [fits.getheader(f) for f in table["path"]]
            cuts = prompt_cuts(table, cuts, headers=headers, nsigma=nsigma, fixed=fixed_cuts)
        else:
            say("warning", "interactive selection needs a notebook; using the automatic cuts")

    keep = apply_cuts(table, cuts)
    if keep.sum() < min_keep and keep.any():
        # legal, but a coadd of one or two frames is rarely what was meant and nothing
        # downstream objects: the products simply end up named after those frames
        say(
            "warning",
            f"Quality cuts {cuts} leave only {int(keep.sum())} of {len(table)} images — "
            f"the coadd will be named after them and carry their header",
        )

    if plot_path:
        try:
            plot_selection(table, cuts, keep, out_path=plot_path)
            say("debug", f"Selection plot saved to {plot_path}")
        except Exception as e:
            say("warning", f"Could not plot the selection: {e}")

    for name, ok in zip(table["name"], keep):
        if not ok:
            say("debug", f"Rejected by quality selection: {name}")
    say("info", f"Quality selection keeps {int(keep.sum())}/{len(table)} images (cuts: {cuts})")
    rejected = rejection_counts(table, cuts)
    if rejected:
        say("info", "Rejected per cut (overlapping; 'alone' = would be kept without it): "
                    + ", ".join(f"{m} {n} ({alone} alone)" for m, (n, alone) in rejected.items()))  # fmt: skip
    return keep, cuts


def resolve_fixed_cuts(fixed_cuts, metrics=None, extra=None, warn=None):
    """``{"seeing": "<=2.8", "bin2fwhm": "<=5.6"}`` -> numeric cuts plus auto-declared metrics.

    The operator string is the same grammar the interactive prompt speaks: it carries the
    direction, so a metric not in the default set is bound to the header key of its own
    name with no separate ``extra`` mapping. Plain numbers are allowed only for metrics
    that already exist; a categorical takes its bitstring. Every cut is inclusive, so ``<``
    and ``>`` are refused rather than read as ``<=`` / ``>=``. A list of a ``>=`` and a ``<=``
    string, ``{"seeing": [">=2", "<=3"]}``, keeps both bounds. A single operator against a
    known metric's direction flips it for this selection, keeping the worse side, and says
    so through ``warn`` (a callable taking the message; `warnings.warn` by default).
    """
    if not fixed_cuts:
        return {}, extra
    known = {name: (keys, direction) for name, keys, direction in normalize_metrics(metrics, extra)}
    numeric, declared = {}, {}
    for raw, value in fixed_cuts.items():
        name = raw.lower()
        if name in CATEGORICAL_METRICS:
            if isinstance(value, (list, tuple)):
                raise ValueError(f'{name} takes a binary string, bit 64 leftmost, as in "{name}": "1000011"')
            numeric[name] = ppflag_mask(value) if isinstance(value, str) else float(value)
            continue
        specs = []
        for item in value if isinstance(value, (list, tuple)) else [value]:
            text = str(item).strip()
            _refuse_strict(text)
            op = next((o for o in ("<=", ">=") if text.startswith(o)), None)
            specs.append((op, text[2:].strip() if op else item))
        bounds = _interval(name, specs) if len(specs) == 2 else None
        if bounds is not None:
            if name not in known:
                declared[name] = (name.upper(), "lower")
            numeric[name] = bounds
            continue
        if len(specs) != 1:
            raise ValueError(f'{name}: one cut, or a ">=" and a "<=" pair for both bounds, not {value!r}')
        op, spec = specs[0]
        number = _as_number(spec)
        if number is None:
            raise ValueError(
                f'{name}: cannot read {value!r}; write "{name}": "<=3.1", ">=3.1", [">=2", "<=3"] or a number'
            )
        if op:
            direction = "lower" if op == "<=" else "higher"
            if name not in known:
                declared[name] = (name.upper(), direction)
            elif known[name][1] != direction:
                declared[name] = (known[name][0], direction)
                (warn or warnings.warn)(f"{name}: {value!r} keeps the worse side; {known[name][1]} is better")
        elif name not in known:
            raise ValueError(f'{name} is not a known metric; give it a direction, e.g. "{name}": "<={value}"')
        numeric[name] = number
    if declared:
        if extra is None:
            extra = declared
        elif isinstance(extra, dict):
            extra = {**extra, **declared}
        else:
            extra = tuple(extra) + tuple((n, k, d) for n, (k, d) in declared.items())
    return numeric, extra


def cuts_grammar(cuts: dict, directions: dict[str, str]) -> dict[str, str | list[str]]:
    """Numeric cuts -> config operator grammar; inverse of `resolve_fixed_cuts`."""

    def spec(m, v):
        if m in CATEGORICAL_METRICS:
            return ppflag_spec(v)
        if isinstance(v, tuple):
            return [f">={v[0]:.6g}", f"<={v[1]:.6g}"]
        return f"{'<=' if directions.get(m) == 'lower' else '>='}{v:.6g}"

    return {m: spec(m, v) for m, v in cuts.items()}


def select_images(
    names: list[str],
    headers: list,
    mode: Literal["auto", "interactive"] = "auto",
    nsigma: float = -1.0,
    plot_path: str | None = None,
    logger=None,
    metrics=None,
    extra=None,
    fixed_cuts: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, float], Table]:
    """Keep-mask over ``names``, from already-read headers. Returns (mask, cuts, table)."""
    fixed_cuts, extra = resolve_fixed_cuts(fixed_cuts, metrics, extra, warn=logger.warning if logger else None)
    table, used_keys = collect_metrics(names, headers, metrics=metrics, extra=extra)
    if used_keys and logger is not None:
        logger.info(f"Quality selection on {', '.join(f'{m}={k}' for m, k in used_keys.items())}")
    keep, cuts = select_from_table(
        table,
        mode=mode,
        nsigma=nsigma,
        plot_path=plot_path,
        logger=logger,
        headers=headers,
        fixed_cuts=fixed_cuts,
    )
    return keep, cuts, table


def _tabulate(rows: list[dict], columns: list[str]) -> list[str]:
    """Rows as lines with the columns vertically aligned; a cell a row lacks is blank."""
    cells = [[str(row.get(c, "")) for c in columns] for row in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(columns))]
    return ["  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in cells]


def select_ingredients(
    obj: str,
    cuts: dict | None = None,
    filter_cuts: dict[str, dict] | None = None,
    nsigma: float = -1.0,
    save: bool = False,
    logger=None,
) -> dict[str, tuple[list[str], dict[str, str]]]:
    """One object against its raw budget: status per frame, cuts per filter, kept images, ImCoadd's figures; one joint figure."""
    import matplotlib.pyplot as plt

    from ..path import PathHandler
    from ..services.database.census import budget_status, census_from_raw
    from ..utils import collapse

    def say(level, msg):
        print(msg)
        if logger is not None:
            getattr(logger, level)(msg)

    census = census_from_raw(objects=[obj], qa=None)
    status = budget_status(census)
    budget = census.loc[status.index]
    if len(census) > len(budget):
        say("info", f"{obj} {len(census) - len(budget)} duplicate raw rows dropped")
    filters = sorted(set(budget["filter"]))
    filter_cuts = filter_cuts or {}
    for label, given in [("cuts", cuts), *((f"filter_cuts[{f!r}]", v) for f, v in filter_cuts.items())]:
        if given is not None and not isinstance(given, dict):
            raise TypeError(f'{label} should be a dict such as {{"depth": ">=16.1"}}, not {type(given).__name__}')
    unknown = set(filter_cuts) - set(filters)
    if unknown:
        raise ValueError(f"filter_cuts names filters without raw frames of {obj}: {sorted(unknown)}")
    resolved = {}
    for filt in filters:
        merged = {**(cuts or {}), **filter_cuts.get(filt, {})}
        resolved[filt] = resolve_fixed_cuts(merged, warn=lambda msg: say("warning", f"{obj} {filt} {msg}"))
    # new metrics join the shared table; a flipped default metric only changes its own filter's directions
    base = {name for name, _, _ in IMAGE_QA_METRICS}
    extra = {m: spec for _, own in resolved.values() for m, spec in (own or {}).items() if m not in base}
    images = budget.single_path.fillna(budget.raw_path).tolist()
    table, _ = metrics_for_paths_from_image_qa(images, extra=extra)
    table["status"] = status.tolist()
    has_single = budget.single_path.notna().to_numpy()

    selection, drawn, rows, notes = {}, [], [], []
    for filt in filters:
        fixed, own = resolved[filt]
        mask = (budget["filter"] == filt).to_numpy()
        paths = [p for p, m in zip(images, mask) if m]
        group = table[mask]
        group.remove_columns([m for m in extra if m not in (own or {})])
        group.meta["directions"] = {**table.meta["directions"], **{m: d for m, (_, d) in (own or {}).items()}}
        state = np.asarray(group["status"])
        usable = state == "usable"
        ok = group[usable]
        chosen = suggest_cuts(ok, nsigma=nsigma, fixed=fixed)
        keep = np.zeros(len(group), dtype=bool)
        keep[usable] = apply_cuts(ok, chosen)
        kept = [p for p, k in zip(paths, keep) if k]
        selection[filt] = (kept, cuts_grammar(chosen, group.meta["directions"]))
        singles = [p for p, s in zip(images, has_single & mask) if s]
        drawn.append((group, chosen, keep, kept, singles))
        row = {"object": obj, "filter": filt, "kept": f"{len(kept)}/{len(ok)} kept", **cut_parts(ok, chosen)}
        for name, cause in (("sanity", "sanity-rejected"), ("unprocessed", "not yet processed")):
            count = int((state == name).sum())
            row[name] = f"{count} {cause}" if count else ""
        row["note"] = "no single, no figure saved" if save and not singles else ""
        rows.append(row)
        unmeasured = int(np.all([~np.isfinite(np.asarray(ok[m], float)) for m in directions_of(ok)], axis=0).sum())
        if unmeasured:
            notes.append(f"{obj} {filt} without image_qa metrics, passing every cut: {unmeasured}")
    columns = ["object", "filter", "kept", *table.meta["directions"], "sanity", "unprocessed", "note"]
    for line in _tabulate(rows, columns):
        say("info", line)
    for note in notes:
        say("warning", note)

    rendered, ph = [], None
    for group, chosen, keep, kept, singles in drawn:
        out_path = None
        if save and singles:
            ph = PathHandler(kept or singles, is_pipeline=True, is_multi_epoch=True)
            stem = os.path.splitext(os.path.basename(collapse(ph.sciproc_output_yml, force=True)))[0]
            out_path = os.path.join(collapse(ph.figure_dir, force=True), f"{stem}_imcoadd_selection.jpg")
        fig = plot_selection(group, chosen, keep, out_path=out_path)
        fig.canvas.draw()
        rendered.append(np.asarray(fig.canvas.buffer_rgba()))
        plt.close(fig)
    joint_path = (
        os.path.join(os.path.dirname(ph._coadd_dir[0]), f"{obj}_imcoadd_selection.jpg") if ph is not None else None
    )
    _joint_figure(rendered, out_path=joint_path)
    return selection


def _joint_figure(images: list[np.ndarray], out_path: str | None = None) -> None:
    """Pixel arrays stacked in one column, narrower ones right-padded white; saved at 100 dpi, shown in a notebook."""
    import matplotlib.pyplot as plt

    width = max(r.shape[1] for r in images)
    stacked = np.vstack([np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0)), constant_values=255) for r in images])
    if out_path:
        plt.imsave(out_path, stacked, dpi=100)
    if in_notebook():
        from IPython.display import display

        # dpi 1 keeps inches equal to pixels; at dpi 100 Agg truncates 290.4 * 100 to 29039 rows.
        joint = plt.figure(figsize=(width, len(stacked)), dpi=1)
        ax = joint.add_axes((0, 0, 1, 1))
        ax.imshow(stacked, interpolation="nearest", aspect="auto")
        ax.set_axis_off()
        display(joint)
        plt.close(joint)


def join_selection_figures(files: list[str], out_path: str | None = None) -> None:
    """Existing selection figures (jpg/png paths) stacked into one joint image; saved when out_path is given, shown in a notebook."""
    import matplotlib.pyplot as plt

    images = []
    for f in files:
        img = plt.imread(f)
        if img.dtype != np.uint8:  # png comes back as floats in [0, 1]
            img = np.round(img * 255).astype(np.uint8)
        images.append(img[..., :3])
    _joint_figure(images, out_path=out_path)
