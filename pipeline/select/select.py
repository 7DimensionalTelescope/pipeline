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

import os
from typing import Literal

import numpy as np
from astropy.table import Table

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
)

# same shape, but the keys are image_qa column names
IMAGE_QA_METRICS = (
    ("seeing", ("seeing", "seeingmn"), "lower"),
    ("ellipticity", ("ellipmn",), "lower"),
    ("depth", ("ul5_5",), "higher"),
)

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

    ``metrics`` replaces the default set, ``extra`` appends to whatever it resolves to.
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
    return base + as_triples(extra)


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
        bad = [c for c in candidates if c not in known]
        if bad:
            raise ValueError(f"{metric!r}: not image_qa columns: {bad}")
        resolved.append((metric, candidates[0], direction))

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


def suggest_cuts(table: Table, nsigma: float = 1.0) -> dict[str, float]:
    """Threshold per metric at median +/- nsigma * robust sigma, on the bad side."""
    cuts = {}
    for metric, direction in directions_of(table).items():
        values = np.asarray(table[metric], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        center, sigma = float(np.median(finite)), _robust_sigma(values)
        cuts[metric] = center + nsigma * sigma if direction == "lower" else center - nsigma * sigma
    return cuts


def apply_cuts(table: Table, cuts: dict[str, float]) -> np.ndarray:
    """Boolean keep-mask. A metric that is absent or NaN never rejects an image."""
    keep = np.ones(len(table), dtype=bool)
    for metric, direction in directions_of(table).items():
        if metric not in cuts:
            continue
        values = np.asarray(table[metric], dtype=float)
        passes = values <= cuts[metric] if direction == "lower" else values >= cuts[metric]
        keep &= passes | ~np.isfinite(values)
    return keep


def plot_selection(table: Table, cuts: dict[str, float], keep: np.ndarray, out_path: str | None = None):
    """One panel per metric pair with the cut lines drawn; kept in colour, rejected in grey.

    Pairwise rather than an N-D projection: the cuts are independent 1-D thresholds, and a
    projected cube makes it impossible to see which side of one a point is on. With extra
    metrics the pair count grows quadratically, so panels wrap onto a grid.
    """
    import matplotlib.pyplot as plt

    metrics = list(directions_of(table))
    pairs = [(a, b) for i, a in enumerate(metrics) for b in metrics[i + 1 :]]
    if not pairs:
        return None

    ncols = min(3, len(pairs))
    nrows = -(-len(pairs) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows), squeeze=False)
    flat = [ax for row in axes for ax in row]
    for ax in flat[len(pairs) :]:
        ax.set_visible(False)
    for i, (ax, (xk, yk)) in enumerate(zip(flat, pairs)):
        ax.scatter(table[xk][~keep], table[yk][~keep], s=26, c="0.75", marker="x", label=f"cut ({int((~keep).sum())})")
        ax.scatter(table[xk][keep], table[yk][keep], s=26, c="tab:blue", label=f"keep ({int(keep.sum())})")
        if xk in cuts:
            ax.axvline(cuts[xk], color="tab:red", ls="--", lw=1)
        if yk in cuts:
            ax.axhline(cuts[yk], color="tab:red", ls="--", lw=1)
        ax.set_xlabel(xk)
        ax.set_ylabel(yk)
        if i == 0:  # one legend is enough; repeating it just eats panel space
            ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"ImCoadd input selection: {int(keep.sum())}/{len(table)} kept")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
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


def _ask_float(prompt: str, current: float) -> float:
    """Read one threshold, re-asking until it parses. Blank keeps ``current``."""
    while True:
        answer = input(f"  {prompt} [{current:.4g}]: ").strip()
        if not answer:
            return current
        try:
            return float(answer)
        except ValueError:
            print(f"    {answer!r} is not a number — try again, or press Enter to keep {current:.4g}")


def _ask_new_metrics(table: Table, headers: list) -> bool:
    """Offer to add metrics from header keys until a blank line.

    Only a blank line ends this; anything unusable is reported and re-asked, so a typo
    never falls through to be read as the next threshold.
    """
    added = False
    while True:
        answer = input("  add metric (HEADER_KEY lower|higher, blank to finish): ").strip()
        if not answer:
            return added
        parts = answer.split()
        if len(parts) != 2 or parts[1].lower() not in ("lower", "higher"):
            print("    expected e.g. 'AIRMASS lower' or 'EXPTIME higher'")
            continue
        key, direction = parts[0].upper(), parts[1].lower()
        name = key.lower()
        if name in table.colnames:
            print(f"    {name} is already there")
            continue
        if not any(h.get(key) is not None for h in headers):
            print(f"    no {key} in any of these headers")
            continue
        table[name] = [_value(h.get(key), key) for h in headers]
        table.meta.setdefault("directions", {})[name] = direction
        print(f"    added {name} from {key} ({direction} is better)")
        added = True


def prompt_cuts(table: Table, cuts: dict[str, float], headers: list | None = None, nsigma: float = 1.0) -> dict:
    """Show the suggestion, take edits, redraw, repeat until nothing changes.

    Blank on every threshold accepts what is drawn and returns. Any edit — a new
    threshold, or a metric added from a header key — redraws with the consequences
    before asking again, so you never commit to a cut you have not seen applied.
    ``headers`` enables the add-a-metric prompt; without it only the existing metrics
    can be re-cut.
    """
    import matplotlib.pyplot as plt

    chosen = dict(cuts)
    while True:
        # a metric added last round has no cut yet; give it the suggestion now so the
        # redraw below shows its line rather than an axis with nothing on it
        suggested = suggest_cuts(table, nsigma=nsigma)
        for metric in directions_of(table):
            if metric not in chosen and suggested.get(metric) is not None:
                chosen[metric] = suggested[metric]

        keep = apply_cuts(table, chosen)
        fig = plot_selection(table, chosen, keep)
        plt.show()
        if fig is not None:
            plt.close(fig)
        print(f"  {int(keep.sum())}/{len(table)} kept — Enter to accept, or type new values")

        changed = False
        for metric, direction in directions_of(table).items():
            if metric not in chosen:
                continue
            side = "keep <=" if direction == "lower" else "keep >="
            new = _ask_float(f"{metric:<12} {side}", chosen[metric])
            changed |= new != chosen[metric]
            chosen[metric] = new

        if headers is not None:
            changed |= _ask_new_metrics(table, headers)

        if not changed:
            return chosen


def select_from_table(
    table: Table,
    mode: Literal["auto", "interactive"] = "auto",
    nsigma: float = 1.0,
    plot_path: str | None = None,
    logger=None,
    headers: list | None = None,
    min_keep: int = 3,
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

    cuts = suggest_cuts(table, nsigma=nsigma)
    if mode == "interactive":
        if in_notebook():
            if headers is None and "path" in table.colnames:
                from astropy.io import fits

                headers = [fits.getheader(f) for f in table["path"]]
            cuts = prompt_cuts(table, cuts, headers=headers, nsigma=nsigma)
        else:
            say("warning", "interactive selection needs a notebook; using the automatic cuts")

    keep = apply_cuts(table, cuts)
    if not keep.any():
        say("warning", f"Quality cuts {cuts} would reject everything; keeping all instead")
        keep = np.ones(len(table), dtype=bool)
    elif keep.sum() < min_keep:
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
    return keep, cuts


def select_images(
    names: list[str],
    headers: list,
    mode: Literal["auto", "interactive"] = "auto",
    nsigma: float = 1.0,
    plot_path: str | None = None,
    logger=None,
    metrics=None,
    extra=None,
) -> tuple[np.ndarray, dict[str, float], Table]:
    """Keep-mask over ``names``, from already-read headers. Returns (mask, cuts, table)."""
    table, used_keys = collect_metrics(names, headers, metrics=metrics, extra=extra)
    if used_keys and logger is not None:
        logger.info(f"Quality selection on {', '.join(f'{m}={k}' for m, k in used_keys.items())}")
    keep, cuts = select_from_table(
        table, mode=mode, nsigma=nsigma, plot_path=plot_path, logger=logger, headers=headers
    )
    return keep, cuts, table
