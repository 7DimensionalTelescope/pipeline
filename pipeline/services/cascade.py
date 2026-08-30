"""Plan and queue the reprocessing cascade of regenerated master frames.

Five drained batches: (1) chained masters via three cumulative calib_types sweeps,
(2) singles, (3) nightly science with -overwrite, (4) multi-epoch science with -overwrite,
(5) crossfilter with -overwrite.
Read-only unless submit() is called; one phase (and sweep) per call.
Rationale and measurements: .claude/memory/services.md "cascade.py".
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

from ..const import (
    CALIB_TYPE_BIAS,
    CALIB_TYPE_DARK,
    CALIB_TYPE_FLAT,
    CONFIG_TYPE_CROSSFILTER,
    CONFIG_TYPE_PREPROCESS,
    CONFIG_TYPE_SCIENCE,
    IMAGE_TYPE_SINGLE,
    INPUT_TYPE_REPROCESS,
    MASTER_IMAGE_TYPES,
    TASK_STATUS_PAUSED,
    TASK_STATUS_PROCESSING,
    TASK_STATUS_READY,
)
from ..path.path import PathHandler
from ..utils import atleast_1d
from .database.query import free_query
from .database.recipes import image_names

# depth-bounded downward closure; the cap terminates cycles (real graph is <=5 deep)
_CLOSURE = """
WITH RECURSIVE seed AS (
    SELECT id FROM image_qa WHERE image_name = ANY(%s)
),
down(id, depth) AS (
        SELECT d.derived_image_id, 1
        FROM image_qa_dependency d JOIN seed ON d.source_image_id = seed.id
    UNION
        SELECT d.derived_image_id, down.depth + 1
        FROM image_qa_dependency d JOIN down ON d.source_image_id = down.id
        WHERE down.depth < %s
)
"""

MASTER_SWEEPS = (
    [CALIB_TYPE_BIAS],
    [CALIB_TYPE_BIAS, CALIB_TYPE_DARK],
    [CALIB_TYPE_BIAS, CALIB_TYPE_DARK, CALIB_TYPE_FLAT],
)


@dataclass
class CascadePlan:
    """What a set of regenerated masters obliges, and in what order."""

    seeds: List[str] = field(default_factory=list)
    masters: List[tuple] = field(default_factory=list)  # (image_name, image_type)
    master_configs: List[tuple] = field(default_factory=list)  # phase 1: (name, config_file)
    preprocess_configs: List[tuple] = field(default_factory=list)  # phase 2: (name, config_file)
    science_configs: List[tuple] = field(default_factory=list)  # phase 3: (name, config_file, nightdate)
    crossfilter_configs: List[tuple] = field(default_factory=list)  # phase 5: (name, config_file, nightdate)
    unregenerable: List[tuple] = field(default_factory=list)  # (config_stem, why)
    skipped: List[tuple] = field(default_factory=list)  # (config_name, why)
    counts: dict = field(default_factory=dict)

    @property
    def nightly_science(self) -> List[tuple]:
        return [c for c in self.science_configs if c[2] is not None]

    @property
    def multiepoch_science(self) -> List[tuple]:
        return [c for c in self.science_configs if c[2] is None]

    def report(self) -> str:
        lines = [
            f"Cascade from {len(self.seeds)} regenerated master frame(s): "
            + ", ".join(self.seeds[:4])
            + ("..." if len(self.seeds) > 4 else ""),
            "",
            "  affected products   " + ", ".join(f"{n} {k}" for k, n in sorted(self.counts.items())),
            f"  chained masters     {len(self.masters)}"
            + (" (" + ", ".join(f"{sum(1 for m in self.masters if m[1] == k)} {k}" for k in MASTER_IMAGE_TYPES) + ")" if self.masters else ""),  # fmt: skip
            f"  phase 1  masters    {len(self.master_configs)} preprocess configs × 3 sweeps",
            f"  phase 2  singles    {len(self.preprocess_configs)} preprocess configs",
            f"  phase 3  science    {len(self.nightly_science)} nightly configs",
            f"  phase 4  science    {len(self.multiepoch_science)} multi-epoch configs",
            f"  phase 5  crossfilter {len(self.crossfilter_configs)} configs",
        ]
        if self.unregenerable:
            lines += ["", f"  CANNOT REGENERATE ({len(self.unregenerable)}) — the chain stops here and"
                          " everything below stays stale:"]
            lines += [f"    {name}: {why}" for name, why in self.unregenerable[:20]]
            if len(self.unregenerable) > 20:
                lines.append(f"    ... and {len(self.unregenerable) - 20} more")
        if self.skipped:
            lines += ["", f"  skipped configs ({len(self.skipped)}):"]
            lines += [f"    {name}: {why}" for name, why in self.skipped[:20]]
            if len(self.skipped) > 20:
                lines.append(f"    ... and {len(self.skipped) - 20} more")
        return "\n".join(lines)


def _closure(names: List[str], max_depth: int) -> List[tuple]:
    """(image_name, image_type, process_status_id, nightdate) of everything below the seeds."""
    return free_query(
        _CLOSURE
        + """
        SELECT DISTINCT i.image_name, i.image_type, i.process_status_id, i.nightdate
        FROM down JOIN image_qa i ON i.id = down.id
        """,
        (names, max_depth),
    )


def _master_config_stems(master_names: List[str]) -> dict:
    """{master name: preprocess config stem} from the names alone; a fetched master's image_qa row can be owned by the fetching config, so process_status_id must not be used here."""
    if not master_names:
        return {}
    stems = atleast_1d(PathHandler(list(master_names), is_pipeline=True)._preproc_config_stem)
    return dict(zip(master_names, stems))


def _resolve_configs(stems) -> tuple:
    """({stem: config_file} for the ones that can run, [(stem, why)] for the ones that cannot)."""
    stems = sorted(set(s for s in stems if s))
    if not stems:
        return {}, []
    known = {
        name: (config_file, sanity)
        for name, config_file, sanity in free_query(
            "SELECT name, config_file, sanity FROM process_status"
            " WHERE name = ANY(%s) AND config_type = %s",
            (stems, CONFIG_TYPE_PREPROCESS),
        )
    }
    runnable, blocked = {}, []
    for stem in stems:
        if stem not in known:
            blocked.append((stem, "no preprocess config in process_status"))
            continue
        config_file, sanity = known[stem]
        if sanity is False:
            blocked.append((stem, "human-rejected (sanity=False)"))
        elif not config_file or not os.path.exists(config_file):
            blocked.append((stem, "config file is missing on disk"))
        elif config_file.startswith("/home/"):
            # the queue daemon cannot traverse /home/*: the task would exit 1 with no log
            blocked.append((stem, f"config under /home is unreadable by the queue daemon: {config_file}"))
        else:
            runnable[stem] = config_file
    return runnable, blocked


def plan(seed_images, max_depth: int = 12) -> CascadePlan:
    """Everything a set of ALREADY-regenerated master frames obliges. Read-only."""
    names = image_names(seed_images)
    registered = free_query("SELECT image_name FROM image_qa WHERE image_name = ANY(%s)", (names,))
    found = sorted({r[0] for r in registered})
    if not found:
        raise ValueError(
            f"none of these are registered in image_qa: {names[:3]}{'...' if len(names) > 3 else ''}. "
            "An unregistered seed yields an empty plan indistinguishable from 'nothing is affected'."
        )
    missing = sorted(set(names) - set(found))

    out = CascadePlan(seeds=found)
    if missing:
        out.skipped += [(m, "not registered in image_qa") for m in missing]

    rows = _closure(found, max_depth)
    seen = set(found)
    counts = {}
    master_names, science_ps_ids = [], set()
    counted = set()
    for name, image_type, ps_id, nightdate in rows:
        if name in seen:
            continue
        # count each name once, but keep every owning config for phase 3
        if name not in counted:
            counted.add(name)
            counts[image_type] = counts.get(image_type, 0) + 1
            if image_type in MASTER_IMAGE_TYPES:
                master_names.append((name, image_type))
        if image_type not in MASTER_IMAGE_TYPES:
            science_ps_ids.add(ps_id)
    out.counts = counts

    # ---- phase 1: the configs that can rebuild the chained masters ----
    out.masters = sorted(master_names, key=lambda m: (MASTER_IMAGE_TYPES.index(m[1]), m[0]))
    stem_of = _master_config_stems([n for n, _ in out.masters])
    master_runnable, blocked = _resolve_configs(stem_of.values())
    out.master_configs = sorted(master_runnable.items())
    out.unregenerable = blocked

    # ---- phase 2: the configs that must recalibrate the affected singles ----
    # a larger set than phase 1: a regenerated flat has no master descendants but stale singles
    affected_singles = free_query(
        _CLOSURE
        + """
        SELECT DISTINCT i.nightdate, i.unit
        FROM down JOIN image_qa i ON i.id = down.id
        WHERE i.image_type = %s AND i.nightdate IS NOT NULL AND i.unit IS NOT NULL
        """,
        (found, max_depth, IMAGE_TYPE_SINGLE),
    )
    single_stems = {f"{nightdate}_{unit}" for nightdate, unit in affected_singles}
    single_runnable, single_blocked = _resolve_configs(single_stems)
    out.preprocess_configs = sorted({**master_runnable, **single_runnable}.items())
    known_blocked = {s for s, _ in out.unregenerable}
    out.unregenerable += [(s, w) for s, w in single_blocked if s not in known_blocked]

    # ---- phase 3: the science configs ----
    # Primary source is the product graph: the configs that own something in the closure.
    science = {}
    if science_ps_ids:
        for name, config_file, nightdate, sanity in free_query(
            "SELECT name, config_file, nightdate, sanity FROM process_status"
            " WHERE id = ANY(%s) AND config_type = %s",
            (sorted(science_ps_ids), CONFIG_TYPE_SCIENCE),
        ):
            science[name] = (config_file, nightdate, sanity)

    # second source: configs whose singles carry no dependency edges at all (partial-edge: known-bugs.md)
    if out.preprocess_configs:
        for name, config_file, nightdate, sanity in free_query(
            "SELECT DISTINCT p.name, p.config_file, p.nightdate, p.sanity"
            " FROM process_status_dependency d"
            " JOIN process_status p ON p.name = d.derived_config_name"
            " WHERE d.source_config_name = ANY(%s) AND p.config_type = %s"
            "   AND EXISTS (SELECT 1 FROM image_qa i"
            "               WHERE i.process_status_id = p.id AND i.image_type = %s"
            "                 AND NOT EXISTS (SELECT 1 FROM image_qa_dependency e"
            "                                 WHERE e.derived_image_id = i.id))",
            ([n for n, _ in out.preprocess_configs], CONFIG_TYPE_SCIENCE, IMAGE_TYPE_SINGLE),
        ):
            science.setdefault(name, (config_file, nightdate, sanity))

    for name, (config_file, nightdate, sanity) in sorted(science.items()):
        if sanity is False:
            out.skipped.append((name, "human-rejected (sanity=False)"))
        elif not config_file or not os.path.exists(config_file):
            out.skipped.append((name, "config file is missing on disk"))
        elif config_file.startswith("/home/"):
            out.skipped.append((name, f"config under /home is unreadable by the queue daemon: {config_file}"))
        else:
            out.science_configs.append((name, config_file, nightdate))

    # ---- phase 5: the crossfilter configs ----
    # Primary: white rows in the closure own crossfilter process rows; secondary: the config
    # graph catches crossfilter children of rerun science configs whose white was never built.
    crossfilter = {}
    if science_ps_ids:
        for name, config_file, nightdate, sanity in free_query(
            "SELECT name, config_file, nightdate, sanity FROM process_status"
            " WHERE id = ANY(%s) AND config_type = %s",
            (sorted(science_ps_ids), CONFIG_TYPE_CROSSFILTER),
        ):
            crossfilter[name] = (config_file, nightdate, sanity)
    if out.science_configs:
        for name, config_file, nightdate, sanity in free_query(
            "SELECT DISTINCT p.name, p.config_file, p.nightdate, p.sanity"
            " FROM process_status_dependency d"
            " JOIN process_status p ON p.name = d.derived_config_name"
            " WHERE d.source_config_name = ANY(%s) AND p.config_type = %s",
            ([n for n, _, _ in out.science_configs], CONFIG_TYPE_CROSSFILTER),
        ):
            crossfilter.setdefault(name, (config_file, nightdate, sanity))

    for name, (config_file, nightdate, sanity) in sorted(crossfilter.items()):
        if sanity is False:
            out.skipped.append((name, "human-rejected (sanity=False)"))
        elif not config_file or not os.path.exists(config_file):
            out.skipped.append((name, "config file is missing on disk"))
        elif config_file.startswith("/home/"):
            out.skipped.append((name, f"config under /home is unreadable by the queue daemon: {config_file}"))
        else:
            out.crossfilter_configs.append((name, config_file, nightdate))

    return out


def _queued_rows(config_files):
    """[(config, status)] of system-queue rows whose config is in `config_files`."""
    import sqlite3

    from ..const import SCHEDULER_DB_PATH

    targets = set(config_files)
    if not targets or SCHEDULER_DB_PATH is None or not os.path.exists(SCHEDULER_DB_PATH):
        return []
    with sqlite3.connect(SCHEDULER_DB_PATH, timeout=10) as conn:
        rows = conn.execute("SELECT config, status FROM scheduler").fetchall()
    return [(c, s) for c, s in rows if c in targets]


def submit(
    plan: CascadePlan,
    phase: int,
    sweep: int = None,
    base_priority: int = 1,
    dry_run: bool = True,
    input_type: str = INPUT_TYPE_REPROCESS,
):
    """Queue ONE phase of the plan -- and for phase 1, ONE calib_types sweep.

    One drained batch per call: the scheduler dedupes on config path alone, so this refuses
    (RuntimeError) while any target config still has a Ready/Processing/Paused row, and
    replaces drained rows via overwrite_schedule (never signals a non-Processing task).
    Resubmit failed cascade tasks through here rather than rerun_failed_tasks: kwargs
    survive both since v1.10.44, but only this checks the drain discipline.
    dry_run=True (default) queues a write-nothing sizing pass -- a LOWER bound; phases 3-4
    have no sizing pass and raise on dry_run. Returns the Scheduler, or None if empty.
    """
    from .scheduler import Scheduler

    if phase == 1:
        if sweep not in (1, 2, 3):
            raise ValueError("phase 1 needs sweep=1, 2 or 3 (cumulative calib_types); run them in order")
        configs = [f for _, f in plan.master_configs]
        extra = ["-master_frame_only", "-calib_types", *MASTER_SWEEPS[sweep - 1]]
        priority, kw = base_priority + 1, {}
    elif phase == 2:
        configs = [f for _, f in plan.preprocess_configs]
        extra = []
        priority, kw = base_priority + 1, {}
    elif phase in (3, 4):
        # nightly (3) strictly before multi-epoch (4); -overwrite because sciproc skipping is flag-based
        if dry_run:
            raise ValueError(f"phase {phase} has no sizing pass; call with dry_run=False to queue the reruns")
        configs = [f for _, f, _ in (plan.nightly_science if phase == 3 else plan.multiepoch_science)]
        extra = []
        priority, kw = base_priority, {"overwrite_science": True}
    elif phase == 5:
        # after phases 3 and 4 drained; -overwrite because crossfilter skipping is flag-based too
        if dry_run:
            raise ValueError(f"phase {phase} has no sizing pass; call with dry_run=False to queue the reruns")
        configs = [f for _, f, _ in plan.crossfilter_configs]
        extra = []
        priority, kw = base_priority, {"overwrite_crossfilter": True}
    else:
        raise ValueError(
            f"phase must be 1, 2, 3 (nightly science), 4 (multi-epoch science) or 5 (crossfilter), not {phase!r}"
        )

    if not configs:
        return None

    busy = [
        (c, s) for c, s in _queued_rows(configs) if s in (TASK_STATUS_READY, TASK_STATUS_PROCESSING, TASK_STATUS_PAUSED)
    ]
    if busy:
        listing = ", ".join(f"{os.path.basename(c)} [{s}]" for c, s in busy[:5])
        raise RuntimeError(
            f"{len(busy)} target config(s) still queued ({listing}{', ...' if len(busy) > 5 else ''}); "
            "wait for the queue to drain before submitting the next batch"
        )

    if dry_run:  # phases 3-5 already raised
        extra = extra + ["-dry_run"]

    sc = Scheduler.from_list(
        configs,
        base_priority=priority,
        use_system_queue=True,
        overwrite_schedule=True,
        input_type=input_type,
        extra_kwargs=extra or None,
        **kw,
    )
    sc.start_system_queue()
    return sc
