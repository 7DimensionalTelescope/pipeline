"""Plan the reprocessing cascade of one or more regenerated master frames.

A regenerated master carries a fresh IMAGEID, which invalidates every product built from
it: other master frames first, then the calibrated singles, then the coadds and diffs above
them. This module answers "what must rerun, in what order" from `image_qa_dependency`, and
optionally submits it. It never decides that something is stale -- `Preprocess` does that
per frame from the IMCID cards ([api-recipes.md] `ingredient_change`); this only routes the
work.

Three phases, and the barriers between them are the whole point:

    1. masters   the affected preprocess configs, master_frame_only, in three CUMULATIVE
                 calib_types sweeps: [bias], [bias,dark], [bias,dark,flat]
    2. singles   the same preprocess configs, normal run, overwrite=False
    3. science   the affected science configs, overwrite=True, nightly before multi-epoch

Why phase 1 cannot be folded into phase 2: a config both consumes masters from other nights
and produces its own, and `Preprocess.run` interleaves master generation with science
reduction inside each group. Night A's flat may be built from night B's bias while night B's
singles use night A's flat, so the order is B-masters -> A-flat -> science-of-both, which no
single pass over configs can produce.

Why the sweeps are cumulative rather than one kind each: kind order (bias -> dark -> flat) is
the real topological order -- shortest-path depth is not, because a flat records its bias
directly as well as through the dark -- but a later kind needs the earlier ones *loaded* for
PPFLAG to propagate. An already-built bias is never rebuilt in sweeps 2 and 3, so the repeat
costs one tolerant_search per group.

Read-only unless `submit()` is called explicitly (owner's rule: dependency bookkeeping must
not touch run configs).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

from ..path.path import PathHandler
from ..utils import atleast_1d
from .database.query import free_query
from .database.recipes import image_names

# Downward closure over the dependency graph. Depth-bounded rather than visited-bounded:
# UNION dedupes ids but one id can legitimately reappear deeper, so the cap is what
# terminates a cycle. The real graph is 5 deep at most, so 12 never binds.
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

MASTER_SWEEPS = (["bias"], ["bias", "dark"], ["bias", "dark", "flat"])


@dataclass
class CascadePlan:
    """What a set of regenerated masters obliges, and in what order."""

    seeds: List[str] = field(default_factory=list)
    masters: List[tuple] = field(default_factory=list)  # (image_name, image_type)
    master_configs: List[tuple] = field(default_factory=list)  # phase 1: (name, config_file)
    preprocess_configs: List[tuple] = field(default_factory=list)  # phase 2: (name, config_file)
    science_configs: List[tuple] = field(default_factory=list)  # phase 3: (name, config_file, nightdate)
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
            + (" (" + ", ".join(f"{sum(1 for m in self.masters if m[1] == k)} {k}" for k in ("bias", "dark", "flat")) + ")" if self.masters else ""),  # fmt: skip
            f"  phase 1  masters    {len(self.master_configs)} preprocess configs × 3 sweeps",
            f"  phase 2  singles    {len(self.preprocess_configs)} preprocess configs",
            f"  phase 3  science    {len(self.science_configs)} configs "
            f"({len(self.nightly_science)} nightly, {len(self.multiepoch_science)} multi-epoch)",
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
    """{master name: preprocess config stem}, from the names alone.

    Derived from the name, not from `image_qa.process_status_id`: a master fetched by another
    night can have its row taken over by the fetching config (520 such rows live), and only
    the config for the master's own (nightdate, unit) holds the raw calibration frames that
    can rebuild it. The basename carries both, and PathHandler builds the stem exactly the way
    the config itself was named. One batched parse, no database.
    """
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
            " WHERE name = ANY(%s) AND config_type = 'preprocess'",
            (stems,),
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
        else:
            runnable[stem] = config_file
    return runnable, blocked


def plan(seed_images, max_depth: int = 12) -> CascadePlan:
    """Everything a set of regenerated master frames obliges. Read-only.

    `seed_images` are the masters that were ALREADY regenerated -- names or paths. They are
    not themselves scheduled; their consumers are.
    """
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
    master_names, science_ps_ids, single_stems = [], set(), set()
    for name, image_type, ps_id, nightdate in rows:
        if name in seen:
            continue
        counts[image_type] = counts.get(image_type, 0) + 1
        if image_type in ("bias", "dark", "flat"):
            master_names.append((name, image_type))
        else:
            science_ps_ids.add(ps_id)
    out.counts = counts

    # ---- phase 1: the configs that can rebuild the chained masters ----
    out.masters = sorted(master_names, key=lambda m: (("bias", "dark", "flat").index(m[1]), m[0]))
    stem_of = _master_config_stems([n for n, _ in out.masters])
    master_runnable, blocked = _resolve_configs(stem_of.values())
    out.master_configs = sorted(master_runnable.items())
    out.unregenerable = blocked

    # ---- phase 2: the configs that must recalibrate the affected singles ----
    # A different, usually larger set than phase 1: a regenerated FLAT has no master
    # descendants at all, so phase 1 is empty for it, yet every single it calibrated must be
    # reduced again. Nothing else does that -- the science stages run on the singles that
    # exist, they do not rebuild pixels. Stems come from image_qa's own nightdate and unit,
    # which for a single is its real unit (unlike a coadd's, which is only a naming token).
    affected_singles = free_query(
        _CLOSURE
        + """
        SELECT DISTINCT i.nightdate, i.unit
        FROM down JOIN image_qa i ON i.id = down.id
        WHERE i.image_type = 'single' AND i.nightdate IS NOT NULL AND i.unit IS NOT NULL
        """,
        (found, max_depth),
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
            " WHERE id = ANY(%s) AND config_type = 'science'",
            (sorted(science_ps_ids),),
        ):
            science[name] = (config_file, nightdate, sanity)

    # Second source, deliberately narrow: a config under an affected preprocess config whose
    # singles carry NO dependency edges at all is invisible to the walk above (0.17% of
    # singles live). Taking every child of an affected preprocess config instead would be a
    # gross over-reach -- measured, one flat seed goes from 26 configs to 128, and the other
    # 102 own nothing that changed.
    if out.preprocess_configs:
        for name, config_file, nightdate, sanity in free_query(
            "SELECT DISTINCT p.name, p.config_file, p.nightdate, p.sanity"
            " FROM process_status_dependency d"
            " JOIN process_status p ON p.name = d.derived_config_name"
            " WHERE d.source_config_name = ANY(%s) AND p.config_type = 'science'"
            "   AND EXISTS (SELECT 1 FROM image_qa i"
            "               WHERE i.process_status_id = p.id AND i.image_type = 'single'"
            "                 AND NOT EXISTS (SELECT 1 FROM image_qa_dependency e"
            "                                 WHERE e.derived_image_id = i.id))",
            ([n for n, _ in out.preprocess_configs],),
        ):
            science.setdefault(name, (config_file, nightdate, sanity))

    for name, (config_file, nightdate, sanity) in sorted(science.items()):
        if sanity is False:
            out.skipped.append((name, "human-rejected (sanity=False)"))
        elif not config_file or not os.path.exists(config_file):
            out.skipped.append((name, "config file is missing on disk"))
        else:
            out.science_configs.append((name, config_file, nightdate))

    return out


def submit(
    plan: CascadePlan,
    phases=(1, 2, 3),
    base_priority: int = 1,
    dry_run: bool = True,
    input_type: str = "Reprocess",
):
    """Queue the plan. `dry_run=True` (the default) queues sizing passes that write nothing.

    Returns the list of `Scheduler` objects it created, in submission order. Phases are
    submitted one call at a time on purpose: **phase N+1 must not be queued until phase N has
    drained**, because the queue runs tasks concurrently and a science config calibrated
    against a master that is still being rebuilt is exactly the corruption this exists to
    prevent. There is no cross-config dependency mechanism to lean on -- `dependent_idx` is
    only ever preprocess->science within one blueprint.

    `dry_run` reaches the preprocess phases only. `cli/data_reduction` has no such flag and
    would reject it; phase 3 has no sizing pass, because the plan's config count already is
    one -- a science rerun is all-or-nothing per config.
    """
    from .scheduler import Scheduler

    master_files = [f for _, f in plan.master_configs]
    preprocess_files = [f for _, f in plan.preprocess_configs]
    science_files = [f for _, f, _ in plan.nightly_science] + [f for _, f, _ in plan.multiepoch_science]
    submitted = []

    def _queue(configs, extra, priority, **kw):
        if not configs:
            return None
        sc = Scheduler.from_list(
            configs,
            base_priority=priority,
            use_system_queue=True,
            input_type=input_type,
            extra_kwargs=list(extra) or None,
            **kw,
        )
        sc.start_system_queue()
        submitted.append(sc)
        return sc

    sizing = ["-dry_run"] if dry_run else []
    if 1 in phases:
        for sweep in MASTER_SWEEPS:
            _queue(master_files, ["-master_frame_only", "-calib_types", *sweep] + sizing, base_priority + 1)
    if 2 in phases:
        _queue(preprocess_files, sizing, base_priority + 1)
    if 3 in phases and not dry_run:
        # overwrite=True: sciproc skipping is flag-based and will not notice an IMCID change.
        _queue(science_files, [], base_priority, overwrite_science=True)

    return submitted
