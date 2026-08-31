"""Regenerate the daily coadds of a target-night, then build its white image and run phot7ds."""

import argparse
import inspect
import os
import time

from pipeline.config import CrossFilterConfiguration, SciProcConfiguration
from pipeline.config.utils import get_key
from pipeline.const import ALL_FILTERS, CONFIG_TYPE_CROSSFILTER, CONFIG_TYPE_SCIENCE, WHITE_FILTER
from pipeline.const.crossfilter import PHOT7DS_SPEC, WHITE_COADD_SPEC
from pipeline.const.run import DEFAULT_CROSSFILTER_PROCESSES
from pipeline.const.sciproc import (
    COADD_PHOTOMETRY_SPEC,
    COADD_SPEC,
    DIFFERENCE_PHOTOMETRY_SPEC,
    SCIPROCESS_REGISTRY,
    SINGLE_PHOTOMETRY_SPEC,
    SUBTRACTION_SPEC,
)
from pipeline.errors.errors import EmptyInputAfterSanityRejectionError, PrerequisiteNotMetError
from pipeline.imcoadd.white import WhiteImage
from pipeline.path.path import CrossFilterPathHandler
from pipeline.run import run_crossfilter_reduction, run_scidata_reduction
from pipeline.services.database import free_query
from pipeline.services.pipeline_lock import enforce_pipeline_lock
from pipeline.version import MIN_SCIPROC_RUNTIME_VERSION_MAP, is_below_min

# parents come from the same filter set WhiteImage counts, so widening the guard widens the campaign
_counted = inspect.signature(WhiteImage._confirm_input_completeness).parameters.get("counted_filters")
COUNTED_FILTERS = sorted(_counted.default) if _counted else sorted(ALL_FILTERS)
# an error or warning from these stages is not evidence against the parent's single_photometry
REGENERATED_PROCESS_CODES = [
    spec.error_code for spec in SCIPROCESS_REGISTRY.specs if spec.progress_start >= COADD_SPEC.progress_start
]


def ready_target_nights(nightdates=None, targets=None, minimum_filters=3, warning_free=False):
    """Target-nights whose every raw-observed counted filter has a clean config through single_photometry."""
    raw_where = ["f.name = ANY(%s)", "COALESCE(tl.name, t.name) IS NOT NULL"]
    raw_params = [COUNTED_FILTERS]
    status_where = ["config_type = %s", "config_file IS NOT NULL", "filter = ANY(%s)"]
    status_params = [CONFIG_TYPE_SCIENCE, COUNTED_FILTERS]
    if nightdates:
        raw_where.append("n.date = ANY(%s)")
        raw_params.append(list(nightdates))
        status_where.append("nightdate = ANY(%s)")
        status_params.append(list(nightdates))
    if targets:
        raw_where.append("COALESCE(tl.name, t.name) = ANY(%s)")
        raw_params.append(list(targets))
        status_where.append("object = ANY(%s)")
        status_params.append(list(targets))

    ready_clauses = [
        "l.config_file IS NOT NULL",
        "l.progress >= %s",
        "(l.errors IS NULL OR l.errors / 100 = ANY(%s))",
    ]
    ready_params = [SINGLE_PHOTOMETRY_SPEC.progress_end, REGENERATED_PROCESS_CODES]
    if warning_free:
        ready_clauses.append(
            "NOT EXISTS (SELECT 1 FROM jsonb_array_elements_text(COALESCE(l.warnings, '[]'::jsonb)) w"
            " WHERE (w::int) / 100 <> ALL(%s))"
        )
        ready_params.append(REGENERATED_PROCESS_CODES)

    rows = free_query(
        "WITH observed AS ("
        "  SELECT COALESCE(tl.name, t.name) AS object, n.date AS nightdate, f.name AS filter"
        "    FROM survey_scienceframe sf"
        "    JOIN survey_night n ON sf.night_id = n.id"
        "    LEFT JOIN facility_filter f ON sf.filter_id = f.id"
        "    LEFT JOIN survey_target t ON sf.target_id = t.id"
        "    LEFT JOIN survey_tile tl ON sf.tile_id = tl.id"
        f"   WHERE {' AND '.join(raw_where)}"
        "   GROUP BY 1, 2, 3),"
        " latest AS ("
        "  SELECT DISTINCT ON (object, nightdate, filter)"
        "         object, nightdate, filter, progress, errors, warnings, config_file"
        f"    FROM process_status WHERE {' AND '.join(status_where)}"
        "   ORDER BY object, nightdate, filter, updated_at DESC, id DESC),"
        " joined AS ("
        "  SELECT o.object, o.nightdate, o.filter, l.config_file,"
        f"        ({' AND '.join(ready_clauses)}) AS ready"
        "    FROM observed o"
        "    LEFT JOIN latest l ON l.object = o.object AND l.nightdate = o.nightdate AND l.filter = o.filter)"
        " SELECT object, nightdate, count(*), array_agg(config_file ORDER BY filter) FILTER (WHERE ready)"
        "   FROM joined GROUP BY object, nightdate"
        "  HAVING count(*) = count(*) FILTER (WHERE ready) AND count(*) >= %s"
        "  ORDER BY nightdate DESC, object",
        raw_params + status_params + ready_params + [minimum_filters],
        statement_timeout_ms=600000,
    )
    return [
        {
            "target": row[0],
            "nightdate": str(row[1]),
            "observed_filters": row[2],
            "science_configs": sorted(os.path.abspath(path) for path in row[3]),
        }
        for row in rows
    ]


def completed_target_nights(suffix=None):
    """Target-nights whose cross-filter config of THIS identity already completed phot7ds."""
    rows = free_query(
        "SELECT DISTINCT object, nightdate, config_file FROM process_status"
        " WHERE config_type = %s AND status = %s AND config_file IS NOT NULL",
        (CONFIG_TYPE_CROSSFILTER, f"{PHOT7DS_SPEC.name}-completed"),
    )
    # white_photometry ends at progress 100 too, and a suffixed config must not mask the default one
    tail = f"_{suffix}" if suffix else ""
    done = set()
    for object_name, nightdate, config_file in rows:
        stem = os.path.splitext(os.path.basename(config_file))[0]
        if stem == f"{object_name}_{WHITE_FILTER}_{nightdate}{tail}":
            done.add((object_name, str(nightdate)))
    return done


def parent_plan(science_configs):
    """Per parent: whether single_photometry is flagged, and whether its coadd is below the version floor."""
    plan = []
    for config_file in science_configs:
        node = SciProcConfiguration(config_file, write=False, logger=False, verbose=False).node
        plan.append(
            {
                "config_file": config_file,
                "single_photometry": bool(get_key(node.flag, SINGLE_PHOTOMETRY_SPEC.name, False)),
                "coadd": bool(get_key(node.flag, COADD_SPEC.name, False)),
                "coadd_outdated": is_below_min(
                    get_key(node.imcoadd, "runtime_version"),
                    MIN_SCIPROC_RUNTIME_VERSION_MAP[COADD_SPEC.config_section],
                ),
                # the photometry stamp is shared with single_photometry, so it is tested on its own stage
                "photometry_outdated": is_below_min(
                    get_key(node.photometry, "runtime_version"),
                    MIN_SCIPROC_RUNTIME_VERSION_MAP[COADD_PHOTOMETRY_SPEC.config_section],
                ),
            }
        )
    return plan


def invalidate_difference_stages(config_file):
    """A replaced coadd leaves its difference products undescended, so their flags are cleared for a rerun."""
    config = SciProcConfiguration(config_file, logger=False, verbose=False)
    for spec in (SUBTRACTION_SPEC, DIFFERENCE_PHOTOMETRY_SPEC):
        if get_key(config.node.flag, spec.name, False):
            setattr(config.node.flag, spec.name, False)


def coadd_parents(plan, overwrite=False):
    """Coadd and coadd-photometer every parent, per stage, regenerating what is below its version floor."""
    done, rejected, failed = 0, 0, []
    for entry in plan:
        redo_coadd = overwrite or entry["coadd_outdated"]
        try:
            run_scidata_reduction(entry["config_file"], processes=[COADD_SPEC.name], overwrite=redo_coadd)
            run_scidata_reduction(
                entry["config_file"],
                processes=[COADD_PHOTOMETRY_SPEC.name],
                overwrite=overwrite or entry["photometry_outdated"],
            )
            if redo_coadd:
                invalidate_difference_stages(entry["config_file"])
            done += 1
        except EmptyInputAfterSanityRejectionError:
            # a sanity-rejected parent is evidence the white guard accepts, not a campaign failure
            rejected += 1
        except Exception as error:
            failed.append(f"{os.path.basename(entry['config_file'])}: {type(error).__name__}: {error}")
    return done, rejected, failed


def load_or_create_config(science_configs, working_dir=None, is_pipeline=True, suffix=None, overwrite=False):
    """The target's cross-filter config, its parent list rewritten whenever the campaign's set differs."""
    expected_coadds = CrossFilterConfiguration._science_config_coadds(science_configs)
    output_yml = CrossFilterPathHandler(
        expected_coadds,
        working_dir=working_dir,
        is_pipeline=is_pipeline,
        config_suffix=suffix,
    ).crossfilter.output_yml

    if os.path.exists(output_yml) and not overwrite:
        config = CrossFilterConfiguration(output_yml)
        existing = {os.path.abspath(path) for path in (config.node.input.science_configs or [])}
        # a wider declared list (Blueprint's, with the uncounted filters) is left alone: rewriting it
        # would clear the white flags on every pass and the guard cuts those filters anyway
        if not set(science_configs).issubset(existing):
            config.set_science_configs(science_configs)
        return config

    config = CrossFilterConfiguration(
        science_configs,
        working_dir=working_dir,
        overwrite=overwrite,
        is_pipeline=is_pipeline,
        config_suffix=suffix,
    )
    config.record_discovery(config.node.input.source_raw_images, "science_configs_db")
    return config


def run_target_night(entry, args, processes):
    """Coadds first, then the cross-filter config and its chain; returns the config file and a detail line."""
    plan = parent_plan(entry["science_configs"])
    unready = [item["config_file"] for item in plan if not item["single_photometry"]]
    if unready:
        raise PrerequisiteNotMetError(f"{len(unready)} parent(s) have no single_photometry flag: {unready[:3]}")

    if args.crossfilter_only:
        detail = f"{len(plan)} parent(s), coadd step skipped"
    else:
        outdated = sum(1 for item in plan if item["coadd_outdated"])
        done, rejected, failed = coadd_parents(plan, overwrite=args.overwrite_coadd)
        detail = f"coadd {done}/{len(plan)} ok ({outdated} regenerated), {rejected} rejected, {len(failed)} failed"
        if failed:
            raise RuntimeError(f"{len(failed)} of {len(plan)} parent coadd(s) failed: {failed[0]}")

    config = load_or_create_config(
        entry["science_configs"],
        working_dir=args.working_dir,
        is_pipeline=args.pipeline,
        suffix=args.suffix,
        overwrite=args.overwrite_config,
    )
    run_crossfilter_reduction(config, processes=processes, overwrite=args.overwrite)
    return config.config_file, detail


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Coadd a target-night's ready science configs, then build its white image and run phot7ds",
        epilog=(
            "Plan the whole backlog, newest night first:\n"
            "  python run_examples/run_white_campaign.py\n"
            "Ten target-nights in tmux:\n"
            "  python run_examples/run_white_campaign.py --limit 10 --execute \\\n"
            "      2>&1 | tee 2026-08-30_white_campaign_tee.log\n"
            "Sandbox the cross-filter half of one target, writing nothing to the processed tree:\n"
            "  python run_examples/run_white_campaign.py --nightdates 2026-07-12 --targets T21659 \\\n"
            "      --crossfilter-only --no-pipeline --working-dir /tmp/white-campaign --execute\n"
            "The coadd step always rewrites the parents' own production configs and products, so\n"
            "--working-dir sandboxes the white products only and --no-pipeline requires\n"
            "--crossfilter-only.\n"
            "A target-night is skipped only when a cross-filter config of the SAME identity (suffix\n"
            "included) has status phot7ds-completed; white_photometry also ends at progress 100, so\n"
            "progress alone would mask it. A regenerated coadd clears its parent's subtraction and\n"
            "difference_photometry flags, since those products no longer descend from it.\n"
            "Selection is anchored in the raw database: every counted filter observed for the\n"
            "target-night must have a config past single_photometry whose recorded error, if any,\n"
            "belongs to a stage this campaign regenerates. Coadds below the version floor in\n"
            "pipeline/version.py are regenerated; target-nights already past phot7ds are skipped.\n"
            "Re-running is idempotent: a current stage whose flag is True is skipped, so a target\n"
            "left 'failed' by a transient database error just runs again.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--nightdates", nargs="+", help="Restrict to these nights instead of the whole backlog")
    parser.add_argument("--targets", nargs="+", help="Restrict to these targets")
    parser.add_argument("--limit", type=int, help="Process only the first N ready target-nights")
    parser.add_argument(
        "--warning-free",
        action="store_true",
        help="Also require no astrometry or single_photometry warning on any parent",
    )
    parser.add_argument("--processes", nargs="+", default=DEFAULT_CROSSFILTER_PROCESSES)
    parser.add_argument("--white-only", action="store_true", help="Shorthand for --processes white_coadd")
    parser.add_argument("--crossfilter-only", action="store_true", help="Skip the parent coadds; white products only")
    parser.add_argument("--execute", action="store_true", help="Without this, only the plan is printed")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cross-filter products")
    parser.add_argument("--overwrite-coadd", action="store_true", help="Regenerate every coadd, not only stale ones")
    parser.add_argument(
        "--overwrite-config",
        action="store_true",
        help="Rewrite the cross-filter YAML from the base, clearing flag: and the per-stage runtime_version:",
    )
    parser.add_argument("--suffix", help="Alternate cross-filter config and product identity")
    parser.add_argument("--working-dir")
    parser.add_argument("--pipeline", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    processes = [WHITE_COADD_SPEC.name] if args.white_only else list(args.processes)
    if not args.pipeline and not args.crossfilter_only:
        parser.error("--no-pipeline needs --crossfilter-only: the coadd step writes the parents' production products")
    if not args.pipeline and not args.working_dir:
        parser.error("--no-pipeline needs --working-dir: without it the white products land in the processed tree")
    enforce_pipeline_lock(action="white campaign", requested_is_pipeline=args.pipeline)

    if not args.pipeline and args.working_dir:
        os.chdir(os.path.abspath(args.working_dir))

    entries = ready_target_nights(args.nightdates, args.targets, warning_free=args.warning_free)
    done = set() if args.overwrite else completed_target_nights(args.suffix)
    remaining = [e for e in entries if (e["target"], e["nightdate"]) not in done]
    print(
        f"{len(entries)} ready target-night(s), {len(entries) - len(remaining)} already past phot7ds,"
        f" {len(COUNTED_FILTERS)} counted filters, processes={processes}",
        flush=True,
    )
    if args.limit:
        print(f"Limiting to the first {args.limit} of {len(remaining)}", flush=True)
        remaining = remaining[: args.limit]

    outcomes = []
    for entry in remaining:
        start = time.time()
        outcome, detail, config_file = "completed", "", ""
        try:
            if not args.execute:
                outcome = "planned"
                detail = f"{entry['observed_filters']} observed filter(s), {len(entry['science_configs'])} parent(s)"
            else:
                config_file, detail = run_target_night(entry, args, processes)
        except PrerequisiteNotMetError as error:
            outcome, detail = "held", str(error)
        except EmptyInputAfterSanityRejectionError as error:
            outcome, detail = "rejected", str(error)
        except Exception as error:
            outcome, detail = "failed", f"{type(error).__name__}: {error}"

        outcomes.append((entry["nightdate"], entry["target"], outcome))
        print(
            f"  {entry['nightdate']}  {entry['target']:<14} {outcome:<9} {time.time() - start:8.1f}s  {config_file}",
            flush=True,
        )
        if detail:
            print(f"    {detail}", flush=True)

    print("\nSummary")
    for outcome in ("planned", "completed", "held", "rejected", "failed"):
        count = sum(1 for _, _, value in outcomes if value == outcome)
        if count:
            print(f"  {outcome:<9} {count}")
    for nightdate, target, outcome in outcomes:
        if outcome == "failed":
            print(f"  failed: {nightdate} {target}")

    raise SystemExit(1 if any(outcome == "failed" for _, _, outcome in outcomes) else 0)
