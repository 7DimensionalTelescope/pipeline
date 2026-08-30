"""Run the cross-filter chain for every target of past nightdates, one target at a time."""

import argparse
import os
import time

from pipeline.config import CrossFilterConfiguration
from pipeline.const import CONFIG_TYPE_SCIENCE
from pipeline.const.crossfilter import WHITE_COADD_SPEC
from pipeline.const.run import DEFAULT_CROSSFILTER_PROCESSES
from pipeline.errors.errors import EmptyInputAfterSanityRejectionError, PrerequisiteNotMetError
from pipeline.path.path import CrossFilterPathHandler
from pipeline.run import run_crossfilter_reduction
from pipeline.services.database import free_query


def targets_on_date(nightdate):
    rows = free_query(
        "SELECT DISTINCT object FROM process_status"
        " WHERE config_type = %s AND nightdate = %s AND config_file IS NOT NULL"
        " ORDER BY object",
        (CONFIG_TYPE_SCIENCE, nightdate),
    )
    return [row[0] for row in rows]


def science_configs_of(target, nightdate):
    """Newest science config per filter, the cross-filter parent set of one target and night."""
    rows = free_query(
        "SELECT DISTINCT ON (filter) config_file FROM process_status"
        " WHERE config_type = %s AND object = %s AND nightdate = %s AND config_file IS NOT NULL"
        " ORDER BY filter, updated_at DESC, id DESC",
        (CONFIG_TYPE_SCIENCE, target, nightdate),
    )
    return sorted(os.path.abspath(row[0]) for row in rows)


def load_or_create_config(science_configs, args):
    """The target's cross-filter config: the existing YAML when its parents match, otherwise a new one."""
    expected_coadds = CrossFilterConfiguration._science_config_coadds(science_configs)
    output_yml = CrossFilterPathHandler(
        expected_coadds,
        working_dir=args.working_dir,
        is_pipeline=args.pipeline,
        config_suffix=args.suffix,
    ).crossfilter.output_yml

    if os.path.exists(output_yml) and not args.overwrite_config:
        config = CrossFilterConfiguration(output_yml)
        existing = sorted(os.path.abspath(path) for path in (config.node.input.science_configs or []))
        if existing != science_configs:
            raise FileExistsError(f"{output_yml} has different parents; use --overwrite-config or --suffix")
        return config

    config = CrossFilterConfiguration(
        science_configs,
        working_dir=args.working_dir,
        overwrite=args.overwrite_config,
        is_pipeline=args.pipeline,
        config_suffix=args.suffix,
    )
    config.record_discovery(config.node.input.source_raw_images, "science_configs_db")
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill white coadds (and phot7ds) from the science configs of past nights",
        epilog=(
            "Plan first, then run in tmux:\n"
            "  python run_examples/run_white_coadd_backfill.py 2025-11-25\n"
            "  python run_examples/run_white_coadd_backfill.py 2025-11-25 --execute --white-only \\\n"
            "      2>&1 | tee 2026-08-30_white_backfill_tee.log\n"
            "Sandbox one target, writing nothing to the processed tree or Postgres:\n"
            "  python run_examples/run_white_coadd_backfill.py 2025-11-25 --targets T08147 \\\n"
            "      --no-pipeline --working-dir /tmp/white-backfill --execute\n"
            "Daily configs only; a multi-epoch target goes through run_examples/run_crossfilter.py.\n"
            "Re-running a night is cheap and idempotent: a stage whose flag is already True is\n"
            "skipped, so a target left 'failed' by a transient database error just runs again.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("nightdates", nargs="+")
    parser.add_argument("--targets", nargs="+", help="Restrict to these targets instead of every target of the night")
    parser.add_argument("--processes", nargs="+", default=DEFAULT_CROSSFILTER_PROCESSES)
    parser.add_argument("--white-only", action="store_true", help="Shorthand for --processes white_coadd")
    parser.add_argument("--execute", action="store_true", help="Without this, only the plan is printed")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cross-filter products")
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
    outcomes = []

    if not args.pipeline and args.working_dir:
        os.chdir(os.path.abspath(args.working_dir))

    for nightdate in args.nightdates:
        targets = args.targets or targets_on_date(nightdate)
        print(f"\n{nightdate}: {len(targets)} target(s), processes={processes}", flush=True)

        for target in targets:
            start = time.time()
            outcome, detail, config_file = "completed", "", ""
            try:
                science_configs = science_configs_of(target, nightdate)
                if not args.execute:
                    # load_or_create_config writes a YAML, so the plan stops at discovery
                    outcome, detail = "planned", f"{len(science_configs)} science parent(s)"
                else:
                    config = load_or_create_config(science_configs, args)
                    config_file = config.config_file
                    run_crossfilter_reduction(config, processes=processes, overwrite=args.overwrite)
            except PrerequisiteNotMetError as error:
                outcome, detail = "held", str(error)
            except EmptyInputAfterSanityRejectionError as error:
                outcome, detail = "rejected", str(error)
            except Exception as error:
                outcome, detail = "failed", f"{type(error).__name__}: {error}"

            outcomes.append((nightdate, target, outcome))
            print(f"  {target:<12} {outcome:<9} {time.time() - start:7.1f}s  {config_file}", flush=True)
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
