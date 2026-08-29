import argparse
import os

from pipeline.config import CrossFilterConfiguration
from pipeline.path.name import NameHandler
from pipeline.services.database import RawFrameQuery
from pipeline.services.scheduler import Scheduler
from pipeline.utils import atleast_1d
from pipeline.wrapper import DataReduction


def query_raw(target, nightdate, target_field):
    query = RawFrameQuery().on_date(nightdate)
    query = {
        "target": query.for_target,
        "tile": query.for_tile,
        "object": query.object_name_contains,
    }[target_field](target)
    query.fetch()
    return query.files()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preview preproc -> sciproc -> crossfilter dependencies in memory")
    parser.add_argument("nightdate")
    parser.add_argument("--target")
    parser.add_argument("--target-field", choices=("target", "tile", "object"), default="target")
    parser.add_argument("--filters", nargs="+")
    parser.add_argument("--raw-files", nargs="+")
    parser.add_argument("--suffix")
    parser.add_argument("--pipeline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--working-dir")
    parser.add_argument("--overwrite-config", action="store_true")
    args = parser.parse_args()

    if not args.pipeline and args.working_dir:
        os.chdir(os.path.abspath(args.working_dir))

    if args.raw_files:
        raw_files = args.raw_files
    else:
        if not args.target:
            parser.error("--target is required unless --raw-files is supplied")
        raw_files = query_raw(args.target, args.nightdate, args.target_field)
    if not raw_files:
        raise RuntimeError("No raw inputs found")

    discovered_filters = sorted(set(atleast_1d(NameHandler(raw_files).filter)))
    if args.filters and discovered_filters != sorted(set(args.filters)):
        raise RuntimeError(f"Expected filters {sorted(set(args.filters))}, discovered {discovered_filters}")

    reduction = DataReduction(
        list_of_images=raw_files,
        use_db=False,
        is_pipeline=args.pipeline,
        enable_crossfilter=True,
        crossfilter_suffix=args.suffix,
    )
    reduction.create_config(overwrite=args.overwrite_config)
    reduction.blueprint.create_schedule(input_type="Daily")
    schedule = reduction.schedule
    scheduler = Scheduler(schedule.copy(), use_system_queue=False)

    cross_rows = schedule[schedule["config_type"] == "crossfilter"]
    if not len(cross_rows):
        raise RuntimeError("No cross-filter row was generated")
    for cross_row in cross_rows:
        cross_index = int(cross_row["index"])
        parents = [row for row in schedule if cross_index in list(row["dependent_idx"] or [])]
        science_parents = [row for row in parents if row["config_type"] == "science"]
        config = CrossFilterConfiguration(cross_row["config"], write=False, logger=False)
        scheduled = {row["config"] for row in science_parents}
        expected = set(config.node.input.science_configs)
        if scheduled != expected:
            raise AssertionError(f"Dependency mismatch: scheduled={scheduled}, expected={expected}")
        if cross_row["is_ready"] or int(cross_row["readiness"]) != 100 - len(science_parents):
            raise AssertionError("Cross-filter row was not held behind every science parent")

        failure_probe = Scheduler(schedule.copy(), use_system_queue=False)
        failure_probe.mark_done(int(science_parents[0]["index"]), return_code=1)
        held = failure_probe.schedule[failure_probe.schedule["index"] == cross_index][0]
        if held["is_ready"] or int(held["readiness"]) != int(cross_row["readiness"]):
            raise AssertionError("A failed science parent incorrectly released the cross-filter row")

        rejected_probe = Scheduler(schedule.copy(), use_system_queue=False)
        rejected_probe.mark_done(int(science_parents[0]["index"]), return_code=2)
        for row in science_parents[1:]:
            rejected_probe.mark_done(int(row["index"]), return_code=0)
        released_after_rejection = rejected_probe.schedule[
            rejected_probe.schedule["index"] == cross_index
        ][0]
        if not released_after_rejection["is_ready"]:
            raise AssertionError("A sanity-rejected science parent did not count as resolved")

        print(f"\n{config.node.name}: {len(science_parents)} science parents")
        for row in science_parents:
            print(f"  {row['index']:>3} -> {cross_index:>3}  {row['config']}")
        for row in science_parents:
            scheduler.mark_done(int(row["index"]), return_code=0)
        final = scheduler.schedule[scheduler.schedule["index"] == cross_index][0]
        if not final["is_ready"] or final["status"] != "Ready":
            raise AssertionError("Completion notices from all science parents did not release the cross-filter row")
        print(f"  released in memory: status={final['status']}, readiness={final['readiness']}")

    print("\nNo system-queue database was opened or modified.")
