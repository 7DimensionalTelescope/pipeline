"""Create the cross-filter configs of past nightdates, with optional dependency mirroring and queue submission."""

import argparse
import os

from pipeline.config import CrossFilterConfiguration
from pipeline.const import INPUT_TYPE_REPROCESS
from pipeline.const.crossfilter import WHITE_COADD_SPEC
from pipeline.imcoadd.white import WhiteImage
from pipeline.services.scheduler import Scheduler
from pipeline.services.utils import CrossFilterGroup
from pipeline.wrapper import DataReduction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill cross-filter configs for nights processed before they existed",
        epilog=(
            "Plan, then create the YAMLs and record their parents:\n"
            "  python run_examples/backfill_crossfilter_configs.py 2025-11-25\n"
            "  python run_examples/backfill_crossfilter_configs.py 2025-11-25 --execute --mirror \\\n"
            "      2>&1 | tee 2026-08-30_crossfilter_configs_tee.log\n"
            "Hand the white coadds to the queue daemon as well:\n"
            "  python run_examples/backfill_crossfilter_configs.py 2025-11-25 --execute --mirror --queue\n"
            "\n"
            "--execute goes through Blueprint.create_config, the same path as\n"
            "pipeline/cli/run_date <dates> --crossfilter-config-only (which already takes a\n"
            "comma-separated date list). That path also touches the night's upstream configs:\n"
            "each preprocess YAML is rebuilt from ref/preproc_base.yml and rewritten (so a manual\n"
            "preprocess.designated_masterframes entry is lost), each science YAML is re-opened,\n"
            "and a cross-filter YAML whose recorded science_configs no longer match the raw\n"
            "inventory is rewritten by set_science_configs, which clears its flag: booleans and\n"
            "sets input.parents_changed, so the next run redoes the white image.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("nightdates", nargs="+")
    parser.add_argument("--execute", action="store_true", help="Without this, only the plan is printed")
    parser.add_argument("--overwrite-config", action="store_true", help="Rewrite existing cross-filter YAMLs")
    parser.add_argument("--mirror", action="store_true", help="Record the declared science parents in Postgres")
    parser.add_argument("--queue", action="store_true", help="Add the cross-filter configs to the system queue")
    parser.add_argument("--white-only", action="store_true", help="Queue white_coadd without phot7ds")
    parser.add_argument("--overwrite-data", action="store_true", help="Queue the rows with -overwrite")
    parser.add_argument("--base-priority", type=int, default=1)
    parser.add_argument("--input-type", default=INPUT_TYPE_REPROCESS)
    parser.add_argument("--suffix", help="Alternate cross-filter config and product identity")
    args = parser.parse_args()

    created = []

    for nightdate in args.nightdates:
        dr = DataReduction(
            [nightdate],
            use_db=True,
            is_pipeline=True,
            enable_crossfilter=True,
            crossfilter_suffix=args.suffix,
        )
        groups = [group for group in dr.blueprint.groups.values() if isinstance(group, CrossFilterGroup)]
        print(f"\n{nightdate}: {len(groups)} cross-filter group(s)", flush=True)
        for group in groups:
            # reading group.config would create the YAML; key and source count are the whole plan
            print(f"  {group.key}: {len(group.source_groups)} science group(s)", flush=True)

        if not args.execute:
            continue

        dr.create_config(overwrite=False, overwrite_crossfilter=args.overwrite_config)
        for config_file in dr.crossfilter_configs:
            config = CrossFilterConfiguration(config_file, write=False, logger=False)
            white_image = config.node.input.white_image
            state = "white image present" if white_image and os.path.exists(white_image) else "no white image yet"
            edges = WhiteImage.record_config_dependencies(config.node, None) if args.mirror else None
            created.append(config_file)
            mirrored = f"{edges} parent edge(s)" if args.mirror else "parents not mirrored"
            print(f"  {os.path.basename(config_file):<40} {state}, {mirrored}", flush=True)

    if not args.execute:
        print("\nPlan only, nothing created — pass --execute")
    elif args.queue and created:
        processes = [WHITE_COADD_SPEC.name] if args.white_only else None
        scheduler = Scheduler.from_list(
            created,
            base_priority=args.base_priority,
            use_system_queue=True,
            input_type=args.input_type,
            crossfilter_processes=processes,
            overwrite_crossfilter=args.overwrite_data,
        )
        queued = {str(row["config"]) for row in scheduler.schedule} & set(created)
        scheduler.start_system_queue()
        print(f"\n{len(created)} cross-filter config(s) created, {len(queued)} in the scheduler DB")
    else:
        print(f"\n{len(created)} cross-filter config(s) created")
