"""Report which targets of a nightdate pass or hold WhiteImage's availability guard."""

import argparse
import tempfile

from pipeline.config import CrossFilterConfiguration
from pipeline.const import CONFIG_TYPE_SCIENCE
from pipeline.errors.errors import EmptyInputAfterSanityRejectionError, PrerequisiteNotMetError
from pipeline.imcoadd.white import WhiteImage
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
    return sorted(row[0] for row in rows)


def check_target(target, nightdate, working_dir):
    """Guard verdict for one target: ready / held / rejected, with the counts behind it."""
    science_configs = science_configs_of(target, nightdate)
    config = CrossFilterConfiguration(
        science_configs, write=False, logger=False, working_dir=working_dir, is_pipeline=False
    )
    # overwrite=True only to skip the version-escalation notice; the guard itself ignores it
    white = WhiteImage(config, overwrite=True)
    # constructed DB-free, then flipped so only the read-only RawFrameQuery branch of the guard runs
    config.node.settings.is_pipeline = True

    verdict, detail = "ready", ""
    try:
        white._confirm_input_completeness()
    except PrerequisiteNotMetError as error:
        verdict, detail = "held", str(error)
    except EmptyInputAfterSanityRejectionError as error:
        verdict, detail = "rejected", str(error)

    node = config.node
    return {
        "target": target,
        "verdict": verdict,
        "parents": len(science_configs),
        "filters": len(node.input.used_filters or []),
        "rejected": len(node.input.sanity_rejected_science_configs or []),
        "detail": detail,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preflight the cross-filter availability guard for a night, without writing anything",
        epilog=(
            "Every target of one night:\n"
            "  python run_examples/check_white_availability.py 2025-11-25\n"
            "One target:\n"
            "  python run_examples/check_white_availability.py 2025-11-25 --targets T08147\n"
            "Daily configs only; a multi-epoch target goes through run_examples/run_crossfilter.py.\n"
            "A busy Postgres can exhaust its connection slots and turn a target into 'error';\n"
            "re-run those by name, the check is read-only and repeatable.\n"
            "'filters' counts confirmed coadds and is only filled on a 'ready' verdict — the guard\n"
            "raises before it records them, so a held target reports the reason in its detail line.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("nightdates", nargs="+")
    parser.add_argument("--targets", nargs="+", help="Restrict to these targets instead of every target of the night")
    parser.add_argument(
        "--verdicts", nargs="+", help="Print only these verdicts (ready held rejected); errors always print"
    )
    args = parser.parse_args()

    results = []
    with tempfile.TemporaryDirectory(prefix="white-guard-") as working_dir:
        for nightdate in args.nightdates:
            targets = args.targets or targets_on_date(nightdate)
            print(f"\n{nightdate}: {len(targets)} target(s)", flush=True)
            for target in targets:
                try:
                    result = check_target(target, nightdate, working_dir)
                except Exception as error:
                    result = {
                        "target": target,
                        "verdict": "error",
                        "parents": 0,
                        "filters": 0,
                        "rejected": 0,
                        "detail": f"{type(error).__name__}: {error}",
                    }
                result["nightdate"] = nightdate
                results.append(result)
                if args.verdicts and result["verdict"] not in args.verdicts and result["verdict"] != "error":
                    continue
                counts = f"parents={result['parents']:>3} rejected={result['rejected']:>3}"
                if result["verdict"] == "ready":
                    counts += f" filters={result['filters']:>3}"
                print(f"  {result['target']:<12} {result['verdict']:<9} {counts}", flush=True)
                if result["detail"]:
                    print(f"    {result['detail']}", flush=True)

    print("\nSummary")
    for verdict in ("ready", "held", "rejected", "error"):
        count = sum(1 for result in results if result["verdict"] == verdict)
        print(f"  {verdict:<9} {count}")
