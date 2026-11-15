#!/usr/bin/env python
"""
Refresh the stored master-association plan (pipeline.masterframe_plan, masterframe_plan_master), whole units.

One process per unit when several are given; without units, every unit whose plan is missing or stale.

Usage:
    python run_refresh_masterframe_plan.py 7DT02
    python run_refresh_masterframe_plan.py 7DT02 7DT03 --dry-run
    python run_refresh_masterframe_plan.py
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser(description="Refresh the stored master-association plan by unit.")
    parser.add_argument("units", nargs="*", help="units, e.g. 7DT02; none: every unit missing or stale")
    parser.add_argument("--dry-run", action="store_true", help="compute and report, write nothing")
    args = parser.parse_args()
    if len(args.units) == 1:
        from pipeline.services.masterframe_association import refresh_plan

        report = refresh_plan(args.units, dry_run=args.dry_run)
        print(report.to_string(index=False), flush=True)
        return 0 if report.message.isin(["refreshed", "dry run: nothing written"]).all() else 1
    units = args.units
    if not units:
        from pipeline.services.database.masterframe_plan import plan_status

        status = plan_status()
        units = list(status.unit[status.stale])
    flags = ["--dry-run"] if args.dry_run else []
    processes = [subprocess.Popen([sys.executable, __file__, unit, *flags]) for unit in units]
    return max([p.wait() for p in processes], default=0)


if __name__ == "__main__":
    sys.exit(main())
