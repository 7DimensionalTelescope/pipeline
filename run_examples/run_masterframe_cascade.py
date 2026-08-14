#!/usr/bin/env python
"""
Plan -- and optionally queue -- the reprocessing cascade of regenerated master frames.

Give it master frames that were ALREADY regenerated; it walks image_qa_dependency and
emits the work in four ordered batches:

  --phase 1 --sweep 1|2|3   rebuild chained masters (-master_frame_only, cumulative
                            -calib_types sweeps: bias / bias+dark / bias+dark+flat)
  --phase 2                 recalibrate the affected singles (overwrite=False)
  --phase 3 --execute       rerun the affected nightly science configs with -overwrite
  --phase 4 --execute       rerun the multi-epoch science configs, after 3 drained

Read-only by default. --submit queues ONE batch; without --execute, phases 1-2 queue a
-dry_run sizing pass (a lower bound). Wait for each batch to drain before the next; if a
task fails, resubmit through this script (rerun_failed_tasks wipes the sweep flags).

Usage:
    python run_masterframe_cascade.py flat_m675_7DT02_20260704_1x1_gain2750_C31166
    python run_masterframe_cascade.py --from-file seeds.txt --show-configs
    python run_masterframe_cascade.py <seed> --submit --phase 1 --sweep 1            # sizing
    python run_masterframe_cascade.py <seed> --submit --phase 1 --sweep 1 --execute  # real
"""

from __future__ import annotations

import argparse
import sys
import time

from pipeline.services.cascade import plan, submit, MASTER_SWEEPS


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("seeds", nargs="*", help="regenerated master frames, by name or path")
    parser.add_argument("--from-file", help="file with one seed per line, instead of/in addition to positional")
    parser.add_argument("--max-depth", type=int, default=12, help="how far to follow the chain (default 12)")
    parser.add_argument("--show-configs", action="store_true", help="list every config, not just the counts")
    parser.add_argument("--show-masters", action="store_true", help="list the chained master frames")
    parser.add_argument("--submit", action="store_true", help="queue the plan (default: print and stop)")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4],
                        help="which ONE phase to submit; the scheduler dedupes on config path, so "
                             "batches sharing configs must be queued one at a time, drained in between")
    parser.add_argument("--sweep", type=int, choices=[1, 2, 3],
                        help="with --phase 1: which cumulative calib_types sweep ("
                             + "; ".join(f"{i+1}=" + "+".join(s) for i, s in enumerate(MASTER_SWEEPS))
                             + "), run in order and drained in between")
    parser.add_argument("--execute", action="store_true",
                        help="with --submit, do the real work instead of a -dry_run sizing pass")
    parser.add_argument("--base-priority", type=int, default=1, help="scheduler base priority (default 1)")
    args = parser.parse_args()

    seeds = list(args.seeds)
    if args.from_file:
        with open(args.from_file) as fh:
            seeds += [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    if not seeds:
        parser.error("give at least one master frame, positionally or with --from-file")

    st = time.time()
    p = plan(seeds, max_depth=args.max_depth)
    print(p.report())
    print(f"\n  planned in {time.time() - st:.2f}s")

    if args.show_masters and p.masters:
        print("\n  chained masters (regenerate in this order):")
        for name, image_type in p.masters:
            print(f"    {image_type:<5} {name}")

    if args.show_configs:
        for label, configs in (
            ("phase 1  masters", [c for c in p.master_configs]),
            ("phase 2  singles", [c for c in p.preprocess_configs]),
            ("phase 3  science (nightly)", [(n, f) for n, f, _ in p.nightly_science]),
            ("phase 4  science (multi-epoch)", [(n, f) for n, f, _ in p.multiepoch_science]),
        ):
            if configs:
                print(f"\n  {label}:")
                for name, config_file in configs:
                    print(f"    {name:<34} {config_file}")

    if not args.submit:
        print("\n  Read-only. Re-run with --submit --phase N to queue a phase.")
        return

    if args.phase is None:
        print(
            "\n  --submit needs --phase (exactly one). The batches are ordered and must not overlap:\n"
            "    --phase 1 --sweep 1|2|3   rebuild the chained masters ("
            + ", ".join("+".join(s) for s in MASTER_SWEEPS)
            + ")\n"
            "    --phase 2                 recalibrate the singles\n"
            "    --phase 3 --execute       rerun the nightly science configs with -overwrite\n"
            "    --phase 4 --execute       rerun the multi-epoch science configs with -overwrite\n"
            "  Wait for each batch to drain before submitting the next. If a task fails,\n"
            "  resubmit through this script -- rerun_failed_tasks would wipe the sweep flags."
        )
        sys.exit(2)

    if args.phase == 1 and args.sweep is None:
        print("\n  --phase 1 needs --sweep 1, 2 or 3 (run them in order, drained in between).")
        sys.exit(2)

    if p.unregenerable and args.phase == 1:
        print(
            f"\n  {len(p.unregenerable)} master frame(s) cannot be regenerated (see above). "
            "Everything below them stays stale; the cascade will be incomplete."
        )

    dry = not args.execute
    if dry and args.phase in (3, 4):
        print(
            f"\n  Phase {args.phase} has no sizing pass -- cli/data_reduction takes no -dry_run, and a science\n"
            "  rerun is all-or-nothing per config. The config count above is the size.\n"
            "  Re-run with --execute to queue it."
        )
        sys.exit(2)

    batch = f"phase {args.phase}" + (f" sweep {args.sweep}" if args.phase == 1 else "")
    print(f"\n  Submitting {batch}" + ("  [-dry_run sizing pass]" if dry else "  [FOR REAL]"))
    sc = submit(p, phase=args.phase, sweep=args.sweep, base_priority=args.base_priority, dry_run=dry)
    if sc is None:
        print("  nothing to queue for this phase")
    else:
        print(f"  queued {len(sc.schedule)} task(s) to the system queue")


if __name__ == "__main__":
    main()
