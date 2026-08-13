#!/usr/bin/env python
"""
Plan -- and optionally queue -- the reprocessing cascade of regenerated master frames.

Give it the master frames that were ALREADY regenerated. It walks image_qa_dependency for
everything built from them and emits the work in three phases:

  phase 1  masters   the preprocess configs that own the raw calibration frames of the
                     CHAINED masters (a rebuilt bias invalidates the darks and flats made
                     from it), run master_frame_only in three cumulative calib_types sweeps
                     -- bias, then bias+dark, then bias+dark+flat. Kind order is the real
                     topological order; blast-radius depth is not, because a flat records
                     its bias directly as well as through the dark.

  phase 2  singles   the preprocess configs that must recalibrate the affected science
                     singles. A different set: a regenerated FLAT has no master descendants,
                     so phase 1 is empty for it, yet every single it calibrated is stale.
                     Nothing else rebuilds those pixels -- the science stages run on the
                     singles that already exist.

  phase 3  science   the affected science configs, with -overwrite. Science stage skipping
                     is flag-based and will never notice an IMCID change on its own.

Read-only by default: it prints the plan and stops. --submit queues it, and even then the
default is a sizing pass (-dry_run on every task) that writes nothing -- pass --execute to
do the actual work.

**The phases are not independent.** Do not start phase N+1 until phase N has drained: the
queue runs tasks concurrently, and a science config calibrated against a master that is
still being rebuilt is exactly what this exists to prevent. Submit one phase per invocation
and watch the queue between them.

Usage:
    python run_masterframe_cascade.py flat_m675_7DT02_20260704_1x1_gain2750_C31166
    python run_masterframe_cascade.py /balmer/.../bias_7DT06_20260711_1x1_gain2750_C31093.fits
    python run_masterframe_cascade.py --from-file seeds.txt --show-configs
    python run_masterframe_cascade.py <seed> --submit --phase 1            # sizing pass
    python run_masterframe_cascade.py <seed> --submit --phase 1 --execute  # for real
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
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], action="append",
                        help="which phase to submit; repeatable. Default: refuse and ask, so that "
                             "phases are never queued together by accident")
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
            ("phase 3  science", [(n, f) for n, f, _ in p.nightly_science + p.multiepoch_science]),
        ):
            if configs:
                print(f"\n  {label}:")
                for name, config_file in configs:
                    print(f"    {name:<34} {config_file}")

    if not args.submit:
        print("\n  Read-only. Re-run with --submit --phase N to queue a phase.")
        return

    if not args.phase:
        print(
            "\n  --submit needs --phase. The phases are ordered and must not overlap:\n"
            "    --phase 1   rebuild the chained masters (3 sweeps: "
            + ", ".join("+".join(s) for s in MASTER_SWEEPS)
            + ")\n"
            "    --phase 2   recalibrate the singles\n"
            "    --phase 3   rerun the science configs with -overwrite\n"
            "  Wait for each to drain before submitting the next."
        )
        sys.exit(2)

    if p.unregenerable and 1 in args.phase:
        print(
            f"\n  {len(p.unregenerable)} master frame(s) cannot be regenerated (see above). "
            "Everything below them stays stale; the cascade will be incomplete."
        )

    dry = not args.execute
    phases = sorted(set(args.phase))
    if dry and phases == [3]:
        print(
            "\n  Phase 3 has no sizing pass -- cli/data_reduction takes no -dry_run, and a science\n"
            "  rerun is all-or-nothing per config. The config count above is the size.\n"
            "  Re-run with --execute to queue it."
        )
        sys.exit(2)

    print(f"\n  Submitting phase(s) {phases}" + ("  [-dry_run sizing pass]" if dry else "  [FOR REAL]"))
    schedulers = submit(p, phases=phases, base_priority=args.base_priority, dry_run=dry)
    print(f"  queued {len(schedulers)} batch(es) to the system queue")


if __name__ == "__main__":
    main()
