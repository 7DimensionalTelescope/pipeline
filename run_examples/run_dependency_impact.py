#!/usr/bin/env python
"""
Report which products a regeneration invalidated, and which configs must rerun.

Read-only. It reports; it does not flip flags, queue anything, or touch a config.

Two questions it answers:

  --stale (default)  What is invalidated right now? A regenerated file takes over the
                     image_qa row of the version it replaced, so an edge whose recorded
                     source_imageid no longer matches that row's current imageid names
                     an image built from pixels that are gone. Those images, plus
                     everything downstream of them, are the answer.

  --config NAME      What would rerunning this config invalidate? Seeds the same walk
                     with the images that config produced, and reports what consumes
                     them -- the blast radius, before paying for it.

The walk is transitive: a coadd stacked from an invalidated single is invalidated too,
even though its own edge still matches, because the single was not regenerated, only
invalidated. A diff off that coadd follows, and so on.

Note that an astrometric re-solve is deliberately not a regeneration -- the WCS changes
but the IMAGEID does not -- so it starts no chain here and this report will not show it.

Usage:
    python run_dependency_impact.py
    python run_dependency_impact.py --nightdate-from 2026-05-01 --nightdate-to 2026-07-31
    python run_dependency_impact.py --config T21659_m600_2026-07-12
    python run_dependency_impact.py --show-images --limit 50
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.services.database.image_qa_dependency import ImageQADependency


def seeds_from_staleness(dep, args):
    """Images whose ingredients have since been regenerated."""
    edges = dep.find_stale_edges(
        nightdate_from=args.nightdate_from, nightdate_to=args.nightdate_to
    )
    if not edges:
        return [], []
    seeds = sorted({row[0] for row in edges})
    return seeds, edges


def seeds_from_config(dep, config_name):
    """Images produced by one config."""
    rows = dep.execute_query(
        "SELECT qa.id FROM image_qa qa"
        " JOIN process_status ps ON ps.id = qa.process_status_id"
        " WHERE ps.name = %s",
        (config_name,),
    )
    return [r[0] for r in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="seed from one config's products instead of from detected staleness")
    parser.add_argument("--nightdate-from", help="earliest nightdate of the invalidated image (YYYY-MM-DD)")
    parser.add_argument("--nightdate-to", help="latest nightdate of the invalidated image (YYYY-MM-DD)")
    parser.add_argument("--max-depth", type=int, default=10, help="how far to follow the chain (default 10)")
    parser.add_argument("--show-images", action="store_true", help="also list the affected images")
    parser.add_argument("--limit", type=int, default=40, help="max rows to print per section (default 40)")
    args = parser.parse_args()

    dep = ImageQADependency()
    st = time.time()

    if args.config:
        seeds = seeds_from_config(dep, args.config)
        if not seeds:
            print(f"No images registered for config {args.config!r}")
            return
        print(f"{args.config}: {len(seeds)} products")
        include_seeds = False
        edges = []
    else:
        seeds, edges = seeds_from_staleness(dep, args)
        if not seeds:
            print("No stale dependencies found in scope.")
            return
        print(f"{len(edges)} stale edges over {len(seeds)} directly invalidated images")
        by_role = Counter(row[4] for row in edges)
        for role, n in by_role.most_common():
            print(f"  via {role}: {n}")
        for row in edges[: args.limit]:
            print(f"  {row[1]}  <-  {row[3]} ({row[4]}: used {row[5][:8]}, now {row[6][:8]})")
        if len(edges) > args.limit:
            print(f"  ... and {len(edges) - args.limit} more")
        include_seeds = True

    downstream = dep.impacted_images(seeds, max_depth=args.max_depth)
    print(f"\n{len(downstream)} images downstream of those")
    for image_type, n in Counter(r[2] for r in downstream).most_common():
        depth = min(r[1] for r in downstream if r[2] == image_type)
        print(f"  {image_type}: {n} (first reached at depth {depth})")
    if args.show_images:
        for image_id, depth, image_type, name in downstream[: args.limit]:
            print(f"  depth {depth}  {image_type:7s}  {name}")
        if len(downstream) > args.limit:
            print(f"  ... and {len(downstream) - args.limit} more")

    configs = dep.impacted_configs(seeds, max_depth=args.max_depth, include_seeds=include_seeds)
    if args.config:
        # its own masters feed its own singles, so the walk comes back to it
        configs = [row for row in configs if row[0] != args.config]
    print(f"\n{len(configs)} configs affected:")
    for name, config_type in configs[: args.limit]:
        print(f"  {config_type:10s} {name}")
    if len(configs) > args.limit:
        print(f"  ... and {len(configs) - args.limit} more")

    print(f"\n({time.time() - st:.1f}s)")


if __name__ == "__main__":
    main()
