#!/usr/bin/env python
"""
Backfill process_status_dependency from the products each config actually wrote.

Historically this table held only the schedule's plan ("these configs were queued
together"), which is wrong whenever a run did not follow the plan -- most often a
preprocess config that produced no usable master, sending its science configs to
another night's masters while the plan still named its own night. Stages now
re-derive their own edges as they finish; this fills in every config that ran
before that, by rolling image_qa_dependency up to config level.

A config whose roll-up finds nothing is left alone rather than emptied, so this
never removes an edge it cannot replace.

Usage:
    # Dry-run: how many configs would gain product-derived edges
    python run_backfill_process_status_dependency.py --dry-run

    # Fill only configs that have no product-derived rows yet (default)
    python run_backfill_process_status_dependency.py

    # Re-derive every matched config, including ones already resolved
    python run_backfill_process_status_dependency.py --force-all

    # Restrict scope
    python run_backfill_process_status_dependency.py --nightdate-from 2026-05-01
    python run_backfill_process_status_dependency.py --config-type preprocess --limit 500

    # Parallel (default 8; each worker holds its own connection, mind max_connections)
    python run_backfill_process_status_dependency.py --workers 16
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.services.database.process_status_dependency import ProcessStatusDependency

_thread_local = threading.local()


def _dependency() -> ProcessStatusDependency:
    """One ProcessStatusDependency per worker thread; BaseDatabase is not pooled."""
    if not hasattr(_thread_local, "psd"):
        _thread_local.psd = ProcessStatusDependency()
    return _thread_local.psd


def select_configs(args) -> list:
    """Candidate (process_status_id, name) rows for the requested scope."""
    query = "SELECT id, name FROM process_status WHERE TRUE"
    params = []
    if args.config_type:
        query += " AND config_type = %s"
        params.append(args.config_type)
    if args.nightdate_from:
        query += " AND nightdate >= %s"
        params.append(args.nightdate_from)
    if args.nightdate_to:
        query += " AND nightdate <= %s"
        params.append(args.nightdate_to)
    if args.name:
        query += " AND name = %s"
        params.append(args.name)
    if not args.force_all:
        query += (
            " AND NOT EXISTS (SELECT 1 FROM process_status_dependency d"
            "  WHERE d.derived_config_name = process_status.name AND d.origin = 'product')"
        )
    query += " ORDER BY id"
    if args.limit:
        query += " LIMIT %s"
        params.append(args.limit)

    return ProcessStatusDependency().execute_query(query, tuple(params))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-type", choices=["preprocess", "science"], help="restrict to one config type")
    parser.add_argument("--nightdate-from", help="earliest nightdate (YYYY-MM-DD)")
    parser.add_argument("--nightdate-to", help="latest nightdate (YYYY-MM-DD)")
    parser.add_argument("--name", help="a single config name")
    parser.add_argument("--limit", type=int, help="stop after this many configs")
    parser.add_argument("--force-all", action="store_true", help="re-derive configs already resolved from products")
    parser.add_argument("--dry-run", action="store_true", help="list the scope, write nothing")
    parser.add_argument("--workers", type=int, default=8, help="parallel workers (default 8)")
    args = parser.parse_args()

    configs = select_configs(args)
    print(f"{len(configs)} configs in scope")
    if args.dry_run:
        for pid, name in configs[:20]:
            print(f"  would sync {name} (process_status_id={pid})")
        if len(configs) > 20:
            print(f"  ... and {len(configs) - 20} more")
        return

    if not configs:
        return

    st = time.time()
    synced = 0
    empty = 0
    failed = 0

    def work(row):
        pid, name = row
        return name, _dependency().sync_from_products(pid)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, row): row for row in configs}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                name, n = future.result()
            except Exception as e:
                failed += 1
                print(f"  FAILED {futures[future][1]}: {e}")
                continue
            if n:
                synced += 1
            else:
                empty += 1
            if i % 1000 == 0:
                print(f"  {i}/{len(configs)} ({time.time() - st:.0f}s)")

    print(
        f"Done in {time.time() - st:.0f}s: {synced} configs given product-derived edges,"
        f" {empty} with nothing to roll up (left as they were), {failed} failed"
    )


if __name__ == "__main__":
    main()
