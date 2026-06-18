#!/usr/bin/env python
"""
Backfill image_qa_dependency rows from FITS headers via ImageQADependency.sync().

Each derived image (master dark/flat, calibrated single, coadd, diff) records its
source images in IMCMB*/IMG* header cards. This walks rows that currently have no
dependency edges and re-syncs them from disk. Bias masters are skipped: their
ingredients are raw frames with no image_qa row.

Usage:
    # Dry-run: list rows that would be synced
    python run_backfill_image_qa_dependency.py --dry-run

    # Fill only rows missing dependencies (default)
    python run_backfill_image_qa_dependency.py

    # Re-sync every matched row, even those that already have edges
    python run_backfill_image_qa_dependency.py --force-all

    # Restrict scope
    python run_backfill_image_qa_dependency.py --image-type single --limit 500
    python run_backfill_image_qa_dependency.py --nightdate-from 2026-05-01

    # Parallel I/O + DB (default 16 workers; tune for disk / DB max_connections)
    python run_backfill_image_qa_dependency.py --workers 24
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.services.database.image_qa import ImageQA
from pipeline.services.database.image_qa_dependency import ImageQADependency

DERIVED_TYPES = ["dark", "flat", "single", "coadd", "diff"]

_thread_local = threading.local()


def _get_thread_dep() -> ImageQADependency:
    """One ImageQADependency (independent DB connections) per worker thread."""
    if not hasattr(_thread_local, "dep"):
        _thread_local.dep = ImageQADependency()
    return _thread_local.dep


def _build_select_sql(args: argparse.Namespace) -> tuple[str, dict]:
    conditions = ["image_path IS NOT NULL", "image_path != ''", "image_type = ANY(%(image_types)s)"]
    params: dict = {"image_types": args.image_types or DERIVED_TYPES}

    if not args.force_all:
        conditions.append("NOT EXISTS (SELECT 1 FROM image_qa_dependency d WHERE d.derived_image_id = image_qa.id)")

    if args.nightdate_from is not None:
        conditions.append("nightdate >= %(nightdate_from)s")
        params["nightdate_from"] = args.nightdate_from
    if args.nightdate_to is not None:
        conditions.append("nightdate <= %(nightdate_to)s")
        params["nightdate_to"] = args.nightdate_to

    where_clause = " AND ".join(conditions)
    sql = f"""
        SELECT id, image_path
        FROM image_qa
        WHERE {where_clause}
        ORDER BY id
    """
    if args.limit is not None:
        sql += " LIMIT %(limit)s"
        params["limit"] = args.limit
    return sql, params


def _process_one_row(row: tuple, *, dry_run: bool) -> tuple[str, int, str, int]:
    """Returns (kind, id, path, n_deps); kind is ok | dry_run | skip_missing | error."""
    rid, path = row[0], row[1]

    if not path or not os.path.isfile(path):
        return ("skip_missing", rid, path or "", 0)
    if dry_run:
        return ("dry_run", rid, path, 0)

    try:
        n = _get_thread_dep().sync(path, rid)
        return ("ok", rid, path, n)
    except Exception as e:
        return ("error", rid, path, 0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill image_qa_dependency from FITS headers.")
    parser.add_argument("--dry-run", action="store_true", help="List rows only; do not write.")
    parser.add_argument("--force-all", action="store_true", help="Re-sync matched rows even if edges exist.")
    parser.add_argument("--nightdate-from", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--nightdate-to", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument(
        "--image-type",
        action="append",
        dest="image_types",
        metavar="TYPE",
        help=f"Repeatable. Subset of {DERIVED_TYPES} (default: all of them).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max rows to process.")
    default_workers = min(32, max(4, (os.cpu_count() or 8) * 2))
    parser.add_argument("--workers", type=int, default=default_workers, help=f"Parallel threads (default {default_workers}).")
    args = parser.parse_args()

    sql, params = _build_select_sql(args)
    rows, _ = ImageQA().execute_query(sql, params or None, return_columns=True)

    if not rows:
        print("No matching rows.")
        return 0

    print(f"Selected {len(rows)} row(s). dry_run={args.dry_run} workers={args.workers}")

    synced = 0
    edges = 0
    skipped_missing_file = 0
    no_sources = 0
    errors = 0
    progress_lock = threading.Lock()

    def _handle_result(kind: str, rid: int, path: str, n: int) -> None:
        nonlocal synced, edges, skipped_missing_file, no_sources, errors
        if kind == "ok":
            synced += 1
            edges += n
            if n == 0:
                no_sources += 1
            if synced % 500 == 0:
                print(f"  synced {synced}/{len(rows)} rows, {edges} edges ...")
        elif kind == "dry_run":
            synced += 1
            print(f"would_sync id={rid} path={path}")
        elif kind == "skip_missing":
            skipped_missing_file += 1
        elif kind == "error":
            errors += 1
            print(f"[error] id={rid} {path!r}")

    if args.workers <= 1:
        for row in rows:
            _handle_result(*_process_one_row(row, dry_run=args.dry_run))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_process_one_row, row, dry_run=args.dry_run) for row in rows]
            for fut in as_completed(futures):
                with progress_lock:
                    _handle_result(*fut.result())

    print(
        f"Done. synced={synced}, edges={edges}, no_sources={no_sources}, "
        f"missing_file={skipped_missing_file}, errors={errors}"
    )
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
