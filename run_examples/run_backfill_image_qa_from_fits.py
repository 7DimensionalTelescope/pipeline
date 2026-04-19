#!/usr/bin/env python
"""
Backfill image_qa columns from FITS headers (e.g. seeing, peeing, photometry keywords).

Use this after fixing Photometry QA sync so historical rows with NULL seeing (and other
header-only fields) are updated from disk.

Usage:
    # Dry-run: count rows that would be updated (science, seeing IS NULL)
    python run_backfill_image_qa_from_fits.py --dry-run

    # Apply for all science images missing seeing
    python run_backfill_image_qa_from_fits.py

    # Restrict by nightdate and limit
    python run_backfill_image_qa_from_fits.py --nightdate-from 2026-04-01 --nightdate-to 2026-04-30 --limit 500

    # Re-sync header fields for all matching rows (not only seeing IS NULL)
    python run_backfill_image_qa_from_fits.py --force-all

    # Include masterframe rows (default: science only)
    python run_backfill_image_qa_from_fits.py --include-masterframes

    # Parallel I/O + DB updates (default 16 workers; tune for disk / DB max_connections)
    python run_backfill_image_qa_from_fits.py --workers 24
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.services.database.image_qa import ImageQA, ImageQATable

_thread_local = threading.local()


def _get_thread_image_qa() -> ImageQA:
    """One ImageQA (and thus independent DB connections) per worker thread."""
    if not hasattr(_thread_local, "iq"):
        _thread_local.iq = ImageQA()
    return _thread_local.iq


def _build_select_sql(args: argparse.Namespace) -> tuple[str, dict]:
    conditions = ["image_path IS NOT NULL", "image_path != ''"]
    params: dict = {}

    if not args.force_all:
        conditions.append("seeing IS NULL")

    if not args.include_masterframes:
        conditions.append("image_group = %(image_group)s")
        params["image_group"] = "science"

    if args.image_types:
        conditions.append("image_type = ANY(%(image_types)s)")
        params["image_types"] = args.image_types

    if args.nightdate_from is not None:
        conditions.append("nightdate >= %(nightdate_from)s")
        params["nightdate_from"] = args.nightdate_from
    if args.nightdate_to is not None:
        conditions.append("nightdate <= %(nightdate_to)s")
        params["nightdate_to"] = args.nightdate_to

    where_clause = " AND ".join(conditions)
    sql = f"""
        SELECT id, image_path, process_status_id
        FROM image_qa
        WHERE {where_clause}
        ORDER BY id
    """
    if args.limit is not None:
        sql += " LIMIT %(limit)s"
        params["limit"] = args.limit
    return sql, params


def _payload_from_fits(path: str, process_status_id: int | None) -> dict:
    qa = ImageQATable.from_file(path, process_status_id=process_status_id)
    data = qa.to_dict()
    data.pop("id", None)
    data.pop("created_at", None)
    # Leave updated_at to DB / triggers; omit if None so we do not overwrite with NULL
    data.pop("updated_at", None)
    return data


def _process_one_row(
    columns: list[str],
    row: tuple,
    *,
    dry_run: bool,
) -> tuple[str, int, str, str | None]:
    """
    Returns (kind, id, path, detail) where kind is
    ok | dry_run | skip_missing | skip_os | skip_key | error.
    """
    row_dict = dict(zip(columns, row))
    rid = row_dict["id"]
    path = row_dict["image_path"]
    psid = row_dict.get("process_status_id")

    if not path or not os.path.isfile(path):
        return ("skip_missing", rid, path or "", None)

    if dry_run:
        return ("dry_run", rid, path, None)

    iq = _get_thread_image_qa()
    try:
        payload = _payload_from_fits(path, psid)
        iq.update_data(rid, **payload)
        return ("ok", rid, path, None)
    except OSError as e:
        return ("skip_os", rid, path, str(e))
    except KeyError:
        return ("skip_key", rid, path, None)
    except Exception as e:
        return ("error", rid, path, str(e))


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill image_qa from FITS headers.")
    parser.add_argument("--dry-run", action="store_true", help="List rows only; do not update.")
    parser.add_argument(
        "--force-all",
        action="store_true",
        help="Do not require seeing IS NULL; refresh header-mapped columns for all matched rows.",
    )
    parser.add_argument(
        "--include-masterframes",
        action="store_true",
        help="Also process masterframe rows (default: image_group = science only).",
    )
    parser.add_argument("--nightdate-from", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--nightdate-to", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument(
        "--image-type",
        action="append",
        dest="image_types",
        metavar="TYPE",
        help="Repeatable. e.g. --image-type single --image-type coadd (default: no filter).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max rows to process.")
    default_workers = min(32, max(4, (os.cpu_count() or 8) * 2))
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=f"Parallel threads (default {default_workers}). Use 1 to run sequentially.",
    )
    args = parser.parse_args()

    iq_main = ImageQA()
    sql, params = _build_select_sql(args)
    rows, columns = iq_main.excute_query(sql, params or None, return_columns=True)

    if not rows:
        print("No matching rows.")
        return 0

    print(f"Selected {len(rows)} row(s). dry_run={args.dry_run} workers={args.workers}")

    updated = 0
    skipped_missing_file = 0
    skipped_no_header_gain = 0
    errors = 0
    progress_lock = threading.Lock()

    def _handle_result(kind: str, rid: int, path: str, detail: str | None) -> None:
        nonlocal updated, skipped_missing_file, skipped_no_header_gain, errors
        if kind == "ok":
            updated += 1
        elif kind == "dry_run":
            updated += 1
            print(f"would_update id={rid} path={path}")
        elif kind == "skip_missing":
            skipped_missing_file += 1
            if args.dry_run:
                print(f"[skip missing] id={rid} path={path!r}")
        elif kind == "skip_os":
            skipped_missing_file += 1
            print(f"[skip] id={rid} read error: {detail}")
        elif kind == "skip_key":
            skipped_no_header_gain += 1
            print(f"[skip] id={rid} header/key error on {path!r}")
        elif kind == "error":
            errors += 1
            print(f"[error] id={rid} {path!r}: {detail}")
        if not args.dry_run and kind == "ok" and updated > 0 and updated % 200 == 0:
            print(f"  updated {updated} ...")

    if args.workers <= 1:
        for row in rows:
            kind, rid, path, detail = _process_one_row(columns, row, dry_run=args.dry_run)
            _handle_result(kind, rid, path, detail)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_row = {
                executor.submit(_process_one_row, columns, row, dry_run=args.dry_run): row for row in rows
            }
            for fut in as_completed(future_to_row):
                kind, rid, path, detail = fut.result()
                with progress_lock:
                    _handle_result(kind, rid, path, detail)

    print(
        f"Done. updated={updated}, missing_file={skipped_missing_file}, "
        f"header_errors={skipped_no_header_gain}, errors={errors}"
    )
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
