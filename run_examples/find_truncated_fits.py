#!/usr/bin/env python
"""
Find FITS files that trigger Astropy's truncation warning:

  File may have been truncated: actual file length (...) is smaller than the expected size (...)

Typical causes: incomplete copy, aborted write, or corrupted transfer.

Usage:
    # Paths or directories (recurses into dirs for *.fits / *.fits.fz)
    python find_truncated_fits.py /lyman/data2/processed/2026-04-07/T00626/m575/singles

    # Science frames from image_qa (same filters as run_backfill_image_qa_from_fits.py)
    python find_truncated_fits.py --from-db --nightdate-from 2026-04-01 --limit 5000

    # Write paths to a file
    python find_truncated_fits.py --from-db --output truncated.txt

    # Parallel checks (default scales with CPU; lower if I/O saturates)
    python find_truncated_fits.py --from-db --workers 24
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from astropy.io import fits
from astropy.utils.exceptions import AstropyUserWarning

from pipeline.services.database.image_qa import ImageQA


def _collect_paths_from_args(paths: list[str]) -> list[str]:
    out: list[str] = []
    for p in paths:
        if os.path.isfile(p):
            out.append(os.path.abspath(p))
        elif os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for f in files:
                    if f.endswith(".fits") or f.endswith(".fits.fz"):
                        out.append(os.path.join(root, f))
        else:
            print(f"[skip not found] {p}", file=sys.stderr)
    return sorted(set(out))


def _build_select_sql(args: argparse.Namespace) -> tuple[str, dict]:
    conditions = ["image_path IS NOT NULL", "image_path != ''"]
    params: dict = {}

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
        SELECT id, image_path
        FROM image_qa
        WHERE {where_clause}
        ORDER BY id
    """
    if args.limit is not None:
        sql += " LIMIT %(limit)s"
        params["limit"] = args.limit
    return sql, params


def is_truncated_or_unreadable(path: str) -> tuple[bool, str | None]:
    """
    Returns (True, reason) if the file appears truncated (Astropy warning) or fails to read.
    """
    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always", AstropyUserWarning)
        try:
            with fits.open(path, memmap=False) as hdul:
                for hdu in hdul:
                    if hdu.data is not None:
                        hdu.data.shape  # force read
        except OSError as e:
            return True, f"OSError: {e}"
        except Exception as e:
            return True, f"{type(e).__name__}: {e}"

    for w in wrec:
        msg = str(w.message)
        if "truncated" in msg.lower():
            return True, msg
    return False, None


def _check_one_row(qa_id: int | None, path: str | None) -> tuple[int | None, str, str, str | None]:
    """
    Returns (qa_id, path, outcome, detail).

    outcome is one of: ok | bad | missing
    detail is reason string for bad/missing, else None.
    """
    if not path or not os.path.isfile(path):
        return (qa_id, path or "", "missing", "missing_on_disk")
    truncated, reason = is_truncated_or_unreadable(path)
    if truncated:
        return (qa_id, path, "bad", reason or "unknown")
    return (qa_id, path, "ok", None)


def main() -> int:
    parser = argparse.ArgumentParser(description="List FITS files that are truncated or unreadable.")
    parser.add_argument("paths", nargs="*", help="Files or directories to scan")
    parser.add_argument("--from-db", action="store_true", help="Use image_path from image_qa")
    parser.add_argument(
        "--include-masterframes",
        action="store_true",
        help="With --from-db: do not restrict to image_group = science",
    )
    parser.add_argument("--nightdate-from", type=str, default=None)
    parser.add_argument("--nightdate-to", type=str, default=None)
    parser.add_argument(
        "--image-type",
        action="append",
        dest="image_types",
        metavar="TYPE",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", "-o", type=str, default=None, help="Write one path per line")
    default_workers = min(32, max(4, (os.cpu_count() or 8) * 2))
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=f"Parallel threads (default {default_workers}). Use 1 for sequential.",
    )
    args = parser.parse_args()

    if args.from_db:
        iq = ImageQA()
        sql, params = _build_select_sql(args)
        rows, columns = iq.excute_query(sql, params or None, return_columns=True)
        file_list = []
        if rows:
            for row in rows:
                d = dict(zip(columns, row))
                p = d.get("image_path")
                if p:
                    file_list.append((d.get("id"), p))
    else:
        file_list = []
        for p in _collect_paths_from_args(args.paths):
            file_list.append((None, p))

    if not file_list:
        print("No files to check. Pass paths or use --from-db.", file=sys.stderr)
        return 1

    bad: list[tuple[int | None, str, str]] = []
    checked = 0
    progress_lock = threading.Lock()

    def _handle_done(qa_id: int | None, path: str, outcome: str, detail: str | None) -> None:
        nonlocal checked
        if outcome == "missing":
            if qa_id is not None:
                bad.append((qa_id, path, detail or "missing_on_disk"))
            return
        checked += 1
        if outcome == "bad":
            bad.append((qa_id, path, detail or "unknown"))
        if checked % 500 == 0:
            print(f"  ... checked {checked}", file=sys.stderr)

    if args.workers <= 1:
        for qa_id, path in file_list:
            qid, pth, outcome, detail = _check_one_row(qa_id, path)
            _handle_done(qid, pth, outcome, detail)
    else:
        print(f"Using {args.workers} workers", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_item = {
                executor.submit(_check_one_row, qa_id, path): (qa_id, path) for qa_id, path in file_list
            }
            for fut in as_completed(future_to_item):
                qid, pth, outcome, detail = fut.result()
                with progress_lock:
                    _handle_done(qid, pth, outcome, detail)

    bad.sort(key=lambda t: (t[0] is None, t[0] if t[0] is not None else 0, t[1]))

    out_fp = open(args.output, "w") if args.output else None
    try:
        for qa_id, path, reason in bad:
            line = f"{path}\t{reason}" if qa_id is None else f"{qa_id}\t{path}\t{reason}"
            print(line)
            if out_fp:
                out_fp.write(path + "\n")
    finally:
        if out_fp:
            out_fp.close()

    print(f"\nChecked {checked} existing file(s). Bad: {len(bad)}.", file=sys.stderr)
    return 0 if len(bad) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
