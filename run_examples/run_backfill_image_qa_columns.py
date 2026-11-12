#!/usr/bin/env python
"""
Refresh image_qa rows from the FITS headers on disk, so that the columns added after a row was
registered (rej_proc, numfrac, ellip; 2026-11-12) hold what the header holds. The stage-aware
Checker decides from these columns on its image_qa path; a NULL there is a criterion it cannot see.

Scope: science singles whose row belongs to a science config (Astrometry has taken the row) or
carries a verdict or a zero point, and whose numfrac is NULL. Each row gets what a stage refresh does: ImageQATable.from_file on its
image_path, then ImageQA.update_data on its id. No dependency row is touched.

A row whose header carries another IMAGEID than the row (the file was re-made after the row was
registered) is reset to the header by update_data, as any re-registration is; reported as 'reset'.

Resumable: rows are taken in id order, and the last finished id is kept in <log>.state; a rerun
continues after it (--start-id overrides). A dry run reads the headers and writes nothing.

Run it from the development checkout: the deployed ImageQATable does not have the new fields.

Usage:
    python run_backfill_image_qa_columns.py --dry-run --limit 2000
    python run_backfill_image_qa_columns.py --limit 20000          # a real first slice
    python run_backfill_image_qa_columns.py                        # the rest
    python run_backfill_image_qa_columns.py --workers 8
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from threading import Lock, local

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.const import CONFIG_TYPE_SCIENCE
from pipeline.services.database.image_qa import ImageQA, ImageQATable
from pipeline.services.database.query import free_query

MAX_WORKERS = 32  # header reads over NFS
DEFAULT_WORKERS = 16
CHUNK = 2000  # rows per query and per saved state
DEFAULT_LOG = "/var/log/pipeline/backfill_image_qa_columns.log"

_log_lock = Lock()
_log_fh = None
_thread = local()


def _log(msg: str) -> None:
    with _log_lock:
        print(msg, flush=True)
        if _log_fh is not None:
            _log_fh.write(msg + "\n")
            _log_fh.flush()


def _next_rows(after_id: int, n: int) -> list:
    return free_query(
        """
        SELECT q.id, q.image_path, q.imageid
        FROM image_qa q JOIN process_status p ON p.id = q.process_status_id
        WHERE q.image_group = 'science' AND q.image_type = 'single'
          AND (p.config_type = %s OR q.sanity IS NOT NULL OR q.zp_auto IS NOT NULL)
          AND q.numfrac IS NULL AND q.id > %s
        ORDER BY q.id LIMIT %s
        """,
        (CONFIG_TYPE_SCIENCE, after_id, n),
    )


def _refresh(row: tuple, dry_run: bool) -> str:
    qa_id, path, stored_imageid = row
    if not path or not os.path.exists(path):
        return "missing"
    try:
        record = ImageQATable.from_file(path)
        if not dry_run:
            if not hasattr(_thread, "image_qa"):
                _thread.image_qa = ImageQA()
            _thread.image_qa.update_data(qa_id, **record.to_dict())
    except Exception as e:
        _log(f"[error] id {qa_id} {path} :: {type(e).__name__}: {e}")
        return "error"
    return "refreshed" if record.imageid == stored_imageid else "reset"


def main() -> int:
    global _log_fh
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="read the headers, write nothing")
    p.add_argument("--limit", type=int, default=None, help="stop after this many rows")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--start-id", type=int, default=None, help="continue after this image_qa id (overrides the state)")
    p.add_argument("--log", default=DEFAULT_LOG)
    args = p.parse_args()

    workers = max(1, min(args.workers, MAX_WORKERS))
    state_file = args.log + ".state"
    _log_fh = open(args.log, "a")

    last_id = args.start_id
    if last_id is None:
        last_id = int(open(state_file).read().strip() or 0) if os.path.exists(state_file) else 0
    _log(f"[start] {time.strftime('%F %T')} after id {last_id}, workers {workers}, dry_run {args.dry_run}")

    total, t0, n_done = Counter(), time.time(), 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while args.limit is None or n_done < args.limit:
            n = CHUNK if args.limit is None else min(CHUNK, args.limit - n_done)
            rows = _next_rows(last_id, n)
            if not rows:
                break
            total.update(pool.map(lambda row: _refresh(row, args.dry_run), rows))
            last_id, n_done = rows[-1][0], n_done + len(rows)
            if not args.dry_run:
                with open(state_file, "w") as f:
                    f.write(str(last_id))
            _log(
                f"[progress] {n_done} rows, last id {last_id}, {n_done / (time.time() - t0):.0f} rows/s, {dict(total)}"
            )

    _log(f"[done] {time.strftime('%F %T')} {n_done} rows in {time.time() - t0:.0f} s, last id {last_id}, {dict(total)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
