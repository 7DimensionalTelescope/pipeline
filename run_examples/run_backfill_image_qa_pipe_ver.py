#!/usr/bin/env python
"""
Backfill image_qa.pipe_ver from the PIPE_VER card of each product's FITS header.

Only astrometry (singles), imcoadd (coadds) and subtract (diffs) stamp PIPE_VER, so
masterframes and preprocess-only singles have none and keep NULL.

Usage:
    # Dry-run: read headers, report the version histogram, write nothing
    python run_backfill_image_qa_pipe_ver.py --dry-run

    # Backfill every row whose pipe_ver is still NULL (resumable; default 7 workers)
    python run_backfill_image_qa_pipe_ver.py

    # Restrict by nightdate / image type, or re-read rows that already have a version
    python run_backfill_image_qa_pipe_ver.py --nightdate-from 2026-04-01 --image-type coadd
    python run_backfill_image_qa_pipe_ver.py --force-all
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from astropy.io import fits

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.services.database.image_qa import ImageQA

MAX_WORKERS = 7


def _build_select_sql(args: argparse.Namespace) -> tuple[str, dict]:
    conditions = ["image_path IS NOT NULL", "image_path != ''"]
    params: dict = {}

    if not args.force_all:
        conditions.append("pipe_ver IS NULL")

    if args.image_types:
        conditions.append("image_type = ANY(%(image_types)s)")
        params["image_types"] = args.image_types

    if args.nightdate_from is not None:
        conditions.append("nightdate >= %(nightdate_from)s")
        params["nightdate_from"] = args.nightdate_from
    if args.nightdate_to is not None:
        conditions.append("nightdate <= %(nightdate_to)s")
        params["nightdate_to"] = args.nightdate_to

    sql = f"SELECT id, image_path FROM image_qa WHERE {' AND '.join(conditions)} ORDER BY id"
    if args.limit is not None:
        sql += " LIMIT %(limit)s"
        params["limit"] = args.limit
    return sql, params


def _read_pipe_ver(row: tuple) -> tuple[int, str, str | None]:
    """Returns (id, kind, version) with kind ok | no_version | missing | error."""
    rid, path = row
    try:
        value = str(fits.getval(path, "PIPE_VER")).strip()
    except KeyError:
        return (rid, "no_version", None)
    except FileNotFoundError:
        return (rid, "missing", None)
    except Exception as e:
        return (rid, "error", f"{type(e).__name__}: {e}")
    return ((rid, "ok", value) if value else (rid, "no_version", None))


def _flush(iq: ImageQA, batch: list[tuple[int, str]]) -> None:
    values = ",".join(["(%s::integer, %s::varchar)"] * len(batch))
    params = [x for pair in batch for x in pair]
    iq.execute_query(
        f"UPDATE image_qa AS q SET pipe_ver = v.ver FROM (VALUES {values}) AS v(id, ver) WHERE q.id = v.id",
        params,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill image_qa.pipe_ver from FITS PIPE_VER cards.")
    parser.add_argument("--dry-run", action="store_true", help="Read headers only; do not update.")
    parser.add_argument("--force-all", action="store_true", help="Do not require pipe_ver IS NULL.")
    parser.add_argument("--nightdate-from", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--nightdate-to", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--image-type", action="append", dest="image_types", metavar="TYPE", help="Repeatable.")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to process.")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help=f"Parallel header reads (max {MAX_WORKERS}).")
    parser.add_argument("--batch-size", type=int, default=1000, help="Rows per UPDATE statement.")
    args = parser.parse_args()

    workers = max(1, min(args.workers, MAX_WORKERS))

    iq = ImageQA()
    sql, params = _build_select_sql(args)
    rows = iq.execute_query(sql, params or None)
    if not rows:
        print("No matching rows.")
        return 0

    print(f"Selected {len(rows)} row(s). dry_run={args.dry_run} workers={workers}", flush=True)

    counts = {"ok": 0, "no_version": 0, "missing": 0, "error": 0}
    versions: dict[str, int] = {}
    batch: list[tuple[int, str]] = []
    started = time.time()

    done = 0
    # slice the work so only one batch of futures is alive at a time
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(rows), args.batch_size):
            for rid, kind, value in executor.map(_read_pipe_ver, rows[start : start + args.batch_size]):
                counts[kind] += 1
                if kind == "ok":
                    versions[value] = versions.get(value, 0) + 1
                    batch.append((rid, value))
                elif kind == "error":
                    print(f"[error] id={rid} {value}", flush=True)
            if batch and not args.dry_run:
                _flush(iq, batch)
                batch = []
            done = min(start + args.batch_size, len(rows))
            if done % 20000 < args.batch_size:
                rate = done / (time.time() - started)
                eta = (len(rows) - done) / rate / 3600
                print(f"  {done}/{len(rows)}  {rate:.0f} rows/s  eta {eta:.2f} h  {counts}", flush=True)

    print(f"Version histogram: {dict(sorted(versions.items()))}")
    print(
        f"Done in {(time.time() - started) / 3600:.2f} h. succeeded={counts['ok']}, "
        f"no_version={counts['no_version']}, failed={counts['missing'] + counts['error']} "
        f"(missing_file={counts['missing']}, read_error={counts['error']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
