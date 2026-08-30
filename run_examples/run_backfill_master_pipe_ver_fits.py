#!/usr/bin/env python
"""
Stamp PIPE_VER into the FITS header of master frames that predate preprocess stamping.

Anchors on image_qa: the value written is that row's ``pipe_ver``, which was populated from
``process_status.pipeline_version`` of the run that last registered the master. That is the
version of the RUN, not provably the version that produced the pixels, so the card carries a
comment marking it approximate and is never written over a card that already exists.

Every sampled master header has 7-28 free card slots, so this is an in-place header write
(measured 9 ms/file, size and inode unchanged) - it does not rewrite the 233 MB of pixels.
Where a ``.header`` sidecar exists the card goes there too, because ``get_header`` returns the
sidecar wholesale and would otherwise mask the FITS card.

Resumable: a file that already carries PIPE_VER is skipped, so re-running continues.

Usage:
    python run_backfill_master_pipe_ver_fits.py --dry-run --limit 200
    python run_backfill_master_pipe_ver_fits.py --limit 500          # a real first slice
    python run_backfill_master_pipe_ver_fits.py                      # the rest
    python run_backfill_master_pipe_ver_fits.py --image-type flat --workers 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from astropy.io import fits

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.const import MASTER_IMAGE_TYPES
from pipeline.services.database.image_qa import ImageQA
from pipeline.utils.filesystem import swap_ext
from pipeline.utils.header import read_header_file, write_header_file

MAX_WORKERS = 4  # writes over NFS: half the read-only backfill's cap
COMMENT = "Approx preprocess version from process_status"  # 45 chars; the 80-byte card allows 47
DEFAULT_LOG = "/var/log/pipeline/backfill_master_pipe_ver.log"

_log_lock = Lock()
_log_fh = None


def _log(msg: str) -> None:
    with _log_lock:
        print(msg, flush=True)
        if _log_fh is not None:
            _log_fh.write(msg + "\n")
            _log_fh.flush()


def _select(args: argparse.Namespace) -> tuple[str, dict]:
    conditions = [
        "image_type = ANY(%(types)s)",
        "image_path IS NOT NULL",
        "image_path != ''",
        "pipe_ver IS NOT NULL",
    ]
    params: dict = {"types": args.image_types or list(MASTER_IMAGE_TYPES)}
    if args.nightdate_from:
        conditions.append("nightdate >= %(nd_from)s")
        params["nd_from"] = args.nightdate_from
    if args.nightdate_to:
        conditions.append("nightdate <= %(nd_to)s")
        params["nd_to"] = args.nightdate_to
    sql = f"SELECT id, image_path, pipe_ver FROM image_qa WHERE {' AND '.join(conditions)} ORDER BY id"
    if args.limit is not None:
        sql += " LIMIT %(limit)s"
        params["limit"] = args.limit
    return sql, params


def _stamp(row: tuple, dry_run: bool, force: bool) -> tuple[str, str | None]:
    """Returns (kind, detail); kind is ok | present | missing | error."""
    rid, path, version = row
    try:
        if not os.path.exists(path):
            return ("missing", f"id={rid} {path}")
        if not force and "PIPE_VER" in fits.getheader(path):
            return ("present", None)
        if dry_run:
            return ("ok", None)

        with fits.open(path, mode="update") as hdul:
            hdul[0].header["PIPE_VER"] = (str(version), COMMENT)
            hdul.flush()

        # the sidecar is what get_header returns when it exists, so it must agree
        side = swap_ext(path, "header")
        if os.path.exists(side):
            hdr = read_header_file(side)
            hdr["PIPE_VER"] = (str(version), COMMENT)
            write_header_file(side, hdr)
        return ("ok", None)
    except Exception as e:
        return ("error", f"id={rid} {path} :: {type(e).__name__}: {e}")


def main() -> int:
    global _log_fh
    parser = argparse.ArgumentParser(description="Stamp PIPE_VER into pre-existing master frame FITS headers.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be written; write nothing.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing PIPE_VER card.")
    parser.add_argument("--image-type", action="append", dest="image_types", metavar="TYPE", help="Repeatable.")
    parser.add_argument("--nightdate-from", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--nightdate-to", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to process.")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help=f"Parallel writers (max {MAX_WORKERS}).")
    parser.add_argument("--chunk", type=int, default=500, help="Rows per progress report.")
    parser.add_argument("--log", type=str, default=DEFAULT_LOG, help="Append failures and progress here.")
    args = parser.parse_args()

    workers = max(1, min(args.workers, MAX_WORKERS))

    try:
        _log_fh = open(args.log, "a")
    except OSError as e:
        print(f"Cannot open log {args.log}: {e}; continuing with stdout only", flush=True)

    iq = ImageQA()
    sql, params = _select(args)
    rows = iq.execute_query(sql, params)
    if not rows:
        _log("No matching rows.")
        return 0

    _log(
        f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} start: {len(rows)} row(s), "
        f"dry_run={args.dry_run} force={args.force} workers={workers} ==="
    )

    counts = {"ok": 0, "present": 0, "missing": 0, "error": 0}
    started = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(rows), args.chunk):
            for kind, detail in executor.map(
                lambda r: _stamp(r, args.dry_run, args.force), rows[start : start + args.chunk]
            ):
                counts[kind] += 1
                if detail and kind in ("missing", "error"):
                    _log(f"[{kind}] {detail}")
            done = min(start + args.chunk, len(rows))
            rate = done / max(time.time() - started, 1e-9)
            eta = (len(rows) - done) / rate / 60
            _log(f"  {done}/{len(rows)}  {rate:.0f} files/s  eta {eta:.1f} min  {counts}")

    _log(
        f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} done in {(time.time() - started) / 60:.1f} min: "
        f"stamped={counts['ok']} already_present={counts['present']} "
        f"missing_file={counts['missing']} error={counts['error']} ==="
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
