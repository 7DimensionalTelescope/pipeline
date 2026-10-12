#!/usr/bin/env python
"""
Backfill image_qa.inspcomm (and sanity) from the INSPCOMM / SANITY cards of images sealed before the column
existed: every row with `sanity IS FALSE` is a candidate (the retired `inspectd` rows are a subset).

Idempotent: selects `sanity IS FALSE AND inspcomm IS NULL`, writes only rows whose FITS carries a human INSPCOMM
(the auto-binned master marker is not one), and reports the remainder (no card, file missing, unreadable).

Usage:
    python run_backfill_image_qa_inspcomm.py --dry-run
    python run_backfill_image_qa_inspcomm.py [--limit N] [--workers 4]
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from astropy.io import fits

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.services.database.image_qa import ImageQA

MAX_WORKERS = 4
BINNED_MARKER = "Auto-generated binned master frame"  # set_inspcomm_in_header's marker, not a verdict


def _read_verdict(row: tuple) -> tuple[int, str, tuple | None]:
    """Returns (id, kind, (sanity, inspcomm)) with kind ok | no_card | missing | error."""
    rid, path = row
    try:
        header = fits.getheader(path)
    except FileNotFoundError:
        return rid, "missing", None
    except Exception as e:
        return rid, "error", (None, f"{type(e).__name__}: {e}")
    text = str(header.get("INSPCOMM", "") or "").rstrip()
    if not text or text == BINNED_MARKER or "SANITY" not in header:
        return rid, "no_card", None
    return rid, "ok", (bool(header["SANITY"]), text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill image_qa.inspcomm from FITS INSPCOMM cards.")
    parser.add_argument("--dry-run", action="store_true", help="Read headers only; do not update.")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to process.")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help=f"Parallel header reads (max {MAX_WORKERS}).")
    args = parser.parse_args()

    iq = ImageQA()
    sql = "SELECT id, image_path FROM image_qa WHERE sanity IS FALSE AND inspcomm IS NULL AND image_path IS NOT NULL ORDER BY id"
    params = None
    if args.limit is not None:
        sql += " LIMIT %(limit)s"
        params = {"limit": args.limit}
    rows = iq.execute_query(sql, params)
    if not rows:
        print("No matching rows.")
        return 0
    print(f"Selected {len(rows)} row(s). dry_run={args.dry_run}", flush=True)

    counts = {"ok": 0, "no_card": 0, "missing": 0, "error": 0}
    remainder = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, MAX_WORKERS))) as executor:
        for n, (rid, kind, verdict) in enumerate(executor.map(_read_verdict, rows), 1):
            counts[kind] += 1
            if n % 5000 == 0:
                print(f"  {n}/{len(rows)} {counts}", flush=True)
            if kind != "ok":
                if kind != "no_card":  # machine rejections carry no card; list only what a re-inspection could fix
                    remainder.append((rid, kind, verdict[1] if verdict else None))
                continue
            if not args.dry_run:
                sanity, text = verdict
                iq.execute_query(
                    "UPDATE image_qa SET inspcomm = %(text)s, sanity = %(sanity)s WHERE id = %(id)s AND inspcomm IS NULL",
                    {"text": text, "sanity": sanity, "id": rid},
                )

    print(f"Done: {counts}")
    for rid, kind, detail in remainder:
        print(f"  remainder id={rid} {kind}" + (f" {detail}" if detail else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
