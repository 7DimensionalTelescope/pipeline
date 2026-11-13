#!/usr/bin/env python
"""
Fill the master-selection columns of process_status (max_offset, ignore_sanity_if_no_match,
ignore_lenient_keys_if_no_match, reuse_recorded_masters, designated_masterframes; added 2026-11-12)
for the preprocess configs Preprocess has not rewritten since. masterframe_association reads them
there and opens no yml; a row whose three search settings are NULL is searched with the base settings
and reported 'settings-unrecorded'.

Scope: every preprocess row of process_status, or the rows named by --config. Each row's config_file
is read as plain YAML (no Configuration object: their constructors write), its preprocess section goes
through pipeline.preprocess.reprocess.master_selection, the same function Preprocess writes with, and
ProcessStatus.update_data writes the five values to the row (a None as SQL NULL). A designation path
that does not exist is resolved by its basename on disk, as Preprocess does.

An update moves the row's updated_at (the table's trigger update_process_status_updated_at).

A dry run reads the ymls and writes nothing; both print the counts: relaxations on / off, rows without
a file, unreadable files, designations (relaxations mixed: one of the two on).

Run it from the development checkout: the deployed ProcessStatusTable does not have the new fields.

Usage:
    python run_backfill_master_selection.py --dry-run
    python run_backfill_master_selection.py --config 2026-08-31_7DT11 2026-04-21_7DT01
    python run_backfill_master_selection.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.const import CONFIG_TYPE_PREPROCESS
from pipeline.preprocess.reprocess import master_selection
from pipeline.services.database.process_status import ProcessStatus
from pipeline.services.database.query import free_query

DEFAULT_LOG = "/var/log/pipeline/backfill_master_selection.log"

_log_fh = None


def _log(msg: str) -> None:
    print(msg, flush=True)
    if _log_fh is not None:
        _log_fh.write(msg + "\n")
        _log_fh.flush()


def _rows(names: list | None) -> list:
    if names:
        query = "SELECT id, name, config_file FROM process_status WHERE config_type = %s AND name = ANY(%s) ORDER BY id"
        return free_query(query, (CONFIG_TYPE_PREPROCESS, names))
    query = "SELECT id, name, config_file FROM process_status WHERE config_type = %s ORDER BY id"
    return free_query(query, (CONFIG_TYPE_PREPROCESS,))


def _selection(config_file: str) -> dict:
    with open(config_file) as f:
        yml = yaml.load(f, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader)) or {}
    is_pipeline = bool((yml.get("settings") or {}).get("is_pipeline"))  # Preprocess: get_key(..., False)
    return master_selection(yml.get("preprocess") or {}, is_pipeline)


def main() -> int:
    global _log_fh
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="read the ymls, write nothing")
    p.add_argument("--config", nargs="+", default=None, help="process_status names (default: every preprocess row)")
    p.add_argument("--log", default=DEFAULT_LOG)
    args = p.parse_args()

    _log_fh = open(args.log, "a")
    rows = _rows(args.config)
    _log(f"[start] {time.strftime('%F %T')} {len(rows)} preprocess rows, dry_run {args.dry_run}")
    if args.config and len(rows) != len(set(args.config)):
        _log(f"[warning] not in process_status: {sorted(set(args.config) - {name for _, name, _ in rows})}")

    total, t0 = Counter(), time.time()
    process_status = None if args.dry_run else ProcessStatus()
    for i, (row_id, name, config_file) in enumerate(rows, 1):
        if not config_file or not os.path.exists(config_file):
            total["no file"] += 1
            continue
        try:
            values = _selection(config_file)
        except Exception as e:
            _log(f"[error] id {row_id} {config_file} :: {type(e).__name__}: {e}")
            total["unreadable"] += 1
            continue
        relaxed = {values["ignore_sanity_if_no_match"], values["ignore_lenient_keys_if_no_match"]}
        total[f"relaxations {'mixed' if len(relaxed) > 1 else 'on' if True in relaxed else 'off'}"] += 1
        if values["designated_masterframes"] is not None:
            total["designations"] += 1
            _log(f"[designation] id {row_id} {name}: {json.dumps(values)}")
        if not args.dry_run:
            designated = values["designated_masterframes"]
            values["designated_masterframes"] = None if designated is None else json.dumps(designated)
            process_status.update_data(row_id, **{k: "None" if v is None else v for k, v in values.items()})
            total["written"] += 1
        if i % 1000 == 0:
            _log(f"[progress] {i} rows, {i / (time.time() - t0):.0f} rows/s, {dict(total)}")

    _log(f"[done] {time.strftime('%F %T')} {len(rows)} rows in {time.time() - t0:.0f} s, {dict(total)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
