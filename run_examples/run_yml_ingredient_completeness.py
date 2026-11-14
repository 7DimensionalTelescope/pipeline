#!/usr/bin/env python
"""
yml ingredient completeness: does each preprocess yml list every raw frame the raw DB holds for its
unit-night? A run takes its raw frames from the yml (input.masterframe_images, input.science_images), so
a yml that lists fewer leaves them out of a rerun, while masterframe_association takes the list as complete.

Scope: every preprocess row of process_status, or the rows named by --config. Each row's config_file is
read as plain YAML (no Configuration object: their constructors write) and compared by file name with the
raw frames of its (nightdate, unit) in the raw DB, the list a rebuild of the config takes
(pipeline.run.query_observations). Read-only: SELECTs and yml reads; the table is the only file written.

Output (--out, csv; one row per config, by nightdate and unit): `state` and, per type (bias, dark, flat,
sci), the raw frames in the raw DB and how many of them the yml lists. n_missing = raw frames not listed,
n_unknown = listed files the raw DB does not hold for the unit-night, n_other_path = listed under another
path than the raw DB's. States: complete, incomplete, no raw frame (the raw DB holds none for the
unit-night, so a rebuild would list nothing), no file, unreadable. The incomplete rows are the configs to
write again:

    table = pd.read_csv("/var/log/pipeline/yml_ingredient_completeness.csv")
    nights = list(table.loc[table.state == "incomplete", ["nightdate", "unit"]].itertuples(index=False, name=None))

Usage:
    python run_yml_ingredient_completeness.py --config 2024-08-08_7DT03 2026-08-31_7DT11 --out trial.csv
    python run_yml_ingredient_completeness.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.const import CONFIG_TYPE_PREPROCESS
from pipeline.services.database.const import TABLES
from pipeline.services.database.query import free_query

DEFAULT_OUT = "/var/log/pipeline/yml_ingredient_completeness.csv"
DEFAULT_WORKERS = 8  # one raw-DB query and one yml read over NFS per config
TYPES = ("bias", "dark", "flat", "sci")
COUNTS = [f"{side}_{t}" for t in TYPES for side in ("raw", "listed")]
COLUMNS = ["nightdate", "unit", "name", "state", "n_missing", "n_unknown", "n_other_path", *COUNTS]
COLUMNS += ["config_file", "progress", "status", "pipeline_version"]
RAW_FRAMES = " UNION ALL ".join(
    f"SELECT '{t}', x.file_path FROM {TABLES[t]} x JOIN facility_unit u ON u.id = x.unit_id"
    " JOIN survey_night n ON n.id = x.night_id WHERE n.date = %s AND u.name = %s"
    for t in TYPES
)


def _rows(names: list | None) -> list:
    query = (
        "SELECT name, nightdate::text, unit, config_file, progress, status, pipeline_version FROM process_status"
        " WHERE config_type = %s {} ORDER BY nightdate, unit, id"
    )
    if names:
        return free_query(query.format("AND name = ANY(%s)"), (CONFIG_TYPE_PREPROCESS, names))
    return free_query(query.format(""), (CONFIG_TYPE_PREPROCESS,))


def _leaves(node) -> list:
    """A yml input list, flattened."""
    if isinstance(node, (list, tuple)):
        return [leaf for item in node for leaf in _leaves(item)]
    return [] if node is None else [str(node)]


def _survey(row: tuple) -> dict:
    name, nightdate, unit, config_file, progress, status, pipeline_version = row
    out = dict(nightdate=nightdate, unit=unit, name=name, config_file=config_file)
    out.update(progress=progress, status=status, pipeline_version=pipeline_version)
    if not config_file or not os.path.exists(config_file):
        return {**out, "state": "no file"}
    try:
        with open(config_file) as f:
            section = (yaml.load(f, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader)) or {}).get("input") or {}
        frames = free_query(RAW_FRAMES, [nightdate, unit] * len(TYPES))
    except Exception as e:
        print(f"[error] {name} {config_file} :: {type(e).__name__}: {e}", flush=True)
        return {**out, "state": "unreadable"}
    listed = {
        os.path.basename(p): p for key in ("masterframe_images", "science_images") for p in _leaves(section.get(key))
    }
    raw = {t: {} for t in TYPES}
    for t, path in frames:
        raw[t][os.path.basename(path)] = path
    for t in TYPES:
        have = raw[t].keys() & listed.keys()
        out[f"raw_{t}"], out[f"listed_{t}"] = len(raw[t]), len(have)
        out["n_other_path"] = out.get("n_other_path", 0) + sum(raw[t][b] != listed[b] for b in have)
    out["n_missing"] = sum(out[f"raw_{t}"] - out[f"listed_{t}"] for t in TYPES)
    out["n_unknown"] = len(listed.keys() - set().union(*raw.values()))
    out["state"] = "no raw frame" if not any(raw.values()) else "incomplete" if out["n_missing"] else "complete"
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", nargs="+", default=None, help="process_status names (default: every preprocess row)")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args()

    rows = _rows(args.config)
    print(f"[start] {time.strftime('%F %T')} {len(rows)} preprocess rows, workers {args.workers}", flush=True)
    if args.config and len(rows) != len(set(args.config)):
        print(f"[warning] not in process_status: {sorted(set(args.config) - {row[0] for row in rows})}", flush=True)

    total, missing, t0 = Counter(), Counter(), time.time()
    with open(args.out, "w", newline="") as f, ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for i, result in enumerate(pool.map(_survey, rows), 1):
            writer.writerow(result)
            total[result["state"]] += 1
            if result["state"] == "incomplete":
                missing.update({t: result[f"raw_{t}"] - result[f"listed_{t}"] for t in TYPES})
            if i % 1000 == 0:
                print(f"[progress] {i} rows, {i / (time.time() - t0):.0f} rows/s, {dict(total)}", flush=True)

    print(f"[done] {time.strftime('%F %T')} {len(rows)} rows in {time.time() - t0:.0f} s, {dict(total)}", flush=True)
    print(f"[done] raw frames the incomplete ymls do not list: {dict(missing)}; table: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
