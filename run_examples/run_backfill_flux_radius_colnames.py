#!/usr/bin/env python
"""
Rename the FLUX_RADIUS columns of catalogs written before the flux fractions reached the
column names: FLUX_RADIUS, FLUX_RADIUS_1, FLUX_RADIUS_2 -> FLUX_RADIUS_20, FLUX_RADIUS_50,
FLUX_RADIUS_80. The fractions are read from PHOT_FLUXFRAC in ref/srcExt/main.sex, the same
source Photometry.write_catalog uses, so the two always agree.

Only the TTYPEn cards of the LDAC_OBJECTS header change. That is an in-place header write
(measured 16 ms/file, size and inode unchanged); the 1.2 MB of table data is never rewritten.

Scope: science configs whose process_status.pipeline_version >= 1.10.47, the version that
first put FLUX_RADIUS into main.param. The catalogs are the *_cat.fits of the config's
photometry.input_images, imcoadd.input_images and imcoadd.coadd_image.

Resumable: a catalog that already carries the new names is skipped, so a killed run continues.

A catalog holding a lone FLUX_RADIUS is reported as 'single_fraction' and left alone: it
predates PHOT_FLUXFRAC and its one radius is SExtractor's default fraction 0.5, not 0.2.

Usage:
    python run_backfill_flux_radius_colnames.py --dry-run --limit 200
    python run_backfill_flux_radius_colnames.py --limit 2000       # a real first slice
    python run_backfill_flux_radius_colnames.py                    # the rest
    python run_backfill_flux_radius_colnames.py --workers 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from astropy.io import fits
from packaging.version import Version

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.config import SciProcConfiguration
from pipeline.photometry.utils import get_flux_fractions
from pipeline.services.database.process_status import ProcessStatus

MAX_WORKERS = 8  # writes over NFS
DEFAULT_WORKERS = 4
MIN_VERSION = "1.10.47"  # the version that put FLUX_RADIUS in main.param
DEFAULT_LOG = "/var/log/pipeline/backfill_flux_radius_colnames.log"

_log_lock = Lock()
_log_fh = None


def _log(msg: str) -> None:
    with _log_lock:
        print(msg, flush=True)
        if _log_fh is not None:
            _log_fh.write(msg + "\n")
            _log_fh.flush()


def _select_configs(args: argparse.Namespace) -> list[str]:
    sql = (
        "SELECT config_file, pipeline_version FROM process_status "
        "WHERE config_type = 'science' AND config_file IS NOT NULL AND config_file != '' "
        "ORDER BY id"
    )
    rows = ProcessStatus().execute_query(sql, None)
    minimum = Version(args.min_version)

    def recent(version) -> bool:
        try:
            return Version(str(version)) >= minimum
        except Exception:
            return False

    paths = [path for path, version in rows if recent(version)]
    return paths[: args.limit] if args.limit is not None else paths


def _rename_serialized_columns(header: fits.Header, old_names: list, new_names: list) -> None:
    """Rename inside astropy's serialized-columns YAML, which Table.read parses out of COMMENT cards."""
    indices = [i for i, card in enumerate(header.cards) if card.keyword == "COMMENT"]
    if not indices:
        return

    # unwrap: astropy splits each YAML line into 70-char chunks and marks continuation with a backslash
    lines, continued = [], False
    for value in [str(v) for v in header["COMMENT"]]:
        if continued:
            lines[-1] += value[:70]
        else:
            lines.append(value[:70])
        continued = len(value) == 71

    for old, new in zip(reversed(old_names), reversed(new_names)):
        lines = [line.replace(f"name: {old},", f"name: {new},") for line in lines]

    wrapped = []
    for line in lines:
        if not line:
            wrapped.append("")
            continue
        edges = list(range(0, len(line) + 70, 70))
        chunks = [line[a:b] + "\\" for a, b in zip(edges, edges[1:])]
        chunks[-1] = chunks[-1][:-1]
        wrapped.extend(chunks)

    for index in reversed(indices):  # descending, so each delete hits the card it was meant to
        del header[index]
    for offset, line in enumerate(wrapped):
        header.insert(indices[0] + offset, fits.Card("COMMENT", line))


def _rename_catalog(catalog: str, old_names: list, new_names: list, dry_run: bool) -> str:
    """Returns one of ok | done | single | absent | missing."""
    if not os.path.exists(catalog):
        return "missing"
    with fits.open(catalog, mode="update", checksum=False, memmap=False) as hdul:
        header = hdul[2].header
        present = {header[key]: key for key in header if key.startswith("TTYPE")}
        if new_names[0] in present:
            return "done"
        if old_names[0] not in present:
            return "absent"
        if not all(name in present for name in old_names):
            return "single"
        if dry_run:
            return "ok"
        # descending, so a new name never lands on an old one still waiting its turn
        for old, new in zip(reversed(old_names), reversed(new_names)):
            header[present[old]] = new
        _rename_serialized_columns(header, old_names, new_names)
        hdul[2].add_datasum(when="data unit checksum")
        hdul[2].add_checksum(when="HDU checksum", override_datasum=True)
    return "ok"


def _process_config(config_file: str, old_names: list, new_names: list, dry_run: bool) -> dict:
    try:
        node = SciProcConfiguration(config_file, logger=False, write=False).node
    except Exception as e:
        _log(f"[config] {config_file} :: {type(e).__name__}: {e}")
        return {"config_error": 1}

    images = list(node.photometry.input_images or []) + list(node.imcoadd.input_images or [])
    if node.imcoadd.coadd_image:
        images.append(node.imcoadd.coadd_image)

    counts: dict = {}
    for image in dict.fromkeys(images):
        catalog = image.replace(".fits", "_cat.fits")
        try:
            kind = _rename_catalog(catalog, old_names, new_names, dry_run)
        except Exception as e:
            kind = "error"
            _log(f"[error] {catalog} :: {type(e).__name__}: {e}")
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def main() -> int:
    global _log_fh
    parser = argparse.ArgumentParser(description="Rename FLUX_RADIUS vector columns in existing photometry catalogs.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be renamed; write nothing.")
    parser.add_argument("--min-version", type=str, default=MIN_VERSION, help=f"Default {MIN_VERSION}.")
    parser.add_argument("--limit", type=int, default=None, help="Max configs to process.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"Parallel workers (max {MAX_WORKERS}).")
    parser.add_argument("--chunk", type=int, default=500, help="Configs per progress report.")
    parser.add_argument("--log", type=str, default=DEFAULT_LOG, help="Append failures and progress here.")
    args = parser.parse_args()

    workers = max(1, min(args.workers, MAX_WORKERS))

    try:
        _log_fh = open(args.log, "a")
    except OSError as e:
        print(f"Cannot open log {args.log}: {e}; continuing with stdout only", flush=True)

    fractions = get_flux_fractions()
    old_names = ["FLUX_RADIUS"] + [f"FLUX_RADIUS_{i}" for i in range(1, len(fractions))]
    new_names = [f"FLUX_RADIUS_{round(f * 100)}" for f in fractions]

    configs = _select_configs(args)
    if not configs:
        _log("No matching configs.")
        return 0

    _log(
        f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} start: {len(configs)} config(s), "
        f"{dict(zip(old_names, new_names))}, dry_run={args.dry_run} workers={workers} ==="
    )

    counts: dict = {}
    started = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(configs), args.chunk):
            for result in executor.map(
                lambda c: _process_config(c, old_names, new_names, args.dry_run),
                configs[start : start + args.chunk],
            ):
                for kind, n in result.items():
                    counts[kind] = counts.get(kind, 0) + n
            done = min(start + args.chunk, len(configs))
            rate = done / max(time.time() - started, 1e-9)
            eta = (len(configs) - done) / rate / 60
            _log(f"  {done}/{len(configs)} configs  {rate:.1f} cfg/s  eta {eta:.1f} min  {counts}")

    _log(
        f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} done in {(time.time() - started) / 60:.1f} min: "
        f"renamed={counts.get('ok', 0)} already_renamed={counts.get('done', 0)} "
        f"no_flux_radius={counts.get('absent', 0)} single_fraction={counts.get('single', 0)} "
        f"missing_file={counts.get('missing', 0)} error={counts.get('error', 0)} "
        f"config_error={counts.get('config_error', 0)} ==="
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
