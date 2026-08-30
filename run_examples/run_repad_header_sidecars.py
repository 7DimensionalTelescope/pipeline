#!/usr/bin/env python
"""
Re-pad `.header` sidecars so `reset_header` stops shrinking the FITS header.

`write_fits_image` pads the FITS header to `n_head_blocks`, but `write_header` writes the sidecar
unpadded. `Astrometry` stamps the short sidecar onto the frame (`reset_image_header=True`), the
header loses a block, the data shifts, and the frame is rewritten -- then twice more as the WCS and
photometry cards grow it back. Measured: 3.12x write amplification, 3 inode changes per frame.

Padding each sidecar to the size a reduced frame settles at (8 blocks) makes `reset_header` grow the
frame once, up front; the WCS and photometry cards then fit and the other two rewrites disappear.

Image list comes from the scheduler (what is queued now). Every frame examined is logged to a local
sqlite so a later full sweep can diff this against image_qa.

    python run_repad_header_sidecars.py --limit 200            # dry run, logs only
    python run_repad_header_sidecars.py --limit 200 --apply
"""

import argparse
import math
import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from pipeline.const import SCHEDULER_DB_PATH, TASK_STATUS_PENDING, TASK_STATUS_READY
from pipeline.utils.header import read_header_file, write_header_file, add_padding

LOG_DB = "/var/db/sidecar_fix.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sidecar_fix (
    image TEXT PRIMARY KEY,
    header_file TEXT,
    config TEXT,
    scheduler_index INTEGER,
    pipeline_version TEXT,
    fits_blocks INTEGER,
    cards_before INTEGER,
    blocks_before INTEGER,
    cards_after INTEGER,
    blocks_after INTEGER,
    sidecar_mtime TEXT,
    action TEXT,
    error TEXT,
    ts TEXT
)
"""


BLOCK = 2880
CARD = 80
READ_AHEAD = 32 * BLOCK  # one read covers any header we have ever seen; NFS hates 80-byte reads


def fits_header_blocks(path):
    """Blocks the frame's own header occupies -- the size the sidecar must not undercut."""
    with open(path, "rb") as fh:
        buf, base = fh.read(READ_AHEAD), 0
        while True:
            for off in range(0, len(buf) - CARD + 1, CARD):
                if buf[off : off + 3] == b"END":
                    return ((base + off) // CARD) // 36 + 1
            base += len(buf) - (len(buf) % CARD)
            buf = fh.read(READ_AHEAD)
            if not buf:
                raise ValueError("no END card")


def sidecar_stats(path):
    header = read_header_file(path)
    n = len(header.cards)
    return header, n, math.ceil((n + 1) / 36)


def queued_configs(scheduler_db):
    """Configs the scheduler still has queued."""
    con = sqlite3.connect(f"file:{scheduler_db}?mode=ro", uri=True, timeout=30)
    rows = con.execute(
        'SELECT "index", config FROM scheduler WHERE status IN (?, ?) ORDER BY "index"',
        (TASK_STATUS_READY, TASK_STATUS_PENDING),
    ).fetchall()
    con.close()
    return rows


def already_done(log_db):
    """Images a previous pass settled, so a resumed run does not re-read them off NFS."""
    if not os.path.exists(log_db):
        return set()
    con = sqlite3.connect(f"file:{log_db}?mode=ro", uri=True, timeout=30)
    done = {r[0] for r in con.execute("SELECT image FROM sidecar_fix WHERE action IN ('padded','ok')")}
    con.close()
    return done


def process_status_versions():
    """{config_name: pipeline_version} for every row -- process_status, not image_qa.pipe_ver."""
    import psycopg
    from pipeline.services.database.const import DB_PARAMS

    with psycopg.connect(**DB_PARAMS) as con:
        return dict(con.execute("select name, pipeline_version from pipeline.process_status"))


def one_config(index, config, target, apply_writes, done):
    """All frames of one config. Runs on a worker thread; touches no sqlite."""
    name = os.path.basename(config)[: -len(".yml")]
    out = []
    try:
        node = yaml.safe_load(open(config))
    except Exception as e:
        return [dict(_blank(index, config, config), action="error", error=f"{type(e).__name__}: {e}")]

    for image in (node.get("input", {}) or {}).get("calibrated_images") or []:
        if image in done:
            continue
        row = _blank(index, config, image)
        row["_name"] = name
        header_file = row["header_file"]
        try:
            if not os.path.exists(header_file):
                row["action"] = "missing"
            else:
                row["sidecar_mtime"] = datetime.fromtimestamp(os.path.getmtime(header_file)).isoformat()
                row["fits_blocks"] = fits_header_blocks(image)
                header, row["cards_before"], row["blocks_before"] = sidecar_stats(header_file)
                want = max(target, row["fits_blocks"])
                if row["blocks_before"] >= want:
                    row["action"] = "ok"
                    row["cards_after"], row["blocks_after"] = row["cards_before"], row["blocks_before"]
                else:
                    padded = add_padding(header, want)
                    if apply_writes:
                        write_header_file(header_file, padded)
                        _, row["cards_after"], row["blocks_after"] = sidecar_stats(header_file)
                        row["action"] = "padded"
                    else:
                        row["cards_after"] = len(padded.cards)
                        row["blocks_after"] = math.ceil((len(padded.cards) + 1) / 36)
                        row["action"] = "would_pad"
        except Exception as e:
            row["action"] = "error"
            row["error"] = f"{type(e).__name__}: {e}"
        out.append(row)
    return out


def _blank(index, config, image):
    return dict(image=image, header_file=image.replace(".fits", ".header"), config=config,
                scheduler_index=index, pipeline_version=None, fits_blocks=None, cards_before=None,
                blocks_before=None, cards_after=None, blocks_after=None, sidecar_mtime=None,
                action=None, error=None, ts=datetime.now().isoformat())


INSERT = ("INSERT OR REPLACE INTO sidecar_fix VALUES "
          "(:image,:header_file,:config,:scheduler_index,:pipeline_version,:fits_blocks,"
          ":cards_before,:blocks_before,:cards_after,:blocks_after,:sidecar_mtime,:action,:error,:ts)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many configs (0 = all queued)")
    ap.add_argument("--blocks", type=int, default=8,
                    help="target header blocks; 8 = n_head_blocks, the size a reduced frame settles at")
    ap.add_argument("--apply", action="store_true", help="write the sidecars; otherwise dry run")
    ap.add_argument("--workers", type=int, default=32, help="concurrent configs; this is NFS wait, not CPU")
    ap.add_argument("--redo", action="store_true", help="re-examine images a previous pass settled")
    ap.add_argument("--log-db", default=LOG_DB)
    ap.add_argument("--scheduler-db", default=SCHEDULER_DB_PATH)
    args = ap.parse_args()

    log = sqlite3.connect(args.log_db, timeout=60)
    log.execute(SCHEMA)
    log.commit()

    done = set() if args.redo else already_done(args.log_db)
    configs = queued_configs(args.scheduler_db)
    if args.limit:
        configs = configs[: args.limit]
    versions = process_status_versions()
    print(f"{len(configs)} queued configs | {len(done)} images already settled | "
          f"{len(versions)} process_status versions | {args.workers} workers", flush=True)

    counts, n_rows, t0 = {}, 0, time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one_config, i, c, args.blocks, args.apply, done) for i, c in configs]
        for n_cfg, fut in enumerate(as_completed(futures), 1):
            for row in fut.result():
                row["pipeline_version"] = versions.get(row.pop("_name", None))
                counts[row["action"]] = counts.get(row["action"], 0) + 1
                log.execute(INSERT, row)
                n_rows += 1
            if n_cfg % 2000 == 0:
                log.commit()
                rate = n_rows / max(time.time() - t0, 1e-9)
                print(f"  {n_cfg}/{len(configs)} configs, {n_rows} frames ({rate:.0f}/s): "
                      + " ".join(f"{k}={v}" for k, v in sorted(counts.items())), flush=True)
    log.commit()

    for action, n in sorted(counts.items()):
        print(f"  {action:<10} {n}")
    print(f"log: {args.log_db}  ({n_rows / max(time.time() - t0, 1e-9):.0f} frames/s)")


if __name__ == "__main__":
    main()
