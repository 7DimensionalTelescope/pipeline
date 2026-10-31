#!/usr/bin/env python
"""
Recompute PPFLAG deterministically from the master frames each product actually used, and
backfill it on disk and in image_qa, for every registered image (or a selection).

PPFLAG records the outcome of the master-frame choice, so every bit is a comparison of the
product's keys with the keys of the master it used (the dependency record), not a replay of
the fetch. For a single against each of its bias, dark and flat:
  1   the master's nightdate is not the single's (both the date directory of the path)
  2   kept as the product already carries it (a manual master is not derivable); also set on a
      flat whose bias/dark/source flat has another binning (an auto-binned copy, as the
      pipeline stamps it)
  4   the master's SANITY card is False (the FITS card, which the pipeline's staleness test reads)
  8   a lenient key differs (bias/dark: unit; flat: gain, camera)
  16  the master's nightdate lies outside the single's masterframe walls (the pipeline's dtype-blind
      `get_masterframe_walls`; `--dtype-walls` applies only the changelog rows that wall that dtype)
  32  a hard key differs (binning for all; gain/camera for bias/dark; exptime for the dark;
      filter for the flat)
  64  the bias master is not the one inside the dark master (by IMAGEID, else by name)
plus the OR of the masters' own recomputed PPFLAGs. A master dark is judged against its bias,
a flat against its bias and flatdark (no exptime check for the flatdark) or, for a binned
copy, against its source flat; a bias is 0. A coadd or white is the OR of its inputs, a diff
carries its science coadd's value. Keys come from NameHandler on the name (the authority;
image_qa columns are wrong on some rows), the nightdate from the path's date directory.

The record of what was used: image_qa_dependency edges first; a product without edges falls
back to its own IMCMB (singles: the sidecar; masters: the FITS, their sidecar holds only
statistics), IMG*/IID* (coadds, whites) or SCIIMG (diffs) cards. A master named by a card but
not registered gets its keys from the name and its SANITY from its FITS when the file exists.

Masters are always loaded and evaluated whole, whatever the selection: a single's value needs
its masters' recomputed values. They are WRITTEN only when `master` is among --types and no
selection filter is given (or `--write-masters`).

Disk: the FITS, its `.header` sidecar and the companion maps that copy the header (single
`_weight`/`_ivar`; coadd `_weight`/`_counts`/`_egain`; white `_weight`/`_counts`), through
`update_padded_header_smart` / `update_padded_header_file`. A FITS with no PPFLAG card, no
padding COMMENT and a full last header block can only take the card by growing a block, which
astropy does by rewriting the whole file (temp copy, delete, rename; about 3.5 % of the
card-less singles); such files are reported as "needs rewrite" and skipped unless
`--allow-rewrite`. DB: `image_qa.ppflag`, batched UPDATEs through `ImageQA.execute_query`.
Not touched: `_cat.fits` (its LDAC_IMHEAD text keeps the header as it stood at SExtractor
time; nothing reads PPFLAG from it) and phot7ds catalogs (no PPFLAG card).

Evaluation reads only the DB, one FITS header per master, and one sidecar per single without
edges; a product's own FITS is read at write time. A product whose image_qa.ppflag already
equals the recomputed value is reported unchanged without opening it (`--verify-headers`
opens every FITS).

Usage:
    # Dry-run over everything (default): change table per type, per-bit differences, a CSV
    python run_examples/run_backfill_ppflag.py

    # A selection: objects, nightdate range, units, types (chain order is enforced)
    python run_examples/run_backfill_ppflag.py --objects NGC0253 NGC0628
    python run_examples/run_backfill_ppflag.py --nightdate-from 2024-07-01 --nightdate-to 2025-09-30 --types single

    # Apply (disk + DB), 7 worker threads for the file I/O
    python run_examples/run_backfill_ppflag.py --apply --workers 7 --csv ~/ppflag_apply.csv

Operations: the queue keeps writing products across all years, so pause dispatch for the apply
(or rerun a dry-run afterwards and apply what moved); stop with Ctrl-C and wait for the chunk,
never kill -9 (a growing header is mid-rewrite). Resumable: rerun after an interruption; rows
already applied come back unchanged, rows whose headers were written but whose DB update did
not land are re-evaluated, found equal on disk and only the DB row is updated. A derived type
takes a parent's value from the parent phase of the same invocation, else from the parent's
image_qa.ppflag; with --limit a derived product with unknown inputs is reported, not written.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import numpy as np
from astropy.io import fits

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.const import CALIB_TYPE_BIAS, CALIB_TYPE_DARK, CALIB_TYPE_FLAT  # noqa: E402
from pipeline.const.instrum_log import get_changelog, get_masterframe_walls, normalize_wall  # noqa: E402
from pipeline.const.observation import (  # noqa: E402
    BIAS_GROUP_LENIENT_KEYS,
    DARK_GROUP_LENIENT_KEYS,
    FLAT_GROUP_LENIENT_KEYS,
)
from pipeline.path import PathHandler  # noqa: E402
from pipeline.path.name import NameHandler  # noqa: E402
from pipeline.path.utils import get_nightdate  # noqa: E402
from pipeline.preprocess import ppflag  # noqa: E402
from pipeline.services.database.image_qa import ImageQA  # noqa: E402
from pipeline.services.database.query import free_query  # noqa: E402
from pipeline.utils.header import get_header, update_padded_header_file, update_padded_header_smart  # noqa: E402

PPFLAG_COMMENTS = {
    "coadd": "Preprocessing quality flag (OR of coadded inputs)",
    "white": "Preprocessing quality flag (OR of coadded inputs)",
}
PPFLAG_COMMENT = "Preprocessing quality flag"
TYPES = ("master", "single", "coadd", "white", "diff")
MASTER_TYPES = (CALIB_TYPE_BIAS, CALIB_TYPE_DARK, CALIB_TYPE_FLAT)
LENIENT_KEYS = {
    CALIB_TYPE_BIAS: tuple(BIAS_GROUP_LENIENT_KEYS),
    CALIB_TYPE_DARK: tuple(DARK_GROUP_LENIENT_KEYS),
    CALIB_TYPE_FLAT: tuple(FLAT_GROUP_LENIENT_KEYS),
}
# the keys of a master's name that are neither lenient nor the date: a mismatch is bit 32
HARD_KEYS = {
    CALIB_TYPE_BIAS: ("n_binning", "gain", "camera"),
    CALIB_TYPE_DARK: ("n_binning", "gain", "camera", "exptime"),
    CALIB_TYPE_FLAT: ("n_binning", "filter"),
}
KEY_COLUMNS = ("nightdate", "unit", "gain", "camera", "n_binning", "filter", "exptime")
ROW_COLUMNS = ("id", "image_name", "image_path", "image_type", "imageid", "sanity", "ppflag")
COMPANIONS = {
    "master": (),
    "single": (PathHandler.weight_map, PathHandler.ivar_map),
    "coadd": (PathHandler.weight_map, PathHandler.counts, PathHandler.egain_map),
    "white": (PathHandler.weight_map, PathHandler.counts),
    "diff": (),
}
ROLE_OF_PARENT = {"coadd": "single", "white": "coadd", "diff": "science"}
BIT_NAMES = {1: "date", 2: "manual", 4: "sanity", 8: "lenient", 16: "wall", 32: "hard", 64: "bias"}
FITS_BLOCK = 2880


def _fill_keys(rec: dict) -> dict:
    """Comparison keys from NameHandler on the name, the nightdate from the path's date directory."""
    try:
        h = NameHandler(rec["image_name"] + ".fits")
        for k in KEY_COLUMNS:
            rec[k] = getattr(h, k, None)
    except Exception:
        for k in KEY_COLUMNS:
            rec.setdefault(k, None)
    if rec.get("image_path"):
        rec["nightdate"] = get_nightdate(rec["image_path"]) or rec.get("nightdate")
    return rec


def _same(a, b) -> bool:
    if a is None or b is None:
        return True  # an unknown key cannot be a mismatch
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) < 1e-3
    return str(a) == str(b)


def _as_date(value):
    if value is None or isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def _needs_rewrite(header: fits.Header) -> bool:
    """A card can only be added by growing a block: no PPFLAG, no padding COMMENT, last block full."""
    if "PPFLAG" in header or any(c.keyword == "COMMENT" and not str(c.value).strip() for c in header.cards):
        return False
    return (len(header.cards) + 1) * 80 % FITS_BLOCK == 0


class Backfill:
    def __init__(self, args):
        self.args = args
        self.image_qa = ImageQA()
        self.new_by_id: dict[int, int] = {}
        self.new_by_imageid: dict[str, int] = {}
        self.new_by_name: dict[str, int] = {}
        self.masters_by_id: dict[int, dict] = {}
        self.masters_by_name: dict[str, dict] = {}
        self.master_ingredients: dict[int, dict] = {}  # master id -> {role: ingredient dict}
        self._walls: dict[tuple, tuple] = {}
        self._shared_cache: dict[tuple, bool | None] = {}
        self._unregistered: dict[str, dict] = {}
        self._lock = threading.Lock()
        self.summary: dict[str, Counter] = defaultdict(Counter)
        self.bitdiff: dict[str, Counter] = defaultdict(Counter)
        self.csv_writer = None
        self.selection = bool(args.objects or args.nightdate_from or args.nightdate_to or args.units)

    # ---- DB access
    def _query(self, sql: str, params=None):
        return free_query(sql, params, statement_timeout_ms=1800000)

    def _filters(self, image_types: tuple[str, ...]) -> tuple[str, list]:
        where = ["image_path IS NOT NULL", "image_type = ANY(%s)"]
        params: list = [list(image_types)]
        if self.args.objects:
            where.append("object = ANY(%s)")
            params.append(list(self.args.objects))
        if self.args.nightdate_from:
            where.append("nightdate >= %s")
            params.append(self.args.nightdate_from)
        if self.args.nightdate_to:
            where.append("nightdate <= %s")
            params.append(self.args.nightdate_to)
        if self.args.units:
            where.append("unit = ANY(%s)")
            params.append(list(self.args.units))
        return " AND ".join(where), params

    def _rows(self, image_types: tuple[str, ...], filtered: bool = True, limit: int | None = None):
        """Yield image_qa rows as dicts in id order, keyset-paginated so no page scans the table."""
        if filtered:
            where, params = self._filters(image_types)
        else:
            where, params = "image_path IS NOT NULL AND image_type = ANY(%s)", [list(image_types)]
        last, seen = 0, 0
        while True:
            rows = self._query(
                f"SELECT {', '.join(ROW_COLUMNS)} FROM image_qa WHERE id > %s AND {where} ORDER BY id LIMIT %s",
                [last, *params, self.args.chunk],
            )
            if not rows:
                return
            for row in rows:
                yield dict(zip(ROW_COLUMNS, row))
                seen += 1
                if limit and seen >= limit:
                    return
            last = rows[-1][0]

    def _edges(self, derived_ids: list[int]) -> dict[int, list[tuple]]:
        """derived id -> [(role, source row dict)] with the source's current ppflag and keys."""
        if not derived_ids:
            return {}
        rows = self._query(
            f"SELECT e.derived_image_id, e.dependency_role, e.source_imageid, {', '.join('s.' + c for c in ROW_COLUMNS)}"
            " FROM image_qa_dependency e JOIN image_qa s ON s.id = e.source_image_id WHERE e.derived_image_id = ANY(%s)",
            (derived_ids,),
        )
        out: dict[int, list[tuple]] = defaultdict(list)
        for row in rows:
            src = self.masters_by_id.get(row[3])
            if src is None:
                src = _fill_keys(dict(zip(ROW_COLUMNS, row[3:])))
            src = dict(src, used_imageid=row[2])
            out[int(row[0])].append((row[1], src))
        return out

    # ---- masters
    def _read_master_header(self, master: dict):
        """SANITY and the ingredient cards from the master's FITS (its sidecar holds only statistics)."""
        try:
            header = fits.getheader(master["image_path"])
        except Exception as exc:
            master["header_error"] = repr(exc)
            return
        sanity = header.get("SANITY")
        if sanity is not None:
            master["sanity"] = bool(sanity) if not isinstance(sanity, str) else sanity.strip().upper() != "F"
        master["hdr_ppflag"] = header.get("PPFLAG")
        master["cards"] = [
            (os.path.basename(str(header[k])), header.get("IMCID" + k[5:]))
            for k in header
            if str(k).startswith("IMCMB") and str(header[k]).split("_")[0] in MASTER_TYPES
        ]

    def load_masters(self):
        st = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            rows = [_fill_keys(r) for r in self._rows(MASTER_TYPES, filtered=False)]
        with ThreadPoolExecutor(max_workers=self.args.workers) as pool:
            list(pool.map(self._read_master_header, rows))
        for r in rows:
            self.masters_by_id[r["id"]] = r
            self.masters_by_name[r["image_name"]] = r
        errors = sum(1 for r in rows if r.get("header_error"))
        print(
            f"[master] {len(rows)} masters loaded, FITS headers read, in {time.time() - st:.0f} s"
            + (f"; {errors} unreadable" if errors else ""),
            flush=True,
        )
        return rows

    def _resolve_master(self, name: str) -> dict:
        """A master by basename: its image_qa row, else keys from the name and SANITY/cards from its FITS when the
        file PathHandler points at exists (else SANITY taken as True)."""
        stem = os.path.basename(str(name)).replace(".fits", "")
        row = self.masters_by_name.get(stem)
        if row is not None:
            return row
        with self._lock:
            if stem in self._unregistered:
                return self._unregistered[stem]
        master = {
            "image_name": stem,
            "image_path": None,
            "id": None,
            "imageid": None,
            "sanity": True,
            "ppflag": None,
            "unregistered": True,
        }
        try:
            path = PathHandler(stem + ".fits").preprocess._masterframe
            path = path[0] if isinstance(path, list) else path
            if path and os.path.exists(path):
                master["image_path"] = path
        except Exception:
            pass
        _fill_keys(master)
        if master["image_path"]:
            self._read_master_header(master)
            try:
                master["imageid"] = fits.getval(master["image_path"], "IMAGEID")
            except Exception:
                pass
        with self._lock:
            self._unregistered[stem] = master
        return master

    # ---- rules
    def _walls_of(self, nightdate, unit: str, dtype: str, dtype_aware: bool) -> tuple:
        """The masterframe walls around a night: the pipeline's dtype-blind `get_masterframe_walls`, or the same
        bounds taken only from changelog rows whose `wall` column names this dtype."""
        key = (str(nightdate), unit, dtype, dtype_aware)
        with self._lock:
            if key in self._walls:
                return self._walls[key]
        try:
            unit_no = int(str(unit)[3:])
            if not dtype_aware:
                walls = get_masterframe_walls(str(nightdate), unit_no, dtype=dtype)
            else:
                tbl = normalize_wall(get_changelog(unit_no), dtype=dtype)
                tbl = tbl[tbl[f"wall_{dtype}"]]
                tbl.sort("nightdate")
                dates = np.array(tbl["nightdate"], dtype=object)
                idx = np.searchsorted(dates, _as_date(nightdate), side="right")
                lower = dates[idx - 1] if idx - 1 >= 0 else None
                upper = dates[idx] - timedelta(days=1) if idx < len(dates) else None
                walls = (lower, upper)
        except Exception:
            walls = (None, None)
        with self._lock:
            self._walls[key] = walls
        return walls

    def _outside_walls(self, product: dict, m_night: date, dtype: str, dtype_aware: bool) -> bool:
        lower, upper = self._walls_of(product["nightdate"], product["unit"], dtype, dtype_aware)
        return (lower is not None and m_night < lower) or (upper is not None and m_night > upper)

    def _bits_against(self, product: dict, master: dict, dtype: str, check_exptime: bool = True) -> int:
        """The bits a product earns from one master it used, by comparing their keys."""
        bits = 0
        if not _same(master.get("nightdate"), product.get("nightdate")):
            bits |= ppflag.PPFLAG_DIFFERENT_DATE
        if master.get("sanity") is False:
            bits |= ppflag.PPFLAG_SANITY_F_USED
        if any(not _same(master.get(k), product.get(k)) for k in LENIENT_KEYS[dtype]):
            bits |= ppflag.PPFLAG_LENIENT_KEYS_IGNORED
        hard = [k for k in HARD_KEYS[dtype] if check_exptime or k != "exptime"]
        if product.get("image_type") == CALIB_TYPE_FLAT and not _same(
            master.get("n_binning"), product.get("n_binning")
        ):
            bits |= ppflag.PPFLAG_MANUAL  # an auto-binned flat copy keeps its source's 1x1 ingredients
            hard = [k for k in hard if k != "n_binning"]
        if any(not _same(master.get(k), product.get(k)) for k in hard):
            bits |= ppflag.PPFLAG_HARD_KEYS_IGNORED
        m_night = _as_date(master.get("nightdate"))
        if m_night is not None and product.get("nightdate") and product.get("unit"):
            chosen = self._outside_walls(product, m_night, dtype, self.args.dtype_walls)
            other = self._outside_walls(product, m_night, dtype, not self.args.dtype_walls)
            if chosen:
                bits |= ppflag.PPFLAG_MASTERFRAME_WALL_IGNORED
            if chosen != other:
                product["wall_rule_differs"] = True
        return bits

    def _ingredients(self, rec: dict, edges: list[tuple], roles: tuple[str, ...]) -> dict[str, dict]:
        """role -> master used, from the edges, else from the product's IMCMB cards."""
        found = {role: src for role, src in edges if role in roles}
        if any(role not in found for role in roles if role != CALIB_TYPE_FLAT or rec["type"] != "master"):
            if rec["type"] == "master":
                cards = rec.get("cards")
            else:
                try:
                    header = get_header(rec["image_path"])
                    cards = [(os.path.basename(str(header[k])), None) for k in header if str(k).startswith("IMCMB")]
                except Exception as exc:
                    rec["note"] = f"no edges and header unreadable: {exc!r}"
                    return found
            for stem, _ in cards or []:
                dtype = stem.split("_")[0]
                if dtype in roles and dtype not in found:
                    found[dtype] = self._resolve_master(stem)
            rec["from_cards"] = True
        return found

    def _master_value(self, master: dict) -> int | None:
        if master.get("id") in self.new_by_id:
            return self.new_by_id[master["id"]]
        return master.get("ppflag")

    @staticmethod
    def _keep_manual(old) -> int:
        return (int(old) & ppflag.PPFLAG_MANUAL) if old is not None else 0

    @staticmethod
    def _unregistered_note(role: str, master: dict) -> str:
        return f"{role} {master['image_name']} not registered" + (
            "" if master.get("image_path") else "; keys from the name, SANITY taken as True"
        )

    # ---- evaluation per type
    def _evaluate_master(self, rec: dict, edges: list[tuple]):
        dtype = rec["image_type"]
        old = rec["ppflag"]
        if dtype == CALIB_TYPE_BIAS:
            rec["new"] = self._keep_manual(old)
            return
        roles = (CALIB_TYPE_BIAS,) if dtype == CALIB_TYPE_DARK else MASTER_TYPES
        used = self._ingredients(rec, edges, roles)
        source = used.pop(CALIB_TYPE_FLAT, None)  # a binned copy's source flat
        self.master_ingredients[rec["id"]] = used
        value = self._keep_manual(old)
        if source is not None:
            if source.get("id") not in self.new_by_id and not source.get("unregistered"):
                rec["deferred"] = True  # the source flat is evaluated later in this pass
                return
            value |= ppflag.PPFLAG_MANUAL | (int(self._master_value(source) or 0) & ppflag.PPFLAG_MAX)
        for role, master in used.items():
            value |= self._bits_against(rec, master, role, check_exptime=(dtype != CALIB_TYPE_FLAT))
            value |= int(self._master_value(master) or 0) & ppflag.PPFLAG_MAX
            if master.get("unregistered"):
                rec["note"] = self._unregistered_note(role, master)
        rec["masters"] = " ".join(f"{r}={m['image_name']}" for r, m in used.items()) + (
            f" flat={source['image_name']}" if source else ""
        )
        rec["new"] = value if (used or source) else (old if old is not None else None)
        if not used and not source:
            rec["note"] = "no ingredient known; left as is"

    def _evaluate_single(self, rec: dict, edges: list[tuple]):
        old = rec["ppflag"]
        used = self._ingredients(rec, edges, MASTER_TYPES)
        rec["masters"] = " ".join(f"{r}={m['image_name']}" for r, m in used.items())
        if not used:
            rec["new"] = old
            rec["note"] = rec.get("note") or "no master known; left as is"
            return
        value = self._keep_manual(old)
        for role, master in used.items():
            value |= self._bits_against(rec, master, role)
            value |= int(self._master_value(master) or 0) & ppflag.PPFLAG_MAX
            if master.get("unregistered"):
                rec["note"] = self._unregistered_note(role, master)
        bias, dark = used.get(CALIB_TYPE_BIAS), used.get(CALIB_TYPE_DARK)
        if bias and dark:
            shared = self._bias_shared(bias, dark)
            if shared is None:
                rec["note"] = "; ".join(filter(None, [rec.get("note"), "the dark's bias is unknown, bit 64 not set"]))
            elif not shared:
                value |= ppflag.PPFLAG_BIAS_NOT_SHARED
        rec["new"] = value

    def _bias_shared(self, bias: dict, dark: dict) -> bool | None:
        """Whether the single's bias is the one inside its dark: by the dark's own bias edge or IMCID001/IMCMB001 card
        (read with the masters), else `ppflag.bias_shared_with_dark` on the two files."""
        inside = self.master_ingredients.get(dark.get("id"), {}).get(CALIB_TYPE_BIAS) if dark.get("id") else None
        if inside is not None:
            if inside.get("used_imageid") and bias.get("imageid"):
                return str(inside["used_imageid"]).strip() == str(bias["imageid"]).strip()
            return inside.get("image_name") == bias.get("image_name")
        for stem, imageid in dark.get("cards") or []:
            if stem.startswith("bias_"):
                if imageid and bias.get("imageid"):
                    return str(imageid).strip() == str(bias["imageid"]).strip()
                return stem.replace(".fits", "") == bias.get("image_name")
        key = (bias.get("image_path"), dark.get("image_path"))
        if not all(key):
            return None
        with self._lock:
            if key in self._shared_cache:
                return self._shared_cache[key]
        try:
            shared = ppflag.bias_shared_with_dark(*key)
        except Exception:
            shared = None
        with self._lock:
            self._shared_cache[key] = shared
        return shared

    def _evaluate_derived(self, rec: dict, edges: list[tuple]):
        role = ROLE_OF_PARENT[rec["image_type"]]
        parents = [src for r, src in edges if r == role]
        values = [self.new_by_id.get(p["id"], p["ppflag"]) for p in parents]
        if not parents:
            values, n = self._parent_values_from_cards(rec)
            rec["from_cards"], rec["n_inputs"] = True, n
        else:
            rec["n_inputs"] = len(parents)
        known = [v for v in values if v is not None]
        rec["n_known"] = len(known)
        old = rec["ppflag"]
        if not values:
            rec.update(new=old, note="no ingredient known; left as is")
        elif not known:
            rec.update(new=old, note="no ingredient PPFLAG known; left as is")
        else:
            rec["new"] = ppflag.propagate_ppflag(*known) | self._keep_manual(old)
            if len(known) < len(values):
                rec["partial"] = True
                rec["note"] = f"{len(values) - len(known)} ingredient(s) without a known PPFLAG; not written"

    def _parent_values_from_cards(self, rec: dict) -> tuple[list, int]:
        header = fits.getheader(rec["image_path"])
        if rec["image_type"] == "diff":
            pairs = [(header.get("SCIIMG"), header.get("SCIIID"))] if header.get("SCIIMG") else []
        else:
            pairs = [(header[k], header.get("IID" + k[3:])) for k in header if k.startswith("IMG") and k[3:].isdigit()]
        values = []
        for name, imageid in pairs:
            stem = os.path.basename(str(name)).replace(".fits", "")
            value = self.new_by_imageid.get(imageid) if imageid else None
            if value is None:
                value = self.new_by_name.get(stem)
            if value is None:
                rows = self._query(
                    "SELECT ppflag FROM image_qa WHERE image_name = %s"
                    " ORDER BY (image_path IS NOT NULL) DESC, id DESC LIMIT 1",
                    (stem,),
                )
                value = rows[0][0] if rows else None
            values.append(value)
        return values, len(pairs)

    # ---- writes
    def _write_disk(self, rec: dict) -> dict:
        """Write the new PPFLAG into the FITS, its sidecar and its companions; {file kind: outcome}, never raises."""
        value = int(rec["new"])
        add = {
            "PPFLAG": (value, PPFLAG_COMMENTS.get(rec["type"], PPFLAG_COMMENT))
        }  # a new card gets the type's comment
        keep = {"PPFLAG": value}  # an existing card keeps its comment
        outcome: dict = {}
        path = rec["image_path"]
        try:
            header = fits.getheader(path)
            rec["hdr_old"] = header.get("PPFLAG")
            if rec["hdr_old"] == value:
                outcome["fits"] = "unchanged"
            elif _needs_rewrite(header) and not self.args.allow_rewrite:
                outcome["fits"] = "needs rewrite"
                return outcome
            else:
                update_padded_header_smart(path, keep if "PPFLAG" in header else add)
                outcome["fits"] = "written"
            outcome["sidecar"] = update_padded_header_file(path, add)
            for companion in COMPANIONS[rec["type"]]:
                cpath = companion(path)
                kind = os.path.basename(cpath)[len(os.path.basename(path)) - 5 :].strip("_").replace(".fits", "")
                if not os.path.exists(cpath):
                    outcome[kind] = "missing"
                    continue
                cheader = fits.getheader(cpath)
                if cheader.get("PPFLAG") == value:
                    outcome[kind] = "unchanged"
                elif _needs_rewrite(cheader) and not self.args.allow_rewrite:
                    outcome[kind] = "needs rewrite"
                else:
                    update_padded_header_smart(cpath, keep if "PPFLAG" in cheader else add)
                    outcome[kind] = "written"
        except Exception as exc:
            outcome["error"] = repr(exc)
        return outcome

    def _write_db(self, recs: list[dict]) -> int:
        pairs = [(r["id"], int(r["new"])) for r in recs if r["ppflag"] != r["new"]]
        done = 0
        for start in range(0, len(pairs), 1000):
            batch = pairs[start : start + 1000]
            values = ",".join(["(%s::integer, %s::integer)"] * len(batch))
            try:
                self.image_qa.execute_query(
                    f"UPDATE image_qa AS q SET ppflag = v.flag FROM (VALUES {values}) AS v(id, flag) WHERE q.id = v.id",
                    [x for pair in batch for x in pair],
                )
                done += len(batch)
            except Exception as exc:
                ids = {i for i, _ in batch}
                for r in recs:
                    if r["id"] in ids:
                        r["db_error"] = repr(exc)
                print(f"    DB ERROR on a batch of {len(batch)} rows: {exc!r}", flush=True)
        return done

    # ---- phases
    def _classify(self, rec: dict):
        old, new = rec["ppflag"], rec.get("new")
        if new is None:
            rec["action"] = "skipped"
        elif rec.get("partial"):
            rec["action"] = "partial inputs"
        elif old is None:
            rec["action"] = "card added"
        elif old == new:
            rec["action"] = "unchanged"
        else:
            rec["action"] = "changed"
        if old is not None and new is not None and old != new:
            for bit, desc in BIT_NAMES.items():
                if new & bit and not old & bit:
                    self.bitdiff[rec["type"]][f"+{bit} {desc}"] += 1
                if old & bit and not new & bit:
                    self.bitdiff[rec["type"]][f"-{bit} {desc}"] += 1

    def _verify_header(self, rec: dict):
        """With --verify-headers: a FITS whose card disagrees with the DB value is written too."""
        try:
            rec["hdr_old"] = fits.getheader(rec["image_path"]).get("PPFLAG")
        except Exception as exc:
            rec["note"] = f"header unreadable: {exc!r}"
            return
        if rec["action"] == "unchanged" and rec["hdr_old"] != rec["new"]:
            rec["action"] = "header differs"

    def run_masters(self):
        """Evaluate every master, bias then dark then flat (binned copies after their source flat)."""
        st = time.time()
        rows = self.load_masters()
        write = self.args.apply and "master" in self.args.types and (self.args.write_masters or not self.selection)
        if self.args.apply and "master" in self.args.types and not write:
            print(
                "[master] a selection is given: masters are evaluated but NOT written (--write-masters to force)",
                flush=True,
            )
        n = 0
        for dtype in MASTER_TYPES:
            pending = [dict(r, type="master", note="") for r in rows if r["image_type"] == dtype]
            for _ in range(3):  # binned copies wait for their source flat, which may itself be a copy
                deferred = []
                for start in range(0, len(pending), self.args.chunk):
                    chunk = pending[start : start + self.args.chunk]
                    n += self._process_chunk("master", chunk, write=write, deferrable=True)
                    deferred.extend(r for r in chunk if r.get("deferred"))
                if not deferred:
                    break
                for r in deferred:
                    r.pop("deferred")
                pending = deferred
            for r in pending:
                if r.get("deferred"):
                    r.update(new=None, note="source flat never resolved", final=True)
                    self._process_chunk("master", [r], write=False)
        self._report("master", n, time.time() - st)

    def run_phase(self, phase: str):
        if phase == "master":
            return self.run_masters()
        st, n = time.time(), 0
        print(f"[{phase}] evaluating", flush=True)
        chunk: list[dict] = []
        quiet = contextlib.redirect_stdout(io.StringIO())  # see _process_chunk: NameHandler may print via cam_tracker
        for row in self._rows((phase,), limit=self.args.limit):
            with quiet:
                chunk.append(_fill_keys(dict(row, type=phase, note="")))
            if len(chunk) >= self.args.chunk:
                n += self._process_chunk(phase, chunk, write=self.args.apply)
                chunk = []
                print(f"[{phase}] {n} done, {time.time() - st:.0f} s", flush=True)
        if chunk:
            n += self._process_chunk(phase, chunk, write=self.args.apply)
        self._report(phase, n, time.time() - st)

    def _process_chunk(self, phase: str, chunk: list[dict], write: bool, deferrable: bool = False) -> int:
        edges = self._edges([r["id"] for r in chunk])
        evaluate = {"master": self._evaluate_master, "single": self._evaluate_single}.get(phase, self._evaluate_derived)

        def one(rec):
            if not rec.get("final"):
                try:
                    evaluate(rec, edges.get(rec["id"], []))
                except Exception as exc:
                    rec.update(new=None, note=f"error: {exc!r}")
            return rec

        # cam_tracker prints a line whenever a name without a camera token predates its serial table; keep it out
        # of the log (the main thread alone swaps stdout, so the redirect is thread-safe here)
        with ThreadPoolExecutor(max_workers=self.args.workers) as pool, contextlib.redirect_stdout(io.StringIO()):
            recs = [r for r in pool.map(one, chunk) if not (deferrable and r.get("deferred"))]
        for rec in recs:
            if rec.get("new") is not None and not rec.get("partial"):
                self.new_by_id[rec["id"]] = rec["new"]
                if rec.get("imageid"):
                    self.new_by_imageid[rec["imageid"]] = rec["new"]
                self.new_by_name[rec["image_name"]] = rec["new"]
            self._classify(rec)
        if self.args.verify_headers:
            with ThreadPoolExecutor(max_workers=self.args.workers) as pool:
                list(pool.map(self._verify_header, [r for r in recs if r.get("new") is not None]))
        if write:
            writable = [r for r in recs if r["action"] in ("card added", "changed", "header differs")]
            with ThreadPoolExecutor(max_workers=self.args.workers) as pool:
                outcomes = list(pool.map(self._write_disk, writable))
            clean = []
            for rec, outcome in zip(writable, outcomes):
                rec["disk"] = outcome
                if "error" not in outcome and "needs rewrite" not in outcome.values():
                    clean.append(rec)
            self.summary[phase]["image_qa rows updated"] += self._write_db(clean)
            for rec in clean:
                rec["db_updated"] = rec["ppflag"] != rec["new"] and not rec.get("db_error")
        for rec in recs:
            self.summary[phase][rec["action"]] += 1
            if rec.get("note"):
                note = rec["note"].split(";")[0].split(":")[0]
                self.summary[phase]["note: " + ("master not registered" if "not registered" in note else note)] += 1
            if rec.get("from_cards"):
                self.summary[phase]["evaluated from header cards (no edges)"] += 1
            if rec.get("wall_rule_differs"):
                self.summary[phase]["bit 16 differs under the other wall rule (--dtype-walls)"] += 1
            disk = rec.get("disk", {})
            if "error" in disk:
                self.summary[phase]["WRITE ERRORS"] += 1
                print(f"    WRITE ERROR {rec['image_path']}: {disk['error']}", flush=True)
            if "needs rewrite" in disk.values():
                self.summary[phase]["needs rewrite (skipped; --allow-rewrite)"] += 1
            for kind, out in disk.items():
                if kind != "error":
                    self.summary[phase][f"disk {kind} {out}"] += 1
            if self.args.verbose and rec["action"] != "unchanged":
                print(
                    f"    {rec['action']:14s} {rec['ppflag']}->{rec.get('new')} {rec['image_path']}"
                    f" {rec.get('masters', '')} {rec.get('note', '')}",
                    flush=True,
                )
            self._csv(rec)
        return len(recs)

    def _report(self, phase: str, n: int, elapsed: float):
        counts = self.summary[phase]
        print(
            f"[{phase}] {n} evaluated in {elapsed:.0f} s: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())),
            flush=True,
        )
        if self.bitdiff[phase]:
            print(
                f"[{phase}] bits changed on products that had a card: "
                + ", ".join(f"{k} {v}" for k, v in sorted(self.bitdiff[phase].items())),
                flush=True,
            )

    # ---- output
    CSV_FIELDS = (
        "type",
        "action",
        "id",
        "image_path",
        "ppflag",
        "hdr_old",
        "new",
        "masters",
        "n_inputs",
        "n_known",
        "disk",
        "db_updated",
        "db_error",
        "note",
    )

    def open_csv(self, path: str):
        self._csv_file = open(path, "w", newline="")
        self.csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.CSV_FIELDS, extrasaction="ignore")
        self.csv_writer.writeheader()

    def _csv(self, rec: dict):
        if self.csv_writer is not None and (self.args.csv_all or rec["action"] != "unchanged"):
            self.csv_writer.writerow({k: rec.get(k, "") for k in self.CSV_FIELDS})

    def close_csv(self):
        if self.csv_writer is not None:
            self._csv_file.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objects", nargs="+", help="image_qa.object values (default: every object)")
    ap.add_argument("--nightdate-from", help="YYYY-MM-DD, inclusive (masters are never filtered)")
    ap.add_argument("--nightdate-to", help="YYYY-MM-DD, inclusive")
    ap.add_argument("--units", nargs="+", help="7DTxx values")
    ap.add_argument(
        "--types",
        nargs="+",
        default=list(TYPES),
        choices=TYPES,
        help="which product types to walk (masters are always evaluated; chain order enforced)",
    )
    ap.add_argument(
        "--apply", action="store_true", help="write headers, sidecars, companions and image_qa (default: dry-run)"
    )
    ap.add_argument(
        "--write-masters", action="store_true", help="with a selection, also write the masters (all of them)"
    )
    ap.add_argument(
        "--allow-rewrite",
        action="store_true",
        help="add the card to a FITS whose header must grow a block (astropy rewrites the whole file)",
    )
    ap.add_argument("--workers", type=int, default=4, help="threads for file I/O (default 4)")
    ap.add_argument("--chunk", type=int, default=20000, help="rows per DB page (default 20000)")
    ap.add_argument("--limit", type=int, help="first N rows per non-master type (for a trial)")
    ap.add_argument(
        "--dtype-walls",
        action="store_true",
        help="bit 16 from changelog rows whose wall column names the master's dtype (default: the pipeline's dtype-blind walls)",
    )
    ap.add_argument(
        "--verify-headers",
        action="store_true",
        help="open every FITS: a card that disagrees with the DB value is written too (slow)",
    )
    ap.add_argument("--csv", help="per-image table of the non-unchanged rows (default ./ppflag_<dry|apply>.csv)")
    ap.add_argument("--csv-all", action="store_true", help="also list unchanged rows in the CSV")
    ap.add_argument("-v", "--verbose", action="store_true", help="one line per changed image")
    args = ap.parse_args()
    args.types = [t for t in TYPES if t in args.types]

    print(
        f"{'APPLY' if args.apply else 'DRY-RUN'} objects={args.objects or 'all'} nights={args.nightdate_from}.."
        f"{args.nightdate_to} units={args.units or 'all'} types={args.types} workers={args.workers}",
        flush=True,
    )
    if args.apply and (args.limit or args.types != list(TYPES)):
        print(
            "WARNING: --apply with --limit or a partial --types takes parent values from image_qa as it is now;"
            " see the docstring",
            flush=True,
        )
    backfill = Backfill(args)
    csv_path = args.csv or ("ppflag_apply.csv" if args.apply else "ppflag_dry.csv")
    backfill.open_csv(csv_path)
    try:
        backfill.run_masters()
        for phase in args.types:
            if phase != "master":
                backfill.run_phase(phase)
    finally:
        backfill.close_csv()
        print(f"CSV: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
