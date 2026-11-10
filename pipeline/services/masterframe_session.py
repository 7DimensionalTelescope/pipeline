"""
Visual inspection of master frames (bias, dark, flat), the same verdict loop as InspectionSession.

Targets are master paths, processed singles (expanded to the distinct masters they used) or raw
calibration frames (their master through PathHandler). A text reply rejects the master with that text
as INSPCOMM after showing what the PPFLAG bit-4 cascade would touch and asking once more; the cascade
writes stop at the singles, everything below them is reported only. `u` withdraws the last rejection
and recomputes bit 4, `r` redraws the current master with its raw frames, the other commands are
InspectionSession's. Raw thumbnails are cached under ~/.cache/pipeline_raw_thumbnails/<master stem>/
unless `cache_dir` is given.
"""

import base64
import json
import logging
import os
import random
import re
import signal
import threading
from datetime import datetime
from typing import List, Optional, Union

import numpy as np
from astropy.io import fits

from .. import const
from ..path import PathHandler
from ..path.name import NameHandler
from ..utils import atleast_1d
from .database.base import DatabaseError
from .inspection import INSPCOMM_COMMENT, INSPCOMM_KEY, inspect_images
from .inspection_session import (
    InspectionSession,
    _cell,
    _comment,
    _confirm_short,
    _pending_mark,
    _fmt,
    _hl,
    _inspcomm_table,
    _judged,
    _settle,
    _stem,
    _write_atomic,
    in_notebook,
)


def _imcmb(image: str) -> tuple:
    """([(role, master path)], [raw names]) from the IMCMB cards."""
    names = [str(v).strip() for k, v in fits.getheader(image).items() if k.startswith("IMCMB")]
    types = atleast_1d(NameHandler(names).type) if names else []
    masters = [(t.exposure_type, n) for n, t in zip(names, types) if t.kind == const.NAME_TYPE_MASTER]
    raws = [n for n, t in zip(names, types) if t.kind == const.NAME_TYPE_RAW]
    paths = atleast_1d(PathHandler([n for _, n in masters]).preprocess._masterframe) if masters else []
    return [(role, path) for (role, _), path in zip(masters, paths)], raws


def _calibration_sources(images: List[str]) -> dict:
    """{image: {role: master path}} for bias/dark/flat: image_qa_dependency edges, the IMCMB cards otherwise."""
    from .ppflag_cascade import _source_edges

    found = {image: {} for image in images}
    for image, edges in (_source_edges(images) if images else {}).items():
        for role, source in edges:
            found[image].setdefault(role, source)
    for image in [image for image, roles in found.items() if not roles]:
        for role, path in _imcmb(image)[0]:
            found[image].setdefault(role, path)
    return found


def _masters_of(images: List[str]) -> dict:
    """{image: {bias, dark, flat, flatdark when the flat's own dark differs}}."""
    used = _calibration_sources(list(dict.fromkeys(images)))
    flat_sources = _calibration_sources(list({roles["flat"] for roles in used.values() if "flat" in roles}))
    for roles in used.values():
        flatdark = flat_sources.get(roles.get("flat"), {}).get(const.CALIB_TYPE_DARK)
        if flatdark and flatdark != roles.get(const.CALIB_TYPE_DARK):
            roles["flatdark"] = flatdark
    return used


def masters_of_single(image: str) -> dict:
    """{bias, dark, flat, flatdark?, bpmask?} paths of a processed single."""
    masters = dict(_masters_of([image])[image])
    if const.CALIB_TYPE_DARK in masters:
        masters["bpmask"] = PathHandler.get_bpmask(masters[const.CALIB_TYPE_DARK])
    return masters


def master_ingredients(master: str) -> dict:
    """{masters: [paths], raws: [(original filename, raw DB path or None)]} from the master's IMCMB cards."""
    from .database.const import TABLES
    from .database.query import free_query

    masters, raws = _imcmb(master)
    table = TABLES[NameHandler(master).type.exposure_type]
    query = f"SELECT original_filename, file_path FROM {table} WHERE original_filename = ANY(%s)"
    found = dict(free_query(query, [raws])) if raws else {}
    return {"masters": [path for _, path in masters], "raws": [(name, found.get(name)) for name in raws]}


def master_record(master: str) -> dict:
    """image_qa row of a master (header fallback) + qa: [(key, shown, rule, passed)] from the Checker's criteria."""
    from .checker import Checker
    from .database.query import free_query

    fields = (
        "image_name image_type sanity inspcomm ppflag nightdate unit exptime "
        "clipmed clipstd clipmin clipmax sigmean edgevar shifted"
    ).split()
    rows = free_query(
        f"SELECT {', '.join(fields)} FROM image_qa WHERE image_path = %s OR image_name = %s"
        " ORDER BY (image_path = %s) DESC, id LIMIT 1",
        [master, _stem(master), master],
    )
    if rows:
        record = dict(zip(fields, rows[0]))
    else:
        header = fits.getheader(master)
        record = {field: header.get(field.upper()) for field in fields}
        record.update(image_name=_stem(master), image_type=NameHandler(master).type.exposure_type)
    checker = Checker()
    checker.logger = logging.getLogger(f"{__name__}.quiet")  # a silent logger, not a swapped stdout
    checker.logger.propagate = False
    criteria = checker.load_qa_criteria(const.IMAGE_GROUP_MASTERFRAME).get(str(record["image_type"]).upper(), {})
    symbols = {"lte": "<=", "lt": "<", "gte": ">=", "gt": ">", "eq": "", "neq": "!="}
    record["qa"] = []
    for key, rule in criteria.items():
        limit = rule["value"]
        text = (
            f"{limit[0]}-{limit[1]}"
            if rule["criteria"] == "within"
            else symbols[rule["criteria"]]
            + ({True: "T", False: "F"}.get(limit) if isinstance(limit, bool) else str(limit))
        )
        value = record.get(key.lower())
        if value is None:
            record["qa"].append((key, "-", text, None))
            continue
        shown = ("T" if value else "F") if isinstance(value, bool) else f"{value:.4g}"
        passed = checker.apply_qa_criteria(header={key: value}, dtype=record["image_type"])
        record["qa"].append((key, shown, text, passed))
    return record


def master_figure_paths(master: str) -> dict:
    """{image (stretched), image, histogram, bad-pixel mask (dark)}: existing figure paths or None."""
    typ = NameHandler(master).type.exposure_type
    figures = PathHandler(master).preprocess.figures
    paths = {
        "image (stretched)": getattr(figures, f"_{typ}_contrast", None),
        "image": getattr(figures, f"_{typ}_image"),
        "histogram": getattr(figures, f"_{typ}_hist"),
    }
    if typ == const.CALIB_TYPE_DARK:
        paths["bad-pixel mask"] = PathHandler(PathHandler.get_bpmask(master)).preprocess.figures._bpmask_image
    return {caption: path if path and os.path.exists(path) else None for caption, path in paths.items()}


def _two_lines(draw, text: str, x: float, y: float, w: int, font, fill) -> None:
    parts = text.split("_")
    cut = next(
        (k for k in range(1, len(parts)) if draw.textlength("_".join(parts[: k + 1]), font=font) > w - 8), len(parts)
    )
    for k, line in enumerate(("_".join(parts[:cut]), "_".join(parts[cut:]))):
        if line:
            draw.text((x + (w - draw.textlength(line, font=font)) / 2, y + 18 * k), line, fill=fill, font=font)


def _status_lines(record: dict, font, width: int) -> List[str]:
    """`SANITY T   PPFLAG 0000001`, then the QA items wrapped to `width`."""
    sanity = {True: "T", False: "F"}.get(record["sanity"], "-")
    ppflag = "-" if record["ppflag"] is None else f"{int(record['ppflag']):09b}"
    lines, line = [f"SANITY {sanity}   PPFLAG {ppflag}"], ""
    for key, shown, rule, _ in record["qa"]:
        item = f"{key} {shown} ({rule})"
        if line and font.getlength(f"{line}  {item}") > width:
            lines.append(line)
            line = item
        else:
            line = f"{line}  {item}" if line else item
    return lines + ([line] if line else [])


def _master_columns(master: str, label: str, rows: int) -> List[dict]:
    """Image column (+ histogram column when `rows` is 1) of one master: label, stem, record, cells."""
    record, figures = master_record(master), master_figure_paths(master)
    main = (
        ("image (stretched)", figures["image (stretched)"])
        if figures["image (stretched)"]
        else ("image", figures["image"])
    )
    histogram = ("histogram", figures["histogram"])
    column = {"label": label, "stem": _stem(master), "record": record, "cells": [main, histogram][:rows]}
    return [column] if rows == 2 else [column, {"label": None, "stem": None, "record": None, "cells": [histogram]}]


def _panel(title: str, columns: List[dict], height: int, min_width: int = 0, bottom: int = 0):
    """Canvas with the title, the column headers and the figure rows; returns (canvas, x0, y below the rows)."""
    from PIL import Image, ImageDraw, ImageFont

    font, small = ImageFont.load_default(size=22), ImageFont.load_default(size=16)
    col_w, top = 396, 40
    status = [_status_lines(c["record"], small, col_w - 8) if c["record"] else [] for c in columns]
    head_h = 70 + 17 * max(len(lines) for lines in status) + 8
    row_h = []
    for r in range(max(len(c["cells"]) for c in columns)):
        sizes = [Image.open(c["cells"][r][1]).size for c in columns if r < len(c["cells"]) and c["cells"][r][1]]
        row_h.append(min(max((round((col_w - 6) * h / w) for w, h in sizes), default=200), height) + 24)
    width = max(col_w * len(columns), min_width)
    canvas = Image.new("RGB", (width, top + head_h + sum(row_h) + bottom), "white")
    draw = ImageDraw.Draw(canvas)
    centered = lambda text, x, w, f: x + (w - draw.textlength(text, font=f)) / 2
    draw.text((centered(title, 0, width, font), 8), title, fill="black", font=font)
    x0 = (width - col_w * len(columns)) // 2
    for i, (column, lines) in enumerate(zip(columns, status)):
        x, y = x0 + i * col_w, top
        if column["label"]:
            draw.text((centered(column["label"], x, col_w, font), y), column["label"], fill="black", font=font)
        if column["stem"]:
            _two_lines(draw, column["stem"], x, y + 26, col_w, small, "dimgray")
        failed = column["record"] and any(passed is False for *_, passed in column["record"]["qa"])
        for k, line in enumerate(lines):
            fill = (
                ("red" if column["record"]["sanity"] is False else "black")
                if k == 0
                else "red" if failed else "dimgray"
            )
            draw.text((centered(line, x, col_w, small), y + 68 + 17 * k), line, fill=fill, font=small)
        y += head_h
        for (caption, path), h in zip(column["cells"], row_h):
            if caption:
                draw.text((centered(caption, x, col_w, small), y + 2), caption, fill="black", font=small)
            if caption and path is None:
                draw.rectangle([x + 3, y + 22, x + col_w - 4, y + h - 4], outline="lightgray", width=1)
                draw.text((centered("missing", x, col_w, small), y + h / 2), "missing", fill="gray", font=small)
            elif path:
                im = Image.open(path).convert("RGB")
                im.thumbnail((col_w - 6, h - 26))
                canvas.paste(im, (x + (col_w - im.width) // 2, y + 22))
            y += h
    return canvas, x0, top + head_h + sum(row_h)


def composite_for_single(image: str, height: int = 320):
    """PIL image of the masters a single used: bias | dark | bpmask | flat | flatdark, image over histogram."""
    masters = masters_of_single(image)
    columns = []
    for role in ("bias", "dark", "bpmask", "flat", "flatdark"):
        if role == "bpmask" and const.CALIB_TYPE_DARK in masters:
            mask = master_figure_paths(masters[const.CALIB_TYPE_DARK])["bad-pixel mask"]
            stem = _stem(masters["bpmask"])
            columns.append(
                {"label": role, "stem": stem, "record": None, "cells": [("bad-pixel mask", mask), (None, None)]}
            )
        elif role != "bpmask" and role in masters:
            columns += _master_columns(masters[role], role, rows=2)
    if not columns:
        raise ValueError(f"{os.path.basename(image)}: no master frames found")
    ppflag = fits.getheader(image).get("PPFLAG")
    title = f"{_stem(image)}   PPFLAG {'-' if ppflag is None else format(int(ppflag), '09b')}"
    return _panel(title, columns, height)[0]


def _raw_thumbnail(name: str, path: str, cache: str) -> tuple:
    """(jpg, median, std) of a raw frame, rendered once into `cache`."""
    jpg = os.path.join(cache, f"{_stem(name)}.jpg")
    stats = os.path.join(cache, f"{_stem(name)}.json")
    if not (os.path.exists(jpg) and os.path.exists(stats)):
        handler = signal.getsignal(signal.SIGINT)
        from ..preprocess.plotting import save_fits_as_figures

        signal.signal(signal.SIGINT, handler)  # importing preprocess imports services.queue, which ignores SIGINT

        data = fits.getdata(path)
        sub = data[::8, ::8].astype(np.float32)
        save_fits_as_figures(data, jpg, stretch=True, max_width=660, overwrite=True)
        with open(stats, "w") as f:
            json.dump({"median": float(np.median(sub)), "std": float(np.std(sub))}, f)
    with open(stats) as f:
        values = json.load(f)
    return jpg, values["median"], values["std"]


def composite_for_master(master: str, raws: bool = False, cache_dir: Optional[str] = None):
    """PIL image of a master with its bpmask and ingredient masters; with `raws`, its raw frames below (cached)."""
    from PIL import Image, ImageDraw, ImageFont

    typ = NameHandler(master).type.exposure_type
    ingredients = master_ingredients(master)
    columns = _master_columns(master, typ, rows=1)
    if typ == const.CALIB_TYPE_DARK:
        mask = master_figure_paths(master)["bad-pixel mask"]
        bpmask = PathHandler.get_bpmask(master)
        columns.append({"label": "bpmask", "stem": _stem(bpmask), "record": None, "cells": [("bad-pixel mask", mask)]})
    for path in ingredients["masters"]:
        role = NameHandler(path).type.exposure_type
        label = "flatdark" if typ == const.CALIB_TYPE_FLAT and role == const.CALIB_TYPE_DARK else role
        columns += _master_columns(path, label, rows=1)
    nframes = fits.getheader(master).get("NFRAMES")
    names = ", ".join(_stem(path) for path in ingredients["masters"])
    title = f"{_stem(master)}   NFRAMES {nframes}   ingredients: {names + ' + ' if names else ''}"
    title += f"{len(ingredients['raws'])} raw {typ} frames"
    col_w = 396
    raw_h = round((col_w - 6) * 2 / 3) + 60
    n = len(ingredients["raws"]) if raws else 0
    bottom = 34 + -(-n // 5) * raw_h if n else 0
    canvas, x0, y0 = _panel(title, columns, 320, min_width=5 * col_w if n else 0, bottom=bottom)
    if not n:
        return canvas
    cache = cache_dir or os.path.expanduser(os.path.join("~/.cache/pipeline_raw_thumbnails", _stem(master)))
    os.makedirs(cache, exist_ok=True)
    draw = ImageDraw.Draw(canvas)
    font, small = ImageFont.load_default(size=22), ImageFont.load_default(size=16)
    centered = lambda text, x, w, f: x + (w - draw.textlength(text, font=f)) / 2
    label = f"raw frames of this {typ} ({n}), ZScale stretched on the fly"
    draw.text((centered(label, 0, canvas.width, font), y0 + 6), label, fill="black", font=font)
    grid_x = (canvas.width - 5 * col_w) // 2
    for j, (name, path) in enumerate(ingredients["raws"]):
        row, col = divmod(j, 5)
        x = grid_x + col * col_w + (5 - min(5, n - 5 * row)) * col_w // 2
        y = y0 + 34 + row * raw_h
        _two_lines(draw, _stem(name), x, y, col_w, small, "dimgray")
        if path is None:
            draw.text(
                (centered("not in the raw DB", x, col_w, small), y + raw_h / 2),
                "not in the raw DB",
                fill="gray",
                font=small,
            )
            continue
        jpg, median, std = _raw_thumbnail(name, path, cache)
        im = Image.open(jpg).convert("RGB")
        im.thumbnail((col_w - 6, raw_h - 60))
        canvas.paste(im, (x + (col_w - im.width) // 2, y + 38))
        stat = f"median {median:.1f}   std {std:.1f}"
        draw.text((centered(stat, x, col_w, small), y + 40 + im.height), stat, fill="black", font=small)
    return canvas


class MasterframeSession(InspectionSession):
    """Notebook loop over master frames; a rejection is written with its PPFLAG bit-4 cascade, passes are kept only here."""

    def __init__(
        self,
        targets=None,
        stem: Optional[str] = None,
        log_path: Optional[str] = None,
        by: Optional[str] = None,
        use_database: bool = True,
        overwrite: bool = False,
        report_path: Optional[str] = None,
        shuffle: Union[bool, int] = False,
        log_lines: int = 5,
        raws: bool = False,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(
            None if targets is None else [], stem, log_path, by, use_database, overwrite, report_path, False, log_lines
        )
        self.raws, self.cache_dir = raws, cache_dir
        self.used_by, self._closures = {}, {}
        if targets is None:  # resumed from the log alone
            return
        targets = [targets] if isinstance(targets, (str, dict)) else list(targets)
        paths = [t["path"] if isinstance(t, dict) else t for t in targets]
        paths = [p for p in paths if isinstance(p, str) and p]
        types = atleast_1d(NameHandler(paths).type) if paths else []
        singles = [p for p, t in zip(paths, types) if t.kind == const.NAME_TYPE_CALIBRATED and os.path.exists(p)]
        used = _masters_of(singles) if singles else {}
        raw_masters = [p for p, t in zip(paths, types) if t.kind == const.NAME_TYPE_RAW]
        raw_masters = (
            dict(zip(raw_masters, atleast_1d(PathHandler(raw_masters).preprocess._masterframe))) if raw_masters else {}
        )
        for path, t in zip(paths, types):
            if t.kind == const.NAME_TYPE_MASTER and t.exposure_type in const.CALIB_TYPES:
                expanded = [path]
            elif t.kind == const.NAME_TYPE_RAW and t.exposure_type in const.CALIB_TYPES:
                expanded = [raw_masters[path]]
            elif t.kind == const.NAME_TYPE_CALIBRATED and path in used:
                expanded = list(used[path].values())
                for master in expanded:
                    self.used_by[master] = self.used_by.get(master, 0) + 1
            else:
                print(f"skipped {path}: not a master, a processed single or a raw calibration frame")
                continue
            self.images += [master for master in expanded if master not in self.images]
        if shuffle is not False:
            random.Random(0 if shuffle is True else shuffle).shuffle(self.images)

    def _state(self, master: str) -> dict:
        header = fits.getheader(master)
        return {"sanity": header.get("SANITY"), "inspcomm": header.get(INSPCOMM_KEY), "ppflag": header.get("PPFLAG")}

    def _closure(self, master: str):
        from .inspection import master_closure

        if master not in self._closures:
            self._closures[master] = master_closure([master])
        return self._closures[master]

    def _cascade_counts(self, master: str, plan) -> dict:
        """Derived masters and singles gaining bit 4 in `plan`; coadds, whites, diffs, configs below them."""
        closure = self._closure(master)
        derived = {path for _, path, _ in closure.derived_masters}
        gaining = [path for path, old, new in plan.planned if new & 4 and not (old or 0) & 4]
        return {
            "derived masters": sum(path in derived for path in gaining),
            "singles": sum(path not in derived for path in gaining),
            "coadds": len(closure.products.get(const.IMAGE_TYPE_COADD, [])),
            "whites": len(closure.products.get(const.IMAGE_TYPE_WHITE, [])),
            "diffs": len(closure.products.get(const.IMAGE_TYPE_DIFF, [])),
            "configs": len(closure.configs),
        }

    def _record(
        self,
        master: str,
        state: dict,
        answer: Optional[str],
        note: Optional[str] = None,
        cascade: Optional[dict] = None,
    ) -> dict:
        existing = self._record_of(master)
        if existing and existing["verdict"] == "reject":
            self.undo(master)
            state = self._state(master)  # the header as restored, not as rejected
        fields = {}
        if answer:
            try:
                fields = self._write_verdict(master, answer)
            except (PermissionError, DatabaseError) as e:  # the choice is kept here until flush()
                fields = {"inspcomm": answer, "pending": str(e)}
        if existing in self.records:  # replaced only once the write went through
            self.records.remove(existing)
        record = {
            "index": self.images.index(master) if master in self.images else self.cursor,
            "image": master,
            "name": _stem(master),
            "type": NameHandler(master).type.exposure_type,
            "verdict": (
                "ambiguous" if note is not None else "seen" if answer is None else "reject" if answer else "pass"
            ),
            "inspcomm": None,
            "pending": None,
            "note": note or None,
            "prior_sanity": state["sanity"],
            "prior_inspcomm": state["inspcomm"],
            "ppflag": state["ppflag"],
            "cascade": cascade if answer else None,
            "changed": None,
            "used_by": self.used_by.get(master),
            "time": datetime.now().isoformat(timespec="seconds"),
            "by": self.by,
        }
        record.update(fields)
        self.records.append(record)
        self._write_log()
        return record

    def _write_verdict(self, master: str, text: str) -> dict:
        """The rejection's writes and its cascade; the record fields they settle."""
        from .inspection import cascade_master

        inspect_images(
            master, sanity=False, inspcomm=text, by=self.by, overwrite=self.overwrite, use_database=self.use_database
        )
        changed = len(cascade_master(master, use_database=self.use_database).planned)
        self._webp_cache.clear()  # the status lines of every master pane may change
        return {"inspcomm": fits.getheader(master).get(INSPCOMM_KEY), "changed": changed}

    def _frame_line(self, master: str, state: dict, record: Optional[dict], total: int) -> str:
        qa = "  ".join(f"{key} {shown}" for key, shown, _, _ in master_record(master)["qa"])
        line = (
            f"[{self.cursor + 1}/{total}] {_hl(_judged(record))} {NameHandler(master).type.exposure_type} {_stem(master)}"
            f"  {_hl('SANITY ' + str(state['sanity']))}  {_hl('INSPCOMM ' + (state['inspcomm'] or '-'))}"
            f"  PPFLAG {_fmt(state['ppflag'])}  {qa}"
        )
        if self.use_database:
            closure = self._closure(master)
            counts = [
                f"{len(closure.products.get(kind, []))} {kind}s"
                for kind in (const.IMAGE_TYPE_COADD, const.IMAGE_TYPE_WHITE, const.IMAGE_TYPE_DIFF)
            ]
            line += f"  used by: {len(closure.singles)} singles, {', '.join(counts)} (depth<=10)"
        return line

    def _loop(self, kernel: bool, retain: bool, active: List[str], revise: bool) -> None:
        shown, note, browsing, with_raws = 0, "", False, None
        self._active = active
        self.cursor = 0 if revise else self._resume_position()
        if not revise and active and self.cursor >= len(active):  # everything judged: the last master, open for b
            self.cursor, browsing = len(active) - 1, True
        while 0 <= self.cursor < len(active):
            master = active[self.cursor]
            record = self._record_of(master)
            if record and record["verdict"] != "seen" and not browsing and not revise:
                self.cursor += 1
                continue
            browsing = False
            try:
                state = self._state(master)
                im = composite_for_master(master, raws=self.raws or with_raws == master, cache_dir=self.cache_dir)
                line = self._frame_line(master, state, record, len(active))
            except (OSError, ValueError) as e:
                print(f"skipped {master}: {e}")
                self.cursor += 1
                continue
            with_raws = None
            shown += 1
            self._clear(kernel, retain)
            if note:
                print(note)
            print(line)
            self._display(im, master, kernel)
            _settle()
            try:
                answer = input(
                    "Enter=pass  text=reject+cascade  ?=ambiguous  u=undo  f/b=move  r=raws  q=quit: "
                ).strip()
            except (EOFError, KeyboardInterrupt, NotImplementedError):
                if record is None:  # Ctrl-C quits like q
                    self._record(master, state, None)
                return
            note = ""
            if answer.lower() == "q":
                if record is None:
                    self._record(master, state, None)
                return
            if answer.lower() == "u":
                try:
                    if revise or (record and record["verdict"] != "seen"):  # a judged master reached by f/b: clear it
                        undone = self.undo(master)
                        if undone:
                            self._record(master, self._state(master), None)
                        note = f"cleared {undone['name']}" if undone else "nothing to clear"
                        if revise:
                            self.cursor += 1
                    else:
                        undone = self.undo()
                        note = f"undone {undone['name']}" if undone else "nothing to undo"
                except (PermissionError, DatabaseError) as e:
                    note, browsing = f"not undone: {e}", True
                continue
            if answer.lower() == "r":
                with_raws, browsing = master, True
                continue
            if answer.startswith("?"):
                self._record(master, state, None, note=answer[1:].strip())
                self.cursor += 1
                continue
            move = re.fullmatch(r"(\d*)([fb])", answer.lower())
            if move:
                if record is None:
                    self._record(master, state, None)
                step = int(move.group(1) or 1) * (1 if move.group(2) == "f" else -1)
                self.cursor = min(max(self.cursor + step, 0), len(active) - 1)
                browsing = True
                continue
            if not answer and record and record["verdict"] != "seen":
                self.cursor += 1
                continue
            cascade = None
            if answer:
                from .inspection import cascade_master

                if not _confirm_short(answer):
                    note, browsing = f"not recorded: {answer!r}", True
                    continue
                try:
                    fits.Card(INSPCOMM_KEY, answer, INSPCOMM_COMMENT)
                except ValueError as e:
                    note, browsing = f"not recorded: {e}", True
                    continue
                cascade = self._cascade_counts(
                    master, cascade_master(master, dry_run=True, use_database=self.use_database, sanity=False)
                )
                print(
                    f"cascade: {cascade['derived masters']} derived masters, {cascade['singles']} singles would gain"
                    f" PPFLAG bit 4; below them: {cascade['coadds']} coadds, {cascade['whites']} whites,"
                    f" {cascade['diffs']} diffs, {cascade['configs']} configs"
                )
                _settle()
                try:
                    confirmed = input("y to write, anything else cancels: ").strip().lower() == "y"
                except (EOFError, KeyboardInterrupt, NotImplementedError):
                    confirmed = False
                if not confirmed:
                    note, browsing = f"cancelled: {answer!r} not written", True
                    continue
            try:
                record = self._record(master, state, answer, cascade=cascade)
            except (ValueError, PermissionError, DatabaseError) as e:
                note, browsing = f"not recorded: {e}", True
                continue
            if record.get("pending"):  # shown again with the note; Enter moves on
                note, browsing = f"kept in the session only, not written: {record['pending']}", True
                continue
            if answer and record["inspcomm"] != answer:
                note = f"kept the existing INSPCOMM: {record['inspcomm']}"
            self._closures.pop(master, None)
            self.cursor += 1
        if not shown:
            print("nothing matches" if revise else "nothing left to inspect")

    def undo(self, image: Optional[str] = None) -> Optional[dict]:
        """Reverse the last verdict, or the one on `image`; a rejection is revoked (or its prior INSPCOMM restored) and bit 4 recomputed."""
        matches = [record for record in self.records if image is None or record["image"] == image]
        if not matches:
            return None
        record = matches[-1]
        if record["verdict"] == "reject" and not record.get("pending"):
            from .inspection import cascade_master, revoke_master

            if record["prior_inspcomm"]:
                inspect_images(
                    record["image"],
                    sanity=bool(record["prior_sanity"]),
                    inspcomm=record["prior_inspcomm"],
                    by=self.by,
                    overwrite=True,
                    use_database=self.use_database,
                )
                cascade_master(record["image"], use_database=self.use_database)
            else:
                revoke_master(record["image"], use_database=self.use_database)
            self._closures.pop(record["image"], None)
            self._webp_cache.clear()
        self.records.remove(record)
        active = getattr(self, "_active", None) or self.images
        if record["image"] in active:
            self.cursor = min(self.cursor, active.index(record["image"]))
        self._write_log()
        return record

    def show(self, master: str, raws: Optional[bool] = None, retain: bool = True) -> None:
        """One master composite, no prompt."""
        kernel = in_notebook()
        self._clear(kernel, retain)
        im = composite_for_master(master, raws=self.raws if raws is None else raws, cache_dir=self.cache_dir)
        self._display(im, master, kernel)

    def report(self, path: Optional[str] = None, images: str = "rejections", max_mb: float = 50) -> str:
        """Self-contained Markdown at `path` (default `report_path`): counts, INSPCOMMs, cascades, 1400 px WebP composites for `images` in rejections | all | none."""
        path = path or self.report_path
        if not path:
            raise ValueError("no report path: pass one or construct the session with stem or report_path")
        if images not in ("rejections", "all", "none"):
            raise ValueError(f"images must be rejections, all or none, not {images!r}")
        if path == self.report_path:
            self._report_images = images
        records = list(self.records)
        groups = {v: [r for r in records if r["verdict"] == v] for v in ("reject", "ambiguous", "pass", "seen")}
        rejected, ambiguous, passed, seen = groups.values()
        n_embedded = {"rejections": len(rejected) + len(ambiguous), "all": len(records) - len(seen), "none": 0}[images]
        estimate = 0.12 * n_embedded + 0.0001 * len(records)
        if estimate > max_mb:
            raise ValueError(f"about {estimate:.2f} MB for {n_embedded} composites; images='none' or a smaller pool")
        pending = [record for record in rejected if record.get("pending")]
        comments = {}
        for record in rejected:
            if not record.get("pending"):
                comments[record["inspcomm"]] = comments.get(record["inspcomm"], 0) + 1
        n = len(self.images)
        lines = [
            f"# Master-frame inspection session, {self.started:%Y-%m-%d %H:%M}, inspector {self.by}",
            "",
            f"{len(passed) + len(rejected)} of {n} masters judged: {len(passed)} pass, {len(rejected)} reject"
            f"{f' ({len(pending)} not written)' if pending else ''}; "
            f"{len(ambiguous)} ambiguous; {n - len(passed) - len(rejected) - len(ambiguous)} without a verdict, "
            f"{len(seen)} of them seen.",
            "",
            *_inspcomm_table(comments, len(pending)),
        ]
        for title, group in (("Rejected", rejected), ("Ambiguous", ambiguous), ("Passed", passed)):
            lines += ["", f"## {title}", ""]
            if not group:
                lines.append("none")
                continue
            embed = images == "all" or (images == "rejections" and title != "Passed")
            if not embed:
                lines += ["| master | name | comment | prior SANITY |", "|---|---|---|---|"]
                lines += [
                    f"| {r['index'] + 1}/{n} | {r['name']} | {_cell(_comment(r))} | {r['prior_sanity']} |"
                    for r in group
                ]
                continue
            for record in group:
                cascade = record.get("cascade")
                parts = [
                    _cell(_comment(record)),
                    f"prior SANITY {record['prior_sanity']}",
                    f"ppflag {_fmt(record['ppflag'])}",
                    f"singles in the pool {record['used_by']}" if record.get("used_by") else None,
                    ("cascade " + ", ".join(f"{v} {k}" for k, v in cascade.items())) if cascade else None,
                    f"{record['changed']} PPFLAGs changed" if record.get("changed") is not None else None,
                ]
                data = self._webp(record["image"], lambda m: composite_for_master(m, cache_dir=self.cache_dir))
                lines += [
                    f"### {record['index'] + 1}/{n} {record['name']}: {record['verdict'].upper()}{_pending_mark(record)}",
                    "",
                    "; ".join(part for part in parts if part),
                    "",
                    f"![{record['name']}](data:image/webp;base64,{data})",
                    "",
                ]
        if seen:
            lines += ["", "## Seen without a verdict", "", "| master | name | SANITY |", "|---|---|---|"]
            lines += [f"| {r['index'] + 1}/{n} | {r['name']} | {r['prior_sanity']} |" for r in seen]
        written = _write_atomic(path, lines)
        if path == self.report_path and in_notebook() and threading.current_thread() is not self._report_thread:
            self._schedule_report()
        return written
