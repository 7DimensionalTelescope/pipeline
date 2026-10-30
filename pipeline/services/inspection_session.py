"""
Visual inspection in a notebook: one composite of a frame's check plots, one prompt per frame.
The pool holds processed singles; a raw frame in it is replaced by its single through PathHandler.

Enter passes, text rejects with that text as INSPCOMM (written at once), `?` marks the frame ambiguous
with the clause after it as the comment (kept in the session only; nothing is written), `u` reverses the previous verdict,
`f`/`b` move forward or back (`100f`, `3b`; a frame left without a verdict is recorded as seen), `q`
stops and `run()` continues later. Passes are session knowledge only: the record is the
session's table, a JSONL file when `log_path` is given, and the Markdown report of `report()`, written
on every stop when `report_path` is given.
"""

import base64
import io
import json
import os
import random
import re
import sys
import tempfile
import time
from datetime import datetime
from typing import List, Optional, Union

from astropy.io import fits

from .. import const
from ..const.sciproc import REJECTION_PROCESS_HEADER_KEY
from ..path import PathHandler
from ..utils import atleast_1d
from ..utils.filesystem import latest_unique_filename
from .inspection import INSPCOMM_COMMENT, INSPCOMM_KEY, inspect_images, revoke_inspection


def in_notebook() -> bool:
    """Jupyter kernel check."""
    try:
        from IPython import get_ipython

        return type(get_ipython()).__module__.startswith("ipykernel")
    except Exception:
        return False


def _settle(seconds: float = 0.3) -> None:
    sys.stdout.flush()
    time.sleep(seconds)


def _stem(image: str) -> str:
    return os.path.basename(image).replace(".fits", "")


def _fmt(value) -> str:
    if value is None:
        return "-"
    return f"{value:.3g}" if isinstance(value, float) else str(value)


def _cell(text) -> str:
    return str(text if text is not None else "").replace("|", "\\|").replace("\n", " ")


def figure_paths(image: str) -> dict:
    """{caption: path or None} of the seven check plots, through PathHandler's mkdir-free twins."""
    path = PathHandler(image)
    single_dir = path._single_dir[0]
    if not os.path.isdir(single_dir):
        raise FileNotFoundError(f"{single_dir}: no processed tree for {os.path.basename(image)}")
    factory = path.imcoadd.factory
    interp = factory.stage_images(image, "interp", "")
    candidates = {
        "WCS Check + 3x3 PSF": latest_unique_filename(path.astrometry.figures._evaluation_plot("wcs")),
        "Preprocessed Single Image": path.preprocess.figures._science_processed,
        "Raw Single Image": path.preprocess.figures._science_raw,
        "Zero Point (MAG_AUTO)": path.photometry.figures._zp("MAG_AUTO"),
        "Filter Check": path.photometry.figures._filter_check,
        "Source Mask": factory.source_mask_figures(interp)[0],
        "Background Model": factory.background_figures(interp)[0],
    }
    return {caption: p if p and os.path.exists(p) else None for caption, p in candidates.items()}


def _layout(height: int) -> dict:
    """{caption: (x, y, w, h)}: WCS full height at left, the preprocessed image large with source mask and background below it, raw image over zero point over filter check at the right."""
    cap = 34
    wcs_w = round(0.583 * height)
    col_w = round((height - 3 * cap) / (2 / 3 + 3 / 4 + 1 / 2))
    fov_w = height - 2 * cap
    raw_h = round(col_w * 2 / 3) + cap
    zp_h = round(col_w * 3 / 4) + cap
    fov_h = round(fov_w / 1.5) + cap
    x1, x2, half = wcs_w, wcs_w + fov_w, (height - 2 * cap) // 2
    return {
        "WCS Check + 3x3 PSF": (0, 0, wcs_w, height),
        "Preprocessed Single Image": (x1, 0, fov_w, fov_h),
        "Source Mask": (x1, fov_h, half, height - fov_h),
        "Background Model": (x1 + half, fov_h, fov_w - half, height - fov_h),
        "Raw Single Image": (x2, 0, col_w, raw_h),
        "Zero Point (MAG_AUTO)": (x2, raw_h, col_w, zp_h),
        "Filter Check": (x2, raw_h + zp_h, col_w, height - raw_h - zp_h),
    }


def composite(image: str, height: int = 1000):
    """PIL image of the seven check plots on a landscape canvas; a missing plot leaves an outlined, labeled gap."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default(size=22)
    paths = figure_paths(image)
    boxes = _layout(height)
    top = 34
    width = max(x + w for x, y, w, h in boxes.values())
    canvas = Image.new("RGB", (width, height + top), "white")
    draw = ImageDraw.Draw(canvas)
    centered = lambda text, x, w: x + (w - draw.textlength(text, font=font)) / 2
    draw.text((centered(_stem(image), 0, width), 6), _stem(image), fill="black", font=font)
    for caption, (x, y, w, h) in boxes.items():
        y += top
        draw.text((centered(caption, x, w), y + 2), caption, fill="black", font=font)
        if paths[caption] is None:
            draw.rectangle([x + 3, y + 30, x + w - 4, y + h - 4], outline="gray", width=2)
            draw.text((centered("missing", x, w), y + 30 + (h - 34 - 22) / 2), "missing", fill="gray", font=font)
            continue
        im = Image.open(paths[caption]).convert("RGB")
        im.thumbnail((w - 6, h - 34))
        canvas.paste(im, (x + (w - im.width) // 2, y + 30))
    return canvas


class InspectionSession:
    """Notebook loop over processed singles; rejections are written at once, passes are kept only here."""

    def __init__(
        self,
        images,
        log_path: Optional[str] = None,
        by: Optional[str] = None,
        use_database: bool = True,
        overwrite: bool = False,
        report_path: Optional[str] = None,
        shuffle: Union[bool, int] = False,
    ):
        images = [images] if isinstance(images, str) else list(images)
        self.images = [image for image in images if isinstance(image, str) and image]
        if self.images:
            handler = PathHandler(self.images)
            kinds, singles = atleast_1d(handler.name.type), atleast_1d(handler._processed_images)
            self.images = [
                single if kind.kind == const.NAME_TYPE_RAW else image
                for image, kind, single in zip(self.images, kinds, singles)
            ]
        if shuffle is not False:
            random.Random(0 if shuffle is True else shuffle).shuffle(self.images)
        self.log_path = log_path
        self.report_path = report_path
        self.by = by or os.environ.get("USER")
        self.use_database = use_database
        self.overwrite = overwrite
        self.started = datetime.now()
        self.records: List[dict] = []
        self.cursor = 0
        self._tmpdir = None
        if log_path and os.path.exists(log_path):
            with open(log_path) as f:
                self.records = [json.loads(line) for line in f if line.strip()]

    def _state(self, image: str) -> dict:
        header = fits.getheader(image)
        return {
            "sanity": header.get("SANITY"),
            "inspcomm": header.get(INSPCOMM_KEY),
            "rej_proc": header.get(REJECTION_PROCESS_HEADER_KEY),
            "seeing": header.get("SEEING"),
            "ellip": header.get("ELLIP"),
            "ppflag": header.get("PPFLAG"),
        }

    def _clear(self, kernel: bool, retain: bool) -> None:
        if kernel and not retain:
            from IPython.display import clear_output

            clear_output(wait=True)

    def _display(self, im, image: str, kernel: bool) -> None:
        if kernel:
            from IPython.display import Image, display

            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=82)
            display(Image(data=buf.getvalue()))
        else:
            if self._tmpdir is None:
                self._tmpdir = tempfile.mkdtemp(prefix="inspection_")
            out = os.path.join(self._tmpdir, f"{_stem(image)}.jpg")
            im.save(out, "JPEG", quality=82)
            print(out)

    def _write_log(self) -> None:
        if self.log_path:
            with open(self.log_path, "w") as f:
                f.writelines(json.dumps(record) + "\n" for record in self.records)

    def _record_of(self, image: str) -> Optional[dict]:
        return next((record for record in self.records if record["image"] == image), None)

    def _record(self, image: str, state: dict, answer: Optional[str], note: Optional[str] = None) -> dict:
        existing = self._record_of(image)
        if existing and existing["verdict"] == "reject":
            self.undo(image)
        elif existing:
            self.records.remove(existing)
        inspcomm = None
        if answer:
            fits.Card(INSPCOMM_KEY, answer, INSPCOMM_COMMENT)
            inspect_images(
                image,
                sanity=False,
                inspcomm=answer,
                by=self.by,
                overwrite=self.overwrite,
                use_database=self.use_database,
            )
            inspcomm = fits.getheader(image).get(INSPCOMM_KEY)
        record = {
            "index": self.cursor,
            "image": image,
            "name": _stem(image),
            "verdict": (
                "ambiguous" if note is not None else "seen" if answer is None else "reject" if answer else "pass"
            ),
            "inspcomm": inspcomm,
            "note": note or None,
            "prior_sanity": state["sanity"],
            "prior_inspcomm": state["inspcomm"],
            "prior_rej_proc": state["rej_proc"],
            "seeing": state["seeing"],
            "ellip": state["ellip"],
            "ppflag": state["ppflag"],
            "time": datetime.now().isoformat(timespec="seconds"),
            "by": self.by,
        }
        self.records.append(record)
        self._write_log()
        return record

    def run(self, retain: bool = False) -> "InspectionSession":
        """Enter pass, text reject, u undo, f/b move (100f, 3b), q quit; `retain` keeps every composite in the cell output."""
        kernel = in_notebook()
        if not kernel and not sys.stdin.isatty():
            raise RuntimeError("run() needs a Jupyter kernel or a terminal")
        try:
            self._loop(kernel, retain)
        finally:
            if self.report_path:
                self.report(self.report_path)
        return self

    def _loop(self, kernel: bool, retain: bool) -> None:
        shown, note, browsing = 0, "", False
        self.cursor = 0
        while 0 <= self.cursor < len(self.images):
            image = self.images[self.cursor]
            record = self._record_of(image)
            if record and record["verdict"] != "seen" and not browsing:
                self.cursor += 1
                continue
            browsing = False
            try:
                state = self._state(image)
                im = composite(image)
            except OSError as e:
                print(f"skipped {image}: {e}")
                self.cursor += 1
                continue
            shown += 1
            marker = f"[{record['verdict']}{': ' + record['note'] if record.get('note') else ''}] " if record else ""
            verdict = f"SANITY {state['sanity']}  INSPCOMM {state['inspcomm'] or '-'}"
            if state["rej_proc"]:
                verdict += f"  REJ_PROC {state['rej_proc']}"
            metrics = f"seeing {_fmt(state['seeing'])}  ellip {_fmt(state['ellip'])}  ppflag {_fmt(state['ppflag'])}"
            self._clear(kernel, retain)
            if note:
                print(note)
            print(f"[{self.cursor + 1}/{len(self.images)}] {marker}{_stem(image)}  {metrics}  {verdict}")
            self._display(im, image, kernel)
            _settle()
            try:
                answer = input("Enter=pass  text=reject  ?=ambiguous  u=undo  f/b=move  q=quit: ").strip()
            except (EOFError, KeyboardInterrupt, NotImplementedError):
                return
            note = ""
            if answer.lower() == "q":
                return
            if answer.lower() == "u":
                undone = self.undo()
                note = f"undone {undone['name']}" if undone else "nothing to undo"
                continue
            if answer.startswith("?"):
                self._record(image, state, None, note=answer[1:].strip())
                self.cursor += 1
                continue
            move = re.fullmatch(r"(\d*)([fb])", answer.lower())
            if move:
                if record is None:
                    self._record(image, state, None)
                step = int(move.group(1) or 1) * (1 if move.group(2) == "f" else -1)
                self.cursor = min(max(self.cursor + step, 0), len(self.images) - 1)
                browsing = True
                continue
            try:
                record = self._record(image, state, answer)
            except ValueError as e:
                note = f"not recorded: {e}"
                continue
            if answer and record["inspcomm"] != answer:
                note = f"kept the existing INSPCOMM: {record['inspcomm']}"
            self.cursor += 1
        if not shown:
            print("nothing left to inspect")

    def undo(self, image: Optional[str] = None) -> Optional[dict]:
        """Reverse the last verdict, or the one on `image`; a rejection goes back to the state recorded before it."""
        matches = [record for record in self.records if image is None or record["image"] == image]
        if not matches:
            return None
        record = matches[-1]
        if record["verdict"] == "reject":
            if record["prior_inspcomm"]:
                inspect_images(
                    record["image"],
                    sanity=bool(record["prior_sanity"]),
                    inspcomm=record["prior_inspcomm"],
                    by=self.by,
                    overwrite=True,
                    use_database=self.use_database,
                )
            else:
                revoke_inspection(record["image"], use_database=self.use_database, sanity=record["prior_sanity"])
        self.records.remove(record)
        if record["image"] in self.images:
            self.cursor = min(self.cursor, self.images.index(record["image"]))
        self._write_log()
        return record

    def show(self, image: str, retain: bool = True) -> None:
        """One composite, no prompt."""
        kernel = in_notebook()
        self._clear(kernel, retain)
        self._display(composite(image), image, kernel)

    def table(self):
        """Records as a DataFrame."""
        import pandas as pd

        return pd.DataFrame(self.records)

    def stats(self):
        """Counts per verdict and comment (the INSPCOMM written, or the clause after ? of an ambiguous verdict)."""
        table = self.table()
        if table.empty:
            return table
        table["comment"] = table.inspcomm.where(table.inspcomm.notna(), table.get("note"))
        return table.groupby(["verdict", "comment"], dropna=False).size().rename("n").reset_index()

    def report(self, path: str, images: str = "rejections", max_mb: float = 50) -> str:
        """Self-contained Markdown: counts, a count per INSPCOMM, 1400 px WebP composites for `images` in rejections (rejected and ambiguous) | all | none."""
        if images not in ("rejections", "all", "none"):
            raise ValueError(f"images must be rejections, all or none, not {images!r}")
        rejected = [record for record in self.records if record["verdict"] == "reject"]
        ambiguous = [record for record in self.records if record["verdict"] == "ambiguous"]
        passed = [record for record in self.records if record["verdict"] == "pass"]
        seen = [record for record in self.records if record["verdict"] == "seen"]
        n_embedded = {
            "rejections": len(rejected) + len(ambiguous),
            "all": len(rejected) + len(ambiguous) + len(passed),
            "none": 0,
        }[images]
        estimate = 0.12 * n_embedded + 0.0001 * len(self.records)
        if estimate > max_mb:
            raise ValueError(f"about {estimate:.2f} MB for {n_embedded} composites; images='none' or a smaller pool")
        comments = {}
        for record in rejected:
            comments[record["inspcomm"]] = comments.get(record["inspcomm"], 0) + 1
        lines = [
            f"# Inspection session, {self.started:%Y-%m-%d %H:%M}, inspector {self.by}",
            "",
            f"{len(passed) + len(rejected)} of {len(self.images)} frames judged: {len(passed)} pass, {len(rejected)} reject; "
            f"{len(ambiguous)} ambiguous; {len(self.images) - len(passed) - len(rejected) - len(ambiguous)} without a verdict, "
            f"{len(seen)} of them seen.",
            "",
            "| INSPCOMM | n |",
            "|---|---|",
            *[f"| {_cell(comment)} | {n} |" for comment, n in sorted(comments.items(), key=lambda item: -item[1])],
        ]
        for title, group in (("Rejected", rejected), ("Ambiguous", ambiguous), ("Passed", passed)):
            lines += ["", f"## {title}", ""]
            embed = images == "all" or (images == "rejections" and title != "Passed")
            if not embed:
                lines += ["| frame | name | comment | prior SANITY |", "|---|---|---|---|"]
                lines += [
                    f"| {record['index'] + 1}/{len(self.images)} | {record['name']} | {_cell(record['inspcomm'] or record.get('note'))} | {record['prior_sanity']} |"
                    for record in group
                ]
                continue
            for record in group:
                parts = [
                    _cell(record["inspcomm"] or record.get("note")),
                    f"seeing {_fmt(record['seeing'])}",
                    f"ellipticity {_fmt(record['ellip'])}",
                    f"ppflag {_fmt(record['ppflag'])}",
                    f"prior SANITY {record['prior_sanity']}",
                    f"REJ_PROC {record['prior_rej_proc']}" if record["prior_rej_proc"] else None,
                ]
                im = composite(record["image"])
                im.thumbnail((1400, 10000))
                buf = io.BytesIO()
                im.save(buf, "WEBP", quality=80)
                data = base64.b64encode(buf.getvalue()).decode()
                lines += [
                    f"### {record['index'] + 1}/{len(self.images)} {record['name']}: {record['verdict'].upper()}",
                    "",
                    "; ".join(part for part in parts if part),
                    "",
                    f"![{record['name']}](data:image/webp;base64,{data})",
                    "",
                ]
        if seen:
            lines += ["", "## Seen without a verdict", "", "| frame | name | SANITY |", "|---|---|---|"]
            lines += [
                f"| {record['index'] + 1}/{len(self.images)} | {record['name']} | {record['prior_sanity']} |"
                for record in seen
            ]
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        return path
