"""
Visual inspection in a notebook: one composite of a frame's check plots, one prompt per frame.
The pool holds processed singles; a raw frame in it is replaced by its single through PathHandler.

Enter passes (on a judged frame it only moves to the next waiting one), text rejects with that text as INSPCOMM (written at once; without database permission the choice is kept in the
session as pending until `flush()`), `?` marks the frame ambiguous
with the clause after it as the comment (kept in the session only; nothing is written), `u` reverses the previous verdict
(on a judged frame reached by `f`/`b` it clears that frame),
`f`/`b` move forward or back (`100f`, `3b`; a frame left without a verdict is recorded as seen), `q`
stops and `run()` continues at the frame last left, in the same order (`shuffle` is seeded). `m` shows the master
frames the single used below its composite until the next reply; a reply of one or two letters that is no command asks
for confirmation before it becomes an INSPCOMM.
`run(verdict=..., comment=...)` revises the recorded frames that match (case-insensitive regex on the
comment; `"^$"` is no comment): text or `?` replace the verdict, `u` clears it, Enter keeps it;
`judge(images, answer)` does the same without the pane. `stem` names the log and
report files, `<stem>.jsonl` and `<stem>.md` (`log_path` alone implies `<log stem>.md`). The log's first line is
the pool, so `InspectionSession(stem=...)` without images resumes from the log alone; once every frame is judged,
`run()` opens the last frame for `b`. The last `log_lines` lines of the frame's science config log are
printed under the frame line. Passes are session knowledge only: the record is the
session's table, the JSONL log, and the Markdown report of `report()`, rewritten after every verdict and stop: in a
background thread under a kernel (composites rendered once), so no frame waits for it, in the foreground elsewhere.
"""

import base64
import io
import json
import os
import random
import re
import sys
import tempfile
import threading
import time
from datetime import datetime
from typing import List, Optional, Union

from astropy.io import fits

from .. import const
from ..const.sciproc import REJECTION_PROCESS_HEADER_KEY
from ..path import PathHandler
from ..utils import atleast_1d, collapse
from ..utils.filesystem import latest_unique_filename
from .database.base import DatabaseError
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


def _hl(text) -> str:
    """Highlight: black on bright white, ANSI SGR 30;107."""
    return f"\x1b[30;107m {text} \x1b[0m"


def _judged(record: Optional[dict]) -> str:
    """Frame-line label: earlier verdict with time, `seen, no verdict`, or `new`."""
    if record is None:
        return "new"
    label = "seen, no verdict" if record["verdict"] == "seen" else record["verdict"]
    if record.get("pending"):
        label += " (pending)"
    if record.get("note"):
        label += f": {record['note']}"
    return f"{label} {record['time'][5:16].replace('T', ' ')}"


def _inspcomm_table(comments: dict, pending: int = 0) -> List[str]:
    """Report lines: INSPCOMM frame counts, or an explicit none; the count kept for flush()."""
    if not comments:
        lines = ["INSPCOMM written: none."]
    else:
        rows = [f"| {_cell(comment)} | {n} |" for comment, n in sorted(comments.items(), key=lambda item: -item[1])]
        lines = ["INSPCOMM written, with the number of frames:", "", "| INSPCOMM | frames |", "|---|---|", *rows]
    if pending:
        lines += ["", f"INSPCOMM pending, kept in the session until `flush()`: {pending} frame(s)."]
    return lines


def _pending_mark(record: dict) -> str:
    return " (pending, not written)" if record.get("pending") else ""


def _comment(record: dict) -> str:
    """The record's INSPCOMM or ambiguous note, marked when pending."""
    return (record["inspcomm"] or record.get("note") or "") + _pending_mark(record)


def _write_atomic(path: str, lines: List[str]) -> str:
    """Atomic whole-file write through a temporary twin."""
    tmp = f"{path}.{threading.get_ident()}.tmp"
    try:
        with open(tmp, "w") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def _log_tail(image: str, n: int) -> List[str]:
    """The last two [ERROR] lines (ProcessError.Kind tags) and the last `n` lines of the frame's science config log."""
    log = collapse(PathHandler(image)._sciproc_output_log, raise_error=True)
    if n <= 0 or not os.path.exists(log):
        return []
    with open(log, "rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(f.tell() - 65536, 0))
        lines = f.read().decode(errors="replace").splitlines()
    tail = lines[-n:]
    tagged = [line for line in lines if line.startswith("[ERROR]")][-2:]
    return [line for line in tagged if line not in tail] + tail


def _status_line(image: str) -> Optional[str]:
    """The config's process_status row: status, error code and warnings decoded to ProcessError.Kind."""
    from ..errors import names_from_code
    from .database.query import free_query

    yml = collapse(PathHandler(image)._sciproc_output_yml, raise_error=True)
    rows = free_query("SELECT status, errors, warnings FROM process_status WHERE config_file = %s", (yml,))
    if not rows:
        return None
    status, error, warnings = rows[0]
    parts = [f"process_status {status}"]
    if error:
        parts.append("error %d %s.%s" % (error, *names_from_code(error)))
    if warnings:
        parts.append("warnings " + ", ".join("%d %s.%s" % (code, *names_from_code(code)) for code in warnings))
    return "; ".join(parts)


def _confirm_short(answer: str) -> bool:
    """Confirmation of a 1-2 letter reply that is no command; any other reply passes."""
    if not re.fullmatch(r"[A-Za-z]{1,2}", answer):
        return True
    try:
        return input(f"reject with INSPCOMM '{answer}'? y to confirm, anything else cancels: ").strip().lower() == "y"
    except (EOFError, KeyboardInterrupt, NotImplementedError):
        return False


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
        images=None,
        name: Optional[str] = None,
        log_path: Optional[str] = None,
        by: Optional[str] = None,
        use_database: bool = True,
        overwrite: bool = False,
        report_path: Optional[str] = None,
        shuffle: Union[bool, int] = False,
        log_lines: int = 5,
    ):
        self.log_path = log_path or (f"{name}.jsonl" if name else None)
        self.report_path = report_path or (
            f"{name}.md" if name else os.path.splitext(self.log_path)[0] + ".md" if self.log_path else None
        )
        self.log_lines = log_lines
        self.by = by or os.environ.get("USER")
        self.use_database = use_database
        self.overwrite = overwrite
        self.started = datetime.now()
        self.records: List[dict] = []
        self.cursor = 0
        self._tmpdir = None
        self._webp_cache: dict = {}
        self._report_images = "rejections"
        self._report_lock, self._report_thread, self._report_dirty = threading.Lock(), None, False
        head = None
        if self.log_path and os.path.exists(self.log_path):
            with open(self.log_path) as f:
                lines = [json.loads(line) for line in f if line.strip()]
            head = lines.pop(0) if lines and "pool" in lines[0] else None  # no pool line before 2026-11-04
            self.records = lines
        if head:
            self.started = datetime.fromisoformat(head["started"])
        if images is None:
            if not head and not self.records:
                raise ValueError("no images and no log to resume from")
            self.images = head["pool"] if head else [r["image"] for r in sorted(self.records, key=lambda r: r["index"])]
            return
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
            head = {"pool": self.images, "started": self.started.isoformat(timespec="seconds"), "by": self.by}
            with open(self.log_path, "w") as f:
                f.write(json.dumps(head) + "\n")
                f.writelines(json.dumps(record) + "\n" for record in self.records)
        self._schedule_report()

    def _schedule_report(self) -> None:
        """Report rewrite: a daemon worker under a kernel, the foreground elsewhere."""
        if not self.report_path:
            return
        if not in_notebook():  # a script may end next, so the file is written now
            self._report_pass()
            return
        with self._report_lock:
            self._report_dirty = True
            if self._report_thread is None:
                self._report_thread = threading.Thread(target=self._report_worker, daemon=True)
                self._report_thread.start()

    def _report_pass(self) -> None:
        try:
            self.report(images=self._report_images)
        except Exception as e:
            print(f"report: {e}")

    def _report_worker(self) -> None:
        while True:
            with self._report_lock:
                if not self._report_dirty:
                    self._report_thread = None
                    return
                self._report_dirty = False
            self._report_pass()
            if self._report_dirty:  # a burst of verdicts: one rewrite per 5 s
                time.sleep(5)

    def _webp(self, image: str, render=composite) -> str:
        """Cached base64 WebP of the 1400 px composite."""
        if image not in self._webp_cache:
            im = render(image)
            im.thumbnail((1400, 10000))
            buf = io.BytesIO()
            im.save(buf, "WEBP", quality=80)
            self._webp_cache[image] = base64.b64encode(buf.getvalue()).decode()
        return self._webp_cache[image]

    def _record_of(self, image: str) -> Optional[dict]:
        return next((record for record in self.records if record["image"] == image), None)

    def _record(self, image: str, state: dict, answer: Optional[str], note: Optional[str] = None) -> dict:
        existing = self._record_of(image)
        if existing and existing["verdict"] == "reject":
            self.undo(image)
            state = self._state(image)  # the header as restored, not as rejected
        fields = {}
        if answer:
            fits.Card(INSPCOMM_KEY, answer, INSPCOMM_COMMENT)
            try:
                fields = self._write_verdict(image, answer)
            except (PermissionError, DatabaseError) as e:  # the choice is kept here until flush()
                fields = {"inspcomm": answer, "pending": str(e)}
        if existing in self.records:  # replaced only once the write went through
            self.records.remove(existing)
        record = {
            "index": self.images.index(image) if image in self.images else self.cursor,
            "image": image,
            "name": _stem(image),
            "verdict": (
                "ambiguous" if note is not None else "seen" if answer is None else "reject" if answer else "pass"
            ),
            "inspcomm": None,
            "pending": None,
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
        record.update(fields)
        self.records.append(record)
        self._write_log()
        return record

    def _write_verdict(self, image: str, text: str) -> dict:
        """The rejection's writes; the record fields they settle."""
        inspect_images(
            image, sanity=False, inspcomm=text, by=self.by, overwrite=self.overwrite, use_database=self.use_database
        )
        return {"inspcomm": fits.getheader(image).get(INSPCOMM_KEY)}

    def flush(self) -> List[dict]:
        """Write the rejections kept in the session only; those still pending."""
        still = []
        for record in [r for r in self.records if r.get("pending")]:
            try:
                record.update(self._write_verdict(record["image"], record["inspcomm"]), pending=None)
            except (PermissionError, DatabaseError) as e:
                record["pending"] = str(e)
                still.append(record)
        self._write_log()
        return still

    def run(self, retain: bool = False, verdict=None, comment: Optional[str] = None) -> "InspectionSession":
        """Enter pass, text reject, ? ambiguous, u undo, f/b move (100f, 3b), m masters, q quit; `verdict` and/or `comment` (regex, case-insensitive) revise the recorded frames that match instead."""
        kernel = in_notebook()
        if not kernel and not sys.stdin.isatty():
            raise RuntimeError("run() needs a Jupyter kernel or a terminal")
        revise = verdict is not None or comment is not None
        active = self.matching(verdict, comment) if revise else self.images
        try:
            self._loop(kernel, retain, active, revise)
        finally:
            self._schedule_report()
        return self

    def matching(self, verdict=None, comment: Optional[str] = None) -> List[str]:
        """Pool frames whose record has one of `verdict` and a comment (INSPCOMM or note) matching the `comment` regex, case-insensitive."""
        verdicts = {v.lower() for v in ([verdict] if isinstance(verdict, str) else verdict or [])}
        chosen = []
        for image in self.images:
            record = self._record_of(image)
            if record is None:
                continue
            if verdicts and record["verdict"].lower() not in verdicts:
                continue
            text = record["inspcomm"] or record.get("note") or ""
            if comment is not None and not re.search(comment, text, re.IGNORECASE):
                continue
            chosen.append(image)
        return chosen

    def _resume_position(self) -> int:
        """The frame last left: a seen record is shown again, a judged one is passed; nothing waiting beyond it, the first waiting frame, else the end."""
        at = 0
        for record in reversed(self.records):
            if record["image"] in self.images:
                at = self.images.index(record["image"])
                at = at if record["verdict"] == "seen" else at + 1
                break
        judged = {r["image"]: r["verdict"] for r in self.records}
        waiting = [i for i, image in enumerate(self.images) if judged.get(image, "seen") == "seen"]
        if not any(i >= at for i in waiting):
            at = min(waiting) if waiting else len(self.images)
        return at

    def _loop(self, kernel: bool, retain: bool, active: List[str], revise: bool) -> None:
        shown, note, browsing, masters_for = 0, "", False, None
        self._active = active
        self.cursor = 0 if revise else self._resume_position()
        if not revise and active and self.cursor >= len(active):  # everything judged: the last frame, open for b
            self.cursor, browsing = len(active) - 1, True
        while 0 <= self.cursor < len(active):
            image = active[self.cursor]
            record = self._record_of(image)
            if record and record["verdict"] != "seen" and not browsing and not revise:
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
            verdict = f"{_hl('SANITY ' + str(state['sanity']))}  {_hl('INSPCOMM ' + (state['inspcomm'] or '-'))}"
            if state["rej_proc"]:
                verdict += f"  REJ_PROC {state['rej_proc']}"
            metrics = f"seeing {_fmt(state['seeing'])}  ellip {_fmt(state['ellip'])}  ppflag {_fmt(state['ppflag'])}"
            self._clear(kernel, retain)
            if note:
                print(note)
            print(f"[{self.cursor + 1}/{len(active)}] {_hl(_judged(record))} {_stem(image)}  {metrics}  {verdict}")
            status = _status_line(image) if self.use_database and self.log_lines else None
            if status:
                print(f"    {status}")
            for line in _log_tail(image, self.log_lines):
                print(f"    {line}")
            self._display(im, image, kernel)
            if masters_for == image:
                from .masterframe_session import composite_for_single

                try:
                    self._display(composite_for_single(image), f"{_stem(image)}_masters", kernel)
                except (OSError, ValueError) as e:
                    print(f"masters: {e}")
            _settle()
            try:
                answer = input("Enter=pass  text=reject  ?=ambiguous  u=undo  f/b=move  m=masters  q=quit: ").strip()
            except (EOFError, KeyboardInterrupt, NotImplementedError):
                if record is None:  # Ctrl-C quits like q: the frame on screen is left as seen
                    self._record(image, state, None)
                return
            note = ""
            if answer.lower() == "m":
                masters_for, browsing = None if masters_for == image else image, True
                continue
            masters_for = None
            if answer.lower() == "q":
                if record is None:
                    self._record(image, state, None)
                return
            if answer.lower() == "u":
                try:
                    if revise or (record and record["verdict"] != "seen"):  # a judged frame reached by f/b: clear it
                        undone = self.undo(image)
                        if undone:
                            self._record(image, self._state(image), None)
                        note = f"cleared {undone['name']}" if undone else "nothing to clear"
                        if revise:
                            self.cursor += 1
                    else:
                        undone = self.undo()
                        note = f"undone {undone['name']}" if undone else "nothing to undo"
                except (PermissionError, DatabaseError) as e:
                    note, browsing = f"not undone: {e}", True
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
                self.cursor = min(max(self.cursor + step, 0), len(active) - 1)
                browsing = True
                continue
            if not answer and record and record["verdict"] != "seen":
                self.cursor += 1
                continue
            if answer and not _confirm_short(answer):
                note, browsing = f"not recorded: {answer!r}", True
                continue
            try:
                record = self._record(image, state, answer)
            except (ValueError, PermissionError, DatabaseError) as e:
                note, browsing = f"not recorded: {e}", True
                continue
            if record.get("pending"):  # shown again with the note; Enter moves on
                note, browsing = f"kept in the session only, not written: {record['pending']}", True
                continue
            if answer and record["inspcomm"] != answer:
                note = f"kept the existing INSPCOMM: {record['inspcomm']}"
            self.cursor += 1
        if not shown:
            print("nothing matches" if revise else "nothing left to inspect")

    def judge(self, images, answer: str = "") -> List[dict]:
        """Verdicts without the pane: "" pass, text reject, ?note ambiguous, u clear."""
        records = []
        for image in atleast_1d(images):
            state = self._state(image)
            if answer.lower() == "u":
                if self.undo(image):
                    records.append(self._record(image, self._state(image), None))
            elif answer.startswith("?"):
                records.append(self._record(image, state, None, note=answer[1:].strip()))
            else:
                records.append(self._record(image, state, answer))
        return records

    def masters(self, by: str = "count") -> List[dict]:
        """Distinct masters of the pool: path, name, role, count of singles using it; sorted by `by` (count descending)."""
        from .masterframe_session import _masters_of

        used = {}
        for roles in _masters_of(self.images).values():
            for role, path in roles.items():
                entry = used.setdefault(path, {"path": path, "name": _stem(path), "role": role, "count": 0})
                entry["count"] += 1
        return sorted(used.values(), key=lambda entry: -entry["count"] if by == "count" else entry[by])

    def undo(self, image: Optional[str] = None) -> Optional[dict]:
        """Reverse the last verdict, or the one on `image`; a rejection goes back to the state recorded before it."""
        matches = [record for record in self.records if image is None or record["image"] == image]
        if not matches:
            return None
        record = matches[-1]
        if record["verdict"] == "reject" and not record.get("pending"):
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
        active = getattr(self, "_active", None) or self.images
        if record["image"] in active:
            self.cursor = min(self.cursor, active.index(record["image"]))
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

    def report(self, path: Optional[str] = None, images: str = "rejections", max_mb: float = 50) -> str:
        """Self-contained Markdown at `path` (default `report_path`): counts, a count per INSPCOMM, 1400 px WebP composites for `images` in rejections (rejected and ambiguous) | all | none."""
        path = path or self.report_path
        if not path:
            raise ValueError("no report path: pass one or construct the session with stem or report_path")
        if images not in ("rejections", "all", "none"):
            raise ValueError(f"images must be rejections, all or none, not {images!r}")
        if path == self.report_path:
            self._report_images = images  # the background rewrites keep the last explicit choice
        records = list(self.records)
        rejected = [record for record in records if record["verdict"] == "reject"]
        ambiguous = [record for record in records if record["verdict"] == "ambiguous"]
        passed = [record for record in records if record["verdict"] == "pass"]
        seen = [record for record in records if record["verdict"] == "seen"]
        n_embedded = {
            "rejections": len(rejected) + len(ambiguous),
            "all": len(rejected) + len(ambiguous) + len(passed),
            "none": 0,
        }[images]
        estimate = 0.12 * n_embedded + 0.0001 * len(records)
        if estimate > max_mb:
            raise ValueError(f"about {estimate:.2f} MB for {n_embedded} composites; images='none' or a smaller pool")
        pending = [record for record in rejected if record.get("pending")]
        comments = {}
        for record in rejected:
            if not record.get("pending"):
                comments[record["inspcomm"]] = comments.get(record["inspcomm"], 0) + 1
        lines = [
            f"# Inspection session, {self.started:%Y-%m-%d %H:%M}, inspector {self.by}",
            "",
            f"{len(passed) + len(rejected)} of {len(self.images)} frames judged: {len(passed)} pass, {len(rejected)} reject"
            f"{f' ({len(pending)} not written)' if pending else ''}; "
            f"{len(ambiguous)} ambiguous; {len(self.images) - len(passed) - len(rejected) - len(ambiguous)} without a verdict, "
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
                lines += ["| frame | name | comment | prior SANITY |", "|---|---|---|---|"]
                lines += [
                    f"| {record['index'] + 1}/{len(self.images)} | {record['name']} | {_cell(_comment(record))} | {record['prior_sanity']} |"
                    for record in group
                ]
                continue
            for record in group:
                parts = [
                    _cell(_comment(record)),
                    f"seeing {_fmt(record['seeing'])}",
                    f"ellipticity {_fmt(record['ellip'])}",
                    f"ppflag {_fmt(record['ppflag'])}",
                    f"prior SANITY {record['prior_sanity']}",
                    f"REJ_PROC {record['prior_rej_proc']}" if record["prior_rej_proc"] else None,
                ]
                data = self._webp(record["image"])
                lines += [
                    f"### {record['index'] + 1}/{len(self.images)} {record['name']}: {record['verdict'].upper()}{_pending_mark(record)}",
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
        written = _write_atomic(path, lines)
        if path == self.report_path and in_notebook() and threading.current_thread() is not self._report_thread:
            self._schedule_report()  # a worker pass still rendering an older snapshot lands after this one
        return written
