"""Masters each raw science frame will be reduced with, and the masters those are built from, read from the databases.

The raw-frame budget and the image_qa registry form a master library that Preprocess's own `tolerant_search` probes in
place of the disk, holding each regenerated master as the code would leave it. Read-only, no FITS file opened.
Rationale and measurements: .claude/memory/services.md "masterframe_association.py".
"""

from __future__ import annotations

import fnmatch
import logging
import os
import signal
import time
import traceback
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import psycopg
import yaml

from ..const import (
    BIAS_DEPENDENCY_ROLE,
    CALIB_TYPE_BIAS,
    CALIB_TYPE_DARK,
    CALIB_TYPE_FLAT,
    CONFIG_TYPE_PREPROCESS,
    CONFIG_TYPE_SCIENCE,
    DARK_DEPENDENCY_ROLE,
    DISK_CHANGE_NIGHTDATE,
    DISK_CHANGE_NIGHTDATE_2,
    FLAT_DEPENDENCY_ROLE,
    IMAGE_GROUP_MASTERFRAME,
    MASTER_IMAGE_TYPES,
    NAME_TYPE_MASTER,
    NUM_MIN_CALIB,
    REF_DIR,
    instrum_log,
)
from ..path.cam_tracker import get_camera_serial
from ..path.name import NameHandler
from ..path.path import PathHandler
from ..path.utils import add_a_day, format_camera, format_exptime, get_nightdate
from ..utils import atleast_1d
from ..version import MIN_PREPROC_RUNTIME_VERSION, is_below_min
from .database.const import DB_PARAMS
from .database.masterframe_plan import fingerprints, plan_groups, plan_masters, plan_status, replace_plan
from .database.query import free_query

_SCIENCE_FRAMES = """
SELECT x.file_path, n.date::text, u.name, COALESCE(tl.name, t.name, NULLIF(x.object_name, '')), f.name, x.exptime,
       x.binning_x, x.gain
FROM survey_scienceframe x JOIN facility_unit u ON u.id = x.unit_id JOIN survey_night n ON n.id = x.night_id
LEFT JOIN facility_filter f ON f.id = x.filter_id LEFT JOIN survey_tile tl ON tl.id = x.tile_id
LEFT JOIN survey_target t ON t.id = x.target_id
WHERE {}
"""

_SCIENCE_INVENTORY = """
SELECT DISTINCT x.binning_x, x.gain, f.name, x.exptime
FROM survey_scienceframe x JOIN facility_unit u ON u.id = x.unit_id JOIN survey_night n ON n.id = x.night_id
LEFT JOIN facility_filter f ON f.id = x.filter_id
WHERE u.name = %s AND n.date = %s::date
"""

_RAW_BUDGET = """
SELECT 'bias', u.name, n.date::text, x.binning_x, x.gain, NULL::float8, NULL::text, count(*)
FROM survey_biasframe x JOIN facility_unit u ON u.id = x.unit_id JOIN survey_night n ON n.id = x.night_id
WHERE u.name LIKE ANY(%s)
GROUP BY 2, 3, 4, 5
UNION ALL
SELECT 'dark', u.name, n.date::text, x.binning_x, x.gain, x.exptime, NULL::text, count(*)
FROM survey_darkframe x JOIN facility_unit u ON u.id = x.unit_id JOIN survey_night n ON n.id = x.night_id
WHERE u.name LIKE ANY(%s)
GROUP BY 2, 3, 4, 5, 6
UNION ALL
SELECT 'flat', u.name, n.date::text, x.binning_x, x.gain, NULL::float8, f.name, count(*)
FROM survey_flatframe x JOIN facility_unit u ON u.id = x.unit_id JOIN survey_night n ON n.id = x.night_id
LEFT JOIN facility_filter f ON f.id = x.filter_id
WHERE u.name LIKE ANY(%s)
GROUP BY 2, 3, 4, 5, 7
"""

_REGISTRY = """
SELECT image_name, image_path, sanity,
       NULLIF(btrim(inspcomm), '') IS NOT NULL AND inspcomm <> 'Auto-generated binned master frame',
       COALESCE(created_at <= %s, TRUE), image_type, pipe_ver, {}
FROM image_qa WHERE image_group = %s AND image_type = ANY(%s) AND image_path IS NOT NULL AND image_path LIKE ANY(%s)
"""

_BINNED_COPIES = """
SELECT m.image_path, min(s.image_path)
FROM image_qa m JOIN image_qa_dependency d ON d.derived_image_id = m.id JOIN image_qa s ON s.id = d.source_image_id
WHERE m.image_group = %s AND m.image_type = %s
GROUP BY m.id, m.image_path HAVING count(*) = 1 AND bool_and(d.dependency_role = %s)
"""

_EDGES = """
SELECT d.dependency_role, s.image_path, d.source_imageid, s.imageid
FROM image_qa m JOIN image_qa_dependency d ON d.derived_image_id = m.id JOIN image_qa s ON s.id = d.source_image_id
WHERE m.image_name = %s AND d.dependency_role = ANY(%s)
"""

_CONFIGS = """
SELECT config_file, config_type, nightdate::text, unit, object, filter FROM process_status WHERE config_file = ANY(%s)
"""

_PREPROCESS_CONFIGS = """
SELECT DISTINCT ON (name) name, config_file, max_offset, ignore_sanity_if_no_match, ignore_lenient_keys_if_no_match,
       reuse_recorded_masters, designated_masterframes
FROM process_status WHERE config_type = %s
ORDER BY name, updated_at DESC NULLS LAST, id DESC
"""

_SCIENCE_NIGHTS = """
SELECT DISTINCT n.date::text
FROM survey_scienceframe x JOIN facility_unit u ON u.id = x.unit_id JOIN survey_night n ON n.id = x.night_id
WHERE u.name = %s
"""

_SCIENCE_NIGHT_RANGE = """
SELECT min(n.date), max(n.date)
FROM survey_scienceframe x JOIN facility_unit u ON u.id = x.unit_id JOIN survey_night n ON n.id = x.night_id
WHERE u.name = ANY(%s)
"""

_UNIT_MASTERS = """
SELECT image_name FROM image_qa WHERE image_group = %s AND image_type = ANY(%s) AND image_path LIKE ANY(%s)
"""

_FOREIGN_EDGES = """
SELECT count(*)
FROM image_qa m JOIN image_qa_dependency d ON d.derived_image_id = m.id JOIN image_qa s ON s.id = d.source_image_id
WHERE m.image_group = %s AND m.image_path LIKE %s AND s.image_path NOT LIKE %s AND d.dependency_role = ANY(%s)
"""

_FOREIGN_MASTERS = """
SELECT count(*) FROM image_qa
WHERE image_group = %s AND image_type = ANY(%s) AND (image_path LIKE %s) <> (image_name LIKE %s)
"""

_FOREIGN_DESIGNATIONS = """
SELECT count(*) FROM process_status p
WHERE p.config_type = %s AND p.name LIKE %s AND jsonb_typeof(p.designated_masterframes) = 'array'
  AND EXISTS (
      SELECT 1 FROM jsonb_array_elements(p.designated_masterframes) e
      WHERE regexp_replace(e->>'path', '^.*/', '') NOT LIKE %s
  )
"""


class _Unresolved(Exception):
    """A probe met a regenerated master whose modeled state is not resolved yet."""


class _Library:
    """Masters of the modeled state, probed by tolerant_search through glob / exists / sanity_check."""

    def __init__(
        self,
        trust_machine_rejection: bool,
        as_of: Optional[datetime],
        targets: set,
        veto: Iterable[str] = (),
        units: Optional[Iterable[str]] = None,
    ):
        from .checker import Checker

        self.trust_machine_rejection = trust_machine_rejection
        self.as_of = as_of
        self.targets = targets
        self.veto = {_stem(str(master)) for master in ([veto] if isinstance(veto, (str, os.PathLike)) else veto)}
        self.checker = Checker()
        self.checker.logger = logging.getLogger(__name__)
        criteria = self.checker.load_qa_criteria(IMAGE_GROUP_MASTERFRAME)
        self.raw = defaultdict(list)
        names = ["%"] if units is None else list(units)
        for typ, unit, night, nbin, gain, exptime, filt, n in free_query(_RAW_BUDGET, [names] * 3):
            quality = format_exptime(float(exptime), type="dark") if typ == CALIB_TYPE_DARK else filt
            self.raw[(night, unit)].append((typ, quality, nbin, gain, n))
        self.registry = defaultdict(dict)
        self.registry_dirs = defaultdict(set)
        self.registered_path = {}
        statistics = sorted({key for keys in criteria.values() for key in keys})
        dirs = ["%"] if units is None else [f"%/{unit}/%" for unit in units]
        for name, path, sanity, human, visible, dtype, pipe_ver, *values in free_query(
            _REGISTRY.format(", ".join(statistics)), [as_of, IMAGE_GROUP_MASTERFRAME, list(MASTER_IMAGE_TYPES), dirs]
        ):
            directory = os.path.dirname(path)
            row = (sanity, human, visible, dtype, pipe_ver, dict(zip(statistics, values)))
            self.registry[directory][os.path.basename(path)] = row
            self.registry_dirs[(get_nightdate(directory), os.path.basename(directory))].add(directory)
            self.registered_path[name] = path
        copies = free_query(_BINNED_COPIES, [IMAGE_GROUP_MASTERFRAME, CALIB_TYPE_FLAT, FLAT_DEPENDENCY_ROLE])
        self.copies = {_stem(copy): source for copy, source in copies}
        with open(os.path.join(REF_DIR, "preproc_base.yml")) as f:
            self.base = _read_settings(yaml.safe_load(f))
        self.configs = None
        self.cameras, self.groups_of, self.dirs = {}, {}, {}
        self.searches, self.ingredients, self.settings, self.designations = {}, {}, {}, {}
        self.inventories, self.edges = {}, {}
        self.skipped, self.globs, self.sibling_sources, self.blind = [], [], {}, {}
        self.unrecorded = set()
        for stem in sorted(self.veto - set(self.registered_path)):
            try:
                name = NameHandler(f"{stem}.fits")
                known = f"{stem}.fits" in self.groups(str(name.nightdate), name.unit)
            except ValueError:
                known = False
            if not known:
                raise ValueError(f"veto {stem}: not a registered master or a raw calibration group")

    def camera(self, unit: str, night: str) -> str:
        """Camera token of a unit-night, memoized."""
        if (unit, night) not in self.cameras:
            self.cameras[(unit, night)] = format_camera(get_camera_serial(unit=int(unit[3:]), query_date=night))
        return self.cameras[(unit, night)]

    def groups(self, night: str, unit: str) -> Dict[str, dict]:
        """Raw calibration groups of a unit-night by master basename: frame count and keys."""
        if (night, unit) not in self.groups_of:
            groups = {}
            for typ, quality, nbin, gain, n in self.raw.get((night, unit), []):
                name = _master_basename(typ, quality, unit, night, nbin, gain, self.camera(unit, night))
                group = groups.setdefault(name, {"n": 0, "keys": (typ, quality, nbin, gain)})
                group["n"] += n
            self.groups_of[(night, unit)] = groups
        return self.groups_of[(night, unit)]

    def entries(self, directory: str) -> Tuple[dict, dict]:
        """Masters of a master directory: in the library, and registered after as_of."""
        if directory not in self.dirs:
            night, unit = get_nightdate(directory), os.path.basename(directory)
            groups = self.groups(night, unit) if night else {}
            present, hidden = {}, {}
            if groups and (self.as_of is None or (night, unit) in self.targets):
                for name, path in zip(groups, _canonical(list(groups))):
                    if os.path.dirname(path) == directory and groups[name]["n"] >= NUM_MIN_CALIB:
                        rec = self._record(night, unit, groups[name])
                        if rec["rebuilt"] and _stem(name) not in self.veto:
                            present[name] = rec
            rows = [
                (d, b, row)
                for d in sorted(self.registry_dirs.get((night, unit), ()))
                for b, row in self.registry[d].items()
            ]
            canonical_paths = _canonical([b for _, b, _ in rows]) if rows else []
            for (registered, name, (sanity, human, visible, *rest)), path in zip(rows, canonical_paths):
                canonical = os.path.dirname(path) == directory
                if registered == directory or canonical:
                    if _stem(name) in self.veto:
                        sanity, human = False, True
                    rec = self._record(
                        night, unit, groups.get(name) if canonical else None, (sanity, human, *rest), visible
                    )
                    (present if visible else hidden)[name] = rec
            self.dirs[directory] = present, hidden
        return self.dirs[directory]

    def _record(
        self, night: str, unit: str, group: Optional[dict], row: Optional[tuple] = None, visible: bool = True
    ) -> dict:
        """Library record of a master."""
        sanity, human, dtype, pipe_ver, statistics = row or (None, False, None, None, None)
        n_raw = group["n"] if group else 0
        typ = group["keys"][0] if group else None
        rebuilt = n_raw >= NUM_MIN_CALIB and (self.as_of is None or (night, unit) in self.targets)
        rebuilt = rebuilt and not (sanity is False and human)
        regen = rebuilt and typ in (CALIB_TYPE_DARK, CALIB_TYPE_FLAT)
        if self.as_of is None and row is not None:
            rebuilt = rebuilt and is_below_min(pipe_ver, MIN_PREPROC_RUNTIME_VERSION)  # _resolve adds the ingredients
        failing = [
            f"{key}={value}"
            for key, value in sorted((statistics or {}).items())
            if not self.checker.apply_qa_criteria({key: value}, dtype)
        ]
        return {
            "nightdate": night,
            "unit": unit,
            "n_raw": n_raw,
            "keys": group and group["keys"],
            "registered": row is not None,
            "visible": visible,
            "rebuilt": rebuilt,
            "sanity": sanity,
            "human": human,
            "vetoed": sanity is False and human,
            "failing": failing,
            "regen": regen,
            "resolved": not regen,
            "viable": True,
            "forced": [],
        }

    def record(self, path: str) -> Optional[dict]:
        """Library record of a master path, else its registry record."""
        present, hidden = self.entries(os.path.dirname(path))
        return present.get(os.path.basename(path)) or hidden.get(os.path.basename(path))

    def glob(self, pattern: str) -> List[str]:
        """glob over the library."""
        directory, name = os.path.split(pattern)
        present = self.entries(directory)[0]
        names = fnmatch.filter(present, name)
        for n in names:
            if not present[n]["registered"] and not present[n]["resolved"]:
                raise _Unresolved(os.path.join(directory, n))
        matches = [os.path.join(directory, n) for n in names if present[n]["registered"] or present[n]["viable"]]
        if len(matches) > 1:
            self.globs.append(tuple(matches))
        return matches

    def exists(self, path: str) -> bool:
        """os.path.exists over the library."""
        rec = self.entries(os.path.dirname(path))[0].get(os.path.basename(path))
        if rec is not None and not rec["registered"] and not rec["resolved"]:
            raise _Unresolved(path)
        return rec is not None and (rec["registered"] or rec["viable"])

    def sanity_check(self, path: str) -> bool:
        """False for a master the plan takes as SANITY F: a human veto always, a machine F when trusted."""
        rec = self.entries(os.path.dirname(path))[0][os.path.basename(path)]
        if not rec["resolved"]:
            raise _Unresolved(path)
        rejected = rec["vetoed"] or (self.trust_machine_rejection and _sanity(rec) is False)
        if rejected and not rec["vetoed"] and rec["rebuilt"] and rec["viable"]:
            self.skipped.append(path)
        return not rejected

    def human_rejected(self, path: str) -> bool:
        """SANITY F with a human INSPCOMM in the registry: a veto."""
        return self.entries(os.path.dirname(path))[0][os.path.basename(path)]["vetoed"]


def _sanity(rec: dict) -> Optional[bool]:
    """SANITY the plan uses: the registry's unless rebuilt, else the own verdict and the ingredients'."""
    if not (rec["rebuilt"] and rec["viable"]):
        return rec["sanity"]
    return not (rec["vetoed"] or rec["failing"] or rec["forced"])


def _state(rec: dict) -> str:
    """registered | budget-only | registry-only."""
    if not rec["registered"]:
        return "budget-only"
    return "registered" if rec["n_raw"] >= NUM_MIN_CALIB else "registry-only"


def _master_basename(typ: str, quality, unit: str, night: str, nbin, gain, camera: str) -> str:
    """Master basename of a raw group."""
    return NameHandler._format_masterframe(typ, quality, unit, add_a_day(night), nbin, gain, camera)


def _canonical(basenames: List[str]) -> List[str]:
    """Pipeline paths of master basenames."""
    return atleast_1d(PathHandler(basenames, is_pipeline=True).preprocess._masterframe)


def _stem(path: str) -> str:
    """File stem, as image_qa.image_name."""
    return os.path.basename(path).removesuffix(".fits")


def _read_settings(yml: dict, default_max_offset: Optional[int] = None) -> Tuple[int, bool, bool]:
    """(max_offset, ignore_sanity_if_no_match, ignore_lenient_keys_if_no_match) of a preprocess yml."""
    section = (yml or {}).get("preprocess") or {}
    return (
        section.get("max_offset") or default_max_offset,
        bool(section.get("ignore_sanity_if_no_match")),
        bool(section.get("ignore_lenient_keys_if_no_match")),
    )


def _row_of(library: _Library, night: str, unit: str) -> Optional[SimpleNamespace]:
    """process_status row of a unit-night's preprocess config: config_file and the master-selection columns."""
    if library.configs is None:
        keys = ["config_file", "max_offset", "ignore_sanity_if_no_match", "ignore_lenient_keys_if_no_match"]
        keys += ["reuse_recorded_masters", "designated_masterframes"]
        rows = free_query(_PREPROCESS_CONFIGS, [CONFIG_TYPE_PREPROCESS])
        library.configs = {name: SimpleNamespace(**dict(zip(keys, rest))) for name, *rest in rows}
    return library.configs.get(f"{night}_{unit}")


def _settings_of(library: _Library, night: str, unit: str) -> Tuple[int, bool, bool]:
    """Search settings of a unit-night as its process_status row records them; the base without a row or record."""
    if (night, unit) not in library.settings:
        row = _row_of(library, night, unit)
        settings = library.base
        if row is not None:
            recorded = (row.max_offset, row.ignore_sanity_if_no_match, row.ignore_lenient_keys_if_no_match)
            if recorded == (None, None, None):
                library.unrecorded.add((night, unit))
            else:
                settings = (recorded[0] or library.base[0], bool(recorded[1]), bool(recorded[2]))
        library.settings[(night, unit)] = settings
    return library.settings[(night, unit)]


def _unmodeled(row: Optional[SimpleNamespace]) -> List[str]:
    """What a preprocess row records that the service does not model."""
    issues = []
    if any(e["images"] for e in (row and row.designated_masterframes) or []):
        issues.append("designated_masterframes with images: pins")
    if row and row.reuse_recorded_masters:
        issues.append("reuse_recorded_masters: true")
    return issues


def _designations(library: _Library, night: str, unit: str) -> List[tuple]:
    """Plain designations of a unit-night: (path, NameHandler, role)."""
    if (night, unit) not in library.designations:
        found = []
        row = _row_of(library, night, unit)
        for entry in (row and row.designated_masterframes) or []:
            if entry["images"]:
                continue
            role, path = entry["role"], str(entry["path"])
            if library.record(path) is None:
                path = _canonical([os.path.basename(path)])[0]
            found.append((path, NameHandler(path), role))
        library.designations[(night, unit)] = found
    return library.designations[(night, unit)]


def _designated(library: _Library, owner: Tuple[str, str], role: str, template: str) -> Optional[dict]:
    """The owner yml's designation for one role of a template, as ReprocessMixin dispatches it."""
    entries = _designations(library, *owner)
    if not entries:
        return None
    target = NameHandler(template)
    if role == "flatdark":
        candidates = [
            (name.exptime, path, r)
            for path, name, r in entries
            if name.type.exposure_type == CALIB_TYPE_DARK
            and r in (None, "flatdark")
            and str(name.n_binning) == str(target.n_binning)
        ]
        pinned = [c for c in candidates if c[2] == "flatdark"]
        if len(pinned) > 1:
            detail = f"designated_masterframes: {len(pinned)} flatdark pins match: {[_stem(p) for _, p, _ in pinned]}"
            return _none([("search-error", template, detail)])
        chosen = pinned[0][1] if pinned else min(candidates)[1] if candidates else None
    else:
        matches = [
            path
            for path, name, r in entries
            if name.type.exposure_type == role
            and r in (None, role)
            and str(name.n_binning) == str(target.n_binning)
            and (role != CALIB_TYPE_FLAT or name.filter == target.filter)
            and (role != CALIB_TYPE_DARK or str(name.exptime) == str(target.exptime))
        ]
        if len(matches) > 1:
            detail = (
                f"designated_masterframes: {len(matches)} candidates for the {role} role "
                f"({os.path.basename(template)}): {[os.path.basename(p) for p in matches]}"
            )
            return _none([("search-error", template, detail)])
        chosen = matches[0] if matches else None
    if chosen is None:
        return None
    rec = library.record(chosen)
    defects = _master_checks(library, template, chosen, rec) if rec else []
    return {"path": chosen, "source": chosen, "rule": "designated", "defects": defects}


def _search(library: _Library, template: str, dtype: str, settings: Tuple[int, bool, bool]) -> dict:
    """tolerant_search over the library, memoized: the selected master, its rule and defects."""
    key = (template, dtype, settings)
    if key not in library.searches:
        handler = signal.getsignal(signal.SIGINT)
        from ..preprocess.utils import tolerant_search

        signal.signal(signal.SIGINT, handler)  # importing preprocess imports services.queue, which ignores SIGINT
        max_offset, ignore_sanity, ignore_lenient = settings
        while key not in library.searches:
            library.skipped, library.globs = [], []
            unresolved = None
            try:
                found, flags = tolerant_search(
                    template,
                    dtype,
                    max_offset=max_offset,
                    future=True,
                    ignore_sanity_if_no_match=ignore_sanity,
                    ignore_lenient_keys_if_no_match=ignore_lenient,
                    library=library,
                )
            except _Unresolved as e:
                unresolved = e.args[0]
            except psycopg.Error:
                raise
            except Exception as e:
                tie = isinstance(e, ValueError) and any(
                    f.name == "_pick_master" for f in traceback.extract_tb(e.__traceback__)
                )
                defects = [("tie" if tie else "search-error", template, f"{type(e).__name__}: {e}")]
                library.searches[key] = _none(defects + _attempt_defects(library, template, dtype))
            else:
                library.searches[key] = _selected(library, template, dtype, found, flags)
            if unresolved:
                _resolve(library, unresolved)
        library.searches[key] = _root_blind_check(library, template, dtype, settings, library.searches[key])
    return library.searches[key]


def _root_gap(night: date, boundary: str) -> int:
    """Smallest day offset from a night that probes the other side of a disk-root boundary."""
    first = date.fromisoformat(boundary)
    return (night - first).days + 1 if night >= first else (first - night).days


def _root_blind_check(library: _Library, template: str, dtype: str, settings: tuple, answer: dict) -> dict:
    """other-disk-root defect from a root-blind rerun of the search, where a boundary can matter."""
    text = get_nightdate(template)
    night = text and date.fromisoformat(text)
    gaps = [
        g
        for b in (DISK_CHANGE_NIGHTDATE, DISK_CHANGE_NIGHTDATE_2)
        if night and (g := _root_gap(night, b)) <= settings[0]
    ]
    if not gaps:
        return answer
    gap = min(gaps)
    found = answer["source"]
    rec = found and library.record(found)
    far = rec and abs((date.fromisoformat(rec["nightdate"]) - night).days) >= gap
    if found and "+" not in answer["rule"] and not far:
        return answer
    blind = _blind_search(library, template, dtype, settings)
    if blind is None or (found and _stem(blind) == _stem(found)):
        return answer
    root = os.path.dirname(os.path.dirname(os.path.dirname(blind)))
    return dict(answer, defects=answer["defects"] + [("other-disk-root", template, f"{_stem(blind)} on {root}")])


def _blind_search(library: _Library, template: str, dtype: str, settings: tuple) -> Optional[str]:
    """Master tolerant_search takes when every night's masters are visible under the template's root, memoized."""
    key = ("search", template, dtype, settings)
    if key not in library.blind:
        from ..preprocess.utils import tolerant_search

        view = SimpleNamespace(
            glob=lambda pattern: _blind_glob(library, pattern),
            exists=lambda path: _blind_exists(library, path),
            sanity_check=lambda path: library.sanity_check(_blind_path(library, path)),
            human_rejected=lambda path: library.human_rejected(_blind_path(library, path)),
        )
        max_offset, ignore_sanity, ignore_lenient = settings
        while key not in library.blind:
            library.skipped, library.globs = [], []
            unresolved = None
            try:
                found, _ = tolerant_search(
                    template,
                    dtype,
                    max_offset=max_offset,
                    future=True,
                    ignore_sanity_if_no_match=ignore_sanity,
                    ignore_lenient_keys_if_no_match=ignore_lenient,
                    library=view,
                )
            except _Unresolved as e:
                unresolved = e.args[0]
            except ValueError:
                library.blind[key] = None
            else:
                library.blind[key] = found and _blind_path(library, found)
            if unresolved:
                _resolve(library, unresolved)
    return library.blind[key]


def _blind_view(library: _Library, directory: str) -> Dict[str, str]:
    """Basename -> real path of every library master of the directory's (nightdate, unit), whatever its root."""
    key = ("view", directory)
    if key not in library.blind:
        night, unit = get_nightdate(directory), os.path.basename(directory)
        groups = library.groups(night, unit) if night else {}
        others = set(library.registry_dirs.get((night, unit), ()))
        if groups:
            others.add(os.path.dirname(_canonical([next(iter(groups))])[0]))
        view = {}
        for d in [directory] + sorted(others - {directory}):
            for name in library.entries(d)[0]:
                view.setdefault(name, os.path.join(d, name))
        library.blind[key] = view
    return library.blind[key]


def _blind_path(library: _Library, path: str) -> str:
    """Real path behind a root-blind probe."""
    return _blind_view(library, os.path.dirname(path)).get(os.path.basename(path), path)


def _blind_glob(library: _Library, pattern: str) -> List[str]:
    """glob over the root-blind view."""
    directory, name = os.path.split(pattern)
    view = _blind_view(library, directory)
    matches = []
    for n in fnmatch.filter(view, name):
        rec = library.record(view[n])
        if not rec["registered"] and not rec["resolved"]:
            raise _Unresolved(view[n])
        if rec["registered"] or rec["viable"]:
            matches.append(view[n])
    return matches


def _blind_exists(library: _Library, path: str) -> bool:
    """os.path.exists over the root-blind view."""
    real = _blind_view(library, os.path.dirname(path)).get(os.path.basename(path))
    rec = real and library.record(real)
    if rec and not rec["registered"] and not rec["resolved"]:
        raise _Unresolved(real)
    return bool(rec) and (rec["registered"] or rec["viable"])


def _resolve(library: _Library, path: str) -> None:
    """Viability, rebuild, forced F and ingredients of a dark or flat with raw budget, from its own searches."""
    rec = library.record(path)
    is_flat = rec["keys"][0] == CALIB_TYPE_FLAT
    ingredients = _planned(library, path, (rec["nightdate"], rec["unit"]), is_flat)
    answers = list(ingredients.values())
    viable = all(a["path"] for a in answers)
    if viable and is_flat:
        viable = _owner_dark_resolves(library, rec)
    rec["viable"] = viable
    if library.as_of is None and rec["registered"]:
        rec["rebuilt"] = viable and (rec["rebuilt"] or _changed(library, path, answers))
    rec["forced"] = [_stem(a["source"]) for a in answers if viable and "sanity-ignored" in a["rule"]]
    rec["resolved"] = True
    library.ingredients[path] = ingredients if rec["rebuilt"] else _recorded(library, path, is_flat)


def _changed(library: _Library, path: str, answers: List[dict]) -> bool:
    """Whether a registered master's planned ingredients differ from its recorded edges, moved, or are rebuilt."""
    edges = _edges(library, path)
    if not edges:
        return True  # records no ingredient: "unidentified", as ReprocessMixin._master_change
    recorded = {_stem(source) for _, source, _, _ in edges}
    for answer in answers:
        rec = library.record(answer["source"])
        if rec and not rec["resolved"]:
            _resolve(library, answer["source"])
        if _stem(answer["path"]) not in recorded or (rec and rec["rebuilt"]):
            return True
    return any(now and was != now for _, _, was, now in edges)


def _owner_dark_resolves(library: _Library, rec: dict) -> bool:
    """Whether some group that builds this flat at its unit-night gets its own dark first."""
    night, unit = rec["nightdate"], rec["unit"]
    if (night, unit) not in library.inventories:
        library.inventories[(night, unit)] = free_query(_SCIENCE_INVENTORY, [unit, night])
    _, filt, nbin, gain = rec["keys"]
    exptimes = sorted({e for b, g, f, e in library.inventories[(night, unit)] if (b, g, f) == (nbin, gain, filt)})
    camera = library.camera(unit, night)
    for exptime in exptimes:
        try:
            quality = format_exptime(float(exptime), type="dark")
            template = _canonical([_master_basename(CALIB_TYPE_DARK, quality, unit, night, nbin, gain, camera)])[0]
        except (ValueError, TypeError):
            continue
        if _answer(library, template, CALIB_TYPE_DARK, (night, unit), CALIB_TYPE_DARK)["path"]:
            return True
    return not exptimes


def _none(defects: list) -> dict:
    """No answer, with the defects that say why."""
    return {"path": None, "source": None, "rule": "none", "defects": defects}


def _skipped(library: _Library, template: str) -> list:
    """own-sanity-f or ingredient-sanity-f defect for each rebuilt master the last search attempt skipped."""
    defects = []
    for path in dict.fromkeys(library.skipped):
        rec = library.record(path)
        if rec["failing"]:
            defects.append(("own-sanity-f", template, f"{_stem(path)}: {', '.join(rec['failing'])}"))
        else:
            defects.append(("ingredient-sanity-f", template, f"{_stem(path)}: {', '.join(rec['forced'])}"))
    return defects


def _name_order_ties(library: _Library, template: str, dtype: str) -> list:
    """tie defect for every date of the last attempt where _pick_master ranked bias/flat candidates by name only."""
    if dtype == CALIB_TYPE_DARK or not library.globs:
        return []
    n_binning = NameHandler(template).n_binning
    dates = []
    for matches in dict.fromkeys(library.globs):
        binnings = NameHandler(list(matches)).n_binning
        usable = [(m, b) for m, b in zip(matches, binnings) if not (n_binning and b and n_binning % b)]
        if len(usable) < 2 or n_binning is None or any(b is None for _, b in usable):
            continue
        best = min(abs(b - n_binning) for _, b in usable)
        tied = sorted(_stem(m) for m, b in usable if abs(b - n_binning) == best)
        if len(tied) > 1:
            dates.append(f"{get_nightdate(matches[0])}: {', '.join(tied)}")
    return [("tie", template, "; ".join(dates))] if dates else []


def _attempt_defects(library: _Library, template: str, dtype: str) -> list:
    """Defects of the last search attempt as a whole: own-sanity-f, ingredient-sanity-f and name-order ties."""
    return _skipped(library, template) + _name_order_ties(library, template, dtype)


def _selected(library: _Library, template: str, dtype: str, found: Optional[str], flags) -> dict:
    """Answer of one search: path used, source master, rule label and defects."""
    defects = _attempt_defects(library, template, dtype)
    if not found:
        return _none([("no-candidate", template, "")] + defects)
    path = found
    if flags.ignored_binning:
        name = NameHandler(found)
        name.n_binning = NameHandler(template).n_binning
        path = os.path.join(os.path.dirname(found), name.mflat_basename)
        detail = (
            f"{_stem(path)}: not made yet, copied from {_stem(found)} unless another unit-night's copy exists first"
        )
        defects.append(("binned-copy", template, detail))
    rule = "own-night" if os.path.dirname(path) == os.path.dirname(template) else "fetched-nearest"
    for flag, label in (
        (flags.ignored_sanity, "sanity-ignored"),
        (flags.ignored_lenient_keys, "lenient-keys"),
        (flags.ignored_binning, "binning-relaxed"),
        (flags.ignored_masterframe_walls, "walls-ignored"),
    ):
        rule += f"+{label}" if flag else ""
    rec = library.record(found)
    if flags.ignored_sanity:
        defects.append(("rejected-used", template, f"{_stem(found)}: {'human' if rec['human'] else 'machine'}"))
    return {
        "path": path,
        "source": found,
        "rule": rule,
        "defects": defects + _master_checks(library, template, found, rec),
    }


def _master_checks(library: _Library, template: str, path: str, rec: dict) -> list:
    """Checks of a selected master itself: existing binned copy, raw budget, registry consistency."""
    defects = []
    if source := _copy_source(library, path):
        detail = f"{_stem(path)}: existing copy of {_stem(source)}, not rebuilt when its source is regenerated"
        defects.append(("binned-copy", template, detail))
    if _state(rec) == "registry-only":
        defects.append(("no-raw-budget", template, f"{_stem(path)}: {rec['n_raw']} raw frames"))
    return defects + [("registry-inconsistent", template, issue) for issue in _registry_issues(library, path, rec)]


def _copy_source(library: _Library, path: str) -> Optional[str]:
    """Source flat of an existing auto-binned copy: its only recorded edge, else its finest registered sibling."""
    stem = _stem(path)
    if stem in library.copies:
        return library.copies[stem]
    if stem not in library.sibling_sources:
        library.sibling_sources[stem] = _sibling_source(library, path)
    return library.sibling_sources[stem]


def _sibling_source(library: _Library, path: str) -> Optional[str]:
    """Finest same-directory registered flat of a dividing binning, for a registered flat without raw budget."""
    registered, rec = library.registered_path.get(_stem(path)), library.record(path)
    if registered is None or rec is None or rec["n_raw"] >= NUM_MIN_CALIB:
        return None
    name = NameHandler(registered)
    nbin = name.n_binning
    if name.type.exposure_type != CALIB_TYPE_FLAT or not isinstance(nbin, int):
        return None
    directory = os.path.dirname(registered)
    for finer in range(1, nbin):
        name.n_binning = finer
        if nbin % finer == 0 and name.mflat_basename in library.registry.get(directory, {}):
            return os.path.join(directory, name.mflat_basename)
    return None


def _registry_issues(library: _Library, path: str, rec: dict) -> List[str]:
    """Where the registry contradicts itself or the raw DB for a selected master."""
    issues = []
    if rec["registered"] and rec["sanity"] is None:
        issues.append(f"{_stem(path)}: sanity NULL")
    registered = library.registered_path.get(_stem(path))
    canonical = registered and _canonical([os.path.basename(path)])[0]
    if registered != canonical:
        issues.append(f"{_stem(path)}: registered at {registered}, canonical {canonical}")
    if _state(rec) == "registry-only":
        name = NameHandler(path)
        typ = name.type.exposure_type
        quality = {CALIB_TYPE_DARK: format_exptime(name.exptime, type="dark"), CALIB_TYPE_FLAT: name.filter}.get(typ)
        night, unit = rec["nightdate"], rec["unit"]
        others = {
            _stem(_master_basename(t, q, unit, night, b, g, library.camera(unit, night)))
            for t, q, b, g, _ in library.raw.get((night, unit), [])
            if (t, q, b) == (typ, quality, name.n_binning)
        } - {_stem(path)}
        if others:
            issues.append(f"{_stem(path)}: raw DB keys name it {', '.join(sorted(others))}")
    return issues


def _answer(library: _Library, template: str, dtype: str, owner: Tuple[str, str], role: str) -> dict:
    """The owner yml's designation for the role, else the search under the base settings."""
    designated = _designated(library, owner, role, template)
    return designated or _search(library, template, dtype, library.base)


def _select(library: _Library, template: str, dtype: str, owner: Tuple[str, str], role: str) -> dict:
    """Answer for one role; stale-config when the recorded settings would answer a search otherwise."""
    answer = _answer(library, template, dtype, owner, role)
    settings = _settings_of(library, *owner)
    if settings != library.base and answer["rule"] != "designated":
        yml = _search(library, template, dtype, settings)
        if (yml["path"], yml["rule"]) != (answer["path"], answer["rule"]):
            yml_answer = f"{yml['path'] and _stem(yml['path'])} ({yml['rule']})"
            base_answer = f"{answer['path'] and _stem(answer['path'])} ({answer['rule']})"
            detail = f"yml {settings}: {yml_answer}; base {library.base}: {base_answer}"
            answer = dict(answer, defects=answer["defects"] + [("stale-config", template, detail)])
    return answer


def _ingredients(library: _Library, path: str) -> Dict[str, dict]:
    """bias (and flatdark for a flat) of a master: searched at its unit-night if rebuilt, else recorded."""
    if path not in library.ingredients:
        rec = library.record(path)
        is_flat = NameHandler(path).type.exposure_type == CALIB_TYPE_FLAT
        source = is_flat and not (rec and rec["regen"]) and _copy_source(library, path)
        if source:
            library.ingredients[path] = _ingredients(library, source)
        elif rec and rec["regen"]:
            _resolve(library, path)
        else:
            library.ingredients[path] = _recorded(library, path, is_flat)
    return library.ingredients[path]


def _planned(library: _Library, path: str, owner: Tuple[str, str], is_flat: bool) -> Dict[str, dict]:
    """bias (and flatdark for a flat) the searches at a master's unit-night select."""
    handler = PathHandler(path)
    roles = {"bias": _select(library, handler.preprocess._mbias[0], CALIB_TYPE_BIAS, owner, CALIB_TYPE_BIAS)}
    if is_flat:
        handler.name.exptime = "*"
        roles["flatdark"] = _select(library, handler.preprocess._mdark[0], CALIB_TYPE_DARK, owner, "flatdark")
    return roles


def _edges(library: _Library, path: str) -> list:
    """(dependency_role, source path, recorded source IMAGEID, registry IMAGEID) of a master's edges, memoized."""
    stem = _stem(path)
    if stem not in library.edges:
        roles = [BIAS_DEPENDENCY_ROLE, DARK_DEPENDENCY_ROLE, FLAT_DEPENDENCY_ROLE]
        library.edges[stem] = free_query(_EDGES, [stem, roles])
    return library.edges[stem]


def _recorded(library: _Library, path: str, is_flat: bool) -> Dict[str, dict]:
    """Ingredients of a master that is not regenerated, from its image_qa_dependency edges."""
    dependency_roles = [BIAS_DEPENDENCY_ROLE, DARK_DEPENDENCY_ROLE, FLAT_DEPENDENCY_ROLE]
    edges = defaultdict(list)
    for dependency_role, source, *_ in _edges(library, path):
        edges[dependency_role].append(source)
    bias, dark, flat = (edges[r] for r in dependency_roles)
    if is_flat and not bias and not dark and len(flat) == 1:  # an auto-binned copy records its source flat alone
        return _ingredients(library, flat[0])
    wanted = {"bias": (BIAS_DEPENDENCY_ROLE, bias)}
    if is_flat:
        wanted["flatdark"] = (DARK_DEPENDENCY_ROLE, dark)
    answers = {}
    for role, (dependency_role, sources) in wanted.items():
        if len(sources) == 1:
            source = library.record(sources[0])
            late = library.as_of is not None and source is not None and not source["visible"]
            issue = f"{_stem(path)}: recorded {role} {_stem(sources[0])} registered after as_of"
            defects = [("registry-inconsistent", path, issue)] if late else []
            answers[role] = {"path": sources[0], "source": sources[0], "rule": "recorded", "defects": defects}
        else:
            issue = f"{_stem(path)}: {len(sources)} recorded {dependency_role} edges"
            answers[role] = _none([("registry-inconsistent", path, issue)])
    return answers


def _role_columns(library: _Library, role: str, answer: Optional[dict]) -> dict:
    """Nine columns of one role; the offset is filled by _offsets."""
    rec = library.record(answer["source"]) if answer and answer["source"] else None
    if rec and not rec["resolved"]:
        _resolve(library, answer["source"])
    return {
        role: _stem(answer["path"]) if answer and answer["path"] else None,
        f"{role}_path": answer and answer["path"],
        f"{role}_nightdate": rec and rec["nightdate"],
        f"{role}_unit": rec and rec["unit"],
        f"{role}_offset": None,
        f"{role}_rule": answer and answer["rule"],
        f"{role}_state": rec and _state(rec),
        f"{role}_sanity": rec and _sanity(rec),
        f"{role}_rebuilt": rec and rec["rebuilt"],
    }


def _offsets(columns: dict, night: str, consumers: Dict[str, Optional[str]]) -> dict:
    """Nights from each role's consumer (the frame's night, or the master role named) to its selected master."""
    offsets = {}
    for role, consumer in consumers.items():
        base, own = columns.get(f"{consumer}_nightdate") if consumer else night, columns.get(f"{role}_nightdate")
        offsets[f"{role}_offset"] = (date.fromisoformat(own) - date.fromisoformat(base)).days if own and base else None
    return offsets


def _templates(night: str, unit: str, nbin, gain, camera: str, exptime, filt) -> List[str]:
    """Bias, dark and flat templates of a raw science group."""
    qualities = [None, format_exptime(float(exptime), type="dark"), filt]
    dtypes = [CALIB_TYPE_BIAS, CALIB_TYPE_DARK, CALIB_TYPE_FLAT]
    return _canonical([_master_basename(d, q, unit, night, nbin, gain, camera) for d, q in zip(dtypes, qualities)])


def _frame_roles(library: _Library, night: str, unit: str, templates: List[str]) -> Dict[str, dict]:
    """The seven roles of a raw science group."""
    dtypes = [CALIB_TYPE_BIAS, CALIB_TYPE_DARK, CALIB_TYPE_FLAT]
    roles = {d: _select(library, t, d, (night, unit), d) for d, t in zip(dtypes, templates)}
    dark, flat = roles[CALIB_TYPE_DARK], roles[CALIB_TYPE_FLAT]
    dark = _ingredients(library, dark["source"]) if dark["source"] else {}
    if flat["rule"] == "designated":
        flat = _recorded(library, flat["source"], True)
    else:
        flat = _ingredients(library, flat["source"]) if flat["source"] else {}
    flatdark = flat.get("flatdark")
    flatdark_ingredients = _ingredients(library, flatdark["source"]) if flatdark and flatdark["source"] else {}
    roles.update(flatdark=flatdark, dark_bias=dark.get("bias"), flat_bias=flat.get("bias"))
    roles["flatdark_bias"] = flatdark_ingredients.get("bias")
    return roles


def _science_frames(where: str, params: list) -> List[tuple]:
    """(raw_path, nightdate, unit, obj, filter, exptime, n_binning, gain) of raw science frames."""
    return free_query(_SCIENCE_FRAMES.format(where), params)


def _nightdate(value) -> str:
    """YYYY-MM-DD of a date, datetime, Timestamp, or YYYYMMDD / YYYY-MM-DD string."""
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    text = str(value)
    return f"{text[:4]}-{text[4:6]}-{text[6:]}" if len(text) == 8 and text.isdigit() else text


def _resolve_targets(targets) -> Tuple[List[tuple], List[tuple]]:
    """Raw science frames of the targets, and the unknown-frame defects."""
    items = [targets] if isinstance(targets, (str, os.PathLike, tuple)) else list(targets)
    pairs, paths, configs = [], [], []
    for item in items:
        if isinstance(item, tuple):
            pairs.append((_nightdate(item[0]), str(item[1])))
        elif str(item).endswith((".yml", ".yaml")):
            configs.append(str(item))
        else:
            paths.append(str(item))
    rows, unknown = [], []
    obj = "COALESCE(tl.name, t.name, NULLIF(x.object_name, ''))"
    known = {r[0]: r for r in free_query(_CONFIGS, [configs])} if configs else {}
    for config in configs:
        _, config_type, night, unit, target, filt = known.get(config, (None,) * 6)
        if config_type == CONFIG_TYPE_PREPROCESS:
            pairs.append((night, unit))
        elif config_type == CONFIG_TYPE_SCIENCE:
            where, params = f"{obj} = %s AND f.name = %s", [target, filt]
            if night is not None:
                where, params = where + " AND n.date = %s::date", params + [night]
            found = _science_frames(where, params)
            rows += found
            if not found:
                detail = f"no raw science frame for object {target}, filter {filt}, night {night}"
                unknown.append((night, None, None, config, "unknown-frame", detail, 0))
        else:
            detail = f"config_type {config_type}" if config in known else "not in process_status"
            unknown.append((None, None, None, config, "unknown-frame", detail, 0))
    if pairs:
        nights, units = [n for n, _ in pairs], [u for _, u in pairs]
        where = "(n.date, u.name) IN (SELECT * FROM unnest(%s::date[], %s::text[]))"
        found = _science_frames(where, [nights, units])
        seen = {(r[1], r[2]) for r in found}
        rows += found
        for night, unit in dict.fromkeys(pairs):
            if (night, unit) not in seen:
                unknown.append((night, unit, None, None, "unknown-frame", "no science frame in the raw DB", 0))
    if paths:
        found = _science_frames("x.file_path = ANY(%s)", [paths])
        seen = {r[0] for r in found}
        rows += found
        unknown += [
            (None, None, None, p, "unknown-frame", "not in survey_scienceframe", 1) for p in paths if p not in seen
        ]
    return sorted({r[0]: r for r in rows}.values(), key=lambda r: (r[1], r[2], r[0])), unknown


def _defect_table(tally: Counter) -> pd.DataFrame:
    """Defect rows from (nightdate, unit, role, template, code, detail) counts."""
    columns = ["nightdate", "unit", "role", "template", "code", "detail", "n_frames"]
    return pd.DataFrame([(*key, n) for key, n in tally.items()], columns=columns)


def _flat_source(library: _Library, answer: Optional[dict]) -> Optional[str]:
    """Stem of the source flat when the flat answer is an auto-binned copy, existing or not yet made."""
    if not answer or not answer["path"]:
        return None
    if answer["source"] != answer["path"]:
        return _stem(answer["source"])
    source = _copy_source(library, answer["path"])
    return source and _stem(source)


def _count_defects(tally: Counter, night: str, unit: str, role: str, answer: Optional[dict], n: int) -> None:
    """Add the defects of one role answer, weighted by the rows it serves."""
    for code, template, detail in answer["defects"] if answer else []:
        tally[(night, unit, role, template, code, detail)] += n


def _count_unmodeled(library: _Library, tally: Counter, night: str, unit: str, n: int) -> None:
    """Add the unmodeled requests of a unit-night's preprocess row."""
    row = _row_of(library, night, unit)
    for issue in _unmodeled(row):
        tally[(night, unit, None, row.config_file, "unmodeled", issue)] += n


def _count_unrecorded(library: _Library, tally: Counter, counts: Counter) -> None:
    """One row per unit-night searched with the base settings because its row records none; n = its target rows."""
    for night, unit in sorted(library.unrecorded):
        detail = f"max_offset / ignore_*_if_no_match not recorded in process_status; base {library.base} used"
        config = _row_of(library, night, unit).config_file
        tally[(night, unit, None, config, "settings-unrecorded", detail)] += counts.get((night, unit), 0)


@contextmanager
def _changelog_cached():
    """Each unit's changelog and wall nightdates read once while a run lasts."""
    read, walls = instrum_log.get_changelog, instrum_log.get_wall_nightdates
    if hasattr(read, "cache_info"):  # a run in progress already caches
        yield
        return
    instrum_log.get_changelog = lru_cache(maxsize=None)(read)
    instrum_log.get_wall_nightdates = lru_cache(maxsize=None)(walls)
    try:
        yield
    finally:
        instrum_log.get_changelog, instrum_log.get_wall_nightdates = read, walls


@_changelog_cached()
def associate(
    targets,
    *,
    trust_machine_rejection: bool = True,
    as_of: Optional[datetime] = None,
    veto: Iterable[str] = (),
    units: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Masters each raw science frame will be reduced with, and the cases the rules cannot settle: (table, defects); `veto` plans as if those masters were human-rejected."""
    roles = ["bias", "dark", "flat", "flatdark", "dark_bias", "flat_bias", "flatdark_bias"]
    consumers = dict(zip(roles, [None, None, None, "flat", "dark", "flat", "flatdark"]))
    frames, unknown = _resolve_targets(targets)
    library = _Library(trust_machine_rejection, as_of, {(r[1], r[2]) for r in frames}, veto, units)
    groups, failures = defaultdict(list), {}
    for i, (_, night, unit, _, filt, exptime, nbin, gain) in enumerate(frames):
        try:
            camera = library.camera(unit, night)
        except ValueError as e:
            camera, failures[(night, unit)] = None, f"camera: {type(e).__name__}: {e}"
        groups[(night, unit, nbin, gain, camera, exptime, filt)].append(i)
    rows = [None] * len(frames)
    tally = Counter()
    for key, members in groups.items():
        failure = failures.get((key[0], key[1])) if key[4] is None else None
        if failure is None:
            try:
                templates = _templates(*key)
            except (ValueError, TypeError) as e:
                failure = f"templates: {type(e).__name__}: {e}"
        if failure is None:
            answers = _frame_roles(library, key[0], key[1], templates)
        else:
            answers = dict.fromkeys(roles)
            answers.update({d: _none([]) for d in (CALIB_TYPE_BIAS, CALIB_TYPE_DARK, CALIB_TYPE_FLAT)})
            tally[(key[0], key[1], None, None, "search-error", failure)] += len(members)
        columns = {}
        for role in roles:
            columns.update(_role_columns(library, role, answers[role]))
            if role == CALIB_TYPE_FLAT:
                columns["flat_source"] = _flat_source(library, answers[role])
            _count_defects(tally, key[0], key[1], role, answers[role], len(members))
        columns.update(_offsets(columns, key[0], consumers))
        for i in members:
            raw_path, night, unit, obj, filt, exptime, nbin, gain = frames[i]
            frame = dict(raw_path=raw_path, nightdate=night, unit=unit, obj=obj, filter=filt, exptime=exptime)
            rows[i] = {**frame, "n_binning": nbin, "gain": gain, "camera": key[4], **columns}
    for (night, unit), n in Counter((r[1], r[2]) for r in frames).items():
        _count_unmodeled(library, tally, night, unit, n)
    _count_unrecorded(library, tally, Counter((r[1], r[2]) for r in frames))
    for night, unit, role, template, code, detail, n in unknown:
        tally[(night, unit, role, template, code, detail)] += n
    frame_columns = ["raw_path", "nightdate", "unit", "obj", "filter", "exptime", "n_binning", "gain", "camera"]
    columns = frame_columns + [c for role in roles for c in _role_columns(library, role, None)]
    columns.insert(columns.index("flat_state") + 1, "flat_source")
    return pd.DataFrame(rows, columns=columns), _defect_table(tally)


@_changelog_cached()
def associate_masters(
    masters: Iterable[str],
    *,
    trust_machine_rejection: bool = True,
    as_of: Optional[datetime] = None,
    veto: Iterable[str] = (),
    units: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Masters each master dark or flat will be built with (bias; flatdark for a flat), and the unsettled cases."""
    items = [str(m) for m in ([masters] if isinstance(masters, (str, os.PathLike)) else masters)]
    names = {}
    for item in items:
        try:
            name = NameHandler(item)
        except ValueError:
            name = None
        is_master = name is not None and name.type.kind == NAME_TYPE_MASTER
        names[item] = name if is_master and name.type.exposure_type in MASTER_IMAGE_TYPES else None
    library = _Library(
        trust_machine_rejection, as_of, {(n.nightdate, n.unit) for n in names.values() if n}, veto, units
    )
    rows, tally = [], Counter()
    for item in items:
        path, detail = (item if os.path.dirname(item) else None), "not a master in the library or registry"
        if names[item] and path is None:
            try:
                canonical = _canonical([f"{_stem(item)}.fits"])[0]
            except ValueError as e:
                canonical, detail = None, f"no canonical path: {type(e).__name__}: {e}"
            registered = library.registered_path.get(_stem(item))
            path = canonical if canonical and library.record(canonical) else registered or canonical
        rec = library.record(path) if names[item] and path else None
        if rec is None:
            tally[(None, None, None, item, "unknown-frame", detail)] += 1
            continue
        typ = names[item].type.exposure_type
        answers = _ingredients(library, path) if typ != CALIB_TYPE_BIAS else {}
        row = dict(master=_stem(path), master_path=path, type=typ, nightdate=rec["nightdate"], unit=rec["unit"])
        row.update(state=_state(rec), n_raw=rec["n_raw"], sanity=_sanity(rec), rebuilt=rec["rebuilt"])
        for role in ("bias", "flatdark"):
            row.update(_role_columns(library, role, answers.get(role)))
            _count_defects(tally, rec["nightdate"], rec["unit"], role, answers.get(role), 1)
        row.update(_offsets(row, rec["nightdate"], {"bias": None, "flatdark": None}))
        _count_unmodeled(library, tally, rec["nightdate"], rec["unit"], 1)
        rows.append(row)
    _count_unrecorded(library, tally, Counter((r["nightdate"], r["unit"]) for r in rows))
    columns = ["master", "master_path", "type", "nightdate", "unit", "state", "n_raw", "sanity", "rebuilt"]
    columns += [c for role in ("bias", "flatdark") for c in _role_columns(library, role, None)]
    return pd.DataFrame(rows, columns=columns), _defect_table(tally)


@_changelog_cached()
def veto_impact(masters: Iterable[str], full: bool = False) -> pd.DataFrame:
    """Selections a human rejection of `masters` changes: master in use, alternative, offsets."""
    roles = ["bias", "dark", "flat", "flatdark", "dark_bias", "flat_bias", "flatdark_bias"]
    consumers = dict(zip(roles, [None, None, None, "flat", "dark", "flat", "flatdark"]))
    masters = [masters] if isinstance(masters, (str, os.PathLike)) else list(masters)
    vetoed = {_stem(str(m)) for m in masters}
    units = sorted({NameHandler(f"{stem}.fits").unit for stem in vetoed})
    groups, stored = plan_groups(units), plan_masters(units, allow_stale=True)
    copies = groups[groups.master.isin(vetoed - set(stored.master)) & groups.flat_source.notna()]
    vetoed = (vetoed - set(copies.master)) | set(copies.flat_source)
    masters = sorted(vetoed)
    if full:
        first, last = free_query(_SCIENCE_NIGHT_RANGE, [units])[0]
        targets = sorted(
            {(str(first + timedelta(days=k)), unit) for unit in units for k in range((last - first).days + 1)}
        )
        before, _ = associate(targets)
    else:
        in_use = groups[groups.master.isin(vetoed) | groups.flat_source.isin(vetoed)]
        targets = sorted({(str(night), unit) for night, unit in zip(in_use.nightdate, in_use.unit)})
        before = planned(targets, allow_stale=True)
    after, _ = associate(targets, veto=masters, units=None if full else units)
    shown = ["master", "offset", "sanity", "rule"]
    parts = []
    for role in roles:
        changed = (before[role].fillna("") != after[role].fillna("")) | after[role].isin(vetoed)
        columns = [role, f"{role}_offset", f"{role}_sanity", f"{role}_rule"]
        used = before.loc[changed, ["nightdate", "unit"] + columns].set_axis(["nightdate", "unit"] + shown, axis=1)
        other = after.loc[changed, columns].set_axis([f"alternative_{c}" for c in shown], axis=1)
        consumer = consumers[role]
        part = pd.concat([used, other], axis=1).assign(
            role=role,
            consumer=before.loc[changed, consumer] if consumer else None,
            alternative_consumer=after.loc[changed, consumer] if consumer else None,
        )
        parts.append(part)
    table = pd.concat(parts)
    keys = (
        ["nightdate", "unit", "role", "consumer"]
        + shown
        + ["alternative_consumer"]
        + [f"alternative_{c}" for c in shown]
    )
    table = table.groupby(keys, dropna=False, sort=True).size().reset_index(name="n_frames")
    if full:
        owned = free_query(
            _UNIT_MASTERS,
            [IMAGE_GROUP_MASTERFRAME, [CALIB_TYPE_DARK, CALIB_TYPE_FLAT], [f"%/{unit}/%" for unit in units]],
        )
        used, _ = associate_masters([name for name, in owned])
    else:
        key = ["unit", "nightdate", "filter", "gain"]
        flats = stored[stored.master.isin(vetoed) & (stored.image_type == CALIB_TYPE_FLAT)]
        flat_keys = set(flats[key].itertuples(index=False, name=None))
        same_key = pd.Series(
            [k in flat_keys for k in stored[key].itertuples(index=False, name=None)], index=stored.index
        )
        users = stored[
            stored.image_type.isin([CALIB_TYPE_DARK, CALIB_TYPE_FLAT])
            & (stored.state != "budget-only")
            & (stored.master.isin(vetoed) | stored.bias.isin(vetoed) | stored.flatdark.isin(vetoed) | same_key)
        ]
        offsets = {"image_type": "type", "bias_night_offset": "bias_offset", "flatdark_night_offset": "flatdark_offset"}
        used = users.rename(columns=offsets).assign(nightdate=users.nightdate.astype(str))
    other, _ = associate_masters(list(used.master), veto=masters, units=None if full else units)
    both = used.merge(other, on="master", suffixes=("", "_alternative"))
    seen = {
        "bias": set(table.consumer[table.role.isin(["dark_bias", "flat_bias", "flatdark_bias"])]),
        "flatdark": set(table.consumer[table.role == "flatdark"]),
    }
    rows = []
    for role in ("bias", "flatdark"):
        now, then = both[role].fillna(""), both[f"{role}_alternative"].fillna("")
        for r in both[((now != then) | then.isin(vetoed)) & ~both.master.isin(seen[role])].itertuples(index=False):
            r = r._asdict()
            row = dict(nightdate=r["nightdate"], unit=r["unit"], consumer=r["master"], alternative_consumer=r["master"])
            row.update(role=f"{r['type']}_bias" if role == "bias" else role, n_frames=0)
            row.update({c: r[role if c == "master" else f"{role}_{c}"] for c in shown})
            row.update(
                {
                    f"alternative_{c}": r[f"{role}_alternative" if c == "master" else f"{role}_{c}_alternative"]
                    for c in shown
                }
            )
            rows.append(row)
    return pd.concat([table, pd.DataFrame(rows, columns=table.columns)], ignore_index=True)


def neighbor_masters(master: str, days: int = 30) -> pd.DataFrame:
    """Masters with the keys of `master` on the nights around it: raw frames in the raw DB, and the registered master's verdict."""
    name = NameHandler(f"{_stem(str(master))}.fits")
    typ, unit = name.type.exposure_type, name.unit
    quality = name.filter if typ == CALIB_TYPE_FLAT else None
    if typ == CALIB_TYPE_DARK:
        quality = format_exptime(float(name.exptime), type="dark")
    keys = (typ, quality, name.n_binning, name.gain)
    raw = Counter()
    for t, _, night, nbin, gain, exptime, filt, n in free_query(_RAW_BUDGET, [[unit]] * 3):
        if (t, format_exptime(float(exptime), type="dark") if t == CALIB_TYPE_DARK else filt, nbin, gain) == keys:
            raw[night] += n
    stored = plan_masters([unit])
    registry = {row.master: row for row in stored.itertuples(index=False) if row.state != "budget-only"}
    first = date.fromisoformat(str(name.nightdate))
    columns = ["master", "nightdate", "offset", "n_raw", "state", "pipe_ver", "sanity", "inspcomm"]
    rows = []
    for offset in range(-days, days + 1):
        night = str(first + timedelta(days=offset))
        basename = _master_basename(*keys[:2], unit, night, *keys[2:], name.camera)
        row, n_raw = registry.get(_stem(basename)), raw[night]
        if row is None and n_raw:
            camera = format_camera(get_camera_serial(unit=int(unit[3:]), query_date=night))
            n_raw = n_raw if basename == _master_basename(*keys[:2], unit, night, *keys[2:], camera) else 0
        if not n_raw and row is None:
            continue
        if row is not None:
            state = "registered" if n_raw >= NUM_MIN_CALIB else "registry-only"
        else:
            state = "budget-only" if n_raw >= NUM_MIN_CALIB else "too few raw frames"
        verdict = [row.pipe_ver, row.registry_sanity, row.human] if row is not None else [None] * 3
        rows.append(dict(zip(columns, [_stem(basename), night, offset, n_raw, state, *verdict])))
    return pd.DataFrame(rows, columns=columns)


def library_masters(unit: str) -> pd.DataFrame:
    """Every master of a unit's library, biases included: keys, raw budget, registry verdict."""
    library = _Library(True, None, set(), units=[unit])
    rows = {}
    for night in sorted({n for n, _ in library.raw} | {n for n, _ in library.registry_dirs}):
        groups = library.groups(night, unit)
        directories = [os.path.dirname(_canonical([next(iter(groups))])[0])] if groups else []
        for directory in directories + sorted(library.registry_dirs.get((night, unit), ())):
            for basename, rec in library.entries(directory)[0].items():
                stem, name = _stem(basename), NameHandler(basename)
                if stem in rows:
                    continue
                registered = rec["registered"] and library.registered_path[stem]
                pipe_ver = library.registry[os.path.dirname(registered)][basename][4] if registered else None
                row = dict(master=stem, unit=unit, master_path=os.path.join(directory, basename), nightdate=night)
                row.update(image_type=name.type.exposure_type, n_binning=name.n_binning, gain=name.gain)
                row.update(camera=name.camera, exptime=name.exptime, filter=name.filter, n_raw=rec["n_raw"])
                row.update(state=_state(rec), registry_sanity=rec["sanity"], human=rec["human"], pipe_ver=pipe_ver)
                rows[stem] = row
    return pd.DataFrame(list(rows.values()))


def planned(unit_nights, allow_stale: bool = False) -> pd.DataFrame:
    """associate's table from the stored plan: frames expanded from their groups."""
    roles = ["bias", "dark", "flat", "flatdark", "dark_bias", "flat_bias", "flatdark_bias"]
    stored_columns = ["master", "master_path", "master_nightdate", "master_unit", "night_offset", "rule", "state"]
    stored_columns += ["sanity", "rebuilt"]
    frames, _ = _resolve_targets(unit_nights)
    groups = defaultdict(dict)
    stored = plan_groups(sorted({f[2] for f in frames}), sorted({f[1] for f in frames}), allow_stale=allow_stale)
    for row in stored.itertuples(index=False):
        groups[(str(row.nightdate), row.unit, row.n_binning, row.gain, row.exptime, row.filter)][row.role] = row
    rows = []
    for raw_path, night, unit, obj, filt, exptime, nbin, gain in frames:
        group = groups[(night, unit, nbin, gain, exptime, filt)]
        frame = dict(raw_path=raw_path, nightdate=night, unit=unit, obj=obj, filter=filt, exptime=exptime)
        row = {**frame, "n_binning": nbin, "gain": gain, "camera": group["bias"].camera}
        for role in roles:
            values = [getattr(group[role], c) for c in stored_columns]
            values[2] = values[2] and str(values[2])
            row.update(zip(_role_columns(None, role, None), values))
        row["flat_source"] = group["flat"].flat_source
        rows.append(row)
    frame_columns = ["raw_path", "nightdate", "unit", "obj", "filter", "exptime", "n_binning", "gain", "camera"]
    columns = frame_columns + [c for role in roles for c in _role_columns(None, role, None)]
    columns.insert(columns.index("flat_state") + 1, "flat_source")
    return pd.DataFrame(rows, columns=columns)


def refresh_plan(units: Optional[Iterable[str]] = None, dry_run: bool = False) -> pd.DataFrame:
    """Store each unit's plan, whole units; None: units never refreshed or stale."""
    roles = ["bias", "dark", "flat", "flatdark", "dark_bias", "flat_bias", "flatdark_bias"]
    stored_columns = ["master", "master_path", "master_nightdate", "master_unit", "night_offset", "rule", "state"]
    stored_columns += ["sanity", "rebuilt"]
    master_columns = ["sanity", "rebuilt"] + [f"{r}{c}" for r in ("bias", "flatdark") for c in ("", "_offset")]
    master_columns += [f"{r}_{c}" for r in ("bias", "flatdark") for c in ("rule", "sanity", "rebuilt")]
    if units is None:
        status = plan_status()
        units = list(status.unit[status.stale])
    report = []
    for unit in [units] if isinstance(units, str) else units:
        start = time.time()
        with psycopg.connect(**DB_PARAMS, autocommit=True) as conn:
            lock = conn.execute("SELECT pg_try_advisory_lock(hashtext(%s))", [f"masterframe_plan {unit}"])
            if not lock.fetchone()[0]:
                report.append((unit, "another refresh of this unit is running: nothing done", 0, 0, 0.0))
                continue
            edges = [BIAS_DEPENDENCY_ROLE, DARK_DEPENDENCY_ROLE, FLAT_DEPENDENCY_ROLE]
            unit_dir = f"%/{unit}/%"
            foreign = free_query(_FOREIGN_EDGES, [IMAGE_GROUP_MASTERFRAME, unit_dir, unit_dir, edges])
            masterframe = [IMAGE_GROUP_MASTERFRAME, list(MASTER_IMAGE_TYPES)]
            foreign += free_query(_FOREIGN_MASTERS, [*masterframe, unit_dir, f"%{unit}%"])
            foreign += free_query(_FOREIGN_DESIGNATIONS, [CONFIG_TYPE_PREPROCESS, f"%\\_{unit}", f"%\\_{unit}\\_%"])
            if any(n for n, in foreign):
                message = f"cross-unit edges / registry rows / designations {[n for n, in foreign]}: nothing done"
                report.append((unit, message, 0, 0, time.time() - start))
                continue
            fingerprint, refreshed_at = fingerprints([unit])[unit], datetime.now(timezone.utc)
            table, _ = associate([(night, unit) for night, in free_query(_SCIENCE_NIGHTS, [unit])], units=[unit])
            key = ["unit", "nightdate", "n_binning", "gain", "exptime", "filter"]
            answers = table.drop(columns=["raw_path", "obj"]).drop_duplicates()
            if answers.duplicated(key).any():
                raise ValueError(f"{unit}: a group of science frames with more than one answer")
            answers = answers.merge(table.groupby(key, dropna=False).size().rename("n_frames").reset_index(), on=key)
            parts = []
            for role in roles:
                names = dict(zip(_role_columns(None, role, None), stored_columns))
                part = answers[key + ["camera", "n_frames", *names]].rename(columns=names)
                parts.append(part.assign(role=role, flat_source=answers.flat_source if role == "flat" else None))
            groups = pd.concat(parts, ignore_index=True).astype({"night_offset": "Int64"})
            library = library_masters(unit)
            plan, _ = associate_masters(list(library.master_path), units=[unit])
            offsets = {"bias_offset": "bias_night_offset", "flatdark_offset": "flatdark_night_offset"}
            masters = library.merge(plan[["master"] + master_columns], on="master", how="left").rename(columns=offsets)
            masters = masters.astype({c: "Int64" for c in offsets.values()})
            masters = masters.assign(refreshed_at=refreshed_at, fingerprint=fingerprint)
            if fingerprints([unit])[unit] != fingerprint:
                report.append((unit, "inputs moved during the computation: nothing written", 0, 0, time.time() - start))
                continue
            if not dry_run:
                replace_plan(conn, unit, groups, masters)
            message = "dry run: nothing written" if dry_run else "refreshed"
            report.append((unit, message, len(groups), len(masters), time.time() - start))
    return pd.DataFrame(report, columns=["unit", "message", "group_rows", "master_rows", "seconds"])
