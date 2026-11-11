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
import traceback
from collections import Counter, defaultdict
from datetime import date, datetime
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
)
from ..path.cam_tracker import get_camera_serial
from ..path.name import NameHandler
from ..path.path import PathHandler
from ..path.utils import add_a_day, format_camera, format_exptime, get_nightdate
from ..utils import atleast_1d
from ..version import MIN_PREPROC_RUNTIME_VERSION, is_below_min
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
GROUP BY 2, 3, 4, 5
UNION ALL
SELECT 'dark', u.name, n.date::text, x.binning_x, x.gain, x.exptime, NULL::text, count(*)
FROM survey_darkframe x JOIN facility_unit u ON u.id = x.unit_id JOIN survey_night n ON n.id = x.night_id
GROUP BY 2, 3, 4, 5, 6
UNION ALL
SELECT 'flat', u.name, n.date::text, x.binning_x, x.gain, NULL::float8, f.name, count(*)
FROM survey_flatframe x JOIN facility_unit u ON u.id = x.unit_id JOIN survey_night n ON n.id = x.night_id
LEFT JOIN facility_filter f ON f.id = x.filter_id
GROUP BY 2, 3, 4, 5, 7
"""

_RAW_FILES = """
SELECT 'bias', x.file_path, x.binning_x, x.gain, NULL::float8, NULL::text
FROM survey_biasframe x JOIN facility_unit u ON u.id = x.unit_id JOIN survey_night n ON n.id = x.night_id
WHERE u.name = %(unit)s AND n.date = %(night)s::date
UNION ALL
SELECT 'dark', x.file_path, x.binning_x, x.gain, x.exptime, NULL::text
FROM survey_darkframe x JOIN facility_unit u ON u.id = x.unit_id JOIN survey_night n ON n.id = x.night_id
WHERE u.name = %(unit)s AND n.date = %(night)s::date
UNION ALL
SELECT 'flat', x.file_path, x.binning_x, x.gain, NULL::float8, f.name
FROM survey_flatframe x JOIN facility_unit u ON u.id = x.unit_id JOIN survey_night n ON n.id = x.night_id
LEFT JOIN facility_filter f ON f.id = x.filter_id
WHERE u.name = %(unit)s AND n.date = %(night)s::date
"""

_REGISTRY = """
SELECT image_name, image_path, sanity,
       NULLIF(btrim(inspcomm), '') IS NOT NULL AND inspcomm <> 'Auto-generated binned master frame',
       COALESCE(created_at <= %s, TRUE), image_type, pipe_ver,
       (SELECT jsonb_object_agg(upper(k), v) FROM jsonb_each(to_jsonb(q)) e(k, v) WHERE upper(k) = ANY(%s))
FROM image_qa q WHERE image_group = %s AND image_type = ANY(%s) AND image_path IS NOT NULL
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
SELECT DISTINCT ON (name) name, config_file FROM process_status WHERE config_type = %s
ORDER BY name, updated_at DESC NULLS LAST, id DESC
"""


class _Unresolved(Exception):
    """A probe met a regenerated master whose modeled state is not resolved yet."""


class _Library:
    """Masters of the modeled state, probed by tolerant_search through glob / exists / sanity_check."""

    def __init__(self, trust_machine_rejection: bool, as_of: Optional[datetime], targets: set):
        from .checker import Checker

        self.trust_machine_rejection = trust_machine_rejection
        self.as_of = as_of
        self.targets = targets
        self.checker = Checker()
        self.checker.logger = logging.getLogger(__name__)
        criteria = self.checker.load_qa_criteria(IMAGE_GROUP_MASTERFRAME)
        self.raw = defaultdict(list)
        for typ, unit, night, nbin, gain, exptime, filt, n in free_query(_RAW_BUDGET):
            quality = format_exptime(float(exptime), type="dark") if typ == CALIB_TYPE_DARK else filt
            self.raw[(night, unit)].append((typ, quality, nbin, gain, n))
        self.registry = defaultdict(dict)
        self.registry_dirs = defaultdict(set)
        self.registered_path = {}
        statistics = sorted({key for keys in criteria.values() for key in keys})
        for name, path, sanity, human, visible, *rest in free_query(
            _REGISTRY, [as_of, statistics, IMAGE_GROUP_MASTERFRAME, list(MASTER_IMAGE_TYPES)]
        ):
            directory = os.path.dirname(path)
            self.registry[directory][os.path.basename(path)] = (sanity, human, visible, *rest)
            self.registry_dirs[(get_nightdate(directory), os.path.basename(directory))].add(directory)
            self.registered_path[name] = path
        copies = free_query(_BINNED_COPIES, [IMAGE_GROUP_MASTERFRAME, CALIB_TYPE_FLAT, FLAT_DEPENDENCY_ROLE])
        self.copies = {_stem(copy): source for copy, source in copies}
        with open(os.path.join(REF_DIR, "preproc_base.yml")) as f:
            self.base = _read_settings(yaml.safe_load(f))
        self.configs = None
        self.cameras, self.groups_of, self.dirs, self.ymls = {}, {}, {}, {}
        self.searches, self.ingredients, self.settings, self.designations = {}, {}, {}, {}
        self.inventories, self.raw_files, self.edges = {}, {}, {}
        self.skipped, self.globs, self.sibling_sources, self.blind = [], [], {}, {}

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
                        rec = self._record(night, unit, name, groups[name])
                        if rec["rebuilt"]:
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
                    rec = self._record(
                        night, unit, name, groups.get(name) if canonical else None, (sanity, human, *rest), visible
                    )
                    (present if visible else hidden)[name] = rec
            self.dirs[directory] = present, hidden
        return self.dirs[directory]

    def _record(
        self, night: str, unit: str, name: str, group: Optional[dict], row: Optional[tuple] = None, visible: bool = True
    ) -> dict:
        """Library record of a master."""
        sanity, human, dtype, pipe_ver, statistics = row or (None, False, None, None, None)
        n_raw = group["n"] if group else 0
        typ = group["keys"][0] if group else None
        rebuilt = n_raw >= NUM_MIN_CALIB and (
            self.as_of is None or ((night, unit) in self.targets and _listed_enough(self, night, unit, name))
        )
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
    return os.path.splitext(os.path.basename(path))[0]


def _read_settings(yml: dict, default_max_offset: Optional[int] = None) -> Tuple[int, bool, bool]:
    """(max_offset, ignore_sanity_if_no_match, ignore_lenient_keys_if_no_match) of a preprocess yml."""
    section = (yml or {}).get("preprocess") or {}
    return (
        section.get("max_offset") or default_max_offset,
        bool(section.get("ignore_sanity_if_no_match")),
        bool(section.get("ignore_lenient_keys_if_no_match")),
    )


def _yml_of(library: _Library, night: str, unit: str) -> Tuple[Optional[str], Optional[dict]]:
    """(config path, parsed yml) of a unit-night's preprocess config; None where missing."""
    if (night, unit) not in library.ymls:
        if library.configs is None:
            library.configs = dict(free_query(_PREPROCESS_CONFIGS, [CONFIG_TYPE_PREPROCESS]))
        config, yml = library.configs.get(f"{night}_{unit}"), None
        if config is not None:
            try:
                with open(config) as f:  # libyaml where available: a yml lists hundreds of frames
                    yml = yaml.load(f, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader)) or {}
            except FileNotFoundError:
                pass
        library.ymls[(night, unit)] = config, yml
    return library.ymls[(night, unit)]


def _settings_of(library: _Library, night: str, unit: str) -> Tuple[int, bool, bool]:
    """Search settings of a unit-night: its own preprocess yml when one exists, else the base."""
    if (night, unit) not in library.settings:
        _, yml = _yml_of(library, night, unit)
        library.settings[(night, unit)] = library.base if yml is None else _read_settings(yml, library.base[0])
    return library.settings[(night, unit)]


def _designation_entries(yml: Optional[dict]) -> list:
    """designated_masterframes entries of a yml, as a list."""
    entries = ((yml or {}).get("preprocess") or {}).get("designated_masterframes") or []
    return [entries] if isinstance(entries, (str, dict)) else entries


def _unmodeled(yml: Optional[dict]) -> List[str]:
    """What a yml asks for that the service does not model."""
    issues = []
    if any(isinstance(e, dict) and e.get("images") for e in _designation_entries(yml)):
        issues.append("designated_masterframes with images: pins")
    if ((yml or {}).get("preprocess") or {}).get("reuse_recorded_masters"):
        issues.append("reuse_recorded_masters: true")
    return issues


def _designations(library: _Library, night: str, unit: str) -> List[tuple]:
    """Plain designations of a unit-night: (path, NameHandler, role)."""
    if (night, unit) not in library.designations:
        found = []
        for entry in _designation_entries(_yml_of(library, night, unit)[1]):
            role, path = None, entry
            if isinstance(entry, dict):
                if entry.get("images"):
                    continue
                role = next(k for k in entry if k != "images")
                path = entry[role]
            path = str(path)
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
        return False  # records no ingredient, as ReprocessMixin._master_change
    recorded = {_stem(source) for _, source, _, _ in edges}
    for answer in answers:
        rec = library.record(answer["source"])
        if rec and not rec["resolved"]:
            _resolve(library, answer["source"])
        if _stem(answer["path"]) not in recorded or (rec and rec["rebuilt"]):
            return True
    return any(was and now and was != now for _, _, was, now in edges)


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
    if rec["rebuilt"] and (missing := _missing_inputs(library, found, rec)):
        defects.append(("config-missing-inputs", template, missing))
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


def _listed_enough(library: _Library, night: str, unit: str, name: str) -> bool:
    """Whether the owner yml lists NUM_MIN_CALIB of a master's raw frames."""
    listing = _listing(library, night, unit, name)
    return listing is None or listing[0] >= NUM_MIN_CALIB


def _listing(library: _Library, night: str, unit: str, name: str) -> Optional[Tuple[int, int]]:
    """(listed in the owner yml, in the raw DB) raw frames of a master basename; None without a yml to read."""
    _, yml = _yml_of(library, night, unit)
    if yml is None:
        return None
    if (night, unit) not in library.raw_files:
        files = defaultdict(set)
        camera = library.camera(unit, night)
        for typ, file_path, nbin, gain, exptime, filt in free_query(_RAW_FILES, {"unit": unit, "night": night}):
            quality = format_exptime(float(exptime), type="dark") if typ == CALIB_TYPE_DARK else filt
            files[_master_basename(typ, quality, unit, night, nbin, gain, camera)].add(os.path.basename(file_path))
        listed = {os.path.basename(str(p)) for p in _flatten((yml.get("input") or {}).get("masterframe_images"))}
        library.raw_files[(night, unit)] = files, listed
    files, listed = library.raw_files[(night, unit)]
    return len(files[name] & listed), len(files[name])


def _missing_inputs(library: _Library, path: str, rec: dict) -> Optional[str]:
    """Why a queue run of the owner's existing config would not rebuild a master the modeled state rebuilds."""
    night, unit = rec["nightdate"], rec["unit"]
    listing = _listing(library, night, unit, os.path.basename(path))
    if listing is None:
        config = _yml_of(library, night, unit)[0]
        missing = f"config file missing: {config}" if config else "no preprocess config in process_status"
        return f"{_stem(path)}: {missing}" if library.as_of is None else None
    return f"{_stem(path)}: {listing[0]} listed / {listing[1]} in raw DB" if listing[0] < NUM_MIN_CALIB else None


def _flatten(items) -> list:
    """Leaves of a nested list."""
    if isinstance(items, (list, tuple)):
        return [leaf for item in items for leaf in _flatten(item)]
    return [] if items is None else [items]


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
    """The owner yml's designation for the role, else the search under the owner's settings."""
    designated = _designated(library, owner, role, template)
    return designated or _search(library, template, dtype, _settings_of(library, *owner))


def _select(library: _Library, template: str, dtype: str, owner: Tuple[str, str], role: str) -> dict:
    """Answer for one role; stale-config when the base settings would answer a search otherwise."""
    answer = _answer(library, template, dtype, owner, role)
    settings = _settings_of(library, *owner)
    if settings != library.base and answer["rule"] != "designated":
        base = _search(library, template, dtype, library.base)
        if (base["path"], base["rule"]) != (answer["path"], answer["rule"]):
            yml_answer = f"{answer['path'] and _stem(answer['path'])} ({answer['rule']})"
            base_answer = f"{base['path'] and _stem(base['path'])} ({base['rule']})"
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
    """Eight columns of one role."""
    rec = library.record(answer["source"]) if answer and answer["source"] else None
    if rec and not rec["resolved"]:
        _resolve(library, answer["source"])
    return {
        role: _stem(answer["path"]) if answer and answer["path"] else None,
        f"{role}_path": answer and answer["path"],
        f"{role}_nightdate": rec and rec["nightdate"],
        f"{role}_unit": rec and rec["unit"],
        f"{role}_rule": answer and answer["rule"],
        f"{role}_state": rec and _state(rec),
        f"{role}_sanity": rec and _sanity(rec),
        f"{role}_rebuilt": rec and rec["rebuilt"],
    }


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
    """Add the unmodeled requests of a unit-night's yml."""
    config, yml = _yml_of(library, night, unit)
    for issue in _unmodeled(yml):
        tally[(night, unit, None, config, "unmodeled", issue)] += n


def associate(
    targets, *, trust_machine_rejection: bool = True, as_of: Optional[datetime] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Masters each raw science frame will be reduced with, and the cases the rules cannot settle: (table, defects)."""
    roles = ["bias", "dark", "flat", "flatdark", "dark_bias", "flat_bias", "flatdark_bias"]
    frames, unknown = _resolve_targets(targets)
    library = _Library(trust_machine_rejection, as_of, {(r[1], r[2]) for r in frames})
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
        for i in members:
            raw_path, night, unit, obj, filt, exptime, nbin, gain = frames[i]
            frame = dict(raw_path=raw_path, nightdate=night, unit=unit, obj=obj, filter=filt, exptime=exptime)
            rows[i] = {**frame, "n_binning": nbin, "gain": gain, "camera": key[4], **columns}
    for (night, unit), n in Counter((r[1], r[2]) for r in frames).items():
        _count_unmodeled(library, tally, night, unit, n)
    for night, unit, role, template, code, detail, n in unknown:
        tally[(night, unit, role, template, code, detail)] += n
    frame_columns = ["raw_path", "nightdate", "unit", "obj", "filter", "exptime", "n_binning", "gain", "camera"]
    columns = frame_columns + [c for role in roles for c in _role_columns(library, role, None)]
    columns.insert(columns.index("flat_state") + 1, "flat_source")
    return pd.DataFrame(rows, columns=columns), _defect_table(tally)


def associate_masters(
    masters: Iterable[str], *, trust_machine_rejection: bool = True, as_of: Optional[datetime] = None
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
    library = _Library(trust_machine_rejection, as_of, {(n.nightdate, n.unit) for n in names.values() if n})
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
        _count_unmodeled(library, tally, rec["nightdate"], rec["unit"], 1)
        rows.append(row)
    columns = ["master", "master_path", "type", "nightdate", "unit", "state", "n_raw", "sanity", "rebuilt"]
    columns += [c for role in ("bias", "flatdark") for c in _role_columns(library, role, None)]
    return pd.DataFrame(rows, columns=columns), _defect_table(tally)
