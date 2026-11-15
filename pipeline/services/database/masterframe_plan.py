"""Stored master-association plan (masterframe_plan, masterframe_plan_master): readers, fingerprint, per-unit replace.

Written only by services.masterframe_association.refresh_plan; rationale in .claude/memory/services.md.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Iterable, Optional

import pandas as pd
import psycopg

from ...const import (
    BIAS_DEPENDENCY_ROLE,
    CONFIG_TYPE_PREPROCESS,
    DARK_DEPENDENCY_ROLE,
    DISK_CHANGE_NIGHTDATE,
    DISK_CHANGE_NIGHTDATE_2,
    FLAT_DEPENDENCY_ROLE,
    IMAGE_GROUP_MASTERFRAME,
    MASTER_FRAME_DIR,
    MASTER_FRAME_DIR_2,
    MASTER_IMAGE_TYPES,
    NUM_MIN_CALIB,
    REF_DIR,
)
from ...path.cam_tracker import get_current_camera_serial
from ...version import MIN_PREPROC_RUNTIME_VERSION, __version__
from .const import DB_PARAMS
from .query import free_query

_UNITS = """
SELECT u.name FROM facility_unit u WHERE EXISTS (SELECT 1 FROM survey_scienceframe x WHERE x.unit_id = u.id) ORDER BY 1
"""

_DIGEST = """
SELECT unit, md5(string_agg(line, E'\\n' ORDER BY line)) FROM ({}) part(unit, line) GROUP BY unit
"""

_RAW_CALIBRATION = """
SELECT u.name, ROW('bias', n.date, x.binning_x, x.gain, NULL::float8, NULL::text, count(*))::text
FROM survey_biasframe x JOIN facility_unit u ON u.id = x.unit_id JOIN survey_night n ON n.id = x.night_id
WHERE u.name = ANY(%s)
GROUP BY u.name, n.date, x.binning_x, x.gain
UNION ALL
SELECT u.name, ROW('dark', n.date, x.binning_x, x.gain, x.exptime, NULL::text, count(*))::text
FROM survey_darkframe x JOIN facility_unit u ON u.id = x.unit_id JOIN survey_night n ON n.id = x.night_id
WHERE u.name = ANY(%s)
GROUP BY u.name, n.date, x.binning_x, x.gain, x.exptime
UNION ALL
SELECT u.name, ROW('flat', n.date, x.binning_x, x.gain, NULL::float8, f.name, count(*))::text
FROM survey_flatframe x JOIN facility_unit u ON u.id = x.unit_id JOIN survey_night n ON n.id = x.night_id
LEFT JOIN facility_filter f ON f.id = x.filter_id
WHERE u.name = ANY(%s)
GROUP BY u.name, n.date, x.binning_x, x.gain, f.name
"""

_RAW_SCIENCE = """
SELECT u.name, ROW(n.date, x.binning_x, x.gain, x.exptime, f.name, count(*))::text
FROM survey_scienceframe x JOIN facility_unit u ON u.id = x.unit_id JOIN survey_night n ON n.id = x.night_id
LEFT JOIN facility_filter f ON f.id = x.filter_id
WHERE u.name = ANY(%s)
GROUP BY u.name, n.date, x.binning_x, x.gain, x.exptime, f.name
"""

_MASTERS = """
SELECT u.unit, ROW(q.image_name, q.image_path, q.imageid, q.sanity,
       NULLIF(btrim(q.inspcomm), '') IS NOT NULL AND q.inspcomm <> 'Auto-generated binned master frame',
       q.image_type, q.pipe_ver, {})::text
FROM image_qa q JOIN unnest(%s::text[]) u(unit) ON q.image_path LIKE '%%/' || u.unit || '/%%'
WHERE q.image_group = %s AND q.image_type = ANY(%s)
"""

_EDGES = """
SELECT u.unit, ROW(m.image_name, d.dependency_role, s.image_path, d.source_imageid, s.imageid)::text
FROM image_qa m JOIN unnest(%s::text[]) u(unit) ON m.image_path LIKE '%%/' || u.unit || '/%%'
JOIN image_qa_dependency d ON d.derived_image_id = m.id JOIN image_qa s ON s.id = d.source_image_id
WHERE m.image_group = %s AND m.image_type = ANY(%s) AND d.dependency_role = ANY(%s)
"""

_CONFIGS = """
SELECT DISTINCT ON (p.name) u.unit, ROW(p.name, p.max_offset, p.ignore_sanity_if_no_match,
       p.ignore_lenient_keys_if_no_match, p.designated_masterframes)::text
FROM process_status p JOIN unnest(%s::text[]) u(unit) ON p.name LIKE '%%\\_' || u.unit
WHERE p.config_type = %s
ORDER BY p.name, p.updated_at DESC NULLS LAST, p.id DESC
"""

_STORED = """
SELECT unit, max(refreshed_at), max(fingerprint) FROM pipeline.masterframe_plan_master WHERE unit = ANY(%s) GROUP BY unit
"""

_GROUPS = """
SELECT * FROM pipeline.masterframe_plan
WHERE unit = ANY(%s) AND (%s::date[] IS NULL OR nightdate = ANY(%s::date[]))
  AND (%s::text[] IS NULL OR master = ANY(%s::text[]) OR flat_source = ANY(%s::text[]))
ORDER BY unit, nightdate, n_binning, gain, exptime, filter, role
"""

_MASTER_ROWS = """
SELECT * FROM pipeline.masterframe_plan_master
WHERE unit = ANY(%s) AND (%s::text[] IS NULL OR bias = ANY(%s::text[]) OR flatdark = ANY(%s::text[]))
ORDER BY unit, nightdate, master
"""


def _file_sha(path: str) -> str:
    """sha256 of a file."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def fingerprint_parts(units: Iterable[str]) -> Dict[str, dict]:
    """Per unit, the named parts of everything its plan reads."""
    units = list(units)
    with open(os.path.join(REF_DIR, "qa", "preprocess.json")) as f:
        statistics = sorted({key for section in json.load(f).values() for key in section})
    masterframe = [IMAGE_GROUP_MASTERFRAME, list(MASTER_IMAGE_TYPES)]
    roles = [BIAS_DEPENDENCY_ROLE, DARK_DEPENDENCY_ROLE, FLAT_DEPENDENCY_ROLE]
    queries = {
        "raw_calibration": (_RAW_CALIBRATION, [units] * 3),
        "raw_science": (_RAW_SCIENCE, [units]),
        "masters": (_MASTERS.format(", ".join(f"q.{key}" for key in statistics)), [units, *masterframe]),
        "edges": (_EDGES, [units, *masterframe, roles]),
        "configs": (_CONFIGS, [units, CONFIG_TYPE_PREPROCESS]),
    }
    digests = {part: dict(free_query(_DIGEST.format(sql), params)) for part, (sql, params) in queries.items()}
    common = [__version__, MIN_PREPROC_RUNTIME_VERSION, NUM_MIN_CALIB, MASTER_FRAME_DIR, MASTER_FRAME_DIR_2]
    common += [DISK_CHANGE_NIGHTDATE, DISK_CHANGE_NIGHTDATE_2, _file_sha(os.path.join(REF_DIR, "preproc_base.yml"))]
    common += [_file_sha(os.path.join(REF_DIR, "qa", "preprocess.json"))]
    parts = {}
    for unit in units:
        changelog = os.path.join(REF_DIR, f"InstrumEvent/changelog_unit{int(unit[3:])}.txt")
        parts[unit] = {part: digest.get(unit) for part, digest in digests.items()}
        parts[unit]["instrument"] = [_file_sha(changelog), get_current_camera_serial(int(unit[3:]))]
        parts[unit]["global"] = common
    return parts


def fingerprints(units: Iterable[str]) -> Dict[str, str]:
    """One hash per unit of everything its plan reads."""
    return {
        unit: hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()
        for unit, parts in fingerprint_parts(units).items()
    }


def _units(units: Optional[Iterable[str]]) -> list:
    """Names as a list, a bare string being one; None: every unit of the raw DB with a science frame."""
    if units is None:
        return [unit for unit, in free_query(_UNITS)]
    return [units] if isinstance(units, str) else list(units)


def _read(units: list, query: str, params: list, allow_stale: bool) -> pd.DataFrame:
    """Plan rows and the units' stored fingerprints in one snapshot; raises for a unit never refreshed or stale."""
    with psycopg.connect(**DB_PARAMS) as conn:
        conn.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
        stored = {unit: fingerprint for unit, _, fingerprint in conn.execute(_STORED, [units])}
        cur = conn.execute(query, params)
        table = pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description], dtype=object)
    if not allow_stale:
        current = fingerprints(units)
        missing = [unit for unit in units if unit not in stored]
        stale = [unit for unit in units if unit in stored and stored[unit] != current[unit]]
        if missing or stale:
            raise RuntimeError(f"masterframe plan never refreshed: {missing}, stale: {stale}; run refresh_plan")
    return table


def plan_status(units: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """unit, refreshed_at, stale."""
    units = _units(units)
    stored = {unit: (at, fingerprint) for unit, at, fingerprint in free_query(_STORED, [units])}
    current = fingerprints(units)
    rows = [(unit, *stored.get(unit, (None, None))) for unit in units]
    return pd.DataFrame(
        [(unit, at, fingerprint != current[unit]) for unit, at, fingerprint in rows],
        columns=["unit", "refreshed_at", "stale"],
    )


def plan_groups(
    units: Optional[Iterable[str]] = None,
    nightdates: Optional[Iterable[str]] = None,
    uses: Optional[Iterable[str]] = None,
    allow_stale: bool = False,
) -> pd.DataFrame:
    """Stored group x role rows; uses: a master in any role, or the source of a binned copy."""
    units = _units(units)
    nights = None if nightdates is None else [str(n) for n in nightdates]
    uses = None if uses is None else _units(uses)
    return _read(units, _GROUPS, [units, nights, nights, uses, uses, uses], allow_stale)


def plan_masters(
    units: Optional[Iterable[str]] = None, uses: Optional[Iterable[str]] = None, allow_stale: bool = False
) -> pd.DataFrame:
    """Stored master rows; uses: a master as bias or flatdark."""
    units = _units(units)
    uses = None if uses is None else _units(uses)
    return _read(units, _MASTER_ROWS, [units, uses, uses, uses], allow_stale)


def replace_plan(conn: psycopg.Connection, unit: str, groups: pd.DataFrame, masters: pd.DataFrame) -> None:
    """A unit's rows of both plan tables replaced in one transaction."""
    with conn.transaction():
        for table, rows in (("masterframe_plan", groups), ("masterframe_plan_master", masters)):
            conn.execute(f"DELETE FROM pipeline.{table} WHERE unit = %s", [unit])
            with conn.cursor().copy(f"COPY pipeline.{table} ({', '.join(rows.columns)}) FROM STDIN") as copy:
                for row in rows.astype(object).where(rows.notna(), None).itertuples(index=False):
                    copy.write_row(row)
