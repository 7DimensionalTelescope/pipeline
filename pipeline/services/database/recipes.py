"""Verified read-only recipes over the pipeline tables.

Thin, curated wrappers around ``free_query`` for questions that need a SQL join rather
than path arithmetic. The contract for anything added here:

1. **Read-only.** Writes belong to the table classes, ``services/inspection.py`` or
   ``services/database/sync.py`` -- never here.
2. **Names and paths in, names and paths out.** Never expose or require a row id; a
   caller holding a FITS file should not have to look one up.
3. **Verified against production data before landing**, and documented in
   ``.claude/memory/api-recipes.md`` -- that file is the index for these.
4. **No new abstractions.** If a question needs state or a class, it belongs in a real
   module; if it is answerable from PathHandler/NameHandler or RawFrameQuery, use those
   instead of re-asking the database.
"""

from typing import Iterable, List, Optional, Tuple

import os

from ... import const
from ...const import (
    CALIB_TYPE_DARK,
    CALIB_TYPE_FLAT,
    MASTER_IMAGE_TYPES,
    SCHEDULER_DB_PATH,
    SINGLE_DEPENDENCY_ROLE,
    TASK_STATUS_PENDING,
    TASK_STATUS_PROCESSING,
    TASK_STATUS_READY,
)
from ...version import MIN_SCIPROC_RUNTIME_VERSION_MAP
from .census import BRIEF_QA_COLUMNS, _listed, census_from_raw
from .query import free_query


def units_of(image) -> List[Tuple[str, int]]:
    """Contributing units of a coadd, by frame count. Accepts a name or a path."""
    return free_query(
        """
        SELECT s.unit, COUNT(*) AS n_frames
        FROM image_qa c
        JOIN image_qa_dependency d ON d.derived_image_id = c.id
        JOIN image_qa s            ON s.id = d.source_image_id
        WHERE c.image_name = ANY(%s)
          AND d.dependency_role = %s
        GROUP BY s.unit
        ORDER BY n_frames DESC, s.unit
        """,
        (_registered(image_names(image)), SINGLE_DEPENDENCY_ROLE),
    )


def images_of_unit(unit: str, kind: str = "coadd", limit: Optional[int] = None) -> List[Tuple[str, str]]:
    """(image_name, image_path) of every derived image `unit` contributed a frame to.

    `kind` filters by basename suffix ("coadd", "diff", or None for all). The suffix test
    is `right()` rather than LIKE: an escaped LIKE wildcard is a trap in a non-raw Python
    string, where ESCAPE '\' silently becomes ESCAPE ''.
    """
    where, params = "", [unit, SINGLE_DEPENDENCY_ROLE]
    if kind:
        suffix = f"_{kind}"
        where = "AND right(c.image_name, %s) = %s"
        params += [len(suffix), suffix]
    query = f"""
        SELECT DISTINCT c.image_name, c.image_path
        FROM image_qa s
        JOIN image_qa_dependency d ON d.source_image_id = s.id
        JOIN image_qa c            ON c.id = d.derived_image_id
        WHERE s.unit = %s
          AND d.dependency_role = %s
          {where}
        ORDER BY c.image_name
    """
    if limit:
        query += " LIMIT %s"
        params.append(limit)
    return free_query(query, params)


# transitive descendant walk over image_qa_dependency; the depth cap is a cycle backstop
_DESCENDANTS = r"""
WITH RECURSIVE seed AS (
    SELECT id FROM image_qa WHERE image_name = ANY(%s)
),
down(id, depth) AS (
        SELECT d.derived_image_id, 1
        FROM image_qa_dependency d JOIN seed ON d.source_image_id = seed.id
    UNION
        SELECT d.derived_image_id, down.depth + 1
        FROM image_qa_dependency d JOIN down ON d.source_image_id = down.id
        WHERE down.depth < %s
)
"""


def ingredients_of(image_name, role: Optional[str] = None) -> List[Tuple[str, str, str]]:
    """Direct parents of an image as (image_name, dependency_role, image_path)."""
    where, params = "", [_registered(image_names(image_name))]
    if role:
        where = "AND d.dependency_role = %s"
        params.append(role)
    return free_query(
        f"""
        SELECT s.image_name, d.dependency_role, s.image_path
        FROM image_qa c
        JOIN image_qa_dependency d ON d.derived_image_id = c.id
        JOIN image_qa s            ON s.id = d.source_image_id
        WHERE c.image_name = ANY(%s)
          {where}
        ORDER BY d.dependency_role, s.image_name
        """,
        params,
    )


def blast_radius(image_name, max_depth: int = 12) -> List[Tuple[str, str, int]]:
    """Every product transitively derived from `image`, as (image_name, image_path, depth).

    `depth` is the SHORTEST path, not a topological rank: never order regeneration by it.
    """
    return free_query(
        _DESCENDANTS
        + """
        SELECT i.image_name, i.image_path, MIN(down.depth) AS depth
        FROM down JOIN image_qa i ON i.id = down.id
        GROUP BY i.image_name, i.image_path
        ORDER BY depth, i.image_name
        """,
        (_registered(image_names(image_name)), max_depth),
    )


def configs_to_rerun(image_name, max_depth: int = 12) -> List[Tuple[str, str, int, int]]:
    """Configs owning anything derived from `image`: (name, config_file, depth, n_images).

    `depth` does not order the reruns (shortest path); a science rerun needs -overwrite.
    """
    return free_query(
        _DESCENDANTS
        + """
        SELECT p.name, p.config_file, MIN(down.depth) AS depth, COUNT(DISTINCT i.id) AS n_images
        FROM down
        JOIN image_qa i       ON i.id = down.id
        JOIN process_status p ON p.id = i.process_status_id
        GROUP BY p.name, p.config_file
        ORDER BY depth, p.name
        """,
        (_registered(image_names(image_name)), max_depth),
    )


def configs_missing_products(nightdate: Optional[str] = None, min_progress: int = 1) -> List[Tuple]:
    """Configs claiming progress but owning no registered image_qa row at all.

    DB-level only: it proves nothing was registered, not that nothing is on disk.
    """
    where, params = "", [min_progress]
    if nightdate:
        where = "AND strpos(p.config_file, %s) > 0"
        params.append(f"/{nightdate}/")
    return free_query(
        f"""
        SELECT p.name, p.config_file, p.progress, p.status
        FROM process_status p
        WHERE p.progress >= %s
          AND NOT EXISTS (SELECT 1 FROM image_qa i WHERE i.process_status_id = p.id)
          {where}
        ORDER BY p.name
        """,
        params,
    )


def select_configs_by_min_version(include_errors: bool = False, exclude_queued: bool = True) -> List[Tuple[str, str]]:
    """ONE selection, by the min-version floors: science configs below a floor or short of coadd_photometry, as (config_file, first stage to run); astrometry done, not human-rejected (a machine rejection, whose error code 387/487 is the rejection itself, is retried once below the photometry floor), not queued. Which selection a run uses is an operational decision."""
    rows = free_query(
        """
        WITH ps AS (
            SELECT config_file, object, nightdate, filter, progress, errors,
                   string_to_array(pipeline_version, '.')::int[] AS v,
                   (sanity IS FALSE AND inspected_at IS NULL) AS machine_rejected
            FROM process_status
            WHERE config_type = 'science' AND progress >= 40 AND config_file IS NOT NULL
              AND (sanity IS NOT FALSE
                   OR (inspected_at IS NULL
                       AND (pipeline_version IS NULL OR string_to_array(pipeline_version, '.')::int[] < string_to_array(%s, '.')::int[])))
        )
        SELECT config_file,
               CASE WHEN v IS NULL OR v < string_to_array(%s, '.')::int[] OR progress < 60 THEN 'single_photometry'
                    WHEN v < string_to_array(%s, '.')::int[] OR progress < 70 THEN 'coadd'
                    ELSE 'coadd_photometry' END AS stage
        FROM ps
        WHERE (errors IS NULL OR %s OR machine_rejected)
          AND NOT (v >= string_to_array(%s, '.')::int[] AND progress >= 80)
        ORDER BY count(*) OVER (PARTITION BY object, nightdate) >= 3 DESC, nightdate DESC NULLS LAST, object, filter
        """,
        (
            MIN_SCIPROC_RUNTIME_VERSION_MAP["photometry"],
            MIN_SCIPROC_RUNTIME_VERSION_MAP["photometry"],
            MIN_SCIPROC_RUNTIME_VERSION_MAP["imcoadd"],
            include_errors,
            MIN_SCIPROC_RUNTIME_VERSION_MAP["imcoadd"],
        ),
        statement_timeout_ms=600000,
    )
    if exclude_queued and SCHEDULER_DB_PATH and os.path.exists(SCHEDULER_DB_PATH):
        import sqlite3

        con = sqlite3.connect(f"file:{SCHEDULER_DB_PATH}?mode=ro", uri=True)
        try:
            live = (TASK_STATUS_READY, TASK_STATUS_PENDING, TASK_STATUS_PROCESSING)
            queued = {r[0] for r in con.execute("SELECT config FROM scheduler WHERE status IN (?, ?, ?)", live)}
        finally:
            con.close()
        rows = [row for row in rows if row[0] not in queued]
    return [(config_file, stage) for config_file, stage in rows]


def select_white_target_nights_by_min_version(min_filters: int = 3) -> List[Tuple[str, str, List[str], bool]]:
    """ONE selection, by the min-version floors: target-nights with every science config at the floors and every observed counted filter covered, white missing or stale, as (object, nightdate, parents, overwrite); overwrite when a parent coadd is newer than the white."""
    from .gwportal import RawFrameQuery

    floor = MIN_SCIPROC_RUNTIME_VERSION_MAP["imcoadd"]
    rows = free_query(
        """
        WITH sci AS (
            SELECT object, nightdate, filter, config_file,
                   (string_to_array(pipeline_version, '.')::int[] >= string_to_array(%s, '.')::int[]
                    AND progress >= 80 AND errors IS NULL) AS current
            FROM process_status
            WHERE config_type = 'science' AND nightdate IS NOT NULL AND sanity IS NOT FALSE AND config_file IS NOT NULL
        ),
        tn AS (
            SELECT object, nightdate, bool_and(current) AS all_current,
                   count(*) FILTER (WHERE filter = ANY(%s)) AS n_counted,
                   array_agg(config_file ORDER BY config_file) AS parents, array_agg(filter) AS filters
            FROM sci GROUP BY 1, 2
        ),
        white AS (
            SELECT DISTINCT ON (ps.object, ps.nightdate) ps.object, ps.nightdate, ps.status,
                   string_to_array(ps.pipeline_version, '.')::int[] >= string_to_array(%s, '.')::int[] AS at_floor,
                   w.created_at AS white_at
            FROM process_status ps
            LEFT JOIN (SELECT DISTINCT ON (process_status_id) process_status_id, created_at FROM image_qa
                       WHERE image_type = 'white' ORDER BY process_status_id, created_at DESC) w
                   ON w.process_status_id = ps.id
            WHERE ps.config_type = 'crossfilter' AND ps.nightdate IS NOT NULL
            ORDER BY ps.object, ps.nightdate, ps.updated_at DESC
        ),
        coadd AS (
            SELECT object, nightdate, max(created_at) AS coadd_at FROM image_qa
            WHERE image_type = 'coadd' AND m_epoch IS NOT TRUE GROUP BY 1, 2
        )
        SELECT tn.object, tn.nightdate::text, tn.parents, tn.filters,
               coalesce(coadd.coadd_at > white.white_at, false) AS parents_newer
        FROM tn LEFT JOIN white USING (object, nightdate) LEFT JOIN coadd USING (object, nightdate)
        WHERE tn.all_current AND tn.n_counted >= %s
          AND NOT coalesce(white.status = 'phot7ds-completed' AND white.at_floor
                           AND NOT coalesce(coadd.coadd_at > white.white_at, false), false)
        ORDER BY tn.nightdate DESC, tn.object
        """,
        (floor, list(const.ALL_FILTERS), floor, min_filters),
        statement_timeout_ms=600000,
    )
    # the raw inventory, aggregated by the builder's own SQL: 2M frames are too many to fetch row by row
    inventory = free_query(
        f"SELECT object_name, night::text, filter, bool_or(is_too) FROM ({RawFrameQuery().full_table().sql()}) r GROUP BY 1, 2, 3",
        statement_timeout_ms=600000,
    )
    observed, too = {}, set()
    for obj, night, filt, is_too in inventory:
        if is_too:
            too.add((obj, night))
        elif filt in const.ALL_FILTERS:
            observed.setdefault((obj, night), set()).add(filt)
    due = []
    for obj, night, parents, filters, parents_newer in rows:
        if (obj, night) in too or not observed.get((obj, night), set()) <= set(filters):
            continue
        due.append((obj, night, list(parents), bool(parents_newer)))
    return due


def select_filter_check_rerun_candidates(versions: Iterable[str], filter_check: Optional[dict] = None):
    """Singles processed by `versions` whose stored image_qa cards fail the current filter check, with their science configs: the
    zero-point test (beyond min_separable_wavelength, INF_DEZF > sigma_factor x (EZP_AUTO - INF_DEZF)) or the depth test (ZP_AUTO
    above the filter's zp_max at the exposure; gain-2750 frames only, image_qa holds no EGAIN); a human-vetoed False is left alone.
    `filter_check` defaults to photometry.filter_check of ref/sciproc_base.yml. Columns: config_file, image_name, object, nightdate,
    filter, inf_filt, inf_dezf, ezp_auto, zp_auto, sanity, test (zero_point_error | zp_excursion | both)."""
    import numpy as np
    import pandas as pd
    import yaml

    if filter_check is None:
        with open(os.path.join(const.REF_DIR, "sciproc_base.yml"), encoding="utf-8") as stream:
            filter_check = yaml.safe_load(stream)["photometry"]["filter_check"]
    rows = free_query(
        """
        SELECT p.config_file, s.image_name, p.object, p.nightdate::text, s.filter, s.inf_filt, s.inf_dezf, s.ezp_auto, s.zp_auto,
               s.exptime, s.gain, s.n_binning, s.sanity
        FROM pipeline.image_qa s JOIN pipeline.process_status p ON p.id = s.process_status_id
        WHERE s.image_type = 'single' AND s.pipe_ver = ANY(%s) AND s.zp_auto IS NOT NULL AND p.config_file IS NOT NULL
          AND NOT (s.sanity IS FALSE AND s.inspcomm IS NOT NULL)
        """,
        [list(versions)],
        statement_timeout_ms=600000,
    )
    df = pd.DataFrame(
        rows,
        columns=[
            "config_file",
            "image_name",
            "object",
            "nightdate",
            "filter",
            "inf_filt",
            "inf_dezf",
            "ezp_auto",
            "zp_auto",
            "exptime",
            "gain",
            "n_binning",
            "sanity",
        ],
    )
    center = {f: v["center_nm"] for f, v in const.FILTERS.items()}
    zp_max = {f: v["zp_max"] for f, v in const.FILTERS.items()}
    sep = np.array(
        [
            abs(center[a] - center[b]) if a in center and b in center else np.nan
            for a, b in zip(df["filter"], df.inf_filt)
        ]
    )
    ezp_inf = df.ezp_auto - df.inf_dezf
    zero_point = (
        (sep > filter_check["min_separable_wavelength"])
        & (df.inf_dezf > 0)
        & ((ezp_inf <= 0) | (df.inf_dezf > filter_check["sigma_factor"] * ezp_inf))
    )
    limit = df["filter"].map(zp_max) + 2.5 * np.log10(df.exptime / const.ZP_MAX_AT["exptime"])
    excursion = (df.gain == const.ZP_MAX_AT["gain"]) & (df.n_binning == 1) & (df.zp_auto > limit)
    zero_point &= bool(filter_check["zero_point_error"])
    excursion &= bool(filter_check["zp_excursion"])
    df["test"] = np.select(
        [zero_point & excursion, zero_point, excursion], ["both", "zero_point_error", "zp_excursion"], None
    )
    return df[df.test.notna()].drop(columns=["exptime", "gain", "n_binning"]).reset_index(drop=True)


def census_masters(
    units: Optional[Iterable[str]] = None,
    nightdates: Optional[Tuple[str, str]] = None,
    calib_types: Iterable[str] = MASTER_IMAGE_TYPES,
    statement_timeout_ms: int = 600000,
):
    """One row per raw calibration frame, in the same blocks: raw and master paths, the identity of the master's group
    (type, unit, nightdate, gain, binning, filter | exptime) with the master's pipe_ver, sanity, inspcomm, the night's
    preprocess config and log with their progress, then has_image_qa and the master-frame metrics."""
    import pandas as pd

    from .const import TABLES
    from .image_qa import ImageQATable

    where, params = [], []
    if units is not None:
        where.append("fu.name = ANY(%s)")
        params.append([str(u) for u in _listed(units)])
    if nightdates is not None:
        where.append("n.date BETWEEN %s AND %s")
        params.extend(nightdates)
    metrics = ["ppflag", "shifted", "shftscr", *ImageQATable.MASTERFRAME_COLUMNS]
    columns = [
        "raw_path",
        "master_path",
        "calib_type",
        "nightdate",
        "unit",
        "camera",
        "filter",
        "exptime",
        "obstime",
        "gain",
        "binning",
        "pipe_ver",
        "sanity",
        "inspcomm",
        "config_file",
        "log_file",
        "config_progress",
        "config_errors",
        "config_status",
        "has_image_qa",
        *metrics,
    ]
    rows = []
    for calib_type in _listed(calib_types):
        if calib_type == CALIB_TYPE_FLAT:
            filter_select, filter_join, key = (
                "f.name",
                "LEFT JOIN facility_filter f ON cf.filter_id = f.id",
                "m.filter = r.filter",
            )
        elif calib_type == CALIB_TYPE_DARK:
            filter_select, filter_join, key = "NULL::text", "", "m.exptime = r.exptime"
        else:
            filter_select, filter_join, key = "NULL::text", "", "TRUE"
        rows += free_query(
            f"""
            WITH r AS (
                SELECT cf.id, n.date::text AS nightdate, fu.name AS unit, {filter_select} AS filter, cf.exptime,
                       cf.obstime, cf.gain, cf.binning_x AS binning, cf.file_path AS raw_path
                FROM {TABLES[calib_type]} cf
                JOIN survey_night n ON cf.night_id = n.id JOIN facility_unit fu ON cf.unit_id = fu.id
                {filter_join}
                WHERE {" AND ".join(where) or "TRUE"}),
            m AS (
                SELECT id, unit, nightdate::text AS nightdate, filter, exptime, gain, n_binning, image_path, camera,
                       pipe_ver, sanity, inspcomm, {", ".join(metrics)}
                FROM image_qa WHERE image_type = %s AND image_group = 'masterframe'),
            p AS (
                SELECT DISTINCT ON (nightdate, unit) nightdate::text AS nightdate, unit,
                       config_file, log_file, progress, errors, status
                FROM process_status WHERE config_type = 'preprocess'
                ORDER BY nightdate, unit, updated_at DESC NULLS LAST, id DESC)
            SELECT DISTINCT ON (r.id) r.raw_path, m.image_path, %s::text, r.nightdate, r.unit, m.camera, r.filter,
                   r.exptime, r.obstime, r.gain, r.binning, m.pipe_ver, m.sanity, m.inspcomm,
                   p.config_file, p.log_file, p.progress, p.errors, p.status,
                   m.id IS NOT NULL, {", ".join("m." + c for c in metrics)}
            FROM r
            LEFT JOIN m ON m.unit = r.unit AND m.nightdate = r.nightdate AND m.gain = r.gain
                       AND m.n_binning = r.binning AND {key}
            LEFT JOIN p ON p.nightdate = r.nightdate AND p.unit = r.unit
            ORDER BY r.id, (m.image_path IS NOT NULL) DESC, m.id DESC
            """,
            params + [calib_type, calib_type],
            statement_timeout_ms=statement_timeout_ms,
        )
    census = pd.DataFrame(rows, columns=columns)
    return census.sort_values(["nightdate", "unit", "obstime"], ignore_index=True)


def stage_funnel(census):
    """Per-stage frame counts as fractions of the raw frames (the budget is 1): raw, then the stated rows "sanity false"
    (a registered single with SANITY=F) and "inspcomm" (a human verdict on the single), then preprocess (a registered
    single) -> the science registry over sane frames, and -> in_coadd -> the default crossfilter processes."""
    import pandas as pd

    from ...const.run import DEFAULT_CROSSFILTER_PROCESSES
    from ...const.sciproc import SCIPROCESS_REGISTRY

    registered = census.has_image_qa.fillna(False).astype(bool)
    rejected = registered & census.sanity.eq(False)
    table = census.assign(preprocess=registered, sanity_ok=registered & ~rejected, in_coadd=census.coadd_name.notna())
    science = ["preprocess", "sanity_ok"] + [spec.name for spec in SCIPROCESS_REGISTRY.specs]
    white = ["preprocess", "sanity_ok", "in_coadd"] + list(DEFAULT_CROSSFILTER_PROCESSES)
    reached = pd.concat([table[science].cummin(axis=1), table[white].cummin(axis=1).iloc[:, 2:]], axis=1)
    n = len(table)
    funnel = pd.DataFrame({"frames": reached.sum()})
    funnel.loc["raw"] = n
    funnel.loc["sanity false"] = int(rejected.sum())
    funnel.loc["inspcomm"] = int((registered & census.inspcomm.notna()).sum())
    stages = ["raw", "sanity false", "inspcomm", "preprocess"] + [c for c in reached.columns if c not in ("preprocess", "sanity_ok")]
    funnel = funnel.loc[stages]
    prev = funnel.frames.shift(1)
    prev.loc["preprocess"] = n
    sane = funnel.frames["preprocess"] - funnel.frames["sanity false"]
    prev.loc[science[2]] = sane  # the science chain continues from the sane frames
    prev.loc["in_coadd"] = sane  # the white chain branches off there too
    funnel["lost_here"] = (prev - funnel.frames).fillna(0).astype(int)
    funnel.loc[["sanity false", "inspcomm"], "lost_here"] = 0  # verdicts, stated, not losses
    funnel["fraction_of_raw"] = (funnel.frames / n).round(4)
    funnel["lost_fraction_of_raw"] = (funnel.lost_here / n).round(4)
    return funnel


def image_names(images) -> List[str]:
    """Normalize image path(s)/name(s) to bare image_qa `image_name` values (basename without .fits)."""
    from ...utils import atleast_1d

    names = []
    for image in atleast_1d(images):
        raw = str(image)
        if raw.endswith("/"):
            raise ValueError(f"looks like a directory, not an image: {image!r}")
        name = os.path.basename(raw).strip()
        if name.endswith(".fits"):
            name = name[: -len(".fits")]
        if not name:
            raise ValueError(f"not an image name or path: {image!r}")
        if any(ch in name for ch in "*?["):
            raise ValueError(f"globs are not accepted, pass explicit names: {image!r}")
        names.append(name)
    if not names:
        raise ValueError("no images given")
    return names


def _registered(names: List[str]) -> List[str]:
    """The subset of `names` present in image_qa; raises if none are (an empty result would be indistinguishable from 'no dependencies')."""
    rows = free_query("SELECT image_name FROM image_qa WHERE image_name = ANY(%s)", (names,))
    found = {r[0] for r in rows}
    if not found:
        raise ValueError(
            f"none of these are registered in image_qa: {names[:3]}{'...' if len(names) > 3 else ''}. "
            "An unregistered image yields an empty result that is indistinguishable from 'no dependencies'."
        )
    return sorted(found)
