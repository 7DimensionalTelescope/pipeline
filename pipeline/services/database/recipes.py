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

from typing import List, Optional, Tuple

import os

from ...const import (
    ALL_FILTERS,
    SCHEDULER_DB_PATH,
    SINGLE_DEPENDENCY_ROLE,
    TASK_STATUS_PENDING,
    TASK_STATUS_PROCESSING,
    TASK_STATUS_READY,
)
from ...version import MIN_SCIPROC_RUNTIME_VERSION_MAP
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
    """ONE selection, by the min-version floors: science configs below a floor or short of coadd_photometry, as (config_file, first stage to run); astrometry done, not rejected, not queued. Which selection a run uses is an operational decision."""
    rows = free_query(
        """
        WITH ps AS (
            SELECT config_file, object, nightdate, filter, progress, errors,
                   string_to_array(pipeline_version, '.')::int[] AS v
            FROM process_status
            WHERE config_type = 'science' AND progress >= 40 AND sanity IS NOT FALSE AND config_file IS NOT NULL
        )
        SELECT config_file,
               CASE WHEN v IS NULL OR v < string_to_array(%s, '.')::int[] OR progress < 60 THEN 'single_photometry'
                    WHEN v < string_to_array(%s, '.')::int[] OR progress < 70 THEN 'coadd'
                    ELSE 'coadd_photometry' END AS stage
        FROM ps
        WHERE (errors IS NULL OR %s)
          AND NOT (v >= string_to_array(%s, '.')::int[] AND progress >= 80)
        ORDER BY count(*) OVER (PARTITION BY object, nightdate) >= 3 DESC, nightdate DESC NULLS LAST, object, filter
        """,
        (
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
        (floor, list(ALL_FILTERS), floor, min_filters),
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
        elif filt in ALL_FILTERS:
            observed.setdefault((obj, night), set()).add(filt)
    due = []
    for obj, night, parents, filters, parents_newer in rows:
        if (obj, night) in too or not observed.get((obj, night), set()) <= set(filters):
            continue
        due.append((obj, night, list(parents), bool(parents_newer)))
    return due


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
