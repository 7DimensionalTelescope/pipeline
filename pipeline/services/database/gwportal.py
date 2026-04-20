"""
Comprehensive GWPortal database wrapper.

This module is a single entry point for querying the GWU survey database. It
covers every entity that the official GWPortal REST API exposes -- raw,
processed, and combined science frames (including their TOO variants), tiles,
targets, and master calibration frames (bias/dark/flat) -- through two
interchangeable backends:

* **SQL** -- a direct ``psycopg`` connection to the Postgres instance. This is
  the default because it is ~20x faster than the HTTP API on warm connections
  (see benchmarks in the repository).
* **HTTP** -- the official ``GWPortalClient`` REST wrapper. Used transparently
  as a fallback when SQL credentials are missing, the database is unreachable,
  or the caller explicitly asks for it.

Usage
-----

The quickest path is the unified connector::

    from pipeline.services.database.gwportal import GWPortalQuery

    q = GWPortalQuery("processed")            # SQL first, HTTP on fallback
    rows = q.query(date_start="2025-10-01", date_end="2025-10-05",
                   filter_name="m525")

    # Or pull the result as an astropy.table.Table:
    tbl = q.query_table(date_start="2025-10-01", filter_name="r")

Each entity also has a fluent builder with a narrower, typed-looking API::

    from pipeline.services.database.gwportal import (
        ProcessedFrameQuery, MasterFlatQuery, TileQuery,
    )

    rows = (ProcessedFrameQuery()
            .on_date("2025-10-01")
            .by_units(["7DT01", "7DT02"])
            .with_filter("m525")
            .fetch())

    flats = (MasterFlatQuery()
             .by_unit("7DT01")
             .on_nightdate("2025-09-30")
             .with_filter("g")
             .fetch())

    tiles_near = (TileQuery()
                  .cone_search(ra=180.0, dec=0.0, radius=2.0)
                  .fetch())

Backend selection
-----------------

All query classes accept a ``backend`` keyword (``"auto"`` by default). The
possible values are:

* ``"auto"``  -- try SQL first; on any failure (including missing credentials)
  fall back to HTTP. This is the default.
* ``"sql"``   -- require the direct Postgres connection; raise on failure.
* ``"http"``  -- require the REST API; raise on failure.

You can also pre-select the preferred order programmatically, for example when
a deployment's network topology makes HTTP faster::

    GWPortalQuery("raw", backend="http").query(...)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .const import DB_PARAMS, GWPORTAL_API_KEY, GWPORTAL_BASE_URL
from .gwportal_client import GWPortalClient
from .query import check_db_connection, get_pool

# --------------------------------------------------------------------------- #
# Types and constants
# --------------------------------------------------------------------------- #

DateLike = Union[str, date, datetime]
ListOrOne = Union[str, Sequence[str], None]


class Backend(str, Enum):
    AUTO = "auto"
    SQL = "sql"
    HTTP = "http"

    @classmethod
    def parse(cls, value: Union["Backend", str, None]) -> "Backend":
        if value is None:
            return cls.AUTO
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError as exc:
            raise ValueError(f"Unknown backend {value!r}. Use one of: " f"{[b.value for b in cls]}") from exc


# Valid entities. Mirrors ``GWPortalClient.ENDPOINTS`` plus calib aliases.
ENTITY_ALIASES: Dict[str, str] = {
    "raw": "raw",
    "rawframe": "raw",
    "rawframes": "raw",
    "sci": "raw",
    "science": "raw",
    "scienceframe": "raw",
    "processed": "processed",
    "processedframe": "processed",
    "processedscienceframe": "processed",
    "combined": "combined",
    "combinedframe": "combined",
    "combinedscienceframe": "combined",
    "processed_too": "processed_too",
    "proctoo": "processed_too",
    "processedtoo": "processed_too",
    "combined_too": "combined_too",
    "combtoo": "combined_too",
    "combinedtoo": "combined_too",
    "tile": "tile",
    "tiles": "tile",
    "target": "target",
    "targets": "target",
    "master_bias": "master_bias",
    "masterbias": "master_bias",
    "mbias": "master_bias",
    "master_dark": "master_dark",
    "masterdark": "master_dark",
    "mdark": "master_dark",
    "master_flat": "master_flat",
    "masterflat": "master_flat",
    "mflat": "master_flat",
    # Raw individual calibration frames (SQL-only: the REST API has no
    # endpoint for these).
    "bias": "bias",
    "biasframe": "bias",
    "raw_bias": "bias",
    "dark": "dark",
    "darkframe": "dark",
    "raw_dark": "dark",
    "flat": "flat",
    "flatframe": "flat",
    "raw_flat": "flat",
}


def _normalize_entity(entity: str) -> str:
    key = entity.strip().lower().replace("-", "_")
    try:
        return ENTITY_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unknown entity {entity!r}. Valid: {sorted(set(ENTITY_ALIASES))}") from exc


_TILE_RE = re.compile(r"^T\d+$")


# --------------------------------------------------------------------------- #
# Lean (default) SELECT column lists
# --------------------------------------------------------------------------- #
# These are the hand-picked column lists used when ``full_table`` is False
# (the default). They keep result rows small so common notebook workflows
# stay fast. Pass ``full_table=True`` (or call ``.full_table()`` on a
# builder) to get every native column of the underlying table instead.

_LEAN_COLS_RAW = """
    sf.id,
    sf.original_filename                AS filename,
    sf.file_path                        AS filepath,
    sf.unified_filename,
    sf.obstime,
    sf.mjd,
    sf.exptime,
    sf.airmass,
    sf.fwhm                             AS seeing,
    sf.object_name,
    sf.object_type,
    sf.object_ra,
    sf.object_dec,
    sf.obsnote,
    u.name  AS unit,
    f.name  AS filter,
    t.name  AS target,
    tl.name AS tile,
    n.date  AS night
"""

_LEAN_COLS_PROCESSED = """
    pf.id,
    pf.filename,
    pf.filepath,
    pf.processing_version,
    pf.data_stream,
    pf.reduction_status,
    pf.created_at,
    pf.mjd,
    sf.obstime,
    sf.unified_filename,
    sf.obsnote,
    pf.ra_center,
    pf.dec_center,
    pf.l_center,
    pf.b_center,
    pf.seeing,
    pf.ellip,
    pf.elong,
    pf.skyval,
    pf.skysig,
    pf.ul5,
    pf.zp,
    sf.exptime,
    u.name  AS unit,
    f.name  AS filter,
    t.name  AS target,
    tl.name AS tile,
    n.date  AS night,
    pf.raw_frame_id
"""

_LEAN_COLS_COMBINED = """
    cf.id,
    cf.filename,
    cf.filepath,
    cf.processing_version,
    cf.data_stream,
    cf.reduction_status,
    cf.n_combined,
    cf.total_exptime,
    cf.obs_start,
    cf.obs_end,
    cf.mjd,
    cf.ra_center,
    cf.dec_center,
    cf.l_center,
    cf.b_center,
    cf.seeing,
    cf.ellip,
    cf.elong,
    cf.skyval,
    cf.skysig,
    cf.ul5,
    cf.zp,
    u.name  AS unit,
    f.name  AS filter,
    t.name  AS target,
    tl.name AS tile
"""

_LEAN_COLS_RAW_CALIB = """
    cf.id,
    cf.original_filename AS filename,
    cf.file_path         AS filepath,
    cf.unified_filename,
    cf.obstime,
    cf.mjd,
    cf.exptime,
    cf.binning_x,
    cf.binning_y,
    cf.gain,
    cf.egain,
    cf.instrument,
    cf.ccdtemp,
    cf.is_usable,
    cf.quality_score,
    cf.processing_status,
    u.name AS unit,
    n.date AS night
"""

# Kind-specific extras that are appended to ``_LEAN_COLS_RAW_CALIB`` when
# ``full_table=False``. In full mode these are already covered by ``cf.*``.
_LEAN_CALIB_EXTRAS = {
    "bias": "cf.median_level, cf.noise_level, cf.std_deviation",
    "dark": "cf.dark_current, cf.hotpix_count, cf.median_level",
    "flat": (
        "cf.median_counts, cf.uniformity_rms, "
        "cf.vignetting_level, cf.illumination_gradient"
    ),
}

_LEAN_COLS_TILE = """
    t.id, t.name, t.ra, t.dec, t.l, t.b, t.priority,
    t.survey_program, t.area_sq_deg, t.observation_count,
    t.total_exposure_time, t.first_observed, t.last_observed
"""

_LEAN_COLS_TARGET = """
    t.id, t.name, t.ra, t.dec, t.l, t.b, t.target_type,
    t.description, t.area_sq_deg, t.observation_count,
    t.total_exposure_time, t.first_observed, t.last_observed
"""

_LEAN_COLS_MASTER = {
    "bias": """
        m.id, m.file_path, m.sigma_file_path, m.nightdate,
        m.binning_x, m.binning_y, m.gain, m.camera_serial,
        m.processing_version, m.software_version, m.is_production,
        m.n_combined, m.clip_mean, m.clip_median, m.clip_std,
        m.center_clip_mean, m.center_clip_median, m.center_clip_std,
        u.name AS unit
    """,
    "dark": """
        m.id, m.file_path, m.sigma_file_path, m.nightdate, m.exptime,
        m.binning_x, m.binning_y, m.gain, m.camera_serial,
        m.processing_version, m.software_version, m.is_production,
        m.n_combined, m.clip_mean, m.clip_median, m.clip_std,
        m.center_clip_mean, m.center_clip_median, m.center_clip_std,
        m.master_bias_id, u.name AS unit
    """,
    "flat": """
        m.id, m.file_path, m.sigma_file_path, m.nightdate,
        m.binning_x, m.binning_y, m.gain, m.camera_serial,
        m.processing_version, m.software_version, m.is_production,
        m.n_combined, m.clip_mean, m.clip_median, m.clip_std,
        m.center_clip_mean, m.center_clip_median, m.center_clip_std,
        m.master_bias_id, m.master_dark_id,
        f.name AS filter, u.name AS unit
    """,
}


# Spatial columns per entity. First element is the (ra, dec) point columns,
# second is the polygon column, each in the (radec, galactic) coordinate
# system. ``None`` means that system is not stored (and the SQL backend will
# fall back to transforming coordinates before dispatch, when possible).
_SPATIAL_COLUMNS: Dict[str, Dict[str, Any]] = {
    "raw": {
        "table_alias": "sf",
        "point_radec": ("sf.object_ra", "sf.object_dec"),
        "point_galactic": None,  # no stored galactic point; derive via q3c
        "poly_radec": None,
        "poly_galactic": None,
    },
    "processed": {
        "table_alias": "pf",
        "point_radec": ("pf.ra_center", "pf.dec_center"),
        "point_galactic": ("pf.l_center", "pf.b_center"),
        "poly_radec": "pf.poly",
        "poly_galactic": "pf.poly_galactic",
    },
    "combined": {
        "table_alias": "cf",
        "point_radec": ("cf.ra_center", "cf.dec_center"),
        "point_galactic": ("cf.l_center", "cf.b_center"),
        "poly_radec": "cf.footprint",
        "poly_galactic": "cf.footprint_galactic",
    },
    "tile": {
        "table_alias": "t",
        "point_radec": ("t.ra", "t.dec"),
        "point_galactic": ("t.l", "t.b"),
        "poly_radec": "t.vertices",
        "poly_galactic": "t.vertices_galactic",
    },
    "target": {
        "table_alias": "t",
        "point_radec": ("t.ra", "t.dec"),
        "point_galactic": ("t.l", "t.b"),
        "poly_radec": "t.vertices",
        "poly_galactic": "t.vertices_galactic",
    },
}


def _to_date(value: DateLike) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _as_list(value: ListOrOne) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [str(value)]


# --------------------------------------------------------------------------- #
# SQL backend
# --------------------------------------------------------------------------- #


class _SqlBackend:
    """
    Direct psycopg queries that return the same field shape as the REST API.

    The column lists were chosen to match what ``GWPortalClient`` returns for
    each endpoint, so callers can swap backends without noticing.
    """

    def __init__(self):
        self.pool = get_pool()
        if self.pool is None:
            raise RuntimeError("Postgres connection pool unavailable")
        # Introspection: populated by ``_execute`` (or ``_capture``) so callers
        # can inspect what was actually run.
        self.last_sql: Optional[str] = None
        self.last_params: Optional[Tuple[Any, ...]] = None
        # When True, ``_execute`` records the SQL / params but skips execution
        # and returns ``[]``. Used by the ``dry_run=True`` path on builders.
        self.dry_run: bool = False
        # Cache of table -> [column_name, ...] used by ``_star`` to expand
        # ``<alias>.*`` at query-build time with optional prefix aliasing.
        self._columns_cache: Dict[str, Tuple[str, ...]] = {}

    # -- helpers ---------------------------------------------------------- #
    def _table_columns(self, table: str) -> Tuple[str, ...]:
        """
        Return the column names of ``table`` in the ``public`` schema,
        cached across calls. Used by :meth:`_star` to materialize
        ``<alias>.*`` into an explicit list so we can control aliasing
        (and avoid collisions across JOINs).
        """
        if table in self._columns_cache:
            return self._columns_cache[table]
        sql = (
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = %s "
            "ORDER BY ordinal_position"
        )
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (table,))
                cols = tuple(row[0] for row in cur.fetchall())
        self._columns_cache[table] = cols
        return cols

    def _star(
        self,
        *,
        alias: str,
        table: str,
        prefix: str = "",
        exclude: Sequence[str] = (),
    ) -> str:
        """
        Build an explicit ``alias.col AS [prefix]col`` list for every column
        in ``table``.

        Use this instead of ``alias.*`` whenever more than one ``*`` would
        appear in the same SELECT (to avoid silent column collisions when
        the result rows are zipped into dicts in :meth:`_execute`).

        Parameters
        ----------
        alias : str
            SQL alias used for the table in the FROM/JOIN clause.
        table : str
            Unqualified table name (``public`` schema assumed).
        prefix : str
            Optional prefix prepended to each output column name. Use e.g.
            ``"raw_"`` to disambiguate joined rows.
        exclude : sequence of str
            Column names (unprefixed) to omit from the expansion.
        """
        cols = self._table_columns(table)
        if not cols:
            # Fall back to the SQL star; better than an empty SELECT. Any
            # duplicates will still clobber, but at least the query runs.
            return f"{alias}.*"
        excluded = set(exclude)
        parts: List[str] = []
        for c in cols:
            if c in excluded:
                continue
            if prefix:
                parts.append(f"{alias}.{c} AS {prefix}{c}")
            else:
                parts.append(f"{alias}.{c}")
        return ",\n                ".join(parts)

    def _execute(self, sql: str, params: Sequence[Any]) -> List[Dict[str, Any]]:
        # Always capture before we even try, so a failing query is still
        # inspectable via ``last_sql`` / ``last_params``.
        self.last_sql = sql
        self.last_params = tuple(params)
        if self.dry_run:
            return []
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]

    # -- introspection ---------------------------------------------------- #
    def mogrify(self, sql: Optional[str] = None, params: Optional[Sequence[Any]] = None) -> str:
        """
        Return a fully-substituted SQL string (placeholders replaced with
        properly-quoted literals). Useful for copy/paste into ``psql``.

        If ``sql`` / ``params`` are omitted, the last captured query is used.
        """
        sql = sql if sql is not None else self.last_sql
        params = params if params is not None else self.last_params
        if sql is None:
            return ""
        try:
            from psycopg import ClientCursor
            with self.pool.connection() as conn:
                with ClientCursor(conn) as cur:
                    return cur.mogrify(sql, tuple(params or ()))
        except Exception:
            # Fall back: return the statement with a separate repr of params.
            return f"{sql}\n-- params: {params!r}"

    @staticmethod
    def _apply_date_filters(
        clauses: List[str],
        params: List[Any],
        *,
        col: str,
        date_start: Optional[DateLike],
        date_end: Optional[DateLike],
        days: Optional[int],
    ):
        if days is not None:
            cutoff = datetime.utcnow() - timedelta(days=int(days))
            clauses.append(f"{col} >= %s")
            params.append(cutoff)
            return
        if date_start is not None:
            clauses.append(f"{col} >= %s")
            params.append(_to_date(date_start))
        if date_end is not None:
            clauses.append(f"{col} <= %s")
            params.append(_to_date(date_end))

    @staticmethod
    def _apply_in(clauses: List[str], params: List[Any], col: str, values: Optional[Sequence[Any]]):
        if not values:
            return
        values = list(values)
        if len(values) == 1:
            clauses.append(f"{col} = %s")
            params.append(values[0])
        else:
            ph = ",".join(["%s"] * len(values))
            clauses.append(f"{col} IN ({ph})")
            params.extend(values)

    @staticmethod
    def _apply_spatial(
        clauses: List[str],
        params: List[Any],
        *,
        entity: str,
        coord_sys: Optional[str],
        ra: Optional[float],
        dec: Optional[float],
        gl: Optional[float],
        gb: Optional[float],
        radius: Optional[float],
        polygon: Any,
    ):
        """
        Append SQL clauses for cone / point-in-polygon / polygon-intersect
        searches, using PostGIS for polygon entities and Q3C for point
        entities. Silently no-ops when nothing spatial is requested.

        ``polygon`` may be a JSON string or a list of ``[lon, lat]`` pairs.
        """
        cs = (coord_sys or "radec").lower()
        if cs not in ("radec", "galactic"):
            raise ValueError(f"coord_sys must be 'radec' or 'galactic', got {cs!r}")

        # Pick the correct lon/lat values for the requested system.
        if cs == "radec":
            lon, lat = ra, dec
        else:
            lon, lat = gl, gb

        has_point = lon is not None and lat is not None
        has_polygon = polygon is not None

        if not has_point and not has_polygon:
            return  # nothing to do

        meta = _SPATIAL_COLUMNS.get(entity)
        if meta is None:
            raise NotImplementedError(f"Spatial filters are not supported for entity {entity!r}")

        point_cols = meta[f"point_{cs}"]
        poly_col = meta[f"poly_{cs}"]

        # PostGIS polygon columns in this schema are stored with SRID 4326
        # (WGS84). Any geometry we synthesize has to use the same SRID or
        # ``ST_Intersects`` raises "mixed SRID geometries".
        SRID = 4326

        # --- Cone / point search --------------------------------------- #
        if has_point and not has_polygon:
            if radius is not None:
                # Cone search. Use Q3C when we have a stored point; else
                # intersect a buffered point with the polygon.
                if point_cols is not None:
                    pcol_lon, pcol_lat = point_cols
                    clauses.append(f"q3c_radial_query({pcol_lon}, {pcol_lat}, %s, %s, %s)")
                    params.extend([float(lon), float(lat), float(radius)])
                elif poly_col is not None:
                    clauses.append(f"ST_DWithin({poly_col}, " f"ST_SetSRID(ST_MakePoint(%s, %s), {SRID}), %s)")
                    params.extend([float(lon), float(lat), float(radius)])
                else:
                    raise NotImplementedError(f"{entity!r} has no spatial column for coord_sys={cs!r}")
            else:
                # Point-in-polygon search (no radius).
                if poly_col is not None:
                    clauses.append(f"ST_Contains({poly_col}, " f"ST_SetSRID(ST_MakePoint(%s, %s), {SRID}))")
                    params.extend([float(lon), float(lat)])
                elif point_cols is not None:
                    # Fall back to a tiny 1 arcsec cone as a proxy.
                    pcol_lon, pcol_lat = point_cols
                    clauses.append(f"q3c_radial_query({pcol_lon}, {pcol_lat}, %s, %s, %s)")
                    params.extend([float(lon), float(lat), 1.0 / 3600.0])
                else:
                    raise NotImplementedError(f"{entity!r} has no spatial column for coord_sys={cs!r}")

        # --- Polygon-intersect search ---------------------------------- #
        if has_polygon:
            # Accept both JSON strings and python lists.
            if isinstance(polygon, str):
                try:
                    coords = json.loads(polygon)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Bad polygon JSON: {exc}") from exc
            else:
                coords = polygon
            if not coords or not isinstance(coords, list) or len(coords) < 3:
                raise ValueError("polygon must be a list of at least 3 [lon,lat] pairs")
            # Close the ring if necessary (PostGIS requires a closed ring).
            ring = list(coords)
            if ring[0] != ring[-1]:
                ring = ring + [ring[0]]
            wkt = "POLYGON((" + ",".join(f"{p[0]} {p[1]}" for p in ring) + "))"

            if poly_col is not None:
                clauses.append(f"ST_Intersects({poly_col}, ST_GeomFromText(%s, {SRID}))")
                params.append(wkt)
            elif point_cols is not None:
                pcol_lon, pcol_lat = point_cols
                clauses.append(f"q3c_poly_query({pcol_lon}, {pcol_lat}, %s::double precision[])")
                # Q3C expects a flat array: [lon1, lat1, lon2, lat2, ...].
                # Drop the closing vertex we just added; q3c doesn't need it.
                flat = [c for p in coords for c in (float(p[0]), float(p[1]))]
                params.append(flat)
            else:
                raise NotImplementedError(f"{entity!r} has no spatial column for coord_sys={cs!r}")

    # -- entity dispatchers ---------------------------------------------- #
    def query(self, entity: str, **f: Any) -> List[Dict[str, Any]]:
        fn = getattr(self, f"_q_{entity}", None)
        if fn is None:
            raise NotImplementedError(f"SQL backend does not support {entity!r}")
        return fn(**f)

    # ---- Raw science frames -------------------------------------------- #
    def _q_raw(
        self,
        *,
        date_start: Optional[DateLike] = None,
        date_end: Optional[DateLike] = None,
        days: Optional[int] = None,
        night_date: Optional[DateLike] = None,
        filter_name: ListOrOne = None,
        unit_name: ListOrOne = None,
        object_name: Optional[str] = None,
        obsnote_contains: Optional[str] = None,
        tile_name: ListOrOne = None,
        target_name: ListOrOne = None,
        coord_sys: Optional[str] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        gl: Optional[float] = None,
        gb: Optional[float] = None,
        radius: Optional[float] = None,
        polygon: Any = None,
        full_table: bool = False,
        **_ignored: Any,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []

        self._apply_date_filters(
            clauses,
            params,
            col="sf.obstime",
            date_start=date_start,
            date_end=date_end,
            days=days,
        )
        if night_date is not None:
            clauses.append("n.date = %s")
            params.append(_to_date(night_date))

        self._apply_in(clauses, params, "u.name", _as_list(unit_name))
        self._apply_in(clauses, params, "f.name", _as_list(filter_name))
        self._apply_in(clauses, params, "tl.name", _as_list(tile_name))

        self._apply_spatial(
            clauses,
            params,
            entity="raw",
            coord_sys=coord_sys,
            ra=ra,
            dec=dec,
            gl=gl,
            gb=gb,
            radius=radius,
            polygon=polygon,
        )

        if target_name is not None:
            names = _as_list(target_name) or []
            tile_tgts = [n for n in names if _TILE_RE.match(n)]
            reg_tgts = [n for n in names if not _TILE_RE.match(n)]
            sub: List[str] = []
            if reg_tgts:
                ph = ",".join(["%s"] * len(reg_tgts))
                sub.append(f"t.name IN ({ph})")
                params.extend(reg_tgts)
            if tile_tgts:
                ph = ",".join(["%s"] * len(tile_tgts))
                sub.append(f"sf.tile_id IN (SELECT id FROM survey_tile WHERE name IN ({ph}))")
                params.extend(tile_tgts)
            if sub:
                clauses.append("(" + " OR ".join(sub) + ")")

        if object_name:
            clauses.append("sf.object_name ILIKE %s")
            params.append(f"%{object_name}%")
        if obsnote_contains:
            clauses.append("sf.obsnote ILIKE %s")
            params.append(f"%{obsnote_contains}%")

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        if full_table:
            sf_cols = self._star(alias="sf", table="survey_scienceframe")
            cols = (
                f"{sf_cols},\n"
                "                sf.original_filename AS filename,\n"
                "                sf.file_path         AS filepath,\n"
                "                sf.fwhm              AS seeing,\n"
                "                u.name  AS unit,\n"
                "                f.name  AS filter,\n"
                "                t.name  AS target,\n"
                "                tl.name AS tile,\n"
                "                n.date  AS night"
            )
        else:
            cols = _LEAN_COLS_RAW
        sql = f"""
            SELECT
                {cols}
            FROM survey_scienceframe sf
            JOIN facility_unit u       ON sf.unit_id = u.id
            JOIN survey_night  n       ON sf.night_id = n.id
            LEFT JOIN facility_filter f ON sf.filter_id = f.id
            LEFT JOIN survey_target  t  ON sf.target_id = t.id
            LEFT JOIN survey_tile    tl ON sf.tile_id = tl.id
            {where}
            ORDER BY sf.obstime
        """
        return self._execute(sql, params)

    # ---- Processed science frames -------------------------------------- #
    def _q_processed(
        self,
        *,
        date_start: Optional[DateLike] = None,
        date_end: Optional[DateLike] = None,
        days: Optional[int] = None,
        night_date: Optional[DateLike] = None,
        filter_name: ListOrOne = None,
        unit_name: ListOrOne = None,
        tile_name: ListOrOne = None,
        target_name: ListOrOne = None,
        obsnote_contains: Optional[str] = None,
        data_stream: Optional[str] = None,
        processing_version: Optional[str] = None,
        is_too: Optional[bool] = None,
        is_production: Optional[bool] = None,
        coord_sys: Optional[str] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        gl: Optional[float] = None,
        gb: Optional[float] = None,
        radius: Optional[float] = None,
        polygon: Any = None,
        full_table: bool = False,
        **_ignored: Any,
    ) -> List[Dict[str, Any]]:
        """
        Processed frames have no direct filter/unit/night columns. Everything
        is derived from the underlying raw frame via ``raw_frame_id``.
        """
        clauses: List[str] = []
        params: List[Any] = []

        self._apply_date_filters(
            clauses,
            params,
            col="sf.obstime",
            date_start=date_start,
            date_end=date_end,
            days=days,
        )
        if night_date is not None:
            clauses.append("n.date = %s")
            params.append(_to_date(night_date))
        self._apply_in(clauses, params, "u.name", _as_list(unit_name))
        self._apply_in(clauses, params, "f.name", _as_list(filter_name))
        self._apply_in(clauses, params, "tl.name", _as_list(tile_name))
        if target_name is not None:
            self._apply_in(clauses, params, "t.name", _as_list(target_name))
        if obsnote_contains:
            clauses.append("sf.obsnote ILIKE %s")
            params.append(f"%{obsnote_contains}%")
        if data_stream:
            clauses.append("pf.data_stream = %s")
            params.append(data_stream)
        if processing_version:
            clauses.append("pf.processing_version = %s")
            params.append(processing_version)
        if is_too is not None:
            clauses.append("sf.is_too = %s")
            params.append(bool(is_too))
        if is_production is not None:
            clauses.append("pf.is_production = %s")
            params.append(bool(is_production))

        self._apply_spatial(
            clauses,
            params,
            entity="processed",
            coord_sys=coord_sys,
            ra=ra,
            dec=dec,
            gl=gl,
            gb=gb,
            radius=radius,
            polygon=polygon,
        )

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        if full_table:
            # All native columns of the processed-frame table plus the joined
            # name/date labels. Raw-frame columns are intentionally not
            # included here; if you need them, query ``RawFrameQuery``.
            pf_cols = self._star(alias="pf", table="survey_processedscienceframe")
            cols = (
                f"{pf_cols},\n"
                "                u.name  AS unit,\n"
                "                f.name  AS filter,\n"
                "                t.name  AS target,\n"
                "                tl.name AS tile,\n"
                "                n.date  AS night"
            )
        else:
            cols = _LEAN_COLS_PROCESSED
        sql = f"""
            SELECT
                {cols}
            FROM survey_processedscienceframe pf
            JOIN survey_scienceframe sf ON pf.raw_frame_id = sf.id
            JOIN facility_unit u        ON sf.unit_id = u.id
            JOIN survey_night  n        ON sf.night_id = n.id
            LEFT JOIN facility_filter f ON sf.filter_id = f.id
            LEFT JOIN survey_target  t  ON sf.target_id = t.id
            LEFT JOIN survey_tile    tl ON sf.tile_id = tl.id
            {where}
            ORDER BY sf.obstime
        """
        return self._execute(sql, params)

    # ---- Combined science frames --------------------------------------- #
    def _q_combined(
        self,
        *,
        date_start: Optional[DateLike] = None,
        date_end: Optional[DateLike] = None,
        days: Optional[int] = None,
        filter_name: ListOrOne = None,
        unit_name: ListOrOne = None,
        tile_name: ListOrOne = None,
        target_name: ListOrOne = None,
        data_stream: Optional[str] = None,
        processing_version: Optional[str] = None,
        is_production: Optional[bool] = None,
        coord_sys: Optional[str] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        gl: Optional[float] = None,
        gb: Optional[float] = None,
        radius: Optional[float] = None,
        polygon: Any = None,
        full_table: bool = False,
        **_ignored: Any,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []

        self._apply_date_filters(
            clauses,
            params,
            col="cf.obs_start",
            date_start=date_start,
            date_end=date_end,
            days=days,
        )
        self._apply_in(clauses, params, "u.name", _as_list(unit_name))
        self._apply_in(clauses, params, "f.name", _as_list(filter_name))
        self._apply_in(clauses, params, "tl.name", _as_list(tile_name))
        if target_name is not None:
            self._apply_in(clauses, params, "t.name", _as_list(target_name))
        if data_stream:
            clauses.append("cf.data_stream = %s")
            params.append(data_stream)
        if processing_version:
            clauses.append("cf.processing_version = %s")
            params.append(processing_version)
        if is_production is not None:
            clauses.append("cf.is_production = %s")
            params.append(bool(is_production))

        self._apply_spatial(
            clauses,
            params,
            entity="combined",
            coord_sys=coord_sys,
            ra=ra,
            dec=dec,
            gl=gl,
            gb=gb,
            radius=radius,
            polygon=polygon,
        )

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        if full_table:
            cf_cols = self._star(alias="cf", table="survey_combinedscienceframe")
            cols = (
                f"{cf_cols},\n"
                "                u.name  AS unit,\n"
                "                f.name  AS filter,\n"
                "                t.name  AS target,\n"
                "                tl.name AS tile"
            )
        else:
            cols = _LEAN_COLS_COMBINED
        sql = f"""
            SELECT
                {cols}
            FROM survey_combinedscienceframe cf
            JOIN facility_unit u        ON cf.unit_id = u.id
            LEFT JOIN facility_filter f ON cf.filter_id = f.id
            LEFT JOIN survey_target  t  ON cf.target_id = t.id
            LEFT JOIN survey_tile    tl ON cf.tile_id = tl.id
            {where}
            ORDER BY cf.obs_start
        """
        return self._execute(sql, params)

    # ---- TOO variants are the same tables + is_too/data_stream filter -- #
    def _q_processed_too(self, **f: Any) -> List[Dict[str, Any]]:
        f.setdefault("data_stream", "too")
        return self._q_processed(**f)

    def _q_combined_too(self, **f: Any) -> List[Dict[str, Any]]:
        f.setdefault("data_stream", "too")
        return self._q_combined(**f)

    # ---- Raw individual calibration frames ----------------------------- #
    # These are the rows of ``survey_biasframe`` / ``survey_darkframe`` /
    # ``survey_flatframe`` -- the raw calibration exposures, NOT the stacked
    # masters (which live in ``survey_master*frame`` and are exposed via
    # ``master_bias`` / ``master_dark`` / ``master_flat`` above).
    def _q_bias(self, **f: Any) -> List[Dict[str, Any]]:
        return self._q_raw_calib("bias", **f)

    def _q_dark(self, **f: Any) -> List[Dict[str, Any]]:
        return self._q_raw_calib("dark", **f)

    def _q_flat(self, **f: Any) -> List[Dict[str, Any]]:
        return self._q_raw_calib("flat", **f)

    def _q_raw_calib(
        self,
        kind: str,
        *,
        date_start: Optional[DateLike] = None,
        date_end: Optional[DateLike] = None,
        days: Optional[int] = None,
        night_date: Optional[DateLike] = None,
        unit_name: ListOrOne = None,
        filter_name: ListOrOne = None,  # flat only
        exptime: Optional[float] = None,  # dark mostly; also useful for flat
        min_exptime: Optional[float] = None,
        max_exptime: Optional[float] = None,
        binning: Optional[int] = None,
        gain: Optional[int] = None,
        is_usable: Optional[bool] = None,
        full_table: bool = False,
        **_ignored: Any,
    ) -> List[Dict[str, Any]]:
        """
        Shared SQL for raw bias/dark/flat frames. Kind-specific columns are
        appended to the SELECT list.
        """
        table = {
            "bias": "survey_biasframe",
            "dark": "survey_darkframe",
            "flat": "survey_flatframe",
        }[kind]

        clauses: List[str] = []
        params: List[Any] = []

        self._apply_date_filters(
            clauses, params, col="cf.obstime",
            date_start=date_start, date_end=date_end, days=days,
        )
        if night_date is not None:
            clauses.append("n.date = %s")
            params.append(_to_date(night_date))
        self._apply_in(clauses, params, "u.name", _as_list(unit_name))

        if exptime is not None:
            clauses.append("cf.exptime = %s")
            params.append(float(exptime))
        if min_exptime is not None:
            clauses.append("cf.exptime >= %s")
            params.append(float(min_exptime))
        if max_exptime is not None:
            clauses.append("cf.exptime <= %s")
            params.append(float(max_exptime))
        if binning is not None:
            clauses.append("cf.binning_x = %s AND cf.binning_y = %s")
            params.extend([int(binning), int(binning)])
        if gain is not None:
            clauses.append("cf.gain = %s")
            params.append(int(gain))
        if is_usable is not None:
            clauses.append("cf.is_usable = %s")
            params.append(bool(is_usable))

        # Flats have a filter_id; bias and dark do not.
        filter_join = ""
        filter_select = ""
        if kind == "flat":
            filter_join = "LEFT JOIN facility_filter f ON cf.filter_id = f.id"
            filter_select = ", f.name AS filter"
            if filter_name is not None:
                self._apply_in(clauses, params, "f.name", _as_list(filter_name))
        elif filter_name is not None:
            raise ValueError(
                f"{kind!r} frames have no filter; drop filter_name."
            )

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        if full_table:
            cf_cols = self._star(alias="cf", table=table)
            cols = (
                f"{cf_cols},\n"
                "                cf.original_filename AS filename,\n"
                "                cf.file_path         AS filepath,\n"
                "                u.name AS unit,\n"
                f"                n.date AS night{filter_select}"
            )
        else:
            cols = (
                f"{_LEAN_COLS_RAW_CALIB.rstrip()},"
                f"\n                {_LEAN_CALIB_EXTRAS[kind]}"
                f"{filter_select}"
            )
        sql = f"""
            SELECT
                {cols}
            FROM {table} cf
            JOIN facility_unit u ON cf.unit_id = u.id
            JOIN survey_night  n ON cf.night_id = n.id
            {filter_join}
            {where}
            ORDER BY cf.obstime
        """
        return self._execute(sql, params)

    # ---- Tiles and targets --------------------------------------------- #
    def _q_tile(
        self,
        *,
        name: ListOrOne = None,
        priority: Optional[int] = None,
        survey_program: Optional[str] = None,
        coord_sys: Optional[str] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        gl: Optional[float] = None,
        gb: Optional[float] = None,
        radius: Optional[float] = None,
        polygon: Any = None,
        full_table: bool = False,
        **_ignored: Any,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        self._apply_in(clauses, params, "t.name", _as_list(name))
        if priority is not None:
            clauses.append("t.priority = %s")
            params.append(int(priority))
        if survey_program:
            clauses.append("t.survey_program = %s")
            params.append(survey_program)

        self._apply_spatial(
            clauses,
            params,
            entity="tile",
            coord_sys=coord_sys,
            ra=ra,
            dec=dec,
            gl=gl,
            gb=gb,
            radius=radius,
            polygon=polygon,
        )

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        cols = self._star(alias="t", table="survey_tile") if full_table else _LEAN_COLS_TILE
        sql = f"""
            SELECT {cols}
            FROM survey_tile t
            {where}
            ORDER BY t.name
        """
        return self._execute(sql, params)

    def _q_target(
        self,
        *,
        name: ListOrOne = None,
        target_type: Optional[str] = None,
        coord_sys: Optional[str] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        gl: Optional[float] = None,
        gb: Optional[float] = None,
        radius: Optional[float] = None,
        polygon: Any = None,
        full_table: bool = False,
        **_ignored: Any,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        self._apply_in(clauses, params, "t.name", _as_list(name))
        if target_type:
            clauses.append("t.target_type = %s")
            params.append(target_type)

        self._apply_spatial(
            clauses,
            params,
            entity="target",
            coord_sys=coord_sys,
            ra=ra,
            dec=dec,
            gl=gl,
            gb=gb,
            radius=radius,
            polygon=polygon,
        )

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        cols = self._star(alias="t", table="survey_target") if full_table else _LEAN_COLS_TARGET
        sql = f"""
            SELECT {cols}
            FROM survey_target t
            {where}
            ORDER BY t.name
        """
        return self._execute(sql, params)

    # ---- Master calibrations ------------------------------------------- #
    def _q_master_bias(self, **f: Any) -> List[Dict[str, Any]]:
        return self._q_master("bias", **f)

    def _q_master_dark(self, **f: Any) -> List[Dict[str, Any]]:
        return self._q_master("dark", **f)

    def _q_master_flat(self, **f: Any) -> List[Dict[str, Any]]:
        return self._q_master("flat", **f)

    def _q_master(
        self,
        kind: str,
        *,
        unit_name: ListOrOne = None,
        nightdate: Optional[DateLike] = None,
        date_start: Optional[DateLike] = None,
        date_end: Optional[DateLike] = None,
        binning: Optional[int] = None,
        gain: Optional[int] = None,
        camera_serial: Optional[str] = None,
        processing_version: Optional[str] = None,
        is_production: Optional[bool] = None,
        filter_name: Optional[str] = None,
        exptime: Optional[float] = None,
        min_exptime: Optional[float] = None,
        max_exptime: Optional[float] = None,
        full_table: bool = False,
        **_ignored: Any,
    ) -> List[Dict[str, Any]]:
        table = {
            "bias": "survey_masterbiasframe",
            "dark": "survey_masterdarkframe",
            "flat": "survey_masterflatframe",
        }[kind]
        clauses: List[str] = []
        params: List[Any] = []
        self._apply_in(clauses, params, "u.name", _as_list(unit_name))
        if nightdate is not None:
            clauses.append("m.nightdate = %s")
            params.append(_to_date(nightdate))
        if date_start is not None:
            clauses.append("m.nightdate >= %s")
            params.append(_to_date(date_start))
        if date_end is not None:
            clauses.append("m.nightdate <= %s")
            params.append(_to_date(date_end))
        if binning is not None:
            clauses.append("m.binning_x = %s AND m.binning_y = %s")
            params.extend([int(binning), int(binning)])
        if gain is not None:
            clauses.append("m.gain = %s")
            params.append(int(gain))
        if camera_serial:
            clauses.append("m.camera_serial = %s")
            params.append(camera_serial)
        if processing_version:
            clauses.append("m.processing_version = %s")
            params.append(processing_version)
        if is_production is not None:
            clauses.append("m.is_production = %s")
            params.append(bool(is_production))

        if kind == "dark":
            if exptime is not None:
                clauses.append("m.exptime = %s")
                params.append(float(exptime))
            if min_exptime is not None:
                clauses.append("m.exptime >= %s")
                params.append(float(min_exptime))
            if max_exptime is not None:
                clauses.append("m.exptime <= %s")
                params.append(float(max_exptime))
        if kind == "flat" and filter_name:
            clauses.append("f.name = %s")
            params.append(filter_name)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        join_filter = (
            "LEFT JOIN facility_filter f ON m.filter_id = f.id"
            if kind == "flat"
            else ""
        )
        if full_table:
            m_cols = self._star(alias="m", table=table)
            if kind == "flat":
                cols = f"{m_cols}, u.name AS unit, f.name AS filter"
            else:
                cols = f"{m_cols}, u.name AS unit"
        else:
            cols = _LEAN_COLS_MASTER[kind]

        sql = f"""
            SELECT {cols}
            FROM {table} m
            JOIN facility_unit u ON m.unit_id = u.id
            {join_filter}
            {where}
            ORDER BY m.nightdate DESC, m.id
        """
        return self._execute(sql, params)


# --------------------------------------------------------------------------- #
# HTTP backend
# --------------------------------------------------------------------------- #


class _HttpBackend:
    """Adapter over ``GWPortalClient`` that exposes the same ``query()`` API."""

    def __init__(self, client: Optional[GWPortalClient] = None):
        self.client = client or GWPortalClient()
        # Introspection counterparts of _SqlBackend.last_sql / last_params.
        self.last_url: Optional[str] = None
        self.last_params: Optional[Dict[str, Any]] = None
        self.dry_run: bool = False

    # Entities that have no REST endpoint and therefore can only be queried
    # through the SQL backend.
    _SQL_ONLY_ENTITIES = frozenset({"bias", "dark", "flat"})

    def query(self, entity: str, **filters: Any) -> List[Dict[str, Any]]:
        if entity in self._SQL_ONLY_ENTITIES:
            raise NotImplementedError(
                f"The GWPortal REST API has no endpoint for raw {entity} "
                "frames; use Backend.SQL (or Backend.AUTO with SQL "
                "available)."
            )
        params = self._translate(entity, filters)
        # Capture the would-be request so callers can inspect it.
        self.last_params = dict(params)
        try:
            endpoint = self.client.ENDPOINTS.get(entity, entity)
            base = (self.client.base_url or "").rstrip("/")
            self.last_url = f"{base}/api/{endpoint.lstrip('/')}"
        except Exception:
            self.last_url = None
        if self.dry_run:
            return []
        return self.client.fetch_all(entity, **params)

    def request_str(self) -> str:
        """Human-readable summary of the last HTTP request."""
        if not self.last_url:
            return ""
        try:
            from urllib.parse import urlencode
            qs = urlencode(self.last_params or {}, doseq=True)
        except Exception:
            qs = repr(self.last_params)
        return f"GET {self.last_url}?{qs}" if qs else f"GET {self.last_url}"

    @staticmethod
    def _translate(entity: str, f: Dict[str, Any]) -> Dict[str, Any]:
        """Map our common filter vocabulary to the REST API's parameter names."""
        out: Dict[str, Any] = {}

        # Direct passthroughs
        passthrough = {
            "days",
            "date_start",
            "date_end",
            "night_date",
            "filter_name",
            "unit_name",
            "tile_name",
            "target_name",
            "object_name",
            "obsnote_contains",
            "target_type",
            "coord_sys",
            "ra",
            "dec",
            "gl",
            "gb",
            "radius",
            "polygon",
            "data_stream",
            "processing_version",
            "binning",
            "gain",
            "camera_serial",
            "nightdate",
            "exptime",
            "min_exptime",
            "max_exptime",
            "offset",
            "neighbors",
            "exptime_near",
            "all_versions",
        }
        for k in passthrough:
            if k in f and f[k] is not None:
                out[k] = f[k]

        # Tile/Target endpoints expect ``name`` rather than ``tile_name``/
        # ``target_name`` when querying the entity directly.
        if entity == "tile" and "name" in f and f["name"] is not None:
            out["tile_name"] = f["name"]
        if entity == "target" and "name" in f and f["name"] is not None:
            out["target_name"] = f["name"]

        # Some params are list-like in SQL but the API prefers scalars.
        # Drop the list form and let the caller iterate if they need to.
        for k in ("filter_name", "unit_name", "tile_name", "target_name"):
            if k in out and isinstance(out[k], (list, tuple)):
                if len(out[k]) == 1:
                    out[k] = out[k][0]
                else:
                    # Keep first; the REST API does not support IN (...) here.
                    # Callers who need multi-value filters should prefer SQL.
                    out[k] = out[k][0]

        # ``is_production`` is not in the REST API for master endpoints directly;
        # they use ``production_only``.
        if entity.startswith("master_") and "is_production" in f and f["is_production"]:
            out["production_only"] = "true"

        # Normalize polygon: ensure the ring is closed (REST server's GEOS is
        # strict) and JSON-encode to string.
        if "polygon" in out:
            p = out["polygon"]
            if isinstance(p, str):
                try:
                    p = json.loads(p)
                except json.JSONDecodeError:
                    # Leave string as-is if we can't parse; server will
                    # give a clearer error than we can.
                    p = None
            if isinstance(p, list) and len(p) >= 3:
                ring = list(p)
                if ring[0] != ring[-1]:
                    ring = ring + [ring[0]]
                out["polygon"] = json.dumps(ring)
            elif isinstance(out["polygon"], str):
                # Keep the original string; nothing else we can do.
                pass

        return out


# --------------------------------------------------------------------------- #
# Unified connector
# --------------------------------------------------------------------------- #


class GWPortalQuery:
    """
    Unified query object spanning all entities and both backends.

    Example
    -------
    >>> q = GWPortalQuery("processed")
    >>> rows = q.query(date_start="2025-10-01", filter_name="m525")
    >>> tbl  = q.query_table(date_start="2025-10-01", filter_name="m525")

    Parameters
    ----------
    entity : str
        One of ``raw``, ``processed``, ``combined``, ``processed_too``,
        ``combined_too``, ``tile``, ``target``, ``master_bias``,
        ``master_dark``, ``master_flat`` (plus common aliases).
    backend : str, default ``"auto"``
        ``"auto"`` tries SQL first and falls back to HTTP. Use ``"sql"`` or
        ``"http"`` to force a transport.
    http_client : GWPortalClient, optional
        A pre-built REST client. If omitted one is created lazily using the
        credentials in ``.env``.
    """

    def __init__(
        self,
        entity: str = "raw",
        backend: Union[Backend, str] = Backend.AUTO,
        http_client: Optional[GWPortalClient] = None,
    ):
        self.entity = _normalize_entity(entity)
        self.backend = Backend.parse(backend)
        self._http_client = http_client
        self._sql: Optional[_SqlBackend] = None
        self._http: Optional[_HttpBackend] = None
        self.last_backend_used: Optional[Backend] = None
        # Introspection populated after each call to :meth:`query`.
        self.last_sql: Optional[str] = None
        self.last_params: Optional[Any] = None
        self.last_url: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Backend availability
    # ------------------------------------------------------------------ #
    def _sql_available(self) -> bool:
        if not all(DB_PARAMS.get(k) for k in ("dbname", "user", "host")):
            return False
        return check_db_connection()

    def _http_available(self) -> bool:
        return bool(GWPORTAL_BASE_URL and GWPORTAL_API_KEY)

    def _get_sql(self) -> _SqlBackend:
        if self._sql is None:
            self._sql = _SqlBackend()
        return self._sql

    def _get_http(self) -> _HttpBackend:
        if self._http is None:
            self._http = _HttpBackend(self._http_client)
        return self._http

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def query(self, *, dry_run: bool = False, **filters: Any) -> List[Dict[str, Any]]:
        """
        Run the query and return a list of dict rows.

        Parameters
        ----------
        dry_run : bool
            If True, build and capture the backend request (SQL or HTTP URL)
            without actually executing it. The captured info is available on
            ``self.last_sql`` / ``self.last_params`` / ``self.last_url``.
        """
        order = self._backend_order()
        errors: List[Tuple[Backend, Exception]] = []
        for be in order:
            try:
                if be is Backend.SQL:
                    sql_be = self._get_sql()
                    sql_be.dry_run = dry_run
                    try:
                        rows = sql_be.query(self.entity, **filters)
                    finally:
                        # Propagate introspection regardless of outcome.
                        self.last_sql = sql_be.last_sql
                        self.last_params = sql_be.last_params
                        self.last_url = None
                        sql_be.dry_run = False
                else:
                    http_be = self._get_http()
                    http_be.dry_run = dry_run
                    try:
                        rows = http_be.query(self.entity, **filters)
                    finally:
                        self.last_url = http_be.last_url
                        self.last_params = http_be.last_params
                        self.last_sql = None
                        http_be.dry_run = False
                self.last_backend_used = be
                return rows
            except Exception as exc:  # noqa: BLE001 - want broad fallback
                errors.append((be, exc))
        # Exhausted all backends. When the caller asked for exactly one
        # backend, re-raise the underlying exception unchanged so the type
        # (e.g. NotImplementedError) is preserved.
        if len(errors) == 1:
            raise errors[0][1]
        details = "; ".join(f"{b.value}: {e}" for b, e in errors)
        raise RuntimeError(f"All backends failed for entity={self.entity!r}. {details}")

    def query_table(self, **filters: Any):
        """Same as :meth:`query` but returns an ``astropy.table.Table``."""
        from astropy.table import Table

        rows = self.query(**filters)
        return Table(rows) if rows else Table()

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def explain(self, *, interpolate: bool = True) -> str:
        """
        Return a human-readable description of the most recent query.

        * SQL backend: the SQL statement, with placeholders substituted when
          ``interpolate=True`` (via psycopg's ``ClientCursor.mogrify``).
        * HTTP backend: ``GET <url>?<querystring>``.

        Returns an empty string if nothing has been executed yet.
        """
        if self.last_backend_used is Backend.SQL and self.last_sql is not None:
            if interpolate and self._sql is not None:
                return self._sql.mogrify(self.last_sql, self.last_params)
            return f"{self.last_sql}\n-- params: {self.last_params!r}"
        if self.last_backend_used is Backend.HTTP and self._http is not None:
            return self._http.request_str()
        return ""

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #
    def _backend_order(self) -> List[Backend]:
        if self.backend is Backend.SQL:
            return [Backend.SQL]
        if self.backend is Backend.HTTP:
            return [Backend.HTTP]
        # AUTO: prefer SQL, fall back to HTTP.
        order: List[Backend] = []
        if self._sql_available():
            order.append(Backend.SQL)
        if self._http_available():
            order.append(Backend.HTTP)
        if not order:
            raise RuntimeError(
                "No backend available. Configure Postgres credentials "
                "(DBNAME/DBUSER/DBHOST/DBPASSWORD) or set GWPORTAL_BASE_URL "
                "and GWPORTAL_API_KEY in your .env."
            )
        return order

    def __repr__(self) -> str:
        return f"GWPortalQuery(entity={self.entity!r}, backend={self.backend.value!r})"


# --------------------------------------------------------------------------- #
# Fluent per-entity builders
# --------------------------------------------------------------------------- #


@dataclass
class _BaseQueryBuilder:
    """Shared state for every fluent builder."""

    entity: str = field(init=False, default="")
    backend: Union[Backend, str] = Backend.AUTO
    _filters: Dict[str, Any] = field(default_factory=dict)
    _results: Optional[List[Dict[str, Any]]] = field(default=None, repr=False)
    _query: Optional["GWPortalQuery"] = field(default=None, repr=False)
    last_backend_used: Optional[Backend] = field(default=None, repr=False)
    last_sql: Optional[str] = field(default=None, repr=False)
    last_params: Optional[Any] = field(default=None, repr=False)
    last_url: Optional[str] = field(default=None, repr=False)

    # -- shared fluent methods --------------------------------------- #
    def where(self, **filters: Any):
        """Escape hatch: set any filter directly."""
        self._filters.update({k: v for k, v in filters.items() if v is not None})
        return self

    def full_table(self, flag: bool = True):
        """
        Return every native column of the underlying table on the next
        ``fetch()`` / ``fetch_table()``. Default is the lean, hand-picked
        column set which is noticeably faster for large result sets.

        Only affects the SQL backend; the HTTP backend's column set is
        controlled server-side.
        """
        if flag:
            self._filters["full_table"] = True
        else:
            self._filters.pop("full_table", None)
        return self

    def reset(self):
        self._filters.clear()
        self._results = None
        self._query = None
        self.last_backend_used = None
        self.last_sql = None
        self.last_params = None
        self.last_url = None
        return self

    def _run(self, *, dry_run: bool = False) -> List[Dict[str, Any]]:
        """Shared SQL/HTTP dispatch that also copies introspection state."""
        q = GWPortalQuery(self.entity, backend=self.backend)
        try:
            rows = q.query(dry_run=dry_run, **self._filters)
        finally:
            self._query = q
            self.last_backend_used = q.last_backend_used
            self.last_sql = q.last_sql
            self.last_params = q.last_params
            self.last_url = q.last_url
        return rows

    def fetch(self, *, full_table: bool = False) -> List[Dict[str, Any]]:
        """
        Execute the query and return rows.

        Pass ``full_table=True`` to include every native column of the
        underlying table. This is a one-shot shortcut for
        ``self.full_table().fetch()``.
        """
        if full_table:
            self._filters["full_table"] = True
        self._results = self._run()
        return self._results

    def fetch_table(self, *, full_table: bool = False):
        """
        Same as :meth:`fetch` but returns an ``astropy.table.Table``.

        Pass ``full_table=True`` for every native column.
        """
        from astropy.table import Table

        rows = self.fetch(full_table=full_table)
        return Table(rows) if rows else Table()

    def files(self, key: Optional[str] = None) -> List[str]:
        """Return the list of file paths from the most recent fetch."""
        if self._results is None:
            self.fetch()
        if key is None:
            # Prefer 'filepath' (REST/SQL processed/combined) then 'file_path'
            # (master frames) then 'filename'.
            for k in ("filepath", "file_path", "filename"):
                if self._results and k in self._results[0]:
                    key = k
                    break
        return [r.get(key) for r in (self._results or []) if r.get(key)]

    # -- introspection ------------------------------------------------ #
    def sql(self, *, interpolate: bool = True, dry_run: bool = True) -> str:
        """
        Return the backend query as a string.

        By default this is a *dry run*: the query is built (and, for SQL,
        placeholders are substituted by the database driver) but nothing is
        executed. Pass ``dry_run=False`` to re-use the statement from the
        most recent :meth:`fetch` call.

        Works for both backends:

        * SQL  -> returns the ``SELECT`` statement. With ``interpolate=True``
          (default) placeholders are substituted via psycopg's
          ``ClientCursor.mogrify`` so the string is copy/paste-ready for
          ``psql``.
        * HTTP -> returns ``GET <url>?<querystring>``.
        """
        if dry_run or self._query is None:
            self._run(dry_run=True)
        if self.last_backend_used is Backend.HTTP:
            return self._query.explain() if self._query is not None else (self.last_url or "")
        if self.last_sql is None:
            return ""
        if interpolate and self._query is not None:
            return self._query.explain(interpolate=True)
        return f"{self.last_sql}\n-- params: {self.last_params!r}"

    def print_sql(self, *, interpolate: bool = True, dry_run: bool = True) -> None:
        """Convenience: print the result of :meth:`sql`."""
        print(self.sql(interpolate=interpolate, dry_run=dry_run))

    def explain(self, *, interpolate: bool = True) -> str:
        """Describe the most recent executed query (no dry run)."""
        return self.sql(interpolate=interpolate, dry_run=False)


# ---------- Mixins for common filter groups --------------------------- #


class _DateFilterMixin:
    def on_date(self, night: DateLike):
        self._filters["night_date"] = _to_date(night)
        return self

    def between(self, start: DateLike, end: DateLike):
        self._filters["date_start"] = _to_date(start)
        self._filters["date_end"] = _to_date(end)
        return self

    def since(self, days: int):
        self._filters["days"] = int(days)
        return self


class _FrameFilterMixin:
    def by_units(self, units: ListOrOne):
        self._filters["unit_name"] = _as_list(units) or None
        return self

    def by_unit(self, unit: str):
        self._filters["unit_name"] = unit
        return self

    def with_filter(self, filt: ListOrOne):
        self._filters["filter_name"] = _as_list(filt) or None
        return self

    def for_tile(self, tile: ListOrOne):
        self._filters["tile_name"] = _as_list(tile) or None
        return self

    def for_target(self, target: ListOrOne):
        self._filters["target_name"] = _as_list(target) or None
        return self

    def obsnote(self, text: str):
        self._filters["obsnote_contains"] = text
        return self


class _SpatialFilterMixin:
    """Spatial builder methods used by frame, tile, and target queries."""

    def cone_search(
        self,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        radius: float = 1.0,
        *,
        l: Optional[float] = None,
        b: Optional[float] = None,
    ):
        """
        Cone search around a point.

        Call with either ``(ra, dec)`` in degrees **or** ``(gl, gb)``. Radius is
        in degrees. If both a RA/Dec pair and an L/B pair are given, RA/Dec
        takes precedence.
        """
        if ra is not None and dec is not None:
            self._filters["coord_sys"] = "radec"
            self._filters["ra"] = float(ra)
            self._filters["dec"] = float(dec)
        elif l is not None and b is not None:
            self._filters["coord_sys"] = "galactic"
            self._filters["gl"] = float(l)
            self._filters["gb"] = float(b)
        else:
            raise ValueError("cone_search requires either (ra, dec) or (gl, gb)")
        self._filters["radius"] = float(radius)
        return self

    def at_point(
        self,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        *,
        gl: Optional[float] = None,
        gb: Optional[float] = None,
    ):
        """Point-in-polygon search (no radius). See :meth:`cone_search`."""
        if ra is not None and dec is not None:
            self._filters["coord_sys"] = "radec"
            self._filters["ra"] = float(ra)
            self._filters["dec"] = float(dec)
        elif gl is not None and gb is not None:
            self._filters["coord_sys"] = "galactic"
            self._filters["gl"] = float(gl)
            self._filters["gb"] = float(gb)
        else:
            raise ValueError("at_point requires either (ra, dec) or (gl, gb)")
        self._filters["radius"] = None
        return self

    def in_polygon(
        self,
        vertices: Sequence[Sequence[float]],
        coord_sys: str = "radec",
    ):
        """
        Polygon-intersection search.

        ``vertices`` is a list of ``[lon, lat]`` pairs, in the coordinate
        system named by ``coord_sys`` (``'radec'`` or ``'galactic'``).
        """
        self._filters["coord_sys"] = coord_sys
        self._filters["polygon"] = list(vertices)
        return self


# ---------- Concrete builders ----------------------------------------- #


class RawFrameQuery(_BaseQueryBuilder, _DateFilterMixin, _FrameFilterMixin, _SpatialFilterMixin):
    """Fluent builder for raw science frames."""

    entity: str = "raw"

    def __init__(self, backend: Union[Backend, str] = Backend.AUTO):
        super().__init__(backend=backend)

    def object_name_contains(self, text: str):
        self._filters["object_name"] = text
        return self


class ProcessedFrameQuery(_BaseQueryBuilder, _DateFilterMixin, _FrameFilterMixin, _SpatialFilterMixin):
    entity: str = "processed"

    def __init__(self, backend: Union[Backend, str] = Backend.AUTO):
        super().__init__(backend=backend)

    def data_stream(self, stream: str):
        self._filters["data_stream"] = stream
        return self

    def version(self, v: str):
        self._filters["processing_version"] = v
        return self

    def production_only(self, flag: bool = True):
        self._filters["is_production"] = flag
        return self


class CombinedFrameQuery(_BaseQueryBuilder, _DateFilterMixin, _FrameFilterMixin, _SpatialFilterMixin):
    entity: str = "combined"

    def __init__(self, backend: Union[Backend, str] = Backend.AUTO):
        super().__init__(backend=backend)

    def data_stream(self, stream: str):
        self._filters["data_stream"] = stream
        return self

    def version(self, v: str):
        self._filters["processing_version"] = v
        return self

    def production_only(self, flag: bool = True):
        self._filters["is_production"] = flag
        return self


class ProcessedTooQuery(ProcessedFrameQuery):
    entity: str = "processed_too"


class CombinedTooQuery(CombinedFrameQuery):
    entity: str = "combined_too"

    def all_versions(self, flag: bool = True):
        self._filters["all_versions"] = "true" if flag else None
        return self


class TileQuery(_BaseQueryBuilder, _SpatialFilterMixin):
    entity: str = "tile"

    def __init__(self, backend: Union[Backend, str] = Backend.AUTO):
        super().__init__(backend=backend)

    def named(self, name: ListOrOne):
        self._filters["name"] = _as_list(name) or None
        return self

    def with_priority(self, p: int):
        self._filters["priority"] = int(p)
        return self


class TargetQuery(_BaseQueryBuilder, _SpatialFilterMixin):
    entity: str = "target"

    def __init__(self, backend: Union[Backend, str] = Backend.AUTO):
        super().__init__(backend=backend)

    def named(self, name: ListOrOne):
        self._filters["name"] = _as_list(name) or None
        return self

    def of_type(self, target_type: str):
        self._filters["target_type"] = target_type
        return self


class _BaseMasterQuery(_BaseQueryBuilder):
    """Common fluent methods shared by master bias/dark/flat."""

    def by_unit(self, unit: str):
        self._filters["unit_name"] = unit
        return self

    def on_nightdate(self, d: DateLike):
        self._filters["nightdate"] = _to_date(d)
        return self

    def between(self, start: DateLike, end: DateLike):
        self._filters["date_start"] = _to_date(start)
        self._filters["date_end"] = _to_date(end)
        return self

    def with_camera(self, serial: str):
        self._filters["camera_serial"] = serial
        return self

    def with_binning(self, b: int):
        self._filters["binning"] = int(b)
        return self

    def with_gain(self, g: int):
        self._filters["gain"] = int(g)
        return self

    def version(self, v: str):
        self._filters["processing_version"] = v
        return self

    def production_only(self, flag: bool = True):
        self._filters["is_production"] = flag
        return self


class MasterBiasQuery(_BaseMasterQuery):
    entity: str = "master_bias"

    def __init__(self, backend: Union[Backend, str] = Backend.AUTO):
        super().__init__(backend=backend)


class MasterDarkQuery(_BaseMasterQuery):
    entity: str = "master_dark"

    def __init__(self, backend: Union[Backend, str] = Backend.AUTO):
        super().__init__(backend=backend)

    def with_exptime(self, exptime: float):
        self._filters["exptime"] = float(exptime)
        return self

    def exptime_between(self, lo: float, hi: float):
        self._filters["min_exptime"] = float(lo)
        self._filters["max_exptime"] = float(hi)
        return self


class MasterFlatQuery(_BaseMasterQuery):
    entity: str = "master_flat"

    def __init__(self, backend: Union[Backend, str] = Backend.AUTO):
        super().__init__(backend=backend)

    def with_filter(self, filter_name: str):
        self._filters["filter_name"] = filter_name
        return self


# --------------------------------------------------------------------------- #
# Raw individual calibration-frame builders (SQL-only)
# --------------------------------------------------------------------------- #
# These query the *raw* bias / dark / flat exposures in
# ``survey_biasframe`` / ``survey_darkframe`` / ``survey_flatframe``.
# The REST API does not expose these tables, so they force Backend.SQL.


class _RawCalibFilterMixin(_DateFilterMixin):
    """Filters shared by raw bias / dark / flat builders."""

    def by_units(self, units: ListOrOne):
        self._filters["unit_name"] = _as_list(units) or None
        return self

    def by_unit(self, unit: str):
        self._filters["unit_name"] = unit
        return self

    def with_exptime(self, exptime: float):
        self._filters["exptime"] = float(exptime)
        return self

    def exptime_between(self, lo: float, hi: float):
        self._filters["min_exptime"] = float(lo)
        self._filters["max_exptime"] = float(hi)
        return self

    def with_binning(self, b: int):
        self._filters["binning"] = int(b)
        return self

    def with_gain(self, g: int):
        self._filters["gain"] = int(g)
        return self

    def usable_only(self, flag: bool = True):
        self._filters["is_usable"] = bool(flag)
        return self


class _RawCalibQueryBase(_BaseQueryBuilder, _RawCalibFilterMixin):
    """Base for raw calibration builders. Always runs through SQL."""

    def __init__(self, backend: Union[Backend, str] = Backend.SQL):
        # Accept the argument for API symmetry but force SQL: the REST API
        # has no endpoint for these tables. ``AUTO`` is upgraded to ``SQL``
        # silently; explicit ``HTTP`` will error out at query time with a
        # clear message from ``_HttpBackend``.
        b = Backend(backend) if isinstance(backend, str) else backend
        if b is Backend.AUTO:
            b = Backend.SQL
        super().__init__(backend=b)


class BiasFrameQuery(_RawCalibQueryBase):
    """
    Query raw bias frames (``survey_biasframe``).

    Example::

        BiasFrameQuery().on_date("2024-05-01").by_unit("7DT01").fetch()
    """

    entity: str = "bias"


class DarkFrameQuery(_RawCalibQueryBase):
    """
    Query raw dark frames (``survey_darkframe``).

    Example::

        (DarkFrameQuery()
            .between("2024-05-01", "2024-05-31")
            .with_exptime(300)
            .fetch_table())
    """

    entity: str = "dark"


class FlatFrameQuery(_RawCalibQueryBase):
    """
    Query raw flat frames (``survey_flatframe``).

    Unlike bias/dark, flats carry a filter.

    Example::

        (FlatFrameQuery()
            .on_date("2024-05-01")
            .with_filter("g")
            .by_units(["7DT01", "7DT02"])
            .fetch())
    """

    entity: str = "flat"

    def with_filter(self, filt: ListOrOne):
        self._filters["filter_name"] = _as_list(filt) or None
        return self


# --------------------------------------------------------------------------- #
# Public exports
# --------------------------------------------------------------------------- #

__all__ = [
    "Backend",
    "GWPortalQuery",
    "RawFrameQuery",
    "ProcessedFrameQuery",
    "CombinedFrameQuery",
    "ProcessedTooQuery",
    "CombinedTooQuery",
    "TileQuery",
    "TargetQuery",
    "MasterBiasQuery",
    "MasterDarkQuery",
    "MasterFlatQuery",
    "BiasFrameQuery",
    "DarkFrameQuery",
    "FlatFrameQuery",
]
