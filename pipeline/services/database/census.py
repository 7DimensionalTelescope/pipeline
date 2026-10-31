"""The raw-frame census: one row per raw science frame, and the budget it implies."""

from typing import Iterable, Optional, Tuple

from ... import const
from .query import free_query

# the single's image_qa metrics a brief census carries, in display order
BRIEF_QA_COLUMNS = (
    "rotang",
    "seeing",
    "ellipmn",
    "ul5_5",
    "zp_auto",
    "ezp_auto",
    "backval",
    "backsig",
    "inf_filt",
    "inf_dezf",
    "ppflag",
    "cov_5",
    "stdnumb",
)


def _listed(values) -> list:
    """A bare string is one name, not its characters."""
    return [values] if isinstance(values, str) else list(values)


def census_from_raw(
    objects: Optional[Iterable[str]] = None,
    object_regex: Optional[str] = r"^T\d+$",
    units: Optional[Iterable[str]] = None,
    nightdates: Optional[Tuple[str, str]] = None,
    filters: Optional[Iterable[str]] = None,
    qa: Optional[str] = "brief",
    mepoch: bool = False,
    statement_timeout_ms: int = 1800000,
):
    """One row per raw science frame, in blocks: paths and identity of the single (with pipe_ver, sanity, inspcomm), the
    preprocess and science configs, the process flags (science registry, then the default crossfilter processes),
    has_image_qa and the single's image_qa metrics (`qa`: None | "brief" | "full"), the newest nightly coadd (`mepoch`
    adds n_coadds and the newest multi-epoch coadd), the white with its crossfilter config. `objects` overrides
    `object_regex`; `object_regex=None` lifts the object filter. Raw -> single by (unit, header instant UTC: local_obstime for the
    old raw filename patterns, whose obstime is the filename's wall clock); twins resolve to the path-bearing newest row. `object` is the tile, else the target, else the archive's own object_name."""
    filters = const.ALL_FILTERS if filters is None else filters
    import pandas as pd

    from ...const.crossfilter import CROSSFILTERPROCESS_REGISTRY
    from ...const.run import DEFAULT_CROSSFILTER_PROCESSES
    from ...const.sciproc import SCIPROCESS_REGISTRY
    from .image_qa import ImageQATable

    where, params = [], []
    if objects is not None:
        objects = [str(o) for o in _listed(objects)]
        if not objects:
            raise ValueError("objects is empty; pass None to use object_regex")
        where.append("COALESCE(tl.name, t.name, NULLIF(sf.object_name, '')) = ANY(%s)")
        params.append(objects)
    elif object_regex is not None:
        where.append("COALESCE(tl.name, t.name, NULLIF(sf.object_name, '')) ~ %s")
        params.append(object_regex)
    if units is not None:
        where.append("fu.name = ANY(%s)")
        params.append([str(u) for u in _listed(units)])
    if nightdates is not None:
        where.append("n.date BETWEEN %s AND %s")
        params.extend(nightdates)
    where.append("f.name = ANY(%s)")
    params.append(_listed(filters))

    single = [
        "image_path",
        "camera",
        "pipe_ver",
        "sanity",
        "inspcomm",
    ]  # of the single, always on, in the identity block
    if qa == "full":
        live = {
            row[0]
            for row in free_query(
                "SELECT column_name FROM information_schema.columns"
                " WHERE table_schema = 'pipeline' AND table_name = 'image_qa'"
            )
        }
        # never repeated: ids, the type, the raw frame's own keys, the identity block, the master-frame set
        skip = {"id", "process_status_id", "image_name", "image_type", "image_group", "m_epoch", "nightdate", "unit"}
        skip |= {"filter", "object", "exptime", *single, *ImageQATable.MASTERFRAME_COLUMNS}
        qa_cols = [c for c in ImageQATable.__annotations__ if c in live and c not in skip]
    elif qa == "brief":
        qa_cols = list(BRIEF_QA_COLUMNS)
    elif qa is None:
        qa_cols = []
    else:
        raise ValueError(f"qa must be None, 'brief' or 'full', not {qa!r}")
    s_select = ", ".join(f"q.{c}" for c in single + qa_cols)
    qa_out = "".join(f", s.{c}" for c in qa_cols)
    mepoch_select = mepoch_out = ""
    if mepoch:
        mepoch_select = """,
                       (array_agg(c.image_name ORDER BY c.id DESC) FILTER (WHERE c.m_epoch IS TRUE))[1] AS mepoch_coadd_name,
                       (array_agg(c.image_path ORDER BY c.id DESC) FILTER (WHERE c.m_epoch IS TRUE))[1] AS mepoch_coadd_path"""
        mepoch_out = ", sc.n_coadds, sc.mepoch_coadd_name, sc.mepoch_coadd_path"

    identity = [
        "raw_path",
        "single_path",
        "object",
        "nightdate",
        "filter",
        "unit",
        "camera",
        "obstime",
        "exptime",
        "is_too",
        "pipe_ver",
        "sanity",
        "inspcomm",
    ]
    config = [
        "preproc_config_file",
        "config_file",
        "config_progress",
        "config_errors",
        "config_status",
        "config_sanity",
    ]
    config += ["config_inspcomm"]
    image_qa = ["has_image_qa", *qa_cols]
    coadd = ["coadd_name", "coadd_path", "coadd_sanity", "coadd_inspcomm", "coadd_zp_auto"]
    coadd += ["n_coadds", "mepoch_coadd_name", "mepoch_coadd_path"] if mepoch else []
    white = [
        "white_name",
        "white_path",
        "crossfilter_file",
        "crossfilter_progress",
        "crossfilter_errors",
        "crossfilter_status",
    ]
    # raw frame + single (one per frame: path-bearing, newest) + config + coadds + white + crossfilter; the coadd and
    # white lookups are LATERAL so they run as index probes per single, not as a CTE rescanned per row
    census = pd.DataFrame(
        free_query(
            f"""
            WITH r AS (
                SELECT sf.id, COALESCE(tl.name, t.name, NULLIF(sf.object_name, '')) AS object, n.date::text AS nightdate, f.name AS filter,
                       fu.name AS unit, h.obstime, (h.obstime AT TIME ZONE 'UTC') AS obstime_utc, sf.exptime,
                       sf.is_too, sf.file_path AS raw_path
                FROM survey_scienceframe sf
                JOIN survey_night n ON sf.night_id = n.id JOIN facility_unit fu ON sf.unit_id = fu.id
                JOIN facility_filter f ON sf.filter_id = f.id
                LEFT JOIN survey_tile tl ON sf.tile_id = tl.id LEFT JOIN survey_target t ON sf.target_id = t.id
                CROSS JOIN LATERAL (SELECT CASE WHEN sf.filename_pattern LIKE 'old%%' THEN sf.local_obstime ELSE sf.obstime END AS obstime) h
                WHERE {" AND ".join(where)}),
            s AS (
                SELECT DISTINCT ON (r.id) r.id AS rid, q.id AS sid, {s_select}
                FROM r JOIN image_qa q ON q.image_type = 'single' AND q.unit = r.unit AND q.date_obs = r.obstime_utc
                ORDER BY r.id, (q.image_path IS NOT NULL) DESC, q.id DESC),
            pre AS (
                SELECT DISTINCT ON (nightdate, unit) nightdate::text AS nightdate, unit, config_file
                FROM process_status WHERE config_type = 'preprocess'
                ORDER BY nightdate, unit, updated_at DESC NULLS LAST, id DESC),
            cfg AS (
                SELECT DISTINCT ON (object, nightdate, filter) object, nightdate::text AS nightdate, filter,
                       config_file, progress, errors, status, sanity, inspcomm
                FROM process_status WHERE config_type = 'science'
                ORDER BY object, nightdate, filter, updated_at DESC NULLS LAST, id DESC),
            cf AS (
                SELECT DISTINCT ON (object, nightdate) object, nightdate::text AS nightdate,
                       config_file, progress, errors, status
                FROM process_status WHERE config_type = 'crossfilter'
                ORDER BY object, nightdate, updated_at DESC NULLS LAST, id DESC)
            SELECT r.raw_path, s.image_path, r.object, r.nightdate, r.filter, r.unit, s.camera, r.obstime, r.exptime,
                   r.is_too, s.pipe_ver, s.sanity, s.inspcomm,
                   pre.config_file, cfg.config_file, cfg.progress, cfg.errors, cfg.status, cfg.sanity, cfg.inspcomm,
                   s.sid IS NOT NULL{qa_out},
                   sc.coadd_name, sc.coadd_path, sc.coadd_sanity, sc.coadd_inspcomm, sc.coadd_zp_auto{mepoch_out},
                   w.image_name, w.image_path, cf.config_file, cf.progress, cf.errors, cf.status
            FROM r
            LEFT JOIN s ON s.rid = r.id
            LEFT JOIN pre ON pre.nightdate = r.nightdate AND pre.unit = r.unit
            LEFT JOIN cfg ON cfg.object = r.object AND cfg.nightdate = r.nightdate AND cfg.filter = r.filter
            LEFT JOIN LATERAL (
                SELECT count(*) AS n_coadds,
                       (array_agg(c.image_name ORDER BY c.id DESC) FILTER (WHERE c.m_epoch IS NOT TRUE))[1] AS coadd_name,
                       (array_agg(c.image_path ORDER BY c.id DESC) FILTER (WHERE c.m_epoch IS NOT TRUE))[1] AS coadd_path,
                       (array_agg(c.sanity ORDER BY c.id DESC) FILTER (WHERE c.m_epoch IS NOT TRUE))[1] AS coadd_sanity,
                       (array_agg(c.inspcomm ORDER BY c.id DESC) FILTER (WHERE c.m_epoch IS NOT TRUE))[1] AS coadd_inspcomm,
                       (array_agg(c.zp_auto ORDER BY c.id DESC) FILTER (WHERE c.m_epoch IS NOT TRUE))[1] AS coadd_zp_auto,
                       (array_agg(c.id ORDER BY c.id DESC) FILTER (WHERE c.m_epoch IS NOT TRUE))[1] AS coadd_id{mepoch_select}
                FROM image_qa_dependency d JOIN image_qa c ON c.id = d.derived_image_id AND c.image_type = 'coadd'
                WHERE d.source_image_id = s.sid AND d.dependency_role = 'single') sc ON TRUE
            LEFT JOIN LATERAL (
                SELECT wq.image_name, wq.image_path
                FROM image_qa_dependency d JOIN image_qa wq ON wq.id = d.derived_image_id AND wq.image_type = 'white'
                WHERE d.source_image_id = sc.coadd_id AND d.dependency_role = 'coadd'
                ORDER BY wq.id DESC LIMIT 1) w ON TRUE
            LEFT JOIN cf ON cf.object = r.object AND cf.nightdate = r.nightdate
            """,
            params,
            statement_timeout_ms=statement_timeout_ms,
        ),
        columns=identity + config + image_qa + coadd + white,
    )
    census = census.sort_values(["object", "nightdate", "filter", "obstime"], ignore_index=True)

    # a process is done when its config reached the process's progress end; config_errors stays a column of its own
    crossfilter = [spec for spec in CROSSFILTERPROCESS_REGISTRY.specs if spec.name in DEFAULT_CROSSFILTER_PROCESSES]
    for spec in SCIPROCESS_REGISTRY.specs:
        census[spec.name] = census.config_progress.ge(spec.progress_end)
    for spec in crossfilter:
        census[spec.name] = census.crossfilter_progress.ge(spec.progress_end)
    flags = [spec.name for spec in SCIPROCESS_REGISTRY.specs] + [spec.name for spec in crossfilter]
    return census[identity + config + flags + image_qa + coadd + white]


def budget_status(census):
    """Twin raw rows dropped; per exposure "sanity", "usable" or "unprocessed"."""
    import pandas as pd

    from ...const.sciproc import SINGLE_PHOTOMETRY_SPEC

    budget = census.loc[census.single_path.isna().sort_values(kind="stable").index]
    budget = budget.drop_duplicates(["object", "filter", "unit", "obstime"]).sort_index()
    status = pd.Series("unprocessed", index=budget.index)
    status[budget[SINGLE_PHOTOMETRY_SPEC.name] & budget.single_path.notna()] = "usable"
    status[budget.sanity.eq(False)] = "sanity"
    return status
