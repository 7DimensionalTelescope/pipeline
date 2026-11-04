"""Sync image_qa_dependency rows from FITS header keywords.

Each processed image records its source images both by name (IMCMB*/IMG*) and by
IMAGEID (IMCID*/IID*, SCIIID/REFIID). This module reads the ID cards and keeps
the image_qa_dependency table in sync with what is actually on disk: one call to
ImageQADependency.sync() fully replaces the dependency rows for a given derived
image.

Every case a derived image can present:

    derived | cards read            | ingredient found there              | role recorded
    --------+-----------------------+-------------------------------------+---------------
    bias    | IMCMB001+ / IMCID001+ | raw bias frames                     | (dropped)
    dark    | IMCMB001+ / IMCID001+ | raw dark frames                     | (dropped)
            |                       | master bias                         | bias
    flat    | IMCMB001+ / IMCID001+ | raw flat frames                     | (dropped)
            |                       | master bias                         | bias
            |                       | master dark (the "flatdark")        | dark
            |                       | sibling master flats, other filters | flat
    single  | IMCMB001+ / IMCID001+ | raw light frame                     | (dropped)
            |                       | master bias                         | bias
            |                       | master dark                         | dark
            |                       | master flat                         | flat
    coadd   | IMG00000+ / IID00000+ | the singles in the stack            | single
            |                       | a source coadd                       | coadd
    white   | IMG00000+ / IID00000+ | its source coadds                    | coadd
    diff    | SCIIMG / SCIIID       | science coadd                       | science
            | REFIMG / REFIID       | reference coadd                     | reference
            | IMG*/IID* (inherited) | the science coadd's singles         | (ignored)

A diff's IMG*/IID* cards are inherited from its science coadd and name that
coadd's singles, so they are ignored here: only the two direct parents are
recorded and the rest of the chain is reached by walking the sources' own rows.

Resolution is by IMAGEID alone. Names are not unique across master
regenerations, which is the ambiguity IMAGEID exists to remove, so an ingredient
carrying no ID card is dropped: a raw frame (which has no image_qa row anyway)
or a product written before the ID cards existed.

Each edge stores both identities of its source. source_image_id is the image_qa
row, which belongs to a path and so follows whatever version currently occupies
it; source_imageid is the version that was actually used. They agree until the
ingredient is regenerated, so comparing them is the staleness test:

    SELECT ... FROM image_qa_dependency d JOIN image_qa s ON s.id = d.source_image_id
     WHERE d.source_imageid IS NOT NULL AND d.source_imageid <> s.imageid

which finds every image built from an ingredient version since superseded. A NULL
source_imageid is an edge written before this column existed: unknown, not stale.
find_stale_edges runs exactly that; impacted_images then walks the graph downward
from those images, because a coadd stacked from a stale single is stale too even
though its own edge still matches -- the single was not regenerated, only invalidated.

An astrometric re-solve is deliberately not a new version: the WCS changes but the
IMAGEID does not, so re-running astrometry does not invalidate the coadds built from
the image. Only the stages that mint a fresh IMAGEID -- preprocess, imcoadd,
imsubtract -- start an invalidation chain here.

The complete dependency_role vocabulary:

    role      | derived from                   | notes
    ----------+--------------------------------+-----------------------------------------------
    bias      | NameType.exposure_type, master | in use
    dark      | NameType.exposure_type, master | in use
    flat      | NameType.exposure_type, master | in use; a flat can be built from other flats
    biassig   | NameType.exposure_type, master | never resolves: no sig frame holds an image_qa
    darksig   | NameType.exposure_type, master |   row, so its ID card matches nothing; also
    flatsig   | NameType.exposure_type, master |   outside the CHECK constraint
    single    | NameType.image_type, calibrated| in use
    coadd     | NameType.image_type, calibrated| in use; source is a coadd
    diff      | NameType.image_type, calibrated| allowed since 2026-08-30; no source is a diff yet
    white     | NameType.image_type, calibrated| allowed since 2026-08-30; nothing coadds a white
    science   | SCIIID card, not NameHandler   | in use, diff only
    reference | REFIID card, not NameHandler   | in use, diff only
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import pandas as pd
from astropy.io import fits

from ...const import (
    CALIB_TYPES,
    COADD_DEPENDENCY_ROLE,
    IMAGE_DEPENDENCY_ROLES,
    IMAGE_TYPE_DIFF,
    IMAGE_TYPE_SINGLE,
    NAME_TYPE_CALIBRATED,
    NAME_TYPE_MASTER,
    REFERENCE_DEPENDENCY_ROLE,
    SCIENCE_DEPENDENCY_ROLE,
)
from .base import BaseDatabase
from ...utils import atleast_1d

# A diff's two parents are both coadds, so NameHandler would call them both
# "coadd"; only the card itself says which is the science one.
DIRECT_PARENT_KEYS = (
    ("SCIIID", SCIENCE_DEPENDENCY_ROLE),
    ("REFIID", REFERENCE_DEPENDENCY_ROLE),
)

# Downward closure over the graph. The depth bound is what keeps a cycle finite:
# UNION already dedupes ids, but an id can legitimately reappear at a greater depth.
_IMPACT_CTE = """
WITH RECURSIVE impact(id, depth) AS (
    SELECT derived_image_id, 1
      FROM image_qa_dependency WHERE source_image_id = ANY(%s)
  UNION
    SELECT d.derived_image_id, i.depth + 1
      FROM image_qa_dependency d JOIN impact i ON d.source_image_id = i.id
     WHERE i.depth < %s
)
"""


def _id_key(name_key: str) -> Optional[str]:
    """IMCMB001 -> IMCID001, IMG00000 -> IID00000."""
    if name_key.startswith("IMCMB"):
        return "IMCID" + name_key[5:]
    if name_key.startswith("IMG") and name_key[3:].isdigit():
        return "IID" + name_key[3:]
    return None


def parse_ingredients(fits_file: str) -> List[Dict[str, str]]:
    """Return {role, imageid} dicts for the source images stamped in the header.

    NameHandler classifies each referenced name into its role, as it always has;
    the paired ID card only says which image that role belongs to. Raw frames are
    dropped (no image_qa row), as is any ingredient lacking an ID card.
    """
    from ...path.name import NameHandler

    try:
        header = fits.getheader(fits_file)
    except Exception:
        return []

    direct = [
        {"role": role, "imageid": str(header[key]).strip()}
        for key, role in DIRECT_PARENT_KEYS
        if str(header.get(key, "")).strip()
    ]
    if direct:
        return direct

    # A diff carries IMG*/IID* cards inherited from its science coadd, naming that
    # coadd's singles. Never read those as the diff's own ingredients: a diff has
    # its parents in SCIIID/REFIID or it has nothing recordable.
    if atleast_1d(NameHandler(fits_file).type)[0].image_type == IMAGE_TYPE_DIFF:
        return []

    pairs = []
    for name_key in sorted(header.keys()):
        id_key = _id_key(name_key)
        if id_key is None:
            continue
        name = str(header[name_key]).strip()
        imageid = str(header.get(id_key, "")).strip()
        if name and imageid:
            pairs.append((name, imageid))
    if not pairs:
        return []

    types = atleast_1d(NameHandler([name for name, _ in pairs]).type)

    out: List[Dict[str, str]] = []
    seen = set()
    for (_, imageid), typ in zip(pairs, types):
        if typ.kind in (NAME_TYPE_MASTER, NAME_TYPE_CALIBRATED):
            role = typ.image_type  # a master carries its exposure_type; a science product its image_type
        else:
            continue  # raw frame: no image_qa row
        if imageid not in seen:
            seen.add(imageid)
            out.append({"role": role, "imageid": imageid})

    return out


class ImageQADependency(BaseDatabase):
    """Manages image_qa_dependency records."""

    def __init__(self, db_params=None):
        self._table_name = "image_qa_dependency"
        super().__init__(db_params)

    @property
    def table_name(self) -> str:
        return self._table_name

    @property
    def pyTable(self):
        return None  # junction table; no ORM dataclass needed

    def sync(self, derived_file: str, derived_qa_id: int) -> int:
        """Rebuild dependency rows for derived_qa_id from the FITS header.

        Deletes all existing rows for derived_qa_id, then re-inserts from the
        current on-disk header.  No-ops (returns 0) when the file does not
        exist or carries no trackable dependency keys.

        Returns the number of rows inserted.
        """
        if not os.path.exists(derived_file):
            return 0

        ingredients = parse_ingredients(derived_file)
        if not ingredients:
            return 0

        # One image can fill two roles -- a diff whose science and reference are the same
        # coadd -- and the key is (derived, source, role), so keep every role per id.
        roles: Dict[str, List[str]] = {}
        for ing in ingredients:
            per_id = roles.setdefault(ing["imageid"], [])
            if ing["role"] not in per_id:
                per_id.append(ing["role"])

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Resolve each ingredient IMAGEID to a single image_qa id. A row is
                # one (process, image) pair, so a master fetched by N nights holds N
                # rows -- all the same file. Take the earliest, which is normally the
                # process that generated it, and is stable across repeated syncs.
                placeholders = ",".join(["%s"] * len(roles))
                cur.execute(
                    f"SELECT DISTINCT ON (imageid) imageid, id"
                    f" FROM image_qa WHERE imageid IN ({placeholders})"
                    f" ORDER BY imageid, (image_path IS NOT NULL) DESC, id",
                    list(roles),
                )
                matched = cur.fetchall()
                if not matched:
                    return 0

                # Full replacement: delete old rows, insert current ones.
                cur.execute(
                    "DELETE FROM image_qa_dependency WHERE derived_image_id = %s",
                    (derived_qa_id,),
                )

                # source_image_id follows the path, so it tracks whatever version now sits
                # there; source_imageid pins the version actually used, which is the only
                # way to tell later that the ingredient has since been regenerated.
                insert_data = [
                    (derived_qa_id, source_id, role, imageid)
                    for imageid, source_id in matched
                    for role in roles[imageid]
                ]
                if insert_data:
                    cur.executemany(
                        "INSERT INTO image_qa_dependency"
                        " (derived_image_id, source_image_id, dependency_role, source_imageid)"
                        " VALUES (%s, %s, %s, %s)",
                        insert_data,
                    )
            conn.commit()

        return len(insert_data)

    def find_stale_edges(
        self,
        nightdate_from: Optional[str] = None,
        nightdate_to: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[tuple]:
        """Edges whose source was regenerated after the derived image was built.

        Returns (derived_image_id, derived_name, source_image_id, source_name, role,
        used_imageid, current_imageid). The nightdate range filters the derived image.
        """
        # MATERIALIZED is load-bearing: stale edges are a handful in millions, so folding
        # this into the outer query lets the planner walk the PK index looking for them
        # (minutes). Forced first, it is one hash join down to ~100 rows, then lookups.
        query = (
            "WITH stale AS MATERIALIZED ("
            "  SELECT d.derived_image_id, d.source_image_id, d.dependency_role,"
            "         d.source_imageid, s.imageid AS current_imageid"
            "    FROM image_qa_dependency d"
            "    JOIN image_qa s ON s.id = d.source_image_id"
            "   WHERE d.source_imageid IS NOT NULL AND d.source_imageid <> s.imageid"
            ")"
            " SELECT st.derived_image_id, qa_d.image_name, st.source_image_id, qa_s.image_name,"
            " st.dependency_role, st.source_imageid, st.current_imageid"
            " FROM stale st"
            " JOIN image_qa qa_d ON qa_d.id = st.derived_image_id"
            " JOIN image_qa qa_s ON qa_s.id = st.source_image_id"
            " WHERE TRUE"
        )
        params: List[object] = []
        if nightdate_from is not None:
            query += " AND qa_d.nightdate >= %s"
            params.append(nightdate_from)
        if nightdate_to is not None:
            query += " AND qa_d.nightdate <= %s"
            params.append(nightdate_to)
        query += " ORDER BY st.derived_image_id"
        if limit is not None:
            query += " LIMIT %s"
            params.append(limit)
        return self.execute_query(query, tuple(params))

    def impacted_images(self, seed_image_ids: List[int], max_depth: int = 10) -> List[tuple]:
        """Images reachable downward from the seeds: (image_id, depth, image_type, image_name).

        The seeds themselves are not included -- these are the images that consumed
        them, then whatever consumed those, to max_depth.
        """
        seeds = [int(i) for i in atleast_1d(seed_image_ids) if i is not None]
        if not seeds:
            return []
        return self.execute_query(
            f"{_IMPACT_CTE}"
            " SELECT i.id, min(i.depth), qa.image_type, qa.image_name"
            " FROM impact i JOIN image_qa qa ON qa.id = i.id"
            " GROUP BY i.id, qa.image_type, qa.image_name ORDER BY 2, 1",
            (seeds, max_depth),
        )

    def impacted_configs(
        self, seed_image_ids: List[int], max_depth: int = 10, include_seeds: bool = True
    ) -> List[tuple]:
        """Configs owning any image downstream of the seeds: (config_name, config_type).

        ``include_seeds`` also returns the configs that own the seed images, which is
        what an invalidation sweep wants (the stale images are themselves affected) and
        what a "who consumes my output" query does not.
        """
        seeds = [int(i) for i in atleast_1d(seed_image_ids) if i is not None]
        if not seeds:
            return []
        scope = "SELECT id FROM impact" + (" UNION SELECT unnest(%s)" if include_seeds else "")
        params: List[object] = [seeds, max_depth] + ([seeds] if include_seeds else [])
        return self.execute_query(
            f"{_IMPACT_CTE}"
            " SELECT DISTINCT ps.name, ps.config_type"
            f" FROM ({scope}) affected"
            " JOIN image_qa qa ON qa.id = affected.id"
            " JOIN process_status ps ON ps.id = qa.process_status_id"
            " ORDER BY 2, 1",
            tuple(params),
        )

    def count_recorded(self, image_names: List[str]) -> int:
        """How many of `image_names` have an image_qa row with at least one dependency edge."""
        if not image_names:
            return 0
        rows = self.execute_query(
            "SELECT count(DISTINCT q.image_name) FROM image_qa q "
            "JOIN image_qa_dependency d ON d.derived_image_id = q.id WHERE q.image_name = ANY(%(names)s)",
            {"names": list(image_names)},
        )
        return rows[0][0] if rows else 0


def _qa_rows(execute, images) -> Dict[str, List[tuple]]:
    """image path or name -> [(id, image_path, image_type, sanity)]: the row at its path wins, else every row of its name."""
    names = {image: os.path.basename(image).replace(".fits", "") for image in atleast_1d(images) if image}
    if not names:
        return {}
    by_name: Dict[str, List[tuple]] = {}
    for row in execute(
        "SELECT id, image_name, image_path, image_type, sanity FROM image_qa WHERE image_name = ANY(%s) ORDER BY id",
        (sorted(set(names.values())),),
    ):
        by_name.setdefault(row[1], []).append((row[0], *row[2:]))
    return {
        image: [row for row in by_name.get(name, []) if row[1] == image] or by_name.get(name, [])
        for image, name in names.items()
    }


def stale_products(
    seeds: Optional[List[str]] = None,
    objects: Optional[List[str]] = None,
    max_depth: int = 10,
    check_disk: bool = True,
) -> tuple:
    """Downstream products of SANITY F seeds, and stale edges: (products, stale) DataFrames; read-only."""
    from concurrent.futures import ThreadPoolExecutor

    dependency = ImageQADependency()
    with dependency.get_connection() as conn:
        conn.read_only = True
        conn.execute("SET statement_timeout = '120s'")

        def execute(query, params=None):
            return conn.execute(query, params).fetchall()

        if seeds:
            seed_ids = sorted({row[0] for rows in _qa_rows(execute, seeds).values() for row in rows})
        elif objects:
            seed_ids = [
                row[0]
                for row in execute(
                    "SELECT id FROM image_qa WHERE sanity IS FALSE AND image_type = ANY(%s)"
                    " AND (image_name LIKE ANY(%s) OR object = ANY(%s))",
                    (
                        [IMAGE_TYPE_SINGLE, *CALIB_TYPES],
                        [obj.replace("_", r"\_") + r"\_%" for obj in atleast_1d(objects)],
                        list(atleast_1d(objects)),
                    ),
                )
            ]
        else:
            seed_ids = [row[0] for row in execute("SELECT id FROM image_qa WHERE sanity IS FALSE")]

        rows = []
        if seed_ids:
            rows = execute(
                "WITH RECURSIVE impact(seed, id, depth) AS ("
                "  SELECT source_image_id, derived_image_id, 1 FROM image_qa_dependency"
                "   WHERE source_image_id = ANY(%s)"
                " UNION"
                "  SELECT i.seed, d.derived_image_id, i.depth + 1"
                "    FROM image_qa_dependency d JOIN impact i ON d.source_image_id = i.id"
                "   WHERE i.depth < %s"
                ")"
                " SELECT qa.id, qa.image_type, qa.image_name, qa.image_path, min(i.depth), count(DISTINCT i.seed),"
                " qa.sanity, ps.name"
                " FROM impact i JOIN image_qa qa ON qa.id = i.id"
                " LEFT JOIN process_status ps ON ps.id = qa.process_status_id"
                " GROUP BY qa.id, ps.name ORDER BY 5, 2, 3",
                (seed_ids, max_depth),
            )

    products = pd.DataFrame(
        [row[1:] for row in rows],
        columns=["image_type", "image_name", "image_path", "depth", "n_seeds", "sanity", "config_name"],
    )
    on_disk = [None] * len(products)
    if check_disk:
        with ThreadPoolExecutor(8) as pool:
            on_disk = list(pool.map(lambda path: bool(path) and os.path.exists(path), products["image_path"]))
    products.insert(products.columns.get_loc("sanity") + 1, "on_disk", on_disk)

    edges = dependency.find_stale_edges()
    if seeds or objects:
        scope = set(seed_ids) | {row[0] for row in rows}
        edges = [edge for edge in edges if edge[0] in scope]
    types = dict(
        dependency.execute_query(
            "SELECT id, image_type FROM image_qa WHERE id = ANY(%s)",
            (sorted({edge[0] for edge in edges} | {edge[2] for edge in edges}),),
        )
        if edges
        else []
    )
    stale = pd.DataFrame(
        [(e[1], types.get(e[0]), e[3], types.get(e[2]), e[4], e[5], e[6]) for e in edges],
        columns=[
            "derived_name",
            "derived_type",
            "source_name",
            "source_type",
            "role",
            "used_imageid",
            "current_imageid",
        ],
    )
    return products, stale


def _markdown_table(frame, limit: int, by: str) -> List[str]:
    """Markdown table of the first `limit` rows; the rest counted by `by`."""
    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    lines += [
        "| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |"
        for row in frame.head(limit).itertuples(index=False)
    ]
    if len(frame) > limit:
        rest = frame.iloc[limit:][by].value_counts()
        lines += ["", f"... {len(frame) - limit} more rows: " + ", ".join(f"{k} {n}" for k, n in rest.items())]
    return lines


def stale_products_report(products, stale, path: Optional[str] = None) -> str:
    """Markdown of stale_products: counts by type, top configs, the tables; written to `path` when given."""
    counts = products.groupby("image_type").agg(
        n=("image_name", "size"),
        sanity_false=("sanity", lambda s: int(s.eq(False).sum())),
        not_on_disk=("on_disk", lambda s: int(s.eq(False).sum())),
    )
    configs = products["config_name"].value_counts().head(20).rename_axis("config_name").reset_index(name="n")
    lines = [
        "# Stale products",
        "",
        f"{len(products)} downstream products, {len(stale)} stale edges.",
        "",
        "## Counts by type",
        "",
        *_markdown_table(counts.reset_index(), len(counts), "image_type"),
        "",
        "## Top configs",
        "",
        *_markdown_table(configs, 20, "config_name"),
        "",
        "## Downstream products",
        "",
        *_markdown_table(products, 200, "image_type"),
        "",
        "## Stale edges",
        "",
        *_markdown_table(stale, 200, "derived_type"),
        "",
    ]
    text = "\n".join(lines)
    if path:
        with open(path, "w") as f:
            f.write(text)
    return text
