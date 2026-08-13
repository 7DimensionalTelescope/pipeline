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
          AND d.dependency_role = 'single'
        GROUP BY s.unit
        ORDER BY n_frames DESC, s.unit
        """,
        (_registered(image_names(image)),),
    )


def images_of_unit(unit: str, kind: str = "coadd", limit: Optional[int] = None) -> List[Tuple[str, str]]:
    """(image_name, image_path) of every derived image `unit` contributed a frame to.

    `kind` filters by basename suffix ("coadd", "diff", or None for all). The suffix test
    is `right()` rather than LIKE: an escaped LIKE wildcard is a trap in a non-raw Python
    string, where ESCAPE '\' silently becomes ESCAPE ''.
    """
    where, params = "", [unit]
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
          AND d.dependency_role = 'single'
          {where}
        ORDER BY c.image_name
    """
    if limit:
        query += " LIMIT %s"
        params.append(limit)
    return free_query(query, params)


# Descendant walk over image_qa_dependency: source -> derived, transitively. The chain
# bias -> dark -> flat -> single -> coadd -> diff is fully recorded (master-to-master
# edges exist), so one recursion reaches every affected product. UNION (not UNION ALL)
# dedupes, and the depth cap is a cycle backstop.
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

    Depth 1 = direct consumers. Regenerating a master frame makes this the set of images
    that no longer descend from what produced them.

    `depth` is the SHORTEST path, which is not a topological rank and must not be used to
    order regeneration: a master bias reaches the flat it made both through the dark
    (depth 2) and directly, since a flat records its bias too, so the flat also comes out
    at depth 1 alongside that dark. Regenerating in depth order would rebuild the flat from
    the dark it is about to replace.
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

    The actionable form of `blast_radius`, and it inherits that function's caveat: `depth`
    is a shortest path, not a topological rank, so it does not order the reruns. Preprocess
    configs must run before the science configs that consume their products, and a
    preprocess config that only fetches a master cannot regenerate it -- only the config
    owning its raw calibration frames can.

    A preprocess rerun does pick the change up on its own (`Preprocess` compares IMCID
    against the masters it would select now, preprocess.py `_sci_needs_processing` /
    `_master_ingredients_changed`), but a science rerun does not: stage skipping there is
    flag-based (run.py:100). Flip the stage `flag:` booleans, or rerun with overwrite=True.
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


def image_names(images) -> List[str]:
    """Normalize image path(s) or name(s) to bare `image_name` values as stored in image_qa.

    Accepts a str/Path or a list of them. `image_name` is the basename without ".fits",
    so both "/lyman/.../UDS_m525_..._coadd.fits" and "UDS_m525_..._coadd" are valid input.
    Raises ValueError on anything that cannot be a name (empty, a directory, a glob).

    A list is accepted for CONVENIENCE, not speed: seeding the recursive walk with an array
    was measured 2x SLOWER than looping single seeds (8 masters: 0.65 s batched vs 0.30 s
    looped), because one union traversal is bigger than several small ones. Same answers
    either way; both are sub-second.
    """
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
    """The subset of `names` present in image_qa, raising if none are."""
    rows = free_query("SELECT image_name FROM image_qa WHERE image_name = ANY(%s)", (names,))
    found = {r[0] for r in rows}
    if not found:
        raise ValueError(
            f"none of these are registered in image_qa: {names[:3]}{'...' if len(names) > 3 else ''}. "
            "An unregistered image yields an empty result that is indistinguishable from 'no dependencies'."
        )
    return sorted(found)
