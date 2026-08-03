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

The complete dependency_role vocabulary:

    role      | derived from                    | notes
    ----------+---------------------------------+----------------------------------------------
    bias      | NameHandler.type[1], master     | in use
    dark      | NameHandler.type[1], master     | in use
    flat      | NameHandler.type[1], master     | in use; a flat can be built from other flats
    biassig   | NameHandler.type[1], master     | never resolves: no sig frame holds an image_qa
    darksig   | NameHandler.type[1], master     |   row, so its ID card matches nothing
    flatsig   | NameHandler.type[1], master     |
    single    | NameHandler.type[2], calibrated | in use
    coadded   | NameHandler.type[2], calibrated | only if a coadd is named in an IMCMB*/IMG*
              |                                 |   card; a diff's parent coadds do not come
              |                                 |   through here, they use SCIIID/REFIID
    science   | SCIIID card, not NameHandler    | in use, diff only
    reference | REFIID card, not NameHandler    | in use, diff only
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from astropy.io import fits

from .base import BaseDatabase
from ...utils import atleast_1d

# A diff's two parents are both coadds, so NameHandler would call them both
# "coadded"; only the card itself says which is the science one.
DIRECT_PARENT_KEYS = (("SCIIID", "science"), ("REFIID", "reference"))


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
    if atleast_1d(NameHandler(fits_file).type)[0][3] == "difference":
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
        kind = typ[0]
        if kind == "master":
            role = typ[1]  # bias / dark / flat, or their sig variants
        elif kind == "calibrated":
            role = typ[2]  # single / coadded
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
