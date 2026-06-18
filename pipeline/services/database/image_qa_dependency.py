"""Sync image_qa_dependency rows from FITS header keywords.

Each processed image records its source images in IMCMB*/IMG* header cards.
This module parses those cards and keeps the image_qa_dependency table in sync
with what is actually on disk: one call to ImageQADependency.sync() fully
replaces the dependency rows for a given derived image.

Header conventions (from the pipeline write paths):
  - Masters (bias/dark/flat) and single science: IMCMB001, IMCMB002, ...
  - Coadd and diff: IMG00000, IMG00001, ...
NameHandler classifies each referenced file into its role; raw individual
frames are dropped because they have no image_qa row.
"""
from __future__ import annotations

import os
from typing import Dict, List

from astropy.io import fits

from .base import BaseDatabase
from ...utils import atleast_1d


def parse_ingredients(fits_file: str) -> List[Dict[str, str]]:
    """Return {role, name} dicts for the source images named in the header.

    Reads IMCMB*/IMG* cards and lets NameHandler classify each referenced file:
    masters map to their calib role (bias/dark/flat), calibrated frames to
    single/coadded. Raw frames are dropped (no image_qa row).
    """
    from ...path.name import NameHandler

    try:
        header = fits.getheader(fits_file)
    except Exception:
        return []

    values = [
        str(header[k]).strip()
        for k in sorted(header.keys())
        if k.startswith("IMCMB") or (k.startswith("IMG") and k[3:].isdigit())
    ]
    values = [v for v in values if v]
    if not values:
        return []

    nh = NameHandler(values)
    types = atleast_1d(nh.type)
    stems = atleast_1d(nh.stem)

    out: List[Dict[str, str]] = []
    for typ, stem in zip(types, stems):
        kind = typ[0]
        if kind == "master":
            role = typ[1]  # bias / dark / flat
        elif kind == "calibrated":
            role = typ[2]  # single / coadded
        else:
            continue  # raw frame: no image_qa row
        out.append({"role": role, "name": stem})

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

        names = [ing["name"] for ing in ingredients]
        name_to_role: Dict[str, str] = {ing["name"]: ing["role"] for ing in ingredients}

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Resolve each ingredient name to a single image_qa id. Names are
                # not unique (a reused master gets one row per process), so pick
                # latest-row-wins, preferring rows that carry an on-disk path.
                placeholders = ",".join(["%s"] * len(names))
                cur.execute(
                    f"SELECT DISTINCT ON (image_name) id, image_name"
                    f" FROM image_qa WHERE image_name IN ({placeholders})"
                    f" ORDER BY image_name, (image_path IS NOT NULL) DESC,"
                    f" created_at DESC NULLS LAST, id DESC",
                    names,
                )
                matched = cur.fetchall()
                if not matched:
                    return 0

                # Full replacement: delete old rows, insert current ones.
                cur.execute(
                    "DELETE FROM image_qa_dependency WHERE derived_image_id = %s",
                    (derived_qa_id,),
                )

                insert_data = [
                    (derived_qa_id, row[0], name_to_role[row[1]])
                    for row in matched
                    if row[1] in name_to_role
                ]
                if insert_data:
                    cur.executemany(
                        "INSERT INTO image_qa_dependency"
                        " (derived_image_id, source_image_id, dependency_role)"
                        " VALUES (%s, %s, %s)",
                        insert_data,
                    )
            conn.commit()

        return len(insert_data)
