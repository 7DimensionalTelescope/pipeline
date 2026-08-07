"""Config-level dependency mirror for process_status.

The scheduler (SQLite) is the operational source of truth for which configs
depend on which: a parent config's ``dependent_idx`` lists the configs that
become runnable once it finishes.  That store must keep working even if the
PostgreSQL database is offline, so this table is only a *mirror* of it, kept
for querying/analysis alongside process_status.

Edges are stored by config *name* (the process_status.name stem), not by
process_status.id, because:
  - the scheduler identifies configs by path/name, so mirroring never depends
    on a process_status row already existing (pending configs are captured);
  - it stays writable when process_status rows have not been created yet.

Direction mirrors image_qa_dependency: the *derived* config depends on the
*source* config, and ``dependency_role`` records the source's config_type
(e.g. "preprocess" for a science config, "science" for a future downstream
type).  The structure is agnostic to the number of config types, so adding a
new type that depends on science needs no schema change.

Two writers fill this table and ``origin`` says which one owns a row:

    schedule | the plan, from Scheduler.mirror_dependencies: "these configs were
             |   queued together".  Available before anything has run, and wrong
             |   whenever a run does not follow the plan -- a preprocess config
             |   that produced no usable master sends its science configs to
             |   another night's masters, and the plan still names its own night.
    product  | the fact, from sync_from_products: a roll-up of image_qa_dependency,
             |   which is resolved by IMAGEID from the headers of the files that
             |   were actually written.  Correct across reprocessing by
             |   construction, because it is re-derived from the products.

A product row therefore supersedes a schedule row and the plan mirror will not
overwrite one; the plan only fills configs that have not produced anything yet.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List, Optional, Tuple

from .base import BaseDatabase

# (derived_config_name, source_config_name, dependency_role)
Edge = Tuple[str, str, Optional[str]]

# image_qa_dependency rolled up to config level, for one derived config.
ROLLUP_QUERY = """
SELECT DISTINCT ps_d.name, ps_s.name, ps_s.config_type
  FROM image_qa qa_d
  JOIN image_qa_dependency iqd ON iqd.derived_image_id = qa_d.id
  JOIN image_qa qa_s ON qa_s.id = iqd.source_image_id
  JOIN process_status ps_d ON ps_d.id = qa_d.process_status_id
  JOIN process_status ps_s ON ps_s.id = qa_s.process_status_id
 WHERE qa_d.process_status_id = %s
   AND qa_s.process_status_id <> qa_d.process_status_id
"""


class ProcessStatusDependency(BaseDatabase):
    """Manages process_status_dependency records (config-level dependencies)."""

    _table_ready = False

    def __init__(self, db_params=None):
        self._table_name = "process_status_dependency"
        super().__init__(db_params)

    @property
    def table_name(self) -> str:
        return self._table_name

    @property
    def pyTable(self):
        return None  # junction table; no ORM dataclass needed

    def ensure_table(self) -> None:
        """Create the table and its indexes if they do not exist (idempotent)."""
        if type(self)._table_ready:
            return
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS process_status_dependency (
                        derived_config_name VARCHAR NOT NULL,
                        source_config_name  VARCHAR NOT NULL,
                        dependency_role     VARCHAR,
                        origin              VARCHAR DEFAULT 'schedule',
                        created_at          TIMESTAMP DEFAULT NOW(),
                        PRIMARY KEY (derived_config_name, source_config_name)
                    )
                    """
                )
                # every row that predates this column was written by the plan mirror
                cur.execute(
                    "ALTER TABLE process_status_dependency"
                    " ADD COLUMN IF NOT EXISTS origin VARCHAR DEFAULT 'schedule'"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS ix_psd_source"
                    " ON process_status_dependency (source_config_name)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS ix_psd_derived"
                    " ON process_status_dependency (derived_config_name)"
                )
            conn.commit()
        type(self)._table_ready = True

    def replace_dependencies(self, edges: Iterable[Edge], origin: str = "schedule") -> int:
        """Full-replace the dependency rows for every derived config in ``edges``.

        For each distinct derived config appearing in ``edges``, its existing
        rows are deleted and re-inserted from ``edges``.  Derived configs not
        mentioned are left untouched.  Returns the number of rows inserted.

        Writing with ``origin="schedule"`` skips any derived config that already
        holds product-derived rows, so re-queueing a config that has already run
        cannot replace what it did with what it was expected to do.
        """
        by_derived = defaultdict(list)
        for derived, source, role in edges:
            if not derived or not source:
                continue
            by_derived[derived].append((source, role))

        if not by_derived:
            return 0

        self.ensure_table()

        inserted = 0
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                if origin == "schedule":
                    cur.execute(
                        "SELECT DISTINCT derived_config_name FROM process_status_dependency"
                        " WHERE origin = 'product' AND derived_config_name = ANY(%s)",
                        (list(by_derived),),
                    )
                    for (resolved,) in cur.fetchall():
                        by_derived.pop(resolved, None)
                    if not by_derived:
                        return 0

                for derived, pairs in by_derived.items():
                    cur.execute(
                        "DELETE FROM process_status_dependency WHERE derived_config_name = %s",
                        (derived,),
                    )
                    rows = [(derived, source, role, origin) for (source, role) in pairs]
                    cur.executemany(
                        "INSERT INTO process_status_dependency"
                        " (derived_config_name, source_config_name, dependency_role, origin)"
                        " VALUES (%s, %s, %s, %s)"
                        " ON CONFLICT (derived_config_name, source_config_name)"
                        " DO UPDATE SET dependency_role = EXCLUDED.dependency_role,"
                        " origin = EXCLUDED.origin",
                        rows,
                    )
                    inserted += len(rows)
            conn.commit()

        return inserted

    def sync_from_products(self, process_status_id: int) -> int:
        """Rebuild one config's edges from image_qa_dependency, i.e. from what it really used.

        Rolls the image-level graph up to config level: every source image that
        belongs to a different config becomes one edge, with the source's
        config_type as the role.  The (derived, source) primary key allows one
        role per config pair, so the ingredient roles that produced the edge
        (bias/dark/flat/single/...) stay at image level, where they are already
        recorded.

        Returns 0 without deleting anything when the roll-up finds nothing, the
        same rule ImageQADependency.sync follows: a config whose images are not
        yet registered must not have its existing edges wiped.
        """
        edges = self.execute_query(ROLLUP_QUERY, (process_status_id,))
        if not edges:
            return 0
        return self.replace_dependencies(edges, origin="product")

    def get_sources(self, derived_config_name: str, role: Optional[str] = None) -> List[tuple]:
        """Return ``(source_config_name, dependency_role)`` rows this config depends on."""
        query = (
            "SELECT source_config_name, dependency_role"
            " FROM process_status_dependency WHERE derived_config_name = %s"
        )
        params: List[object] = [derived_config_name]
        if role is not None:
            query += " AND dependency_role = %s"
            params.append(role)
        return self.execute_query(query, tuple(params))

    def get_derived(self, source_config_name: str, role: Optional[str] = None) -> List[tuple]:
        """Return ``(derived_config_name, dependency_role)`` rows that depend on this config."""
        query = (
            "SELECT derived_config_name, dependency_role"
            " FROM process_status_dependency WHERE source_config_name = %s"
        )
        params: List[object] = [source_config_name]
        if role is not None:
            query += " AND dependency_role = %s"
            params.append(role)
        return self.execute_query(query, tuple(params))
