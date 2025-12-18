from ..services.database.query import free_query, TABLES

# Build a single UNION ALL query across all frame tables
_UNION_SQL = " UNION ALL ".join([f'SELECT file_path, unified_filename FROM "{tbl}"' for tbl in TABLES.values()])


def unified_name_from_path(full_path: str) -> str | None:
    """
    Given a full file path, return its unified_filename, or None if not found.
    """
    sql = f"""
    SELECT unified_filename
    FROM (
        {_UNION_SQL}
    ) AS all_frames
    WHERE file_path = %s
    LIMIT 1;
    """
    rows = free_query(sql, (full_path,))
    return rows[0][0] if rows else None


def unified_names_from_paths(paths: list[str]) -> dict[str, str | None]:
    """
    Resolve many paths in one query. Returns {path: unified_filename or None}.
    """
    if not paths:
        return {}
    sql = f"""
    SELECT file_path, unified_filename
    FROM (
        {_UNION_SQL}
    ) AS all_frames
    WHERE file_path = ANY(%s);
    """
    rows = free_query(sql, (paths,))
    found = {fp: uf for fp, uf in rows}
    return {p: found.get(p) for p in paths}  # return None if not found
