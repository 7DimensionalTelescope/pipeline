import os
from ..services.database.query import free_query, TABLES


# Build a single UNION ALL query across all frame tables
_UNION_SQL = " UNION ALL ".join(
    [f'SELECT original_filename, file_path, unified_filename FROM "{tbl}"' for tbl in TABLES.values()]
)


def unified_name_from_path(full_path: str) -> str | None:
    """
    Given a full file path, return its unified_filename, or None if not found.
    """

    # direct comparison if full path
    if os.path.dirname(full_path):
        sql = f"""
        SELECT unified_filename
        FROM (
            {_UNION_SQL}
        ) AS all_frames
        WHERE original_filename = %s AND file_path = %s
        LIMIT 1;
        """
        rows = free_query(sql, (os.path.basename(full_path), full_path))

    # if basename only, compare original_filename
    else:
        sql = f"""
        SELECT unified_filename
        FROM (
            {_UNION_SQL}
        ) AS all_frames
        WHERE original_filename = %s
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

    any_basename = any(not os.path.dirname(p) for p in paths)

    if not any_basename:
        # Exact match mode if all inputs are full paths
        sql = f"""
        SELECT file_path, unified_filename
        FROM (
            {_UNION_SQL}
        ) AS all_frames
        WHERE original_filename = ANY(%s);
        """
        rows = free_query(sql, ([os.path.basename(p) for p in paths],))
        found = {fp: uf for fp, uf in rows}
        # return {p: found.get(p) for p in paths}
        return [found.get(p) for p in paths]

    # Basename match mode for ALL inputs
    basenames = [os.path.basename(p) for p in paths]  # return None if not found

    sql = f"""
    SELECT original_filename, unified_filename
    FROM (
        {_UNION_SQL}
    ) AS all_frames
    WHERE original_filename = ANY(%s);
    """
    rows = free_query(sql, (basenames,))

    # Map basename -> unified_filename (last one wins if duplicates)
    base_map: dict[str, str] = {}
    for base, uf in rows:
        base_map[base] = uf

    return [base_map.get(p) for p in paths]
    # return {p: base_map.get(os.path.basename(p)) for p in paths}
