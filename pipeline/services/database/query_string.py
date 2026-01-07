query_all_columns = """
    SELECT *
    FROM {table_name}
    ORDER BY {order_by}
"""

query_column_by_name = """
    SELECT *
    FROM {table_name}
    WHERE name = %(name)s
    LIMIT 1
"""

query_column_by_id = """
    SELECT *
    FROM {table_name}
    WHERE id = %(id)s
    LIMIT 1
"""

query_clear_table = """
    DELETE FROM {table_name}
"""

query_by_params = """
    SELECT *
    FROM {table_name}
    WHERE {params}
"""

query_insert = """
    INSERT INTO {table_name} ({columns}) VALUES ({values})
    RETURNING id
"""

query_update = """
    UPDATE {table_name}
    SET {params}
    WHERE id = %(id)s
    RETURNING id
"""

query_delete = """
    DELETE FROM {table_name}
    WHERE id = %(id)s
"""
