from .const import dbname, user, host, port, password
import psycopg

conn = psycopg.connect(
    dbname=dbname,
    user=user,
    host=host,
    port=port,
    password=password,
)

with conn.cursor() as cur:
    cur.execute(
        """
        SELECT DISTINCT ON (software_used)
            original_filename, file_path, unified_filename, software_used
        FROM survey_scienceframe
        WHERE software_used IN (%s, %s)
        ORDER BY software_used, original_filename
        LIMIT 2;
    """,
        ("nina", "tcspy"),
    )
    for row in cur.fetchall():
        print(row)

conn.close()
