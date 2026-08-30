#!/usr/bin/env python
import argparse

from pipeline.const import (
    COADD_DEPENDENCY_ROLE,
    IMAGE_DEPENDENCY_ROLES,
    IMAGE_TYPE_COADD,
    IMAGE_TYPE_WHITE,
    WHITE_FILTER,
)
from pipeline.services.database.image_qa_dependency import ImageQADependency


parser = argparse.ArgumentParser(description="Register white images and their coadd source role in image_qa")
parser.add_argument("--apply", action="store_true", help="Apply the schema and row migration")
args = parser.parse_args()

db = ImageQADependency()
constraint = "image_qa_dependency_dependency_role_check"
roles_sql = ", ".join(f"'{role}'" for role in IMAGE_DEPENDENCY_ROLES)

with db.get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM image_qa WHERE filter = %s AND image_type IS DISTINCT FROM %s",
            (WHITE_FILTER, IMAGE_TYPE_WHITE),
        )
        white_rows = cur.fetchone()[0]
        cur.execute(
            "SELECT COUNT(*) FROM image_qa_dependency d"
            " JOIN image_qa derived ON derived.id = d.derived_image_id"
            " JOIN image_qa source ON source.id = d.source_image_id"
            " WHERE derived.filter = %s AND source.image_type = %s"
            " AND d.dependency_role IS DISTINCT FROM %s",
            (WHITE_FILTER, IMAGE_TYPE_COADD, COADD_DEPENDENCY_ROLE),
        )
        dependency_rows = cur.fetchone()[0]

print(f"White image rows to migrate: {white_rows}")
print(f"White-to-coadd dependency rows to migrate: {dependency_rows}")
if not args.apply:
    print("Dry run only; pass --apply to write changes.")
    raise SystemExit(0)

with db.get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute(f"ALTER TABLE image_qa_dependency DROP CONSTRAINT IF EXISTS {constraint}")
        cur.execute(
            f"ALTER TABLE image_qa_dependency ADD CONSTRAINT {constraint}"
            f" CHECK (dependency_role IN ({roles_sql})) NOT VALID"
        )
    conn.commit()

with db.get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE image_qa SET image_type = %s"
            " WHERE filter = %s AND image_type IS DISTINCT FROM %s",
            (IMAGE_TYPE_WHITE, WHITE_FILTER, IMAGE_TYPE_WHITE),
        )
        updated_white = cur.rowcount
        cur.execute(
            "UPDATE image_qa_dependency d SET dependency_role = %s"
            " FROM image_qa derived, image_qa source"
            " WHERE derived.id = d.derived_image_id AND source.id = d.source_image_id"
            " AND derived.image_type = %s AND source.image_type = %s"
            " AND d.dependency_role IS DISTINCT FROM %s",
            (COADD_DEPENDENCY_ROLE, IMAGE_TYPE_WHITE, IMAGE_TYPE_COADD, COADD_DEPENDENCY_ROLE),
        )
        updated_dependencies = cur.rowcount
    conn.commit()

with db.get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute(f"ALTER TABLE image_qa_dependency VALIDATE CONSTRAINT {constraint}")
    conn.commit()

print(f"Updated white image rows: {updated_white}")
print(f"Updated white-to-coadd dependency rows: {updated_dependencies}")
