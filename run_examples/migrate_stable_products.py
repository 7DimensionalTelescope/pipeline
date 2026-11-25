#!/usr/bin/env python
import argparse
import re

from pipeline.const import IMAGE_DEPENDENCY_ROLES
from pipeline.services.database.image_qa_dependency import ImageQADependency


parser = argparse.ArgumentParser(description="Let image_qa hold stable products made outside a queue process")
parser.add_argument("--apply", action="store_true", help="Apply the schema migration")
args = parser.parse_args()

db = ImageQADependency()
constraint = "image_qa_dependency_dependency_role_check"
roles_sql = ", ".join(f"'{role}'" for role in IMAGE_DEPENDENCY_ROLES)
lock_timeout = "2s"

nullable_sql = ["ALTER TABLE pipeline.image_qa ALTER COLUMN process_status_id DROP NOT NULL"]
check_sql = [
    f"ALTER TABLE pipeline.image_qa_dependency DROP CONSTRAINT IF EXISTS {constraint}",
    f"ALTER TABLE pipeline.image_qa_dependency ADD CONSTRAINT {constraint}"
    f" CHECK (dependency_role IN ({roles_sql})) NOT VALID",
]
validate_sql = [f"ALTER TABLE pipeline.image_qa_dependency VALIDATE CONSTRAINT {constraint}"]

with db.get_connection() as conn:
    conn.read_only = True
    with conn.cursor() as cur:
        cur.execute(
            "SELECT is_nullable FROM information_schema.columns"
            " WHERE table_schema = 'pipeline' AND table_name = 'image_qa' AND column_name = 'process_status_id'"
        )
        nullable = cur.fetchone()[0] == "YES"
        cur.execute(
            "SELECT pg_get_constraintdef(oid), convalidated FROM pg_constraint"
            " WHERE conname = %s AND conrelid = 'pipeline.image_qa_dependency'::regclass",
            (constraint,),
        )
        check = cur.fetchone()

current_roles = set(re.findall(r"'([^']+)'", check[0])) if check else set()
if current_roles - set(IMAGE_DEPENDENCY_ROLES):
    raise SystemExit(
        f"{constraint} allows roles the code does not know: {sorted(current_roles - set(IMAGE_DEPENDENCY_ROLES))}"
    )
check_current = current_roles == set(IMAGE_DEPENDENCY_ROLES)

steps = [
    (nullable_sql, not nullable),
    (check_sql, not check_current),
    (validate_sql, not (check_current and check[1])),
]

print(f"image_qa.process_status_id nullable: {nullable}")
print(f"{constraint}: {check[0] if check else None} (validated: {check[1] if check else None})")
print(f"Roles to allow: {', '.join(IMAGE_DEPENDENCY_ROLES)}")
for statements, needed in steps:
    for sql in statements:
        print(f"{('Running' if args.apply else 'Would run') if needed else 'Already done'}: {sql}")
if not args.apply:
    print(f"Dry run only (lock_timeout {lock_timeout} per transaction); pass --apply to write changes.")
    raise SystemExit(0)

for statements, needed in steps:
    if not needed:
        continue
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SET LOCAL lock_timeout = '{lock_timeout}'")
            for sql in statements:
                cur.execute(sql)
        conn.commit()

print("Done.")
