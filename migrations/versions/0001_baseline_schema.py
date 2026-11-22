"""baseline schema

Revision ID: 0001
Revises:
Create Date: 2026-10-10
"""
from pathlib import Path

from alembic import op

# ---------------------------------------------------------------
# Names of this migration
# ---------------------------------------------------------------
# The number of this migration
revision = "0001"
# The number of the migration before. This is the first one, so there is none.
down_revision = None
# We do not need these two lines. They stay empty.
branch_labels = None
depends_on = None

# ---------------------------------------------------------------
# Where is our schema?
# ---------------------------------------------------------------
# The file migrations/sql/0001_schema.sql with all 11 tables
SQL_FILE = Path(__file__).resolve().parent.parent / "sql" / "0001_schema.sql"


def upgrade():
    # "upgrade" means: the database gets something new.
    # 1. Read the SQL file
    sql = SQL_FILE.read_text(encoding="utf-8")
    # 2. Run the SQL in the database
    op.get_bind().exec_driver_sql(sql)


def downgrade():
    # "downgrade" means: take the change back.
    # This is not possible here, otherwise all tables and data would be gone.
    raise NotImplementedError("The first migration cannot be taken back.")