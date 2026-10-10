"""baseline schema

Revision ID: 0001
Revises:
Create Date: 2026-10-10
"""
from pathlib import Path

from alembic import op

# ---------------------------------------------------------------
# Namen dieser Migration
# ---------------------------------------------------------------
# Die Nummer dieser Migration
revision = "0001"
# Die Nummer der Migration davor. Das ist die erste, also gibt es keine.
down_revision = None
# Diese zwei Zeilen brauchen wir nicht. Sie bleiben leer.
branch_labels = None
depends_on = None

# ---------------------------------------------------------------
# Wo liegt unser Schema?
# ---------------------------------------------------------------
# Die Datei migrations/sql/0001_schema.sql mit allen 11 Tabellen
SQL_DATEI = Path(__file__).resolve().parent.parent / "sql" / "0001_schema.sql"


def upgrade():
    # "upgrade" heißt: Die Datenbank bekommt etwas Neues.
    # 1. Die SQL-Datei lesen
    sql = SQL_DATEI.read_text(encoding="utf-8")
    # 2. Das SQL in der Datenbank ausführen
    op.get_bind().exec_driver_sql(sql)


def downgrade():
    # "downgrade" heißt: Die Änderung zurücknehmen.
    # Das geht hier nicht, sonst wären alle Tabellen und Daten weg.
    raise NotImplementedError("Die erste Migration kann man nicht zurücknehmen.")