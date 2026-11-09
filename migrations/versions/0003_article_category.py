"""artikel kategorie

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-09
"""
import sqlalchemy as sa
from alembic import op

# ---------------------------------------------------------------
# Namen dieser Migration
# ---------------------------------------------------------------
revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade():
    # 1. Neue Spalte category: Text, darf leer sein (alte Artikel haben noch kein Thema)
    op.add_column("articles", sa.Column("category", sa.String(32), nullable=True))

    # 2. Regel in der Datenbank: nur diese 5 Themen sind erlaubt (leer ist auch erlaubt)
    op.create_check_constraint(
        "ck_articles_category",
        "articles",
        "category IN ('urban_growth', 'forests', 'coast', 'agriculture', 'methodology')",
    )


def downgrade():
    # In umgekehrter Reihenfolge zurück
    op.drop_constraint("ck_articles_category", "articles", type_="check")
    op.drop_column("articles", "category")