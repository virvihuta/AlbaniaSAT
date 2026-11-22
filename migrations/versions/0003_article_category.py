"""article category

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-09
"""
import sqlalchemy as sa
from alembic import op

# ---------------------------------------------------------------
# Names of this migration
# ---------------------------------------------------------------
revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade():
    # 1. New column category: text, may be empty (old articles have no topic yet)
    op.add_column("articles", sa.Column("category", sa.String(32), nullable=True))

    # 2. Rule in the database: only these 5 topics are allowed (empty is allowed too)
    op.create_check_constraint(
        "ck_articles_category",
        "articles",
        "category IN ('urban_growth', 'forests', 'coast', 'agriculture', 'methodology')",
    )


def downgrade():
    # Back in the opposite order
    op.drop_constraint("ck_articles_category", "articles", type_="check")
    op.drop_column("articles", "category")