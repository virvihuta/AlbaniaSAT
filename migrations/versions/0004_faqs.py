"""faqs table

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-14
"""
import sqlalchemy as sa
from alembic import op

# ---------------------------------------------------------------
# Names of this migration
# ---------------------------------------------------------------
revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

# The 4 current questions from frontend/src/components/Faq.jsx (position 0 to 3)
START_FAQS = [
    (
        "Where does the imagery come from?",
        "From the Copernicus Sentinel-2 mission, accessed through Google Earth Engine. "
        "Sentinel-2 photographs all of Albania every few days at 10 m resolution.",
    ),
    (
        "How accurate is the map?",
        "The AlbaniaSAT model reaches 80.94% accuracy on held-out Albanian test patches. "
        "Single pixels can be wrong, but trends over larger areas are far more reliable.",
    ),
    (
        "Why not use an existing European model?",
        "Models trained on European data such as EuroSAT reach only 19.08% on Albanian patches. "
        "Albanian land looks different, so we trained on Albanian ground truth.",
    ),
    (
        "Which years can I compare?",
        "The classification covers 2017 to 2024, so you can compare how land cover changed over these years.",
    ),
]


def upgrade():
    # 1. New table faqs
    faqs = op.create_table(
        "faqs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("is_published", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    # 2. Insert the 4 current questions, in their order
    op.bulk_insert(
        faqs,
        [{"question": question, "answer": answer, "position": index}
         for index, (question, answer) in enumerate(START_FAQS)],
    )


def downgrade():
    # Removes the table and the FAQs in it
    op.drop_table("faqs")