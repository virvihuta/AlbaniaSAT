"""probabilities with class codes

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-23
"""
from alembic import op

# ---------------------------------------------------------------
# Names of this migration
# ---------------------------------------------------------------
revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None

# ---------------------------------------------------------------
# The keys in predictions.probabilities change, the numbers stay the same
# ---------------------------------------------------------------
# {"0": 0.93, ...} becomes {"broadleaf_forest": 0.93, ...}
IDS_TO_CODES = """
    UPDATE predictions p
    SET probabilities = (
        SELECT jsonb_object_agg(c.code, e.value)
        FROM jsonb_each(p.probabilities) AS e
        JOIN land_cover_classes c ON c.id = CAST(e.key AS int)
    )
    WHERE p.probabilities ->> '0' IS NOT NULL
"""

# {"broadleaf_forest": 0.93, ...} becomes {"0": 0.93, ...}
CODES_TO_IDS = """
    UPDATE predictions p
    SET probabilities = (
        SELECT jsonb_object_agg(CAST(c.id AS text), e.value)
        FROM jsonb_each(p.probabilities) AS e
        JOIN land_cover_classes c ON c.code = e.key
    )
    WHERE p.probabilities ->> 'broadleaf_forest' IS NOT NULL
"""


def upgrade():
    # Class ids as keys become class codes (Virvi's decision: codes everywhere)
    op.execute(IDS_TO_CODES)


def downgrade():
    # Back: class codes become class ids again
    op.execute(CODES_TO_IDS)