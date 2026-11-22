"""class codes, names and mean_confidence

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-09
"""
import sqlalchemy as sa
from alembic import op

# ---------------------------------------------------------------
# Names of this migration
# ---------------------------------------------------------------
# The number of this migration
revision = "0002"
# The number of the migration before
down_revision = "0001"
# We do not need these two lines. They stay empty.
branch_labels = None
depends_on = None

# ---------------------------------------------------------------
# The 7 classes: codes from Virvi, names from Suisa
# ---------------------------------------------------------------
# id: (code, name_en, name_de, name_sq)
NEW_CLASSES = {
    0: ("broadleaf_forest", "Broadleaf forest", "Laubwald", "Pyll gjethegjerë"),
    1: ("coniferous_forest", "Coniferous forest", "Nadelwald", "Pyll halor"),
    2: ("cropland", "Cropland", "Ackerland", "Tokë bujqësore"),
    3: ("grass_shrub", "Grassland and shrubland", "Grünland und Buschland", "Livadhe dhe shkurre"),
    4: ("olive_grove", "Olive groves", "Olivenhaine", "Ullishte"),
    5: ("built_up", "Built-up", "Siedlung", "Zonë e ndërtuar"),
    6: ("water", "Water", "Gewässer", "Ujëra"),
}

# The old names, so that downgrade can write them back
# id: (name_en, name_de, name_sq)
OLD_CLASSES = {
    0: ("Broad-leaved Forest", "Laubwald", "Pyje gjethegjere"),
    1: ("Coniferous Forest", "Nadelwald", "Pyje halore"),
    2: ("Agricultural", "Ackerland", "Toke bujqesore"),
    3: ("Grassland and Shrubland", "Gruenland und Buschland", "Kullota dhe shkurre"),
    4: ("Olive Groves", "Olivenhaine", "Ullishta"),
    5: ("Urban", "Siedlung", "Zona urbane"),
    6: ("Water", "Gewaesser", "Ujera"),
}

UPDATE_NAMES = """
    UPDATE land_cover_classes
    SET name_en = :name_en, name_de = :name_de, name_sq = :name_sq
    WHERE id = :id
"""


def upgrade():
    # "upgrade" means: the database gets something new.

    # 1. New column code. First it may be empty, because the classes have no code yet.
    op.add_column("land_cover_classes", sa.Column("code", sa.String(32), nullable=True))

    # 2. Write the code and the new names for every class
    for class_id, (code, name_en, name_de, name_sq) in NEW_CLASSES.items():
        op.execute(sa.text("UPDATE land_cover_classes SET code = :code WHERE id = :id")
                   .bindparams(id=class_id, code=code))
        op.execute(sa.text(UPDATE_NAMES)
                   .bindparams(id=class_id, name_en=name_en, name_de=name_de, name_sq=name_sq))

    # 3. Now every class has a code: it may not be empty anymore and must be unique
    op.alter_column("land_cover_classes", "code", nullable=False)
    op.create_unique_constraint("uq_land_cover_classes_code", "land_cover_classes", ["code"])

    # 4. New column mean_confidence: decimal number from 0 to 1, may be empty (old rows)
    op.add_column("region_year_summaries", sa.Column("mean_confidence", sa.Float(), nullable=True))
    op.create_check_constraint(
        "ck_region_year_summaries_mean_confidence",
        "region_year_summaries",
        "mean_confidence >= 0 AND mean_confidence <= 1",
    )


def downgrade():
    # "downgrade" means: take the change back, exactly in the opposite order.

    # 4. Remove mean_confidence
    op.drop_constraint("ck_region_year_summaries_mean_confidence", "region_year_summaries", type_="check")
    op.drop_column("region_year_summaries", "mean_confidence")

    # 3. and 1. Remove the column code
    op.drop_constraint("uq_land_cover_classes_code", "land_cover_classes", type_="unique")
    op.drop_column("land_cover_classes", "code")

    # 2. Write the old names back
    for class_id, (name_en, name_de, name_sq) in OLD_CLASSES.items():
        op.execute(sa.text(UPDATE_NAMES)
                   .bindparams(id=class_id, name_en=name_en, name_de=name_de, name_sq=name_sq))