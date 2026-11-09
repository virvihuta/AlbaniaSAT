"""klassen codes, namen und mean_confidence

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-09
"""
import sqlalchemy as sa
from alembic import op

# ---------------------------------------------------------------
# Namen dieser Migration
# ---------------------------------------------------------------
# Die Nummer dieser Migration
revision = "0002"
# Die Nummer der Migration davor
down_revision = "0001"
# Diese zwei Zeilen brauchen wir nicht. Sie bleiben leer.
branch_labels = None
depends_on = None

# ---------------------------------------------------------------
# Die 7 Klassen: Codes von Virvi, Namen von Suisa
# ---------------------------------------------------------------
# id: (code, name_en, name_de, name_sq)
NEUE_KLASSEN = {
    0: ("broadleaf_forest", "Broadleaf forest", "Laubwald", "Pyll gjethegjerë"),
    1: ("coniferous_forest", "Coniferous forest", "Nadelwald", "Pyll halor"),
    2: ("cropland", "Cropland", "Ackerland", "Tokë bujqësore"),
    3: ("grass_shrub", "Grassland and shrubland", "Grünland und Buschland", "Livadhe dhe shkurre"),
    4: ("olive_grove", "Olive groves", "Olivenhaine", "Ullishte"),
    5: ("built_up", "Built-up", "Siedlung", "Zonë e ndërtuar"),
    6: ("water", "Water", "Gewässer", "Ujëra"),
}

# Die alten Namen, damit downgrade sie zurückschreiben kann
# id: (name_en, name_de, name_sq)
ALTE_KLASSEN = {
    0: ("Broad-leaved Forest", "Laubwald", "Pyje gjethegjere"),
    1: ("Coniferous Forest", "Nadelwald", "Pyje halore"),
    2: ("Agricultural", "Ackerland", "Toke bujqesore"),
    3: ("Grassland and Shrubland", "Gruenland und Buschland", "Kullota dhe shkurre"),
    4: ("Olive Groves", "Olivenhaine", "Ullishta"),
    5: ("Urban", "Siedlung", "Zona urbane"),
    6: ("Water", "Gewaesser", "Ujera"),
}

UPDATE_NAMEN = """
    UPDATE land_cover_classes
    SET name_en = :name_en, name_de = :name_de, name_sq = :name_sq
    WHERE id = :id
"""


def upgrade():
    # "upgrade" heißt: Die Datenbank bekommt etwas Neues.

    # 1. Neue Spalte code. Zuerst darf sie leer sein, weil die Klassen noch keinen Code haben.
    op.add_column("land_cover_classes", sa.Column("code", sa.String(32), nullable=True))

    # 2. Für jede Klasse den Code und die neuen Namen eintragen
    for class_id, (code, name_en, name_de, name_sq) in NEUE_KLASSEN.items():
        op.execute(sa.text("UPDATE land_cover_classes SET code = :code WHERE id = :id")
                   .bindparams(id=class_id, code=code))
        op.execute(sa.text(UPDATE_NAMEN)
                   .bindparams(id=class_id, name_en=name_en, name_de=name_de, name_sq=name_sq))

    # 3. Jetzt hat jede Klasse einen Code: Er darf nicht mehr leer sein und muss einmalig sein
    op.alter_column("land_cover_classes", "code", nullable=False)
    op.create_unique_constraint("uq_land_cover_classes_code", "land_cover_classes", ["code"])

    # 4. Neue Spalte mean_confidence: Kommazahl von 0 bis 1, darf leer sein (alte Zeilen)
    op.add_column("region_year_summaries", sa.Column("mean_confidence", sa.Float(), nullable=True))
    op.create_check_constraint(
        "ck_region_year_summaries_mean_confidence",
        "region_year_summaries",
        "mean_confidence >= 0 AND mean_confidence <= 1",
    )


def downgrade():
    # "downgrade" heißt: Die Änderung zurücknehmen, genau in umgekehrter Reihenfolge.

    # 4. mean_confidence wieder weg
    op.drop_constraint("ck_region_year_summaries_mean_confidence", "region_year_summaries", type_="check")
    op.drop_column("region_year_summaries", "mean_confidence")

    # 3. und 1. Spalte code wieder weg
    op.drop_constraint("uq_land_cover_classes_code", "land_cover_classes", type_="unique")
    op.drop_column("land_cover_classes", "code")

    # 2. Die alten Namen zurückschreiben
    for class_id, (name_en, name_de, name_sq) in ALTE_KLASSEN.items():
        op.execute(sa.text(UPDATE_NAMEN)
                   .bindparams(id=class_id, name_en=name_en, name_de=name_de, name_sq=name_sq))