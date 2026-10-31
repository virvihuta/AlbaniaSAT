# Endpoints für Regionen und Klassifikation (Regis)
import os

from fastapi import APIRouter, HTTPException
from sqlalchemy import create_engine, text

# Verbindung zur Datenbank (Adresse kommt aus der .env)
engine = create_engine(os.environ["DATABASE_URL"])

# Router: eine Gruppe von Endpoints
router = APIRouter()


@router.get("/regions")
def list_regions():
    """Alle Regionen, sortiert nach Name (ohne Geometrie)."""
    with engine.connect() as db:
        rows = db.execute(
            text("""
                SELECT id, name, code, level, parent_id, area_km2
                FROM regions
                ORDER BY name
            """)
        ).mappings().all()
    return [dict(r) for r in rows]


@router.get("/classification/{region_id}/{year}")
def get_classification(region_id: int, year: int):
    """Ergebnis der Klassifikation für eine Region in einem Jahr."""
    with engine.connect() as db:
        row = db.execute(
            text("""
                SELECT s.region_id, r.name AS region_name, s.year,
                       s.patch_count, s.class_distribution, s.change_vs_previous,
                       s.ai_summary, s.generated_at,
                       s.dominant_class_id, c.name_en, c.name_de, c.name_sq, c.color_hex
                FROM region_year_summaries s
                JOIN regions r ON r.id = s.region_id
                LEFT JOIN land_cover_classes c ON c.id = s.dominant_class_id
                WHERE s.region_id = :region_id AND s.year = :year
                ORDER BY s.generated_at DESC
                LIMIT 1
            """),
            {"region_id": region_id, "year": year},
        ).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Keine Daten für diese Region und dieses Jahr")
    return dict(row)


@router.get("/classes")
def list_classes():
    """Die 7 Klassen mit Namen in 3 Sprachen und Farbe (für die Legende)."""
    with engine.connect() as db:
        rows = db.execute(
            text("""
                SELECT id, name_en, name_de, name_sq, color_hex
                FROM land_cover_classes
                ORDER BY id
            """)
        ).mappings().all()
    return [dict(r) for r in rows]

