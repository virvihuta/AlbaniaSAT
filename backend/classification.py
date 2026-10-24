# Endpoints für Regionen und Klassifikation (Regis)
import os

from fastapi import APIRouter
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