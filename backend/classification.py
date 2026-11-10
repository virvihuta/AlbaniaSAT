# Endpoints für Regionen und Klassifikation (Regis)
import json

from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from db import engine

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
                       s.ai_summary, s.generated_at, s.mean_confidence,
                       s.dominant_class_id, c.code AS dominant_code,
                       c.name_en, c.name_de, c.name_sq, c.color_hex
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
    """Die 7 Klassen mit Code, Namen in 3 Sprachen und Farbe (für die Legende)."""
    with engine.connect() as db:
        rows = db.execute(
            text("""
                SELECT id, code, name_en, name_de, name_sq, color_hex
                FROM land_cover_classes
                ORDER BY id
            """)
        ).mappings().all()
    return [dict(r) for r in rows]


@router.get("/classification/{region_id}")
def get_classification_years(region_id: int):
    """Alle Jahre einer Region, sortiert nach Jahr (für die Zeitreihe)."""
    with engine.connect() as db:
        # 1. Gibt es die Region?
        region = db.execute(
            text("SELECT id, name FROM regions WHERE id = :region_id"),
            {"region_id": region_id},
        ).mappings().first()
        if not region:
            raise HTTPException(status_code=404, detail="Region nicht gefunden")

        # 2. Alle Jahre dieser Region holen (pro Jahr nur das neueste Ergebnis)
        rows = db.execute(
            text("""
                SELECT DISTINCT ON (year)
                       year, patch_count, class_distribution, dominant_class_id, mean_confidence
                FROM region_year_summaries
                WHERE region_id = :region_id
                ORDER BY year, generated_at DESC
            """),
            {"region_id": region_id},
        ).mappings().all()

    # 3. Antwort: Region und Liste der Jahre
    return {
        "region_id": region["id"],
        "region_name": region["name"],
        "years": [dict(r) for r in rows],
    }


@router.get("/regions/geojson")
def regions_geojson():
    """Grenzen aller Regionen als GeoJSON (für die Karte)."""
    with engine.connect() as db:
        rows = db.execute(
            text("""
                SELECT id, name, code, ST_AsGeoJSON(geom, 6) AS geometry
                FROM regions
                ORDER BY name
            """)
        ).mappings().all()

    # Jede Region wird ein "Feature": Form + Informationen
    features = []
    for r in rows:
        features.append({
            "type": "Feature",
            "geometry": json.loads(r["geometry"]),
            "properties": {"id": r["id"], "name": r["name"], "code": r["code"]},
        })

    # Alle Features zusammen sind eine "FeatureCollection"
    return {"type": "FeatureCollection", "features": features}


@router.get("/years")
def list_years():
    """Alle Jahre, für die es Ergebnisse gibt (für den Jahr-Filter auf der Karte)."""
    with engine.connect() as db:
        rows = db.execute(
            text("SELECT DISTINCT year FROM region_year_summaries ORDER BY year")
        ).mappings().all()
    return [r["year"] for r in rows]