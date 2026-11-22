# Endpoints for regions and classification (Regis)
import json

from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from db import engine

# Router: a group of endpoints
router = APIRouter(tags=["Regions and classification"])


@router.get("/regions")
def list_regions():
    """All regions, sorted by name (without geometry)."""
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
    """Classification result for one region in one year."""
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
        raise HTTPException(status_code=404, detail="No data for this region and year")
    return dict(row)


@router.get("/classes")
def list_classes():
    """The 7 classes with code, names in 3 languages and color (for the legend)."""
    with engine.connect() as db:
        rows = db.execute(
            text("""
                SELECT id, code, name_en, name_de, name_sq, color_hex
                FROM land_cover_classes
                ORDER BY id
            """)
        ).mappings().all()
    return [dict(r) for r in rows]


def calculate_change(previous, current):
    """Share this year minus share last year, for each class. None for the first year."""
    if previous is None or current is None:
        return None
    change = {}
    for code in current:
        change[code] = round(current[code] - previous.get(code, 0), 4)
    return change


@router.get("/classification/{region_id}")
def get_classification_years(region_id: int):
    """All years of one region, sorted by year (for the time series)."""
    with engine.connect() as db:
        # 1. Does the region exist?
        region = db.execute(
            text("SELECT id, name FROM regions WHERE id = :region_id"),
            {"region_id": region_id},
        ).mappings().first()
        if not region:
            raise HTTPException(status_code=404, detail="Region not found")

        # 2. Get all years of this region (only the newest result per year)
        rows = db.execute(
            text("""
                SELECT DISTINCT ON (year)
                       year, patch_count, class_distribution, dominant_class_id,
                       mean_confidence, ai_summary
                FROM region_year_summaries
                WHERE region_id = :region_id
                ORDER BY year, generated_at DESC
            """),
            {"region_id": region_id},
        ).mappings().all()

    # 3. Add the change compared to the year before
    years = []
    previous = None
    for r in rows:
        year = dict(r)
        year["change_vs_previous"] = calculate_change(previous, year["class_distribution"])
        years.append(year)
        previous = year["class_distribution"]

    # 4. Answer: region and list of years
    return {
        "region_id": region["id"],
        "region_name": region["name"],
        "years": years,
    }


@router.get("/regions/geojson")
def regions_geojson():
    """Borders of all regions as GeoJSON (for the map)."""
    with engine.connect() as db:
        rows = db.execute(
            text("""
                SELECT id, name, code, ST_AsGeoJSON(geom, 6) AS geometry
                FROM regions
                ORDER BY name
            """)
        ).mappings().all()

    # Each region becomes a "feature": shape + information
    features = []
    for r in rows:
        features.append({
            "type": "Feature",
            "geometry": json.loads(r["geometry"]),
            "properties": {"id": r["id"], "name": r["name"], "code": r["code"]},
        })

    # All features together are a "FeatureCollection"
    return {"type": "FeatureCollection", "features": features}


@router.get("/years")
def list_years():
    """All years that have results (for the year filter on the map)."""
    with engine.connect() as db:
        rows = db.execute(
            text("SELECT DISTINCT year FROM region_year_summaries ORDER BY year")
        ).mappings().all()
    return [r["year"] for r in rows]