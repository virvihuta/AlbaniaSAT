# Endpoints für die Patches (Bild-Ausschnitte) auf der Karte (Regis)
import json
import os
from typing import Optional

from fastapi import APIRouter
from sqlalchemy import text

from db import engine

# Router: eine Gruppe von Endpoints
router = APIRouter()

# Anfang der Bild-Adresse in Supabase Storage (Bucket "patches", öffentlich).
# Nicht geheim. Kann in der .env mit PATCH_IMAGE_BASE_URL geändert werden.
IMAGE_BASE_URL = os.environ.get(
    "PATCH_IMAGE_BASE_URL",
    "https://rvakheaddvgnokymcwgr.supabase.co/storage/v1/object/public/patches/",
)


@router.get("/patches/geojson")
def patches_geojson(region_id: Optional[int] = None, year: Optional[int] = None):
    """Patches als Punkte für die Karte, mit Vorhersage und Bild-Link. Optional ?region_id=10&year=2021."""
    with engine.connect() as db:
        rows = db.execute(text("""
            SELECT DISTINCT ON (p.id)
                p.id, p.region_id, p.year, p.reference_class_id,
                ST_AsGeoJSON(p.center_geom, 6) AS geometry,
                pr.predicted_class_id,
                CAST(pr.confidence AS float) AS confidence,
                :base || p.preview_path AS image_url
            FROM patches p
            LEFT JOIN predictions pr ON pr.patch_id = p.id
            WHERE (:region_id IS NULL OR p.region_id = :region_id)
              AND (:year IS NULL OR p.year = :year)
            ORDER BY p.id, pr.model_version_id DESC
        """), {"base": IMAGE_BASE_URL, "region_id": region_id, "year": year}).mappings().all()

    features = []
    for r in rows:
        features.append({
            "type": "Feature",
            "geometry": json.loads(r["geometry"]),
            "properties": {
                "id": r["id"],
                "region_id": r["region_id"],
                "year": r["year"],
                "predicted_class_id": r["predicted_class_id"],
                "confidence": r["confidence"],
                "reference_class_id": r["reference_class_id"],
                "image_url": r["image_url"],
            },
        })
    return {"type": "FeatureCollection", "features": features}