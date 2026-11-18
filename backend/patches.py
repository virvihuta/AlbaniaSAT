# Endpoints for the patches (image tiles) on the map (Regis)
import json
import os

from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from db import engine

router = APIRouter()

# Start of the image address in Supabase Storage (bucket "patches", public).
# Not a secret. Can be changed in the .env with PATCH_IMAGE_BASE_URL.
IMAGE_BASE_URL = os.environ.get(
    "PATCH_IMAGE_BASE_URL",
    "https://rvakheaddvgnokymcwgr.supabase.co/storage/v1/object/public/patches/",
)


# Must stay before /patches/{patch_id}, otherwise FastAPI would think "years" is an id
@router.get("/patches/years")
def patch_years():
    """Years that have patches, oldest first (for the year buttons on the map)."""
    with engine.connect() as db:
        rows = db.execute(text("SELECT DISTINCT year FROM patches ORDER BY year")).mappings().all()
    return [r["year"] for r in rows]


@router.get("/patches/geojson")
def patches_geojson(region_id: int | None = None, year: int | None = None):
    """Patches as points for the map, with prediction, model version and image link. Optional ?region_id=10&year=2021."""
    with engine.connect() as db:
        rows = db.execute(text("""
            SELECT DISTINCT ON (p.id)
                p.id, p.region_id, p.year, p.reference_class_id,
                ST_AsGeoJSON(p.center_geom, 6) AS geometry,
                pr.predicted_class_id,
                CAST(pr.confidence AS float) AS confidence,
                mv.label AS model_version,
                :base || p.preview_path AS image_url
            FROM patches p
            LEFT JOIN predictions pr ON pr.patch_id = p.id
            LEFT JOIN model_versions mv ON mv.id = pr.model_version_id
            WHERE (:region_id IS NULL OR p.region_id = :region_id)
              AND (:year IS NULL OR p.year = :year)
            ORDER BY p.id, pr.model_version_id DESC
        """), {"base": IMAGE_BASE_URL, "region_id": region_id, "year": year}).mappings().all()

    # Each patch becomes a GeoJSON feature: a point with information
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
                "model_version": r["model_version"],
                "reference_class_id": r["reference_class_id"],
                "image_url": r["image_url"],
            },
        })
    return {"type": "FeatureCollection", "features": features}


# Must stay at the end, after /patches/years and /patches/geojson
@router.get("/patches/{patch_id}")
def get_patch(patch_id: int):
    """One patch with image link, prediction and the probabilities of all classes."""
    with engine.connect() as db:
        row = db.execute(text("""
            SELECT p.id, p.region_id, p.year, p.reference_class_id,
                   pr.predicted_class_id,
                   CAST(pr.confidence AS float) AS confidence,
                   pr.probabilities,
                   mv.label AS model_version,
                   :base || p.preview_path AS image_url
            FROM patches p
            LEFT JOIN predictions pr ON pr.patch_id = p.id
            LEFT JOIN model_versions mv ON mv.id = pr.model_version_id
            WHERE p.id = :patch_id
            ORDER BY pr.model_version_id DESC
            LIMIT 1
        """), {"base": IMAGE_BASE_URL, "patch_id": patch_id}).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Patch not found")
    return dict(row)