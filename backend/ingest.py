# Virvi sends her results here (Regis)
import json
import os

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from db import engine

router = APIRouter(tags=["Ingest"])


# The data Virvi sends
class ClassificationResult(BaseModel):
    region_id: int
    year: int
    model_version_id: int
    class_distribution: dict
    dominant_class_id: int
    patch_count: int
    ai_summary: str = ""
    # How sure the model is on average (0 to 1). Optional.
    mean_confidence: float | None = Field(default=None, ge=0, le=1)


@router.post("/ingest/classification", status_code=201)
def ingest(data: ClassificationResult, x_api_key: str = Header()):
    """Saves the classification result of one region and year (needs the ingest API key)."""
    # 1. Only with the right key
    if x_api_key != os.environ["INGEST_API_KEY"]:
        raise HTTPException(status_code=401, detail="Wrong API key")

    # 2. Prepare the data (the distribution as JSON text)
    values = data.model_dump()
    values["class_distribution"] = json.dumps(data.class_distribution)

    # 3. Write to the database (insert new or replace the old result)
    try:
        with engine.begin() as db:
            db.execute(
                text("""
                    INSERT INTO region_year_summaries
                        (region_id, year, model_version_id, class_distribution,
                         dominant_class_id, patch_count, ai_summary, mean_confidence, generated_at)
                    VALUES
                        (:region_id, :year, :model_version_id, CAST(:class_distribution AS jsonb),
                         :dominant_class_id, :patch_count, :ai_summary, :mean_confidence, now())
                    ON CONFLICT (region_id, year, model_version_id) DO UPDATE SET
                        class_distribution = EXCLUDED.class_distribution,
                        dominant_class_id = EXCLUDED.dominant_class_id,
                        patch_count = EXCLUDED.patch_count,
                        ai_summary = EXCLUDED.ai_summary,
                        mean_confidence = EXCLUDED.mean_confidence,
                        generated_at = now()
                """),
                values,
            )
    except IntegrityError:
        raise HTTPException(status_code=400, detail="Wrong ID (region, model or class does not exist)") from None

    return {"status": "saved"}