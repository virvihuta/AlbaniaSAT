# Virvi schickt hier ihre Ergebnisse (Regis)
import json
import os
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

engine = create_engine(os.environ["DATABASE_URL"])
router = APIRouter()


# Diese Daten schickt Virvi
class Ergebnis(BaseModel):
    region_id: int
    year: int
    model_version_id: int
    class_distribution: dict
    dominant_class_id: int
    patch_count: int
    ai_summary: str = ""
    # Wie sicher das Modell im Durchschnitt ist (0 bis 1). Freiwillig.
    mean_confidence: Optional[float] = Field(default=None, ge=0, le=1)


@router.post("/ingest/classification", status_code=201)
def ingest(data: Ergebnis, x_api_key: str = Header()):
    # 1. Nur mit dem richtigen Schlüssel
    if x_api_key != os.environ["INGEST_API_KEY"]:
        raise HTTPException(status_code=401, detail="Falscher Schlüssel")

    # 2. Daten vorbereiten (die Verteilung als JSON-Text)
    werte = data.model_dump()
    werte["class_distribution"] = json.dumps(data.class_distribution)

    # 3. In die Datenbank schreiben (neu einfügen oder altes Ergebnis ersetzen)
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
                werte,
            )
    except IntegrityError:
        raise HTTPException(status_code=400, detail="Falsche ID (Region, Modell oder Klasse gibt es nicht)")

    return {"status": "gespeichert"}