# Endpoint for the results of the trained models (Regis)
from fastapi import APIRouter
from sqlalchemy import text

from db import engine

router = APIRouter()


@router.get("/metrics")
def list_metrics():
    """Results of all trained models, newest first (accuracy and F1 score from 0 to 1)."""
    with engine.connect() as db:
        rows = db.execute(text("""
            SELECT id, label, architecture, input_bands,
                   CAST(accuracy AS float) AS accuracy,
                   CAST(f1_macro AS float) AS f1_macro,
                   trained_at, notes
            FROM model_versions
            ORDER BY trained_at DESC, id DESC
        """)).mappings().all()
    return [dict(r) for r in rows]