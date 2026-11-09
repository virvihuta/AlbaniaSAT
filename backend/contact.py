# Endpoints für das Kontaktformular der Website (Regis)
import os
from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

from security import require_role

# Verbindung zur Datenbank (Adresse kommt aus der .env)
engine = create_engine(os.environ["DATABASE_URL"])

# Router: eine Gruppe von Endpoints
router = APIRouter()

# Nur der Admin darf die Nachrichten lesen
admin_only = require_role("admin")

# Gleicher Filter für die Liste und für das Zählen (ohne year: alle Jahre)
YEAR_FILTER = "WHERE (:year IS NULL OR EXTRACT(YEAR FROM created_at) = :year)"


# Diese Daten schickt das Formular
class Kontakt(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    message: str = Field(min_length=10, max_length=2000)


@router.post("/contact", status_code=201)
def send_contact(data: Kontakt):
    """Speichert eine Nachricht aus dem Kontaktformular (öffentlich)."""
    with engine.begin() as db:
        db.execute(text("""
                INSERT INTO contact_messages (name, email, message)
                VALUES (:name, :email, :message)
            """), data.model_dump())
    return {"status": "gesendet"}


@router.get("/admin/contact")
def list_contact_messages(
    year: Optional[int] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user=Depends(admin_only),
):
    """Kontaktnachrichten Seite für Seite, die neuesten zuerst (nur admin). Optional ?year=2026."""
    with engine.connect() as db:
        total = db.execute(
            text(f"SELECT count(*) FROM contact_messages {YEAR_FILTER}"),
            {"year": year},
        ).scalar()
        rows = db.execute(
            text(f"""
                SELECT id, name, email, message, created_at
                FROM contact_messages
                {YEAR_FILTER}
                ORDER BY created_at DESC, id DESC
                LIMIT :limit OFFSET :offset
            """),
            {"year": year, "limit": page_size, "offset": (page - 1) * page_size},
        ).mappings().all()
    return {"items": [dict(r) for r in rows], "total": total, "page": page, "page_size": page_size}


@router.get("/admin/contact/years")
def list_contact_years(user=Depends(admin_only)):
    """Jahre mit Kontaktnachrichten und wie viele, neuestes Jahr zuerst (nur admin)."""
    with engine.connect() as db:
        rows = db.execute(text("""
            SELECT CAST(EXTRACT(YEAR FROM created_at) AS int) AS year, count(*) AS count
            FROM contact_messages
            GROUP BY year
            ORDER BY year DESC
        """)).mappings().all()
    return [dict(r) for r in rows]