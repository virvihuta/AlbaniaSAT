# Endpoints für das Kontaktformular der Website (Regis)
import os

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

from security import require_role

# Verbindung zur Datenbank (Adresse kommt aus der .env)
engine = create_engine(os.environ["DATABASE_URL"])

# Router: eine Gruppe von Endpoints
router = APIRouter()

# Nur der Admin darf die Nachrichten lesen
admin_only = require_role("admin")


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
def list_contact_messages(user=Depends(admin_only)):
    """Alle Kontaktnachrichten, die neuesten zuerst (nur admin)."""
    with engine.connect() as db:
        rows = db.execute(text("""
                SELECT id, name, email, message, created_at
                FROM contact_messages
                ORDER BY created_at DESC
            """)).mappings().all()
    return [dict(r) for r in rows]