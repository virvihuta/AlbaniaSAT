# Endpoints for the contact form of the website (Regis)

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import text

from db import engine
from security import require_role

# Router: a group of endpoints
router = APIRouter(tags=["Contact"])

# Only the admin may read the messages
admin_only = require_role("admin")

# Same filter for the list and for counting (without year: all years)
YEAR_FILTER = "WHERE (:year IS NULL OR EXTRACT(YEAR FROM created_at) = :year)"


# The data the form sends
class ContactData(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    message: str = Field(min_length=10, max_length=2000)


# The data the admin sends to mark a message as read
class ReadStatus(BaseModel):
    is_read: bool


@router.post("/contact", status_code=201)
def send_contact(data: ContactData):
    """Saves a message from the contact form (public)."""
    with engine.begin() as db:
        db.execute(text("""
                INSERT INTO contact_messages (name, email, message)
                VALUES (:name, :email, :message)
            """), data.model_dump())
    return {"status": "sent"}


@router.get("/admin/contact")
def list_contact_messages(
    year: int | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user=Depends(admin_only),
):
    """Contact messages page by page, newest first (admin only). Optional ?year=2026."""
    with engine.connect() as db:
        total = db.execute(
            text(f"SELECT count(*) FROM contact_messages {YEAR_FILTER}"),
            {"year": year},
        ).scalar()
        rows = db.execute(
            text(f"""
                SELECT id, name, email, message, created_at, status <> 'new' AS is_read
                FROM contact_messages
                {YEAR_FILTER}
                ORDER BY created_at DESC, id DESC
                LIMIT :limit OFFSET :offset
            """),
            {"year": year, "limit": page_size, "offset": (page - 1) * page_size},
        ).mappings().all()
    return {"items": [dict(r) for r in rows], "total": total, "page": page, "page_size": page_size}


# /years and /unread-count must stay before /{message_id}
@router.get("/admin/contact/years")
def list_contact_years(user=Depends(admin_only)):
    """Years with contact messages and how many, newest year first (admin only)."""
    with engine.connect() as db:
        rows = db.execute(text("""
            SELECT CAST(EXTRACT(YEAR FROM created_at) AS int) AS year, count(*) AS count
            FROM contact_messages
            GROUP BY year
            ORDER BY year DESC
        """)).mappings().all()
    return [dict(r) for r in rows]


@router.get("/admin/contact/unread-count")
def unread_count(user=Depends(admin_only)):
    """How many messages are not read yet (for the number in the menu)."""
    with engine.connect() as db:
        count = db.execute(text("SELECT count(*) FROM contact_messages WHERE status = 'new'")).scalar()
    return {"count": count}


@router.patch("/admin/contact/{message_id}")
def mark_read(message_id: int, data: ReadStatus, user=Depends(admin_only)):
    """Marks a message as read or unread (admin only)."""
    # In the database 'read' means read and 'new' means unread
    new_status = "read" if data.is_read else "new"
    with engine.begin() as db:
        result = db.execute(
            text("UPDATE contact_messages SET status = :status WHERE id = :id"),
            {"status": new_status, "id": message_id},
        )
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Message not found")
    return {"id": message_id, "is_read": data.is_read}