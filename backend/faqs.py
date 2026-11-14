# Endpoints for the FAQ on the contact page (Regis)
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text

from db import engine
from security import require_role

router = APIRouter()

# Admin and redakteur may manage the FAQs (like the articles)
editor = require_role("admin", "redakteur")


# Data for a new or changed FAQ
class FaqData(BaseModel):
    question: str = Field(min_length=5, max_length=200)
    answer: str = Field(min_length=10, max_length=2000)
    is_published: bool = True


# New order of the FAQs: the ids from top to bottom
class FaqOrder(BaseModel):
    ids: list[int] = Field(min_length=1)


@router.get("/faqs")
def list_faqs():
    """Published FAQs for the contact page, in the order set in the admin area."""
    with engine.connect() as db:
        rows = db.execute(text("""
            SELECT id, question, answer
            FROM faqs
            WHERE is_published
            ORDER BY position, id
        """)).mappings().all()
    return [dict(r) for r in rows]


@router.get("/admin/faqs")
def list_all_faqs(user=Depends(editor)):
    """All FAQs, also hidden ones, in their order (admin and redakteur)."""
    with engine.connect() as db:
        rows = db.execute(text("""
            SELECT id, question, answer, position, is_published, updated_at
            FROM faqs
            ORDER BY position, id
        """)).mappings().all()
    return [dict(r) for r in rows]


@router.post("/admin/faqs", status_code=201)
def create_faq(data: FaqData, user=Depends(editor)):
    """New FAQ at the end of the list."""
    with engine.begin() as db:
        new_id = db.execute(
            text("""
                INSERT INTO faqs (question, answer, is_published, position)
                VALUES (:question, :answer, :is_published,
                        (SELECT COALESCE(MAX(position), -1) + 1 FROM faqs))
                RETURNING id
            """),
            data.model_dump(),
        ).scalar()
    return {"id": new_id}


# Must stay before /admin/faqs/{faq_id}, otherwise FastAPI thinks "order" is an id
@router.put("/admin/faqs/order")
def reorder_faqs(data: FaqOrder, user=Depends(editor)):
    """Sets the position of each FAQ by its place in the list (for the up/down buttons)."""
    with engine.begin() as db:
        for position, faq_id in enumerate(data.ids):
            result = db.execute(
                text("UPDATE faqs SET position = :position WHERE id = :id"),
                {"position": position, "id": faq_id},
            )
            if result.rowcount == 0:
                raise HTTPException(status_code=404, detail=f"FAQ {faq_id} not found")
    return {"ok": True}


@router.put("/admin/faqs/{faq_id}")
def update_faq(faq_id: int, data: FaqData, user=Depends(editor)):
    """Changes question, answer and visibility of one FAQ."""
    with engine.begin() as db:
        result = db.execute(
            text("""
                UPDATE faqs
                SET question = :question, answer = :answer, is_published = :is_published,
                    updated_at = now()
                WHERE id = :id
            """),
            {**data.model_dump(), "id": faq_id},
        )
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="FAQ not found")
    return {"ok": True}


@router.delete("/admin/faqs/{faq_id}")
def delete_faq(faq_id: int, user=Depends(editor)):
    """Deletes one FAQ."""
    with engine.begin() as db:
        result = db.execute(text("DELETE FROM faqs WHERE id = :id"), {"id": faq_id})
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="FAQ not found")
    return {"ok": True}