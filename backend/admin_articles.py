# Article management for the private area: only admin and redakteur.
import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

from security import require_role

engine = create_engine(os.environ["DATABASE_URL"])
editor = require_role("admin", "redakteur")
router = APIRouter(prefix="/admin/articles", dependencies=[Depends(editor)])


class ArticleData(BaseModel):
    title: str
    slug: str
    summary: Optional[str] = None
    content: str
    image_url: Optional[str] = None
    status: str = "draft"  # "draft" or "published"


def check(data: ArticleData):
    if data.status not in ("draft", "published"):
        raise HTTPException(status_code=400, detail="Status must be draft or published")
    if not data.title.strip() or not data.slug.strip() or not data.content.strip():
        raise HTTPException(status_code=400, detail="Title, slug and content are required")


@router.get("")
def list_all():
    """All articles, also drafts, last changed first."""
    with engine.connect() as db:
        rows = db.execute(text(
            "SELECT id, title, slug, status, published_at, updated_at FROM articles ORDER BY updated_at DESC"
        )).mappings().all()
    return [dict(r) for r in rows]


@router.get("/{article_id}")
def get_one(article_id: int):
    with engine.connect() as db:
        row = db.execute(text("SELECT * FROM articles WHERE id = :id"), {"id": article_id}).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Article not found")
    return dict(row)


@router.post("")
def create(data: ArticleData, user=Depends(editor)):
    check(data)
    try:
        with engine.begin() as db:
            new_id = db.execute(text("""
                INSERT INTO articles (title, slug, summary, content, image_url, status, author_id, published_at)
                VALUES (:title, :slug, :summary, :content, :image_url, :status, :author,
                        CASE WHEN :status = 'published' THEN CURRENT_TIMESTAMP END)
                RETURNING id
            """), {**data.model_dump(), "author": int(user["sub"])}).scalar()
    except IntegrityError:
        raise HTTPException(status_code=409, detail="This slug is already used by another article")
    return {"id": new_id}


@router.put("/{article_id}")
def update(article_id: int, data: ArticleData):
    check(data)
    try:
        with engine.begin() as db:
            result = db.execute(text("""
                UPDATE articles SET title = :title, slug = :slug, summary = :summary, content = :content,
                    image_url = :image_url, status = :status, updated_at = CURRENT_TIMESTAMP,
                    published_at = CASE WHEN :status = 'published' THEN COALESCE(published_at, CURRENT_TIMESTAMP) END
                WHERE id = :id
            """), {**data.model_dump(), "id": article_id})
    except IntegrityError:
        raise HTTPException(status_code=409, detail="This slug is already used by another article")
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Article not found")
    return {"ok": True}


@router.delete("/{article_id}")
def delete(article_id: int):
    with engine.begin() as db:
        result = db.execute(text("DELETE FROM articles WHERE id = :id"), {"id": article_id})
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Article not found")
    return {"ok": True}