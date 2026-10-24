# Public news endpoints: only published articles, newest first.
import os

from fastapi import APIRouter, HTTPException
from sqlalchemy import create_engine, text

engine = create_engine(os.environ["DATABASE_URL"])
router = APIRouter(prefix="/articles")

COLUMNS = """
    a.id, a.title, a.slug, a.summary, a.image_url, a.published_at,
    u.display_name AS author
"""


@router.get("")
def list_articles(q: str = ""):
    """All published articles. With ?q=... only those whose title or summary contains the text."""
    with engine.connect() as db:
        rows = db.execute(
            text(f"""
                SELECT {COLUMNS}
                FROM articles a LEFT JOIN users u ON u.id = a.author_id
                WHERE a.status = 'published'
                  AND (lower(a.title) LIKE :q OR lower(coalesce(a.summary, '')) LIKE :q)
                ORDER BY a.published_at DESC
            """),
            {"q": f"%{q.lower()}%"},
        ).mappings().all()
    return [dict(r) for r in rows]


@router.get("/{slug}")
def get_article(slug: str):
    """One published article with its full content."""
    with engine.connect() as db:
        row = db.execute(
            text(f"""
                SELECT {COLUMNS}, a.content
                FROM articles a LEFT JOIN users u ON u.id = a.author_id
                WHERE a.status = 'published' AND a.slug = :slug
            """),
            {"slug": slug},
        ).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Article not found")
    return dict(row)