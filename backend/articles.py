# Public news endpoints: only published articles, newest first.
import os

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import create_engine, text

engine = create_engine(os.environ["DATABASE_URL"])
router = APIRouter(prefix="/articles")

COLUMNS = """
    a.id, a.title, a.slug, a.summary, a.image_url, a.published_at,
    u.display_name AS author
"""

# Same filter for the list and for the count
FILTER = """
    WHERE a.status = 'published'
      AND (lower(a.title) LIKE :q OR lower(coalesce(a.summary, '')) LIKE :q)
"""


@router.get("")
def list_articles(q: str = "", page: int = Query(1, ge=1), page_size: int = Query(9, ge=1, le=50)):
    """Published articles, page by page. With ?q=... only those whose title or summary contains the text."""
    search = f"%{q.lower()}%"
    with engine.connect() as db:
        total = db.execute(
            text(f"SELECT count(*) FROM articles a {FILTER}"),
            {"q": search},
        ).scalar()
        rows = db.execute(
            text(f"""
                SELECT {COLUMNS}
                FROM articles a LEFT JOIN users u ON u.id = a.author_id
                {FILTER}
                ORDER BY a.published_at DESC, a.id DESC
                LIMIT :limit OFFSET :offset
            """),
            {"q": search, "limit": page_size, "offset": (page - 1) * page_size},
        ).mappings().all()
    return {"items": [dict(r) for r in rows], "total": total, "page": page, "page_size": page_size}


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