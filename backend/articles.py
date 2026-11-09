# Public news endpoints: only published articles, newest first.
import os
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import create_engine, text

engine = create_engine(os.environ["DATABASE_URL"])
router = APIRouter(prefix="/articles")

# Allowed topics (the same list as the check in the database). Other values → 422
Category = Literal["urban_growth", "forests", "coast", "agriculture", "methodology"]

COLUMNS = """
    a.id, a.title, a.slug, a.summary, a.image_url, a.published_at, a.category,
    u.display_name AS author
"""

# Same filter for the list and for the count
FILTER = """
    WHERE a.status = 'published'
      AND (lower(a.title) LIKE :q OR lower(coalesce(a.summary, '')) LIKE :q)
      AND (:year IS NULL OR EXTRACT(YEAR FROM a.published_at) = :year)
      AND (:category IS NULL OR a.category = :category)
"""


@router.get("")
def list_articles(
    q: str = "",
    year: Optional[int] = None,
    category: Optional[Category] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(9, ge=1, le=50),
):
    """Published articles, page by page. Optional search (?q=...), year (?year=2025) and topic (?category=forests)."""
    params = {"q": f"%{q.lower()}%", "year": year, "category": category}
    with engine.connect() as db:
        total = db.execute(
            text(f"SELECT count(*) FROM articles a {FILTER}"),
            params,
        ).scalar()
        rows = db.execute(
            text(f"""
                SELECT {COLUMNS}
                FROM articles a LEFT JOIN users u ON u.id = a.author_id
                {FILTER}
                ORDER BY a.published_at DESC, a.id DESC
                LIMIT :limit OFFSET :offset
            """),
            {**params, "limit": page_size, "offset": (page - 1) * page_size},
        ).mappings().all()
    return {"items": [dict(r) for r in rows], "total": total, "page": page, "page_size": page_size}


# Must stay before /{slug}, otherwise FastAPI thinks "years" is a slug
@router.get("/years")
def list_years():
    """Years with published articles and how many, newest year first."""
    with engine.connect() as db:
        rows = db.execute(text("""
            SELECT CAST(EXTRACT(YEAR FROM published_at) AS int) AS year, count(*) AS count
            FROM articles
            WHERE status = 'published'
            GROUP BY year
            ORDER BY year DESC
        """)).mappings().all()
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