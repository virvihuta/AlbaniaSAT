# Adds two example news articles so the news page has something to show.
# Usage: python seed_articles.py
import os

from sqlalchemy import create_engine, text

engine = create_engine(os.environ["DATABASE_URL"])

articles = [
    {
        "title": "AlbaniaSAT reaches 80.94 % accuracy",
        "slug": "albaniasat-reaches-80-percent-accuracy",
        "summary": "Our model now classifies seven land cover classes across Albania.",
        "content": "After several training rounds, the AlbaniaSAT model reaches 80.94 % accuracy on our own Albanian test data. It recognises forest, shrubland, grassland, cropland, built-up areas, bare ground and water.",
    },
    {
        "title": "A new Sentinel-2 dataset for Albania",
        "slug": "new-sentinel-2-dataset-for-albania",
        "summary": "About 7,800 satellite image patches were labeled for training.",
        "content": "Until now there was no labeled satellite dataset for Albania. With Google Earth Engine we collected about 7,800 Sentinel-2 patches and labeled them with ESA WorldCover and CORINE data.",
    },
]

with engine.begin() as db:
    for a in articles:
        db.execute(
            text("""
                INSERT INTO articles (title, slug, summary, content, status, published_at)
                VALUES (:title, :slug, :summary, :content, 'published', CURRENT_TIMESTAMP)
            """),
            a,
        )
print(f"Added {len(articles)} articles")