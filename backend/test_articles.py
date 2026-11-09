# Tests für die Pagination in articles.py und admin_articles.py (Regis)
# Diese Tests schreiben nichts in die Datenbank.
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_articles_format():
    r = client.get("/articles?page=1&page_size=1")
    assert r.status_code == 200
    data = r.json()
    assert data["page"] == 1
    assert data["page_size"] == 1
    assert len(data["items"]) <= 1
    assert "total" in data


def test_articles_page_size_too_big():
    r = client.get("/articles?page_size=51")
    assert r.status_code == 422


def test_articles_page_zero():
    r = client.get("/articles?page=0")
    assert r.status_code == 422


def test_admin_articles_without_token():
    r = client.get("/admin/articles")
    assert r.status_code == 401


def test_articles_years():
    r = client.get("/articles/years")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_articles_year_without_articles():
    r = client.get("/articles?year=1990")
    assert r.status_code == 200
    assert r.json()["total"] == 0


def test_articles_wrong_category():
    r = client.get("/articles?category=abc")
    assert r.status_code == 422


def test_articles_category_filter():
    r = client.get("/articles?category=forests")
    assert r.status_code == 200
    assert "total" in r.json()


def test_admin_articles_wrong_status():
    r = client.get("/admin/articles?status=abc")
    assert r.status_code in (401, 422)