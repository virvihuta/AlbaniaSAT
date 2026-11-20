# Tests for the settings in main.py (CORS and health check) (Regis)
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_cors_allows_local_frontend():
    r = client.get("/", headers={"Origin": "http://localhost:5173"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_blocks_unknown_website():
    r = client.get("/", headers={"Origin": "https://unknown-website.example"})
    assert "access-control-allow-origin" not in r.headers


def test_health_head():
    response = client.head("/")
    assert response.status_code == 200