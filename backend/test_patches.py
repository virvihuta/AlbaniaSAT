# Tests für die Endpoints in patches.py (Regis)
# Diese Tests schreiben nichts in die Datenbank.
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_patches_geojson():
    r = client.get("/patches/geojson?region_id=10&year=2021")
    assert r.status_code == 200
    data = r.json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) > 0


def test_patches_have_point_and_image_url():
    r = client.get("/patches/geojson?region_id=10&year=2021")
    first = r.json()["features"][0]
    assert first["geometry"]["type"] == "Point"
    assert first["properties"]["image_url"].startswith("https://")


def test_patches_wrong_type():
    r = client.get("/patches/geojson?region_id=abc")
    assert r.status_code == 422