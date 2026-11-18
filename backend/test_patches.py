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


def test_patch_years():
    r = client.get("/patches/years")
    assert r.status_code == 200
    years = r.json()
    assert isinstance(years, list)
    # Oldest year first
    assert years == sorted(years)


def test_patches_geojson_has_model_version():
    response = client.get("/patches/geojson?year=2021")
    assert response.status_code == 200
    first = response.json()["features"][0]
    assert first["properties"]["model_version"] == "AlbaniaSAT-SoftCon-7class-v1"


def test_patch_by_id():
    response = client.get("/patches/1")
    assert response.status_code == 200
    patch = response.json()
    assert patch["id"] == 1
    assert patch["probabilities"] is not None
    assert patch["image_url"].startswith("https://")


def test_patch_by_id_not_found():
    response = client.get("/patches/999999")
    assert response.status_code == 404