# Tests für die Endpoints in classification.py und ingest.py (Regis)
# Starten im Ordner backend: pytest -v
from fastapi.testclient import TestClient

from main import app

# Ein "Fake-Browser", der Anfragen an unsere App schickt
client = TestClient(app)


def test_regions():
    r = client.get("/regions")
    assert r.status_code == 200
    assert len(r.json()) == 12


def test_regions_geojson():
    r = client.get("/regions/geojson")
    assert r.status_code == 200
    assert r.json()["type"] == "FeatureCollection"
    assert len(r.json()["features"]) == 12


def test_classes():
    r = client.get("/classes")
    assert r.status_code == 200
    assert len(r.json()) == 7
    assert r.json()[0]["name_de"] == "Laubwald"


def test_classification_years():
    r = client.get("/classification/10")
    assert r.status_code == 200
    assert r.json()["region_name"] == "Shkodër"


def test_classification_unknown_region():
    r = client.get("/classification/99")
    assert r.status_code == 404


def test_classification_wrong_type():
    r = client.get("/classification/abc/2021")
    assert r.status_code == 422


def test_ingest_wrong_key():
    data = {
        "region_id": 10,
        "year": 2021,
        "model_version_id": 1,
        "class_distribution": {"1": 0.5},
        "dominant_class_id": 1,
        "patch_count": 1,
    }
    r = client.post("/ingest/classification", json=data, headers={"X-API-Key": "falsch"})
    assert r.status_code == 401


def test_ingest_missing_data():
    r = client.post("/ingest/classification", json={}, headers={"X-API-Key": "falsch"})
    assert r.status_code == 422