# Tests for GET /metrics (Regis)
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    models = response.json()
    assert len(models) >= 1
    for model in models:
        assert 0 <= model["accuracy"] <= 1
        assert 0 <= model["f1_macro"] <= 1