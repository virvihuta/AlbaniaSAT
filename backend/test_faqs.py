# Tests for the endpoints in faqs.py (Regis)
# These tests don't write anything to the database.
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_faqs_public():
    r = client.get("/faqs")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_admin_faqs_without_token():
    r = client.get("/admin/faqs")
    assert r.status_code == 401


def test_create_faq_without_token():
    data = {"question": "Is this a test?", "answer": "Yes, this is only a test.", "is_published": True}
    r = client.post("/admin/faqs", json=data)
    assert r.status_code == 401


def test_reorder_faqs_without_token():
    r = client.put("/admin/faqs/order", json={"ids": [1, 2]})
    assert r.status_code == 401