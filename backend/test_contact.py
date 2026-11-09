# Tests für die Endpoints in contact.py (Regis)
# Diese Tests schreiben nichts in die Datenbank.
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_contact_message_too_short():
    data = {"name": "Regis", "email": "test@test.com", "message": "kurz"}
    r = client.post("/contact", json=data)
    assert r.status_code == 422


def test_contact_wrong_email():
    data = {"name": "Regis", "email": "keine-email", "message": "Das ist eine Testnachricht."}
    r = client.post("/contact", json=data)
    assert r.status_code == 422


def test_admin_contact_without_token():
    r = client.get("/admin/contact")
    assert r.status_code == 401


def test_admin_contact_years_without_token():
    r = client.get("/admin/contact/years")
    assert r.status_code == 401


def test_unread_count_without_token():
    r = client.get("/admin/contact/unread-count")
    assert r.status_code == 401


def test_mark_read_without_token():
    r = client.patch("/admin/contact/1", json={"is_read": True})
    assert r.status_code == 401