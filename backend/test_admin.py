# Tests for the admin endpoints with a login token (Regis)
# The tokens are made directly with create_token, so no login and no user in the database is needed.
# These tests only read from the database or check error cases. They never save anything.
from fastapi.testclient import TestClient

from main import app
from security import create_token

client = TestClient(app)

# Headers with a valid token, like after a login
ADMIN = {"Authorization": "Bearer " + create_token(1, "admin")}
REDAKTEUR = {"Authorization": "Bearer " + create_token(1, "redakteur")}


def test_login_with_wrong_password():
    data = {"email": "nobody@example.com", "password": "wrong-password"}
    r = client.post("/auth/login", json=data)
    assert r.status_code == 401


def test_invalid_token():
    r = client.get("/admin/faqs", headers={"Authorization": "Bearer not-a-real-token"})
    assert r.status_code == 401


def test_admin_articles_with_counts():
    r = client.get("/admin/articles?page=1&page_size=5", headers=ADMIN)
    assert r.status_code == 200
    data = r.json()
    assert set(data["counts"]) == {"all", "published", "draft"}
    assert len(data["items"]) <= 5


def test_admin_articles_wrong_status_with_login():
    r = client.get("/admin/articles?status=abc", headers=ADMIN)
    assert r.status_code == 422


def test_admin_article_not_found():
    r = client.get("/admin/articles/999999", headers=ADMIN)
    assert r.status_code == 404


def test_admin_contact_list():
    r = client.get("/admin/contact?page=1&page_size=5", headers=ADMIN)
    assert r.status_code == 200
    assert "total" in r.json()


def test_admin_contact_years_and_unread_count():
    assert client.get("/admin/contact/years", headers=ADMIN).status_code == 200
    r = client.get("/admin/contact/unread-count", headers=ADMIN)
    assert r.status_code == 200
    assert "count" in r.json()


def test_admin_contact_not_allowed_for_redakteur():
    r = client.get("/admin/contact", headers=REDAKTEUR)
    assert r.status_code == 403


def test_mark_read_unknown_message():
    r = client.patch("/admin/contact/999999", json={"is_read": True}, headers=ADMIN)
    assert r.status_code == 404


def test_admin_faqs_list():
    r = client.get("/admin/faqs", headers=ADMIN)
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_create_faq_question_too_short():
    data = {"question": "Hi", "answer": "Only a test answer.", "is_published": True}
    r = client.post("/admin/faqs", json=data, headers=ADMIN)
    assert r.status_code == 422


def test_reorder_unknown_faq():
    r = client.put("/admin/faqs/order", json={"ids": [999999]}, headers=ADMIN)
    assert r.status_code == 404


def test_update_unknown_faq():
    data = {"question": "Test question?", "answer": "Only a test answer.", "is_published": True}
    r = client.put("/admin/faqs/999999", json=data, headers=ADMIN)
    assert r.status_code == 404


def test_delete_unknown_faq_as_redakteur():
    # A redakteur may manage FAQs, so this is 404 (unknown id) and not 403
    r = client.delete("/admin/faqs/999999", headers=REDAKTEUR)
    assert r.status_code == 404