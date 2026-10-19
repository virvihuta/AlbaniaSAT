# Tests the login without Supabase: uses a small temporary SQLite database.
# Run from the backend folder: python test_auth.py
import os

os.environ["DATABASE_URL"] = "sqlite:///test_auth.db"
os.environ["JWT_SECRET"] = "only-for-testing-" + "x" * 32

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from security import hash_password, require_role

# 1. Fake users table with one admin
engine = create_engine(os.environ["DATABASE_URL"])
with engine.begin() as db:
    db.execute(text("DROP TABLE IF EXISTS users"))
    db.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY, display_name TEXT, email TEXT, password_hash TEXT, role TEXT, is_active BOOLEAN DEFAULT 1, last_login_at TIMESTAMP)"))
    db.execute(
        text("INSERT INTO users (display_name, email, password_hash, role) VALUES ('Suisa', 'suisa@test.al', :p, 'admin')"),
        {"p": hash_password("geheim123")},
    )

# 2. Small app with the login and one protected endpoint
from auth import router

app = FastAPI()
app.include_router(router)


@app.get("/secret")
def secret(user=Depends(require_role("admin"))):
    return {"ok": True}


client = TestClient(app)

# 3. The tests
wrong = client.post("/auth/login", json={"email": "suisa@test.al", "password": "falsch"})
print("Wrong password      ->", wrong.status_code, "(should be 401)")

right = client.post("/auth/login", json={"email": "suisa@test.al", "password": "geheim123"})
print("Right password      ->", right.status_code, "(should be 200)", right.json().get("user"))

token = right.json()["token"]
print("Secret with token   ->", client.get("/secret", headers={"Authorization": "Bearer " + token}).status_code, "(should be 200)")
print("Secret without token->", client.get("/secret").status_code, "(should be 401 or 403)")

os.remove("test_auth.db")