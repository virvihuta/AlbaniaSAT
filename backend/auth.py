# Login endpoint: checks the password against the hash and returns a JWT.
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text

from security import create_token, verify_password

engine = create_engine(os.environ["DATABASE_URL"])
router = APIRouter(prefix="/auth")


class LoginData(BaseModel):
    email: str
    password: str


@router.post("/login")
def login(data: LoginData):
    with engine.connect() as db:
        user = db.execute(
            text("SELECT id, display_name, email, password_hash, role, is_active FROM users WHERE email = :email"),
            {"email": data.email},
        ).mappings().first()

    # Same message for wrong email and wrong password, so nobody can guess emails
    if not user or not verify_password(data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Wrong email or password")
    if not user["is_active"]:
        raise HTTPException(status_code=403, detail="This account is disabled")
    # Only admins and editors may log in, visitors need no account
    if user["role"] not in ("admin", "redakteur"):
        raise HTTPException(status_code=403, detail="Not allowed")

    with engine.begin() as db:
        db.execute(text("UPDATE users SET last_login_at = CURRENT_TIMESTAMP WHERE id = :id"), {"id": user["id"]})

    return {
        "token": create_token(user["id"], user["role"]),
        "user": {"name": user["display_name"], "email": user["email"], "role": user["role"]},
    }