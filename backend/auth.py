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
            text("SELECT id, name, email, password_hash, role FROM users WHERE email = :email"),
            {"email": data.email},
        ).mappings().first()

    # Same message for wrong email and wrong password, so nobody can guess emails
    if not user or not verify_password(data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Wrong email or password")

    return {
        "token": create_token(user["id"], user["role"]),
        "user": {"name": user["name"], "email": user["email"], "role": user["role"]},
    }
