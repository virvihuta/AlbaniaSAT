# Password hashing and JWT tokens for the admin login.
import os
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

JWT_SECRET = os.environ["JWT_SECRET"]  # long random text, only in .env
TOKEN_HOURS = 8


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def create_token(user_id: int, role: str) -> str:
    expires = datetime.now(timezone.utc) + timedelta(hours=TOKEN_HOURS)
    return jwt.encode({"sub": str(user_id), "role": role, "exp": expires}, JWT_SECRET, algorithm="HS256")


def require_role(*roles: str):
    """Use on protected endpoints, e.g. Depends(require_role("admin", "redakteur"))."""

    def check(credentials=Depends(HTTPBearer())):
        try:
            payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=["HS256"])
        except jwt.PyJWTError:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        if payload["role"] not in roles:
            raise HTTPException(status_code=403, detail="Not allowed")
        return payload

    return check