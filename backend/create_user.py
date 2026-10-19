# Creates an admin or redakteur (editor) account with a hashed password.
# Usage: python create_user.py
import getpass
import os

from sqlalchemy import create_engine, text

from security import hash_password

engine = create_engine(os.environ["DATABASE_URL"])

name = input("Name: ")
email = input("Email: ")
role = input("Role (admin/redakteur): ").strip()
password = getpass.getpass("Password (min. 8 characters): ")

if role not in ("admin", "redakteur"):
    raise SystemExit("Role must be admin or redakteur")
if len(password) < 8:
    raise SystemExit("Password is too short")

with engine.begin() as db:
    db.execute(
        text("INSERT INTO users (display_name, email, password_hash, role) VALUES (:n, :e, :p, :r)"),
        {"n": name, "e": email, "p": hash_password(password), "r": role},
    )
print(f"Created {role} account for {email}")