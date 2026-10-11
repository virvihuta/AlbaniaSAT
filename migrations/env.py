# Diese Datei sagt Alembic, wie es sich mit der Datenbank verbindet.

import os

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import create_engine, pool

# 1. Die Datei .env lesen. Dort steht unsere DATABASE_URL mit dem Passwort.
load_dotenv()

# 2. Die Adresse der Datenbank aus .env holen.
url = os.environ.get("DATABASE_URL")
if not url:
    raise RuntimeError("DATABASE_URL fehlt. Steht sie in der Datei .env?")
# 3. Mit der Datenbank verbinden.
#    NullPool heißt: Die Verbindung wird danach gleich wieder geschlossen.
engine = create_engine(url, poolclass=pool.NullPool)

# 4. Die Migrationen ausführen.
#    target_metadata ist None, weil wir die Migrationen selbst
#    als SQL schreiben und kein Python-Modell haben.
with engine.connect() as connection:
    context.configure(connection=connection, target_metadata=None)
    with context.begin_transaction():
        context.run_migrations()