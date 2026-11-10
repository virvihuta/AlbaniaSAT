# Verbindung zur Datenbank, einmal für das ganze Backend (Regis)
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine

# Die .env laden (DATABASE_URL, JWT_SECRET, INGEST_API_KEY).
# Gibt es keine .env (z. B. später auf Render), passiert nichts, dann kommen die Werte vom Server.
load_dotenv()

# Eine Verbindung für alle Dateien
engine = create_engine(os.environ["DATABASE_URL"])