# Database connection, once for the whole backend (Regis)
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine

# Load the .env (DATABASE_URL, JWT_SECRET, INGEST_API_KEY).
# Without a .env (e.g. on Render) nothing happens, then the values come from the server.
load_dotenv()

# One shared engine for all files. pool_pre_ping checks each connection before using it
# and reconnects if it is dead (e.g. after the Wi-Fi reconnects or the server woke up).
engine = create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)