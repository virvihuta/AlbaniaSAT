# Einstellungen für Pytest (Regis)
from dotenv import load_dotenv

# Lädt DATABASE_URL usw. aus der .env, bevor die Tests starten
load_dotenv()

# test_auth.py ist ein normales Script von Suisa (python test_auth.py),
# kein Pytest-Test. Pytest soll es nicht starten.
collect_ignore = ["test_auth.py"]