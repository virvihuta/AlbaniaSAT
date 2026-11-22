# Settings for Pytest (Regis)
from dotenv import load_dotenv

# Loads DATABASE_URL etc. from the .env before the tests start
load_dotenv()

# test_auth.py is a normal script from Suisa (python test_auth.py),
# not a Pytest test. Pytest should not run it.
collect_ignore = ["test_auth.py"]