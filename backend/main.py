# Starts the AlbaniaSAT API. Run from the backend folder: uvicorn main:app --reload
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from admin_articles import router as admin_articles_router
from articles import router as articles_router
from auth import router as auth_router
from classification import router as classification_router
from contact import router as contact_router
from faqs import router as faqs_router
from ingest import router as ingest_router
from patches import router as patches_router

app = FastAPI(title="AlbaniaSAT API")

# Websites that may call the API. FRONTEND_URL can hold several addresses, separated by commas,
# e.g. "https://albaniasat.onrender.com,http://localhost:5173".
# Without FRONTEND_URL only the local Vite dev server is allowed.
FRONTEND_URLS = [url.strip() for url in os.environ.get("FRONTEND_URL", "http://localhost:5173").split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_URLS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(articles_router)
app.include_router(admin_articles_router)
app.include_router(classification_router)
app.include_router(ingest_router)
app.include_router(contact_router)
app.include_router(patches_router)
app.include_router(faqs_router)


@app.get("/")
def health():
    return {"status": "ok"}