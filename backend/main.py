# Starts the AlbaniaSAT API. Run from the backend folder: uvicorn main:app --reload
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from admin_articles import router as admin_articles_router
from articles import router as articles_router
from auth import router as auth_router

app = FastAPI(title="AlbaniaSAT API")

# Allow the frontend (Vite dev server) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(articles_router)
app.include_router(admin_articles_router)


@app.get("/")
def health():
    return {"status": "ok"}