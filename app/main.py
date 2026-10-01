"""FastAPI entry point: app instance, router registration, health check.

Single monolith (see CLAUDE.md): all 7 modules run in this one process and
are wired together through the routers.

Run with:  venv/Scripts/python -m uvicorn app.main:app --reload
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.db.database import init_db
from app.logging_config import configure_logging
from app.routers import paddleocr, query, studio_tools, upload

STATIC_DIR = Path(__file__).parent / "static"

# Uvicorn only configures its own loggers; this makes app.* INFO logs visible too
# (LOG_FORMAT=json switches to one JSON object per line).
configure_logging(settings.log_format)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    init_db()
    yield


app = FastAPI(
    title="DocBlendAI",
    description="Confidence-aware multi-format document QA assistant.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(upload.router)
app.include_router(query.router)
app.include_router(paddleocr.router)
app.include_router(studio_tools.router)
# Experience Center assets (studio.css/js, sample images) and the QA page's shared files.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    """Demo UI: upload documents, ask questions, see reliability labels and sources."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/studio", include_in_schema=False)
def studio() -> FileResponse:
    """Experience Center: OCR / document parsing with calibrated confidence, side by side."""
    return FileResponse(STATIC_DIR / "studio.html")
