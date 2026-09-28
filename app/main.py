"""FastAPI entry point: app instance, router registration, health check.

Single monolith (see CLAUDE.md): all 7 modules run in this one process and
are wired together through the routers.

Run with:  venv/Scripts/python -m uvicorn app.main:app --reload
"""

import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import settings
from app.db.database import init_db
from app.routers import preview, query, upload

STATIC_DIR = Path(__file__).parent / "static"

# Uvicorn only configures its own loggers; this makes app.* INFO logs visible too.
logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def _warm_up_models() -> None:
    """Load (and run once) the models a scanned or handwritten upload needs.

    Loading TrOCR and docTR takes ~15 s and their first run is slower than later
    ones; doing both at startup, off the request path, keeps that out of uploads.
    A model that cannot load is simply loaded (and reported) on first use instead.
    """
    from PIL import Image

    from app.modules import htr_extractor, line_segmentation, ocr_extractor

    blank_line, blank_page = Image.new("L", (384, 64), 255), Image.new("L", (850, 1100), 255)
    steps = [("HTR model", lambda: htr_extractor.recognize_lines([blank_line]))]
    if settings.htr_segmenter == "doctr":
        steps.append(("line detector", lambda: line_segmentation.detect_lines(blank_page)))
    if ocr_extractor._engine() == "doctr":
        steps.append(("docTR OCR", lambda: ocr_extractor.ocr_image(blank_page)))
    for name, step in steps:
        try:
            step()
        except Exception as e:  # noqa: BLE001 - a warm-up failure must never stop the server
            logger.warning("Could not preload %s: %s", name, e)
    logger.info("OCR/HTR models loaded")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    init_db()
    if settings.preload_models:
        threading.Thread(target=_warm_up_models, name="model-warm-up", daemon=True).start()
    yield


app = FastAPI(
    title="DocBlendAI",
    description="Confidence-aware multi-format document QA assistant.",
    version="0.1.0",
    lifespan=lifespan,
)

# The UI is normally served at "/", but people also open app/static/index.html directly
# (double-click: origin "null") or through an editor's live preview on another localhost
# port. Let those local pages call the API; nothing outside this machine is allowed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["null"],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(query.router)
app.include_router(preview.router)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    """Demo UI: upload documents, ask questions, see reliability labels and sources."""
    return FileResponse(STATIC_DIR / "index.html")
