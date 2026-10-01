"""PaddleOCR experience-center API routes."""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.config import settings
from app.modules.paddleocr_service import PaddleOCRUnavailableError, paddleocr_service

router = APIRouter(prefix="/api", tags=["paddleocr"])

_ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".pdf"}
_MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def _safe_suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        allowed = ", ".join(sorted(_ALLOWED_SUFFIXES))
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Upload one of: {allowed}"
        )
    return suffix


async def _save_upload(file: UploadFile) -> Path:
    suffix = _safe_suffix(file.filename)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    path = settings.upload_dir / f"paddle-{uuid.uuid4().hex}{suffix}"
    total = 0
    try:
        with path.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > _MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status.HTTP_413_CONTENT_TOO_LARGE,
                        "File exceeds the 25 MB limit",
                    )
                output.write(chunk)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def _page_response(page: object) -> dict[str, object]:
    lines = []
    for line in page.lines:
        calibrated = line.raw_confidence
        lines.append(
            {
                "text": line.text,
                "polygon": line.polygon,
                "raw_confidence": line.raw_confidence,
                "calibrated_confidence": calibrated,
                "calibration_status": "uncalibrated",
                "needs_review": calibrated < settings.review_threshold,
            }
        )
    return {"page_index": page.page_index, "source": page.source, "lines": lines}


@router.get("/health")
def paddleocr_health() -> dict[str, object]:
    return {
        "status": "ok",
        "engine": "paddleocr",
        "device": settings.paddle_device,
        "vl": paddleocr_service.vl_status(),
    }


@router.post("/ocr")
async def run_ocr(
    file: UploadFile = File(...),
    language: str = Form("en"),
) -> dict[str, object]:
    path = await _save_upload(file)
    started = time.perf_counter()
    try:
        pages = paddleocr_service.ocr_file(path, language)
        return {
            "pipeline": "ocr",
            "language": language,
            "pages": [_page_response(page) for page in pages],
            "processing_time_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    except PaddleOCRUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    finally:
        path.unlink(missing_ok=True)


@router.post("/parse")
async def parse_document(
    file: UploadFile = File(...),
    language: str = Form("en"),
) -> dict[str, object]:
    path = await _save_upload(file)
    started = time.perf_counter()
    try:
        result = paddleocr_service.parse_file(path, language)
        result.update(
            {
                "pipeline": "structure",
                "language": language,
                "processing_time_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        )
        return result
    except PaddleOCRUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    finally:
        path.unlink(missing_ok=True)
