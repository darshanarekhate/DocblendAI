"""PaddleOCR Experience Center HTTP API (prefix /api): docs/experience_center_contract.md §3.

Supports Modules 1-3 for the Experience Center UI (/studio): upload + validation, OCR / document
parsing jobs with progress, run history, human review edits, and confidence calibration.
Heavy work runs on the job manager's worker thread (app/modules/ocr_jobs.py); `wait=true`
blocks the request until the job is done and returns the full result.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import APIRouter, Body, File, Form, HTTPException, Query, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse

from app.config import BASE_DIR, settings
from app.modules import paddleocr_service as svc
from app.modules import preprocess as preprocess_steps
from app.modules.ocr_jobs import EditError, Job, JobBusyError, job_manager, page_image_path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["experience-center"])

SAMPLE_DATASET = BASE_DIR / "evaluation" / "calibration_sample"
ZIP_MAX_FILES = 5000
CHUNK = 1024 * 1024


def _max_upload_bytes() -> int:
    return settings.paddle_max_upload_mb * 1024 * 1024


def _zip_max_uncompressed() -> int:
    return 4 * _max_upload_bytes()


# -- form validation ---------------------------------------------------------------------------


def _bad(message: str) -> HTTPException:
    return HTTPException(status.HTTP_400_BAD_REQUEST, message)


def _language(language: str) -> str:
    language = (language or "en").strip()
    if language not in svc.LANGUAGE_CODES:
        raise _bad(f"Unsupported language {language!r}; use one of: {', '.join(sorted(svc.LANGUAGE_CODES))}")
    return language


def _review_threshold(value: str | None) -> float:
    if value is None or not str(value).strip():
        return settings.review_threshold
    try:
        threshold = float(value)
    except ValueError:
        raise _bad("review_threshold must be a number between 0 and 1") from None
    if not 0.0 <= threshold <= 1.0:
        raise _bad("review_threshold must be between 0 and 1")
    return threshold


def _preprocess(value: str | None) -> dict[str, bool]:
    """Contract §5 options, validated with app.modules.preprocess (accepted, not applied yet)."""
    if value is None or not value.strip():
        return preprocess_steps.parse_options(None)
    try:
        return preprocess_steps.parse_options(json.loads(value))
    except json.JSONDecodeError:
        raise _bad("preprocess must be a JSON object, e.g. {\"deskew\": true}") from None
    except preprocess_steps.PreprocessOptionsError as exc:
        raise _bad(str(exc)) from None


def _flag(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


# -- uploads -----------------------------------------------------------------------------------


async def _save_upload(file: UploadFile, suffix: str) -> Path:
    """Stream the upload into a temp file, enforcing settings.paddle_max_upload_mb (413)."""
    fd, name = tempfile.mkstemp(prefix="docblend-ec-", suffix=suffix)
    path = Path(name)
    total = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while chunk := await file.read(CHUNK):
                total += len(chunk)
                if total > _max_upload_bytes():
                    raise HTTPException(
                        status.HTTP_413_CONTENT_TOO_LARGE,
                        f"File is larger than {settings.paddle_max_upload_mb} MB",
                    )
                out.write(chunk)
        if total == 0:
            raise _bad("The uploaded file is empty")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _upload_kind(filename: str | None) -> tuple[str, str]:
    suffix = Path(filename or "").suffix.lower()
    kind = svc.kind_of_suffix(suffix)
    if kind is None:
        allowed = sorted(svc.IMAGE_SUFFIXES | svc.PDF_SUFFIXES | svc.OFFICE_SUFFIXES)
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Unsupported file type; upload one of: {', '.join(allowed)}")
    return suffix, kind


def _require_available(pipeline: str) -> None:
    state = paddle_status(pipeline)
    if not state["available"]:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, state["message"])


def paddle_status(pipeline: str) -> dict[str, Any]:
    return svc.paddleocr_service.pipeline_status(pipeline)


async def _finish(job: Job, wait: bool) -> JSONResponse:
    if not wait:
        return JSONResponse({"id": job.id, "status": job.status}, status_code=status.HTTP_202_ACCEPTED)
    await run_in_threadpool(job_manager.wait, job)
    if job.unavailable:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, job.error or "Pipeline unavailable")
    return JSONResponse(job_manager.get_result(job.id) or {"id": job.id, "status": job.status})


async def _submit_document(
    file: UploadFile, pipeline: str, language: str, wait: str | bool, review_threshold: str | None,
    preprocess: str | None,
) -> JSONResponse:
    suffix, kind = _upload_kind(file.filename)
    lang = _language(language)
    threshold = _review_threshold(review_threshold)
    _preprocess(preprocess)
    if kind == "office":
        pipeline = "office"
    _require_available(pipeline)

    path = await _save_upload(file, suffix)
    submitted = False
    try:
        try:
            await run_in_threadpool(svc.validate_input, path, kind)
        except svc.UnreadableInputError as exc:
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"{file.filename}: {exc}") from exc
        filename = Path(file.filename or f"upload{suffix}").name[:255]
        job = await run_in_threadpool(job_manager.submit_run, path, filename, pipeline, lang, threshold)
        submitted = True  # the job now owns (and deletes) the temp file
    finally:
        if not submitted:
            path.unlink(missing_ok=True)
    return await _finish(job, _flag(wait))


# -- routes ------------------------------------------------------------------------------------


@router.get("/health")
def experience_health() -> dict[str, Any]:
    """Engine versions, pipeline availability, calibration state, limits (nothing is loaded)."""
    calibration: dict[str, Any] = {"status": "uncalibrated", "method": None}
    try:
        from app.calibration import load_active

        active = load_active()
        if active is not None:
            calibration = {"status": "calibrated", "method": getattr(active, "method", None)}
    except Exception:
        logger.exception("could not read the active calibrator")
    return {
        "status": "ok",
        "engine": svc.paddleocr_service.engine_info(),
        "pipelines": {name: paddle_status(name) for name in svc.PIPELINES},
        "calibration": calibration,
        "review_threshold": settings.review_threshold,
        "max_upload_mb": settings.paddle_max_upload_mb,
        "languages": svc.LANGUAGES,
    }


@router.post("/ocr")
async def run_ocr(
    file: UploadFile = File(...),
    language: str = Form("en"),
    wait: str = Form("false"),
    review_threshold: str | None = Form(None),
    preprocess: str | None = Form(None),
) -> JSONResponse:
    """Text OCR (PP-OCR) of an image or PDF; Office files are converted to Markdown instead."""
    return await _submit_document(file, "ocr", language, wait, review_threshold, preprocess)


@router.post("/parse")
async def run_parse(
    file: UploadFile = File(...),
    pipeline: str = Form("structure"),
    language: str = Form("en"),
    wait: str = Form("false"),
    review_threshold: str | None = Form(None),
    preprocess: str | None = Form(None),
) -> JSONResponse:
    """Document parsing: layout blocks, tables, Markdown (PP-StructureV3, or PaddleOCR-VL)."""
    pipeline = (pipeline or "structure").strip().lower()
    if pipeline not in ("structure", "vl"):
        raise _bad("pipeline must be 'structure' or 'vl'")
    return await _submit_document(file, pipeline, language, wait, review_threshold, preprocess)


@router.get("/results")
def list_results(q: str | None = Query(None, max_length=200), limit: int = Query(50, ge=1, le=500)) -> list[dict[str, Any]]:
    """History, newest first; q searches file names and recognised text."""
    return job_manager.history(q.strip() if q else None, limit)


@router.get("/results/{run_id}")
def get_result(run_id: str) -> dict[str, Any]:
    result = job_manager.get_result(run_id)
    if result is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such result")
    return result


@router.delete("/results/{run_id}")
def delete_result(run_id: str) -> dict[str, Any]:
    try:
        deleted = job_manager.delete(run_id)
    except JobBusyError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    if not deleted:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such result")
    return {"id": run_id, "deleted": True}


@router.patch("/results/{run_id}/lines")
def edit_lines(run_id: str, edits: list[dict[str, Any]] = Body(...)) -> dict[str, Any]:
    """Human review: [{"page": 0, "line": 3, "text": "fixed"}] -> the updated result."""
    try:
        result = job_manager.edit_lines(run_id, edits)
    except JobBusyError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except EditError as exc:
        raise _bad(str(exc)) from exc
    if result is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such result")
    return result


@router.get("/results/{run_id}/pages/{index}/image")
def page_image(run_id: str, index: int) -> FileResponse:
    if not run_id.isalnum() or index < 0:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such page image")
    path = page_image_path(run_id, index)
    if not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such page image")
    return FileResponse(path, media_type="image/png")


# -- calibration -------------------------------------------------------------------------------


def _safe_extract(zip_path: Path, dest: Path) -> None:
    """Extract a dataset zip, refusing absolute/parent paths, links, and zip bombs (400/413)."""
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Not a zip file: {exc}") from exc
    with zf:
        infos = zf.infolist()
        if len(infos) > ZIP_MAX_FILES:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"The zip holds more than {ZIP_MAX_FILES} files")
        if sum(info.file_size for info in infos) > _zip_max_uncompressed():
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE,
                f"The zip expands to more than {_zip_max_uncompressed() // (1024 * 1024)} MB",
            )
        root = dest.resolve()
        for info in infos:
            name = info.filename.replace("\\", "/")
            parts = PurePosixPath(name).parts
            if (
                name.startswith("/") or ".." in parts or (parts and ":" in parts[0])
                or (info.external_attr >> 16) & 0o170000 == 0o120000  # symlink
            ):
                raise _bad(f"Unsafe path in zip: {info.filename!r}")
            target = (root / name).resolve()
            if root != target and root not in target.parents:
                raise _bad(f"Unsafe path in zip: {info.filename!r}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            with zf.open(info) as src, target.open("wb") as out:
                while chunk := src.read(CHUNK):
                    written += len(chunk)
                    if written > info.file_size:  # header lied about the size
                        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "Corrupt zip (size mismatch)")
                    out.write(chunk)


def _find_dataset(root: Path) -> Path:
    """The folder holding labels.csv: the zip root or its single top-level folder."""
    candidates = sorted(root.rglob("labels.csv"), key=lambda p: len(p.parts))
    if not candidates or len(candidates[0].relative_to(root).parts) > 2:
        raise _bad("The zip must hold labels.csv (columns image,text) next to the images")
    with candidates[0].open(newline="", encoding="utf-8-sig") as fh:
        header = next(csv.reader(fh), [])
    if not {"image", "text"} <= {h.strip().lower() for h in header}:
        raise _bad("labels.csv needs the columns image,text")
    return candidates[0].parent


@router.post("/calibrate")
async def calibrate(
    file: UploadFile | None = File(None),
    use_sample: str = Form("false"),
    match: str = Form("cer"),
    cer_threshold: str | None = Form(None),
    wait: str = Form("false"),
) -> JSONResponse:
    """Fit a confidence calibrator on a labelled dataset (zip upload or the bundled sample)."""
    match = (match or "cer").strip().lower()
    if match not in ("cer", "exact"):
        raise _bad("match must be 'cer' or 'exact'")
    threshold = settings.calibration_cer_threshold
    if cer_threshold is not None and str(cer_threshold).strip():
        try:
            threshold = float(cer_threshold)
        except ValueError:
            raise _bad("cer_threshold must be a number between 0 and 1") from None
        if not 0.0 <= threshold <= 1.0:
            raise _bad("cer_threshold must be between 0 and 1")
    _require_available("ocr")

    has_file = file is not None and bool(file.filename)
    if _flag(use_sample):
        if not (SAMPLE_DATASET / "labels.csv").is_file():
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "The sample dataset is missing; build it with: venv/Scripts/python -m app.calibration.make_sample",
            )
        job = job_manager.submit_calibration(SAMPLE_DATASET, match, threshold, label="calibration_sample")
        return await _finish(job, _flag(wait))
    if not has_file:
        raise _bad("Upload a .zip (labels.csv + images) as 'file', or set use_sample=true")
    if Path(file.filename).suffix.lower() != ".zip":
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Upload the dataset as a .zip file")

    zip_path = await _save_upload(file, ".zip")
    extract_dir = Path(tempfile.mkdtemp(prefix="docblend-calib-"))
    submitted = False
    try:
        await run_in_threadpool(_safe_extract, zip_path, extract_dir)
        dataset = _find_dataset(extract_dir)
        job = job_manager.submit_calibration(
            dataset, match, threshold, cleanup=extract_dir, label=Path(file.filename).name[:255]
        )
        submitted = True
    finally:
        zip_path.unlink(missing_ok=True)
        if not submitted:
            shutil.rmtree(extract_dir, ignore_errors=True)
    return await _finish(job, _flag(wait))


@router.get("/calibration")
def calibration_report() -> dict[str, Any]:
    path = settings.paddle_calibration_dir / "report.json"
    if not path.is_file():
        return {"status": "uncalibrated"}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("unreadable calibration report %s", path)
        return {"status": "uncalibrated"}
    report.setdefault("status", "calibrated")
    return report


@router.get("/calibration/diagram.png")
def calibration_diagram() -> FileResponse:
    path = settings.paddle_calibration_dir / "reliability_diagram.png"
    if not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No reliability diagram yet: run a calibration first")
    return FileResponse(path, media_type="image/png")
