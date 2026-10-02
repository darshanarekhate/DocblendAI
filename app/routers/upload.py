"""Module 1 — Document Upload.

Responsibility: accept an upload (PDF, image, Word, PowerPoint, Excel, or text; see
file_types.py), save it under settings.upload_dir, and run it as one ingest job
(ocr_jobs.py): detection -> extraction (exact text / PaddleOCR / PP-StructureV3 / TrOCR,
document_extract.py) -> chunking, calibration, content types -> embedding -> ChromaDB
(qa_index.py), plus a Document row linked to the document's Experience Center run
(document_runs.py). Also lists (incl. the shared library), serves (for the document
viewer), and deletes documents from both pages.

Uses from schemas.py: Document, FormatType.
"""

import logging
import mimetypes
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.db.database import get_db
from app.db.models import DocumentORM
from app.models.schemas import Document, FormatType
from app.modules import document_runs, file_types
from app.modules.ocr_jobs import job_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["upload"])


# Plain `def` (not async): it may wait for extraction (blocking), so FastAPI runs it in its threadpool.
@router.post("/upload", response_model=Document, status_code=status.HTTP_201_CREATED)
def upload_document(
    file: UploadFile = File(...),
    format_hint: FormatType | None = Form(
        None,
        description=(
            "Skip automatic detection and force this format (e.g. if detection guesses wrong). "
            "Ignored for Word/PowerPoint/Excel/text files (always typed), and 'typed' is ignored for images."
        ),
    ),
    pipeline: str | None = Form(
        None, description="Scanned pages: 'structure' (layout, tables; slower) or 'ocr' (text lines); default INGEST_PIPELINE"
    ),
    background: bool = Form(
        False, description="Return 202 {doc_id, run_id} at once and process in the background (poll /documents/library)"
    ),
    db: Session = Depends(get_db),
    response: Response = None,
):
    """Ingest a file: 201 = new document; 200 = the same file was already uploaded (that Document is returned).

    The file is read once (document_extract.py: exact text for typed files, PaddleOCR /
    PP-StructureV3 for scans, TrOCR for handwriting), shown in the Experience Center as a run,
    and indexed for questions. Without background the request waits until questions can be
    asked; Gemini's automatic correction of low-confidence lines continues afterwards.
    """
    filename = file.filename or ""
    if file_types.kind_of(filename) is None:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported file type. Upload one of: {file_types.supported_extensions()}",
        )
    pipeline = (pipeline or settings.ingest_pipeline or "structure").strip().lower()
    if pipeline not in ("ocr", "structure"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "pipeline must be 'structure' or 'ocr'")

    doc_id, path = document_runs.new_upload_path(filename)
    with path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    existing = document_runs.find_duplicate(db, path, format_hint)
    if existing is not None:
        # A second copy would only crowd retrieval with identical passages.
        path.unlink(missing_ok=True)
        logger.info("%s is identical to %s; not stored again", file.filename, existing.doc_id)
        if background:
            return JSONResponse({"doc_id": existing.doc_id, "run_id": document_runs.run_for_doc(db, existing.doc_id),
                                 "status": "exists"}, status_code=status.HTTP_200_OK)
        response.status_code = status.HTTP_200_OK
        return Document.model_validate(existing)

    job = job_manager.submit_run(
        path, Path(filename).name[:255] or path.name, pipeline, "en", settings.review_threshold, None,
        format_hint.value if format_hint else None, doc_id=doc_id, keep_failed=background,
    )
    if background:
        return JSONResponse({"doc_id": doc_id, "run_id": job.id, "status": job.status}, status_code=status.HTTP_202_ACCEPTED)
    job_manager.wait_until_ready(job)
    if job.status != "done":
        detail = job.message if job.http_status == 422 and job.message.startswith("File is") else job.error
        raise HTTPException(job.http_status or status.HTTP_500_INTERNAL_SERVER_ERROR, detail or "Upload failed")
    db.expire_all()  # the job wrote the Document row in its own session
    row = db.get(DocumentORM, doc_id)
    logger.info("Uploaded %s as %s", file.filename, doc_id)
    return Document.model_validate(row)


@router.get("/documents/library")
def document_library(q: str | None = Query(None, max_length=200)) -> list[dict]:
    """Documents of both pages (QA and Experience Center) with processing, QA and refinement state."""
    return document_runs.library(q.strip() if q else None)


@router.post("/documents/{doc_id}/experience", status_code=status.HTTP_202_ACCEPTED)
def open_in_experience_center(
    doc_id: str, pipeline: str | None = Query(None, description="structure | ocr"), db: Session = Depends(get_db)
) -> dict:
    """A document uploaded before the two pages were shared: read it into an Experience Center run
    (and rebuild its QA chunks from that reading). 409 if it already has a run."""
    row = db.get(DocumentORM, doc_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    run_id = document_runs.run_for_doc(db, doc_id)
    if run_id is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Already in the Experience Center as run {run_id}")
    path = Path(row.file_path)
    if not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "The uploaded file is missing")
    pipeline = (pipeline or settings.ingest_pipeline).strip().lower()
    if pipeline not in ("ocr", "structure"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "pipeline must be 'structure' or 'ocr'")
    job = job_manager.submit_run(
        path, document_runs.display_name(doc_id, row.file_path), pipeline, "en", settings.review_threshold,
        None, row.format_type.value, doc_id=doc_id, existing_doc=True,
    )
    return {"doc_id": doc_id, "run_id": job.id, "status": job.status}


@router.get("/documents", response_model=list[Document])
def list_documents(db: Session = Depends(get_db)) -> list[Document]:
    return [Document.model_validate(row) for row in db.query(DocumentORM).all()]


@router.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(doc_id: str, db: Session = Depends(get_db)) -> None:
    """Remove a document from both pages: its chunks (ChromaDB), its record (SQLite), the uploaded
    file, and its Experience Center run (result, versions, page images).

    Past answers stay in the history; their sources simply no longer list this document.
    """
    if db.get(DocumentORM, doc_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    run_id = document_runs.run_for_doc(db, doc_id)
    if run_id is not None and job_manager.active_refine(run_id) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "This document is being refined; delete it when that has finished.")
    db.close()  # release this session's read before another session deletes the rows (SQLite)
    document_runs.delete_document(doc_id)


# Types mimetypes may not know on every system (Windows reads them from the registry).
MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


@router.get("/documents/{doc_id}/file", response_class=FileResponse)
def get_document_file(doc_id: str, db: Session = Depends(get_db)) -> FileResponse:
    """The original uploaded file, shown inline (for the document viewer)."""
    row = db.get(DocumentORM, doc_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    path = Path(row.file_path).resolve()
    # The path comes only from the Document row; still refuse anything outside the upload folder.
    if not path.is_relative_to(settings.upload_dir.resolve()) or not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "The uploaded file is missing")
    media_type = MEDIA_TYPES.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    name = path.name[len(doc_id) + 1 :] if path.name.startswith(f"{doc_id}_") else path.name
    return FileResponse(
        path,
        media_type=media_type,
        filename=name,
        content_disposition_type="inline",
        headers={"X-Content-Type-Options": "nosniff"},
    )
