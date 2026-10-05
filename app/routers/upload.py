"""Module 1 — Document Upload.

Responsibility: accept an upload (PDF, image, Word, PowerPoint, or text; see
file_types.py), save it under settings.upload_dir, run it through
detection -> extraction (parse/OCR/HTR) -> chunking ->
calibration -> content-type labeling -> embedding -> ChromaDB, and record a
Document row.

Uses from schemas.py: Document, FormatType.

Two ways in:
- POST /upload reads the file before replying (201 + the Document).
- POST /upload/background (the web page) saves the file, replies within a second (202),
  and reads it in the background; GET /uploads/pending lists files still being read
  (or that failed), so the page can show "Reading..." and pick the document up when done.
  Handwritten pages take ~25 s each on a laptop CPU; this keeps that off the upload.
"""

import hashlib
import logging
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

from fastapi import (
    APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, Response, UploadFile, status,
)
from pydantic import BaseModel
from pdfplumber.utils.exceptions import PdfminerException
from sqlalchemy.orm import Session

from app.config import settings
from app.db.database import get_db
from app.db.models import DocumentORM
from app.models.schemas import Document, FormatType
from app.modules import (
    chunker,
    confidence_capture,
    content_type,
    embedder,
    file_types,
    format_detection,
    htr_correction,
    htr_extractor,
    ocr_extractor,
    vector_store,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["upload"])


FORMAT_HINT_HELP = (
    "Skip automatic detection and force this format (e.g. if detection guesses wrong). "
    "Ignored for Word/PowerPoint/text files (always typed), and 'typed' is ignored for images."
)

# Background uploads still being read (or failed): doc_id -> PendingUpload. In memory only:
# a file whose reading was cut short by a server restart is simply not in the list afterwards.
_pending: dict[str, "PendingUpload"] = {}
_pending_lock = threading.Lock()
FAILED_KEPT_SECONDS = 600  # a failed upload stays listed this long, so the page can report it


class PendingUpload(BaseModel):
    doc_id: str
    name: str  # the original file name
    status: str  # "reading", "failed", or "duplicate" (already uploaded: doc_id is that document)
    detail: str | None = None  # why it failed
    started: float = 0.0  # time.time() when the upload arrived


def _save(file: UploadFile) -> tuple[str, Path]:
    filename = file.filename or ""
    if file_types.kind_of(filename) is None:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported file type. Upload one of: {file_types.supported_extensions()}",
        )
    doc_id = uuid.uuid4().hex
    # Keep the original name in the stored path so the UI can show it (the Document entity has no name field).
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).stem)[:80] or "document"
    path = settings.upload_dir / f"{doc_id}_{safe_name}{Path(filename).suffix.lower()}"
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    return doc_id, path


# Plain `def` (not async): PDF parsing, OCR, and HTR are blocking, so FastAPI runs this in its threadpool.
@router.post("/upload", response_model=Document, status_code=status.HTTP_201_CREATED)
def upload_document(
    file: UploadFile = File(...),
    format_hint: FormatType | None = Form(None, description=FORMAT_HINT_HELP),
    db: Session = Depends(get_db),
    response: Response = None,
) -> Document:
    """Ingest a file. 201 = new document; 200 = the same file was already uploaded (that Document is returned)."""
    doc_id, path = _save(file)

    existing = _same_file_already_uploaded(path, format_hint, db)
    if existing is not None:
        # A second copy would only crowd retrieval with identical passages.
        path.unlink(missing_ok=True)
        logger.info("%s is identical to %s; not stored again", file.filename, existing.doc_id)
        response.status_code = status.HTTP_200_OK
        return existing

    try:
        document = _ingest(doc_id, path, format_hint, db)
    except Exception:
        # Any failure (rejection, service error, or bug) leaves no stray upload behind.
        path.unlink(missing_ok=True)
        raise
    logger.info("Uploaded %s as %s", file.filename, doc_id)
    return document


@router.post("/upload/background", response_model=PendingUpload, status_code=status.HTTP_202_ACCEPTED)
def upload_in_background(
    request: Request,
    background: BackgroundTasks,
    file: UploadFile = File(...),
    format_hint: FormatType | None = Form(None, description=FORMAT_HINT_HELP),
    db: Session = Depends(get_db),
    response: Response = None,
) -> PendingUpload:
    """Save the file and reply at once (202); it is read in the background (see GET /uploads/pending).

    200 with status "duplicate" = the same file was already uploaded; doc_id is that document.
    """
    name = file.filename or ""
    doc_id, path = _save(file)
    existing = _same_file_already_uploaded(path, format_hint, db)
    if existing is not None:
        path.unlink(missing_ok=True)
        response.status_code = status.HTTP_200_OK
        return PendingUpload(doc_id=existing.doc_id, name=name, status="duplicate", started=time.time())

    job = PendingUpload(doc_id=doc_id, name=name, status="reading", started=time.time())
    with _pending_lock:
        _pending[doc_id] = job
    # The request's own session closes with the request; the background read opens its own
    # (through the same dependency, so a test's database override applies too).
    session_factory = request.app.dependency_overrides.get(get_db, get_db)
    background.add_task(_ingest_in_background, doc_id, path, format_hint, session_factory, name)
    logger.info("Saved %s as %s; reading it in the background", name, doc_id)
    return job


def _ingest_in_background(doc_id: str, path: Path, format_hint: FormatType | None, session_factory, name: str) -> None:
    sessions = session_factory()
    db = next(sessions)
    try:
        _ingest(doc_id, path, format_hint, db)
        with _pending_lock:
            _pending.pop(doc_id, None)
        logger.info("Uploaded %s as %s", name, doc_id)
    except Exception as e:  # noqa: BLE001 - reported to the page, never raised into the server
        path.unlink(missing_ok=True)
        detail = e.detail if isinstance(e, HTTPException) else "Reading the file failed; try again."
        if not isinstance(e, HTTPException):
            logger.exception("Background upload of %s failed", name)
        with _pending_lock:
            _pending[doc_id] = PendingUpload(doc_id=doc_id, name=name, status="failed", detail=detail, started=time.time())
    finally:
        sessions.close()


@router.get("/uploads/pending", response_model=list[PendingUpload])
def pending_uploads() -> list[PendingUpload]:
    """Background uploads still being read, and ones that failed in the last few minutes."""
    now = time.time()
    with _pending_lock:
        for doc_id, job in list(_pending.items()):
            if job.status == "failed" and now - job.started > FAILED_KEPT_SECONDS:
                del _pending[doc_id]
        return sorted(_pending.values(), key=lambda job: job.started)


@router.delete("/uploads/pending/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def dismiss_failed_upload(doc_id: str) -> None:
    """Forget a failed background upload once the page has shown the error."""
    with _pending_lock:
        job = _pending.get(doc_id)
        if job is not None and job.status == "failed":
            del _pending[doc_id]


@router.get("/documents", response_model=list[Document])
def list_documents(db: Session = Depends(get_db)) -> list[Document]:
    return [Document.model_validate(row) for row in db.query(DocumentORM).all()]


@router.delete("/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(doc_id: str, db: Session = Depends(get_db)) -> None:
    """Remove a document: its chunks (ChromaDB), its record (SQLite), and the uploaded file.

    Past answers stay in the history; their sources simply no longer list this document.
    """
    row = db.get(DocumentORM, doc_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    vector_store.delete_document(doc_id)
    file_path = Path(row.file_path)
    db.delete(row)
    db.commit()
    file_path.unlink(missing_ok=True)
    logger.info("Deleted %s (%s)", doc_id, file_path.name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _same_file_already_uploaded(path: Path, format_hint: FormatType | None, db: Session) -> Document | None:
    """An earlier Document with byte-identical content (and a compatible format), if any.

    A format_hint that differs from the stored format is a request to re-read the
    file another way, so it is not treated as a duplicate.
    """
    size, digest = path.stat().st_size, None
    for row in db.query(DocumentORM).all():
        other = Path(row.file_path)
        if other == path or not other.is_file() or other.stat().st_size != size:
            continue
        if format_hint is not None and row.format_type != format_hint:
            continue
        digest = digest or _sha256(path)
        if _sha256(other) == digest:
            return Document.model_validate(row)
    return None


def _ingest(doc_id: str, path: Path, format_hint: FormatType | None, db: Session) -> Document:
    """Detect -> extract -> chunk -> calibrate -> label -> embed -> store. Raises HTTPException on expected failures."""
    unreadable = f"File is not a readable {path.suffix} file (damaged, or not what its extension says)"
    try:
        pages = format_detection.page_count(str(path))
        # Detection reads a few sample pages; they are reused below instead of read twice.
        format_type, already_read = format_detection.resolve(str(path), format_hint)
    except (PdfminerException, file_types.UnreadableFileError):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, unreadable)
    except (ocr_extractor.OCRUnavailableError, htr_extractor.HTRUnavailableError) as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e))

    document = Document(doc_id=doc_id, file_path=str(path), format_type=format_type, page_count=pages)
    try:
        extracted = format_detection.extract(document, already_read)
        if format_type is FormatType.HANDWRITTEN:
            extracted = htr_correction.correct_pages(extracted)
    except file_types.UnreadableFileError:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, unreadable)
    except (ocr_extractor.OCRUnavailableError, htr_extractor.HTRUnavailableError) as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e))

    numbered = chunker.chunk_pages_numbered(doc_id, extracted)
    chunks = [chunk for _, chunk in numbered]
    if not chunks:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "No readable text found in this file")
    chunks = confidence_capture.apply_calibration(chunks, format_type)
    chunks = content_type.label_chunks(chunks)

    try:
        chunks = embedder.embed_chunks(chunks)
    except embedder.EmbeddingError as e:
        logger.error("Embedding failed for %s: %s", doc_id, e)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Embedding service failed; try again later")
    vector_store.add_chunks(doc_id, chunks, {chunk.chunk_id: loc for loc, chunk in numbered})

    try:
        db.add(DocumentORM(**document.model_dump()))
        db.commit()
    except Exception:
        # Keep ChromaDB and SQLite consistent: no vectors without a Document row.
        vector_store.delete_document(doc_id)
        raise

    mean_conf = sum(c.calibrated_conf for c in chunks) / len(chunks)
    logger.info(
        "Ingested %s: %s, %d pages, %d chunks, mean calibrated confidence %.2f",
        doc_id, format_type.value, pages, len(chunks), mean_conf,
    )
    return document
