"""Module 1 — Document Upload.

Responsibility: accept an upload (PDF, image, Word, PowerPoint, or text; see
file_types.py), save it under settings.upload_dir, run it through
detection -> extraction (parse/OCR/HTR) -> chunking ->
calibration -> content-type labeling -> embedding -> ChromaDB, and record a
Document row.

Uses from schemas.py: Document, FormatType.
"""

import hashlib
import logging
import re
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
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
    htr_extractor,
    ocr_extractor,
    vector_store,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["upload"])


# Plain `def` (not async): PDF parsing, OCR, and HTR are blocking, so FastAPI runs this in its threadpool.
@router.post("/upload", response_model=Document, status_code=status.HTTP_201_CREATED)
def upload_document(
    file: UploadFile = File(...),
    format_hint: FormatType | None = Form(
        None,
        description=(
            "Skip automatic detection and force this format (e.g. if detection guesses wrong). "
            "Ignored for Word/PowerPoint/text files (always typed), and 'typed' is ignored for images."
        ),
    ),
    db: Session = Depends(get_db),
    response: Response = None,
) -> Document:
    """Ingest a file. 201 = new document; 200 = the same file was already uploaded (that Document is returned)."""
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
        format_type = format_detection.resolve_format(str(path), format_hint)
    except (PdfminerException, file_types.UnreadableFileError):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, unreadable)
    except (ocr_extractor.OCRUnavailableError, htr_extractor.HTRUnavailableError) as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e))

    document = Document(doc_id=doc_id, file_path=str(path), format_type=format_type, page_count=pages)
    try:
        extracted = format_detection.extract(document)
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
    vector_store.add_chunks(doc_id, chunks, {chunk.chunk_id: page for page, chunk in numbered})

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
