"""Document preview routes — let users look at what they uploaded (supports Module 1).

Responsibility: show an uploaded document in the web UI without leaving the app.
- PDFs and images: one page at a time, rendered to PNG (the same renderer
  OCR/HTR use, so multi-page TIFFs and scans look exactly as the engines saw them)
- Word, PowerPoint, text: the text as parsed, one entry per page/slide

API-only views; nothing here is a synopsis entity.
"""

import io

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import DocumentORM
from app.modules import text_parser
from app.modules.file_types import TEXT_KINDS, UnreadableFileError, kind_of
from app.modules.pdf_render import render_pages

router = APIRouter(tags=["preview"])

PREVIEW_DPI = 110  # sharp enough to read on screen, small enough to load quickly


class DocumentPreview(BaseModel):
    doc_id: str
    kind: str  # "image" = fetch /pages/{n}; "text" = text_pages holds the content
    page_count: int
    text_pages: list[str] | None = None


def _document(doc_id: str, db: Session) -> DocumentORM:
    row = db.get(DocumentORM, doc_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return row


@router.get("/documents/{doc_id}/preview", response_model=DocumentPreview)
def preview(doc_id: str, db: Session = Depends(get_db)) -> DocumentPreview:
    row = _document(doc_id, db)
    if kind_of(row.file_path) in TEXT_KINDS:
        try:
            pages = [text for text, _ in text_parser.parse(row.file_path)]
        except (OSError, UnreadableFileError):
            raise HTTPException(status.HTTP_410_GONE, "The uploaded file is no longer readable")
        return DocumentPreview(doc_id=doc_id, kind="text", page_count=len(pages), text_pages=pages)
    return DocumentPreview(doc_id=doc_id, kind="image", page_count=row.page_count)


@router.get("/documents/{doc_id}/pages/{page}", response_class=Response)
def page_image(doc_id: str, page: int, db: Session = Depends(get_db)) -> Response:
    """Page `page` (1-based) of a PDF or image, as PNG."""
    row = _document(doc_id, db)
    if kind_of(row.file_path) in TEXT_KINDS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "This document has no page images; use /preview")
    if not 1 <= page <= row.page_count:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Page {page} does not exist")
    try:
        *_, image = render_pages(row.file_path, PREVIEW_DPI, max_pages=page)
    except (OSError, UnreadableFileError, ValueError):
        raise HTTPException(status.HTTP_410_GONE, "The uploaded file is no longer readable")
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})
