"""Document preview routes — let users look at what they uploaded (supports Module 1).

Responsibility: show an uploaded document in the web UI without leaving the app.
- PDFs and images: one page at a time, rendered to PNG (the same renderer
  OCR/HTR use, so multi-page TIFFs and scans look exactly as the engines saw them)
- Word, PowerPoint, text: the text as parsed, one entry per page/slide
- /view: the whole document on one plain page, opened in a new browser tab

API-only views; nothing here is a synopsis entity.
"""

import html
import io
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import HTMLResponse
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


def _original_name(row: DocumentORM) -> str:
    base = Path(row.file_path).name  # stored as "<doc_id>_<original name>"
    return base[len(row.doc_id) + 1 :] if base.startswith(row.doc_id + "_") else base


VIEW_STYLE = """
  * { box-sizing: border-box; }
  body { margin: 0; background: #eef5fc; color: #1c2a3a; font: 17px/1.6 "Times New Roman", Times, serif; }
  header { padding: 16px 24px; background: #fff; border-bottom: 1px solid #d5e3f1; }
  h1 { margin: 0; font-size: 22px; }
  header p { margin: 2px 0 0; color: #5a6b7d; font-size: 15px; }
  main { max-width: 860px; margin: 0 auto; padding: 20px 16px 48px; }
  h2 { margin: 24px 0 8px; font-size: 16px; color: #5a6b7d; font-weight: normal; }
  img, pre { display: block; width: 100%; margin: 0; background: #fff; border: 1px solid #d5e3f1; border-radius: 6px; }
  pre { padding: 18px 20px; white-space: pre-wrap; font: inherit; }
"""


@router.get("/documents/{doc_id}/view", response_class=HTMLResponse)
def view(doc_id: str, db: Session = Depends(get_db)) -> HTMLResponse:
    """The whole document on one page: page images for PDFs/images, the parsed text otherwise."""
    row = _document(doc_id, db)
    name = html.escape(_original_name(row))
    if kind_of(row.file_path) in TEXT_KINDS:
        try:
            texts = [text for text, _ in text_parser.parse(row.file_path)]
        except (OSError, UnreadableFileError):
            raise HTTPException(status.HTTP_410_GONE, "The uploaded file is no longer readable")
        bodies = [f"<pre>{html.escape(t) or '(This page has no text.)'}</pre>" for t in texts]
    else:
        bodies = [
            f'<img src="/documents/{doc_id}/pages/{n}" alt="{name}, page {n}" loading="lazy">'
            for n in range(1, row.page_count + 1)
        ]
    pages = "".join(f"<h2>Page {n}</h2>{body}" for n, body in enumerate(bodies, 1))
    count = f"{len(bodies)} page{'' if len(bodies) == 1 else 's'}"
    return HTMLResponse(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{name}</title>'
        f"<style>{VIEW_STYLE}</style></head><body>"
        f"<header><h1>{name}</h1><p>{count}</p></header><main>{pages}</main></body></html>"
    )
