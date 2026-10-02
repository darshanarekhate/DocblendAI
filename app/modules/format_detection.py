"""Module 2 — Format Detection & Text Extraction (router).

Responsibility: decide whether a document is typed, scanned, or handwritten,
then route it to the matching extractor (text_parser / ocr_extractor /
htr_extractor).

- Word, PowerPoint, and text files are always typed.
- PDFs with a text layer are typed; other PDFs and all images are scanned or
  handwritten, decided by which recognition engine reads them better.

Uses from schemas.py: Document, FormatType.
"""

import pdfplumber

from app.config import settings
from app.models.schemas import Document, FormatType
from app.modules import confidence_capture, htr_extractor, ocr_extractor, text_parser
from app.modules.file_types import TEXT_KINDS, FileKind, kind_of
from app.modules.pdf_render import image_page_count

# Pages sampled to tell printed scans from handwriting (keeps detection fast on long PDFs).
OCR_SAMPLE_PAGES = 2

# A page counts as "has a text layer" above this many extracted characters
# (filters out stray page numbers / headers on scanned pages).
MIN_TEXT_CHARS_PER_PAGE = 25
# Share of pages that must have a text layer for the PDF to count as typed.
MIN_TEXT_PAGE_RATIO = 0.5


def page_count(file_path: str) -> int:
    """Pages for PDFs, frames for images, slides for PowerPoint; Word and text files count as 1.

    Raises PdfminerException for unreadable PDFs and file_types.UnreadableFileError
    for other unreadable files.
    """
    kind = kind_of(file_path)
    if kind is FileKind.IMAGE:
        return image_page_count(file_path)
    if kind in TEXT_KINDS:
        return len(text_parser.parse(file_path))
    with pdfplumber.open(file_path) as pdf:
        return len(pdf.pages)


def has_text_layer(file_path: str) -> bool:
    """True if enough pages carry extractable text (i.e. a typed/digital PDF). Images never do.

    Raises pdfplumber's PdfminerException if the file is not a readable PDF.
    """
    if kind_of(file_path) is FileKind.IMAGE:
        return False
    with pdfplumber.open(file_path) as pdf:
        pages = pdf.pages
        if not pages:
            return False
        text_pages = sum(len((p.extract_text() or "").strip()) >= MIN_TEXT_CHARS_PER_PAGE for p in pages)
    return text_pages / len(pages) >= MIN_TEXT_PAGE_RATIO


def _mean_conf(pages: list[tuple[str, float]]) -> float:
    chars = sum(len(text) for text, _ in pages)
    return sum(conf * len(text) for text, conf in pages) / chars if chars else 0.0


def detect_format(file_path: str) -> FormatType:
    """Classify the document as typed, scanned (printed), or handwritten (see _detect)."""
    return _detect(file_path)[0]


Pages = list[tuple[str, float]]


def _detect(file_path: str) -> tuple[FormatType, Pages]:
    """Classify the document as typed, scanned (printed), or handwritten.

    Also returns the sample pages already read with the winning engine (the
    first ones of the document), so extraction does not read them a second time.

    Word/PowerPoint/text files are typed. A PDF with a text layer is typed.
    Otherwise (other PDFs, images) the first OCR_SAMPLE_PAGES are
    OCRed: if Tesseract is confident (calibrated mean >= settings.scanned_min_ocr_conf)
    the PDF is scanned. If not, the sample is also read with HTR and the
    engine with the higher calibrated confidence wins. Low OCR confidence
    alone does not mean handwriting: a bad photocopy of print also OCRs
    poorly, and HTR reads it far worse.

    "scanned" therefore means "OCR reads it best": very regular handwriting
    that Tesseract reads confidently is routed to OCR, which is the better
    engine for it. Use the upload's format_hint to override.

    Raises PdfminerException for unreadable PDFs,
    ocr_extractor.OCRUnavailableError if Tesseract is not installed, and
    htr_extractor.HTRUnavailableError if HTR is needed but cannot load.
    """
    if kind_of(file_path) in TEXT_KINDS or has_text_layer(file_path):
        return FormatType.TYPED, []

    ocr_pages = ocr_extractor.ocr_file(file_path, max_pages=OCR_SAMPLE_PAGES)
    ocr_conf = confidence_capture.calibrate(_mean_conf(ocr_pages), FormatType.SCANNED)
    if ocr_conf >= settings.scanned_min_ocr_conf:
        return FormatType.SCANNED, ocr_pages

    htr_pages = htr_extractor.htr_file(file_path, max_pages=OCR_SAMPLE_PAGES)
    htr_conf = confidence_capture.calibrate(_mean_conf(htr_pages), FormatType.HANDWRITTEN)
    if htr_conf > ocr_conf:
        return FormatType.HANDWRITTEN, htr_pages
    return FormatType.SCANNED, ocr_pages


def resolve(file_path: str, hint: FormatType | None) -> tuple[FormatType, Pages]:
    """The user's format hint if it can apply, else automatic detection.

    Word/PowerPoint/text files are always typed (there is nothing to OCR), and
    an image can never be typed (it has no text layer), so those hints are ignored.
    Returns the format and the pages detection already read with that format's
    engine (empty when the hint was used); pass them on to extract().
    """
    kind = kind_of(file_path)
    if kind in TEXT_KINDS:
        return FormatType.TYPED, []
    if hint is None or (kind is FileKind.IMAGE and hint is FormatType.TYPED):
        return _detect(file_path)
    return hint, []


def resolve_format(file_path: str, hint: FormatType | None) -> FormatType:
    return resolve(file_path, hint)[0]


def extract(document: Document, already_read: Pages = ()) -> Pages:
    """Dispatch to the right extractor. Returns (page_text, raw_conf) per page.

    already_read: the document's first pages as detection read them with this
    format's engine; only the pages after them are read here.
    """
    if document.format_type is FormatType.TYPED:
        return text_parser.parse(document.file_path)
    read = ocr_extractor.ocr_file if document.format_type is FormatType.SCANNED else htr_extractor.htr_file
    done = list(already_read)
    return done + read(document.file_path, first_page=len(done))
