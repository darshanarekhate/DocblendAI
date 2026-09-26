"""Module 2 — OCR path (scanned/printed PDFs and images).

Responsibility: rasterize pages and run an OCR engine, returning page text
plus a confidence (0-1) as raw_conf for Module 3. raw_conf is the mean of the
engine's word confidences, weighted by word length, so it tracks the share
of characters recognized confidently.

Two engines (settings.ocr_engine):
- Tesseract via pytesseract: needs the Tesseract binary on the system (see CLAUDE.md).
- docTR (mindee/doctr): DBNet text detection + CRNN recognition in PyTorch,
  installed with pip alone. "auto" uses Tesseract when it is installed and
  docTR otherwise, so the app runs on a laptop without Tesseract.

Uses from schemas.py: nothing directly; output feeds chunker.py -> RecognizedChunk.
"""

import logging
import shutil
from contextlib import closing
from functools import lru_cache
from pathlib import Path

import pytesseract
from PIL import Image

from app.config import settings
from app.modules.pdf_render import denoise, render_pages

logger = logging.getLogger(__name__)

# The UB Mannheim Windows installer's default location (not added to PATH by default).
_WINDOWS_DEFAULT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
# A clean page OCRs in 1-3 s. A page Tesseract is still chewing on after this is
# extreme noise, and reading it as unreadable beats blocking the upload for minutes.
OCR_TIMEOUT_SECONDS = 30


class OCRUnavailableError(RuntimeError):
    """No OCR engine is available (Tesseract not found, or docTR could not load)."""


@lru_cache
def _tesseract_cmd(configured: str) -> str:
    for candidate in (configured, shutil.which("tesseract"), str(_WINDOWS_DEFAULT)):
        if candidate and Path(candidate).is_file():
            return candidate
    raise OCRUnavailableError(
        "Tesseract is not installed. Install it (Windows: winget install UB-Mannheim.TesseractOCR) "
        "or set TESSERACT_CMD in .env to the full path of tesseract.exe."
    )


def _engine() -> str:
    if settings.ocr_engine != "auto":
        return settings.ocr_engine
    try:
        _tesseract_cmd(settings.tesseract_cmd)
        return "tesseract"
    except OCRUnavailableError:
        return "doctr"


def ocr_image(image: Image.Image) -> tuple[str, float]:
    """OCR one page image. Returns (text, raw_conf); a page with no words gives ("", 0.0)."""
    if _engine() == "doctr":
        return _doctr_ocr(image)
    return _tesseract_ocr(image)


@lru_cache
def _doctr_predictor():
    try:
        from doctr.models import ocr_predictor

        return ocr_predictor("db_resnet50", "crnn_vgg16_bn", pretrained=True, assume_straight_pages=True)
    except (ImportError, OSError, RuntimeError) as e:
        raise OCRUnavailableError(
            f"docTR OCR could not load ({e}). Install Tesseract (Windows: winget install "
            "UB-Mannheim.TesseractOCR) or run: pip install python-doctr"
        ) from e


def _doctr_ocr(image: Image.Image) -> tuple[str, float]:
    import numpy as np

    result = _doctr_predictor()([np.asarray(denoise(image).convert("RGB"))])
    # docTR blocks ~ paragraphs: lines joined by newlines, blocks by a blank line (as for Tesseract).
    blocks, weighted, chars = [], 0.0, 0
    for block in result.pages[0].blocks:
        lines = []
        for line in block.lines:
            words = [w for w in line.words if w.value.strip()]
            for w in words:
                weighted += float(w.confidence) * len(w.value)
                chars += len(w.value)
            if words:
                lines.append(" ".join(w.value for w in words))
        if lines:
            blocks.append("\n".join(lines))
    if not chars:
        return "", 0.0
    return "\n\n".join(blocks), weighted / chars


def _tesseract_ocr(image: Image.Image) -> tuple[str, float]:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd(settings.tesseract_cmd)
    try:
        data = pytesseract.image_to_data(
            denoise(image), output_type=pytesseract.Output.DICT, timeout=OCR_TIMEOUT_SECONDS
        )
    except RuntimeError as e:  # pytesseract raises RuntimeError("Tesseract process timeout")
        if "timeout" not in str(e).lower():
            raise
        logger.warning("OCR gave up on a page after %ss; treating it as unreadable", OCR_TIMEOUT_SECONDS)
        return "", 0.0

    # Rebuild text from word boxes: words on the same line joined by spaces,
    # lines by newlines, paragraphs by a blank line. conf == -1 marks non-word boxes.
    lines: dict[tuple[int, int, int], list[str]] = {}
    weighted, chars = 0.0, 0
    for word, conf, block, par, line in zip(
        data["text"], data["conf"], data["block_num"], data["par_num"], data["line_num"]
    ):
        word = word.strip()
        if not word or float(conf) < 0:
            continue
        lines.setdefault((block, par, line), []).append(word)
        weighted += float(conf) / 100 * len(word)
        chars += len(word)

    if not chars:
        return "", 0.0

    parts: list[str] = []
    prev_par = None
    for (block, par, _), words in lines.items():  # dicts keep Tesseract's reading order
        if prev_par is not None and (block, par) != prev_par:
            parts.append("")
        parts.append(" ".join(words))
        prev_par = (block, par)
    return "\n".join(parts), weighted / chars


def _quick_conf(image: Image.Image) -> float:
    """OCR confidence of a half-size copy: cheap enough to compare orientations."""
    small = image.reduce(2) if max(image.size) > 1600 else image
    return ocr_image(small)[1]


def auto_orient(page: Image.Image) -> Image.Image:
    """Turn a sideways or upside-down page upright.

    Phone photos and scans often arrive rotated with no (or a wrong) EXIF tag,
    and both OCR and HTR then read garbage. Tesseract's orientation detection
    (OSD) proposes a rotation, but its confidence on short or handwritten pages
    is low, so the proposal is verified: the page is OCRed as-is and rotated,
    and the orientation that reads with higher confidence wins. Upright pages
    (OSD says 0) cost one OSD call and nothing else.

    Without Tesseract, or when OSD cannot decide (too little text), the page is
    returned unchanged.
    """
    try:
        pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd(settings.tesseract_cmd)
        osd = pytesseract.image_to_osd(page, output_type=pytesseract.Output.DICT, timeout=OCR_TIMEOUT_SECONDS)
    except (OCRUnavailableError, pytesseract.TesseractError, RuntimeError):
        return page
    rotate = int(osd.get("rotate", 0)) % 360
    if rotate == 0:
        return page

    # OSD's rotate direction is ambiguous across versions for 90/270, so try both turns.
    candidates = [page, page.rotate(-rotate, expand=True)]
    if rotate in (90, 270):
        candidates.append(page.rotate(rotate, expand=True))
    best = max(candidates, key=_quick_conf)
    if best is not page:
        logger.info("Page was rotated (OSD suggested %d degrees); turned it upright", rotate)
    return best


def ocr_file(file_path: str, max_pages: int | None = None) -> list[tuple[str, float]]:
    """Return (page_text, raw_conf) for each page of a PDF or image (or the first max_pages)."""
    # closing(): if OCR fails mid-document, release the PDF now, not at garbage
    # collection (Windows cannot delete a file that is still open).
    with closing(render_pages(file_path, settings.ocr_dpi, max_pages)) as pages:
        return [ocr_image(auto_orient(img)) for img in pages]
