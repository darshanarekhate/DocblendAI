"""Supports Module 2 — turn PDF pages and image files into page images for OCR and HTR.

Responsibility: yield one grayscale PIL image per page:
- PDF: rasterize with pypdfium2 (already a pdfplumber dependency, so no
  Poppler install is needed)
- image files: each image is one page (multi-page TIFF: one page per frame);
  phone photos are turned upright from their EXIF orientation, transparency
  is flattened onto white, and very large photos are scaled down

Uses from schemas.py: nothing; returns PIL images to ocr_extractor/htr_extractor.
"""

import threading
from collections.abc import Iterator

import pypdfium2 as pdfium
from PIL import Image, ImageFilter, ImageOps, ImageSequence, UnidentifiedImageError

from app.modules.file_types import FileKind, UnreadableFileError, kind_of

PDF_POINTS_PER_INCH = 72
_PDFIUM_LOCK = threading.RLock()  # see render_pages
# Images carry no reliable page size, so treat each as an A4 page (11.69 in tall)
# and shrink anything whose long side exceeds that at the requested DPI.
# A 12-megapixel phone photo is ~4000 px; OCR at 300 DPI needs at most ~3500 px.
PAGE_LONG_SIDE_INCHES = 11.69
# An image whose long side is below this share of the target size is enlarged to the target.
MIN_SHARE_OF_PAGE = 0.6
MAX_ENLARGE = 4.0  # at most 4x: a small snippet (one cropped line) should not become a poster


def denoise(image: Image.Image) -> Image.Image:
    """Remove photocopy/scan speckle before recognition.

    A 3x3 median filter deletes isolated specks but keeps strokes. Without it,
    Tesseract can spend minutes on a noisy page (every speck is a candidate
    component) and HTR line segmentation mistakes speckle for text rows.
    """
    return image.filter(ImageFilter.MedianFilter(3))


def render_pages(file_path: str, dpi: int, max_pages: int | None = None, first_page: int = 0) -> Iterator[Image.Image]:
    """Yield each page (PDF page or image frame) as a grayscale PIL image at about the given DPI.

    max_pages limits to the document's first max_pages pages; first_page (0-based) skips
    the pages before it, e.g. ones already read while detecting the format.
    """
    if kind_of(file_path) is FileKind.IMAGE:
        yield from _image_pages(file_path, dpi, max_pages, first_page)
        return

    # pdfium is not thread-safe: the web page loads several page images at once and background
    # uploads render pages too, and overlapping calls fail with "Failed to load page" (HTTP 500,
    # broken preview images). Every pdfium call runs under one lock; the lock is not held while
    # the caller works on a yielded page (OCR/HTR), so other renders are not held up by that.
    with _PDFIUM_LOCK:
        pdf = pdfium.PdfDocument(file_path)
        count = len(pdf) if max_pages is None else min(len(pdf), max_pages)
    try:
        for i in range(first_page, count):
            with _PDFIUM_LOCK:
                page = pdf[i]
                try:
                    image = page.render(scale=dpi / PDF_POINTS_PER_INCH).to_pil().convert("L")
                finally:
                    page.close()
            yield image
    finally:
        with _PDFIUM_LOCK:
            pdf.close()


def _open_image(file_path: str) -> Image.Image:
    try:
        return Image.open(file_path)
    except (UnidentifiedImageError, OSError) as e:
        raise UnreadableFileError(f"not a readable image: {e}") from e


def image_page_count(file_path: str) -> int:
    with _open_image(file_path) as img:
        return getattr(img, "n_frames", 1)


def _to_page(frame: Image.Image, dpi: int) -> Image.Image:
    page = ImageOps.exif_transpose(frame.copy())  # copy(): the frame object is reused by the iterator
    if page.mode in ("RGBA", "LA", "PA") or "transparency" in page.info:
        white = Image.new("RGBA", page.size, (255, 255, 255, 255))
        page = Image.alpha_composite(white, page.convert("RGBA"))
    page = page.convert("L")

    # Bring every page to about the requested DPI for an A4 page: shrink big phone photos, and
    # enlarge small ones (web or chat images of ~500-1000 px), whose text lines are only ~12 px
    # tall: line detection and TrOCR then read noise. A 465x660 note page went from CER 1.84 to
    # 0.21 (and from 49 s to 14 s, as fewer junk fragments are read) once enlarged.
    max_side = int(dpi * PAGE_LONG_SIDE_INCHES)
    if max(page.size) > max_side or max(page.size) < MIN_SHARE_OF_PAGE * max_side:
        scale = min(max_side / max(page.size), MAX_ENLARGE)
        page = page.resize((round(page.width * scale), round(page.height * scale)), Image.LANCZOS)
    return page


def _image_pages(file_path: str, dpi: int, max_pages: int | None, first_page: int = 0) -> Iterator[Image.Image]:
    img = _open_image(file_path)
    try:
        for i, frame in enumerate(ImageSequence.Iterator(img)):
            if max_pages is not None and i >= max_pages:
                break
            if i < first_page:
                continue
            try:
                yield _to_page(frame, dpi)
            except OSError as e:  # truncated / corrupt pixel data surfaces only on decode
                raise UnreadableFileError(f"image data is damaged: {e}") from e
    finally:
        img.close()
