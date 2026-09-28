"""Supports Module 2 — turn PDF pages and image files into page images for OCR and HTR.

Responsibility: yield one grayscale PIL image per page:
- PDF: rasterize with pypdfium2 (already a pdfplumber dependency, so no
  Poppler install is needed)
- image files: each image is one page (multi-page TIFF: one page per frame);
  phone photos are turned upright from their EXIF orientation, transparency
  is flattened onto white, and very large photos are scaled down

Uses from schemas.py: nothing; returns PIL images to ocr_extractor/htr_extractor.
"""

from collections.abc import Iterator

import pypdfium2 as pdfium
from PIL import Image, ImageFilter, ImageOps, ImageSequence, UnidentifiedImageError

from app.modules.file_types import FileKind, UnreadableFileError, kind_of

PDF_POINTS_PER_INCH = 72
# Images carry no reliable page size, so treat each as an A4 page (11.69 in tall)
# and shrink anything whose long side exceeds that at the requested DPI.
# A 12-megapixel phone photo is ~4000 px; OCR at 300 DPI needs at most ~3500 px.
PAGE_LONG_SIDE_INCHES = 11.69


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

    pdf = pdfium.PdfDocument(file_path)
    try:
        count = len(pdf) if max_pages is None else min(len(pdf), max_pages)
        for i in range(first_page, count):
            page = pdf[i]
            try:
                yield page.render(scale=dpi / PDF_POINTS_PER_INCH).to_pil().convert("L")
            finally:
                page.close()
    finally:
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

    max_side = int(dpi * PAGE_LONG_SIDE_INCHES)
    if max(page.size) > max_side:
        scale = max_side / max(page.size)
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
