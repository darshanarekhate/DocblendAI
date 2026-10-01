"""PaddleOCR Experience Center engine: text OCR, document parsing, office conversion.

Supports Module 2 (text extraction) and Module 3 (confidence capture + calibration) for the
Experience Center (/studio, /api/*). Interface: docs/experience_center_contract.md §2 and §4.

Two layers:
- pure functions (`normalise_ocr_page`, `normalise_structure_page`, `parse_table_html`,
  `merge_word_tokens`, `apply_calibration`, `summarize`, ...) that turn PaddleOCR's
  `result.json["res"]` dicts into the contract's result object. They never import paddle, so
  tests run them on recorded fixture JSON.
- `PaddleOCRService` (singleton `paddleocr_service`): lazy, thread-safe model loading once per
  (pipeline, language), input rendering (images, multi-frame TIFF, PDFs via PyMuPDF) and inference.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
import os
import re
import threading
import time
from collections.abc import Callable
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
PDF_SUFFIXES = frozenset({".pdf"})
OFFICE_SUFFIXES = frozenset({".docx", ".xlsx", ".pptx"})
PIPELINES = ("ocr", "structure", "vl", "office")

# PaddleOCR language codes offered in the UI (PaddleOCR picks the matching recognition model).
# CLAUDE.md keeps multilingual scripts out of the QA scope; this list is Experience Center only.
LANGUAGES: list[dict[str, str]] = [
    {"code": "en", "name": "English"},
    {"code": "ch", "name": "Chinese (simplified) + English"},
    {"code": "chinese_cht", "name": "Chinese (traditional)"},
    {"code": "japan", "name": "Japanese"},
    {"code": "korean", "name": "Korean"},
    {"code": "fr", "name": "French"},
    {"code": "de", "name": "German"},
    {"code": "es", "name": "Spanish"},
    {"code": "it", "name": "Italian"},
    {"code": "pt", "name": "Portuguese"},
    {"code": "ru", "name": "Russian"},
    {"code": "ar", "name": "Arabic"},
    {"code": "hi", "name": "Hindi"},
    {"code": "mr", "name": "Marathi"},
    {"code": "ta", "name": "Tamil"},
    {"code": "te", "name": "Telugu"},
]
LANGUAGE_CODES = frozenset(lang["code"] for lang in LANGUAGES)

# Longest image side fed to the models; PaddleOCR's own detector limit is 4000 px.
MAX_IMAGE_SIDE = 4000

VL_DISABLED_MESSAGE = (
    "PaddleOCR-VL is disabled: set PADDLE_ENABLE_VL=true; needs a GPU or a lot of RAM/patience."
)

ProgressFn = Callable[[float, str], None]


class PaddleOCRUnavailableError(RuntimeError):
    """A pipeline cannot run here: PaddleOCR missing, disabled, or its models failed to load."""


class UnreadableInputError(ValueError):
    """The uploaded file cannot be opened as the type its extension claims."""


# ---------------------------------------------------------------------------------------------
# Pure normalisation (no paddle import): PaddleOCR res dicts -> contract §4 page objects
# ---------------------------------------------------------------------------------------------


def _clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number:  # NaN
        return 0.0
    return max(0.0, min(1.0, number))


def _box(values: Any) -> list[float] | None:
    try:
        x0, y0, x1, y1 = (float(v) for v in list(values)[:4])
    except (TypeError, ValueError):
        return None
    return [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]


def _polygon(points: Any) -> list[list[float]]:
    out = []
    for point in points or []:
        try:
            out.append([round(float(point[0]), 2), round(float(point[1]), 2)])
        except (TypeError, ValueError, IndexError):
            continue
    return out


def _box_of_polygon(polygon: list[list[float]]) -> list[float] | None:
    if not polygon:
        return None
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def merge_word_tokens(tokens: list[Any], boxes: list[Any]) -> list[dict[str, Any]]:
    """PaddleOCR's per-line `text_word`/`text_word_boxes` -> words.

    The tokens include ' ' separators and split punctuation ("set", "."), so consecutive
    non-space tokens are merged into one word with the union of their boxes.
    """
    words: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for token, raw_box in zip(tokens or [], boxes or []):
        text = str(token)
        if not text.strip():
            current = None
            continue
        box = _box(raw_box)
        if box is None:
            continue
        if current is None:
            current = {"text": text.strip(), "box": box}
            words.append(current)
        else:
            current["text"] += text.strip()
            b = current["box"]
            current["box"] = [min(b[0], box[0]), min(b[1], box[1]), max(b[2], box[2]), max(b[3], box[3])]
    return words


def _lines_from_ocr_res(res: dict[str, Any], page_index: int) -> list[dict[str, Any]]:
    """rec_texts/rec_scores/rec_polys are index-aligned (dt_polys is not: never use it here)."""
    texts = list(res.get("rec_texts") or [])
    scores = list(res.get("rec_scores") or [])
    polys = list(res.get("rec_polys") or [])
    rec_boxes = list(res.get("rec_boxes") or [])
    word_tokens = list(res.get("text_word") or [])
    word_boxes = list(res.get("text_word_boxes") or [])
    lines = []
    for i, text in enumerate(texts):
        polygon = _polygon(polys[i]) if i < len(polys) else []
        box = _box(rec_boxes[i]) if i < len(rec_boxes) else None
        box = box or _box_of_polygon(polygon)
        if not polygon and box:
            polygon = [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]
        words = (
            merge_word_tokens(word_tokens[i], word_boxes[i])
            if i < len(word_tokens) and i < len(word_boxes)
            else []
        )
        raw = _clamp01(scores[i]) if i < len(scores) else 0.0
        lines.append({
            "id": f"p{page_index}-l{i}",
            "text": str(text),
            "polygon": polygon,
            "box": box,
            "raw_confidence": round(raw, 6),
            "calibrated_confidence": round(raw, 6),
            "needs_review": False,
            "edited": False,
            "words": words,
        })
    return lines


def _empty_page(index: int, width: int | None, height: int | None, image_url: str | None) -> dict[str, Any]:
    return {
        "index": index, "width": width, "height": height, "image_url": image_url,
        "lines": [], "markdown": "", "blocks": [], "tables": [],
    }


def lines_markdown(lines: list[dict[str, Any]]) -> str:
    return "\n".join(line["text"] for line in lines)


def normalise_ocr_page(
    res: dict[str, Any], index: int, width: int | None = None, height: int | None = None,
    image_url: str | None = None,
) -> dict[str, Any]:
    """Text-OCR `result.json["res"]` -> contract page (lines only; markdown = lines joined)."""
    page = _empty_page(index, width or res.get("width"), height or res.get("height"), image_url)
    page["lines"] = _lines_from_ocr_res(res, index)
    page["markdown"] = lines_markdown(page["lines"])
    return page


class _TableParser(HTMLParser):
    """Collects the text of each <tr>'s <td>/<th> cells (colspan repeats the text)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._colspan = 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._close_row()
            self._row = []
        elif tag in ("td", "th"):
            self._close_cell()
            if self._row is None:
                self._row = []
            self._cell = []
            try:
                self._colspan = max(1, min(50, int(dict(attrs).get("colspan") or 1)))
            except ValueError:
                self._colspan = 1
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th"):
            self._close_cell()
        elif tag == "tr":
            self._close_row()

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def _close_cell(self) -> None:
        if self._cell is not None and self._row is not None:
            text = " ".join("".join(self._cell).split())
            self._row.extend([text] * self._colspan)
        self._cell = None
        self._colspan = 1

    def _close_row(self) -> None:
        self._close_cell()
        if self._row is not None:
            self.rows.append(self._row)
        self._row = None

    def close(self) -> None:
        super().close()
        self._close_row()


def parse_table_html(html: str) -> list[list[str]]:
    """'<table><tr><td>a</td>...' -> [["a", ...], ...] (first table's rows and any nested ones flattened)."""
    parser = _TableParser()
    parser.feed(html or "")
    parser.close()
    return parser.rows


_HTML_TABLE_RE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)


def tables_from_markdown(markdown: str) -> list[dict[str, Any]]:
    """Tables embedded in Markdown, as HTML (<table>) or pipe tables -> contract table objects."""
    tables = []
    for match in _HTML_TABLE_RE.finditer(markdown or ""):
        tables.append({"box": None, "html": match.group(0), "rows": parse_table_html(match.group(0)), "cells": []})
    block: list[str] = []
    for line in (markdown or "").splitlines() + [""]:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|") and len(stripped) > 1:
            block.append(stripped)
            continue
        if len(block) >= 2:
            rows = [[c.strip() for c in row.strip("|").split("|")] for row in block]
            rows = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c or "-") for c in r)]
            tables.append({"box": None, "html": None, "rows": rows, "cells": []})
        block = []
    return tables


def normalise_structure_page(
    res: dict[str, Any], markdown: str | None, index: int, width: int | None = None,
    height: int | None = None, image_url: str | None = None,
) -> dict[str, Any]:
    """PP-StructureV3 (or PaddleOCR-VL) `result.json["res"]` -> contract page.

    lines from overall_ocr_res, blocks from parsing_res_list, tables from table_res_list (their
    boxes are the boxes of the "table" blocks, in the same order), markdown from the pipeline's
    own markdown when given, else rebuilt from the blocks.
    """
    page = _empty_page(index, width or res.get("width"), height or res.get("height"), image_url)
    overall = res.get("overall_ocr_res") or {}
    page["lines"] = _lines_from_ocr_res(overall, index) if isinstance(overall, dict) else []

    blocks = []
    for block in res.get("parsing_res_list") or []:
        if not isinstance(block, dict):
            continue
        blocks.append({
            "type": str(block.get("block_label") or "text"),
            "box": _box(block.get("block_bbox") or []),
            "content": str(block.get("block_content") or "").strip(),
        })
    page["blocks"] = blocks

    table_boxes = [b["box"] for b in blocks if b["type"] == "table"]
    tables = []
    for i, table in enumerate(res.get("table_res_list") or []):
        if not isinstance(table, dict):
            continue
        html = str(table.get("pred_html") or "")
        tables.append({
            "box": table_boxes[i] if i < len(table_boxes) else None,
            "html": html,
            "rows": parse_table_html(html),
            "cells": [b for b in (_box(c) for c in table.get("cell_box_list") or []) if b],
        })
    page["tables"] = tables

    if markdown is None:
        parts = []
        for block in blocks:
            content = block["content"]
            if not content:
                continue
            if block["type"] == "doc_title":
                parts.append(f"# {content}")
            elif block["type"] == "paragraph_title":
                parts.append(f"## {content}")
            else:
                parts.append(content)
        markdown = "\n\n".join(parts)
    page["markdown"] = markdown.strip()
    return page


def office_page(markdown: str) -> dict[str, Any]:
    """Office (docx/xlsx/pptx) conversion -> one image-less page holding markdown + tables."""
    page = _empty_page(0, None, None, None)
    page["markdown"] = markdown.strip()
    page["tables"] = tables_from_markdown(markdown)
    return page


def all_lines(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [line for page in pages for line in page.get("lines", [])]


def apply_calibration(pages: list[dict[str, Any]], review_threshold: float) -> dict[str, Any]:
    """Fill calibrated_confidence / needs_review on every line; returns the contract's calibration object."""
    from app.calibration import calibrate_confidences, load_active

    lines = all_lines(pages)
    calibrated, status = calibrate_confidences([line["raw_confidence"] for line in lines]) if lines else ([], None)
    if status is None:  # no lines: still report whether a calibrator is active
        status = "calibrated" if _safe_active() is not None else "uncalibrated"
    for line, value in zip(lines, calibrated):
        line["calibrated_confidence"] = round(_clamp01(value), 6)
        line["needs_review"] = (not line.get("edited")) and line["calibrated_confidence"] < review_threshold
    method = None
    if status == "calibrated":
        active = _safe_active(load_active)
        method = getattr(active, "method", None)
    return {"status": status, "method": method, "review_threshold": review_threshold}


def _safe_active(loader: Callable[[], Any] | None = None) -> Any:
    try:
        if loader is None:
            from app.calibration import load_active as loader
        return loader()
    except Exception:  # a broken calibrator file must not break OCR
        logger.exception("could not load the active calibrator")
        return None


def summarize(pages: list[dict[str, Any]]) -> dict[str, Any]:
    lines = all_lines(pages)
    n = len(lines)
    return {
        "page_count": len(pages),
        "line_count": n,
        "mean_raw_confidence": round(sum(l["raw_confidence"] for l in lines) / n, 6) if n else None,
        "mean_calibrated_confidence": round(sum(l["calibrated_confidence"] for l in lines) / n, 6) if n else None,
        "needs_review": sum(1 for l in lines if l["needs_review"]),
    }


def full_text(pages: list[dict[str, Any]]) -> str:
    """Searchable text of a run: line texts, or the markdown when a page has no lines (office)."""
    parts = []
    for page in pages:
        parts.append(lines_markdown(page["lines"]) if page.get("lines") else page.get("markdown", ""))
    return "\n".join(p for p in parts if p)


def join_markdown(pages: list[dict[str, Any]]) -> str:
    return "\n\n".join(page["markdown"] for page in pages if page.get("markdown"))


# ---------------------------------------------------------------------------------------------
# Input rendering
# ---------------------------------------------------------------------------------------------


def kind_of_suffix(suffix: str) -> str | None:
    suffix = suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in PDF_SUFFIXES:
        return "pdf"
    if suffix in OFFICE_SUFFIXES:
        return "office"
    return None


def validate_input(path: Path, kind: str) -> None:
    """Open the file the way its extension claims; raise UnreadableInputError if it is not that."""
    if kind == "image":
        from PIL import Image, UnidentifiedImageError

        try:
            with Image.open(path) as img:
                img.verify()
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
            raise UnreadableInputError(f"not a readable image: {exc}") from exc
    elif kind == "pdf":
        import pymupdf

        try:  # opened from bytes: a failed open must not leave the file locked (Windows)
            with pymupdf.open(stream=path.read_bytes(), filetype="pdf") as doc:
                if doc.needs_pass:
                    raise UnreadableInputError("the PDF is password-protected")
                if doc.page_count < 1:
                    raise UnreadableInputError("the PDF has no pages")
        except UnreadableInputError:
            raise
        except Exception as exc:
            raise UnreadableInputError(f"not a readable PDF: {exc}") from exc
    elif kind == "office":
        import zipfile

        try:
            with zipfile.ZipFile(path) as zf:
                if "[Content_Types].xml" not in zf.namelist():
                    raise UnreadableInputError("not an Office Open XML file")
        except zipfile.BadZipFile as exc:
            raise UnreadableInputError(f"not an Office Open XML file: {exc}") from exc
    else:
        raise UnreadableInputError(f"unsupported file kind {kind!r}")


def _to_rgb(frame: Any) -> Any:
    from PIL import Image, ImageOps

    image = ImageOps.exif_transpose(frame.copy())
    if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info:
        white = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(white, image.convert("RGBA"))
    image = image.convert("RGB")
    if max(image.size) > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / max(image.size)
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)
    return image


def render_pages(path: Path, kind: str, out_dir: Path) -> tuple[list[Path], int]:
    """Write each page (image frame or PDF page) as out_dir/page-<n>.png.

    Returns (page image paths, total pages in the source); at most settings.paddle_max_pages are
    rendered. These PNGs are what the models read, so result coordinates are in their pixels.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    max_pages = max(1, settings.paddle_max_pages)
    paths: list[Path] = []
    if kind == "image":
        from PIL import Image, ImageSequence

        try:
            with Image.open(path) as img:
                total = getattr(img, "n_frames", 1)
                for i, frame in enumerate(ImageSequence.Iterator(img)):
                    if i >= max_pages:
                        break
                    target = out_dir / f"page-{i}.png"
                    _to_rgb(frame).save(target)
                    paths.append(target)
        except OSError as exc:
            raise UnreadableInputError(f"image data is damaged: {exc}") from exc
        return paths, total
    if kind == "pdf":
        import pymupdf

        try:
            with pymupdf.open(path, filetype="pdf") as doc:
                total = doc.page_count
                for i in range(min(total, max_pages)):
                    pix = doc[i].get_pixmap(dpi=settings.paddle_pdf_dpi, alpha=False)
                    target = out_dir / f"page-{i}.png"
                    pix.save(target)
                    paths.append(target)
        except Exception as exc:
            raise UnreadableInputError(f"could not render the PDF: {exc}") from exc
        return paths, total
    raise UnreadableInputError(f"cannot render {kind!r} files to pages")


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as img:
        return img.size


def _result_res(result: Any) -> dict[str, Any]:
    """PaddleX result object (or plain dict in tests) -> its res dict."""
    payload = None
    try:
        payload = result.json
    except AttributeError:
        payload = None
    if payload is None and isinstance(result, dict):
        payload = result
    if isinstance(payload, dict) and isinstance(payload.get("res"), dict):
        return payload["res"]
    return payload if isinstance(payload, dict) else {}


def _result_markdown(result: Any) -> str | None:
    try:
        md = result.markdown
    except AttributeError:
        return None
    if isinstance(md, dict):
        text = md.get("markdown_texts")
        return text if isinstance(text, str) else None
    return md if isinstance(md, str) else None


# ---------------------------------------------------------------------------------------------
# Model construction (the only place paddle is imported; tests replace _construct_model)
# ---------------------------------------------------------------------------------------------


def _import_paddleocr() -> Any:
    # torch must load before paddle on Windows: once paddle's OpenMP/MKL DLLs are loaded, torch
    # (TrOCR/docTR, same process) fails with "WinError 127 ... shm.dll". The reverse order works.
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    import paddleocr

    return paddleocr


def _model_names(lang: str) -> dict[str, str]:
    """Configured det/rec models for English; other languages let PaddleOCR choose by lang."""
    if lang != "en":
        return {}
    return {
        "text_detection_model_name": settings.paddle_det_model,
        "text_recognition_model_name": settings.paddle_rec_model,
    }


def _construct_model(pipeline: str, lang: str) -> Any:
    paddleocr = _import_paddleocr()
    common = {"device": settings.paddle_device, "enable_mkldnn": settings.paddle_enable_mkldnn}
    if pipeline == "ocr":
        return paddleocr.PaddleOCR(
            lang=lang,
            ocr_version=settings.paddle_ocr_version,
            use_doc_orientation_classify=settings.paddle_use_orientation,
            use_doc_unwarping=settings.paddle_use_unwarping,
            use_textline_orientation=settings.paddle_use_textline_orientation,
            return_word_box=True,
            **_model_names(lang),
            **common,
        )
    if pipeline == "structure":
        return paddleocr.PPStructureV3(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_seal_recognition=False,
            use_chart_recognition=False,
            formula_recognition_model_name=settings.paddle_formula_model,
            **_model_names(lang),
            **common,
        )
    if pipeline == "vl":
        return paddleocr.PaddleOCRVL(
            pipeline_version=settings.paddle_vl_version,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            **common,
        )
    raise ValueError(f"unknown pipeline {pipeline!r}")


def _package_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


class PaddleOCRService:
    """Process-wide PaddleOCR engine (use the `paddleocr_service` singleton).

    Each (pipeline, lang) model is constructed once, under its own lock (two requests for the same
    model wait for one load), and inference on a model is serialized: PaddleX predictors are not
    documented as thread-safe.
    """

    _instance: PaddleOCRService | None = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> PaddleOCRService:
        with cls._instance_lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._init_state()
                cls._instance = instance
            return cls._instance

    def _init_state(self) -> None:
        self._registry_lock = threading.Lock()
        self._models: dict[tuple[str, str], Any] = {}
        self._locks: dict[tuple[str, str], threading.Lock] = {}
        self._errors: dict[str, str] = {}

    # -- models ------------------------------------------------------------------------------

    def _lock_for(self, key: tuple[str, str]) -> threading.Lock:
        with self._registry_lock:
            return self._locks.setdefault(key, threading.Lock())

    def get_model(self, pipeline: str, lang: str = "en", progress: ProgressFn | None = None) -> Any:
        """The loaded model for (pipeline, lang), constructing it on first use."""
        status = self.pipeline_status(pipeline)
        if not status["available"]:
            raise PaddleOCRUnavailableError(status["message"])
        key = (pipeline, lang)
        if key in self._models:
            return self._models[key]
        with self._lock_for(key):
            if key in self._models:
                return self._models[key]
            if progress:
                progress(0.0, f"Loading the {pipeline} model ({lang}); the first run takes a while")
            started = time.perf_counter()
            try:
                model = _construct_model(pipeline, lang)
            except Exception as exc:
                message = f"Could not load the {pipeline} pipeline ({lang}): {type(exc).__name__}: {exc}"
                self._errors[pipeline] = message
                logger.exception("PaddleOCR %s/%s failed to load", pipeline, lang)
                raise PaddleOCRUnavailableError(message) from exc
            self._errors.pop(pipeline, None)
            self._models[key] = model
            logger.info("PaddleOCR %s/%s loaded in %.1f s", pipeline, lang, time.perf_counter() - started)
            return model

    def _predict(self, pipeline: str, lang: str, image_path: Path) -> Any:
        model = self.get_model(pipeline, lang)
        with self._lock_for((pipeline, lang)):
            results = model.predict(str(image_path))
            return next(iter(results), None)

    def unload(self) -> None:
        """Drop every loaded model (tests; frees memory)."""
        with self._registry_lock:
            self._models.clear()
            self._errors.clear()

    # -- status --------------------------------------------------------------------------------

    @staticmethod
    def installed() -> bool:
        return importlib.util.find_spec("paddleocr") is not None

    def pipeline_status(self, pipeline: str) -> dict[str, Any]:
        """{available, message} without loading anything (cheap: used by /api/health)."""
        if pipeline not in PIPELINES:
            return {"available": False, "message": f"unknown pipeline {pipeline!r}"}
        if not self.installed():
            return {"available": False, "message": "PaddleOCR is not installed (pip install -r requirements.txt)"}
        if pipeline == "vl":
            if not settings.paddle_enable_vl:
                return {"available": False, "message": VL_DISABLED_MESSAGE}
            if "vl" in self._errors:  # a failed VL load is not retried: it is minutes of work
                return {"available": False, "message": self._errors["vl"]}
        loaded = any(key[0] == pipeline for key in self._models)
        messages = {
            "ocr": "PP-OCR text detection + recognition (loaded)" if loaded else "PP-OCR text detection + recognition (loads on first use, a few seconds)",
            "structure": "PP-StructureV3 layout, tables, formulas (loaded)" if loaded else "PP-StructureV3 layout, tables, formulas (loads on first use, ~35 s; ~1 min per page on CPU)",
            "vl": "PaddleOCR-VL (loaded)" if loaded else "PaddleOCR-VL (loads on first use; slow on CPU)",
            "office": "docx/xlsx/pptx to Markdown (no OCR)",
        }
        message = messages[pipeline]
        if pipeline in self._errors:
            message += f"; last load failed: {self._errors[pipeline]}"
        return {"available": True, "message": message}

    @staticmethod
    def engine_info() -> dict[str, Any]:
        return {
            "paddleocr": _package_version("paddleocr"),
            "paddlepaddle": _package_version("paddlepaddle", "paddlepaddle-gpu"),
            "ocr_version": settings.paddle_ocr_version,
            "device": settings.paddle_device,
        }

    # -- pipelines -----------------------------------------------------------------------------

    def ocr_page(self, image_path: Path, index: int, lang: str = "en", image_url: str | None = None) -> dict[str, Any]:
        result = self._predict("ocr", lang, image_path)
        width, height = _image_size(image_path)
        return normalise_ocr_page(_result_res(result) if result is not None else {}, index, width, height, image_url)

    def structure_page(
        self, image_path: Path, index: int, lang: str = "en", pipeline: str = "structure",
        image_url: str | None = None,
    ) -> dict[str, Any]:
        result = self._predict(pipeline, lang, image_path)
        width, height = _image_size(image_path)
        if result is None:
            return normalise_structure_page({}, "", index, width, height, image_url)
        return normalise_structure_page(_result_res(result), _result_markdown(result), index, width, height, image_url)

    def recognize_lines(self, image_path: str | Path, lang: str = "en") -> list[tuple[str, float]]:
        """Text OCR of one image: [(line text, raw confidence)] in reading order (calibration CLI)."""
        import tempfile

        with tempfile.TemporaryDirectory(prefix="docblend-paddle-") as tmp:
            pages, _ = render_pages(Path(image_path), "image", Path(tmp))
            if not pages:
                return []
            page = self.ocr_page(pages[0], 0, lang)
        return [(line["text"], line["raw_confidence"]) for line in page["lines"]]

    def convert_office(self, path: Path) -> str:
        if not self.installed():
            raise PaddleOCRUnavailableError("PaddleOCR is not installed (pip install -r requirements.txt)")
        try:
            paddleocr = _import_paddleocr()
            return str(paddleocr.doc2md_convert(str(path)).markdown or "")
        except ImportError as exc:
            raise PaddleOCRUnavailableError(f"Office conversion is unavailable: {exc}") from exc

    def process(
        self, path: Path, pipeline: str, lang: str, pages_dir: Path,
        image_url: Callable[[int], str] | None = None, progress: ProgressFn | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Run one uploaded file through a pipeline -> (contract pages, note or None).

        Page images are written to pages_dir/page-<n>.png (kept: the UI draws boxes on them).
        Confidence calibration is applied separately (`apply_calibration`).
        """
        progress = progress or (lambda fraction, message: None)
        if pipeline == "office":
            progress(0.1, "Converting the document to Markdown")
            return [office_page(self.convert_office(path))], None

        kind = kind_of_suffix(path.suffix)
        if kind not in ("image", "pdf"):
            raise UnreadableInputError(f"{path.suffix} files cannot be OCR'd")
        self.get_model(pipeline, lang, progress)  # load first, so its time is not "page 1"
        progress(0.0, "Preparing pages")
        page_paths, total = render_pages(path, kind, pages_dir)
        note = None
        if total > len(page_paths):
            note = f"Only the first {len(page_paths)} of {total} pages were processed (PADDLE_MAX_PAGES)."
        pages = []
        n = len(page_paths)
        for i, page_path in enumerate(page_paths):
            progress(i / n, f"Page {i + 1} of {n}")
            url = image_url(i) if image_url else None
            if pipeline == "ocr":
                pages.append(self.ocr_page(page_path, i, lang, url))
            else:
                pages.append(self.structure_page(page_path, i, lang, pipeline, url))
        progress(1.0, "Finishing")
        return pages, note


paddleocr_service = PaddleOCRService()
