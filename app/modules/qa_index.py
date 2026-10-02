"""Supports Modules 3-5 — turn an extracted document (Experience Center result) into QA chunks.

Responsibility: one place that decides what the QA page retrieves for a document, from the
same lines /studio shows (including reviewed and LLM-refined text):

- Module 2 text: a page's exact extracted text ("text", typed and classic pages) as long as
  none of its lines was edited; otherwise its lines (or PP-StructureV3 blocks) in reading order.
- Module 3 confidence: the char-weighted mean of the lines' raw / calibrated confidence. Lines
  whose text came from an LLM never raise it: they count at most reliability.LLM_TEXT_MAX_CONF,
  and a chunk containing any of them is capped there too (so its answers are at most Moderate).
- Module 4 content type: from the layout when PP-StructureV3 found it (table blocks -> table,
  figure/chart/image blocks -> image, everything else -> paragraph), otherwise the rule-based
  content_type.classify on each chunk.

Each chunk keeps its citation (chunker.Location: page and range of the page's non-empty lines,
counted across the page's segments in reading order). index_document embeds the chunks and
replaces the document's chunks in ChromaDB.

Uses from schemas.py: ContentType, RecognizedChunk.
"""

from __future__ import annotations

import logging
import re
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any

from app.models.schemas import ContentType, RecognizedChunk
from app.modules import chunker, content_type, embedder, spelling, vector_store
from app.modules.chunker import Location
from app.modules.reliability import LLM_SOURCES, LLM_TEXT_MAX_CONF
from app.modules.text_parser import CELL_SEPARATOR

logger = logging.getLogger(__name__)

TABLE_BLOCKS = frozenset({"table"})
IMAGE_BLOCKS = frozenset({"image", "figure", "chart", "seal", "figure_title", "chart_title", "vision_footnote"})
CHUNK_SIZE, CHUNK_OVERLAP = 1000, 200  # chunker's defaults (keep chunks comparable across paths)


@dataclass
class Segment:
    page_no: int  # 1-based, for source citations
    text: str
    raw_conf: float
    calibrated_conf: float
    content_type: ContentType | None  # None: classify each chunk by its text
    llm: bool = False  # contains LLM-written text


def _line_conf(line: dict[str, Any], key: str) -> float:
    value = float(line.get(key, 0.0) or 0.0)
    return min(value, LLM_TEXT_MAX_CONF) if line.get("source") in LLM_SOURCES else value


def _confidences(lines: list[dict[str, Any]], fallback: tuple[float, float]) -> tuple[float, float, bool]:
    chars = sum(len(l["text"]) for l in lines)
    llm = any(l.get("source") in LLM_SOURCES for l in lines)
    if not chars:
        return fallback[0], fallback[1], llm
    raw = sum(_line_conf(l, "raw_confidence") * len(l["text"]) for l in lines) / chars
    cal = sum(_line_conf(l, "calibrated_confidence") * len(l["text"]) for l in lines) / chars
    if llm:
        raw, cal = min(raw, LLM_TEXT_MAX_CONF), min(cal, LLM_TEXT_MAX_CONF)
    return raw, cal, llm


def _center_in(line: dict[str, Any], box: list[float] | None) -> bool:
    if not box or not line.get("box"):
        return False
    x0, y0, x1, y1 = line["box"]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def _rows_text(lines: list[dict[str, Any]]) -> str:
    """Lines of a table area grouped into rows by vertical overlap, cells two spaces apart."""
    rows: list[list[dict[str, Any]]] = []
    for line in sorted(lines, key=lambda l: (l["box"][1] + l["box"][3]) / 2):
        y0, y1 = line["box"][1], line["box"][3]
        if rows:
            top = min(l["box"][1] for l in rows[-1])
            bottom = max(l["box"][3] for l in rows[-1])
            if min(bottom, y1) - max(top, y0) > 0.5 * min(bottom - top, y1 - y0):
                rows[-1].append(line)
                continue
        rows.append([line])
    return "\n".join(CELL_SEPARATOR.join(l["text"] for l in sorted(r, key=lambda l: l["box"][0])) for r in rows)


def _block_segments(page: dict[str, Any], page_no: int, fallback: tuple[float, float]) -> list[Segment]:
    lines = page.get("lines") or []
    used: set[int] = set()
    tables = iter(page.get("tables") or [])
    segments: list[Segment] = []
    prose: list[tuple[str, list[dict[str, Any]]]] = []  # titles/paragraphs merged into one segment

    def flush() -> None:
        if prose:
            text = "\n".join(t for t, _ in prose if t)
            raw, cal, llm = _confidences([l for _, ls in prose for l in ls], fallback)
            segments.append(Segment(page_no, text, raw, cal, ContentType.PARAGRAPH, llm))
            prose.clear()

    for block in page.get("blocks") or []:
        inside = [i for i, l in enumerate(lines) if i not in used and _center_in(l, block.get("box"))]
        used.update(inside)
        block_lines = [lines[i] for i in inside]
        kind = block.get("type") or "text"
        table = next(tables, None) if kind in TABLE_BLOCKS else None
        if kind in TABLE_BLOCKS:
            flush()
            edited = any(l.get("edited") for l in block_lines)
            if block_lines and (edited or not (table and table.get("rows"))):
                text = _rows_text(block_lines)
            else:
                text = "\n".join(CELL_SEPARATOR.join(c for c in row if c) for row in (table or {}).get("rows") or [])
            raw, cal, llm = _confidences(block_lines, fallback)
            if text.strip():
                segments.append(Segment(page_no, text, raw, cal, ContentType.TABLE, llm))
        elif kind in IMAGE_BLOCKS:
            flush()
            text = "\n".join(l["text"] for l in block_lines) or str(block.get("content") or "")
            raw, cal, llm = _confidences(block_lines, fallback)
            if text.strip():
                segments.append(Segment(page_no, text, raw, cal, ContentType.IMAGE, llm))
        else:
            edited = any(l.get("edited") for l in block_lines)
            text = "\n".join(l["text"] for l in block_lines) if (block_lines and (edited or not block.get("content"))) else str(block.get("content") or "")
            prose.append((text.strip(), block_lines))
    rest = [l for i, l in enumerate(lines) if i not in used]
    if rest:
        prose.append(("\n".join(l["text"] for l in rest), rest))
    flush()
    return segments


def segments(result: dict[str, Any]) -> list[Segment]:
    out: list[Segment] = []
    for i, page in enumerate(result.get("pages") or []):
        lines = page.get("lines") or []
        raw, cal, llm = _confidences(lines, (1.0, 1.0))
        exact = page.get("text")
        if exact is not None and not any(l.get("edited") for l in lines):
            if exact.strip():
                out.append(Segment(i + 1, exact, raw, cal, None, llm))
        elif page.get("blocks") and any(b.get("box") for b in page["blocks"]):
            out.extend(_block_segments(page, i + 1, (raw, cal)))
        elif lines:
            out.append(Segment(i + 1, "\n".join(l["text"] for l in lines), raw, cal, None, llm))
        elif (page.get("markdown") or "").strip():
            out.append(Segment(i + 1, page["markdown"], raw, cal, None, llm))
    return out


_LINE_START = re.compile(r"^[^\S\n]*\S", re.MULTILINE)  # a non-empty line (as chunker counts them)


def build_chunks(doc_id: str, result: dict[str, Any]) -> list[tuple[Location, RecognizedChunk]]:
    """(citation, chunk) pairs with raw/calibrated confidence and content type set (not embedded)."""
    out: list[tuple[Location, RecognizedChunk]] = []
    lines_before: dict[int, int] = {}  # page -> non-empty lines in its earlier segments
    for seg in segments(result):
        line_starts = [m.start() for m in _LINE_START.finditer(seg.text)]
        offset = lines_before.get(seg.page_no, 0)
        for text, first, last in chunker._split(seg.text, CHUNK_SIZE, CHUNK_OVERLAP):
            if not text.strip():
                continue
            location = Location(seg.page_no, offset + bisect_right(line_starts, first), offset + bisect_right(line_starts, last))
            chunk = RecognizedChunk(
                chunk_id=f"{doc_id}:{len(out)}", text=text, raw_conf=round(seg.raw_conf, 6),
                calibrated_conf=round(seg.calibrated_conf, 6),
                content_type=seg.content_type or content_type.classify(text),
            )
            out.append((location, chunk))
        lines_before[seg.page_no] = offset + len(line_starts)
    return out


class NoTextError(ValueError):
    """The document has no readable text to index."""


def index_document(doc_id: str, result: dict[str, Any]) -> list[RecognizedChunk]:
    """Embed the document's chunks and replace its chunks in ChromaDB (old ones stay if embedding fails).

    Raises NoTextError, embedder.EmbeddingError.
    """
    numbered = build_chunks(doc_id, result)
    if not numbered:
        raise NoTextError("No readable text found in this file")
    chunks = embedder.embed_chunks([c for _, c in numbered])
    vector_store.delete_document(doc_id)
    vector_store.add_chunks(doc_id, chunks, {c.chunk_id: loc for (loc, _), c in zip(numbered, chunks)})
    spelling.invalidate()
    logger.info("Indexed %s: %d chunks", doc_id, len(chunks))
    return chunks
