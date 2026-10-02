"""Supports Modules 4/5 — text chunking.

Responsibility: split extracted page text into overlapping chunks, carrying
each page's raw_conf onto its chunks.

Uses from schemas.py: RecognizedChunk.
"""

import re
from bisect import bisect_right
from typing import NamedTuple

from app.models.schemas import RecognizedChunk

# A word plus the whitespace after it. Keeping the original whitespace (not
# re-joining with spaces) preserves line breaks that Module 4 needs to spot tables.
_TOKEN = re.compile(r"\S+\s*")


def chunk_pages(
    doc_id: str,
    pages: list[tuple[str, float]],
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[RecognizedChunk]:
    """Turn (page_text, raw_conf) pages into RecognizedChunks with unique chunk_ids.

    Chunks never cross a page boundary, so each chunk keeps its own page's
    raw_conf. Splits happen between words; a single word longer than
    chunk_size becomes its own chunk. chunk_id is "<doc_id>:<n>", stable
    across re-runs so ChromaDB upserts replace rather than duplicate.
    """
    return [chunk for _, chunk in chunk_pages_numbered(doc_id, pages, chunk_size, overlap)]


class Location(NamedTuple):
    """Where a chunk sits in its document, for citations: 1-based page and line range.

    Lines are the page's non-empty text lines (blank separator lines are not
    counted), so they match what a reader counts on the page.
    """

    page: int
    first_line: int
    last_line: int


def chunk_pages_numbered(
    doc_id: str,
    pages: list[tuple[str, float]],
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[tuple[Location, RecognizedChunk]]:
    """chunk_pages, with each chunk's page and line range (for source citations)."""
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    out: list[tuple[Location, RecognizedChunk]] = []
    for page_no, (page_text, raw_conf) in enumerate(pages, 1):
        line_starts = [m.start() for m in re.finditer(r"^[^\S\n]*\S", page_text, re.MULTILINE)]
        for text, first, last in _split(page_text, chunk_size, overlap):
            location = Location(page_no, bisect_right(line_starts, first), bisect_right(line_starts, last))
            out.append((location, RecognizedChunk(chunk_id=f"{doc_id}:{len(out)}", text=text, raw_conf=raw_conf)))
    return out


def _split(text: str, chunk_size: int, overlap: int) -> list[tuple[str, int, int]]:
    """Split into overlapping pieces: (piece, offset of its first char, offset of its last char)."""
    matches = list(_TOKEN.finditer(text))
    tokens = [m.group() for m in matches]
    pieces: list[tuple[str, int, int]] = []
    start = 0
    while start < len(tokens):
        end, length = start, 0
        while end < len(tokens) and (end == start or length + len(tokens[end].rstrip()) <= chunk_size):
            length += len(tokens[end])
            end += 1
        last = matches[end - 1].start() + len(tokens[end - 1].rstrip()) - 1
        pieces.append(("".join(tokens[start:end]).strip(), matches[start].start(), last))
        if end == len(tokens):
            break

        # Step back over whole words until ~overlap chars repeat, always advancing at least one word.
        next_start, back = end, 0
        while next_start - 1 > start and back + len(tokens[next_start - 1]) <= overlap:
            next_start -= 1
            back += len(tokens[next_start])
        start = next_start
    return pieces
