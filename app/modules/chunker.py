"""Supports Modules 4/5 — text chunking.

Responsibility: split extracted page text into overlapping chunks, carrying
each page's raw_conf onto its chunks.

Uses from schemas.py: RecognizedChunk.
"""

import re

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


def chunk_pages_numbered(
    doc_id: str,
    pages: list[tuple[str, float]],
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[tuple[int, RecognizedChunk]]:
    """chunk_pages, with each chunk's 1-based page number (for source citations)."""
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    out: list[tuple[int, RecognizedChunk]] = []
    for page_no, (page_text, raw_conf) in enumerate(pages, 1):
        for text in _split(page_text, chunk_size, overlap):
            out.append((page_no, RecognizedChunk(chunk_id=f"{doc_id}:{len(out)}", text=text, raw_conf=raw_conf)))
    return out


def _split(text: str, chunk_size: int, overlap: int) -> list[str]:
    tokens = _TOKEN.findall(text)
    pieces: list[str] = []
    start = 0
    while start < len(tokens):
        end, length = start, 0
        while end < len(tokens) and (end == start or length + len(tokens[end].rstrip()) <= chunk_size):
            length += len(tokens[end])
            end += 1
        pieces.append("".join(tokens[start:end]).strip())
        if end == len(tokens):
            break

        # Step back over whole words until ~overlap chars repeat, always advancing at least one word.
        next_start, back = end, 0
        while next_start - 1 > start and back + len(tokens[next_start - 1]) <= overlap:
            next_start -= 1
            back += len(tokens[next_start])
        start = next_start
    return pieces
