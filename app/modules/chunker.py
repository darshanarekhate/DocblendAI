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
        for start, end, is_table in _segments(page_text):
            piece_list = (
                _split_table(page_text[start:end], chunk_size) if is_table
                else _split(page_text[start:end], chunk_size, overlap)
            )
            for text, first, last in piece_list:
                first, last = first + start, last + start
                location = Location(page_no, bisect_right(line_starts, first), bisect_right(line_starts, last))
                out.append((location, RecognizedChunk(chunk_id=f"{doc_id}:{len(out)}", text=text, raw_conf=raw_conf)))
    return out


# --- tables get chunks of their own ---------------------------------------------------------
# text_parser.py writes tables as "Table: A | B" followed by rows "A: x | B: y" (or plain
# "x | y" rows). Mixed into a prose chunk, a table matches table questions ("customers in
# February?") poorly, and a split table loses its header. So each table, with the short line
# just before it (its caption, e.g. "Dataset contains:"), becomes its own chunk(s), split by
# rows with the header line repeated at the top of each part.

TABLE_HEADER = "Table: "
CAPTION_MAX_CHARS = 120


def _is_table_line(line: str) -> bool:
    line = line.strip()
    return line.startswith(TABLE_HEADER) or " | " in line


def _segments(text: str) -> list[tuple[int, int, bool]]:
    """Split a page into (start, end, is_table) character spans, in order."""
    lines = [(m.start(), m.end(), m.group()) for m in re.finditer(r"[^\n]*\n?", text) if m.group()]
    spans: list[tuple[int, int, bool]] = []
    i = 0
    while i < len(lines):
        if not _is_table_line(lines[i][2]):
            i += 1
            continue
        j = i
        while j < len(lines) and (_is_table_line(lines[j][2]) or not lines[j][2].strip()):
            j += 1
        start = lines[i][0]
        # Take the caption line along (the last non-empty line before the table), if short.
        k = i - 1
        while k >= 0 and not lines[k][2].strip():
            k -= 1
        if k >= 0 and len(lines[k][2].strip()) <= CAPTION_MAX_CHARS and not _is_table_line(lines[k][2]):
            start = lines[k][0]
        spans.append((start, lines[j - 1][1], True))
        i = j
    out, pos = [], 0
    for start, end, _ in spans:
        if text[pos:start].strip():
            out.append((pos, start, False))
        out.append((start, end, True))
        pos = end
    if text[pos:].strip():
        out.append((pos, len(text), False))
    return out


def _split_table(text: str, chunk_size: int) -> list[tuple[str, int, int]]:
    """Split a table span by rows; every part after the first starts with the header line again."""
    rows = [(m.start(), m.group().rstrip()) for m in re.finditer(r"[^\n]+", text) if m.group().strip()]
    header = next((row for _, row in rows if row.strip().startswith(TABLE_HEADER)), None)
    pieces: list[tuple[str, int, int]] = []
    current: list[tuple[int, str]] = []

    def flush() -> None:
        lines = [row for _, row in current]
        if pieces and header and header not in lines:
            lines.insert(0, header)
        first, (last_start, last_row) = current[0][0], current[-1]
        pieces.append(("\n".join(line.strip() for line in lines), first, last_start + len(last_row) - 1))

    for start, row in rows:
        size = sum(len(r) + 1 for _, r in current) + len(row) + (len(header) + 1 if pieces and header else 0)
        if current and size > chunk_size:
            flush()
            current = []
        current.append((start, row))
    if current:
        flush()
    return pieces


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
