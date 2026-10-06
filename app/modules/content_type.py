"""Module 4 — Content-Type Identification.

Responsibility: label each chunk as table, paragraph, or image content.

Rule-based on the chunk's line structure (works the same for parsed, OCR, and
HTR text):
- table: most lines look like rows: several cells split by wide gaps, tabs,
  or pipes; number-heavy lines (marks, results, measurements); or short
  label-value lines ending in a number.
- image: text that belongs to a figure: a "Figure N"/"Fig. N" caption, or
  mostly short fragmentary lines (diagram and axis labels).
- paragraph: everything else (running prose).

Uses from schemas.py: ContentType, RecognizedChunk.
"""

import re

from app.models.schemas import ContentType, RecognizedChunk

MIN_LINES = 3  # fewer lines than this is too little structure to call a table/figure
TABLE_LINE_RATIO = 0.6
IMAGE_SHORT_LINE_RATIO = 0.6
SHORT_LINE_MAX_WORDS = 3
LABEL_VALUE_MAX_TOKENS = 4

_CELL_SPLIT = re.compile(r"\s{2,}|\t|\|")
_NUMBER = re.compile(r"^[-+(]?\d[\d.,:/%)]*$")
_CAPTION = re.compile(r"^\s*(figure|fig\.?)\s*\d+", re.IGNORECASE)
_SENTENCE_END = re.compile(r"[.!?](\s|$)")


def _is_table_row(line: str) -> bool:
    cells = [c for c in _CELL_SPLIT.split(line.strip()) if c]
    if len(cells) >= 3:
        return True
    # The parser's table format (text_parser.py): "Table: A | B" header lines and
    # "A: x | B: y" rows; a two-column table has only two cells per row.
    if line.startswith("Table: ") or (len(cells) >= 2 and "|" in line and all(":" in c for c in cells)):
        return True
    tokens = line.split()
    numbers = sum(bool(_NUMBER.match(t)) for t in tokens)
    if numbers >= 2 and numbers / len(tokens) >= 0.4:
        return True
    # Label-value row ("Asha 91"): parsed PDFs collapse column gaps to one space,
    # so a two-column table only shows up as short lines ending in a number.
    return 2 <= len(tokens) <= LABEL_VALUE_MAX_TOKENS and bool(_NUMBER.match(tokens[-1]))


def classify(text: str) -> ContentType:
    """Return the content type of one piece of text."""
    lines = [line for line in text.splitlines() if line.strip()]
    if lines and _CAPTION.match(lines[0]):
        return ContentType.IMAGE
    if len(lines) < MIN_LINES:
        return ContentType.PARAGRAPH

    if sum(_is_table_row(line) for line in lines) / len(lines) >= TABLE_LINE_RATIO:
        return ContentType.TABLE

    short = sum(len(line.split()) <= SHORT_LINE_MAX_WORDS for line in lines)
    if short / len(lines) >= IMAGE_SHORT_LINE_RATIO and not _SENTENCE_END.search(text):
        return ContentType.IMAGE
    return ContentType.PARAGRAPH


def label_chunks(chunks: list[RecognizedChunk]) -> list[RecognizedChunk]:
    """Return copies of the chunks with content_type set."""
    return [c.model_copy(update={"content_type": classify(c.text)}) for c in chunks]
