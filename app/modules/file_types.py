"""Supports Modules 1-2 — which upload formats DocBlendAI accepts.

Responsibility: map a file's extension to its kind, so upload validation,
format detection, and extraction agree on what a file is.

- PDF: typed (text layer), scanned, or handwritten; detected per file
- IMAGE (photos, scans): scanned or handwritten, never typed (no text layer)
- DOCX, PPTX, TXT: always typed; their text is read directly

Uses from schemas.py: nothing.
"""

from enum import Enum
from pathlib import Path


class FileKind(str, Enum):
    PDF = "pdf"
    IMAGE = "image"
    DOCX = "docx"
    PPTX = "pptx"
    TXT = "txt"


EXTENSIONS: dict[str, FileKind] = {
    ".pdf": FileKind.PDF,
    ".jpg": FileKind.IMAGE,
    ".jpeg": FileKind.IMAGE,
    ".png": FileKind.IMAGE,
    ".tif": FileKind.IMAGE,
    ".tiff": FileKind.IMAGE,
    ".bmp": FileKind.IMAGE,
    ".webp": FileKind.IMAGE,
    ".docx": FileKind.DOCX,
    ".pptx": FileKind.PPTX,
    ".txt": FileKind.TXT,
}

# Kinds whose text is read directly: never OCR/HTR, so always FormatType.TYPED.
TEXT_KINDS = {FileKind.DOCX, FileKind.PPTX, FileKind.TXT}


class UnreadableFileError(ValueError):
    """The file is damaged, or its content does not match its extension."""


def kind_of(path: str | Path) -> FileKind | None:
    """The file's kind from its extension, or None if unsupported."""
    return EXTENSIONS.get(Path(path).suffix.lower())


def supported_extensions() -> str:
    return ", ".join(sorted(EXTENSIONS))
