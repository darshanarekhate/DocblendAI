"""Shared Pydantic schemas for all 7 DocBlendAI modules.

Single source of truth for the entities in the synopsis Problem Analysis table
(see CLAUDE.md). Field names here must match that table exactly: every module,
router, and ORM model imports from this file.

Entities: Document, RecognizedChunk, Query, RetrievalResult, Answer.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class FormatType(str, Enum):
    """Document format detected by Module 2; decides the extraction path."""

    TYPED = "typed"  # direct parsing
    SCANNED = "scanned"  # OCR (pytesseract)
    HANDWRITTEN = "handwritten"  # HTR (TrOCR)


class ContentType(str, Enum):
    """Chunk content type assigned by Module 4."""

    TABLE = "table"
    PARAGRAPH = "paragraph"
    IMAGE = "image"


class ReliabilityLabel(str, Enum):
    """Four-tier answer reliability assigned by Module 6."""

    CERTAIN = "Certain"
    MODERATE = "Moderate"
    UNCERTAIN = "Uncertain"
    UNREADABLE = "Unreadable"


class _Schema(BaseModel):
    # Allows building schemas straight from SQLAlchemy rows (Model.model_validate(row)).
    model_config = ConfigDict(from_attributes=True)


class Document(_Schema):
    """An uploaded file (Module 1), classified by Module 2."""

    doc_id: str
    file_path: str
    format_type: FormatType
    page_count: int = Field(ge=0)


class RecognizedChunk(_Schema):
    """A unit of extracted text, stored in ChromaDB (Modules 2-5).

    calibrated_conf and vector are filled by later pipeline stages
    (Module 3 and the embedder), so they start as None.
    """

    chunk_id: str
    text: str
    content_type: ContentType = ContentType.PARAGRAPH
    raw_conf: float = Field(ge=0.0, le=1.0)
    calibrated_conf: float | None = Field(default=None, ge=0.0, le=1.0)
    vector: list[float] | None = None


class Query(_Schema):
    """A user question (input to Modules 5-7)."""

    query_id: str
    question_text: str = Field(min_length=1)
    user_id: str


class RetrievalResult(_Schema):
    """One retrieved chunk with its scores (Module 5)."""

    chunk_id: str
    similarity: float
    confidence: float
    combined_score: float


class Answer(_Schema):
    """Generated answer (Module 7) with its reliability tier (Module 6)."""

    answer_id: str
    answer_text: str
    reliability_label: ReliabilityLabel
