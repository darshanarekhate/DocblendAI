"""SQLAlchemy ORM models for the relational entities.

Mirrors the Pydantic schemas in app/models/schemas.py for Document, Query,
Answer, and RetrievalResult. RecognizedChunk is not stored here: it lives in
ChromaDB (Module 5, vector_store.py).
"""

from sqlalchemy import Enum as SAEnum
from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base
from app.models.schemas import FormatType, ReliabilityLabel


class DocumentORM(Base):
    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String, primary_key=True)
    file_path: Mapped[str] = mapped_column(String)
    format_type: Mapped[FormatType] = mapped_column(SAEnum(FormatType), index=True)
    page_count: Mapped[int] = mapped_column(Integer)


class QueryORM(Base):
    __tablename__ = "queries"

    query_id: Mapped[str] = mapped_column(String, primary_key=True)
    question_text: Mapped[str] = mapped_column(Text)
    user_id: Mapped[str] = mapped_column(String, index=True)


class AnswerORM(Base):
    __tablename__ = "answers"

    answer_id: Mapped[str] = mapped_column(String, primary_key=True)
    # Not in the synopsis entity (so not in schemas.py): links the answer to its question.
    query_id: Mapped[str] = mapped_column(ForeignKey("queries.query_id"), index=True)
    answer_text: Mapped[str] = mapped_column(Text)
    reliability_label: Mapped[ReliabilityLabel] = mapped_column(SAEnum(ReliabilityLabel), index=True)


class RetrievalResultORM(Base):
    __tablename__ = "retrieval_results"

    # Surrogate key: the same chunk_id can be retrieved for many queries, so it cannot be the PK.
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Not in the synopsis entity (so not in schemas.py): which question retrieved this chunk.
    query_id: Mapped[str] = mapped_column(ForeignKey("queries.query_id"), index=True)
    chunk_id: Mapped[str] = mapped_column(String, index=True)
    similarity: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    combined_score: Mapped[float] = mapped_column(Float)


class OCRRunORM(Base):
    """Experience Center run history (/studio, /api/results): one row per OCR/parse job.

    NOT a synopsis entity (so not in schemas.py): it stores the PaddleOCR Experience Center's
    result objects (docs/experience_center_contract.md §4) for history, search and review edits.
    Page images live in settings.paddle_runs_dir/<id>/page-<n>.png.
    """

    __tablename__ = "ocr_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    filename: Mapped[str] = mapped_column(String)
    pipeline: Mapped[str] = mapped_column(String, index=True)
    language: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, index=True)  # queued | running | done | error
    created_at: Mapped[str] = mapped_column(String, index=True)  # ISO 8601 UTC, sorts as text
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mean_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)  # mean calibrated
    processing_time_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    full_text: Mapped[str] = mapped_column(Text, default="")  # line texts, searched by ?q=
    result_json: Mapped[str] = mapped_column(Text, default="{}")  # the full result object (§4)
