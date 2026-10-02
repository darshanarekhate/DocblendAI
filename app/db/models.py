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


class RunVersionORM(Base):
    """History of an Experience Center run's text: every extraction, refinement and edit.

    NOT a synopsis entity. A run's current result lives in ocr_runs.result_json; each change
    (re-extraction, accepted/discarded LLM refinement, manual edit, restore) is recorded here so
    the user can compare versions or roll back. kind:
    - "extraction" / "edit" / "restore": data_json is the full result object after that change
    - "refinement": data_json is an LLM proposal (per-line original/refined text, diff, decision);
      status pending | accepted | partial | discarded
    """

    __tablename__ = "run_versions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(String, index=True)
    seq: Mapped[int] = mapped_column(Integer)  # 1, 2, ... per run
    kind: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="")
    label: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[str] = mapped_column(String)  # ISO 8601 UTC
    data_json: Mapped[str] = mapped_column(Text, default="{}")


class DocumentRunORM(Base):
    """Links a QA document (documents.doc_id) to its Experience Center run (ocr_runs.id).

    NOT a synopsis entity: a link table, so neither the documents table (synopsis Document)
    nor ocr_runs needs a new column. One upload from either page creates both, one run each.
    """

    __tablename__ = "document_runs"

    doc_id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(String, unique=True, index=True)
