"""Question-answering routes — entry point to Modules 5, 6, 7.

Responsibility: accept a user question, run retrieval -> reliability
classification -> LLM answer generation, and let clients fetch a stored answer.

Uses from schemas.py: Query, Answer.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.config import settings
from app.db.database import get_db
from app.db.models import AnswerORM, QueryORM, RetrievalResultORM
from app.models.schemas import Answer, ContentType, Query, RetrievalResult
from app.modules import embedder, llm_answer, reliability, retrieval, vector_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


# Plain `def`: embedding and generation are blocking network calls, so FastAPI runs this in its threadpool.
@router.post("/ask", response_model=Answer)
def ask(query: Query, db: Session = Depends(get_db)) -> Answer:
    if db.get(QueryORM, query.query_id) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, f"query_id {query.query_id!r} already exists")

    try:
        hits = retrieval.retrieve(query, settings.top_k)
        results = [r for _, r in hits]
        label = reliability.classify_reliability(results)
        answer = llm_answer.generate_answer(query, [c for c, _ in hits], label)
    except (embedder.EmbeddingError, llm_answer.LLMError) as e:
        logger.error("Answering %s failed: %s", query.query_id, e)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "The answer service (LLM) failed; try again later")

    db.add(QueryORM(**query.model_dump()))
    db.add(AnswerORM(**answer.model_dump(), query_id=query.query_id))
    db.add_all(RetrievalResultORM(**r.model_dump(), query_id=query.query_id) for r in results)
    db.commit()

    logger.info("Answered %s: %s from %d chunks", query.query_id, answer.reliability_label.value, len(results))
    return answer


@router.get("/answer/{answer_id}", response_model=Answer)
def get_answer(answer_id: str, db: Session = Depends(get_db)) -> Answer:
    row = db.get(AnswerORM, answer_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Answer not found")
    return Answer.model_validate(row)


class AnswerSource(RetrievalResult):
    """A retrieved chunk behind an answer: its scores plus the text itself.

    API-only view (not a synopsis entity): lets users see why an answer got
    its reliability label.
    """

    doc_id: str
    page: int | None = None  # 1-based; None for chunks stored before page numbers were recorded
    content_type: ContentType
    text: str


@router.get("/answer/{answer_id}/sources", response_model=list[AnswerSource])
def get_answer_sources(answer_id: str, db: Session = Depends(get_db)) -> list[AnswerSource]:
    """Retrieved chunks for an answer, best combined_score first."""
    row = db.get(AnswerORM, answer_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Answer not found")
    results = (
        db.query(RetrievalResultORM)
        .filter_by(query_id=row.query_id)
        .order_by(RetrievalResultORM.combined_score.desc())
        .all()
    )
    chunks = vector_store.get_chunks([r.chunk_id for r in results])
    pages = vector_store.get_pages([r.chunk_id for r in results])
    sources = []
    for r in results:
        chunk = chunks.get(r.chunk_id)
        if chunk is None:  # document deleted since the question was asked
            continue
        sources.append(
            AnswerSource(
                **RetrievalResult.model_validate(r).model_dump(),
                doc_id=r.chunk_id.rsplit(":", 1)[0],
                page=pages.get(r.chunk_id),
                content_type=chunk.content_type,
                text=chunk.text,
            )
        )
    return sources
