"""Question-answering routes — entry point to Modules 5, 6, 7.

Responsibility: accept a user question, run retrieval -> reliability
classification -> LLM answer generation, and let clients fetch a stored answer.

Uses from schemas.py: Query, Answer.
"""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import settings
from app.db.database import get_db
from app.db.models import AnswerORM, QueryORM, RetrievalResultORM
from app.models.schemas import Answer, ContentType, Query, RetrievalResult
from app.modules import embedder, llm_answer, reliability, retrieval, vector_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


Turn = tuple[str, str]  # (question, answer) from earlier in a conversation


def _retrieve(query: Query, history: list[Turn] | None):
    """Retrieve for the question; in a conversation, also for (previous question + this one).

    A follow-up like "explain it in more detail" has nothing to match on its own,
    so the previous question is searched with it and the better-scored chunks win.
    A self-contained new question still finds its own chunks through the first search.
    """
    hits = retrieval.retrieve(query, settings.top_k)
    if not history:
        return hits
    follow_up = query.model_copy(update={"question_text": f"{history[-1][0]}\n{query.question_text}"})
    best = {chunk.chunk_id: (chunk, result) for chunk, result in hits}
    for chunk, result in retrieval.retrieve(follow_up, settings.top_k):
        if chunk.chunk_id not in best or result.combined_score > best[chunk.chunk_id][1].combined_score:
            best[chunk.chunk_id] = (chunk, result)
    return sorted(best.values(), key=lambda pair: pair[1].combined_score, reverse=True)[: settings.top_k]


def answer_and_store(query: Query, db: Session, history: list[Turn] | None = None) -> Answer:
    """Modules 5 -> 6 -> 7 for one question, then store Query, Answer, and RetrievalResults.

    Raises HTTPException 409 for a reused query_id and 502 if Gemini fails.
    """
    if db.get(QueryORM, query.query_id) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, f"query_id {query.query_id!r} already exists")

    try:
        hits = _retrieve(query, history)
        results = [r for _, r in hits]
        label = reliability.classify_reliability(results)
        answer = llm_answer.generate_answer(query, [c for c, _ in hits], label, history=history)
    except (embedder.EmbeddingError, llm_answer.LLMError) as e:
        logger.error("Answering %s failed: %s", query.query_id, e)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "The answer service (LLM) failed; try again later")

    db.add(QueryORM(**query.model_dump()))
    db.add(AnswerORM(**answer.model_dump(), query_id=query.query_id))
    db.add_all(RetrievalResultORM(**r.model_dump(), query_id=query.query_id) for r in results)
    db.commit()

    logger.info("Answered %s: %s from %d chunks", query.query_id, answer.reliability_label.value, len(results))
    return answer


# Plain `def`: embedding and generation are blocking network calls, so FastAPI runs this in its threadpool.
@router.post("/ask", response_model=Answer)
def ask(query: Query, db: Session = Depends(get_db)) -> Answer:
    return answer_and_store(query, db)


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
    # Citation: 1-based page and non-empty-line range; None for chunks stored before these were recorded.
    page: int | None = None
    first_line: int | None = None
    last_line: int | None = None
    content_type: ContentType
    text: str


@router.get("/answer/{answer_id}/sources", response_model=list[AnswerSource])
def get_answer_sources(answer_id: str, db: Session = Depends(get_db)) -> list[AnswerSource]:
    """Retrieved chunks for an answer, best combined_score first."""
    row = db.get(AnswerORM, answer_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Answer not found")
    return sources_for_query(row.query_id, db)


def sources_for_query(query_id: str, db: Session) -> list[AnswerSource]:
    """The chunks retrieved for a question, with text and citation, best combined_score first."""
    results = (
        db.query(RetrievalResultORM)
        .filter_by(query_id=query_id)
        .order_by(RetrievalResultORM.combined_score.desc())
        .all()
    )
    chunks = vector_store.get_chunks([r.chunk_id for r in results])
    locations = vector_store.get_locations([r.chunk_id for r in results])
    sources = []
    for r in results:
        chunk = chunks.get(r.chunk_id)
        if chunk is None:  # document deleted since the question was asked
            continue
        sources.append(
            AnswerSource(
                **RetrievalResult.model_validate(r).model_dump(),
                doc_id=r.chunk_id.rsplit(":", 1)[0],
                **locations.get(r.chunk_id, {}),
                content_type=chunk.content_type,
                text=chunk.text,
            )
        )
    return sources


# --- chat: a question plus the recent turns of the current conversation --------------------------
# Nothing about the conversation is saved: the page keeps its own turns and sends the last few,
# so follow-ups ("explain that in more detail") can be understood. Each question is still stored
# as the usual Query / Answer / RetrievalResult rows, exactly like /ask.


class ChatTurn(BaseModel):
    question: str
    answer: str


class ChatRequest(BaseModel):
    question_text: str = Field(min_length=1)
    history: list[ChatTurn] = Field(default_factory=list, description="Earlier turns, oldest first")


class ChatReply(BaseModel):
    answer: Answer
    sources: list[AnswerSource]


CHAT_USER = "web"  # the demo has no accounts
CHAT_CONTEXT_TURNS = 3


@router.post("/chat", response_model=ChatReply)
def chat(body: ChatRequest, db: Session = Depends(get_db)) -> ChatReply:
    query = Query(query_id=uuid.uuid4().hex, question_text=body.question_text.strip(), user_id=CHAT_USER)
    history = [(t.question, t.answer) for t in body.history[-CHAT_CONTEXT_TURNS:]]
    answer = answer_and_store(query, db, history=history)
    return ChatReply(answer=answer, sources=sources_for_query(query.query_id, db))
