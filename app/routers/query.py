"""Question-answering routes — entry point to Modules 5, 6, 7.

Responsibility: accept a user question, run retrieval -> reliability
classification -> LLM answer generation, and let clients fetch a stored answer.
Questions can be limited to some documents (doc_ids), /chat answers follow-up questions
with the recent turns of a conversation, and /ask/suggest offers a spelling-corrected
question ("Did you mean") built from the selected documents' words.

Uses from schemas.py: Query, Answer.
"""

import logging
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi import Query as QueryParam  # schemas.Query is the synopsis entity
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import settings
from app.db.database import get_db
from app.db.models import AnswerORM, DocumentORM, QueryORM, RetrievalResultORM
from app.models.schemas import Answer, ContentType, Query, RetrievalResult
from app.modules import embedder, llm_answer, reliability, retrieval, spelling, vector_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])

DOC_IDS_DESCRIPTION = "Only use these documents (omitted, null, or empty = all documents)."


class AskRequest(Query):
    """POST /ask body: a Query plus an optional document filter.

    API-only wrapper (not a synopsis entity), so schemas.Query stays as the synopsis defines it.
    """

    doc_ids: list[str] | None = Field(default=None, description=DOC_IDS_DESCRIPTION)


def _check_doc_ids(doc_ids: list[str] | None, db: Session) -> list[str] | None:
    """The filter without duplicates, or None for all documents. 404 if any doc_id is unknown."""
    if not doc_ids:
        return None
    wanted = list(dict.fromkeys(doc_ids))
    found = {doc_id for (doc_id,) in db.query(DocumentORM.doc_id).filter(DocumentORM.doc_id.in_(wanted))}
    missing = [d for d in wanted if d not in found]
    if missing:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown document(s): {', '.join(missing)}")
    return wanted


Turn = tuple[str, str]  # (question, answer) from earlier in a conversation


def _retrieve(query: Query, history: list[Turn] | None, doc_ids: list[str] | None = None):
    """Retrieve for the question; in a conversation, also for (previous question + this one).

    A follow-up like "explain it in more detail" has nothing to match on its own,
    so the previous question is searched with it and the better-scored chunks win.
    A self-contained new question still finds its own chunks through the first search.
    """
    hits = retrieval.retrieve(query, settings.top_k, doc_ids)
    if not history:
        return hits
    follow_up = query.model_copy(update={"question_text": f"{history[-1][0]}\n{query.question_text}"})
    best = {chunk.chunk_id: (chunk, result) for chunk, result in hits}
    for chunk, result in retrieval.retrieve(follow_up, settings.top_k, doc_ids):
        if chunk.chunk_id not in best or result.combined_score > best[chunk.chunk_id][1].combined_score:
            best[chunk.chunk_id] = (chunk, result)
    return sorted(best.values(), key=lambda pair: pair[1].combined_score, reverse=True)[: settings.top_k]


def answer_and_store(
    query: Query, db: Session, history: list[Turn] | None = None, doc_ids: list[str] | None = None
) -> Answer:
    """Modules 5 -> 6 -> 7 for one question, then store Query, Answer, and RetrievalResults.

    doc_ids: answer only from these documents (None = all uploaded documents; already checked).

    Raises HTTPException 409 for a reused query_id, and 503 / 429 / 502 with the reason when
    Gemini is not configured, out of quota, or failing.
    """
    if db.get(QueryORM, query.query_id) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, f"query_id {query.query_id!r} already exists")

    try:
        hits = _retrieve(query, history, doc_ids)
        results = [r for _, r in hits]
        label = reliability.classify_reliability(results)
        answer = llm_answer.generate_answer(query, [c for c, _ in hits], label, history=history)
    except (embedder.EmbeddingError, llm_answer.LLMError) as e:
        # The message says what went wrong (no key, quota used up, overloaded) so the UI can show it.
        logger.error("Answering %s failed: %s", query.query_id, e)
        raise HTTPException(getattr(e, "status", status.HTTP_502_BAD_GATEWAY), str(e))

    db.add(QueryORM(**query.model_dump()))
    db.add(AnswerORM(**answer.model_dump(), query_id=query.query_id))
    db.add_all(RetrievalResultORM(**r.model_dump(), query_id=query.query_id) for r in results)
    db.commit()

    logger.info("Answered %s: %s from %d chunks", query.query_id, answer.reliability_label.value, len(results))
    return answer


# Plain `def`: embedding and generation are blocking network calls, so FastAPI runs this in its threadpool.
@router.post("/ask", response_model=Answer)
def ask(
    request: AskRequest,
    doc_id: list[str] | None = QueryParam(
        None, description="Answer only from these documents (repeat the parameter); same as doc_ids in the body"
    ),
    db: Session = Depends(get_db),
) -> Answer:
    """Answer one question. The documents can be chosen in the body (doc_ids) or as ?doc_id=... parameters."""
    query = Query.model_validate(request.model_dump(exclude={"doc_ids"}))
    chosen = request.doc_ids if request.doc_ids is not None else doc_id
    return answer_and_store(query, db, doc_ids=_check_doc_ids(chosen, db))


class SuggestRequest(BaseModel):
    question_text: str = Field(min_length=1)
    doc_ids: list[str] | None = Field(default=None, description=DOC_IDS_DESCRIPTION)


class Correction(BaseModel):
    word: str
    replacement: str


class SpellingSuggestion(BaseModel):
    """API-only view (not a synopsis entity): a "Did you mean" suggestion; suggestion is null if none."""

    original: str
    suggestion: str | None
    corrections: list[Correction]
    source: Literal["fuzzy", "gemini"] | None


# Plain `def`: the optional Gemini rewrite is a blocking network call.
@router.post("/ask/suggest", response_model=SpellingSuggestion)
def suggest_spelling(request: SuggestRequest, db: Session = Depends(get_db)) -> SpellingSuggestion:
    """Spelling-corrected question from the selected documents' words (fuzzy match, then Gemini if configured)."""
    result = spelling.suggest(request.question_text, _check_doc_ids(request.doc_ids, db))
    return SpellingSuggestion(
        original=result.original,
        suggestion=result.suggestion,
        corrections=[Correction(word=w, replacement=r) for w, r in result.corrections],
        source=result.source,
    )


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
    doc_ids: list[str] | None = Field(None, description="Answer only from these documents; default: all")


class ChatReply(BaseModel):
    answer: Answer
    sources: list[AnswerSource]


CHAT_USER = "web"  # the demo has no accounts
CHAT_CONTEXT_TURNS = 3


@router.post("/chat", response_model=ChatReply)
def chat(body: ChatRequest, db: Session = Depends(get_db)) -> ChatReply:
    query = Query(query_id=uuid.uuid4().hex, question_text=body.question_text.strip(), user_id=CHAT_USER)
    history = [(t.question, t.answer) for t in body.history[-CHAT_CONTEXT_TURNS:]]
    # An empty selection means "no documents" here (the page sends what is ticked); unknown ids are 404.
    doc_ids = _check_doc_ids(body.doc_ids, db) if body.doc_ids else body.doc_ids
    answer = answer_and_store(query, db, history=history, doc_ids=doc_ids)
    return ChatReply(answer=answer, sources=sources_for_query(query.query_id, db))
