"""Chat routes — conversations over the uploaded documents, with saved history.

Responsibility: group questions into chats, answer each new question with
the recent turns as context (so follow-ups like "explain that" work), and
list, reopen, rename, and delete past chats.

Every question still goes through the same pipeline as /ask (Modules 5-7)
and is stored as the usual Query / Answer / RetrievalResult rows; a chat
only records which questions belong together and in what order.

API-only views; nothing here is a synopsis entity.
"""

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import AnswerORM, ConversationORM, ConversationTurnORM, QueryORM, RetrievalResultORM
from app.models.schemas import Answer, Query
from app.routers.query import AnswerSource, answer_and_store, sources_for_query

router = APIRouter(tags=["chat"])

NEW_CHAT_TITLE = "New chat"
TITLE_LENGTH = 60
CHAT_USER = "web"  # the demo has no accounts; every chat belongs to the web UI user
CONTEXT_TURNS = 3  # earlier turns sent along with a new question


class ConversationSummary(BaseModel):
    conversation_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    turn_count: int


class ChatTurn(BaseModel):
    query_id: str
    question_text: str
    asked_at: datetime
    answer: Answer
    sources: list[AnswerSource]


class Conversation(ConversationSummary):
    turns: list[ChatTurn]


class NewConversation(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class Rename(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class Message(BaseModel):
    question_text: str = Field(min_length=1)


def _now() -> datetime:
    return datetime.now()


def _get(conversation_id: str, db: Session) -> ConversationORM:
    row = db.get(ConversationORM, conversation_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Chat not found")
    return row


def _turn_rows(conversation_id: str, db: Session) -> list[ConversationTurnORM]:
    return (
        db.query(ConversationTurnORM)
        .filter_by(conversation_id=conversation_id)
        .order_by(ConversationTurnORM.id)
        .all()
    )


def _summary(row: ConversationORM, db: Session) -> ConversationSummary:
    count = db.query(ConversationTurnORM).filter_by(conversation_id=row.conversation_id).count()
    return ConversationSummary(
        conversation_id=row.conversation_id, title=row.title,
        created_at=row.created_at, updated_at=row.updated_at, turn_count=count,
    )


def _chat_turn(turn: ConversationTurnORM, db: Session) -> ChatTurn | None:
    query = db.get(QueryORM, turn.query_id)
    answer = db.query(AnswerORM).filter_by(query_id=turn.query_id).first()
    if query is None or answer is None:
        return None
    return ChatTurn(
        query_id=turn.query_id,
        question_text=query.question_text,
        asked_at=turn.asked_at,
        answer=Answer.model_validate(answer),
        sources=sources_for_query(turn.query_id, db),
    )


@router.get("/conversations", response_model=list[ConversationSummary])
def list_conversations(db: Session = Depends(get_db)) -> list[ConversationSummary]:
    """All chats, most recently active first."""
    rows = db.query(ConversationORM).order_by(ConversationORM.updated_at.desc()).all()
    return [_summary(row, db) for row in rows]


@router.post("/conversations", response_model=ConversationSummary, status_code=status.HTTP_201_CREATED)
def create_conversation(body: NewConversation | None = None, db: Session = Depends(get_db)) -> ConversationSummary:
    now = _now()
    row = ConversationORM(
        conversation_id=uuid.uuid4().hex,
        title=(body.title.strip() if body and body.title and body.title.strip() else NEW_CHAT_TITLE),
        created_at=now,
        updated_at=now,
    )
    db.add(row)
    db.commit()
    return _summary(row, db)


@router.get("/conversations/{conversation_id}", response_model=Conversation)
def get_conversation(conversation_id: str, db: Session = Depends(get_db)) -> Conversation:
    row = _get(conversation_id, db)
    turns = [t for t in (_chat_turn(turn, db) for turn in _turn_rows(conversation_id, db)) if t is not None]
    return Conversation(**_summary(row, db).model_dump(), turns=turns)


@router.patch("/conversations/{conversation_id}", response_model=ConversationSummary)
def rename_conversation(conversation_id: str, body: Rename, db: Session = Depends(get_db)) -> ConversationSummary:
    row = _get(conversation_id, db)
    row.title = body.title.strip()
    db.commit()
    return _summary(row, db)


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: str, db: Session = Depends(get_db)) -> None:
    """Delete a chat and its questions, answers, and retrieval results."""
    row = _get(conversation_id, db)
    query_ids = [t.query_id for t in _turn_rows(conversation_id, db)]
    db.query(ConversationTurnORM).filter_by(conversation_id=conversation_id).delete()
    if query_ids:
        db.query(RetrievalResultORM).filter(RetrievalResultORM.query_id.in_(query_ids)).delete(synchronize_session=False)
        db.query(AnswerORM).filter(AnswerORM.query_id.in_(query_ids)).delete(synchronize_session=False)
        db.query(QueryORM).filter(QueryORM.query_id.in_(query_ids)).delete(synchronize_session=False)
    db.delete(row)
    db.commit()


# Plain `def`: answering makes blocking network calls, so FastAPI runs this in its threadpool.
@router.post("/conversations/{conversation_id}/messages", response_model=ChatTurn, status_code=status.HTTP_201_CREATED)
def send_message(conversation_id: str, body: Message, db: Session = Depends(get_db)) -> ChatTurn:
    """Ask a question in a chat. The last few turns go along so follow-up questions work."""
    row = _get(conversation_id, db)
    history = []
    for turn in _turn_rows(conversation_id, db)[-CONTEXT_TURNS:]:
        query = db.get(QueryORM, turn.query_id)
        answer = db.query(AnswerORM).filter_by(query_id=turn.query_id).first()
        if query is not None and answer is not None:
            history.append((query.question_text, answer.answer_text))

    query = Query(query_id=uuid.uuid4().hex, question_text=body.question_text.strip(), user_id=CHAT_USER)
    answer_and_store(query, db, history=history)

    now = _now()
    db.add(ConversationTurnORM(conversation_id=conversation_id, query_id=query.query_id, asked_at=now))
    if row.title == NEW_CHAT_TITLE:  # name the chat after its first question
        title = query.question_text
        row.title = title if len(title) <= TITLE_LENGTH else title[: TITLE_LENGTH - 1].rstrip() + "…"
    row.updated_at = now
    db.commit()
    return _chat_turn(db.query(ConversationTurnORM).filter_by(query_id=query.query_id).one(), db)
