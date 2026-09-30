"""Chat: a question plus the recent turns the page sends along (nothing about the chat is saved)."""

from types import SimpleNamespace

import pytest

from app.modules import llm_answer, retrieval
from tests.test_multipage import PAGES, _upload


@pytest.fixture
def fake_llm(monkeypatch):
    """Answers "Answer N." and records every prompt sent."""
    fake = SimpleNamespace(prompts=[])

    def _generate(prompt: str) -> str:
        fake.prompts.append(prompt)
        return f"Answer {len(fake.prompts)}."

    monkeypatch.setattr(llm_answer, "_generate", _generate)
    return fake


def _chat(client, question: str, history=()):
    return client.post("/chat", json={"question_text": question, "history": list(history)})


def test_chat_answers_with_sources(client, pdf_file, fake_llm) -> None:
    _upload(client, pdf_file(PAGES))
    reply = _chat(client, "What does lexical analysis convert?").json()
    assert reply["answer"]["answer_text"] == "Answer 1."
    assert reply["sources"][0]["page"] == 3
    assert "Earlier in this conversation" not in fake_llm.prompts[0]


def test_follow_up_uses_the_turns_sent_with_it(client, pdf_file, fake_llm) -> None:
    _upload(client, pdf_file(PAGES))
    history = [{"question": "What does lexical analysis convert?", "answer": "Characters into tokens."}]
    _chat(client, "Explain that in more detail", history)
    assert "Q: What does lexical analysis convert?\nA: Characters into tokens." in fake_llm.prompts[0]


def test_follow_up_also_searches_with_the_previous_question(client, pdf_file, fake_llm, monkeypatch) -> None:
    _upload(client, pdf_file(PAGES))
    searched = []
    original = retrieval.retrieve
    monkeypatch.setattr(retrieval, "retrieve", lambda q, k=5, *rest: searched.append(q.question_text) or original(q, k, *rest))
    _chat(client, "And the parser?", [{"question": "What does lexical analysis convert?", "answer": "Tokens."}])
    assert searched == ["And the parser?", "What does lexical analysis convert?\nAnd the parser?"]


def test_only_the_last_few_turns_are_used(client, pdf_file, fake_llm) -> None:
    _upload(client, pdf_file(PAGES))
    history = [{"question": f"Question {n}?", "answer": f"Reply {n}."} for n in range(1, 7)]
    _chat(client, "One more?", history)
    assert "Question 3?" not in fake_llm.prompts[0] and "Question 4?" in fake_llm.prompts[0]


def test_empty_question_is_rejected(client) -> None:
    assert _chat(client, "").status_code == 422


def test_no_chat_history_endpoints(client) -> None:
    assert client.get("/conversations").status_code == 404
