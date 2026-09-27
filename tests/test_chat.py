"""Chat: conversations with saved history and follow-up questions."""

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


def _new_chat(client, **body) -> str:
    resp = client.post("/conversations", json=body or None)
    assert resp.status_code == 201
    return resp.json()["conversation_id"]


def _say(client, chat_id: str, text: str):
    return client.post(f"/conversations/{chat_id}/messages", json={"question_text": text})


def test_new_chat_is_empty_and_listed(client) -> None:
    chat_id = _new_chat(client)
    [summary] = client.get("/conversations").json()
    assert summary["conversation_id"] == chat_id and summary["title"] == "New chat" and summary["turn_count"] == 0
    assert client.get(f"/conversations/{chat_id}").json()["turns"] == []


def test_message_is_answered_saved_and_titles_the_chat(client, pdf_file, fake_llm) -> None:
    _upload(client, pdf_file(PAGES))
    chat_id = _new_chat(client)

    turn = _say(client, chat_id, "What does lexical analysis convert?").json()

    assert turn["answer"]["answer_text"] == "Answer 1."
    assert turn["sources"] and turn["sources"][0]["page"] == 3
    saved = client.get(f"/conversations/{chat_id}").json()
    assert saved["title"] == "What does lexical analysis convert?"
    assert [t["question_text"] for t in saved["turns"]] == ["What does lexical analysis convert?"]


def test_follow_up_carries_earlier_turns(client, pdf_file, fake_llm) -> None:
    _upload(client, pdf_file(PAGES))
    chat_id = _new_chat(client)
    _say(client, chat_id, "What does lexical analysis convert?")
    _say(client, chat_id, "Explain that in more detail")

    first, second = fake_llm.prompts
    assert "Earlier in this conversation" not in first
    assert "Q: What does lexical analysis convert?\nA: Answer 1." in second
    turns = client.get(f"/conversations/{chat_id}").json()["turns"]
    assert [t["answer"]["answer_text"] for t in turns] == ["Answer 1.", "Answer 2."]


def test_follow_up_also_searches_with_the_previous_question(client, pdf_file, fake_llm, monkeypatch) -> None:
    _upload(client, pdf_file(PAGES))
    searched = []
    original = retrieval.retrieve
    monkeypatch.setattr(retrieval, "retrieve", lambda q, k=5: searched.append(q.question_text) or original(q, k))
    chat_id = _new_chat(client)
    _say(client, chat_id, "What does lexical analysis convert?")
    _say(client, chat_id, "And the parser?")
    assert searched[-2:] == ["And the parser?", "What does lexical analysis convert?\nAnd the parser?"]


def test_chats_are_kept_apart(client, pdf_file, fake_llm) -> None:
    _upload(client, pdf_file(PAGES))
    a, b = _new_chat(client), _new_chat(client)
    _say(client, a, "What does lexical analysis convert?")
    _say(client, b, "How many layers does the OSI model have?")
    assert "Earlier in this conversation" not in fake_llm.prompts[1]  # b's first question ignores a
    assert len(client.get(f"/conversations/{a}").json()["turns"]) == 1


def test_most_recently_used_chat_is_listed_first(client, pdf_file, fake_llm) -> None:
    _upload(client, pdf_file(PAGES))
    old, new = _new_chat(client), _new_chat(client)
    _say(client, old, "What does lexical analysis convert?")
    assert [c["conversation_id"] for c in client.get("/conversations").json()] == [old, new]


def test_rename_and_delete(client, pdf_file, fake_llm) -> None:
    _upload(client, pdf_file(PAGES))
    chat_id = _new_chat(client)
    turn = _say(client, chat_id, "What does lexical analysis convert?").json()

    assert client.patch(f"/conversations/{chat_id}", json={"title": "Compilers"}).json()["title"] == "Compilers"
    assert client.delete(f"/conversations/{chat_id}").status_code == 204

    assert client.get("/conversations").json() == []
    assert client.get(f"/conversations/{chat_id}").status_code == 404
    assert client.get(f"/answer/{turn['answer']['answer_id']}").status_code == 404  # its answers went too


def test_unknown_chat_and_empty_message(client) -> None:
    assert _say(client, "nope", "hi").status_code == 404
    chat_id = _new_chat(client)
    assert _say(client, chat_id, "").status_code == 422


def test_long_first_question_gives_a_short_title(client, pdf_file, fake_llm) -> None:
    _upload(client, pdf_file(PAGES))
    chat_id = _new_chat(client)
    _say(client, chat_id, "Explain " + "in great detail " * 10)
    title = client.get(f"/conversations/{chat_id}").json()["title"]
    assert len(title) <= 60 and title.endswith("…")
