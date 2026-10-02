"""Spelling correction ("Did you mean"): fuzzy matching, the validated Gemini rewrite, and POST /ask/suggest."""

from collections.abc import Iterator
from types import SimpleNamespace

import pytest

from app.config import settings
from app.models.schemas import RecognizedChunk
from app.modules import llm_answer, spelling, vector_store
from tests.conftest import fake_embed

RAG_TEXT = (
    "Retrieval-Augmented Generation combines a retriever with a language model. "
    "The retriever selects relevant passages and the model answers from them."
)
DB_TEXT = "Normalization removes redundancy. Second normal form removes partial dependencies."


@pytest.fixture(autouse=True)
def fresh_vocabulary() -> Iterator[None]:
    """Each test has its own ChromaDB, so never reuse a vocabulary cached by another test."""
    spelling.invalidate()
    yield
    spelling.invalidate()


@pytest.fixture
def no_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "gemini_api_key", "")


@pytest.fixture
def fake_gemini(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Gemini configured but faked: set .reply (or .error) and read .calls."""
    fake = SimpleNamespace(reply="NO_CHANGE", error=None, calls=[])

    def _generate(prompt: str, system_instruction: str = llm_answer.SYSTEM_INSTRUCTION) -> str:
        fake.calls.append((prompt, system_instruction))
        if fake.error:
            raise fake.error
        return fake.reply

    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(llm_answer, "_generate", _generate)
    return fake


def _store(doc_id: str, text: str) -> None:
    [vector] = fake_embed([text], "")
    chunk = RecognizedChunk(chunk_id=f"{doc_id}:0", text=text, raw_conf=1.0, vector=vector)
    vector_store.add_chunks(doc_id, [chunk])


# --- fuzzy matching ------------------------------------------------------------


def test_fuzzy_corrects_misspelled_document_terms(chroma, no_gemini) -> None:
    _store("rag", RAG_TEXT)

    result = spelling.suggest("What does the retrival step selcts?")

    assert result.suggestion == "What does the retrieval step selects?"
    assert result.corrections == [("retrival", "retrieval"), ("selcts", "selects")]
    assert result.source == "fuzzy"


def test_fuzzy_keeps_capitalisation_and_punctuation(chroma, no_gemini) -> None:
    _store("rag", RAG_TEXT)
    assert spelling.suggest("Retreiver?").suggestion == "Retriever?"
    assert spelling.suggest("RETREIVER, please").suggestion == "RETRIEVER, please"


def test_nothing_to_correct_gives_no_suggestion(chroma, no_gemini) -> None:
    _store("rag", RAG_TEXT)
    result = spelling.suggest("What does the retriever select?")
    assert (result.suggestion, result.corrections, result.source) == (None, [], None)


def test_skips_stopwords_numbers_short_words_and_inflections(chroma, no_gemini) -> None:
    _store("rag", RAG_TEXT)
    # "explain" (stopword list), "2nf" (digit), "lmo" (short), "passage" (inflection of "passages"),
    # "zebra" (no close document word): none of them is a typo of a document term.
    assert spelling.suggest("Explain 2NF lmo passage zebra").suggestion is None


def test_vocabulary_is_limited_to_selected_documents(chroma, no_gemini) -> None:
    _store("rag", RAG_TEXT)
    _store("db", DB_TEXT)
    assert spelling.suggest("normalisation", ["db"]).suggestion == "normalization"
    assert spelling.suggest("normalisation", ["rag"]).suggestion is None
    assert "normalization" in spelling.vocabulary()


def test_no_documents_gives_no_suggestion(chroma, no_gemini) -> None:
    assert spelling.suggest("retrival").suggestion is None


def test_vocabulary_is_cached_until_invalidated(chroma, no_gemini, monkeypatch) -> None:
    _store("rag", RAG_TEXT)
    calls = []
    original = vector_store.get_texts
    monkeypatch.setattr(vector_store, "get_texts", lambda doc_ids=None: calls.append(doc_ids) or original(doc_ids))

    spelling.suggest("retrival")
    spelling.suggest("selcts")
    assert len(calls) == 1

    spelling.invalidate()
    spelling.suggest("retrival")
    assert len(calls) == 2


# --- Gemini rewrite --------------------------------------------------------------


def test_gemini_rewrite_is_used_when_it_only_uses_document_words(chroma, fake_gemini) -> None:
    _store("rag", RAG_TEXT)
    fake_gemini.reply = "What does the retriever choose from the passages?"

    result = spelling.suggest("What does the retrevr choose from the pasages?")

    assert result.source == "gemini"
    assert result.suggestion == "What does the retriever choose from the passages?"
    assert result.corrections == [("retrevr", "retriever"), ("pasages", "passages")]
    prompt, instruction = fake_gemini.calls[0]
    assert instruction == spelling.REWRITE_INSTRUCTION
    assert "retrevr" in prompt and "retriever" in prompt


def test_gemini_rewrite_with_outside_words_is_discarded(chroma, fake_gemini) -> None:
    _store("rag", RAG_TEXT)
    fake_gemini.reply = "What does the transformer encoder select?"

    result = spelling.suggest("What does the retrival step select?")

    assert result.source == "fuzzy"
    assert result.suggestion == "What does the retrieval step select?"


def test_gemini_error_falls_back_to_fuzzy(chroma, fake_gemini) -> None:
    _store("rag", RAG_TEXT)
    fake_gemini.error = llm_answer.LLMError("quota")

    result = spelling.suggest("retrival")

    assert (result.suggestion, result.source) == ("retrieval", "fuzzy")


def test_gemini_no_change_keeps_fuzzy_result(chroma, fake_gemini) -> None:
    _store("rag", RAG_TEXT)
    fake_gemini.reply = "NO_CHANGE"
    assert spelling.suggest("retrival").source == "fuzzy"


def test_gemini_is_not_called_when_every_word_is_known(chroma, fake_gemini) -> None:
    _store("rag", RAG_TEXT)
    assert spelling.suggest("What does the retriever select?").suggestion is None
    assert fake_gemini.calls == []


def test_gemini_is_not_called_without_api_key(chroma, no_gemini, monkeypatch) -> None:
    _store("rag", RAG_TEXT)
    monkeypatch.setattr(llm_answer, "_generate", lambda *a, **k: pytest.fail("Gemini called without a key"))
    assert spelling.suggest("retrival").source == "fuzzy"


# --- POST /ask/suggest -------------------------------------------------------------


def _upload(client, pdf_file, text: str, name: str) -> str:
    with pdf_file([text], name).open("rb") as f:
        return client.post("/upload", files={"file": (name, f, "application/pdf")}).json()["doc_id"]


def test_suggest_endpoint(client, pdf_file, no_gemini) -> None:
    doc_id = _upload(client, pdf_file, RAG_TEXT, "rag.pdf")

    resp = client.post("/ask/suggest", json={"question_text": "What is retrival?", "doc_ids": [doc_id]})

    assert resp.status_code == 200
    assert resp.json() == {
        "original": "What is retrival?",
        "suggestion": "What is retrieval?",
        "corrections": [{"word": "retrival", "replacement": "retrieval"}],
        "source": "fuzzy",
    }


def test_suggest_endpoint_without_corrections(client, pdf_file, no_gemini) -> None:
    _upload(client, pdf_file, RAG_TEXT, "rag.pdf")
    body = client.post("/ask/suggest", json={"question_text": "What is retrieval?"}).json()
    assert (body["suggestion"], body["corrections"], body["source"]) == (None, [], None)


def test_suggest_endpoint_sees_new_uploads_and_deletions(client, pdf_file, no_gemini) -> None:
    _upload(client, pdf_file, RAG_TEXT, "rag.pdf")
    ask = lambda: client.post("/ask/suggest", json={"question_text": "normalisation"}).json()["suggestion"]  # noqa: E731
    assert ask() is None

    db_id = _upload(client, pdf_file, DB_TEXT, "db.pdf")
    assert ask() == "normalization"

    client.delete(f"/documents/{db_id}")
    assert ask() is None


def test_suggest_endpoint_unknown_doc_is_404(client, no_gemini) -> None:
    resp = client.post("/ask/suggest", json={"question_text": "retrival", "doc_ids": ["nope"]})
    assert resp.status_code == 404


def test_suggest_endpoint_rejects_empty_question(client) -> None:
    assert client.post("/ask/suggest", json={"question_text": ""}).status_code == 422
