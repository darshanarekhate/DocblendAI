"""Pipeline tests, grouped by module (1-7).

OCR/HTR/calibration tests live in test_ocr_htr_confidence.py; content types,
confidence-aware ranking, and reliability tiers in test_content_type_reliability.py.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors

from app.config import settings
from app.db.models import AnswerORM, DocumentORM, QueryORM, RetrievalResultORM
from app.models.schemas import (
    Answer,
    Document,
    FormatType,
    Query,
    RecognizedChunk,
    ReliabilityLabel,
    RetrievalResult,
)
from app.modules import (
    chunker,
    embedder,
    format_detection,
    llm_answer,
    reliability,
    retrieval,
    text_parser,
    vector_store,
)
from tests.conftest import TYPED_PAGE, fake_embed

# --- Module 1: Document Upload ---------------------------------------------


def test_module1_upload_saves_document(client, pdf_file, db_session_factory) -> None:
    path = pdf_file([TYPED_PAGE, TYPED_PAGE])

    with path.open("rb") as f:
        resp = client.post("/upload", files={"file": ("notes.pdf", f, "application/pdf")})

    assert resp.status_code == 201
    doc = Document.model_validate(resp.json())
    assert doc.format_type is FormatType.TYPED
    assert doc.page_count == 2
    assert Path(doc.file_path).is_file()
    assert Path(doc.file_path).name == f"{doc.doc_id}_notes.pdf"  # original name kept for the UI
    with db_session_factory() as db:
        assert db.get(DocumentORM, doc.doc_id) is not None


def test_module1_rejects_unsupported_file_type(client) -> None:
    resp = client.post("/upload", files={"file": ("setup.exe", b"MZ\x90\x00", "application/octet-stream")})
    assert resp.status_code == 415
    assert ".pdf" in resp.json()["detail"] and ".png" in resp.json()["detail"]


def test_module1_rejects_corrupt_pdf_and_cleans_up(client, tmp_path) -> None:
    resp = client.post("/upload", files={"file": ("broken.pdf", b"not a pdf", "application/pdf")})
    assert resp.status_code == 422
    assert not any((tmp_path / "uploads").iterdir())


def test_module1_rejects_pdf_with_no_readable_text(client, pdf_file, tmp_path, fake_ocr, fake_htr) -> None:
    # No text layer, and neither OCR nor HTR finds anything (e.g. blank pages).
    with pdf_file(["", ""]).open("rb") as f:
        resp = client.post("/upload", files={"file": ("blank.pdf", f, "application/pdf")})
    assert resp.status_code == 422
    assert "No readable text" in resp.json()["detail"]
    assert not any((tmp_path / "uploads").iterdir())


def test_outdated_database_fails_fast_with_instructions(tmp_path, monkeypatch) -> None:
    from sqlalchemy import create_engine, text

    from app.db import database

    old = create_engine(f"sqlite:///{(tmp_path / 'old.db').as_posix()}")
    with old.begin() as conn:  # the answers table as it was before step 3 (no query_id)
        conn.execute(text("CREATE TABLE answers (answer_id VARCHAR PRIMARY KEY, answer_text TEXT, reliability_label VARCHAR)"))
    monkeypatch.setattr(database, "engine", old)

    with pytest.raises(database.SchemaOutdatedError, match=r"answers\.query_id.*delete"):
        database.init_db()


def test_current_database_passes_schema_check(tmp_path, monkeypatch) -> None:
    from sqlalchemy import create_engine

    from app.db import database

    monkeypatch.setattr(database, "engine", create_engine(f"sqlite:///{(tmp_path / 'new.db').as_posix()}"))
    database.init_db()  # creates everything; must not raise


# --- Module 2: Format Detection & Text Extraction ---------------------------


def test_module2_detects_typed_pdf(pdf_file) -> None:
    assert format_detection.detect_format(str(pdf_file([TYPED_PAGE]))) is FormatType.TYPED


def test_module2_pdf_without_text_is_not_typed(pdf_file, fake_ocr, fake_htr) -> None:
    assert format_detection.detect_format(str(pdf_file(["", "", TYPED_PAGE]))) is not FormatType.TYPED


def test_module2_text_parser_returns_page_text_with_full_confidence(pdf_file) -> None:
    pages = text_parser.parse_pdf(str(pdf_file([TYPED_PAGE, ""])))

    assert len(pages) == 2
    assert "Retrieval-Augmented Generation" in pages[0][0]
    assert pages[1][0] == ""
    assert all(conf == 1.0 for _, conf in pages)


def test_module2_extract_routes_typed_to_text_parser(pdf_file) -> None:
    path = str(pdf_file([TYPED_PAGE]))
    doc = Document(doc_id="d", file_path=path, format_type=FormatType.TYPED, page_count=1)
    assert format_detection.extract(doc) == text_parser.parse_pdf(path)


# --- Chunker (supports Modules 4/5) ----------------------------------------


def test_chunker_short_page_is_one_chunk() -> None:
    chunks = chunker.chunk_pages("doc", [("A short page.", 0.9)])

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "doc:0"
    assert chunks[0].text == "A short page."
    assert chunks[0].raw_conf == 0.9


def test_chunker_respects_size_and_overlap() -> None:
    words = [f"word{i}" for i in range(400)]
    chunks = chunker.chunk_pages("doc", [(" ".join(words), 1.0)], chunk_size=200, overlap=50)

    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    # Consecutive chunks share words (the overlap) ...
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.text.split()[0] in prev.text.split()
    # ... and together they cover every word, in order.
    assert set(words) == {w for c in chunks for w in c.text.split()}


def test_chunker_skips_empty_pages_and_numbers_ids_across_pages() -> None:
    chunks = chunker.chunk_pages("doc", [("Page one.", 1.0), ("", 1.0), ("Page three.", 0.5)])

    assert [c.chunk_id for c in chunks] == ["doc:0", "doc:1"]
    assert [c.raw_conf for c in chunks] == [1.0, 0.5]


def test_chunker_preserves_line_breaks() -> None:
    chunks = chunker.chunk_pages("doc", [("Name  Marks\nAsha  91\nRavi  84", 1.0)])
    assert chunks[0].text == "Name  Marks\nAsha  91\nRavi  84"


def test_chunker_rejects_overlap_not_smaller_than_size() -> None:
    with pytest.raises(ValueError):
        chunker.chunk_pages("doc", [("text", 1.0)], chunk_size=100, overlap=100)


# --- Module 5: embeddings + vector store (build step 2) ---------------------


class _FakeGenaiClient:
    """Records embed_content batches instead of calling Gemini."""

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple[int, str]] = []
        self.fail = fail
        self.models = self

    def embed_content(self, model, contents, config):
        if self.fail:
            raise genai_errors.APIError(500, {"error": {"message": "boom"}})
        self.calls.append((len(contents), config.task_type))
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1, 0.2]) for _ in contents])


@pytest.fixture
def fake_genai(monkeypatch: pytest.MonkeyPatch):
    def _install(fail: bool = False) -> _FakeGenaiClient:
        fake = _FakeGenaiClient(fail)
        monkeypatch.setattr(settings, "gemini_api_key", "test-key")
        monkeypatch.setattr(embedder, "_client", lambda api_key: fake)
        return fake

    return _install


def test_module5_embedder_splits_into_batches_of_100(fake_genai) -> None:
    fake = fake_genai()
    vectors = embedder.embed_texts(["text"] * 250)

    assert len(vectors) == 250
    assert fake.calls == [(100, "RETRIEVAL_DOCUMENT"), (100, "RETRIEVAL_DOCUMENT"), (50, "RETRIEVAL_DOCUMENT")]


def test_module5_embed_query_uses_query_task_type(fake_genai) -> None:
    fake = fake_genai()
    assert embedder.embed_query("What is RAG?") == [0.1, 0.2]
    assert fake.calls == [(1, "RETRIEVAL_QUERY")]


def test_module5_embedder_requires_api_key(monkeypatch) -> None:
    monkeypatch.setattr(settings, "gemini_api_key", "")
    with pytest.raises(embedder.EmbeddingError, match="GEMINI_API_KEY"):
        embedder.embed_texts(["text"])


def test_module5_embedder_wraps_api_errors(fake_genai) -> None:
    fake_genai(fail=True)
    with pytest.raises(embedder.EmbeddingError):
        embedder.embed_texts(["text"])


def _embedded(doc_id: str, texts: list[str], raw_conf: float = 1.0) -> list[RecognizedChunk]:
    return [
        RecognizedChunk(chunk_id=f"{doc_id}:{i}", text=t, raw_conf=raw_conf, vector=v)
        for i, (t, v) in enumerate(zip(texts, fake_embed(texts, "RETRIEVAL_DOCUMENT")))
    ]


def test_module5_vector_store_search_ranks_best_match_first(chroma) -> None:
    vector_store.add_chunks("doc", _embedded("doc", ["transformers use attention", "ocr reads scanned pages"], 0.8))

    results = vector_store.search(fake_embed(["ocr reads scanned pages"], "RETRIEVAL_QUERY")[0], top_k=2)

    assert [c.chunk_id for c, _ in results] == ["doc:1", "doc:0"]
    best, similarity = results[0]
    assert similarity == pytest.approx(1.0, abs=1e-4)  # identical text -> cosine similarity 1
    assert results[1][1] < similarity
    assert best.text == "ocr reads scanned pages"
    assert best.raw_conf == 0.8
    assert best.calibrated_conf is None


def test_module5_vector_store_upsert_does_not_duplicate(chroma) -> None:
    chunks = _embedded("doc", ["one chunk"])
    vector_store.add_chunks("doc", chunks)
    vector_store.add_chunks("doc", chunks)

    assert len(vector_store.search(chunks[0].vector, top_k=10)) == 1


def test_module5_vector_store_requires_vectors(chroma) -> None:
    with pytest.raises(ValueError):
        vector_store.add_chunks("doc", [RecognizedChunk(chunk_id="doc:0", text="t", raw_conf=1.0)])


def test_module5_vector_store_delete_document(chroma) -> None:
    vector_store.add_chunks("a", _embedded("a", ["alpha text"]))
    vector_store.add_chunks("b", _embedded("b", ["beta text"]))

    vector_store.delete_document("a")

    assert [c.chunk_id for c, _ in vector_store.search(fake_embed(["text"], "")[0], top_k=10)] == ["b:0"]


def test_module5_upload_stores_chunks_in_vector_store(client, pdf_file) -> None:
    with pdf_file([TYPED_PAGE]).open("rb") as f:
        doc_id = client.post("/upload", files={"file": ("notes.pdf", f, "application/pdf")}).json()["doc_id"]

    results = vector_store.search(fake_embed(["retriever language model"], "")[0], top_k=5)

    assert results and all(c.chunk_id.startswith(f"{doc_id}:") for c, _ in results)
    assert "Retrieval-Augmented Generation" in results[0][0].text


def test_module5_upload_rolls_back_when_embedding_fails(client, pdf_file, tmp_path, monkeypatch, db_session_factory) -> None:
    def _fail(texts, task_type):
        raise embedder.EmbeddingError("down")

    monkeypatch.setattr(embedder, "_embed", _fail)

    with pdf_file([TYPED_PAGE]).open("rb") as f:
        resp = client.post("/upload", files={"file": ("notes.pdf", f, "application/pdf")})

    assert resp.status_code == 502
    assert not any((tmp_path / "uploads").iterdir())
    with db_session_factory() as db:
        assert db.query(DocumentORM).count() == 0


# --- Module 5: retrieval ---------------------------------------------------
# (Module 4 and confidence-aware ranking: tests/test_content_type_reliability.py)


def test_module5_retrieve_ranks_by_similarity_at_equal_confidence(chroma, fake_embeddings) -> None:
    vector_store.add_chunks("doc", _embedded("doc", ["transformers use attention", "tesseract ocr confidence"], 1.0))

    hits = retrieval.retrieve(_query("tesseract ocr confidence"), top_k=2)

    assert [c.chunk_id for c, _ in hits] == ["doc:1", "doc:0"]
    result = hits[0][1]
    assert result.combined_score == pytest.approx(retrieval.combined_score(result.similarity, 1.0))
    assert result.confidence == 1.0


def test_module5_retrieve_on_empty_store_returns_nothing(chroma, fake_embeddings) -> None:
    assert retrieval.retrieve(_query("anything"), top_k=5) == []


# --- Module 6: reliability tiers (fully legible evidence) ---------------------


def _result(score: float) -> RetrievalResult:
    return RetrievalResult(chunk_id="c", similarity=score, confidence=1.0, combined_score=score)


@pytest.mark.parametrize(
    ("best", "expected"),
    [
        (0.80, ReliabilityLabel.CERTAIN),
        (reliability.CERTAIN_MIN_SIM, ReliabilityLabel.CERTAIN),
        (0.62, ReliabilityLabel.MODERATE),
        (0.50, ReliabilityLabel.UNCERTAIN),
    ],
)
def test_module6_tier_from_best_score(best, expected) -> None:
    assert reliability.classify_reliability([_result(0.3), _result(best)]) is expected


def test_module6_no_results_is_uncertain() -> None:
    assert reliability.classify_reliability([]) is ReliabilityLabel.UNCERTAIN


def test_module6_worse_of() -> None:
    assert reliability.worse_of(ReliabilityLabel.CERTAIN, ReliabilityLabel.UNCERTAIN) is ReliabilityLabel.UNCERTAIN
    assert reliability.worse_of(ReliabilityLabel.UNREADABLE, ReliabilityLabel.UNCERTAIN) is ReliabilityLabel.UNREADABLE


# --- Module 7: answer generation --------------------------------------------


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch):
    """Replace the Gemini call; set .reply to control the answer, read .prompts to inspect calls."""
    fake = SimpleNamespace(reply="RAG grounds answers in retrieved passages.", prompts=[])

    def _generate(prompt: str) -> str:
        fake.prompts.append(prompt)
        return fake.reply

    monkeypatch.setattr(llm_answer, "_generate", _generate)
    return fake


def _query(text: str, query_id: str = "q1") -> Query:
    return Query(query_id=query_id, question_text=text, user_id="u1")


def _chunk(text: str) -> RecognizedChunk:
    return RecognizedChunk(chunk_id="doc:0", text=text, raw_conf=1.0)


def test_module7_prompt_contains_question_and_numbered_passages(fake_llm) -> None:
    answer = llm_answer.generate_answer(
        _query("What is RAG?"), [_chunk("RAG retrieves passages."), _chunk("OCR reads scans.")], ReliabilityLabel.CERTAIN
    )

    assert answer.answer_text == fake_llm.reply
    assert answer.reliability_label is ReliabilityLabel.CERTAIN
    prompt = fake_llm.prompts[0]
    assert "[1] (paragraph)\nRAG retrieves passages." in prompt
    assert "[2] (paragraph)\nOCR reads scans." in prompt
    assert "Question: What is RAG?" in prompt


def test_module7_not_found_lowers_label_to_uncertain(fake_llm) -> None:
    fake_llm.reply = llm_answer.NOT_FOUND
    answer = llm_answer.generate_answer(_query("Who founded it?"), [_chunk("Unrelated.")], ReliabilityLabel.CERTAIN)

    assert answer.answer_text == llm_answer.NOT_FOUND_ANSWER
    assert answer.reliability_label is ReliabilityLabel.UNCERTAIN


def test_module7_no_chunks_skips_llm(fake_llm) -> None:
    answer = llm_answer.generate_answer(_query("Anything?"), [], ReliabilityLabel.UNCERTAIN)

    assert answer.answer_text == llm_answer.NO_DOCUMENTS_ANSWER
    assert fake_llm.prompts == []


class _FlakyGenaiClient:
    def __init__(self, codes: list[int]) -> None:
        self.codes = list(codes)  # error codes to raise, in order, before succeeding
        self.calls = 0
        self.models = self

    def generate_content(self, model, contents, config):
        self.calls += 1
        if self.codes:
            raise genai_errors.APIError(self.codes.pop(0), {"error": {"message": "err"}})
        return SimpleNamespace(text=" An answer. ")


def _install_flaky(monkeypatch, codes: list[int]) -> _FlakyGenaiClient:
    fake = _FlakyGenaiClient(codes)
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(llm_answer, "_client", lambda api_key: fake)
    monkeypatch.setattr(llm_answer.time, "sleep", lambda s: None)
    return fake


def test_module7_retries_overloaded_model(monkeypatch) -> None:
    fake = _install_flaky(monkeypatch, [503, 429])
    assert llm_answer._generate("prompt") == "An answer."
    assert fake.calls == 3


def test_module7_gives_up_after_max_attempts(monkeypatch) -> None:
    _install_flaky(monkeypatch, [503, 503, 503])
    with pytest.raises(llm_answer.LLMError):
        llm_answer._generate("prompt")


def test_module7_does_not_retry_client_errors(monkeypatch) -> None:
    fake = _install_flaky(monkeypatch, [400])
    with pytest.raises(llm_answer.LLMError):
        llm_answer._generate("prompt")
    assert fake.calls == 1


# --- /ask end to end (Modules 5-7) -----------------------------------------


def _upload(client, pdf_file) -> str:
    with pdf_file([TYPED_PAGE]).open("rb") as f:
        return client.post("/upload", files={"file": ("notes.pdf", f, "application/pdf")}).json()["doc_id"]


def _ask(client, question: str, query_id: str = "q1"):
    return client.post("/ask", json={"query_id": query_id, "question_text": question, "user_id": "u1"})


def test_ask_answers_and_stores_everything(client, pdf_file, fake_llm, db_session_factory) -> None:
    doc_id = _upload(client, pdf_file)

    resp = _ask(client, "What does the retriever select?")

    assert resp.status_code == 200
    answer = Answer.model_validate(resp.json())
    assert answer.answer_text == fake_llm.reply
    assert "Retrieval-Augmented Generation" in fake_llm.prompts[0]
    with db_session_factory() as db:
        assert db.get(QueryORM, "q1").question_text == "What does the retriever select?"
        assert db.get(AnswerORM, answer.answer_id).query_id == "q1"
        rows = db.query(RetrievalResultORM).filter_by(query_id="q1").all()
        assert rows and all(r.chunk_id.startswith(f"{doc_id}:") for r in rows)

    fetched = client.get(f"/answer/{answer.answer_id}")
    assert fetched.status_code == 200
    assert Answer.model_validate(fetched.json()) == answer


def test_ask_rejects_duplicate_query_id(client, pdf_file, fake_llm) -> None:
    _upload(client, pdf_file)
    assert _ask(client, "First?").status_code == 200
    assert _ask(client, "Second?").status_code == 409


def test_ask_returns_502_and_stores_nothing_when_llm_fails(client, pdf_file, monkeypatch, db_session_factory) -> None:
    _upload(client, pdf_file)

    def _fail(prompt):
        raise llm_answer.LLMError("down")

    monkeypatch.setattr(llm_answer, "_generate", _fail)

    assert _ask(client, "What is RAG?").status_code == 502
    with db_session_factory() as db:
        assert db.query(QueryORM).count() == 0


def test_get_unknown_answer_is_404(client) -> None:
    assert client.get("/answer/does-not-exist").status_code == 404
    assert client.get("/answer/does-not-exist/sources").status_code == 404


def test_answer_sources_show_text_and_scores(client, pdf_file, fake_llm) -> None:
    doc_id = _upload(client, pdf_file)
    answer = _ask(client, "What does the retriever select?").json()

    sources = client.get(f"/answer/{answer['answer_id']}/sources").json()

    assert sources, "the answer should list the chunks it was built from"
    top = sources[0]
    assert top["doc_id"] == doc_id
    assert "Retrieval-Augmented Generation" in top["text"]
    assert top["content_type"] == "paragraph"
    assert top["confidence"] == 1.0
    scores = [s["combined_score"] for s in sources]
    assert scores == sorted(scores, reverse=True)


def test_list_documents(client, pdf_file) -> None:
    assert client.get("/documents").json() == []
    doc_id = _upload(client, pdf_file)
    [doc] = client.get("/documents").json()
    assert doc["doc_id"] == doc_id and doc["format_type"] == "typed"


def test_frontend_is_served(client) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "DocBlendAI" in resp.text and "/conversations" in resp.text
