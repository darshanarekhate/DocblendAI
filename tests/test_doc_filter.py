"""Document filter: retrieval and POST /ask limited to the selected documents (doc_ids)."""

from types import SimpleNamespace

import pytest

from app.models.schemas import Query, RecognizedChunk
from app.modules import llm_answer, retrieval, vector_store
from tests.conftest import fake_embed

OS_PAGE = "Process scheduling decides which process runs next on the processor."
DB_PAGE = "Second normal form removes partial dependencies from a relation."


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Replace the Gemini call; read .prompts to see which passages reached the model."""
    fake = SimpleNamespace(prompts=[])

    def _generate(prompt: str, system_instruction: str = "") -> str:
        fake.prompts.append(prompt)
        return "An answer."

    monkeypatch.setattr(llm_answer, "_generate", _generate)
    return fake


def _store(doc_id: str, text: str) -> None:
    [vector] = fake_embed([text], "")
    chunk = RecognizedChunk(chunk_id=f"{doc_id}:0", text=text, raw_conf=1.0, calibrated_conf=1.0, vector=vector)
    vector_store.add_chunks(doc_id, [chunk])


def _upload(client, pdf_file, text: str, name: str) -> str:
    with pdf_file([text], name).open("rb") as f:
        resp = client.post("/upload", files={"file": (name, f, "application/pdf")})
    assert resp.status_code == 201
    return resp.json()["doc_id"]


def _ask(client, question: str, query_id: str = "q1", **extra):
    return client.post("/ask", json={"query_id": query_id, "question_text": question, "user_id": "u1", **extra})


def test_vector_store_search_only_returns_selected_documents(chroma) -> None:
    _store("os", OS_PAGE)
    _store("db", DB_PAGE)
    [query_vector] = fake_embed(["process scheduling"], "")

    assert {c.chunk_id for c, _ in vector_store.search(query_vector, 10)} == {"os:0", "db:0"}
    assert [c.chunk_id for c, _ in vector_store.search(query_vector, 10, ["db"])] == ["db:0"]
    assert {c.chunk_id for c, _ in vector_store.search(query_vector, 10, ["db", "os"])} == {"os:0", "db:0"}


def test_no_filter_means_all_documents_and_empty_finds_nothing(chroma) -> None:
    # vector_store: None = all documents, [] = none selected (main's /chat sends what is ticked).
    # /ask treats an empty doc_ids body as "all" before it gets here (test_ask_without_doc_ids_still_uses_all_documents).
    _store("os", OS_PAGE)
    _store("db", DB_PAGE)
    [query_vector] = fake_embed(["anything"], "")
    assert len(vector_store.search(query_vector, 10, None)) == 2
    assert vector_store.search(query_vector, 10, []) == []


def test_get_texts_by_document(chroma) -> None:
    _store("os", OS_PAGE)
    _store("db", DB_PAGE)
    assert vector_store.get_texts(["os"]) == [OS_PAGE]
    assert sorted(vector_store.get_texts()) == sorted([OS_PAGE, DB_PAGE])


def test_retrieve_respects_doc_filter(chroma, fake_embeddings) -> None:
    _store("os", OS_PAGE)
    _store("db", DB_PAGE)
    query = Query(query_id="q", question_text="process scheduling", user_id="u")

    assert retrieval.retrieve(query, 5)[0][0].chunk_id == "os:0"
    assert [c.chunk_id for c, _ in retrieval.retrieve(query, 5, ["db"])] == ["db:0"]


def test_ask_uses_only_selected_documents(client, pdf_file, fake_llm) -> None:
    _upload(client, pdf_file, OS_PAGE, "os.pdf")
    db_id = _upload(client, pdf_file, DB_PAGE, "db.pdf")

    resp = _ask(client, "What does process scheduling decide?", doc_ids=[db_id])

    assert resp.status_code == 200
    assert "Second normal form" in fake_llm.prompts[0]
    assert "Process scheduling" not in fake_llm.prompts[0]
    sources = client.get(f"/answer/{resp.json()['answer_id']}/sources").json()
    assert {s["doc_id"] for s in sources} == {db_id}


def test_ask_without_doc_ids_still_uses_all_documents(client, pdf_file, fake_llm) -> None:
    _upload(client, pdf_file, OS_PAGE, "os.pdf")
    _upload(client, pdf_file, DB_PAGE, "db.pdf")

    assert _ask(client, "process scheduling and normal form").status_code == 200
    assert _ask(client, "process scheduling", query_id="q2", doc_ids=None).status_code == 200
    assert _ask(client, "normal form", query_id="q3", doc_ids=[]).status_code == 200
    assert "Process scheduling" in fake_llm.prompts[0] and "Second normal form" in fake_llm.prompts[0]


def test_ask_with_unknown_doc_id_is_404_and_stores_nothing(client, pdf_file, fake_llm) -> None:
    known = _upload(client, pdf_file, OS_PAGE, "os.pdf")

    resp = _ask(client, "process scheduling", doc_ids=[known, "nope"])

    assert resp.status_code == 404
    assert "nope" in resp.json()["detail"]
    assert fake_llm.prompts == []
    # The query_id was not used up, so the same question can be asked again.
    assert _ask(client, "process scheduling", doc_ids=[known]).status_code == 200


def test_stored_query_has_no_doc_ids_field(client, pdf_file, fake_llm, db_session_factory) -> None:
    from app.db.models import QueryORM

    doc_id = _upload(client, pdf_file, OS_PAGE, "os.pdf")
    assert _ask(client, "process scheduling", doc_ids=[doc_id]).status_code == 200
    with db_session_factory() as db:
        row = db.get(QueryORM, "q1")
    assert (row.question_text, row.user_id) == ("process scheduling", "u1")
