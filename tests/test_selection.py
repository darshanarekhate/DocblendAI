"""Answering from selected documents only (the checkboxes in the web UI)."""

from tests.test_multipage import _upload, fake_llm  # noqa: F401  (fixture)

NETWORKS = ["TCP provides reliable, ordered delivery of a byte stream between applications."]
DATABASES = ["Second normal form removes partial dependencies of attributes on a composite key."]


def _two_docs(client, pdf_file) -> tuple[str, str]:
    nets = _upload(client, pdf_file(NETWORKS, name="networks.pdf")).json()["doc_id"]
    dbms = _upload(client, pdf_file(DATABASES, name="dbms.pdf")).json()["doc_id"]
    return nets, dbms


def _ask(client, question: str, qid: str, doc_ids: list[str] | None = None) -> set[str]:
    params = [("doc_id", d) for d in doc_ids] if doc_ids is not None else None
    body = {"query_id": qid, "question_text": question, "user_id": "u"}
    answer = client.post("/ask", json=body, params=params).json()
    return {s["doc_id"] for s in client.get(f"/answer/{answer['answer_id']}/sources").json()}


def test_ask_uses_every_document_by_default(client, pdf_file, fake_llm) -> None:  # noqa: F811
    nets, dbms = _two_docs(client, pdf_file)
    assert _ask(client, "What does TCP provide?", "all") == {nets, dbms}


def test_ask_uses_only_the_selected_document(client, pdf_file, fake_llm) -> None:  # noqa: F811
    nets, dbms = _two_docs(client, pdf_file)
    # Even a question about networks is answered from the selected DBMS notes only.
    assert _ask(client, "What does TCP provide?", "only-dbms", [dbms]) == {dbms}
    assert _ask(client, "What does second normal form remove?", "only-nets", [nets]) == {nets}


def test_chat_uses_only_the_selected_document(client, pdf_file, fake_llm) -> None:  # noqa: F811
    nets, dbms = _two_docs(client, pdf_file)
    reply = client.post("/chat", json={"question_text": "What does TCP provide?", "doc_ids": [dbms]}).json()
    assert {s["doc_id"] for s in reply["sources"]} == {dbms}


def test_frontend_sends_the_selection(client) -> None:
    html = client.get("/").text
    assert 'type: "checkbox"' in html and "doc_id=" in html
