"""Document viewer: GET /documents/{doc_id}/file serves the original upload inline."""

from pathlib import Path

from app.db.models import DocumentORM
from app.models.schemas import FormatType
from tests.conftest import TYPED_PAGE


def _upload(client, name: str, data: bytes, content_type: str) -> dict:
    resp = client.post("/upload", files={"file": (name, data, content_type)})
    assert resp.status_code == 201
    return resp.json()


def test_pdf_is_served_inline_with_original_name(client, pdf_file) -> None:
    data = pdf_file([TYPED_PAGE]).read_bytes()
    doc = _upload(client, "Lecture Notes.pdf", data, "application/pdf")

    resp = client.get(f"/documents/{doc['doc_id']}/file")

    assert resp.status_code == 200
    assert resp.content == data
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.headers["content-disposition"] == 'inline; filename="Lecture_Notes.pdf"'
    assert resp.headers["x-content-type-options"] == "nosniff"


def test_text_file_is_served_as_text(client) -> None:
    doc = _upload(client, "notes.txt", TYPED_PAGE.encode(), "text/plain")

    resp = client.get(f"/documents/{doc['doc_id']}/file")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert resp.text == TYPED_PAGE


def test_unknown_document_is_404(client) -> None:
    assert client.get("/documents/nope/file").status_code == 404


def test_missing_file_is_404(client, pdf_file) -> None:
    doc = _upload(client, "notes.pdf", pdf_file([TYPED_PAGE]).read_bytes(), "application/pdf")
    Path(doc["file_path"]).unlink()

    resp = client.get(f"/documents/{doc['doc_id']}/file")

    assert resp.status_code == 404
    assert "missing" in resp.json()["detail"]


def test_path_outside_upload_folder_is_refused(client, tmp_path, db_session_factory) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("not an upload")
    with db_session_factory() as db:
        db.add(DocumentORM(doc_id="evil", file_path=str(secret), format_type=FormatType.TYPED, page_count=1))
        db.commit()

    assert client.get("/documents/evil/file").status_code == 404


def test_deleted_document_file_is_404(client, pdf_file) -> None:
    doc = _upload(client, "notes.pdf", pdf_file([TYPED_PAGE]).read_bytes(), "application/pdf")
    client.delete(f"/documents/{doc['doc_id']}")
    assert client.get(f"/documents/{doc['doc_id']}/file").status_code == 404
