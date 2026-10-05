"""Background uploads (POST /upload/background, GET /uploads/pending): the web page gets a reply
at once and the file is read afterwards. TestClient runs background tasks before returning, so
by the time a call returns here the reading has finished."""

from tests.test_multipage import PAGES, _upload


def _background(client, path, **form):
    with path.open("rb") as f:
        return client.post("/upload/background", files={"file": (path.name, f, "application/octet-stream")}, data=form)


def test_background_upload_replies_with_a_reading_job(client, pdf_file) -> None:
    resp = _background(client, pdf_file(PAGES, name="notes.pdf"))
    assert resp.status_code == 202
    job = resp.json()
    assert (job["status"], job["name"]) == ("reading", "notes.pdf")


def test_background_upload_becomes_a_document(client, pdf_file) -> None:
    doc_id = _background(client, pdf_file(PAGES)).json()["doc_id"]
    assert client.get("/uploads/pending").json() == []
    [doc] = client.get("/documents").json()
    assert (doc["doc_id"], doc["page_count"], doc["format_type"]) == (doc_id, 3, "typed")


def test_background_upload_uses_the_chosen_type(client, pdf_file, fake_htr) -> None:
    fake_htr.text, fake_htr.conf = "handwritten notes about graphs", 0.7
    _background(client, pdf_file([""]), format_hint="handwritten")
    assert client.get("/documents").json()[0]["format_type"] == "handwritten"


def test_same_file_again_is_reported_as_duplicate(client, pdf_file) -> None:
    path = pdf_file(PAGES)
    first = _upload(client, path).json()["doc_id"]
    resp = _background(client, path)
    assert resp.status_code == 200
    assert resp.json()["status"] == "duplicate" and resp.json()["doc_id"] == first
    assert len(client.get("/documents").json()) == 1


def test_failed_background_upload_is_listed_then_dismissed(client, tmp_path) -> None:
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"this is not a pdf")
    doc_id = _background(client, broken).json()["doc_id"]

    [job] = client.get("/uploads/pending").json()
    assert job["doc_id"] == doc_id and job["status"] == "failed" and "not a readable" in job["detail"]
    assert client.get("/documents").json() == []

    assert client.delete(f"/uploads/pending/{doc_id}").status_code == 204
    assert client.get("/uploads/pending").json() == []


def test_unsupported_type_is_refused_at_once(client, tmp_path) -> None:
    sheet = tmp_path / "marks.xlsx"
    sheet.write_bytes(b"PK")
    assert _background(client, sheet).status_code == 415


def test_frontend_uploads_in_the_background(client) -> None:
    html = client.get("/").text
    assert "/upload/background" in html and "/uploads/pending" in html
