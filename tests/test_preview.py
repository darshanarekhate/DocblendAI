"""Document preview: page images for PDFs/images, parsed text for Office/text files."""

import io
from pathlib import Path

from PIL import Image

from tests.test_multipage import PAGES, _upload, _write_paged_docx


def test_pdf_preview_lists_pages_and_serves_png(client, pdf_file) -> None:
    doc_id = _upload(client, pdf_file(PAGES)).json()["doc_id"]

    info = client.get(f"/documents/{doc_id}/preview").json()
    assert (info["kind"], info["page_count"], info["text_pages"]) == ("image", 3, None)

    resp = client.get(f"/documents/{doc_id}/pages/3")
    assert resp.status_code == 200 and resp.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(resp.content)).width > 500


def test_page_out_of_range_is_404(client, pdf_file) -> None:
    doc_id = _upload(client, pdf_file(PAGES)).json()["doc_id"]
    assert client.get(f"/documents/{doc_id}/pages/0").status_code == 404
    assert client.get(f"/documents/{doc_id}/pages/4").status_code == 404


def test_image_upload_preview(client, tmp_path: Path, fake_ocr) -> None:
    fake_ocr.text, fake_ocr.conf = "a scanned page", 0.9
    path = tmp_path / "scan.png"
    Image.new("RGB", (400, 300), "white").save(path)
    doc_id = _upload(client, path).json()["doc_id"]
    assert client.get(f"/documents/{doc_id}/preview").json()["kind"] == "image"
    assert client.get(f"/documents/{doc_id}/pages/1").headers["content-type"] == "image/png"


def test_docx_preview_is_text_per_page(client, tmp_path) -> None:
    doc_id = _upload(client, _write_paged_docx(tmp_path / "notes.docx")).json()["doc_id"]
    info = client.get(f"/documents/{doc_id}/preview").json()
    assert info["kind"] == "text" and info["text_pages"] == PAGES
    assert client.get(f"/documents/{doc_id}/pages/1").status_code == 400


def test_preview_unknown_document_is_404(client) -> None:
    assert client.get("/documents/nope/preview").status_code == 404
    assert client.get("/documents/nope/pages/1").status_code == 404


def test_local_pages_opened_outside_the_server_may_call_the_api(client) -> None:
    """index.html opened as a file (origin "null") or from a localhost live preview works;
    other websites are not allowed to call the local API."""
    for origin in ("null", "http://127.0.0.1:5500", "http://localhost:3000"):
        resp = client.get("/health", headers={"Origin": origin})
        assert resp.headers.get("access-control-allow-origin") == origin
    assert "access-control-allow-origin" not in client.get("/health", headers={"Origin": "https://evil.example"}).headers
