"""Document preview: page images for PDFs/images, parsed text for Office/text files."""

import io
from pathlib import Path

from docx import Document as new_docx
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
    assert client.get("/documents/nope/view").status_code == 404


# --- View: the whole document on one page, opened in a new tab ---


def test_view_pdf_shows_every_page_image(client, pdf_file) -> None:
    doc_id = _upload(client, pdf_file(PAGES, name="unit_notes.pdf")).json()["doc_id"]
    resp = client.get(f"/documents/{doc_id}/view")
    assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/html")
    assert "<title>unit_notes.pdf</title>" in resp.text
    assert all(f'src="/documents/{doc_id}/pages/{n}"' in resp.text for n in (1, 2, 3))


def test_view_docx_shows_text_escaped(client, tmp_path) -> None:
    doc = new_docx()
    doc.add_paragraph("Use <b> tags & compare x < y")
    doc.save(tmp_path / "notes.docx")
    doc_id = _upload(client, tmp_path / "notes.docx").json()["doc_id"]
    text = client.get(f"/documents/{doc_id}/view").text
    assert "Use &lt;b&gt; tags &amp; compare x &lt; y" in text and "<b>" not in text


def test_frontend_links_to_view(client) -> None:
    assert "/view`" in client.get("/").text and "href: viewUrl" in client.get("/").text


def test_local_pages_opened_outside_the_server_may_call_the_api(client) -> None:
    """index.html opened as a file (origin "null") or from a localhost live preview works;
    other websites are not allowed to call the local API."""
    for origin in ("null", "http://127.0.0.1:5500", "http://localhost:3000"):
        resp = client.get("/health", headers={"Origin": origin})
        assert resp.headers.get("access-control-allow-origin") == origin
    assert "access-control-allow-origin" not in client.get("/health", headers={"Origin": "https://evil.example"}).headers


def test_pages_render_correctly_when_requested_at_the_same_time(pdf_file) -> None:
    """The page loads every page image at once; pdfium fails under overlapping calls without the lock."""
    from concurrent.futures import ThreadPoolExecutor

    from app.modules.pdf_render import render_pages

    path = str(pdf_file([f"Page {n} text." for n in range(1, 9)]))

    def page(n):
        *_, image = render_pages(path, 110, max_pages=n, first_page=n - 1)
        return image.size

    with ThreadPoolExecutor(max_workers=8) as pool:
        sizes = list(pool.map(page, [n for n in range(1, 9) for _ in range(4)]))
    assert len(sizes) == 32 and len(set(sizes)) == 1


def test_preview_page_renders_only_that_page(client, pdf_file, monkeypatch) -> None:
    from app.modules import pdf_render

    doc_id = _upload(client, pdf_file(PAGES)).json()["doc_id"]
    rendered = []
    original = pdf_render.pdfium.PdfDocument.__getitem__
    monkeypatch.setattr(pdf_render.pdfium.PdfDocument, "__getitem__", lambda self, i: rendered.append(i) or original(self, i))
    assert client.get(f"/documents/{doc_id}/pages/3").status_code == 200
    assert rendered == [2]
