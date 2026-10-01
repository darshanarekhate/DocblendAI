"""Experience Center engine: normalisers on real recorded PaddleOCR output, and the service with fake models."""

import copy
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

import app.calibration as calibration
from app.config import settings
from app.modules import paddleocr_service as svc
from tests.conftest import make_pdf
from tests.paddle_fakes import STRUCTURE_MARKDOWN, STRUCTURE_RES, TEXT_RES, FakeFactory

# Captured at import, before the autouse guard replaces it, to test the real constructor's kwargs.
REAL_CONSTRUCT = svc._construct_model


@pytest.fixture
def fake_models(monkeypatch: pytest.MonkeyPatch) -> FakeFactory:
    factory = FakeFactory()
    monkeypatch.setattr(svc, "_construct_model", factory)
    monkeypatch.setattr(svc.PaddleOCRService, "installed", staticmethod(lambda: True))
    return factory


# -- normalisers -------------------------------------------------------------------------------


def test_ocr_page_from_real_result():
    page = svc.normalise_ocr_page(TEXT_RES, 0, 1000, 700, "/img")
    assert (page["width"], page["height"], page["image_url"]) == (1000, 700, "/img")
    assert len(page["lines"]) == 15
    first = page["lines"][0]
    assert first["id"] == "p0-l0" and first["text"] == "Calibration Report"
    assert first["box"] == [37, 33, 390, 76]
    assert len(first["polygon"]) == 4
    assert first["raw_confidence"] == pytest.approx(0.9727, abs=1e-4)
    assert first["edited"] is False and first["needs_review"] is False
    assert [w["text"] for w in first["words"]] == ["Calibration", "Report"]
    # punctuation tokens merge into the word before them, boxes are unioned
    words = page["lines"][2]["words"]
    assert words[-1]["text"] == "scores."
    assert words[-1]["box"][0] < words[-1]["box"][2]
    assert page["markdown"].splitlines()[0] == "Calibration Report"
    assert page["blocks"] == [] and page["tables"] == []


def test_polygons_come_from_rec_polys_not_dt_polys():
    res = copy.deepcopy(TEXT_RES)
    res["dt_polys"] = list(reversed(res["dt_polys"]))[:5]  # misaligned, as after score filtering
    page = svc.normalise_ocr_page(res, 2)
    for line, poly in zip(page["lines"], TEXT_RES["rec_polys"]):
        assert line["polygon"] == [[float(x), float(y)] for x, y in poly]
    assert page["lines"][3]["id"] == "p2-l3"


def test_missing_boxes_and_scores_are_tolerated():
    page = svc.normalise_ocr_page({"rec_texts": ["a", "b"], "rec_scores": [1.7], "rec_polys": [[[0, 0], [4, 0], [4, 2], [0, 2]]]}, 0)
    a, b = page["lines"]
    assert a["box"] == [0, 0, 4, 2] and a["raw_confidence"] == 1.0  # clamped
    assert b["polygon"] == [] and b["box"] is None and b["raw_confidence"] == 0.0


def test_merge_word_tokens():
    words = svc.merge_word_tokens(["Hi", " ", "there", ",", " ", "you"], [[0, 0, 2, 1], [2, 0, 3, 1], [3, 0, 8, 1], [8, 0, 9, 1], [9, 0, 10, 1], [10, 0, 13, 1]])
    assert words == [
        {"text": "Hi", "box": [0, 0, 2, 1]},
        {"text": "there,", "box": [3, 0, 9, 1]},
        {"text": "you", "box": [10, 0, 13, 1]},
    ]


def test_structure_page_from_real_result():
    page = svc.normalise_structure_page(STRUCTURE_RES, STRUCTURE_MARKDOWN, 0)
    assert (page["width"], page["height"]) == (1000, 700)
    assert [b["type"] for b in page["blocks"]] == ["paragraph_title", "text", "table"]
    assert page["blocks"][0]["content"] == "Calibration Report"
    assert len(page["lines"]) == 15 and page["lines"][0]["words"] == []
    (table,) = page["tables"]
    assert table["box"] == page["blocks"][2]["box"] == [38, 198, 699, 399]
    assert table["rows"][0] == ["Method", "ECE", "Brier"]
    assert table["rows"][3] == ["Isotonic", "0.021", "0.117"]
    assert len(table["rows"]) == 4 and len(table["cells"]) == 12
    assert table["html"].startswith("<html>")
    assert page["markdown"] == STRUCTURE_MARKDOWN.strip()


def test_structure_markdown_rebuilt_from_blocks_when_missing():
    page = svc.normalise_structure_page(STRUCTURE_RES, None, 0)
    assert page["markdown"].startswith("## Calibration Report\n\nThis page compares")
    assert svc.normalise_structure_page({}, "", 1)["lines"] == []  # VL-style result without OCR


def test_parse_table_html_handles_th_colspan_entities_and_br():
    rows = svc.parse_table_html(
        "<table><thead><tr><th colspan='2'>A &amp; B</th></tr></thead>"
        "<tr><td>x<br>y</td><td> 1 </td></tr><tr><td></td></tr></table>"
    )
    assert rows == [["A & B", "A & B"], ["x y", "1"], [""]]
    assert svc.parse_table_html("") == []


def test_tables_from_markdown_html_and_pipe():
    md = "# T\n\n<table><tr><td>a</td><td>b</td></tr></table>\n\n| h1 | h2 |\n|---|:---:|\n| 1 | 2 |\n\ntext"
    html_table, pipe_table = svc.tables_from_markdown(md)
    assert html_table["rows"] == [["a", "b"]]
    assert pipe_table["rows"] == [["h1", "h2"], ["1", "2"]] and pipe_table["html"] is None


def test_apply_calibration_uncalibrated_uses_raw_and_threshold():
    pages = [svc.normalise_ocr_page(TEXT_RES, 0)]
    info = svc.apply_calibration(pages, 0.98)
    assert info == {"status": "uncalibrated", "method": None, "review_threshold": 0.98}
    lines = pages[0]["lines"]
    assert all(l["calibrated_confidence"] == l["raw_confidence"] for l in lines)
    assert [l["text"] for l in lines if l["needs_review"]] == ["Calibration Report"]  # 0.9727 < 0.98
    summary = svc.summarize(pages)
    assert summary["page_count"] == 1 and summary["line_count"] == 15 and summary["needs_review"] == 1
    assert 0.97 < summary["mean_raw_confidence"] <= 1


def test_apply_calibration_uses_the_active_calibrator(monkeypatch):
    monkeypatch.setattr(calibration, "calibrate_confidences", lambda c: ([x / 2 for x in c], "calibrated"))
    monkeypatch.setattr(calibration, "load_active", lambda: SimpleNamespace(method="isotonic"))
    pages = [svc.normalise_ocr_page(TEXT_RES, 0)]
    info = svc.apply_calibration(pages, 0.7)
    assert info["status"] == "calibrated" and info["method"] == "isotonic"
    line = pages[0]["lines"][0]
    assert line["calibrated_confidence"] == pytest.approx(line["raw_confidence"] / 2, abs=1e-6)
    assert svc.summarize(pages)["needs_review"] == 15


def test_office_page_and_empty_summary():
    page = svc.office_page("# Doc\n\n| a | b |\n|---|---|\n| 1 | 2 |\n")
    assert page["width"] is None and page["image_url"] is None and page["lines"] == []
    assert page["tables"][0]["rows"] == [["a", "b"], ["1", "2"]]
    assert svc.summarize([page])["mean_raw_confidence"] is None
    assert svc.full_text([page]).startswith("# Doc")


# -- input rendering ---------------------------------------------------------------------------


def test_render_png_flattens_alpha(tmp_path):
    src = tmp_path / "a.png"
    Image.new("RGBA", (40, 20), (0, 0, 0, 0)).save(src)
    paths, total = svc.render_pages(src, "image", tmp_path / "out")
    assert total == 1 and [p.name for p in paths] == ["page-0.png"]
    with Image.open(paths[0]) as img:
        assert img.mode == "RGB" and img.size == (40, 20) and img.getpixel((0, 0)) == (255, 255, 255)


def test_render_multiframe_tiff_one_page_per_frame(tmp_path):
    src = tmp_path / "scan.tiff"
    frames = [Image.new("L", (30, 30), v) for v in (0, 128, 255)]
    frames[0].save(src, save_all=True, append_images=frames[1:])
    paths, total = svc.render_pages(src, "image", tmp_path / "out")
    assert total == 3 and len(paths) == 3


def test_render_pdf_at_configured_dpi_and_page_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "paddle_pdf_dpi", 72)
    monkeypatch.setattr(settings, "paddle_max_pages", 2)
    src = tmp_path / "doc.pdf"
    src.write_bytes(make_pdf(["one", "two", "three"]))
    paths, total = svc.render_pages(src, "pdf", tmp_path / "out")
    assert total == 3 and len(paths) == 2
    with Image.open(paths[0]) as img:
        assert img.size == (612, 792)  # US Letter at 72 DPI


@pytest.mark.parametrize(
    ("name", "kind", "data"),
    [("x.png", "image", b"not an image"), ("x.pdf", "pdf", b"%PDF-1.4 garbage"), ("x.docx", "office", b"PK nope")],
)
def test_validate_input_rejects_disguised_files(tmp_path, name, kind, data):
    path = tmp_path / name
    path.write_bytes(data)
    with pytest.raises(svc.UnreadableInputError):
        svc.validate_input(path, kind)


def test_validate_input_accepts_real_files(tmp_path):
    png = tmp_path / "a.png"
    Image.new("RGB", (5, 5)).save(png)
    svc.validate_input(png, "image")
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(make_pdf(["hi"]))
    svc.validate_input(pdf, "pdf")
    docx = tmp_path / "a.docx"
    with zipfile.ZipFile(docx, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
    svc.validate_input(docx, "office")


# -- service with fake models ------------------------------------------------------------------


def test_models_load_once_per_pipeline_and_language_even_concurrently(fake_models, monkeypatch):
    slow = threading.Event()
    original = fake_models.__call__

    def slow_construct(pipeline, lang):
        slow.wait(0.05)
        return original(pipeline, lang)

    monkeypatch.setattr(svc, "_construct_model", slow_construct)
    service = svc.paddleocr_service
    got = []
    threads = [threading.Thread(target=lambda: got.append(service.get_model("ocr", "en"))) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert fake_models.built == [("ocr", "en")]
    assert len({id(m) for m in got}) == 1
    service.get_model("ocr", "fr")
    service.get_model("structure", "en")
    assert fake_models.built == [("ocr", "en"), ("ocr", "fr"), ("structure", "en")]


def test_process_pdf_writes_page_images_and_reports_progress(fake_models, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "paddle_pdf_dpi", 72)
    src = tmp_path / "doc.pdf"
    src.write_bytes(make_pdf(["one", "two"]))
    events = []
    pages, note = svc.paddleocr_service.process(
        src, "ocr", "en", tmp_path / "run", lambda i: f"/img/{i}", lambda f, m: events.append((f, m))
    )
    assert note is None
    assert [p["index"] for p in pages] == [0, 1]
    assert pages[1]["image_url"] == "/img/1" and (pages[1]["width"], pages[1]["height"]) == (612, 792)
    assert pages[1]["lines"][0]["id"] == "p1-l0"
    assert (tmp_path / "run" / "page-1.png").is_file()
    assert ("Page 2 of 2") in [m for _, m in events]
    assert len(fake_models.models[("ocr", "en")].calls) == 2


def test_process_notes_truncated_pdfs(fake_models, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "paddle_max_pages", 1)
    src = tmp_path / "doc.pdf"
    src.write_bytes(make_pdf(["one", "two"]))
    pages, note = svc.paddleocr_service.process(src, "ocr", "en", tmp_path / "run")
    assert len(pages) == 1 and "first 1 of 2" in note


def test_process_structure(fake_models, tmp_path):
    src = tmp_path / "page.png"
    Image.new("RGB", (100, 70), "white").save(src)
    pages, _ = svc.paddleocr_service.process(src, "structure", "en", tmp_path / "run")
    assert pages[0]["tables"][0]["rows"][0] == ["Method", "ECE", "Brier"]
    assert pages[0]["width"] == 100  # size of the image the model saw


def test_recognize_lines(fake_models, tmp_path):
    src = tmp_path / "line.jpg"
    Image.new("RGB", (60, 20), "white").save(src)
    lines = svc.paddleocr_service.recognize_lines(src)
    assert lines[0] == ("Calibration Report", pytest.approx(0.9727, abs=1e-4))
    assert len(lines) == 15 and all(isinstance(t, str) and 0 <= c <= 1 for t, c in lines)


def test_load_failure_is_unavailable_with_message(fake_models):
    fake_models.fail = MemoryError("out of RAM")
    with pytest.raises(svc.PaddleOCRUnavailableError, match="out of RAM"):
        svc.paddleocr_service.get_model("structure", "en")
    status = svc.paddleocr_service.pipeline_status("structure")
    assert status["available"] is True and "out of RAM" in status["message"]


def test_vl_disabled_by_default_and_failed_vl_load_sticks(fake_models, monkeypatch):
    status = svc.paddleocr_service.pipeline_status("vl")
    assert status["available"] is False and "PADDLE_ENABLE_VL=true" in status["message"]
    with pytest.raises(svc.PaddleOCRUnavailableError):
        svc.paddleocr_service.get_model("vl", "en")
    assert fake_models.built == []
    monkeypatch.setattr(settings, "paddle_enable_vl", True)
    fake_models.fail = RuntimeError("no GPU")
    with pytest.raises(svc.PaddleOCRUnavailableError):
        svc.paddleocr_service.get_model("vl", "en")
    status = svc.paddleocr_service.pipeline_status("vl")
    assert status["available"] is False and "no GPU" in status["message"]


def test_not_installed_is_unavailable(monkeypatch):
    monkeypatch.setattr(svc.PaddleOCRService, "installed", staticmethod(lambda: False))
    assert svc.paddleocr_service.pipeline_status("ocr")["available"] is False
    with pytest.raises(svc.PaddleOCRUnavailableError, match="not installed"):
        svc.paddleocr_service.get_model("ocr")


def test_real_constructor_passes_the_configured_options(monkeypatch):
    calls = []

    def recorder(name):
        return lambda **kwargs: calls.append((name, kwargs)) or name

    fake_module = SimpleNamespace(
        PaddleOCR=recorder("PaddleOCR"), PPStructureV3=recorder("PPStructureV3"), PaddleOCRVL=recorder("PaddleOCRVL")
    )
    monkeypatch.setattr(svc, "_import_paddleocr", lambda: fake_module)
    REAL_CONSTRUCT("ocr", "en")
    REAL_CONSTRUCT("ocr", "fr")
    REAL_CONSTRUCT("structure", "en")
    REAL_CONSTRUCT("vl", "en")
    (_, ocr_en), (_, ocr_fr), (_, structure), (_, vl) = calls
    assert ocr_en["enable_mkldnn"] is settings.paddle_enable_mkldnn and ocr_en["device"] == settings.paddle_device
    assert ocr_en["return_word_box"] is True and ocr_en["ocr_version"] == settings.paddle_ocr_version
    assert ocr_en["text_recognition_model_name"] == settings.paddle_rec_model
    assert "text_recognition_model_name" not in ocr_fr and ocr_fr["lang"] == "fr"
    assert structure["formula_recognition_model_name"] == settings.paddle_formula_model
    assert structure["use_seal_recognition"] is False and structure["use_chart_recognition"] is False
    assert vl["pipeline_version"] == settings.paddle_vl_version
    with pytest.raises(ValueError):
        REAL_CONSTRUCT("bogus", "en")


def test_convert_office_uses_doc2md(monkeypatch, tmp_path):
    monkeypatch.setattr(svc.PaddleOCRService, "installed", staticmethod(lambda: True))
    seen = []
    fake_module = SimpleNamespace(doc2md_convert=lambda p: seen.append(p) or SimpleNamespace(markdown="# Hi"))
    monkeypatch.setattr(svc, "_import_paddleocr", lambda: fake_module)
    pages, note = svc.paddleocr_service.process(tmp_path / "a.docx", "office", "en", tmp_path / "run")
    assert pages[0]["markdown"] == "# Hi" and seen == [str(tmp_path / "a.docx")]
    assert not (tmp_path / "run").exists()
