"""Build step 4 tests: OCR, HTR, format detection, and confidence calibration (Modules 2-3).

Tesseract and TrOCR are replaced by fakes, so these run without either installed.
"""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.config import settings
from app.models.schemas import FormatType, RecognizedChunk
from app.modules import confidence_capture, format_detection, htr_extractor, ocr_extractor, vector_store
from app.modules.pdf_render import render_pages
from tests.conftest import TYPED_PAGE, fake_embed

# --- PDF rendering ------------------------------------------------------------


def test_render_pages_gives_one_grayscale_image_per_page(pdf_file) -> None:
    images = list(render_pages(str(pdf_file([TYPED_PAGE, ""])), dpi=72))

    assert len(images) == 2
    assert images[0].mode == "L"
    assert images[0].size == (612, 792)  # US Letter at 72 DPI


def test_render_pages_respects_max_pages(pdf_file) -> None:
    assert len(list(render_pages(str(pdf_file(["a", "b", "c"])), dpi=36, max_pages=2))) == 2


# --- OCR (Tesseract) -----------------------------------------------------------


def _tesseract_data(words: list[tuple[str, float, int, int, int]]) -> dict:
    """Build a pytesseract image_to_data dict from (text, conf, block, par, line) rows."""
    return {
        "text": [w[0] for w in words],
        "conf": [w[1] for w in words],
        "block_num": [w[2] for w in words],
        "par_num": [w[3] for w in words],
        "line_num": [w[4] for w in words],
    }


@pytest.fixture
def fake_tesseract(monkeypatch):
    def _install(words):
        monkeypatch.setattr(ocr_extractor, "_tesseract_cmd", lambda configured: "tesseract")
        monkeypatch.setattr(
            ocr_extractor.pytesseract, "image_to_data", lambda image, output_type, timeout: _tesseract_data(words)
        )

    return _install


def test_ocr_timeout_reads_as_unreadable(monkeypatch) -> None:
    def _slow(image, output_type, timeout):
        raise RuntimeError("Tesseract process timeout")

    monkeypatch.setattr(ocr_extractor, "_tesseract_cmd", lambda configured: "tesseract")
    monkeypatch.setattr(ocr_extractor.pytesseract, "image_to_data", _slow)
    assert ocr_extractor.ocr_image(Image.new("L", (10, 10))) == ("", 0.0)


def test_denoise_removes_speckle_but_keeps_strokes() -> None:
    from app.modules.pdf_render import denoise

    page = Image.new("L", (200, 100), 255)
    draw = ImageDraw.Draw(page)
    draw.rectangle((20, 40, 180, 50), fill=0)  # a stroke
    for x, y in [(5, 5), (100, 90), (190, 10)]:  # isolated specks
        page.putpixel((x, y), 0)

    arr = np.asarray(denoise(page))
    assert arr[45, 100] < 50  # stroke kept
    assert arr[5, 5] > 200 and arr[90, 100] > 200  # specks gone


def test_ocr_rebuilds_lines_and_paragraphs(fake_tesseract) -> None:
    fake_tesseract(
        [
            ("", -1, 1, 0, 0),  # block box, not a word
            ("Deep", 90, 1, 1, 1),
            ("learning", 90, 1, 1, 1),
            ("models", 90, 1, 1, 2),
            ("Results", 90, 1, 2, 1),
        ]
    )
    text, _ = ocr_extractor.ocr_image(Image.new("L", (10, 10)))
    assert text == "Deep learning\nmodels\n\nResults"


def test_ocr_confidence_is_length_weighted_mean(fake_tesseract) -> None:
    # 8 chars at 100% and 2 chars at 50% -> (8*1.0 + 2*0.5) / 10 = 0.9
    fake_tesseract([("abcdefgh", 100, 1, 1, 1), ("ij", 50, 1, 1, 1)])
    _, conf = ocr_extractor.ocr_image(Image.new("L", (10, 10)))
    assert conf == pytest.approx(0.9)


def test_ocr_blank_page(fake_tesseract) -> None:
    fake_tesseract([("", -1, 1, 0, 0), ("  ", 95, 1, 1, 1)])
    assert ocr_extractor.ocr_image(Image.new("L", (10, 10))) == ("", 0.0)


def test_ocr_reports_missing_tesseract(monkeypatch, tmp_path) -> None:
    ocr_extractor._tesseract_cmd.cache_clear()
    monkeypatch.setattr(ocr_extractor.shutil, "which", lambda name: None)
    monkeypatch.setattr(ocr_extractor, "_WINDOWS_DEFAULT", tmp_path / "missing.exe")
    try:
        with pytest.raises(ocr_extractor.OCRUnavailableError, match="winget install"):
            ocr_extractor.ocr_image(Image.new("L", (10, 10)))
    finally:
        ocr_extractor._tesseract_cmd.cache_clear()


# --- HTR (TrOCR) ---------------------------------------------------------------


def _page_with_lines(n: int, size=(800, 1000)) -> Image.Image:
    """A white page with n dark horizontal 'handwriting' strokes."""
    page = Image.new("L", size, 255)
    draw = ImageDraw.Draw(page)
    for i in range(n):
        top = 100 + i * 150
        draw.rectangle((80, top, 700, top + 30), fill=20)
    return page


def test_htr_segments_one_image_per_text_line() -> None:
    lines = htr_extractor.segment_lines(_page_with_lines(3))

    assert len(lines) == 3
    assert all(img.width < 800 and 30 <= img.height < 60 for img in lines)


def test_htr_blank_page_has_no_lines() -> None:
    assert htr_extractor.segment_lines(Image.new("L", (800, 1000), 255)) == []


def test_htr_ignores_specks() -> None:
    page = Image.new("L", (800, 1000), 255)
    ImageDraw.Draw(page).rectangle((100, 100, 104, 103), fill=0)  # tiny dot, not a line
    assert htr_extractor.segment_lines(page) == []


def test_htr_page_joins_lines_with_length_weighted_confidence(monkeypatch) -> None:
    monkeypatch.setattr(htr_extractor, "recognize_lines", lambda lines: [("abcdefgh", 1.0), ("", 0.1), ("ij", 0.5)])

    text, conf = htr_extractor.htr_image(_page_with_lines(3))

    assert text == "abcdefgh\nij"  # empty recognitions dropped
    assert conf == pytest.approx(0.9)


def test_htr_page_with_nothing_recognized(monkeypatch) -> None:
    monkeypatch.setattr(htr_extractor, "recognize_lines", lambda lines: [])
    assert htr_extractor.htr_image(Image.new("L", (100, 100), 255)) == ("", 0.0)


# --- Format detection -----------------------------------------------------------


def test_detect_confident_ocr_as_scanned(pdf_file, fake_ocr) -> None:
    fake_ocr.text, fake_ocr.conf = "Printed words", 0.92
    assert format_detection.detect_format(str(pdf_file(["", "", "", ""]))) is FormatType.SCANNED
    assert fake_ocr.calls == format_detection.OCR_SAMPLE_PAGES  # only samples, not every page


def test_detect_handwriting_when_htr_beats_unconfident_ocr(pdf_file, fake_ocr, fake_htr) -> None:
    fake_ocr.text, fake_ocr.conf = "sc ribb le", 0.31
    fake_htr.text, fake_htr.conf = "scribble notes", 0.8
    assert format_detection.detect_format(str(pdf_file([""]))) is FormatType.HANDWRITTEN


def test_detect_bad_photocopy_stays_scanned(pdf_file, fake_ocr, fake_htr) -> None:
    # Regression: a poor photocopy OCRs badly, but HTR reads print even worse.
    fake_ocr.text, fake_ocr.conf = "Regularizat1on adds a pena1ty", 0.45
    fake_htr.text, fake_htr.conf = "Ruglaiz", 0.1
    assert format_detection.detect_format(str(pdf_file([""]))) is FormatType.SCANNED


def test_detect_confident_ocr_skips_htr(pdf_file, fake_ocr, monkeypatch) -> None:
    fake_ocr.text, fake_ocr.conf = "Printed words", 0.9
    monkeypatch.setattr(htr_extractor, "htr_image", lambda image: pytest.fail("HTR should not run"))
    assert format_detection.detect_format(str(pdf_file([""]))) is FormatType.SCANNED


def test_detect_text_layer_skips_ocr(pdf_file, fake_ocr) -> None:
    assert format_detection.detect_format(str(pdf_file([TYPED_PAGE]))) is FormatType.TYPED
    assert fake_ocr.calls == 0


# --- Module 3: confidence calibration ---------------------------------------------


def test_calibration_typed_is_always_certain() -> None:
    assert confidence_capture.calibrate(0.2, FormatType.TYPED) == 1.0


def test_calibration_is_identity_until_fitted() -> None:
    assert confidence_capture.calibrate(0.63, FormatType.SCANNED) == 0.63


def test_calibration_uses_saved_table() -> None:
    confidence_capture.save_calibration(FormatType.HANDWRITTEN, [(0.2, 0.1), (0.8, 0.6)])

    assert confidence_capture.calibrate(0.5, FormatType.HANDWRITTEN) == pytest.approx(0.35)  # interpolated
    assert confidence_capture.calibrate(0.95, FormatType.HANDWRITTEN) == pytest.approx(0.6)  # clamped to last knot
    assert confidence_capture.calibrate(0.5, FormatType.SCANNED) == 0.5  # other formats unaffected
    assert settings.calibration_file.is_file()


def test_fit_calibration_bins_and_is_monotonic() -> None:
    samples = [(0.15, 0.30), (0.15, 0.40), (0.55, 0.80), (0.65, 0.60), (0.95, 0.97)]

    knots = confidence_capture.fit_calibration(samples, bins=10)

    ys = [y for _, y in knots]
    assert ys == sorted(ys), "calibration must never decrease"
    assert knots[0] == (0.0, 0.0)  # anchor
    assert knots[1] == (0.15, 0.35)  # bin mean
    assert knots[2] == (0.6, 0.7)  # 0.55->0.80 and 0.65->0.60 pooled
    assert knots[-1] == (0.95, 0.97)


def test_fit_calibration_with_only_good_samples_keeps_low_scores_low() -> None:
    # Regression: all-good samples gave a single knot, and interpolation then
    # mapped an illegible page (raw 0.10) to 0.98.
    knots = confidence_capture.fit_calibration([(0.94, 0.98), (0.96, 0.99), (0.95, 0.97)], bins=5)
    confidence_capture.save_calibration(FormatType.HANDWRITTEN, knots)

    assert confidence_capture.calibrate(0.10, FormatType.HANDWRITTEN) < 0.15


def test_fit_calibration_needs_samples() -> None:
    with pytest.raises(ValueError):
        confidence_capture.fit_calibration([])


def test_apply_calibration_sets_calibrated_conf() -> None:
    chunks = [RecognizedChunk(chunk_id="d:0", text="t", raw_conf=0.4)]
    assert confidence_capture.apply_calibration(chunks, FormatType.SCANNED)[0].calibrated_conf == 0.4
    assert chunks[0].calibrated_conf is None  # original untouched


# --- Upload end to end with OCR / HTR ----------------------------------------------


def _upload(client, pdf_path, **form):
    with pdf_path.open("rb") as f:
        return client.post("/upload", files={"file": ("doc.pdf", f, "application/pdf")}, data=form)


def _stored_chunks():
    return vector_store.search(fake_embed(["scanned handwritten notes text"], "")[0], top_k=20)


def test_upload_scanned_pdf_goes_through_ocr_and_calibration(client, pdf_file, fake_ocr) -> None:
    fake_ocr.text, fake_ocr.conf = "Scanned notes about optical character recognition text.", 0.88

    resp = _upload(client, pdf_file(["", ""]))

    assert resp.status_code == 201
    assert resp.json()["format_type"] == "scanned"
    chunks = [c for c, _ in _stored_chunks()]
    assert len(chunks) == 2
    assert all(c.raw_conf == pytest.approx(0.88) and c.calibrated_conf == pytest.approx(0.88) for c in chunks)


def test_upload_handwritten_pdf_goes_through_htr(client, pdf_file, fake_ocr, fake_htr) -> None:
    fake_ocr.text, fake_ocr.conf = "sc ribb le", 0.25  # detection: handwriting
    fake_htr.text, fake_htr.conf = "Handwritten notes text", 0.7

    resp = _upload(client, pdf_file([""]))

    assert resp.status_code == 201
    assert resp.json()["format_type"] == "handwritten"
    [(chunk, _)] = _stored_chunks()
    assert chunk.text == "Handwritten notes text"
    assert chunk.calibrated_conf == pytest.approx(0.7)


def test_upload_format_hint_skips_detection(client, pdf_file, fake_ocr, fake_htr) -> None:
    fake_htr.text, fake_htr.conf = "Handwritten notes text", 0.6

    resp = _upload(client, pdf_file([""]), format_hint="handwritten")

    assert resp.status_code == 201
    assert resp.json()["format_type"] == "handwritten"
    assert fake_ocr.calls == 0


def test_upload_without_tesseract_is_503_and_cleans_up(client, pdf_file, tmp_path, monkeypatch) -> None:
    def _missing(image):
        raise ocr_extractor.OCRUnavailableError("Tesseract is not installed")

    monkeypatch.setattr(ocr_extractor, "ocr_image", _missing)

    resp = _upload(client, pdf_file([""]))

    assert resp.status_code == 503
    assert "Tesseract" in resp.json()["detail"]
    assert not any((tmp_path / "uploads").iterdir())


def test_upload_without_htr_model_is_503(client, pdf_file, monkeypatch) -> None:
    def _missing(image):
        raise htr_extractor.HTRUnavailableError("model not downloaded")

    monkeypatch.setattr(htr_extractor, "htr_image", _missing)

    assert _upload(client, pdf_file([""]), format_hint="handwritten").status_code == 503


def test_retrieval_reports_calibrated_confidence(client, pdf_file, fake_ocr) -> None:
    from app.models.schemas import Query
    from app.modules import retrieval

    fake_ocr.text, fake_ocr.conf = "Scanned notes about optical character recognition text.", 0.8
    confidence_capture.save_calibration(FormatType.SCANNED, [(0.0, 0.0), (1.0, 0.5)])  # halves confidence
    _upload(client, pdf_file([""]), format_hint="scanned")

    [(_, result)] = retrieval.retrieve(Query(query_id="q", question_text="scanned notes", user_id="u"), top_k=5)

    assert result.confidence == pytest.approx(0.4)


def test_otsu_threshold_separates_ink_from_paper() -> None:
    gray = np.array([[20] * 10 + [240] * 90], dtype=np.uint8)
    t = htr_extractor._otsu_threshold(gray)
    assert 20 <= t < 240
