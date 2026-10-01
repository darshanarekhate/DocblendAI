"""Experience Center extras: preprocessing, exporters, structured logging, KIE/translation plugins."""

import csv
import io
import json
import logging
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pymupdf
import pytest
from docx import Document
from PIL import Image, ImageDraw, ImageFont

from app.config import settings
from app.logging_config import JsonFormatter
from app.modules import exporters, preprocess
from app.plugins import PLUGINS, PluginError
from app.plugins import base as plugin_base


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def text_image(lines=("Confidence calibration", "Reliability diagram"), size=(700, 220)) -> np.ndarray:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(lines):
        draw.text((40, 40 + 70 * i), line, font=_font(34), fill="black")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def rotate(image: np.ndarray, degrees: float) -> np.ndarray:
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderValue=(255, 255, 255))


# --- preprocessing ---


def test_parse_options_defaults_and_validation():
    assert preprocess.parse_options(None) == {s: False for s in preprocess.STEPS}
    assert preprocess.parse_options({"deskew": True})["deskew"] is True
    with pytest.raises(preprocess.PreprocessOptionsError):
        preprocess.parse_options({"sharpen": True})
    with pytest.raises(preprocess.PreprocessOptionsError):
        preprocess.parse_options({"deskew": "yes"})
    with pytest.raises(preprocess.PreprocessOptionsError):
        preprocess.parse_options(["deskew"])


@pytest.mark.parametrize("tilt", [4.0, -6.0])
def test_deskew_levels_tilted_text(tilt):
    tilted = rotate(text_image(), tilt)
    assert abs(preprocess.skew_angle(tilted)) > 2
    levelled = preprocess.deskew(tilted)
    assert abs(preprocess.skew_angle(levelled)) < 1.0


def test_deskew_leaves_level_and_blank_pages_alone():
    level = text_image()
    assert preprocess.skew_angle(level) == 0.0
    blank = np.full((100, 100, 3), 255, np.uint8)
    assert preprocess.deskew(blank) is blank


def test_apply_runs_enabled_steps_only():
    image = text_image()
    assert preprocess.apply(image, preprocess.parse_options({})) is image
    out = preprocess.apply(image, {"deskew": False, "denoise": True, "contrast": True, "binarize": True})
    assert out.ndim == 3 and out.shape == image.shape
    assert set(np.unique(out)) <= {0, 255}  # binarized


def test_apply_accepts_grayscale():
    gray = cv2.cvtColor(text_image(), cv2.COLOR_BGR2GRAY)
    assert preprocess.apply(gray, {"contrast": True}).ndim == 3


# --- exporters ---


def make_result(pipeline="ocr", tables=None, markdown=None):
    lines = [
        {"id": "p0-l0", "text": "Confidence calibration", "box": [40, 40, 420, 80], "raw_confidence": 0.99,
         "calibrated_confidence": 0.95, "needs_review": False, "edited": False},
        {"id": "p0-l1", "text": "Reliability diagram", "box": [40, 110, 380, 150], "raw_confidence": 0.8,
         "calibrated_confidence": 0.6, "needs_review": True, "edited": True},
    ]
    return {
        "id": "abc", "status": "done", "pipeline": pipeline, "filename": "notes.png",
        "markdown": markdown or "Confidence calibration\nReliability diagram",
        "pages": [{"index": 0, "width": 700, "height": 220, "lines": lines, "markdown": markdown,
                   "blocks": [], "tables": tables or []}],
    }


TABLE = {"box": [0, 0, 10, 10], "html": "", "rows": [["Method", "ECE"], ["Raw", "0.081"], ["Isotonic", "0.021"]]}


def test_txt_json_md_exports():
    result = make_result()
    assert exporters.to_txt(result).decode() == "Confidence calibration\nReliability diagram"
    assert json.loads(exporters.to_json(result))["id"] == "abc"
    assert "Reliability diagram" in exporters.to_md(result).decode()


def test_txt_multi_page_has_page_headers():
    result = make_result()
    result["pages"].append({**result["pages"][0], "index": 1})
    text = exporters.to_txt(result).decode()
    assert "===== Page 1 =====" in text and "===== Page 2 =====" in text


def test_structure_markdown_preferred():
    result = make_result(pipeline="structure", markdown="# Report\n\nBody text.")
    assert exporters.to_md(result).decode().startswith("# Report")


def test_csv_single_and_zip_multi():
    content, media, ext = exporters.export(make_result(tables=[TABLE]), "csv")
    assert ext == "csv" and media.startswith("text/csv")
    rows = list(csv.reader(io.StringIO(content.decode("utf-8-sig"))))
    assert rows == TABLE["rows"]

    html_table = {"box": None, "html": "<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>"}
    content, media, ext = exporters.export(make_result(tables=[TABLE, html_table]), "csv")
    assert ext == "zip"
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = archive.namelist()
        assert names == ["table-p1-1.csv", "table-p1-2.csv"]
        assert archive.read(names[1]).decode("utf-8-sig").splitlines() == ["A,B", "1,2"]


def test_csv_without_tables_is_an_error():
    with pytest.raises(exporters.ExportError):
        exporters.export(make_result(), "csv")


def test_unknown_format():
    with pytest.raises(exporters.ExportError):
        exporters.export(make_result(), "xlsx")


def test_docx_from_markdown_with_html_table():
    markdown = (
        "## Calibration Report\n\nThis page compares **raw** and calibrated scores.\n\n"
        "- lower ECE is better\n1. fit\n\n"
        '<div style="text-align: center;"><html><body><table border="1"><tr><td>Method</td><td>ECE</td></tr>'
        "<tr><td>Raw</td><td>0.081</td></tr></table></body></html></div>\n\n"
        "| a | b |\n|---|---|\n| 1 | 2 |\n"
    )
    doc = Document(io.BytesIO(exporters.markdown_to_docx(markdown, title="t")))
    texts = [p.text for p in doc.paragraphs]
    assert "Calibration Report" in texts
    assert any("raw" in t and "calibrated" in t for t in texts)
    assert "lower ECE is better" in texts
    assert [[c.text for c in row.cells] for row in doc.tables[0].rows] == [["Method", "ECE"], ["Raw", "0.081"]]
    assert [[c.text for c in row.cells] for row in doc.tables[1].rows] == [["a", "b"], ["1", "2"]]


def test_searchable_pdf_has_selectable_text(tmp_path: Path):
    image_path = tmp_path / "page-0.png"
    cv2.imwrite(str(image_path), text_image())
    content, media, ext = exporters.export(make_result(), "pdf", [image_path], dpi=150)
    assert media == "application/pdf" and ext == "pdf"
    with pymupdf.open(stream=content, filetype="pdf") as doc:
        page = doc[0]
        assert page.get_text().split() == ["Confidence", "calibration", "Reliability", "diagram"]
        # The hidden word sits where OCR found it (box 40..420 px at 150 DPI -> points).
        hit = page.search_for("Reliability")[0]
        assert abs(hit.x0 - 40 * 72 / 150) < 3


def test_searchable_pdf_needs_page_images():
    with pytest.raises(exporters.ExportError):
        exporters.export(make_result(), "pdf", [None])


# --- logging ---


def test_json_formatter_includes_extra_fields():
    record = logging.LogRecord("app.x", logging.INFO, __file__, 1, "job %s done", ("j1",), None)
    record.duration_ms = 12.5
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "job j1 done"
    assert payload["level"] == "INFO" and payload["duration_ms"] == 12.5


# --- plugins ---


def test_plugins_disabled_without_key(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "")
    for plugin in PLUGINS.values():
        assert plugin.info()["enabled"] is False
        assert "GEMINI_API_KEY" in plugin.info()["message"]
    with pytest.raises(PluginError):
        plugin_base.gemini("x", "y")


def test_kie_drops_values_not_in_text(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "test")
    reply = json.dumps({"title": "Calibration Report", "author": "Invented Person", "key terms": ["ECE", "made up"]})
    monkeypatch.setattr("app.plugins.kie.gemini", lambda prompt, system, as_json=False: reply)
    text = "Calibration Report\nLower ECE means better calibrated scores."
    out = PLUGINS["kie"].run(text, {"fields": ["title", "author", "key terms"]})
    assert out == {"fields": {"title": "Calibration Report", "author": None, "key terms": ["ECE"]}}


def test_kie_validates_fields(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "test")
    with pytest.raises(PluginError):
        PLUGINS["kie"].run("text", {"fields": "title"})


def test_translate(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "test")
    seen = {}

    def fake(prompt, system, as_json=False):
        seen["prompt"] = prompt
        return "# Rapport"

    monkeypatch.setattr("app.plugins.translate.gemini", fake)
    out = PLUGINS["translate"].run("# Report", {"target_language": "French"})
    assert out == {"target_language": "French", "markdown": "# Rapport", "truncated": False}
    assert seen["prompt"].startswith("Target language: French")
    with pytest.raises(PluginError):
        PLUGINS["translate"].run("# Report", {})


def test_parse_json_tolerates_fences():
    assert plugin_base.parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(PluginError):
        plugin_base.parse_json("not json")


def test_deskew_ignores_speckle_and_grey_scan_borders():
    # noisy_scan.png: speckled, low contrast, rotated 2.2 degrees clockwise with grey corners.
    sample = cv2.imread(str(Path(__file__).parent.parent / "app" / "static" / "samples" / "noisy_scan.png"))
    assert preprocess.skew_angle(sample) == pytest.approx(2.2, abs=0.3)
    assert preprocess.skew_angle(preprocess.deskew(sample)) == 0.0
