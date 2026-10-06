"""Ruled tables in scanned pages/images, read cell by cell (table_ocr.py).

No Tesseract needed: every cell of a drawn table is filled with its own light grey
shade, and a stand-in for cell OCR names the cell from that shade. So these tests
check the grid (rows, columns, merged cells, open edges), not character recognition.
"""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.config import settings
from app.modules import ocr_extractor, table_ocr

ROW_H, COL_W, X0, Y0 = 60, 220, 60, 80


def _table_page(cells, n_rows, n_cols, missing=(), border=True, size=(1000, 900), spans=None):
    """Draw a table. cells: {(row, col): (label, shade)}; missing: separators not drawn,
    as ("v", row, col) = no line right of (row, col), ("h", row, col) = no line below it;
    spans: {(row, col): (rows, cols)} for merged cells, whose fill covers the whole span."""
    page = Image.new("L", size, 255)
    draw = ImageDraw.Draw(page)
    for (r, c), (_, shade) in cells.items():
        rs, cs = (spans or {}).get((r, c), (1, 1))
        draw.rectangle((X0 + c * COL_W + 6, Y0 + r * ROW_H + 6, X0 + (c + cs) * COL_W - 6, Y0 + (r + rs) * ROW_H - 6), fill=shade)
    for r in range(n_rows + 1):
        for c in range(n_cols):
            if ("h", r - 1, c) in missing or (not border and r in (0, n_rows)):
                continue
            y = Y0 + r * ROW_H
            draw.line((X0 + c * COL_W, y, X0 + (c + 1) * COL_W, y), fill=0, width=3)
    for c in range(n_cols + 1):
        for r in range(n_rows):
            if ("v", r, c - 1) in missing or (not border and c in (0, n_cols)):
                continue
            x = X0 + c * COL_W
            draw.line((x, Y0 + r * ROW_H, x, Y0 + (r + 1) * ROW_H), fill=0, width=3)
    return page


@pytest.fixture
def read_cells(monkeypatch):
    """Cell OCR stand-in: names a cell from its fill shade; counts calls."""
    labels: dict[int, str] = {}
    calls = []

    def fake(crop):
        calls.append(crop.size)
        values = np.asarray(crop.convert("L")).ravel()
        values = values[values < 250]
        if values.size == 0:
            return "", 0.0
        shade = int(np.bincount(values).argmax())
        return labels.get(shade, ""), 0.9

    monkeypatch.setattr(table_ocr, "_ocr_cell", fake)
    monkeypatch.setattr(table_ocr, "_ocr", lambda image: ("", 0.0))  # text around the table
    return labels, calls


def _cells(rows):
    """{(r, c): (label, shade)} from a list of rows of labels; each label gets its own shade."""
    out, shade = {}, 160
    for r, row in enumerate(rows):
        for c, label in enumerate(row):
            if label is not None:
                out[(r, c)] = (label, shade)
                shade += 3
    return out


def _labels(cells):
    return {shade: label for label, shade in cells.values()}


def test_a_simple_table_keeps_values_with_their_row_and_column(read_cells) -> None:
    labels, _ = read_cells
    cells = _cells([["Name", "Marks", "Grade"], ["Asha", "91", "A"], ["Ravi", "84", "B"]])
    labels.update(_labels(cells))
    text, conf = table_ocr.read_page(_table_page(cells, 3, 3))
    assert text.splitlines() == [
        "Table: Name | Marks | Grade",
        "Name: Asha | Marks: 91 | Grade: A",
        "Name: Ravi | Marks: 84 | Grade: B",
    ]
    assert conf == pytest.approx(0.9)


def test_merged_header_over_two_rows_and_columns(read_cells) -> None:
    """Like the C++/C#/Java table: "Criteria" spans two rows, "Languages" spans the columns."""
    labels, _ = read_cells
    cells = _cells([["Criteria", "Languages", None, None], [None, "C++", "C#", "Java"], ["Keywords", "62", "87", "50"]])
    labels.update(_labels(cells))
    missing = {("h", 0, 0), ("v", 0, 1), ("v", 0, 2)}  # Criteria / Languages are merged cells
    spans = {(0, 0): (2, 1), (0, 1): (1, 3)}
    text, _ = table_ocr.read_page(_table_page(cells, 3, 4, missing=missing, spans=spans))
    assert text.splitlines() == [
        "Table: Criteria | C++ | C# | Java",
        "Criteria: Keywords | C++: 62 | C#: 87 | Java: 50",
    ]


def test_a_merged_cell_is_read_once(read_cells) -> None:
    labels, calls = read_cells
    cells = _cells([["Week 1 summary", None, None], ["Mon", "Tue", "Wed"]])
    labels.update(_labels(cells))
    page = _table_page(cells, 2, 3, missing={("v", 0, 0), ("v", 0, 1)}, spans={(0, 0): (1, 3)})
    text, _ = table_ocr.read_page(page)
    assert text.splitlines() == ["Week 1 summary", "Mon | Tue | Wed"]  # no usable header row
    assert len(calls) == 4  # 1 merged + 3 cells


def test_a_table_without_an_outer_border(read_cells) -> None:
    labels, _ = read_cells
    cells = _cells([["Interaction", "Possibility to specify the user behaviours"],
                    ["Multimedia", "Possibility to support graphics, audio and video"],
                    ["Tool CASE", "Availability of a tool supporting the methodology"]])
    labels.update(_labels(cells))
    text, _ = table_ocr.read_page(_table_page(cells, 3, 2, border=False))
    lines = text.splitlines()
    assert "Interaction | Possibility to specify the user behaviours" in lines  # a data row, not a header
    assert "Tool CASE | Availability of a tool supporting the methodology" in lines


def test_a_page_without_a_table_is_left_to_whole_page_ocr(read_cells) -> None:
    page = Image.new("L", (1000, 900), 255)
    ImageDraw.Draw(page).rectangle((100, 100, 600, 130), fill=0)  # a bar, not a grid
    assert table_ocr.read_page(page) is None


def test_ocr_file_reads_tables_cell_by_cell(read_cells, monkeypatch, tmp_path) -> None:
    labels, _ = read_cells
    cells = _cells([["Protocol", "Port"], ["HTTP", "80"], ["HTTPS", "443"]])
    labels.update(_labels(cells))
    path = tmp_path / "ports.png"
    _table_page(cells, 3, 2, size=(2400, 3400)).save(path)  # page-sized: not enlarged
    monkeypatch.setattr(ocr_extractor, "ocr_image", lambda image: pytest.fail("whole-page OCR used for a table page"))
    [(text, _)] = ocr_extractor.ocr_file(str(path))
    assert "Protocol: HTTPS | Port: 443" in text.splitlines()

    monkeypatch.setattr(settings, "ocr_tables", False)
    monkeypatch.setattr(ocr_extractor, "ocr_image", lambda image: ("whole page", 0.8))
    assert ocr_extractor.ocr_file(str(path)) == [("whole page", 0.8)]
