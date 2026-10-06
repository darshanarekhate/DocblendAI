"""Module 2 (OCR path) — read ruled tables in scanned pages and images cell by cell.

Whole-page OCR reads a table badly: Tesseract segments the page into text blocks,
so a table comes out column by column (all names, then all descriptions), values
go missing next to the ruling lines, and a cell wrapped over several lines is
split up. Here the table's ruling lines are found first (OpenCV), the grid of
cells is rebuilt (merged cells included: a missing separator line joins the two
grid cells), and every cell is OCRed on its own. The table is then written like
the typed-document tables (text_parser._grid_lines):

    Table: Criteria | C++ | C# | Java
    Criteria: Keywords | C++: 62 | C#: 87 | Java: 50

Text above, between and below tables is OCRed band by band, so the page keeps
its reading order. A page without a ruled table returns None and is OCRed as a
whole, as before.
"""

import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Ruling lines: straight runs at least this share of the page width / height.
H_LINE_MIN = 1 / 12
V_LINE_MIN = 1 / 30
# A row/column separator must cover at least this share of the table's width/height
# (a merged header cell interrupts the column lines, so these are not 1.0).
ROW_LINE_COVER = 0.5
COL_LINE_COVER = 0.25
# Between two grid cells, a separator counts as present if it covers this share of the cell edge.
EDGE_COVER = 0.5
MIN_ROWS, MIN_COLS = 2, 2
CELL_INSET = 3  # px kept away from the ruling lines when cropping a cell
MIN_CELL = 15  # px: boundaries closer than this are one (thick or double) line
LINE_SLACK = 6  # px: a separator may wander this far from its average position (slightly tilted scans)
MAX_CELLS = 400  # beyond this the page is OCRed as a whole (too slow cell by cell)


def _masks(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    h, w = gray.shape
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    h_lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, int(w * H_LINE_MIN)), 1)))
    v_lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, int(h * V_LINE_MIN)))))
    return h_lines > 0, v_lines > 0


def _clusters(positions: np.ndarray, gap: int = MIN_CELL) -> list[int]:
    """Centres of runs of nearby positions: a thick line, or a double border, is one line."""
    out, run = [], []
    for p in positions.tolist():
        if run and p - run[-1] > gap:
            out.append(int(np.mean(run)))
            run = []
        run.append(p)
    if run:
        out.append(int(np.mean(run)))
    return out


def _table_boxes(h_mask: np.ndarray, v_mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Bounding boxes (x0, y0, x1, y1) of line grids big enough to be tables, top to bottom."""
    import cv2

    grid = ((h_mask | v_mask) * 255).astype(np.uint8)
    grid = cv2.dilate(grid, np.ones((5, 5), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(grid, connectivity=8)
    h, w = h_mask.shape
    boxes = []
    for i in range(1, count):
        x, y, bw, bh = stats[i, :4]
        if bw > 0.15 * w and bh > 0.03 * h:
            boxes.append((int(x), int(y), int(x + bw), int(y + bh)))
    return sorted(boxes, key=lambda b: b[1])


def _grid(h_mask, v_mask, box):
    """Row and column boundaries inside a table box, or None if it is not a table."""
    x0, y0, x1, y1 = box
    hs, vs = h_mask[y0:y1, x0:x1], v_mask[y0:y1, x0:x1]
    rows = _clusters(np.flatnonzero(hs.mean(axis=1) >= ROW_LINE_COVER)) or []
    cols = _clusters(np.flatnonzero(vs.mean(axis=0) >= COL_LINE_COVER)) or []
    inner_cols = [c for c in cols if 6 < c < (x1 - x0) - 6]
    if not inner_cols:
        return None  # no column separator inside: a box or underlines, not a table
    # Open edges (a table without an outer border, or cut off by the image edge) are boundaries too.
    if not rows or rows[0] > MIN_CELL:
        rows = [0] + rows
    if rows[-1] < (y1 - y0) - MIN_CELL:
        rows.append(y1 - y0 - 1)
    if not cols or cols[0] > MIN_CELL:
        cols = [0] + cols
    if cols[-1] < (x1 - x0) - MIN_CELL:
        cols.append(x1 - x0 - 1)
    if len(rows) - 1 < MIN_ROWS or len(cols) - 1 < MIN_COLS or (len(rows) - 1) * (len(cols) - 1) > MAX_CELLS:
        return None
    return rows, cols


def _covered(mask: np.ndarray, line: int, start: int, end: int, axis: int) -> bool:
    """Is there a separator line at `line` between start..end (axis 0: vertical line at x=line)?"""
    lo, hi = max(0, line - LINE_SLACK), line + LINE_SLACK + 1
    band = mask[start:end, lo:hi] if axis == 0 else mask[lo:hi, start:end]
    if band.size == 0:
        return True  # at the table edge
    covered = band.any(axis=1 if axis == 0 else 0)
    return covered.mean() >= EDGE_COVER


def _cell_groups(h_mask, v_mask, box, rows, cols) -> list[tuple[int, int, int, int, int, int]]:
    """Merged-cell-aware cells: (row, col, y0, y1, x0, x1), each group of merged grid cells once."""
    x0, y0, x1, y1 = box
    hs, vs = h_mask[y0:y1, x0:x1], v_mask[y0:y1, x0:x1]
    n_r, n_c = len(rows) - 1, len(cols) - 1
    parent = list(range(n_r * n_c))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for r in range(n_r):
        for c in range(n_c):
            if c + 1 < n_c and not _covered(vs, cols[c + 1], rows[r] + 2, rows[r + 1] - 2, axis=0):
                parent[find(r * n_c + c + 1)] = find(r * n_c + c)  # no line to the right: merged
            if r + 1 < n_r and not _covered(hs, rows[r + 1], cols[c] + 2, cols[c + 1] - 2, axis=1):
                parent[find((r + 1) * n_c + c)] = find(r * n_c + c)  # no line below: merged
    groups: dict[int, list[tuple[int, int]]] = {}
    for r in range(n_r):
        for c in range(n_c):
            groups.setdefault(find(r * n_c + c), []).append((r, c))
    cells = []
    for members in groups.values():
        r0, c0 = min(m[0] for m in members), min(m[1] for m in members)
        r1, c1 = max(m[0] for m in members), max(m[1] for m in members)
        if len(members) != (r1 - r0 + 1) * (c1 - c0 + 1):
            # Not a rectangle: a gap in a ruling line chained in a cell that is not really merged
            # (e.g. a header spanning three columns + one cell below it). Keep each row's run of
            # neighbouring columns together (the spanning header stays one cell) and split the rest.
            for r in sorted({m[0] for m in members}):
                row_cols = sorted(c for rr, c in members if rr == r)
                run = [row_cols[0]]
                for c in row_cols[1:] + [None]:
                    if c is not None and c == run[-1] + 1:
                        run.append(c)
                        continue
                    cells.append((r, run[0], rows[r], rows[r + 1], cols[run[0]], cols[run[-1] + 1]))
                    if c is not None:
                        run = [c]
            continue
        cells.append((r0, c0, rows[r0], rows[r1 + 1], cols[c0], cols[c1 + 1]))
    return cells


def _ocr(image: Image.Image) -> tuple[str, float]:
    from app.modules import ocr_extractor  # late import: ocr_extractor imports this module

    return ocr_extractor.ocr_image(image)


def _ocr_cell(image: Image.Image) -> tuple[str, float]:
    from app.modules import ocr_extractor

    return ocr_extractor.ocr_cell(image)


def _read_table(page: Image.Image, h_mask, v_mask, box) -> tuple[list[str], float, int] | None:
    from app.modules.text_parser import _grid_lines

    grid_info = _grid(h_mask, v_mask, box)
    if grid_info is None:
        return None
    rows, cols = grid_info
    x0, y0, _, _ = box
    grid = [["" for _ in range(len(cols) - 1)] for _ in range(len(rows) - 1)]
    weighted, chars = 0.0, 0
    for r, c, cy0, cy1, cx0, cx1 in _cell_groups(h_mask, v_mask, box, rows, cols):
        crop = page.crop((x0 + cx0 + CELL_INSET, y0 + cy0 + CELL_INSET, x0 + cx1 - CELL_INSET, y0 + cy1 - CELL_INSET))
        if crop.width < 8 or crop.height < 8:
            continue
        text, conf = _ocr_cell(crop)
        # One cell = one line, however it wrapped. A "|" is the edge of a ruling line read as
        # a character (a cell cannot contain the separator anyway).
        text = " ".join(text.replace("|", " ").split())
        grid[r][c] = text
        weighted += conf * len(text)
        chars += len(text)
    lines = _grid_lines(grid)
    if not lines:
        return None
    return lines, (weighted / chars if chars else 0.0), chars


def read_page(page: Image.Image) -> tuple[str, float] | None:
    """(text, raw_conf) for a page with ruled tables read cell by cell; None if it has no table."""
    gray = np.asarray(page.convert("L"))
    h_mask, v_mask = _masks(gray)
    boxes = _table_boxes(h_mask, v_mask)
    tables = []
    for box in boxes:
        result = _read_table(page, h_mask, v_mask, box)
        if result is not None:
            tables.append((box, result))
    if not tables:
        return None

    parts, weighted, chars, y = [], 0.0, 0, 0
    width, height = page.size

    def text_band(top: int, bottom: int) -> None:
        nonlocal weighted, chars
        if bottom - top < 12:
            return
        text, conf = _ocr(page.crop((0, top, width, bottom)))
        if text.strip():
            parts.append(text.strip())
            weighted += conf * len(text)
            chars += len(text)

    for (bx0, by0, bx1, by1), (lines, conf, n) in tables:
        text_band(y, by0)
        parts.append("\n".join(lines))
        weighted += conf * n
        chars += n
        y = max(y, by1)
    text_band(y, height)
    logger.info("Read %d table(s) cell by cell", len(tables))
    return "\n\n".join(parts), (weighted / chars if chars else 0.0)
