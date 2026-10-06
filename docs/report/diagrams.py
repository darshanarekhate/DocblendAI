"""SVG diagrams for the DocBlendAI project report (docs/report/build_report.py).

Small drawing helpers (boxes, decisions, arrows, use-case ellipses, actors) and one function per
diagram. Coordinates are in SVG user units; each diagram scales to the page width when printed.
"""

from html import escape

INK = "#1f2a37"
MUTED = "#5b6775"
BLUE = "#2f6db5"
GREEN = "#2f7d5b"
AMBER = "#b7791f"
RED = "#b3261e"
PURPLE = "#6b4fa0"
FILL = {
    "blue": "#e6eff9", "green": "#e5f3ec", "amber": "#fdf1dc", "red": "#fbe7e5",
    "purple": "#efe9f8", "grey": "#f1f3f6", "white": "#ffffff", "teal": "#e2f2f3",
}
STROKE = {
    "blue": BLUE, "green": GREEN, "amber": AMBER, "red": RED, "purple": PURPLE, "grey": "#8a96a3",
    "white": "#8a96a3", "teal": "#2a8a91",
}


def svg(width: int, height: int, body: str, title: str) -> str:
    return (
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="{escape(title)}" class="diagram" font-family="Segoe UI, Arial, sans-serif">'
        '<defs>'
        f'<marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker>'
        f'<marker id="arrg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker>'
        '</defs>'
        f'{body}</svg>'
    )


def _lines(text: str, cx: float, cy: float, size: float, weight: str = "normal", color: str = INK,
           anchor: str = "middle", line_gap: float = 1.25) -> str:
    rows = text.split("\n")
    first = cy - (len(rows) - 1) * size * line_gap / 2
    out = []
    for i, row in enumerate(rows):
        w = weight
        if row.startswith("**") and row.endswith("**"):
            row, w = row[2:-2], "bold"
        out.append(
            f'<text x="{cx:.1f}" y="{first + i * size * line_gap:.1f}" font-size="{size}" font-weight="{w}" '
            f'fill="{color}" text-anchor="{anchor}" dominant-baseline="middle">{escape(row)}</text>'
        )
    return "".join(out)


def box(x, y, w, h, text, kind="blue", size=12, bold_first=True, rx=8, dashed=False) -> str:
    rows = text.split("\n")
    if bold_first and rows and not rows[0].startswith("**"):
        rows[0] = f"**{rows[0]}**"
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{FILL[kind]}" stroke="{STROKE[kind]}" stroke-width="1.4"{dash}/>'
        + _lines("\n".join(rows), x + w / 2, y + h / 2, size)
    )


def pill(x, y, w, h, text, kind="green", size=12) -> str:
    return box(x, y, w, h, text, kind, size, rx=h / 2)


def diamond(cx, cy, w, h, text, kind="amber", size=11.5) -> str:
    pts = f"{cx},{cy - h / 2} {cx + w / 2},{cy} {cx},{cy + h / 2} {cx - w / 2},{cy}"
    return (
        f'<polygon points="{pts}" fill="{FILL[kind]}" stroke="{STROKE[kind]}" stroke-width="1.4"/>'
        + _lines(text, cx, cy, size)
    )


def arrow(points, label=None, label_at=None, color=INK, dashed=False, size=11, label_anchor="middle") -> str:
    d = " ".join(f"{x},{y}" for x, y in points)
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    marker = "arrg" if color != INK else "arr"
    out = f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="1.4"{dash} marker-end="url(#{marker})"/>'
    if label:
        if label_at is None:
            (x1, y1), (x2, y2) = points[0], points[1]
            label_at = ((x1 + x2) / 2 + 6, (y1 + y2) / 2)
        lx, ly = label_at
        out += (
            f'<text x="{lx}" y="{ly}" font-size="{size}" fill="{MUTED}" font-style="italic" '
            f'text-anchor="{label_anchor}" dominant-baseline="middle">{escape(label)}</text>'
        )
    return out


def line(points, color=MUTED, dashed=False) -> str:
    d = " ".join(f"{x},{y}" for x, y in points)
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    return f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="1.3"{dash}/>'


def ellipse(cx, cy, rx, ry, text, kind="blue", size=11.5) -> str:
    return (
        f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{FILL[kind]}" stroke="{STROKE[kind]}" stroke-width="1.3"/>'
        + _lines(text, cx, cy, size)
    )


def actor(x, y, label, size=12) -> str:
    return (
        f'<circle cx="{x}" cy="{y}" r="11" fill="#fff" stroke="{INK}" stroke-width="1.6"/>'
        f'<line x1="{x}" y1="{y + 11}" x2="{x}" y2="{y + 42}" stroke="{INK}" stroke-width="1.6"/>'
        f'<line x1="{x - 18}" y1="{y + 22}" x2="{x + 18}" y2="{y + 22}" stroke="{INK}" stroke-width="1.6"/>'
        f'<line x1="{x}" y1="{y + 42}" x2="{x - 14}" y2="{y + 64}" stroke="{INK}" stroke-width="1.6"/>'
        f'<line x1="{x}" y1="{y + 42}" x2="{x + 14}" y2="{y + 64}" stroke="{INK}" stroke-width="1.6"/>'
        + _lines(label, x, y + 84, size, weight="bold")
    )


def label(x, y, text, size=12, color=MUTED, weight="normal", anchor="start") -> str:
    return _lines(text, x, y, size, weight=weight, color=color, anchor=anchor)


def band(x, y, w, h, title, kind) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{FILL[kind]}" stroke="{STROKE[kind]}" stroke-width="1" opacity="0.55"/>'
        + f'<text x="{x + 12}" y="{y + 17}" font-size="12.5" font-weight="bold" fill="{STROKE[kind]}">{escape(title)}</text>'
    )


# ------------------------------------------------------------------------------------------------
# 1. Layered architecture
# ------------------------------------------------------------------------------------------------


def architecture() -> str:
    W = 780
    b = []
    y = 8
    rows = [
        ("1  PRESENTATION LAYER  (browser, plain HTML / CSS / JavaScript)", "blue", [
            "QA page  /\nindex.html: upload, documents,\nask & chat, viewer",
            "Experience Center  /studio\nstudio.html / .js: page + boxes,\ntabs, exports, calibration",
            "Shared  layout.js / .css\nlayout view, LLM change marks,\nMarkdown + table sanitiser",
            "API clients\nSwagger UI /docs,\ncurl, tests",
        ]),
        ("2  API LAYER  (FastAPI routers, one monolith)", "teal", [
            "upload.py\n/upload  /documents\n/library  /file",
            "query.py\n/ask  /chat\n/ask/suggest  /answer",
            "preview.py\n/documents/{id}/\npreview  /pages  /view",
            "paddleocr.py\n/api/ocr  /api/parse\n/api/results  /calibrate",
            "refine.py\n/changes /revert\n/reapply /versions",
            "studio_tools.py\n/export  /preprocess\n/plugins",
        ]),
        ("3  PIPELINE LAYER  (background jobs, one worker thread)", "green", [
            "ocr_jobs.py\nupload job: extract,\nrefine, index; progress",
            "document_extract.py\nModule 2: routes typed /\nscanned / handwritten",
            "refinement.py\n+ text_diff.py: LLM\nrefinement, versions",
            "qa_index.py\nModules 3-5: chunks,\ncitations, LLM cap",
            "document_runs.py\nQA document <-> run,\nlibrary, delete both",
        ]),
        ("4  INTELLIGENCE LAYER  (models and algorithms)", "purple", [
            "Recognition\nPaddleOCR PP-OCRv5,\nPP-StructureV3, TrOCR,\ndocTR, Tesseract",
            "Confidence\nconfidence_capture.py,\napp/calibration (temp.,\nPlatt, isotonic)",
            "Retrieval & trust\nembedder, vector_store,\nretrieval (combined score),\nreliability (4 tiers)",
            "Language model\nllm_answer.py (answers),\nplugins: refine, KIE,\ntranslate; spelling.py",
        ]),
        ("5  STORAGE LAYER", "amber", [
            "SQLite (SQLAlchemy)\ndocuments, queries, answers,\nretrieval_results, ocr_runs,\nrun_versions, document_runs",
            "ChromaDB\nchunk vectors (768-d) +\ndoc_id, page, lines,\ncontent type, confidence",
            "Files  data/\nuploads/, paddle_runs/\n(page images),\ncalibration files",
        ]),
        ("6  EXTERNAL SERVICES", "grey", [
            "Google Gemini API\ngemini-3.5-flash-lite (+ fallback),\ngemini-embedding-001",
            "Model hubs (first run only)\nHugging Face, PaddleX:\nweights cached locally",
        ]),
    ]
    heights = [86, 86, 86, 100, 92, 74]
    for (title, kind, items), h in zip(rows, heights):
        b.append(band(6, y, W - 12, h + 26, title, kind))
        n = len(items)
        gap = 10
        bw = (W - 40 - gap * (n - 1)) / n
        for i, text in enumerate(items):
            b.append(box(20 + i * (bw + gap), y + 24, bw, h - 4, text, kind if kind != "grey" else "white", size=10.5))
        y += h + 26 + 12
    # flow arrows between layers (right margin)
    return svg(W, y, "".join(b), "Layered architecture of DocBlendAI")


# ------------------------------------------------------------------------------------------------
# 2. Use case diagram
# ------------------------------------------------------------------------------------------------


def use_cases() -> str:
    W, H = 800, 760
    b = [f'<rect x="185" y="10" width="455" height="{H - 20}" rx="14" fill="#fbfcfe" stroke="{INK}" stroke-width="1.5"/>',
         label(412, 30, "DocBlendAI system", 14, INK, "bold", "middle")]
    main = [
        ("Upload a document\n(PDF, image, Office, text)", "blue"),
        ("View document\n(pages / layout / original)", "blue"),
        ("Choose documents\nfor answers", "blue"),
        ("Ask a question /\nchat follow-up", "blue"),
        ("See answer, reliability\ntier and cited sources", "blue"),
        ("Get \"Did you mean\"\nsuggestion", "blue"),
        ("Review LLM changes:\nshow original / revert / retry", "blue"),
        ("Correct a line\nmanually", "blue"),
        ("Export result (TXT, layout,\nMD, JSON, PDF, DOCX, CSV)", "blue"),
        ("OCR / parse in the\nExperience Center", "blue"),
        ("Clean up image\n(deskew, denoise, ...)", "blue"),
        ("Delete a document", "blue"),
        ("Fit confidence\ncalibration", "purple"),
        ("Run evaluation\n(CER, WER, QA)", "purple"),
    ]
    cx, rx, ry = 318, 118, 21
    ys = [62 + i * 49 for i in range(len(main))]
    for (text, kind), y in zip(main, ys):
        b.append(ellipse(cx, y, rx, ry, text, kind, size=10.5))
    inner = [("Extract text\n(OCR / HTR / parse)", 150), ("Refine with LLM\n(guarded)", 255),
             ("Index for QA\n(chunk, embed)", 360), ("Retrieve &\nlabel reliability", 440)]
    icx, irx, iry = 548, 76, 24
    for text, y in inner:
        b.append(ellipse(icx, y, irx, iry, text, "green", size=10.5))
    # include relations
    inc = [(0, 150), (0, 255), (0, 360), (9, 150), (3, 440), (6, 255)]
    for idx, ty in inc:
        b.append(arrow([(cx + rx - 4, ys[idx]), (icx - irx + 4, ty)], color=MUTED, dashed=True))
    b.append(label(470, 92, "«include»", 10.5, MUTED, anchor="middle"))
    # actors
    b.append(actor(70, 150, "Student /\nResearcher /\nFaculty"))
    b.append(actor(70, 560, "Developer /\nAdmin"))
    b.append(actor(735, 150, "Google Gemini\n(LLM + embeddings)"))
    b.append(actor(735, 470, "OCR / HTR models\n(local)"))
    user = (90, 180)
    for i in range(12):
        b.append(line([user, (cx - rx, ys[i])]))
    dev = (90, 590)
    for i in (9, 12, 13):
        b.append(line([dev, (cx - rx, ys[i])]))
    gem = (715, 180)
    for ty in (255, 360, 440):
        b.append(line([gem, (icx + irx, ty)]))
    b.append(line([gem, (cx + rx, ys[5])]))
    mod = (715, 500)
    b.append(line([mod, (icx + irx, 150)]))
    b.append(line([mod, (cx + rx, ys[12])]))
    return svg(W, H, "".join(b), "Use case diagram")


# ------------------------------------------------------------------------------------------------
# 3. Upload pipeline (what happens internally when a file is uploaded)
# ------------------------------------------------------------------------------------------------


def upload_flow() -> str:
    W, H = 780, 1080
    C, L, R = 390, 175, 605
    b = []
    b.append(pill(C - 170, 10, 340, 40, "User uploads a file\n(QA page or Experience Center)", "green", 12))
    b.append(arrow([(C, 50), (C, 72)]))
    b.append(box(C - 170, 72, 340, 48, "Module 1  Validate & store\ntype + size check, save to data/uploads/", "blue", 11))
    b.append(arrow([(C, 120), (C, 142)]))
    b.append(box(C - 170, 142, 340, 48, "Start background job\nocr_runs row; progress shown on both pages", "blue", 11))
    b.append(arrow([(C, 190), (C, 214)]))
    b.append(diamond(C, 256, 300, 84, "Typed file?\n(docx/pptx/xlsx/txt or\nPDF with a text layer)"))
    b.append(arrow([(C - 150, 256), (L, 256), (L, 300)], "yes", (C - 175, 246)))
    b.append(arrow([(C + 150, 256), (R, 256), (R, 300)], "no", (C + 172, 246)))
    b.append(box(L - 150, 300, 300, 74, "Exact text extraction\npdfplumber, python-docx, python-pptx,\nopenpyxl  (+ line, word, table boxes)\nconfidence = 1.0", "green", 10.5))
    b.append(box(R - 150, 300, 300, 56, "Render pages\nPyMuPDF 150 DPI (+ optional clean-up)", "purple", 10.5))
    b.append(arrow([(R, 356), (R, 376)]))
    b.append(box(R - 150, 376, 300, 60, "PaddleOCR text OCR or\nPP-StructureV3 layout parsing\nlines, boxes, words, blocks, tables", "purple", 10.5))
    b.append(arrow([(R, 436), (R, 458)]))
    b.append(diamond(R, 498, 260, 80, "Handwritten?\nhint, or Paddle confidence\nlow and TrOCR reads better"))
    b.append(arrow([(R + 130, 498), (R + 168, 498), (R + 168, 567), (R + 120, 567)], "yes", (R + 160, 488)))
    b.append(box(R - 120, 548, 240, 38, "TrOCR reads each\nPaddle line box", "purple", 10.5))
    b.append(arrow([(R - 130, 498), (R - 172, 498), (R - 172, 610), (C + 60, 610)], "no", (R - 150, 488)))
    b.append(arrow([(R, 586), (R, 610), (C + 60, 610)]))
    b.append(arrow([(L, 374), (L, 610), (C - 60, 610)]))
    b.append(box(C - 160, 594, 320, 48, "Module 3  Calibrate line confidence\napp/calibration (Paddle) / confidence tables", "teal", 10.5))
    b.append(arrow([(C, 642), (C, 662)]))
    b.append(box(C - 160, 662, 320, 40, "Save original extraction  (version 1)", "grey", 11))
    b.append(arrow([(C, 702), (C, 724)]))
    b.append(diamond(C, 760, 230, 72, "Gemini\navailable?"))
    b.append(arrow([(C - 115, 760), (L, 760), (L, 800)], "no", (C - 138, 750)))
    b.append(box(L - 140, 800, 280, 52, "Keep unrefined text\n\"Not refined: Gemini unavailable\" + Retry", "red", 10.5))
    b.append(arrow([(C + 115, 760), (R, 760), (R, 790)], "yes", (C + 138, 750)))
    b.append(box(R - 160, 790, 320, 72, "Refine every line with Gemini\n(refine plugin, JSON per line)\nword diff; reject lines changing > 40%\nsave proposal + refined text (v2, v3)", "amber", 10.5))
    b.append(arrow([(L, 852), (L, 892), (C - 60, 892)]))
    b.append(arrow([(R, 862), (R, 892), (C + 60, 892)]))
    b.append(box(C - 170, 876, 340, 58, "Modules 4-5  Index for questions\nchunks + page/line citations + content type;\nLLM-changed text capped at confidence 0.84", "teal", 10.5))
    b.append(arrow([(C, 934), (C, 954)]))
    b.append(box(C - 170, 954, 340, 44, "Embed (gemini-embedding-001, 768-d)\nstore vectors in ChromaDB; Document row", "blue", 10.5))
    b.append(arrow([(C, 998), (C, 1020)]))
    b.append(pill(C - 170, 1020, 340, 44, "Ready: ask questions, view layout,\nsee LLM changes, export", "green", 11.5))
    return svg(W, H, "".join(b), "Upload pipeline flowchart")


# ------------------------------------------------------------------------------------------------
# 4. Handwriting pipeline
# ------------------------------------------------------------------------------------------------


def handwriting_flow() -> str:
    W, H = 780, 330
    b = []
    top = [
        ("Page image\n(phone photo / scan)", "grey"),
        ("Denoise\nmedian filter", "blue"),
        ("Remove ruled lines\nOpenCV morphology", "blue"),
        ("Whiten background\nshow-through removed", "blue"),
        ("Find line boxes\nPaddleOCR detector", "purple"),
    ]
    bottom = [
        ("Crop each line\n(+15% padding)", "purple"),
        ("TrOCR encoder\nViT reads the image", "purple"),
        ("TrOCR decoder\ngenerates tokens", "purple"),
        ("Line confidence\ngeom. mean of token p\n(temperature T)", "teal"),
        ("Calibrated confidence\nhandwritten table\n(Module 3)", "teal"),
    ]
    bw, bh, gap = 138, 62, 15
    for i, (t, k) in enumerate(top):
        x = 10 + i * (bw + gap)
        b.append(box(x, 20, bw, bh, t, k, 10.5))
        if i:
            b.append(arrow([(x - gap, 51), (x, 51)]))
    b.append(arrow([(10 + 4 * (bw + gap) + bw / 2, 82), (10 + 4 * (bw + gap) + bw / 2, 112), (10 + bw / 2, 112), (10 + bw / 2, 150)]))
    for i, (t, k) in enumerate(bottom):
        x = 10 + i * (bw + gap)
        b.append(box(x, 150, bw, bh + 8, t, k, 10.5))
        if i:
            b.append(arrow([(x - gap, 185), (x, 185)]))
    b.append(label(10, 262, "Why this design: PaddleOCR is a strong line DETECTOR on any page, while TrOCR (trained on handwriting) is the", 11.5, INK))
    b.append(label(10, 282, "better line READER for cursive text. DocBlendAI combines them: Paddle finds where each line is, TrOCR reads it.", 11.5, INK))
    b.append(label(10, 302, "On a test note Paddle alone read \"Opera+ing systems\"; Paddle boxes + TrOCR read \"Operating systems\" correctly.", 11.5, MUTED))
    return svg(W, H, "".join(b), "Handwriting recognition pipeline")


# ------------------------------------------------------------------------------------------------
# 5. LLM refinement
# ------------------------------------------------------------------------------------------------


def refinement_flow() -> str:
    W, H = 780, 820
    C = 390
    b = []
    b.append(pill(C - 190, 10, 380, 48, "Recognised lines with calibrated confidence\n+ page structure (Markdown, tables)", "green", 11))
    b.append(arrow([(C, 58), (C, 80)]))
    b.append(box(C - 190, 80, 380, 44, "Split into parts\n<= 6,000 characters / <= 150 lines per Gemini call", "blue", 10.5))
    b.append(arrow([(C, 124), (C, 146)]))
    b.append(box(C - 190, 146, 380, 72, "Prompt Gemini (refine plugin)\nfix misread words; restore words clearly implied;\nkeep wording, numbers, spacing, tables; no new facts;\nreply JSON: {line_id, refined_text} for every line", "amber", 10.5))
    b.append(arrow([(C, 218), (C, 240)]))
    b.append(diamond(C, 282, 270, 82, "Usable JSON reply?\n(quota / overload / key\nerrors are not retried)"))
    b.append(arrow([(C + 135, 282), (685, 282), (685, 320)], "no", (C + 160, 272)))
    b.append(box(598, 320, 174, 66, "Retry once; then\n\"Not refined: Gemini\nunavailable\" + Retry link", "red", 10.5))
    b.append(arrow([(C, 323), (C, 346)], "yes", (C + 18, 334)))
    b.append(box(C - 190, 346, 380, 52, "Word-level diff per line (text_diff.py)\nequal / replace / insert / delete", "blue", 10.5))
    b.append(arrow([(C, 398), (C, 420)]))
    b.append(diamond(C, 458, 260, 74, "Only spacing changed\n(or nothing)?"))
    b.append(arrow([(C - 130, 458), (150, 458), (150, 495)], "yes", (C - 152, 448)))
    b.append(box(40, 495, 220, 54, "UNCHANGED\nrecognised layout kept", "grey", 10.5))
    b.append(arrow([(C, 495), (C, 518)], "no", (C + 16, 506)))
    b.append(diamond(C, 560, 280, 84, "Change ratio > 0.40?\n(character edit distance /\nlength of the original)"))
    b.append(arrow([(C + 140, 560), (685, 560), (685, 600)], "yes", (C + 163, 550)))
    b.append(box(598, 600, 174, 68, "REJECTED\noriginal text kept;\nline flagged \"needs review\"", "red", 10.5))
    b.append(arrow([(C, 602), (C, 626)], "no", (C + 16, 614)))
    b.append(box(C - 190, 626, 380, 74, "APPLIED  (corrected / inferred)\ntext = refined;  ocr_text = original;\nsource = \"llm\";  llm_diff kept for highlighting\ncorrected words highlighted, inferred words marked", "green", 10.5))
    b.append(arrow([(C, 700), (C, 722)]))
    b.append(box(C - 190, 722, 380, 44, "Save versions: original, proposal, refined text\n(revert / use refined text / compare / restore)", "grey", 10.5))
    b.append(arrow([(C, 766), (C, 786)]))
    b.append(pill(C - 190, 786, 380, 30, "Re-index for QA: LLM text counts at most 0.84 (Moderate)", "teal", 10.5))
    return svg(W, H, "".join(b), "Guarded LLM refinement flowchart")


# ------------------------------------------------------------------------------------------------
# 6. Question answering (RAG)
# ------------------------------------------------------------------------------------------------


def rag_flow() -> str:
    W, H = 780, 980
    C = 390
    b = []
    b.append(pill(C - 180, 10, 360, 46, "Question  (+ last 3 chat turns, + ticked documents)", "green", 11.5))
    b.append(arrow([(C + 180, 33), (680, 33), (680, 70)]))
    b.append(box(590, 70, 182, 60, "In parallel: \"Did you mean\"\nfuzzy match to document words,\nGemini rewrite (validated)", "grey", 10))
    b.append(arrow([(C, 56), (C, 80)]))
    b.append(box(C - 180, 80, 360, 44, "Module 5  Embed the question\ngemini-embedding-001 (RETRIEVAL_QUERY, 768-d)", "blue", 10.5))
    b.append(arrow([(C, 124), (C, 146)]))
    b.append(box(C - 180, 146, 360, 52, "Vector search in ChromaDB (cosine)\ntop_k x 3 = 15 candidates, only the ticked documents", "blue", 10.5))
    b.append(arrow([(C, 198), (C, 220)]))
    b.append(box(C - 180, 220, 360, 52, "Follow-up? also search  previous question + this one\nkeep the better-scored chunks", "blue", 10.5))
    b.append(arrow([(C, 272), (C, 294)]))
    b.append(box(C - 180, 294, 360, 60, "Confidence-aware re-ranking\ncombined = 0.7 x similarity + 0.3 x calibrated confidence\nidentical passages kept once (best copy)", "teal", 10.5))
    b.append(arrow([(C, 354), (C, 376)]))
    b.append(box(C - 180, 376, 360, 52, "Share the 5 slots between relevant documents\n(similarity within 0.05 of the best match)", "teal", 10.5))
    b.append(arrow([(C, 428), (C, 450)]))
    b.append(box(C - 180, 450, 360, 52, "Module 6  Reliability tier from the top passage\nCertain / Moderate / Uncertain / Unreadable", "purple", 10.5))
    b.append(arrow([(C, 502), (C, 524)]))
    b.append(box(C - 180, 524, 360, 86, "Module 7  Prompt Gemini (gemini-3.5-flash-lite)\nnumbered passages + content type; passages with\nconfidence < 0.95 marked \"may contain recognition errors\";\nuse only the passages; reply NOT_FOUND if absent;\nearlier turns only to resolve \"it / that\"", "amber", 10.5))
    b.append(arrow([(C, 610), (C, 632)]))
    b.append(diamond(C, 672, 260, 78, "Answer = NOT_FOUND?"))
    b.append(arrow([(C - 130, 672), (110, 672), (110, 712)], "yes", (C - 152, 662)))
    b.append(box(14, 712, 192, 64, "\"I couldn't find the answer\"\nlabel lowered to at\nleast Uncertain", "red", 10.5))
    b.append(arrow([(C, 711), (C, 740)], "no", (C + 16, 724)))
    b.append(box(C - 180, 740, 360, 44, "Gemini busy / out of quota?\nretry, then the fallback model (gemini-3.5-flash)", "grey", 10.5))
    b.append(arrow([(110, 776), (110, 838), (C - 180, 838)]))
    b.append(arrow([(C, 784), (C, 812)]))
    b.append(box(C - 180, 812, 360, 52, "Store Query, Answer, RetrievalResults (SQLite)\nso sources can be shown and audited", "grey", 10.5))
    b.append(arrow([(C, 864), (C, 888)]))
    b.append(pill(C - 180, 888, 360, 60, "Answer + reliability tier + sources\n(document, page, lines, similarity, readability;\nLLM-corrected words marked)", "green", 10.5))
    return svg(W, H, "".join(b), "Question answering flowchart")


# ------------------------------------------------------------------------------------------------
# 7. Reliability decision
# ------------------------------------------------------------------------------------------------


def reliability_tree() -> str:
    W, H = 780, 400
    b = []
    b.append(box(20, 20, 210, 56, "Top passage\nsimilarity s, confidence c", "blue", 11))
    b.append(arrow([(230, 48), (262, 48)]))
    b.append(diamond(360, 48, 196, 70, "c < 0.35 ?"))
    b.append(arrow([(458, 48), (560, 48)], "yes", (505, 38)))
    b.append(box(560, 22, 200, 52, "UNREADABLE\ntext could not be read", "red", 11))
    b.append(arrow([(360, 83), (360, 120)], "no", (378, 100)))
    b.append(diamond(360, 160, 230, 78, "s >= 0.65 and\nc >= 0.85 ?"))
    b.append(arrow([(475, 160), (560, 160)], "yes", (515, 150)))
    b.append(box(560, 134, 200, 52, "CERTAIN\nstrong match, clear text", "green", 11))
    b.append(arrow([(360, 199), (360, 236)], "no", (378, 216)))
    b.append(diamond(360, 276, 230, 78, "s >= 0.58 and\nc >= 0.60 ?"))
    b.append(arrow([(475, 276), (560, 276)], "yes", (515, 266)))
    b.append(box(560, 250, 200, 52, "MODERATE\nlooser match or partly legible", "amber", 11))
    b.append(arrow([(360, 315), (360, 345)], "no", (378, 330)))
    b.append(box(260, 345, 200, 46, "UNCERTAIN\ncheck the source", "grey", 11))
    b.append(label(20, 140, "LLM-changed text:\nconfidence capped at\n0.84, so evidence\nresting on it can be\nat most MODERATE.\nNOT_FOUND answers:\nat least UNCERTAIN.", 11.5, PURPLE))
    return svg(W, H, "".join(b), "Reliability tier decision")


# ------------------------------------------------------------------------------------------------
# 8. Confidence journey
# ------------------------------------------------------------------------------------------------


def confidence_journey() -> str:
    W, H = 780, 250
    b = []
    steps = [
        ("Raw engine score\nTesseract word conf. /\nTrOCR token prob. /\nPaddle line score", "grey"),
        ("Calibration\nper engine: tables,\ntemperature, Platt,\nisotonic (min. ECE)", "purple"),
        ("Line confidence\n~ probability the line\nis read correctly", "teal"),
        ("Chunk confidence\nchar-weighted mean;\nLLM text <= 0.84", "teal"),
        ("Combined score\n0.7 sim + 0.3 conf\n(ranking)", "blue"),
        ("Reliability tier\nCertain / Moderate /\nUncertain / Unreadable", "green"),
    ]
    bw, gap = 116, 13
    for i, (t, k) in enumerate(steps):
        x = 8 + i * (bw + gap)
        b.append(box(x, 30, bw, 96, t, k, 10))
        if i:
            b.append(arrow([(x - gap, 78), (x, 78)]))
    b.append(label(8, 160, "Every number the user sees as \"readability\" has the same meaning across engines: after calibration a 0.9 means", 11.5, INK))
    b.append(label(8, 180, "\"about 9 in 10 characters / lines are right\", whether the page was typed, printed, scanned or handwritten. That is what", 11.5, INK))
    b.append(label(8, 200, "lets one ranking formula and one set of reliability thresholds work for all formats.", 11.5, INK))
    return svg(W, H, "".join(b), "Confidence journey")


# ------------------------------------------------------------------------------------------------
# 9. Storage
# ------------------------------------------------------------------------------------------------


def _table(x, y, w, name, fields, kind):
    h = 30 + 16 * len(fields)
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#fff" stroke="{STROKE[kind]}" stroke-width="1.4"/>',
           f'<rect x="{x}" y="{y}" width="{w}" height="22" rx="6" fill="{FILL[kind]}" stroke="{STROKE[kind]}" stroke-width="1.4"/>',
           f'<text x="{x + 8}" y="{y + 15}" font-size="11.5" font-weight="bold" fill="{STROKE[kind]}">{escape(name)}</text>']
    for i, f in enumerate(fields):
        out.append(f'<text x="{x + 8}" y="{y + 38 + i * 16}" font-size="10.5" fill="{INK}">{escape(f)}</text>')
    return "".join(out), h


def storage() -> str:
    W, H = 780, 450
    b = []
    tables = [
        (10, 10, "documents  (synopsis Document)", ["doc_id  PK", "file_path", "format_type", "page_count"], "blue"),
        (10, 130, "queries  (synopsis Query)", ["query_id  PK", "question_text", "user_id"], "blue"),
        (10, 245, "answers  (synopsis Answer)", ["answer_id  PK", "query_id  FK", "answer_text", "reliability_label"], "blue"),
        (10, 360, "retrieval_results", ["query_id  FK, chunk_id", "similarity, confidence,", "combined_score"], "blue"),
        (270, 10, "document_runs  (link)", ["doc_id  PK -> documents", "run_id  -> ocr_runs"], "green"),
        (270, 110, "ocr_runs  (Experience Center)", ["id  PK", "filename, pipeline, status", "page_count, mean_confidence", "full_text (search)", "result_json (pages, lines,", "  boxes, confidences, ...)"], "green"),
        (270, 300, "run_versions", ["id, run_id, seq, kind", "extraction | refinement |", "  edit | restore", "data_json (full text state)"], "green"),
        (540, 10, "ChromaDB  collection \"chunks\"", ["id = <doc_id>:<n>", "embedding (768 floats)", "document (chunk text)", "doc_id, page, first_line,", "  last_line, content_type,", "  raw_conf, calibrated_conf"], "purple"),
        (540, 180, "Files  data/", ["uploads/<doc_id>_<name>", "paddle_runs/<run>/page-n.png", "paddle_calibration/", "  calibrator.json, report", "calibration.json (tables)", "docblendai.db (SQLite)"], "amber"),
    ]
    for x, y, name, fields, kind in tables:
        t, _ = _table(x, y, 230, name, fields, kind)
        b.append(t)
    b.append(arrow([(240, 50), (270, 50)], color=MUTED))
    b.append(arrow([(385, 66), (385, 110)], color=MUTED))
    b.append(arrow([(385, 236), (385, 300)], color=MUTED))
    b.append(arrow([(125, 200), (125, 245)], color=MUTED))
    b.append(arrow([(125, 330), (125, 360)], color=MUTED))
    b.append(arrow([(500, 160), (540, 90)], color=MUTED, dashed=True))
    b.append(label(268, 412, "Synopsis entities (schemas.py) are unchanged; the Experience Center adds only new tables.", 11, MUTED))
    return svg(W, H, "".join(b), "Storage model")


# ------------------------------------------------------------------------------------------------
# 10. Deployment
# ------------------------------------------------------------------------------------------------


def deployment() -> str:
    W, H = 780, 300
    b = []
    b.append(band(10, 10, 360, 270, "Option A: local Python venv (development)", "blue"))
    b.append(box(30, 40, 320, 50, "uvicorn app.main:app --reload\nFastAPI app, one process, port 8000", "white", 10.5))
    b.append(box(30, 100, 320, 70, "venv/  Python 3.11\npaddlepaddle 3.3.1, paddleocr 3.7.0,\ntorch (TrOCR, docTR), chromadb, ...", "white", 10.5))
    b.append(box(30, 180, 155, 80, "~/.paddlex,\n~/.cache\nmodel weights", "white", 10.5))
    b.append(box(195, 180, 155, 80, "data/\nSQLite, ChromaDB,\nuploads, runs", "white", 10.5))
    b.append(band(410, 10, 360, 270, "Option B: Docker (other computers)", "green"))
    b.append(box(430, 40, 320, 50, "docker compose up -d\ncontainer docblendai, port 8000, health check", "white", 10.5))
    b.append(box(430, 100, 320, 70, "Image (multi-stage, python:3.11-slim)\nCPU torch, paddle, app code; non-root user;\noptional PRELOAD_MODELS=true", "white", 10.5))
    b.append(box(430, 180, 155, 80, "volume\nmodel-cache\n(/models)", "white", 10.5))
    b.append(box(595, 180, 155, 80, "bind mount\n./data\n(same data)", "white", 10.5))
    return svg(W, H, "".join(b), "Deployment options")


# ------------------------------------------------------------------------------------------------
# 11. Context diagram (big picture)
# ------------------------------------------------------------------------------------------------


def context() -> str:
    W, H = 780, 300
    b = []
    b.append(actor(60, 70, "Student /\nFaculty"))
    b.append(box(200, 40, 380, 210, "", "white", 11))
    b.append(label(390, 62, "DocBlendAI", 15, INK, "bold", "middle"))
    inner = [("Read any document\ntyped / scanned / handwritten", 82), ("Refine the text with an LLM\n(guarded, reversible)", 132),
             ("Answer questions with a\nreliability tier and sources", 182)]
    for t, y in inner:
        b.append(box(225, y, 330, 42, t, "blue", 11))
    b.append(arrow([(95, 110), (200, 110)], "documents", (148, 98)))
    b.append(arrow([(200, 200), (95, 200)], "answers + trust", (148, 188)))
    b.append(box(650, 50, 120, 70, "Google Gemini\nLLM +\nembeddings", "amber", 10.5))
    b.append(box(650, 170, 120, 70, "Local models\nPaddleOCR,\nTrOCR, docTR", "purple", 10.5))
    b.append(arrow([(580, 110), (650, 90)], color=MUTED))
    b.append(arrow([(580, 190), (650, 205)], color=MUTED))
    b.append(label(390, 280, "Everything runs on the user's computer except the Gemini calls.", 11.5, MUTED, anchor="middle"))
    return svg(W, H, "".join(b), "Context diagram")
