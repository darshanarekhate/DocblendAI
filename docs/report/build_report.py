"""Build the DocBlendAI project explanation (HTML + PDF).

    venv/Scripts/python docs/report/build_report.py

Writes docs/report/DocBlendAI_Project_Explanation.html and prints it to
docs/report/DocBlendAI_Project_Explanation.pdf with headless Microsoft Edge (or Chrome).
"""

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import diagrams as d  # noqa: E402

OUT = Path(__file__).parent
HTML = OUT / "DocBlendAI_Project_Explanation.html"
PDF = OUT / "DocBlendAI_Project_Explanation.pdf"

CSS = """
@page { size: A4; margin: 16mm 15mm 18mm 15mm; }
* { box-sizing: border-box; }
body { font-family: 'Segoe UI', Arial, sans-serif; color: #1f2a37; font-size: 10.6pt; line-height: 1.5; margin: 0; }
h1 { font-size: 22pt; margin: 0 0 6px; color: #1d4f8c; }
h2 { font-size: 15pt; color: #1d4f8c; border-bottom: 2px solid #c9d8ec; padding-bottom: 4px; margin: 22px 0 10px; break-after: avoid; }
h3 { font-size: 12pt; color: #2a3b4f; margin: 16px 0 6px; break-after: avoid; }
p { margin: 5px 0 8px; }
ul, ol { margin: 4px 0 8px 20px; padding: 0; }
li { margin: 2px 0; }
.chapter { break-before: page; }
.cover { height: 255mm; display: flex; flex-direction: column; justify-content: center; text-align: center; }
.cover .sub { font-size: 13pt; color: #5b6775; margin: 4px 0; }
.cover .tag { display: inline-block; margin: 18px auto 0; padding: 8px 16px; border-radius: 20px; background: #e6eff9; color: #1d4f8c; font-weight: 600; }
.cover .meta { margin-top: 40px; font-size: 11pt; line-height: 1.8; }
.diagram { width: 100%; height: auto; display: block; margin: 6px 0 4px; }
figure { margin: 8px 0 14px; break-inside: avoid; }
figcaption { font-size: 9.2pt; color: #5b6775; text-align: center; font-style: italic; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 12px; font-size: 9.5pt; break-inside: auto; }
th, td { border: 1px solid #c8d1dc; padding: 4px 6px; vertical-align: top; text-align: left; }
th { background: #eef3f9; color: #1d4f8c; }
tr { break-inside: avoid; }
code, .mono { font-family: Consolas, 'Courier New', monospace; font-size: 9.2pt; background: #f2f4f7; padding: 0 3px; border-radius: 3px; }
pre { font-family: Consolas, monospace; font-size: 8.8pt; background: #f6f8fa; border: 1px solid #dde3ea; border-radius: 6px; padding: 8px 10px; white-space: pre-wrap; break-inside: avoid; }
.box { border-left: 4px solid #2f6db5; background: #f2f7fc; padding: 8px 12px; margin: 8px 0 12px; border-radius: 4px; break-inside: avoid; }
.novel { border-left-color: #2f7d5b; background: #eef8f2; }
.warn { border-left-color: #b7791f; background: #fdf6e7; }
.toc li { margin: 3px 0; }
.tier { display: inline-block; padding: 1px 8px; border-radius: 10px; font-weight: 600; font-size: 9.2pt; }
.t-c { background: #e5f3ec; color: #2f7d5b; } .t-m { background: #fdf1dc; color: #9a6413; }
.t-u { background: #f1f3f6; color: #4a5562; } .t-x { background: #fbe7e5; color: #b3261e; }
.small { font-size: 9.2pt; color: #5b6775; }
.two { display: flex; gap: 14px; } .two > div { flex: 1; }
"""


def fig(svg: str, caption: str) -> str:
    return f"<figure>{svg}<figcaption>{caption}</figcaption></figure>"


def table(head: list[str], rows: list[list[str]]) -> str:
    h = "".join(f"<th>{c}</th>" for c in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>"


def build() -> str:
    s = []
    # ------------------------------------------------------------------ cover
    s.append("""
<div class="cover">
  <div class="sub">B.Tech Final-Year Project &middot; Group C4</div>
  <h1 style="font-size:34pt;margin:14px 0 4px">DocBlendAI</h1>
  <div class="sub" style="font-size:15pt">A Confidence-Aware Multi-Format Document Question-Answering Assistant</div>
  <div class="tag">Project explanation: architecture, models, flowcharts and novelty</div>
  <div class="meta">
    Guide: Prof. R. A. Tiwari<br>
    Department of Computer Science &amp; Engineering<br>
    PRMIT&amp;R, Badnera<br><br>
    <span class="small">Typed &middot; Scanned / printed &middot; Handwritten documents &rarr; trustworthy answers</span>
  </div>
</div>""")

    # ------------------------------------------------------------------ contents
    s.append("""
<div class="chapter"><h2>Contents</h2>
<ol class="toc">
<li>Project in one page (problem, idea, result)</li>
<li>Novelty: what DocBlendAI adds</li>
<li>Use case diagram: who does what</li>
<li>Layered architecture: each layer explained</li>
<li>The seven synopsis modules</li>
<li>What happens internally when a document is uploaded</li>
<li>How text is extracted: typed, scanned, handwritten, tables</li>
<li>Confidence capture and calibration</li>
<li>How the LLM works (1): guarded refinement of extracted text</li>
<li>How the LLM works (2): retrieval-augmented question answering</li>
<li>Reliability tiers</li>
<li>Models used in the project</li>
<li>Data storage</li>
<li>Experience Center, exports and the user interface</li>
<li>Evaluation and results</li>
<li>Deployment, testing and engineering practice</li>
<li>Limitations and future work</li>
<li>Glossary</li>
</ol></div>""")

    # ------------------------------------------------------------------ 1 overview
    s.append(f"""
<div class="chapter"><h2>1. Project in one page</h2>
<h3>The problem</h3>
<p>Students, researchers and faculty study from material in many forms: typed PDFs and slides, printed pages
that were scanned or photographed with a phone, and handwritten class notes. Ordinary &ldquo;chat with your PDF&rdquo;
tools assume clean digital text. When the text comes from OCR or handwriting recognition it contains reading
errors, yet the tool answers just as confidently. The user cannot tell whether an answer rests on clean text
or on a guess built from badly read words.</p>
<h3>The idea</h3>
<p>DocBlendAI reads any academic document, measures how well each line was read, and carries that measurement
all the way to the answer. Every answer comes with a <b>reliability tier</b>
(<span class="tier t-c">Certain</span> <span class="tier t-m">Moderate</span> <span class="tier t-u">Uncertain</span>
<span class="tier t-x">Unreadable</span>) and with the exact page and lines it came from.</p>
{fig(d.context(), "Figure 1. Context diagram: what goes in, what comes out, what runs where.")}
<h3>What the system does, step by step</h3>
<ol>
<li><b>Detects the format</b> of an uploaded file (typed, scanned/printed, handwritten) and corrects the page orientation.</li>
<li><b>Extracts the text</b> with the right method: exact parsing for digital files, PaddleOCR for printed pages,
PaddleOCR line detection + TrOCR reading for handwriting, and PP-StructureV3 for layout and tables.</li>
<li><b>Calibrates the confidence</b> of each recognised line so that, for example, 0.9 really means &ldquo;about 90% right&rdquo;.</li>
<li><b>Refines the text with an LLM</b> (Gemini) automatically, but under strict guards, keeping the original so any change can be reverted.</li>
<li><b>Indexes</b> the text in chunks with page/line citations, content type (table / paragraph / image) and confidence.</li>
<li><b>Answers questions</b> with confidence-aware retrieval (similarity <i>and</i> readability) and Gemini, and labels each answer with a reliability tier.</li>
</ol>
<div class="box"><b>In one sentence:</b> DocBlendAI is a retrieval-augmented question-answering system that treats
<i>&ldquo;how well was this text read?&rdquo;</i> as a first-class signal, from the OCR engine to the final answer.</div>
<h3>Scope</h3>
<p><b>In scope:</b> typed, scanned/printed and handwritten academic documents uploaded as PDF, images
(JPG/PNG/TIFF/BMP/WEBP, including phone photos), Word, PowerPoint, Excel or text files.
<b>Out of scope:</b> live classroom transcription, non-Latin scripts, video.</p>
<h3>Technology stack</h3>
{table(["Area", "Technology"], [
    ["Backend", "Python 3.11, FastAPI (one monolithic application), Uvicorn"],
    ["Recognition", "PaddleOCR 3.7 (PP-OCRv5, PP-StructureV3), TrOCR (Hugging Face transformers), docTR, Tesseract, OpenCV"],
    ["Language model", "Google Gemini via the google-genai SDK: gemini-3.5-flash-lite (answers, refinement), gemini-embedding-001 (vectors)"],
    ["Storage", "SQLite through SQLAlchemy (relational data), ChromaDB (vectors)"],
    ["Frontend", "Plain HTML, CSS and JavaScript (no framework), served by FastAPI"],
    ["Evaluation", "jiwer (CER, WER), own calibration metrics (ECE, MCE, Brier, NLL)"],
    ["Deployment", "Python virtual environment or Docker / Docker Compose"],
])}
</div>""")

    # ------------------------------------------------------------------ 2 novelty
    nov = [
        ("Confidence-aware retrieval (core idea)",
         "Normal RAG ranks passages by semantic similarity only. DocBlendAI ranks by "
         "<code>combined = 0.7 &times; similarity + 0.3 &times; calibrated confidence</code>, so a clearly read passage beats an "
         "equally relevant but badly recognised one. Recognition quality becomes part of retrieval, not an afterthought."),
        ("Four-tier reliability label on every answer",
         "The answer is labelled Certain / Moderate / Uncertain / Unreadable by weighing <i>both</i> retrieval relevance "
         "and recognition confidence of the evidence. The user sees why, through cited sources with similarity and readability values."),
        ("Calibrated confidence that means the same for every engine",
         "Raw scores of Tesseract, TrOCR and PaddleOCR are not comparable (TrOCR is over-confident). DocBlendAI fits temperature "
         "scaling (Ayllon et al., ICDAR 2024), Platt scaling and isotonic regression, chooses the one with the lowest validation "
         "Expected Calibration Error (ECE), and reports ECE, MCE, Brier score and NLL before and after. One threshold set then works for all formats."),
        ("Hybrid handwriting recognition",
         "PaddleOCR is used as a line <i>detector</i>, TrOCR (trained on handwriting) as the line <i>reader</i>. The engine is chosen "
         "automatically by comparing calibrated confidences. Ruled notebook lines are erased with OpenCV first. Example: Paddle alone read "
         "&ldquo;Opera+ing systems&rdquo;; the hybrid read &ldquo;Operating systems&rdquo;."),
        ("Guarded, reversible automatic LLM refinement",
         "Gemini corrects recognition errors line by line, but: it must answer in JSON per line; each change is word-diffed and labelled "
         "<i>corrected</i> or <i>inferred</i>; a line changed by more than 40% is rejected; the original is kept as a version "
         "(revert / reapply / retry); and LLM-written text can never raise reliability (capped at 0.84, i.e. at most Moderate). "
         "When Gemini is unavailable the system keeps working with the unrefined text."),
        ("Format-aware unified extraction",
         "One pipeline detects whether a file is typed, scanned or handwritten and routes it to exact parsing, OCR or HTR, while producing "
         "the same output structure (pages, lines, boxes, words, confidences, blocks, tables) for all of them."),
        ("Content-type identification from layout",
         "PP-StructureV3 layout blocks decide whether a chunk is a table, paragraph or image; tables keep their rows and cells, so table "
         "questions retrieve whole rows. A rule-based classifier is the fallback."),
        ("Traceable answers: page and line citations",
         "Each chunk stores page, first line and last line, so every source under an answer says exactly where it came from, and words "
         "corrected by the LLM are marked in the source."),
        ("One shared document library, two views",
         "The QA page and the Experience Center (OCR studio) work on the same documents and versions: a correction made in one is re-indexed "
         "and used by the other immediately."),
        ("Layout-preserving view and export",
         "Text is shown and exported in its original position on the page (columns, tables), not as a flat paragraph."),
    ]
    items = "".join(f'<div class="box novel"><b>N{i}. {t}</b><br>{x}</div>' for i, (t, x) in enumerate(nov, 1))
    s.append(f"""
<div class="chapter"><h2>2. Novelty: what DocBlendAI adds</h2>
<p>Existing tools usually do <i>one</i> of these things: OCR (Tesseract, PaddleOCR), handwriting recognition (TrOCR) or
question answering over digital text (RAG chatbots). None of them carries the recognition uncertainty into the answer.
The table compares DocBlendAI with a typical &ldquo;chat with PDF&rdquo; system.</p>
{table(["Aspect", "Typical RAG chatbot", "DocBlendAI"], [
    ["Input formats", "Digital PDF text only", "Typed, scanned, photographed, handwritten; PDF, images, Office, text"],
    ["Recognition errors", "Ignored (scanned PDFs often give no text)", "Measured per line, calibrated, shown to the user"],
    ["Ranking of passages", "Similarity only", "Similarity + calibrated confidence"],
    ["Trust in the answer", "None shown", "4-tier reliability + cited page/lines + readability"],
    ["LLM use on the text", "None, or silent rewriting", "Guarded refinement: diffed, capped, reversible, never raises trust"],
    ["Handwriting", "Not supported", "Paddle detection + TrOCR reading, ruled-line removal"],
    ["Tables / layout", "Lost (flat text)", "Detected, kept as rows/cells, layout view and export"],
    ["Unanswerable questions", "Often hallucinated", "NOT_FOUND protocol, labelled at least Uncertain"],
])}
<h3>Novel contributions in detail</h3>
{items}
</div>""")

    # ------------------------------------------------------------------ 3 use case
    s.append(f"""
<div class="chapter"><h2>3. Use case diagram: who does what</h2>
{fig(d.use_cases(), "Figure 2. Use case diagram. Blue: user goals; purple: developer goals; green: internal use cases included by them.")}
<h3>Actors</h3>
{table(["Actor", "Role"], [
    ["Student / Researcher / Faculty", "Uploads documents, reads them, asks questions, checks sources, reviews LLM changes, exports text."],
    ["Developer / Admin", "Uses the Experience Center for detailed OCR work, fits calibration on labelled data, runs the evaluation."],
    ["Google Gemini (external)", "Creates embeddings, refines recognised text, writes answers, suggests spelling corrections."],
    ["OCR / HTR models (local)", "PaddleOCR, PP-StructureV3, TrOCR, docTR, Tesseract running on the user's computer."],
])}
<h3>Main use cases</h3>
{table(["Use case", "What happens internally"], [
    ["Upload a document", "File is checked and saved; a background job extracts the text, refines it with Gemini, chunks, embeds and indexes it. Progress is shown live (&ldquo;Extracting page 2 of 5&hellip;&rdquo;, &ldquo;Refining with Gemini&hellip;&rdquo;, &ldquo;Indexing for questions&hellip;&rdquo;, &ldquo;Ready&rdquo;)."],
    ["View document", "Page images with recognised line boxes, a layout view that keeps the original positions, or the original file."],
    ["Choose documents", "Tick-boxes limit retrieval to the selected documents (ChromaDB metadata filter on doc_id)."],
    ["Ask / chat", "Embedding, vector search, confidence-aware re-ranking, reliability tier, Gemini answer; the last 3 turns are sent for follow-ups."],
    ["See answer & sources", "Answer text, tier badge, and up to 5 sources with document, page, lines, similarity and readability."],
    ["&ldquo;Did you mean&rdquo;", "Misspelled question words are matched against the document vocabulary (fuzzy), optionally refined by Gemini."],
    ["Review LLM changes", "Changed words are highlighted; the user can show the original, revert, use the refined text again or retry refinement."],
    ["Correct a line", "Manual edit in the Experience Center; saved as a new version and the document is re-indexed."],
    ["Export", "TXT, layout TXT, Markdown, JSON, searchable PDF, DOCX, CSV (tables)."],
    ["Clean up image", "Deskew, denoise, contrast, binarise, remove ruled lines before OCR."],
    ["Fit calibration", "Labelled lines are used to fit temperature / Platt / isotonic calibrators; best by validation ECE."],
    ["Run evaluation", "Scripts in evaluation/ compute CER, WER, Exact/Semantic match, calibration error and tier accuracy."],
])}
</div>""")

    # ------------------------------------------------------------------ 4 layers
    s.append(f"""
<div class="chapter"><h2>4. Layered architecture: each layer explained</h2>
<p>DocBlendAI is a <b>single FastAPI monolith</b>. Modules are separated by Python files and folders, not by network
services, which keeps setup and the demo simple while four team members could still work on separate modules.</p>
{fig(d.architecture(), "Figure 3. The six layers of DocBlendAI. Requests flow downwards; results flow back up.")}
<h3>Layer 1: Presentation</h3>
<p>Two web pages written in plain HTML/CSS/JavaScript and served by FastAPI itself (no Node build step):</p>
<ul>
<li><b>QA page</b> (<code>/</code>, <code>index.html</code>): upload, document list with tick-boxes and live progress, document viewer
(pages with boxes, layout view, original), chat with reliability badges and sources, &ldquo;LLM changes&rdquo; panel.</li>
<li><b>Experience Center</b> (<code>/studio</code>): detailed OCR workspace: page image with clickable line boxes, tabs for text,
layout, Markdown, tables, JSON, confidence, versions; image clean-up; exports; calibration; plugins.</li>
<li><b>Shared layout module</b> (<code>layout.js</code>): draws text in its page position, marks LLM-corrected and inferred words,
and safely renders Markdown and HTML tables (sanitised).</li>
</ul>
<h3>Layer 2: API</h3>
<p>FastAPI routers expose REST endpoints with automatic validation (Pydantic) and documentation (<code>/docs</code>):</p>
{table(["Router", "Main endpoints", "Purpose"], [
    ["upload.py", "POST /upload, GET /documents, /documents/{{id}}/file, DELETE", "Upload, list, download, delete documents"],
    ["query.py", "POST /ask, /chat, /ask/suggest; GET /answer/{{id}}/sources", "Question answering and sources"],
    ["preview.py", "/documents/{{id}}/preview, /pages, /view", "Page images and viewer data"],
    ["paddleocr.py", "/api/ocr, /api/parse, /api/results, /api/calibrate", "Experience Center OCR / parsing jobs, calibration"],
    ["refine.py", "/changes, /revert, /reapply, /refine, /versions", "LLM changes and version history"],
    ["studio_tools.py", "/export, /preprocess, /plugins", "Exports, image clean-up, plugins (refine, KIE, translate)"],
])}
<h3>Layer 3: Pipeline</h3>
<ul>
<li><b>ocr_jobs.py</b>: a job manager with one worker thread. Each upload becomes a job that runs extraction &rarr; refinement &rarr; indexing
and reports progress. Heavy models are loaded once and reused.</li>
<li><b>document_extract.py</b>: Module 2, chooses typed / scanned / handwritten extraction and returns one common result structure.</li>
<li><b>refinement.py</b> + <b>text_diff.py</b>: the guarded LLM refinement and the word-level diff.</li>
<li><b>qa_index.py</b>: turns a result into chunks with content type, confidence (LLM cap) and page/line citation, then embeds and stores them.</li>
<li><b>document_runs.py</b>: links a QA document to its Experience Center run, so both pages share one library.</li>
</ul>
<h3>Layer 4: Intelligence (models and algorithms)</h3>
<ul>
<li><b>Recognition:</b> PaddleOCR (PP-OCRv5 text detection + recognition), PP-StructureV3 (layout, tables, formulas), TrOCR (handwriting),
docTR and Tesseract (classic engine), OpenCV preprocessing.</li>
<li><b>Confidence:</b> <code>confidence_capture.py</code> (raw scores per engine, temperature scaling) and <code>app/calibration</code>
(temperature, Platt, isotonic; ECE-based selection).</li>
<li><b>Retrieval and trust:</b> <code>embedder.py</code> (Gemini embeddings), <code>vector_store.py</code> (ChromaDB),
<code>retrieval.py</code> (combined score), <code>reliability.py</code> (4 tiers).</li>
<li><b>Language model:</b> <code>llm_answer.py</code> (grounded answers, fallback model), plugins (refine, key-information extraction,
translate), <code>spelling.py</code> (&ldquo;Did you mean&rdquo;).</li>
</ul>
<h3>Layer 5: Storage</h3>
<p><b>SQLite</b> keeps everything relational (documents, questions, answers, retrieval results, OCR runs, versions);
<b>ChromaDB</b> keeps only the chunk vectors and their metadata; the <b>data/</b> folder keeps uploaded files, rendered
page images and calibration files. SQLite needs no server, so every team laptop runs the same setup.</p>
<h3>Layer 6: External services</h3>
<p>Only Google Gemini is called over the network (embeddings, refinement, answers). Model weights for PaddleOCR, TrOCR and docTR
are downloaded once from their hubs and cached; after that recognition runs offline.</p>
</div>""")

    # ------------------------------------------------------------------ 5 modules
    s.append(f"""
<div class="chapter"><h2>5. The seven synopsis modules</h2>
<p>The approved synopsis defines seven modules. The code keeps the same names and boundaries.</p>
{table(["#", "Module", "Main files", "Responsibility"], [
    ["1", "Document Upload", "routers/upload.py, file_types.py", "Accept PDF, images, Office, text; check type/size; store; start the job"],
    ["2", "Format Detection &amp; Text Extraction", "document_extract.py, text_parser.py, paddleocr_service.py, htr_extractor.py, ocr_extractor.py, preprocess.py, line_segmentation.py", "Decide typed / scanned / handwritten; route to parsing / OCR / HTR; correct orientation"],
    ["3", "Confidence Capture &amp; Calibration", "confidence_capture.py, app/calibration/", "Raw line confidence per engine; calibrated probability"],
    ["4", "Content-Type Identification", "content_type.py, qa_index.py", "Table / paragraph / image for each chunk (layout first, rules as fallback)"],
    ["5", "Confidence-Aware Retrieval (RAG)", "embedder.py, vector_store.py, retrieval.py", "Embed, search, re-rank by combined score"],
    ["6", "Reliability Tier Classification", "reliability.py", "Certain / Moderate / Uncertain / Unreadable"],
    ["7", "LLM Answer Generation", "llm_answer.py", "Grounded Gemini answer, NOT_FOUND, fallback model"],
])}
<h3>Synopsis entities (unchanged in <code>schemas.py</code>)</h3>
{table(["Entity", "Fields"], [
    ["Document", "doc_id, file_path, format_type, page_count"],
    ["RecognizedChunk", "chunk_id, text, content_type, raw_conf, calibrated_conf, vector"],
    ["Query", "query_id, question_text, user_id"],
    ["RetrievalResult", "chunk_id, similarity, confidence, combined_score"],
    ["Answer", "answer_id, answer_text, reliability_label"],
])}
</div>""")

    # ------------------------------------------------------------------ 6 upload flow
    s.append(f"""
<div class="chapter"><h2>6. What happens internally when a document is uploaded</h2>
{fig(d.upload_flow(), "Figure 4. Upload pipeline: one background job per document.")}
<h3>Explanation of each step</h3>
<ol>
<li><b>Validate and store.</b> The extension and size are checked against <code>file_types.py</code>; the file is saved under
<code>data/uploads/</code> with a new <code>doc_id</code>.</li>
<li><b>Background job.</b> The request returns immediately; a job (an <code>ocr_runs</code> row) runs in a worker thread. Both pages
poll its progress, so the user sees the document appear with &ldquo;Extracting page i of n&hellip;&rdquo;.</li>
<li><b>Typed check.</b> Word, PowerPoint, Excel, text files and PDFs that already contain a text layer are <i>typed</i>: their text is
read exactly (confidence 1.0), with line and word boxes and tables from pdfplumber.</li>
<li><b>OCR path.</b> Other PDFs are rendered at 150 DPI with PyMuPDF; images are loaded directly. Orientation is corrected and
PaddleOCR (or PP-StructureV3 for layout and tables) recognises lines with boxes and scores.</li>
<li><b>Handwriting decision.</b> If the user marked the file as handwritten, or Paddle's calibrated confidence is low (&lt; 0.6),
TrOCR reads a sample of Paddle's line boxes. Whichever engine reaches the higher calibrated confidence wins.</li>
<li><b>Calibration.</b> Every line gets a raw and a calibrated confidence (Section 8).</li>
<li><b>Original version.</b> The unrefined result is saved as version 1, so it can always be restored.</li>
<li><b>Guarded LLM refinement</b> (Section 9). If Gemini is unavailable the document is still usable, with a clear
&ldquo;Not refined&rdquo; notice and a Retry link.</li>
<li><b>Indexing.</b> Text is split into chunks of about 1000 characters with 200 overlap, each with page/line citation, content
type and confidence; LLM-written text counts at most 0.84.</li>
<li><b>Embedding and storage.</b> Chunks are embedded (768-dimension vectors) and stored in ChromaDB; the <code>Document</code> row is
written to SQLite. The document is now &ldquo;Ready&rdquo;.</li>
</ol>
<h3>Format detection rules</h3>
{table(["Evidence", "Detected format", "Extraction method"], [
    ["Office / text file, or PDF with a usable text layer", "typed", "Direct parsing (exact text)"],
    ["Image or image-only PDF; Paddle mean calibrated confidence &ge; 0.6", "scanned (printed)", "PaddleOCR / PP-StructureV3"],
    ["Paddle confidence low and TrOCR reads the boxes better, or user hint", "handwritten", "Paddle line detection + TrOCR"],
])}
</div>""")

    # ------------------------------------------------------------------ 7 extraction
    s.append(f"""
<div class="chapter"><h2>7. How text is extracted</h2>
<h3>7.1 Typed documents (exact parsing)</h3>
<ul>
<li><b>PDF with text layer:</b> pdfplumber gives every word with its box; words are grouped into lines; tables are detected from ruling lines.</li>
<li><b>Word (.docx):</b> python-docx reads paragraphs and tables in order. <b>PowerPoint (.pptx):</b> python-pptx reads each slide's text frames as a page.
<b>Excel:</b> openpyxl reads each sheet as a table. <b>Text:</b> read as is.</li>
<li>Confidence is 1.0 because no recognition is involved. The exact text is used for QA unless the user edits a line.</li>
</ul>
<h3>7.2 Scanned and photographed printed pages (OCR)</h3>
<ol>
<li><b>Render / load:</b> PyMuPDF renders each PDF page at 150 DPI; images are read with OpenCV.</li>
<li><b>Preprocess (optional):</b> deskew, denoise, contrast (CLAHE), binarise; page orientation is fixed by PaddleOCR's
document-orientation classifier (PP-LCNet).</li>
<li><b>Text detection:</b> PP-OCRv5 detector (DB, a differentiable-binarisation segmentation network) predicts a text-probability map
and turns it into line polygons.</li>
<li><b>Text-line orientation:</b> PP-LCNet classifier flips upside-down lines.</li>
<li><b>Recognition:</b> PP-OCRv5 recogniser (SVTR-style network with CTC decoding) reads each cropped line and returns text and a score.</li>
<li><b>Words:</b> character positions from the recogniser are used to give each word its own box.</li>
</ol>
<h3>7.3 Layout, tables and formulas (PP-StructureV3)</h3>
<ul>
<li><b>Layout detection</b> (PP-DocLayout_plus-L) finds blocks: title, paragraph, table, figure, formula, header, footer &hellip;;
PP-DocBlockLayout orders them for multi-column reading order.</li>
<li><b>Tables:</b> a classifier decides wired/wireless; RT-DETR cell detectors and SLANeXt / SLANet_plus structure models rebuild the table as HTML rows and cells.</li>
<li><b>Formulas:</b> PP-FormulaNet_plus-S converts formulas to LaTeX.</li>
<li>The output gives Markdown for the page, blocks with boxes, and tables; DocBlendAI uses the blocks to set the <b>content type</b> of each chunk.</li>
</ul>
{fig(d.handwriting_flow(), "Figure 5. Handwriting pipeline: Paddle finds the lines, TrOCR reads them.")}
<h3>7.4 Handwritten notes (HTR)</h3>
<p><b>TrOCR</b> (<code>microsoft/trocr-small-handwritten</code>) is a transformer encoder&ndash;decoder: a Vision Transformer encoder splits
the line image into 16&times;16 patches and encodes them; a text decoder generates the line one token at a time, attending to the image.
For each token the decoder gives a probability; the line confidence is their geometric mean after <b>temperature scaling</b>
(dividing the logits by T &gt; 1), which removes TrOCR's over-confidence.</p>
<p>Before TrOCR, OpenCV <b>removes ruled notebook lines</b> (long horizontal / vertical structures found by morphological opening) and
whitens the background, because ruled lines were the main cause of gibberish on real notebook pages.</p>
<h3>7.5 Classic engine (alternative)</h3>
<p>The original pipeline (<code>ingest_pipeline=classic</code>) uses <b>Tesseract</b> for printed text (or <b>docTR</b>
db_resnet50 + crnn_vgg16_bn when Tesseract is not installed), and docTR detection + TrOCR for handwriting. It is still available and
tested, and is used as a baseline in the evaluation.</p>
</div>""")

    # ------------------------------------------------------------------ 8 calibration
    s.append(f"""
<div class="chapter"><h2>8. Confidence capture and calibration</h2>
{fig(d.confidence_journey(), "Figure 6. From a raw engine score to a reliability tier.")}
<h3>Why calibration is needed</h3>
<p>Each engine reports a score, but the scores mean different things. TrOCR often reports 0.99 for a wrong line; Tesseract word
confidences are on a 0&ndash;100 scale; Paddle scores are high even for slightly wrong lines. A model is <b>well calibrated</b> when,
among all lines with confidence 0.8, about 80% are correct. Only then can one formula and one set of thresholds serve every format.</p>
<h3>Methods implemented (<code>app/calibration</code>)</h3>
{table(["Method", "Formula / idea", "Notes"], [
    ["Temperature scaling", "p = &sigma;(logit(s) / T), one parameter T", "Ayllon et al., ICDAR 2024 (synopsis ref. [3]); also applied inside TrOCR on token logits"],
    ["Platt scaling", "p = &sigma;(a &middot; logit(s) + b)", "Two parameters fitted by logistic regression"],
    ["Isotonic regression", "monotone step function (pool-adjacent-violators)", "Add-one prior keeps it away from exact 0 and 1"],
    ["Per-format tables", "piece-wise linear knots, e.g. scanned 0.70&rarr;0.73, 0.94&rarr;0.99", "Used by the classic engine (data/calibration.json)"],
])}
<p>The data are split into fitting and validation parts; the method with the lowest <b>validation ECE</b> is kept.</p>
<h3>Metrics</h3>
<ul>
<li><b>ECE</b> (Expected Calibration Error) = &Sigma;<sub>bins</sub> (n<sub>b</sub>/N) &middot; |accuracy<sub>b</sub> &minus; confidence<sub>b</sub>|</li>
<li><b>MCE</b> (Maximum Calibration Error) = the largest of those bin gaps.</li>
<li><b>Brier score</b> = mean of (p &minus; y)<sup>2</sup>; <b>NLL</b> = mean of &minus;[y log p + (1&minus;y) log(1&minus;p)].</li>
</ul>
<h3>Example result (sample PaddleOCR run, 119 labelled lines, 76% correct)</h3>
{table(["Metric", "Before", "After (isotonic, chosen)"], [
    ["ECE", "0.131", "0.121"], ["MCE", "0.895", "0.375"], ["Brier", "0.103", "0.050"], ["NLL", "0.296", "0.199"],
])}
<p class="small">The classic engine's per-format calibration lowered calibration error from 0.062 to 0.013 (scanned) and from 0.061 to
0.050 (handwritten) on the evaluation set.</p>
</div>""")

    # ------------------------------------------------------------------ 9 refinement
    s.append(f"""
<div class="chapter"><h2>9. How the LLM works (1): guarded refinement of extracted text</h2>
<p>OCR and HTR make small reading errors (&ldquo;proccess&rdquo;, &ldquo;Opera+ing&rdquo;, &ldquo;rn&rdquo; for &ldquo;m&rdquo;).
A large language model knows the language and can fix them, but it may also <i>invent</i> text. DocBlendAI therefore refines
automatically but surrounds the LLM with guards.</p>
{fig(d.refinement_flow(), "Figure 7. Guarded LLM refinement.")}
<h3>The guards</h3>
<ol>
<li><b>Structured output:</b> Gemini receives numbered lines and must reply with JSON <code>{{line_id, refined_text}}</code> for every
line; anything else is rejected. Lines cannot be merged, dropped or invented.</li>
<li><b>Clear instructions:</b> fix misread words, restore words that are clearly implied, keep wording, numbers, spacing and tables,
add no facts.</li>
<li><b>Word diff:</b> each changed line is compared word by word; words are labelled <i>corrected</i> (replaced) or <i>inferred</i>
(inserted), and highlighted in the UI.</li>
<li><b>Change cap:</b> if more than 40% of a line changes, the change is rejected and the line is flagged &ldquo;needs review&rdquo;.</li>
<li><b>Versions:</b> original extraction, LLM proposal and refined text are saved; the user can show the original, revert, reapply or retry.</li>
<li><b>Trust cap:</b> LLM-written lines count at most <b>0.84</b> confidence in QA, so an answer that rests on them is at most
<span class="tier t-m">Moderate</span>; the LLM can never make an answer look <i>more</i> certain.</li>
<li><b>Graceful failure:</b> quota (429), overload (503) or a missing key never block a document; it stays usable unrefined.</li>
</ol>
<div class="box">Live example: a .docx containing &ldquo;proccess&rdquo; was corrected to &ldquo;process&rdquo;. A question about it was answered
<span class="tier t-m">Moderate</span> (evidence is LLM text); after <i>Revert</i> to the exact original text, the same question was
<span class="tier t-c">Certain</span>.</div>
</div>""")

    # ------------------------------------------------------------------ 10 rag
    s.append(f"""
<div class="chapter"><h2>10. How the LLM works (2): retrieval-augmented question answering</h2>
<p><b>Retrieval-Augmented Generation (RAG)</b> means the LLM does not answer from memory: the system first <i>retrieves</i> the
most relevant passages from the user's documents and then asks the LLM to answer <i>only</i> from them. This keeps answers grounded
and lets the system show the sources.</p>
{fig(d.rag_flow(), "Figure 8. Question answering flow (Modules 5, 6 and 7).")}
<h3>Step by step</h3>
<ol>
<li><b>Embedding.</b> gemini-embedding-001 turns the question into a 768-number vector. Chunks were embedded the same way when indexed,
so texts with similar meaning have vectors pointing in similar directions.</li>
<li><b>Vector search.</b> ChromaDB returns the 15 nearest chunks (cosine similarity), restricted to the ticked documents.</li>
<li><b>Follow-ups.</b> In a chat, the previous question is searched together with the new one, so &ldquo;explain it in more detail&rdquo;
still finds the right chunks.</li>
<li><b>Confidence-aware re-ranking.</b> combined = 0.7 &times; similarity + 0.3 &times; calibrated confidence; duplicates removed;
the 5 slots are shared fairly between documents that are almost equally relevant (within 0.05 similarity).</li>
<li><b>Reliability tier</b> from the top passage (Section 11).</li>
<li><b>Prompt.</b> Gemini (gemini-3.5-flash-lite) receives the numbered passages with their content type, a warning on passages read with
confidence below 0.95, the last three chat turns (only to resolve &ldquo;it&rdquo; / &ldquo;that&rdquo;), and the rule: answer only
from the passages, otherwise reply <code>NOT_FOUND</code>.</li>
<li><b>Answer.</b> A NOT_FOUND reply becomes &ldquo;I couldn't find the answer in the document&rdquo; and the label is lowered to at
least Uncertain. If the main model is out of quota or busy, the call is retried on the fallback model (gemini-3.5-flash).</li>
<li><b>Store and show.</b> Query, Answer and RetrievalResults are stored in SQLite; the page shows the answer, tier and sources with page/line.</li>
</ol>
<h3>How the language model itself generates an answer</h3>
<p>Gemini is a transformer language model. The prompt (instructions + passages + question) is split into tokens; self-attention
layers let every token look at every other token, so the model links the question words to the matching passage words. It then
generates the answer one token at a time, each time choosing a likely next token given everything before it. Because the passages are
in the prompt, the most likely continuation is a summary of them; the instructions and the NOT_FOUND rule discourage invented facts.</p>
<h3>Simplified answer prompt</h3>
<pre>You answer questions about the user's documents using ONLY the passages below.
Passages marked "may contain recognition errors" were read by OCR/HTR with lower confidence.
If the passages do not contain the answer, reply exactly NOT_FOUND.

[1] (paragraph, page 2) ... passage text ...
[2] (table, page 3, may contain recognition errors) ... passage text ...

Earlier conversation (only to understand references): ...
Question: What are the four reliability tiers?</pre>
</div>""")

    # ------------------------------------------------------------------ 11 reliability
    s.append(f"""
<div class="chapter"><h2>11. Reliability tiers</h2>
{fig(d.reliability_tree(), "Figure 9. Reliability decision from the top retrieved passage.")}
{table(["Tier", "Rule", "Meaning for the user"], [
    ['<span class="tier t-c">Certain</span>', "similarity &ge; 0.65 and confidence &ge; 0.85", "Strong match on clearly read text: trust it."],
    ['<span class="tier t-m">Moderate</span>', "similarity &ge; 0.58 and confidence &ge; 0.60", "Looser match or partly legible / LLM-corrected text: probably right, glance at the source."],
    ['<span class="tier t-u">Uncertain</span>', "otherwise (or answer not found)", "Weak evidence: check the source before using it."],
    ['<span class="tier t-x">Unreadable</span>', "confidence &lt; 0.35", "The relevant text could not be read reliably."],
])}
<p>The thresholds were tuned on the evaluation set; the similarity thresholds reflect the range of gemini-embedding-001 cosine scores.
Because confidence is calibrated, the same thresholds apply to typed, scanned and handwritten text.</p>
</div>""")

    # ------------------------------------------------------------------ 12 models
    s.append(f"""
<div class="chapter"><h2>12. Models used in the project</h2>
{table(["Model", "Type / architecture", "Used for", "Where"], [
    ["PP-OCRv5 mobile detector", "DB (differentiable binarisation) segmentation CNN", "Finding text lines on printed and handwritten pages", "paddleocr_service.py"],
    ["PP-OCRv5 mobile recogniser", "SVTR-based recogniser with CTC decoding", "Reading printed lines", "paddleocr_service.py"],
    ["PP-LCNet classifiers", "Lightweight CNN", "Page orientation, text-line orientation, table type", "PaddleOCR pipelines"],
    ["PP-DocLayout_plus-L", "RT-DETR object detector", "Layout blocks (title, text, table, figure, formula ...)", "PP-StructureV3"],
    ["PP-DocBlockLayout", "Detector", "Reading order of multi-column pages", "PP-StructureV3"],
    ["SLANeXt_wired, SLANet_plus", "Table structure recognition", "Rows / columns / cells as HTML", "PP-StructureV3"],
    ["RT-DETR-L wired / wireless cell detectors", "Transformer detectors", "Table cell boxes", "PP-StructureV3"],
    ["PP-FormulaNet_plus-S", "Encoder-decoder", "Formulas to LaTeX", "PP-StructureV3"],
    ["PaddleOCR-VL (optional, off)", "Vision-language model", "Alternative document parsing", "paddleocr_service.py"],
    ["TrOCR small handwritten", "ViT encoder + transformer decoder (~62M parameters)", "Reading handwritten lines", "htr_extractor.py"],
    ["docTR db_resnet50 + crnn_vgg16_bn", "DBNet detector + CRNN recogniser", "Classic engine: line detection, OCR fallback", "line_segmentation.py, ocr_extractor.py"],
    ["Tesseract 5", "LSTM OCR engine", "Classic engine printed OCR", "ocr_extractor.py"],
    ["gemini-embedding-001", "Embedding model (768-d output)", "Chunk and question vectors", "embedder.py"],
    ["gemini-3.5-flash-lite", "Large language model", "Answers, refinement, spelling suggestions, plugins", "llm_answer.py, plugins/"],
    ["gemini-3.5-flash (fallback)", "Large language model", "Used when the main model is out of quota or busy", "llm_answer.py"],
])}
<h3>Non-learning algorithms</h3>
<ul>
<li>OpenCV: median denoise, CLAHE contrast, Otsu / adaptive binarisation, deskew (minimum-area rectangle), ruled-line removal (morphology).</li>
<li>Calibration: temperature, Platt, isotonic (PAV) with ECE-based selection.</li>
<li>Word-level diff (sequence matching) and edit-distance change ratio for refinement.</li>
<li>Fuzzy matching against the document vocabulary for &ldquo;Did you mean&rdquo;.</li>
<li>Chunking with overlap and line-based citations; rule-based content-type classifier.</li>
</ul>
</div>""")

    # ------------------------------------------------------------------ 13 storage
    s.append(f"""
<div class="chapter"><h2>13. Data storage</h2>
{fig(d.storage(), "Figure 10. SQLite tables, the ChromaDB collection and the data folder.")}
<ul>
<li><b>Why two stores:</b> ChromaDB is good at nearest-neighbour search on vectors; SQLite is good at relations and filters
(format type, reliability label, versions). Each does what it is best at.</li>
<li><b>Chunk id</b> <code>&lt;doc_id&gt;:&lt;n&gt;</code> links a retrieval result back to its document.</li>
<li><b>Versions:</b> every extraction, refinement, manual edit or restore adds a row to <code>run_versions</code>, so any earlier text can be brought back.</li>
<li><b>Deleting</b> a document removes its file, rows, run and vectors together.</li>
</ul>
</div>""")

    # ------------------------------------------------------------------ 14 UI
    s.append(f"""
<div class="chapter"><h2>14. Experience Center, exports and the user interface</h2>
<h3>QA page</h3>
<ul>
<li>Drag-and-drop upload with optional &ldquo;handwritten&rdquo; hint; the document appears at once with live progress.</li>
<li>Document list with tick-boxes to choose which documents answer questions; delete button.</li>
<li>Viewer: page images with recognised boxes, layout view, or the original file.</li>
<li>Chat: answers with tier badges, expandable sources (document, page, lines, similarity, readability), &ldquo;Did you mean&rdquo;
suggestions, LLM-corrected words marked.</li>
<li>&ldquo;LLM changes&rdquo; panel: each changed line with original and refined text; revert / reapply / retry.</li>
</ul>
<h3>Experience Center (/studio)</h3>
<ul>
<li>Choose pipeline (text OCR or PP-StructureV3), engine options, image clean-up preview.</li>
<li>Page canvas with coloured boxes by confidence; click a line to see raw and calibrated confidence and edit it.</li>
<li>Tabs: Text, Layout, Markdown, Tables, JSON, Confidence (histogram and reliability diagram), Versions.</li>
<li>Low-confidence lines (below 0.70) are flagged for review.</li>
<li>Exports: TXT, layout-preserving TXT, Markdown, JSON, searchable PDF, DOCX, CSV.</li>
<li>Calibration: upload labelled data, fit, see ECE/MCE/Brier/NLL before and after.</li>
<li>Plugins: refine, key-information extraction, translate.</li>
</ul>
</div>""")

    # ------------------------------------------------------------------ 15 evaluation
    s.append(f"""
<div class="chapter"><h2>15. Evaluation and results</h2>
<p>The synopsis metrics are <b>Exact Match, Semantic Match, Character Error Rate (CER)</b> and <b>Word Error Rate (WER)</b>, measured
across the three formats. Scripts are in <code>evaluation/</code>; the report is <code>evaluation/results/report.md</code>.
The current numbers come from a synthetic evaluation set (generated typed, scanned and handwritten pages with known text).</p>
<h3>Metric definitions</h3>
<ul>
<li><b>CER</b> = (substitutions + deletions + insertions) / characters in the reference; <b>WER</b> = the same on words.</li>
<li><b>Exact Match</b>: normalised answer equals the reference. <b>Semantic Match</b>: embedding similarity with the reference above a threshold.</li>
</ul>
<h3>Recognition</h3>
{table(["Format", "CER", "WER", "Format detected correctly"], [
    ["Typed", "0.000", "0.000", "yes"], ["Scanned / printed", "0.366", "0.504", "71%"], ["Handwritten", "0.020", "0.077", "38%"],
])}
<h3>Question answering</h3>
{table(["Question set", "Exact", "Contains", "Semantic", "Correct", "Retrieval hit"], [
    ["Answerable", "19%", "81%", "88%", "88%", "88%"], ["Unanswerable", "&ndash;", "&ndash;", "&ndash;", "100% (NOT_FOUND)", "&ndash;"],
])}
<h3>Reliability tiers vs correctness</h3>
{table(["Tier", "Answers", "Correct"], [["Certain", "14", "100%"], ["Uncertain", "4", "50%"]])}
<p>The tiers behave as intended: <span class="tier t-c">Certain</span> answers were always right, while
<span class="tier t-u">Uncertain</span> answers were right only half the time, so the label tells the user when to check.</p>
<h3>Noise robustness (OHRBench idea, synopsis ref. [2])</h3>
<p><code>evaluation/noise_robustness.py</code> injects graded OCR noise into the text and checks that reliability labels drop as noise rises.</p>
<h3>Live checks on real files</h3>
{table(["File", "Result", "Time (CPU)"], [
    ["Typed PDF", "Exact text, Certain answers", "3.5 s"],
    ["Word file with typo", "Typo corrected by Gemini; Moderate, Certain after revert", "3 s"],
    ["Scanned page", "PaddleOCR text + boxes", "92 s"],
    ["Table image", "Table parsed into correct rows/cells by PP-StructureV3", "63 s"],
    ["Handwritten note", "Read correctly by Paddle boxes + TrOCR", "40 s"],
])}
<div class="box warn"><b>Note:</b> scanned CER on the synthetic set is high because those pages were generated with heavy noise and
blur; real scans read much better. Format detection for handwriting is conservative: unclear cases fall back to OCR, and the user
can set the &ldquo;handwritten&rdquo; hint.</div>
</div>""")

    # ------------------------------------------------------------------ 16 deployment
    s.append(f"""
<div class="chapter"><h2>16. Deployment, testing and engineering practice</h2>
{fig(d.deployment(), "Figure 11. Two ways to run DocBlendAI.")}
<h3>Running</h3>
<pre>venv/Scripts/python -m pip install -r requirements.txt
venv/Scripts/python -m uvicorn app.main:app --reload        # http://localhost:8000

docker compose up -d                                         # same app in a container</pre>
<p>The Gemini key is read from <code>.env</code> (<code>GEMINI_API_KEY</code>); model names and thresholds can be changed there too.</p>
<h3>Testing</h3>
<ul>
<li>About 440 automated tests (pytest) cover every module, router and the full upload &rarr; ask flow.</li>
<li>Tests never call Gemini and never load real OCR/HTR models: fakes replace them (<code>tests/conftest.py</code>), so tests run fast and offline.</li>
<li>A GitHub Actions workflow builds the Docker image.</li>
</ul>
<h3>Team workflow</h3>
<p><code>main &rarr; feature branch &rarr; development &rarr; commit &rarr; push &rarr; pull request &rarr; review &rarr; merge</code>.
Each member owns modules; nobody commits directly to <code>main</code>. The build followed the synopsis order: typed path &rarr; embeddings
&rarr; first end-to-end QA &rarr; OCR/HTR + confidence &rarr; content type + combined score + reliability &rarr; evaluation and UI.</p>
</div>""")

    # ------------------------------------------------------------------ 17 limits
    s.append("""
<div class="chapter"><h2>17. Limitations and future work</h2>
<h3>Limitations</h3>
<ul>
<li>OCR on CPU is slow for long scanned documents (about 1&ndash;1.5 minutes per page with PP-StructureV3).</li>
<li>Answers need the Gemini API (internet and quota); free-tier quotas are small.</li>
<li>Handwriting recognition works line by line and is weaker on very messy or mixed handwriting/print pages.</li>
<li>Calibration is only as good as the labelled data used to fit it; more real labelled pages would improve it.</li>
<li>Latin script only; no multilingual support.</li>
</ul>
<h3>Future work</h3>
<ul>
<li>GPU support and page-parallel processing.</li>
<li>A local open-source LLM option for offline use.</li>
<li>Fine-tuning TrOCR on Indian student handwriting.</li>
<li>A larger real evaluation set (scanned notes and handwritten answer sheets from the department).</li>
<li>User accounts and per-user document libraries.</li>
</ul>
<h2>18. Glossary</h2>
<table><tbody>
<tr><td><b>OCR</b></td><td>Optical Character Recognition: reading printed text from images.</td></tr>
<tr><td><b>HTR</b></td><td>Handwritten Text Recognition.</td></tr>
<tr><td><b>RAG</b></td><td>Retrieval-Augmented Generation: retrieve relevant passages, then let an LLM answer from them.</td></tr>
<tr><td><b>Embedding</b></td><td>A vector of numbers representing the meaning of a text; similar meanings give similar vectors.</td></tr>
<tr><td><b>Cosine similarity</b></td><td>How closely two vectors point the same way (1 = identical direction).</td></tr>
<tr><td><b>Calibration</b></td><td>Adjusting confidence scores so they match the real probability of being correct.</td></tr>
<tr><td><b>ECE</b></td><td>Expected Calibration Error: the average gap between confidence and accuracy.</td></tr>
<tr><td><b>CER / WER</b></td><td>Character / Word Error Rate: edit distance divided by reference length.</td></tr>
<tr><td><b>Chunk</b></td><td>A piece of about 1000 characters of a document, the unit that is retrieved.</td></tr>
<tr><td><b>Transformer</b></td><td>Neural network built on self-attention; used by TrOCR and Gemini.</td></tr>
<tr><td><b>CTC</b></td><td>Connectionist Temporal Classification: decoding used by line recognisers.</td></tr>
</tbody></table>
</div>""")
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><title>DocBlendAI Project Explanation</title>"
        f"<style>{CSS}</style></head><body>{''.join(s)}</body></html>"
    )


def browser() -> str | None:
    for p in (
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ):
        if Path(p).exists():
            return p
    return shutil.which("msedge") or shutil.which("chrome") or shutil.which("chromium")


def main() -> None:
    HTML.write_text(build(), encoding="utf-8")
    print("wrote", HTML)
    exe = browser()
    if not exe:
        print("No Edge/Chrome found: open the HTML and print it to PDF.")
        return
    subprocess.run(
        [exe, "--headless", "--disable-gpu", f"--print-to-pdf={PDF}", "--no-pdf-header-footer",
         HTML.resolve().as_uri()],
        check=True, timeout=180, capture_output=True,
    )
    print("wrote", PDF)


if __name__ == "__main__":
    main()
