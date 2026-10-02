# DocBlendAI

Confidence-aware multi-format document QA assistant: upload typed, scanned, or
handwritten academic documents and ask questions. Every answer carries a reliability
tier (Certain / Moderate / Uncertain / Unreadable).

**Supported uploads:** PDF, images (JPG, PNG, TIFF incl. multi-page, BMP, WEBP;
phone photos are turned upright automatically), Word (.docx, including tables),
PowerPoint (.pptx, one page per slide), Excel (.xlsx), and plain text (.txt). See `CLAUDE.md`
for full project context.

**Team members:** start with [CONTRIBUTING.md](CONTRIBUTING.md) for setup, the Git workflow, and troubleshooting.

## What's new

Full list with files, endpoints, settings and tests: **[CHANGELOG.md](CHANGELOG.md)**.

- **Experience Center** (`/studio`): PaddleOCR text OCR and PP-StructureV3 document parsing with
  the page and the result side by side, confidence-coloured line boxes, Layout / Markdown / JSON /
  Tables tabs, exports (TXT, layout text, Markdown, JSON, searchable PDF, Word, CSV).
- **One document for both pages**: an upload on either page is read once (exact text, PaddleOCR /
  PP-StructureV3, TrOCR for handwriting) and is both a QA document and an Experience Center result.
- **Automatic LLM refinement**: Gemini fixes misread words and restores implied words in every
  upload; changes are marked, the original is kept (Show original / Revert), and LLM text never
  raises an answer above Moderate.
- **Confidence calibration** for PaddleOCR (temperature, Platt, isotonic; ECE / Brier / NLL).
- **QA page**: choose which documents to ask about, "Did you mean", document viewer with layout.
- **Docker**: run everything on another computer with one command (below).

## Two ways to run it

| | [Docker](#run-with-docker-recommended-for-other-computers) | [Local venv](#run-locally-with-a-python-venv) |
|---|---|---|
| Best for | teammates, demos, any OS | developing the code on Windows |
| Install | Docker Desktop | Python 3.11, Tesseract, ~3 GB of packages |
| Start | `scripts/docker-start.ps1` or `docker compose up -d` | `venv/Scripts/python -m uvicorn app.main:app --reload` |

Both use the same `data/` folder and `.env`; run only one of them at a time (they share the
SQLite database).

## Run with Docker (recommended for other computers)

1. **Install Docker Desktop** (https://www.docker.com/products/docker-desktop/; Windows needs WSL 2,
   which the installer sets up). Start it and wait for "Engine running". In Docker Desktop >
   Settings > Resources, give it **at least 8 GB of memory** (PP-StructureV3 + TrOCR + docTR use
   4–6 GB on CPU).
2. **Get the code**: `git clone` the repository (see CONTRIBUTING.md) and open a terminal in it.
3. **Start**, either with the start script (it creates `.env` from `.env.example`, asks for your
   Gemini key without showing it, starts the container, waits until it is ready and opens the app):
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\docker-start.ps1 -Build
   ```
   ```bash
   bash scripts/docker-start.sh --build
   ```
   or by hand:
   ```bash
   cp .env.example .env          # then put your key after GEMINI_API_KEY=
   docker compose up -d --build
   ```
   Then open http://localhost:8000 (QA page) and http://localhost:8000/studio (Experience Center).
   Get a Gemini key at https://aistudio.google.com/apikey; without one, uploads and OCR work but
   answers and LLM refinement are off (the pages say so).

**First run, what to expect**

- **Build**: 10–20 minutes the first time (downloads ~2 GB of Python packages, CPU-only PyTorch);
  later builds after a code change take seconds, after a `requirements.txt` change a few minutes
  (BuildKit pip cache).
- **Image size**: about 4 GB (estimate: the image was not built on the authoring machine). With
  models baked in (`PRELOAD_MODELS=true`) about 2 GB more.
- **Models**: by default PaddleOCR / PP-StructureV3 (~1.2 GB), TrOCR (~0.5 GB) and docTR
  (~160 MB) download on first use into the `model-cache` volume, once (~2 GB in total). To ship them in the image
  instead (no download on a teammate's first OCR run):
  ```bash
  PRELOAD_MODELS=true docker compose build          # Linux/macOS
  ```
  ```powershell
  $env:PRELOAD_MODELS="true"; docker compose build  # Windows PowerShell
  ```
  A new `model-cache` volume is filled from the image automatically.
- **Speed on CPU** (laptop, 4 cores): typed PDF / Word / text a few seconds; text OCR ~10 s per
  page; PP-StructureV3 (default for QA-page uploads) ~60–90 s per scanned page, plus ~35 s to
  load it the first time; handwriting (TrOCR) ~30–40 s per page; Gemini refinement a few seconds
  per document. Set `INGEST_PIPELINE=ocr` in `.env` for faster, text-only reading.
- **RAM**: 8 GB for Docker recommended (4 GB works for typed files and text OCR only).

**Daily use**

```bash
docker compose ps                 # status (healthy = ready)
docker compose logs -f            # follow the logs
docker compose down               # stop (data and models are kept)
docker compose up -d              # start again
```

**Updating**: `git pull`, then `docker compose up -d --build` (only changed layers are rebuilt).
To use a pre-built image instead of building (once the GitHub workflow below publishes it):
```bash
docker compose pull
docker compose up -d --no-build
```
(`-Pull` / `--pull` in the start scripts; a different image: `DOCBLENDAI_IMAGE=ghcr.io/<owner>/docblendai:latest`.)
If the image is private, log in first: `docker login ghcr.io` (GitHub username + a token with `read:packages`).

**Where data is stored**: `./data` on your computer (bind-mounted into the container): SQLite
database `data/docblendai.db`, ChromaDB `data/chroma_db/`, uploads `data/uploads/`,
Experience Center page images `data/paddle_runs/`, PaddleOCR calibration `data/paddle_calibration/`.
`docker compose down` keeps it; models are in the Docker volume `model-cache`
(`docker volume rm docblendai_model-cache` frees ~2 GB; they download again when needed).

**Calibration in Docker**: the fitted PaddleOCR calibrator is not in git; fit it once with
`docker compose exec docblendai python -m app.calibration.calibrate` (2–3 minutes).

**MKL-DNN (oneDNN) on Linux**: it is off by default (`PADDLE_ENABLE_MKLDNN=false`), the setting
known to work everywhere. paddle 3.3's oneDNN kernels crash on Windows; whether they work in the
Linux container was **not tested yet**. To try it (faster on CPU if it works):
```bash
docker compose exec -e PADDLE_ENABLE_MKLDNN=true docblendai python scripts/paddle_smoke_test.py
```
If that ends with `RESULT: PASS`, set `PADDLE_ENABLE_MKLDNN=true` in `.env` and
`docker compose up -d`; if it fails with `ConvertPirAttribute2RuntimeAttribute ... Unimplemented`,
keep it `false`.

**Troubleshooting**

| Problem | Fix |
|---|---|
| `port is already allocated` / port 8000 busy | Stop the other server (e.g. the local venv one), or use another port: `DOCBLENDAI_PORT=8001 docker compose up -d` (start scripts: `-Port 8001` / `--port 8001`), then open http://localhost:8001 |
| Container restarts, `Killed` / exit code 137 in `docker compose logs` | Out of memory: give Docker Desktop more memory (8 GB), or set `INGEST_PIPELINE=ocr` in `.env` |
| "Gemini is not configured" / "Not refined: Gemini unavailable" | Put your key in `.env` (`GEMINI_API_KEY=...`), then `docker compose up -d` (recreates the container with it) |
| "Gemini's quota ... is used up" | Free-tier limit: wait (resets daily) or set `LLM_MODEL` / `LLM_FALLBACK_MODEL` in `.env` |
| `docker: command not found` / "Docker is not running" | Install / start Docker Desktop and wait for "Engine running" |
| `Permission denied` writing `data/` (Linux) | The container runs as uid 1000: `sudo chown -R 1000:1000 data` |
| `env_file ... required` not supported | Update Docker Desktop (Compose 2.24 or newer) |
| First OCR run is slow | Models are downloading (see first run above), or build with `PRELOAD_MODELS=true` |

**Publishing the image (optional)**: `.github/workflows/docker.yml` builds the image on pull
requests and pushes `ghcr.io/<owner>/docblendai` on pushes to `main` and on `v*` tags. To enable:
repository Settings > Actions > General > Workflow permissions > "Read and write permissions";
after the first run, share the package (Packages > docblendai > Package settings) with the team.

## Run locally with a Python venv

1. Python 3.11 virtualenv in `venv/`, then:
   ```bash
   venv/Scripts/python -m pip install -r requirements.txt
   ```
2. Tesseract OCR (for the classic OCR path and page-orientation correction):
   ```bash
   winget install UB-Mannheim.TesseractOCR
   ```
   The default install location is found automatically; otherwise set `TESSERACT_CMD` in `.env`.
3. Copy `.env.example` to `.env` and set `GEMINI_API_KEY`. Never put the key in `.env.example`.

The TrOCR handwriting model (~0.5 GB) and docTR (~160 MB) download automatically on first use.
PaddleOCR models (~1.2 GB for PP-StructureV3, ~20 MB for text OCR) download to `~/.paddlex` on first use
(or all at once: `venv/Scripts/python scripts/preload_models.py`).

Check the PaddleOCR install (imports, models on CPU, one OCR run, one PP-StructureV3 parse):
```bash
venv/Scripts/python scripts/paddle_smoke_test.py
```
On Windows, paddle 3.3 must run with MKL-DNN off (`PADDLE_ENABLE_MKLDNN=false`, the default) and
`torch` must be imported before `paddle` in a process (the service does this); otherwise paddle
crashes in oneDNN, or torch fails with `WinError 127 ... shm.dll`.

Start it:
```bash
venv/Scripts/python -m uvicorn app.main:app --reload
```

- **App:** http://127.0.0.1:8000/ to upload documents, ask questions, and see reliability labels and sources
- **Experience Center:** http://127.0.0.1:8000/studio for PaddleOCR text OCR / document parsing with
  calibrated confidence, side by side with the original (see below); same documents as the QA page
- **API docs:** http://127.0.0.1:8000/docs

If the server refuses to start with "database is out of date", delete `data/docblendai.db`
and the contents of `data/chroma_db/` (local dev data), restart, and re-upload.
With `--reload`, saving a `.py` file restarts the server and interrupts a document being read
(it shows "Interrupted"; upload it again).

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/upload` | Upload a PDF, image, .docx, .pptx, .xlsx or .txt (optional `format_hint`: typed / scanned / handwritten; `pipeline`: structure / ocr; `background=true` returns 202 at once) |
| GET | `/documents/library` | Both pages' documents with reading progress, LLM refinement and QA state |
| GET | `/documents` | List uploaded documents |
| GET | `/documents/{id}/file` | The original uploaded file, inline (document viewer) |
| POST | `/documents/{id}/experience` | Read an earlier upload into the Experience Center |
| DELETE | `/documents/{id}` | Remove a document from both pages (chunks, record, file, Experience Center result) |
| POST | `/ask` | Ask a question → answer + reliability label (optional `doc_ids`: only use these documents) |
| POST | `/ask/suggest` | "Did you mean": spelling-corrected question from the selected documents' words (`question_text`, optional `doc_ids`) |
| GET | `/answer/{id}` | Fetch a stored answer |
| GET | `/answer/{id}/sources` | Chunks behind an answer, with similarity, confidence, content type |

## Experience Center (`/studio`)

A PaddleOCR-powered workspace in DocBlendAI's own design: drop images (JPG/PNG/BMP/TIFF/WEBP),
multi-page PDFs, Office or text files (docx/xlsx/pptx/txt; several at once, processed as a queue),
or pick a built-in sample; choose **Text OCR** (PP-OCRv5 mobile), **Document parsing**
(PP-StructureV3: titles, paragraphs, tables, formulas) or **PaddleOCR-VL** (shown
"unavailable" unless `PADDLE_ENABLE_VL=true`), a language and the document type. The page shows:

- left: the page with every line's box coloured by confidence (green / yellow / red), zoom,
  pan, page navigation; hovering a line highlights its box and vice versa
- right: **Text** (click a line to correct it: human review), **Layout** (the page rebuilt at its
  geometry), **Changes** (what the LLM changed, versions), **Markdown**, **JSON**,
  **Tables** (CSV per table), **Plugins** (key-information extraction, translation; enabled
  only when `GEMINI_API_KEY` is set) and **Calibration** (method, ECE/MCE/Brier/NLL before
  and after, reliability diagram, refit)
- a raw vs calibrated confidence toggle, a "hide below" filter and the review threshold slider
  (lines under it are flagged "needs review")
- optional clean-up before OCR (deskew, denoise, contrast, binarize) with a before/after preview
- exports: TXT, layout text, Markdown, JSON, searchable PDF (page image + invisible text layer), Word, CSV
- the shared Documents list (search by name or recognised text; "Ask questions" opens the QA page)

### One document for both pages, refined by an LLM automatically

An upload from either page (any type) is one background job, shown with its progress on both
pages: "Extracting page 2 of 5…" (typed PDF / Word / PowerPoint / Excel / text: exact text;
scans and photos: PaddleOCR or PP-StructureV3; handwriting: TrOCR reading each of Paddle's line
boxes) → "Refining with Gemini…" → "Ready". The QA page and the Experience Center list the same
documents; deleting one deletes it everywhere.

- **Refinement** (`app/plugins/refine.py`, `app/modules/refinement.py`): Gemini gets every line
  with its calibrated confidence and the page structure, fixes misread words, restores words the
  context clearly implies, and keeps layout, spacing, tables and meaning (no new facts). Changes
  are applied without an approval step; a line that would change more than 40% of its characters
  (`REFINE_MAX_CHANGE`) keeps its recognised text and is flagged.
- **The refined text is used everywhere**: QA answers (re-chunked and re-embedded), the Text and
  Layout views, exports. Corrected words are highlighted and inferred words marked differently
  on both pages (hover shows the original). Each document shows "Refined by LLM" with
  **Show original** and **Revert to original** (re-embeds the original; "Use refined text" goes
  back). The original extraction, Gemini's proposal and every edit are kept as versions.
- **LLM text never raises reliability**: lines the LLM changed count at most as Moderate
  confidence for retrieval and reliability labels.
- **Gemini unavailable** (no `GEMINI_API_KEY`, quota used up, error): the unrefined text is used,
  the document says "Not refined: Gemini unavailable" with a **Retry** link, and QA is not blocked.
- **Layout view** (Experience Center tab and the QA viewer): the page rebuilt at its real geometry
  (words at their boxes, tables as real tables with merged cells, figures and formulas in place,
  zoomable, hover highlights the page image too); Office and text files as structured Markdown.
  "Layout text" export: monospace text with the page's spacing and columns.

### Confidence calibration

PaddleOCR's raw line scores are over-confident on degraded scans. `app/calibration/` fits
temperature scaling, Platt scaling and isotonic regression on lines labelled correct/incorrect
against ground truth (exact match or CER <= `CALIBRATION_CER_THRESHOLD`), keeps the one with the
lowest validation ECE, and saves it with a JSON report and reliability diagram to
`data/paddle_calibration/`. Every OCR line then carries `raw_confidence` and
`calibrated_confidence` (raw values marked "uncalibrated" until a calibrator exists).

```bash
# Regenerate the bundled 40-image sample (evaluation/calibration_sample/, labels.csv + PNGs)
venv/Scripts/python -m app.calibration.make_sample
# Fit on it with the real OCR engine (~2-3 min on CPU), or on your own folder/CSV
venv/Scripts/python -m app.calibration.calibrate
venv/Scripts/python -m app.calibration.calibrate path/to/labels.csv --match exact
```
Your own data: a folder with `labels.csv` (columns `image,text`; `text` holds the image's
ground-truth lines separated by newlines) next to the images; see
[evaluation/calibration_sample/README.md](evaluation/calibration_sample/README.md). The same
folder zipped can be uploaded from the Calibration tab or `POST /api/calibrate`.

On the bundled sample (119 lines, 76% read correctly) isotonic regression is chosen; on the
validation split ECE 0.131 -> 0.121, MCE 0.895 -> 0.375, Brier 0.103 -> 0.050, NLL 0.296 -> 0.199.

## Experience Center API

PaddleOCR text OCR, document parsing (PP-StructureV3) and Office-to-Markdown conversion with
calibrated per-line confidence, used by the `/studio` page. Full shapes are in
[docs/experience_center_contract.md](docs/experience_center_contract.md); code in
`app/routers/paddleocr.py`, `app/modules/paddleocr_service.py` and `app/modules/ocr_jobs.py`.
Jobs run one at a time on a background thread; results (history) are stored in the `ocr_runs`
SQLite table and page images under `data/paddle_runs/`. Models load on first use (text OCR:
a few seconds; PP-StructureV3: ~35 s, then about a minute per page on a CPU laptop).

```bash
# Engine, pipelines (ocr / structure / vl / office), calibration state, languages, limits
curl http://127.0.0.1:8000/api/health

# Text OCR of an image or PDF, waiting for the full result
curl -F file=@scan.pdf -F language=en -F wait=true http://127.0.0.1:8000/api/ocr

# Same, as a background job: returns 202 {"id": ..., "status": "queued"}; poll for progress
curl -F file=@photo.jpg -F review_threshold=0.8 http://127.0.0.1:8000/api/ocr
curl http://127.0.0.1:8000/api/results/<id>

# Layout, tables and Markdown (PP-StructureV3); .docx/.xlsx/.pptx are converted without OCR
curl -F file=@page.png -F pipeline=structure -F wait=true http://127.0.0.1:8000/api/parse
curl -F file=@notes.docx -F wait=true http://127.0.0.1:8000/api/parse

# History (newest first, search file names and recognised text), page image, delete
curl "http://127.0.0.1:8000/api/results?q=calibration&limit=20"
curl -o page0.png http://127.0.0.1:8000/api/results/<id>/pages/0/image
curl -X DELETE http://127.0.0.1:8000/api/results/<id>

# Human review: fix a line's text (the line is marked "edited": true)
curl -X PATCH -H "Content-Type: application/json" \
     -d '[{"page": 0, "line": 3, "text": "corrected text"}]' \
     http://127.0.0.1:8000/api/results/<id>/lines

# Fit confidence calibration on a zip (labels.csv with columns image,text + the images),
# or on the bundled sample; then read the report and reliability diagram
curl -F file=@dataset.zip -F match=cer http://127.0.0.1:8000/api/calibrate
curl -F use_sample=true http://127.0.0.1:8000/api/calibrate
curl http://127.0.0.1:8000/api/calibration
curl -o diagram.png http://127.0.0.1:8000/api/calibration/diagram.png

# Refinement: what Gemini changed, revert to the original / use the refined text again, retry
curl http://127.0.0.1:8000/api/results/<id>/changes
curl -X POST http://127.0.0.1:8000/api/results/<id>/revert
curl -X POST http://127.0.0.1:8000/api/results/<id>/reapply
curl -X POST http://127.0.0.1:8000/api/results/<id>/refine
curl http://127.0.0.1:8000/api/results/<id>/versions

# Exports (current text): txt | layout (spacing kept) | md | json | pdf (searchable) | docx | csv (tables)
curl -OJ http://127.0.0.1:8000/api/results/<id>/export/pdf

# Clean-up before OCR, and its before/after preview (data: URLs)
curl -F file=@photo.jpg -F 'preprocess={"deskew": true, "contrast": true}' -F wait=true http://127.0.0.1:8000/api/ocr
curl -F file=@photo.jpg -F 'preprocess={"deskew": true, "binarize": true}' http://127.0.0.1:8000/api/preprocess

# Plugins (need GEMINI_API_KEY): key-information extraction and translation of a result
curl http://127.0.0.1:8000/api/plugins
curl -H "Content-Type: application/json" -d '{"fields": ["title", "date"]}' http://127.0.0.1:8000/api/results/<id>/plugins/kie
curl -H "Content-Type: application/json" -d '{"target_language": "Hindi"}' http://127.0.0.1:8000/api/results/<id>/plugins/translate
```

Errors are `{"detail": "..."}`: 400 bad form value, 404 unknown id, 413 file larger than
`PADDLE_MAX_UPLOAD_MB`, 415 unsupported or disguised file type, 503 pipeline unavailable
(PaddleOCR-VL stays off unless `PADDLE_ENABLE_VL=true`). Tests never load a Paddle model
(a guard in `tests/conftest.py` fails any test that tries).

## Logging

`LOG_FORMAT=json` gives one JSON log object per line (job ids, pipeline, timings as fields);
in Docker see them with `docker compose logs -f`.

## Testing and evaluation

```bash
venv/Scripts/python -m pytest
venv/Scripts/python -m evaluation.make_dataset
venv/Scripts/python -m evaluation.run_eval --fit-calibration
```

Tests fake Tesseract, TrOCR, docTR, PaddleOCR, and Gemini, so they run offline (403 tests). The evaluation uses the
real engines: it generates a synthetic dataset (typed, degraded scans, handwriting-style
fonts), measures CER/WER, fits confidence calibration into `data/calibration.json`,
and scores question answering (Exact Match, Semantic Match, retrieval hits, accuracy per
reliability tier). The latest report is in [evaluation/results/report.md](evaluation/results/report.md).

## Folder structure

```
app/
├── main.py                  FastAPI app, router registration, /health, serves the UI
├── config.py                Settings from .env (pydantic-settings)
├── logging_config.py        Text or JSON (LOG_FORMAT) logging
├── static/index.html        Demo UI (document QA)
├── static/studio.*          Experience Center UI (/studio); static/samples/ its sample images
├── static/layout.js, .css   Shared by both pages: layout view, LLM change marks, Markdown
├── calibration/             PaddleOCR confidence calibration: calibrators, metrics, selection,
│                            labelling, reliability diagram, CLI (calibrate.py), sample generator
├── plugins/                 Gemini plugins: refine (automatic LLM refinement), KIE, translate
├── routers/
│   ├── upload.py            Module 1: POST /upload, GET/DELETE /documents, GET /documents/{id}/file
│   ├── query.py             POST /ask, POST /ask/suggest, GET /answer/{id}[/sources] (Modules 5-7)
│   ├── paddleocr.py         Experience Center: /api/ocr, /api/parse, results, history, calibration
│   ├── studio_tools.py      Experience Center: exports, preprocessing preview, plugins
│   └── refine.py            LLM refinement: changes, revert, reapply, retry, versions
├── modules/
│   ├── format_detection.py  Module 2: detect format, route to an extractor
│   ├── document_extract.py  Module 2: one extractor for both pages (exact text / PaddleOCR / PP-StructureV3 / TrOCR)
│   ├── text_parser.py       Module 2: direct parsing of typed PDF, Word, PowerPoint, Excel, text
│   ├── file_types.py        Modules 1-2: supported upload formats
│   ├── pdf_render.py        Module 2: PDF pages / image files -> page images, denoise
│   ├── ocr_extractor.py     Module 2: pytesseract OCR for scanned PDFs
│   ├── htr_extractor.py     Module 2: TrOCR HTR for handwritten PDFs
│   ├── confidence_capture.py Module 3: raw -> calibrated confidence
│   ├── content_type.py      Module 4: table / paragraph / image labeling
│   ├── chunker.py           Modules 4/5: split text into chunks
│   ├── embedder.py          Module 5: Gemini embeddings
│   ├── vector_store.py      Module 5: ChromaDB wrapper
│   ├── retrieval.py         Module 5: combined_score ranking
│   ├── spelling.py          Module 5: "Did you mean" question spelling correction
│   ├── reliability.py       Module 6: four-tier reliability labels
│   ├── llm_answer.py        Module 7: Gemini answer generation
│   ├── paddleocr_service.py PaddleOCR / PP-StructureV3 / VL / Office->Markdown, result normalising
│   ├── ocr_jobs.py          Background jobs: extract -> refine -> index, progress, run history
│   ├── document_runs.py     Module 1: QA document <-> Experience Center run, library, delete both
│   ├── qa_index.py          Modules 3-5: QA chunks from a run's lines (content types, LLM cap)
│   ├── refinement.py        Automatic LLM refinement and text versions
│   ├── text_diff.py         Word diff: unchanged / corrected / inferred, change ratio
│   ├── preprocess.py        Deskew / denoise / contrast / binarize (OpenCV)
│   └── exporters.py         TXT / layout text / MD / JSON / searchable PDF / DOCX / CSV exports
├── db/
│   ├── database.py          SQLAlchemy engine/session (SQLite), schema check
│   └── models.py            ORM: Document, Query, Answer, RetrievalResult (+ OCRRun, RunVersion, DocumentRun)
└── models/
    └── schemas.py           Pydantic entities shared by all modules
data/
├── uploads/                 Uploaded files (git-ignored)
├── chroma_db/               ChromaDB persistence (git-ignored)
├── calibration.json         Fitted confidence calibration (committed, shared)
├── paddle_runs/             Experience Center page images per run (git-ignored)
└── paddle_calibration/      Fitted PaddleOCR calibrator, report.json, diagram (git-ignored)
docs/experience_center_contract.md  Experience Center API / result / calibration contract
scripts/                     PaddleOCR smoke test, model preloader, Docker start scripts, sample generator
Dockerfile, docker-compose.yml  Docker image and service (see "Run with Docker")
.github/workflows/docker.yml    Builds / publishes the image to ghcr.io
CHANGELOG.md                 What changed since the first version
evaluation/
├── make_dataset.py          Synthetic typed / scanned / handwritten dataset
├── calibration_sample/      40 labelled images for PaddleOCR calibration
├── run_eval.py              Recognition, calibration, and QA evaluation
└── results/report.md        Latest evaluation report
tests/
├── test_*.py                Unit and API tests (offline)
└── eval_metrics.py          Exact Match, Semantic Match, CER, WER
```
