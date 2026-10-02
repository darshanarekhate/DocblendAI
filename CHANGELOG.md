# Changelog

What changed in DocBlendAI, newest first. The baseline is the original `main` commit `b0030d4`
("DocBlendAI: confidence-aware QA over typed, scanned and handwritten documents"); everything
below is on the branch `feature/experience-center` (`git log b0030d4..HEAD`).

Test counts are from `venv/Scripts/python -m pytest`: 193 tests at `b0030d4`, **403** now
(210 new, all offline: PaddleOCR, TrOCR, docTR, Tesseract and Gemini are faked).

---

## [Unreleased] — `feature/experience-center` (2026-10-01 to 2026-10-02)

### Docker for other computers

- Multi-stage `Dockerfile` on `python:3.11-slim`: a builder stage installs the wheels into
  `/opt/venv` with a BuildKit pip cache; CPU-only `torch==2.14.0` / `torchvision==0.29.0` from the
  PyTorch CPU index; the runtime stage has only system libraries (OpenCV, OpenMP, Tesseract, curl),
  the venv and the app, and runs as the non-root user `app` (uid 1000). Layers are ordered so a
  code change rebuilds in seconds.
- Build arg `PRELOAD_MODELS=true` bakes the app's configured models into the image
  (`scripts/preload_models.py`: PaddleOCR PP-OCRv5 mobile, PP-StructureV3 with
  PP-FormulaNet_plus-S, TrOCR-small, docTR); default `false` (smaller image, models download on
  first use into the `model-cache` volume).
- `docker-compose.yml`: service `docblendai` on port 8000 (`DOCBLENDAI_PORT`), image name
  `ghcr.io/darshanarekhate/docblendai` (`DOCBLENDAI_IMAGE`) so a pre-built image can be pulled,
  optional `.env`, `./data` bind mount + named `model-cache` volume, health check on `/health`,
  `restart: unless-stopped`, memory note (8 GB recommended), `PADDLE_ENABLE_MKLDNN` from `.env`
  (default `false`).
- `scripts/docker-start.ps1` (Windows) and `scripts/docker-start.sh` (Linux/macOS): check Docker,
  create `.env` from `.env.example` asking for the Gemini key (hidden, never printed),
  `docker compose up -d` (`-Build` / `--build`, `-Pull` / `--pull`), wait for `/health`, open the app.
- `.github/workflows/docker.yml`: builds on pull requests; on `main` and `v*` tags pushes to
  `ghcr.io/<owner>/docblendai` (`latest`, branch, semver, sha); manual run can publish `-models`
  images. Not enabled yet (needs the repository's workflow permissions).
- `scripts/paddle_smoke_test.py` honours `PADDLE_ENABLE_MKLDNN` (to try oneDNN inside the container).
- `.gitattributes`: `*.sh` keep LF line endings (so the start script runs on Linux/macOS).
- Files: `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `scripts/preload_models.py`,
  `scripts/docker-start.ps1`, `scripts/docker-start.sh`, `.github/workflows/docker.yml`.

### Shared documents between the QA page and the Experience Center

- One upload from either page (`POST /upload`, `POST /api/ocr`, `POST /api/parse`) is one
  background job that creates both the QA document (chunks, embeddings, reliability) and the
  Experience Center run, linked by `doc_id` (new link table `document_runs`; `schemas.py` and the
  existing tables are unchanged). Both pages list the same documents; deleting one deletes both.
- The QA index is built from the same extraction (`app/modules/qa_index.py`): for scanned and
  image pages the PaddleOCR / PP-StructureV3 text is chunked (Module 2), its line confidences give
  raw / calibrated confidence (Module 3), and PP-StructureV3 block labels give the content type
  (table / paragraph / image, Module 4). Handwritten pages use TrOCR text in Paddle's line boxes;
  typed PDFs, docx, pptx, xlsx and txt keep exact text extraction.
- `POST /upload` still waits and answers `201 Document` (or `200` for a duplicate) by default; the
  UI sends `background=true` (`202 {doc_id, run_id, status}`) and shows progress. New form field
  `pipeline` (`structure` | `ocr`).
- `.xlsx` is now a QA file type (`text_parser.parse_xlsx`).
- QA page: shared document list with live progress, "View in Experience Center"; `/studio`:
  "Documents" list (search by name or text), "Ask questions" link (`/?doc=<id>` preselects it),
  links `/studio?run=<id>&tab=layout|changes`. Older QA uploads can be read into the Experience
  Center ("Open in Experience Center").
- Endpoints: `GET /documents/library?q=`, `POST /documents/{doc_id}/experience`;
  `DELETE /documents/{id}` and `DELETE /api/results/{id}` delete both sides.
- Settings: `EXTRACTION_ENGINE` (`auto` | `paddle` | `classic`), `INGEST_PIPELINE` (`structure`).
- Files: `app/modules/document_extract.py`, `document_runs.py`, `qa_index.py`, `ocr_jobs.py`,
  `app/routers/upload.py`, `app/db/models.py` (`DocumentRunORM`), `app/static/index.html`, `studio.*`.
- Tests: `tests/test_refine.py` (QA upload refined + capped, delete from either page), the
  existing QA tests run unchanged through the new job (`tests/conftest.py` points the job manager
  at the test database and pins the classic engine for them).

### Automatic LLM refinement

- Every upload's text is refined by Gemini automatically and the result is used for QA answers,
  the Text / Layout views and exports: "Extracting page 2 of 5…" → "Refining with Gemini…" →
  "Indexing for questions…" → "Ready". No approval step.
- `refine` plugin (`app/plugins/refine.py`): sends every line with its calibrated confidence and
  the page structure (Markdown / tables), asks for JSON `{line_id, refined_text}` per line, splits
  long documents into parts (≤ 6,000 characters / 150 lines). Fixes misread words, restores words
  the context clearly implies, keeps layout, spacing, tables and meaning, adds no facts.
- Safety: word-level diff per line (`app/modules/text_diff.py`: unchanged / corrected / inferred;
  a split or joined word counts as corrected); a line whose refinement changes more than 40% of
  its characters keeps its recognised text and is flagged (`llm_flag`, needs review).
- Versions (new table `run_versions`): original extraction, Gemini's proposal, refined text,
  edits, reverts and restores are all stored in full; nothing is overwritten.
- Revert to original / use refined text again / retry; each re-chunks and re-embeds the QA document.
- Reliability: text the LLM changed never raises reliability: in the QA index it counts at most
  `reliability.LLM_TEXT_MAX_CONF` (0.84, just below Certain), so such answers are at most Moderate.
- Gemini unavailable (no key, quota 429, overload, unusable reply): the unrefined text is used,
  the document says "Not refined: Gemini unavailable" with a Retry link, QA is never blocked.
  Clear messages for missing key / quota / overload also for answers (`/ask` now returns
  503 / 429 / 502 with the reason) and embeddings (`llm_answer.describe_gemini_error`).
- UI: corrected words highlighted, inferred words marked differently (hover shows the original) on
  both pages (Text tab, Layout, QA viewer, answer sources); "Refined by LLM" badge, Show original,
  Revert to original / Use refined text, Retry; read-only Changes tab with versions (compare,
  restore) in `/studio`.
- Endpoints: `GET /api/results/{id}/changes`, `POST /api/results/{id}/revert`,
  `POST /api/results/{id}/reapply`, `POST /api/results/{id}/refine` (retry, 202 + job),
  `GET /api/results/{id}/versions`, `GET .../versions/{vid}`, `GET .../versions/{vid}/compare`,
  `POST .../versions/{vid}/restore`.
- Settings: `LLM_REFINE` (true), `REFINE_MAX_CHANGE` (0.4).
- Files: `app/plugins/refine.py`, `app/modules/refinement.py`, `text_diff.py`, `reliability.py`,
  `llm_answer.py`, `embedder.py`, `app/routers/refine.py`, `app/plugins/base.py`.
- Tests: `tests/test_refine.py` (22: diff, rejection rule, plugin parts / retries / quota,
  auto-refine on upload, progress messages, fallback without key and on quota, retry, revert and
  re-embedding, changes, versions, manual edits, typed and handwritten files, reliability cap),
  `tests/refine_fakes.py`.
- History: a first version (commit `05fc2ef`) had a manual "Refine with LLM" button with a review /
  accept step; it was replaced by the automatic flow (`9f74687`, `97c2e40`).

### Layout-preserving view and export

- Layout view (`app/static/layout.js` + `layout.css`, shared by both pages): the page rebuilt at
  its real geometry — lines and words at their boxes (word gaps, indentation, column spacing),
  titles bold, tables as real HTML tables with merged cells at their block positions, figures and
  formulas in place; zoomable; hovering a line highlights its box on the page image. Office and
  txt files are shown as structured Markdown and tables (`.txt` with its own spacing).
  In `/studio` as the Layout tab, on the QA page as the viewer's Layout / Original-text modes.
- Layout text export: monospace text with columns and paragraph spacing from the word / line boxes
  (`GET /api/results/{id}/export/layout`, file `<name>.layout.txt`).
- Files: `app/static/layout.js`, `layout.css`, `app/modules/exporters.py` (`page_layout_text`).
- Tests: `tests/test_studio_tools_api.py::test_export_layout_text_keeps_columns_and_spacing`,
  `tests/test_studio_page.py::test_shared_layout_assets_are_served_and_used_by_both_pages`.

### Unified extraction (Module 2) for both pages

- `app/modules/document_extract.py` reads any upload into Experience Center pages: exact text for
  typed files (PDF line / word / table boxes from pdfplumber, page images from PyMuPDF), PaddleOCR
  or PP-StructureV3 for scans, TrOCR reading each of Paddle's line boxes for handwriting.
  Scanned vs handwritten: user hint, else Paddle's calibrated confidence and, when low, a TrOCR
  sample on the same boxes. The classic engines (Tesseract / docTR + TrOCR) remain as a fallback
  (`EXTRACTION_ENGINE=classic`). `/api/ocr` and `/api/parse` accept `format_hint` and `.txt`.
- Lines carry `source` (`text`, `paddleocr`, `trocr`, `tesseract`, `llm`, `human`).

### QA page features (document filter, "Did you mean", document viewer)

- Document checkboxes: answers use only the ticked documents (remembered per browser); `/ask`
  takes optional `doc_ids` via a request wrapper (`AskRequest`), so `schemas.py` is unchanged;
  `vector_store` / `retrieval` filter by `doc_id`.
- "Did you mean": fuzzy match of question words against the selected documents' vocabulary, then
  an optional Gemini rewrite that may use only those documents' terms (`app/modules/spelling.py`).
- Document viewer: the original file inline; source citations open PDFs at `#page=N`.
- Endpoints: `POST /ask` (`doc_ids`), `POST /ask/suggest`, `GET /documents/{doc_id}/file`.
- Files: `app/modules/spelling.py`, `vector_store.py`, `retrieval.py`, `app/routers/query.py`,
  `upload.py`, `app/static/index.html`.
- Tests: `tests/test_doc_filter.py` (8), `tests/test_spelling.py` (18), `tests/test_document_file.py` (6).

### Experience Center (`/studio`)

- A PaddleOCR workspace in DocBlendAI's own design: drag-and-drop (several files, queued),
  built-in samples, pipeline (Text OCR / Document parsing / PaddleOCR-VL shown "unavailable"
  unless enabled), language and document type; page image with confidence-coloured line boxes
  (zoom, pan, page navigation) next to tabs Text (click to correct a line), Layout, Changes,
  Markdown, JSON, Tables, Plugins, Calibration; raw vs calibrated confidence toggle, "hide below"
  filter, review-threshold slider; light / dark theme; export buttons.
- Files: `app/static/studio.html`, `studio.css`, `studio.js`, `app/static/samples/*.png`
  (`scripts/make_studio_samples.py`), route `GET /studio` in `app/main.py`.
- Tests: `tests/test_studio_page.py` (8).

### PaddleOCR service and API

- `PaddleOCRService` (`app/modules/paddleocr_service.py`): singleton, models lazy-loaded once per
  (pipeline, language), thread-safe; text OCR (PP-OCRv5 mobile by default, PP-OCRv6 supported),
  PP-StructureV3 (blocks, tables with cell boxes, formulas, Markdown), PaddleOCR-VL (optional,
  off), Office → Markdown (`doc2md`). Images and multi-page PDFs (PyMuPDF at 150 DPI).
- Job manager (`app/modules/ocr_jobs.py`): one worker thread, progress, run history in SQLite
  (table `ocr_runs`), page images under `data/paddle_runs/`, human corrections.
- Endpoints: `GET /api/health`, `POST /api/ocr`, `POST /api/parse`, `GET /api/results`
  (history, `?q=`), `GET|DELETE /api/results/{id}`, `PATCH /api/results/{id}/lines`,
  `GET /api/results/{id}/pages/{n}/image`. File type and size validation, `wait=true` for
  synchronous use. Contract: `docs/experience_center_contract.md`.
- Settings: `PADDLE_DEVICE`, `PADDLE_OCR_VERSION`, `PADDLE_DET_MODEL`, `PADDLE_REC_MODEL`,
  `PADDLE_ENABLE_MKLDNN`, `PADDLE_USE_ORIENTATION`, `PADDLE_USE_UNWARPING`,
  `PADDLE_USE_TEXTLINE_ORIENTATION`, `PADDLE_PDF_DPI`, `PADDLE_MAX_PAGES`, `PADDLE_MAX_UPLOAD_MB`,
  `PADDLE_FORMULA_MODEL`, `PADDLE_ENABLE_VL`, `PADDLE_VL_VERSION`, `REVIEW_THRESHOLD`.
- Tests: `tests/test_paddleocr_service.py` (28), `tests/test_paddleocr_api.py` (34),
  `tests/paddle_fakes.py`, recorded real results in `tests/fixtures/`; a guard in `conftest.py`
  fails any test that tries to load a real Paddle model.

### Confidence calibration (PaddleOCR)

- `app/calibration/`: temperature scaling, Platt scaling, isotonic regression (pool adjacent
  violators with an add-one prior per block); ECE, MCE, Brier score, NLL, reliability bins;
  stratified train/validation split, best calibrator by validation ECE (tie-break Brier); line
  labelling by exact match or CER ≤ threshold; reliability diagram PNG + JSON report; every OCR
  line returns `raw_confidence` and `calibrated_confidence` ("uncalibrated" until fitted).
- Sample dataset: 40 images, 119 lines in `evaluation/calibration_sample/`
  (`python -m app.calibration.make_sample`); CLI `python -m app.calibration.calibrate`.
  On the sample: isotonic chosen; validation ECE 0.131 → 0.121, Brier 0.103 → 0.050.
- Endpoints: `POST /api/calibrate` (zip or `use_sample=true`), `GET /api/calibration`,
  `GET /api/calibration/diagram.png`.
- Settings: `CALIBRATION_CER_THRESHOLD`; output in `data/paddle_calibration/`.
- Tests: `tests/test_calibration.py` (50).

### Exports, preprocessing, plugins

- Exports: TXT, layout text, Markdown, JSON, searchable PDF (page image + invisible text layer),
  Word (from the Markdown, tables included), CSV (one table, or a zip of several)
  (`app/modules/exporters.py`, `GET /api/results/{id}/export/{fmt}`).
- Preprocessing before OCR: deskew (projection-profile search, robust to speckle and grey scan
  borders), denoise, contrast (CLAHE), binarize, with a before / after preview
  (`app/modules/preprocess.py`, `POST /api/preprocess`, `preprocess` field on `/api/ocr`, `/api/parse`).
- Plugins (Gemini, off without `GEMINI_API_KEY`): key-information extraction (values must occur
  in the text) and translation (keeps Markdown structure) (`app/plugins/`, `GET /api/plugins`,
  `POST /api/results/{id}/plugins/{name}`).
- Tests: `tests/test_experience_extras.py` (22), `tests/test_studio_tools_api.py` (14).

### History and logging

- Run history in SQLite (`ocr_runs`) with search by file name and recognised text; now shown as
  the shared Documents list.
- Structured logging: `LOG_FORMAT=json` gives one JSON object per line with job ids, pipeline and
  timings as fields (`app/logging_config.py`).

### Environment fixes (local venv on Windows)

- Dependencies pinned in `requirements.txt`: `paddlepaddle==3.3.1`,
  `paddleocr[doc-parser,doc2md]==3.7.0` (the extras PP-StructureV3 and Office conversion need),
  `paddlex==3.7.2`, `PyMuPDF==1.28.2`, `transformers==5.17.0`.
- OpenCV: `opencv-python`, `opencv-python-headless` and `opencv-contrib-python` all pinned to
  `4.10.0.84` (paddlex's version) so the three wheels no longer overwrite each other's `cv2`.
- MKL-DNN (oneDNN) off by default (`PADDLE_ENABLE_MKLDNN=false`): paddle 3.3 on Windows CPU fails
  with "ConvertPirAttribute2RuntimeAttribute ... Unimplemented".
- `torch` is imported before `paddle` (otherwise torch fails with "WinError 127 ... shm.dll" in the
  same process as TrOCR / docTR).
- PP-StructureV3 uses `PP-FormulaNet_plus-S` and the mobile OCR models: the default `-L` formula
  model with server OCR models ran out of memory (segfault) on an 8 GB laptop.
- `PyYAML==6.0.2` (paddlex pins it exactly); `pip check` still warns that chromadb's `kubernetes`
  extra wants ≥ 6.0.3 — harmless (APIs identical).
- `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True` set by the service (no connectivity check per start).
- `scripts/paddle_smoke_test.py`: checks the PaddleOCR install (imports, models, one OCR run, one
  PP-StructureV3 parse).

---

## [0.1.0] — `b0030d4`

Confidence-aware QA over typed, scanned and handwritten documents: upload (PDF, images, docx, pptx,
txt), format detection, parsing / Tesseract or docTR OCR / TrOCR HTR, confidence calibration,
content types, confidence-aware retrieval (ChromaDB), four-tier reliability labels, Gemini
answers, evaluation (CER, WER, Exact / Semantic Match). 193 tests.
