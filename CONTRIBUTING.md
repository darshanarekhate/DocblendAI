# Contributing to DocBlendAI

Team guide for Group C4. Read `README.md` for what the project does and `CLAUDE.md`
for the architecture decisions (they are locked; discuss before changing them).

## First-time setup

Two options. **Docker** is the quickest way to run the app on any computer (no Python, Tesseract
or model setup): install Docker Desktop, then from the repository folder run
`powershell -ExecutionPolicy Bypass -File scripts\docker-start.ps1 -Build` (Windows) or
`bash scripts/docker-start.sh --build` (Linux/macOS). Details, first-run times and troubleshooting:
README, "Run with Docker". Use the **local venv** below to develop the code and run the tests.

### Local venv (Windows)

1. **Clone** (you need to be a collaborator on the private repo):
   ```bash
   git clone https://github.com/om-upadhye/DocBlendAI.git
   cd DocBlendAI
   ```
2. **Python 3.11 virtualenv** and dependencies:
   ```bash
   py -3.11 -m venv venv
   venv/Scripts/python -m pip install -r requirements.txt
   ```
   Always run tools through `venv/Scripts/python -m ...` (not the `venv/Scripts/*.exe` launchers).
   This also installs PaddleOCR (`paddlepaddle==3.3.1`, `paddleocr[doc-parser,doc2md]==3.7.0`),
   PyMuPDF and the pinned OpenCV wheels (all three at 4.10.0.84; do not upgrade one of them alone).
   `pip check` warns that `kubernetes` wants PyYAML >= 6.0.3 while paddlex pins 6.0.2: harmless.
   Check the PaddleOCR install with `venv/Scripts/python scripts/paddle_smoke_test.py`
   (`RESULT: PASS`), and optionally download every model at once with
   `venv/Scripts/python scripts/preload_models.py` (~2 GB).
3. **Tesseract OCR** (needed for scanned PDFs, images, and page-orientation correction):
   ```bash
   winget install UB-Mannheim.TesseractOCR
   ```
   It installs to `C:\Program Files\Tesseract-OCR`, which the app finds automatically.
   Optional: without Tesseract the app reads printed pages with docTR instead (`OCR_ENGINE=auto`),
   but page-orientation correction still needs Tesseract.
4. **Gemini API key:** copy `.env.example` to `.env` and put your key after `GEMINI_API_KEY=`.
   Get a key at https://aistudio.google.com/apikey.
   > **Never put a real key in `.env.example`** or any other committed file. `.env` is git-ignored; `.env.example` is not.
5. **Run it:**
   ```bash
   venv/Scripts/python -m uvicorn app.main:app --reload
   ```
   Open http://127.0.0.1:8000/ (QA page), http://127.0.0.1:8000/studio (Experience Center) or
   http://127.0.0.1:8000/docs (API). First uses download the PaddleOCR models (~1.2 GB),
   TrOCR (~0.5 GB) and docTR (~160 MB), once. With `--reload`, saving a `.py` file restarts the
   server and interrupts a document that is being read (upload it again).

## Daily workflow

`main` → feature branch → commits → push → Pull Request → review → merge.

1. Start from an up-to-date `main`:
   ```bash
   git checkout main
   git pull
   git checkout -b feature/<short-name>
   ```
2. Commit small, focused changes with messages that say **what** changed and **why**.
3. Run the tests before pushing:
   ```bash
   venv/Scripts/python -m pytest
   ```
4. Push and open a PR into `main`:
   ```bash
   git push -u origin feature/<short-name>
   ```
5. At least **one teammate reviews** before merging. The author merges after approval.

Rules:
- **Never commit directly to `main`.**
- **Never commit secrets** (`.env`, API keys) or local data (`data/uploads/`, `data/chroma_db/`, `data/*.db` are git-ignored).
- Keep module names and boundaries as listed in `CLAUDE.md` (the 7 synopsis modules).
- `app/models/schemas.py` must match the synopsis entity table exactly.

## Where things live

| Module | File(s) |
|---|---|
| 1. Document Upload | `app/routers/upload.py` |
| 2. Format Detection & Text Extraction | `app/modules/format_detection.py`, `file_types.py`, `text_parser.py` (PDF/Word/PowerPoint/text), `ocr_extractor.py`, `htr_extractor.py`, `pdf_render.py` (PDF pages and images) |
| 3. Confidence Capture & Calibration | `app/modules/confidence_capture.py`, `data/calibration.json` |
| 4. Content-Type Identification | `app/modules/content_type.py` |
| 5. Confidence-Aware Retrieval | `app/modules/chunker.py`, `embedder.py`, `vector_store.py`, `retrieval.py`, `spelling.py` ("Did you mean") |
| 6. Reliability Tier Classification | `app/modules/reliability.py` |
| 7. LLM Answer Generation | `app/modules/llm_answer.py`, `app/routers/query.py` |
| Shared upload pipeline (Modules 1-5) | `app/modules/document_extract.py` (one extractor for both pages: exact text / PaddleOCR / PP-StructureV3 / TrOCR), `ocr_jobs.py` (background job: extract -> refine -> index), `qa_index.py` (QA chunks from a run), `document_runs.py` (QA document <-> Experience Center run) |
| Automatic LLM refinement | `app/plugins/refine.py`, `app/modules/refinement.py`, `text_diff.py`, `app/routers/refine.py` |
| Experience Center (PaddleOCR) | `app/modules/paddleocr_service.py`, `preprocess.py`, `exporters.py`, `app/routers/paddleocr.py`, `studio_tools.py`, `app/plugins/` |
| PaddleOCR confidence calibration | `app/calibration/`, `evaluation/calibration_sample/` |
| Evaluation | `evaluation/`, `tests/eval_metrics.py` |
| Frontend | `app/static/index.html` (QA page), `studio.html/.css/.js` (Experience Center), `layout.js/.css` (shared) |
| Docker | `Dockerfile`, `docker-compose.yml`, `scripts/docker-start.*`, `scripts/preload_models.py`, `.github/workflows/docker.yml` |

API shapes for the Experience Center, refinement and versions: `docs/experience_center_contract.md`.
What changed and where: `CHANGELOG.md` (add an entry when you add a feature).

## Tests and evaluation

- `venv/Scripts/python -m pytest`: 403 offline unit/API tests (~1 minute). Tesseract, TrOCR, docTR, PaddleOCR, embeddings and Gemini are faked, so no key or network is needed. Guards in `tests/conftest.py` fail any test that tries to load a real TrOCR, docTR or Paddle model; uploads use the classic engines (faked) and LLM refinement is off unless a test turns it on.
  - One area at a time: `venv/Scripts/python -m pytest tests/test_refine.py` (LLM refinement, shared documents), `tests/test_paddleocr_api.py` / `test_paddleocr_service.py` (Experience Center engine and API), `tests/test_calibration.py`, `tests/test_studio_tools_api.py` (exports, preprocessing, plugins), `tests/test_doc_filter.py`, `test_spelling.py`, `test_document_file.py` (QA page features).
  - Fakes to reuse: `tests/paddle_fakes.py` (Paddle models from recorded real results in `tests/fixtures/`), `tests/refine_fakes.py` (`FakeGemini` for the `refine` plugin), `fake_embed` / `fake_ocr` / `fake_htr` in `tests/conftest.py`.
- PaddleOCR calibration with the real engine (~3 minutes): `venv/Scripts/python -m app.calibration.calibrate` (writes `data/paddle_calibration/`, git-ignored).
- Full evaluation with the real engines and Gemini (takes ~15 minutes):
  ```bash
  venv/Scripts/python -m evaluation.make_dataset
  venv/Scripts/python -m evaluation.run_eval --fit-calibration
  ```
  Writes `evaluation/results/report.md` and refits `data/calibration.json`. Commit both when you change recognition or calibration.
- HTR temperature scaling (after `make_dataset`): `venv/Scripts/python -m evaluation.fit_htr_temperature`, then put the printed `HTR_TEMPERATURE` in `.env` and rerun `run_eval --fit-calibration`.
- Noise robustness, OHRBench-style (calls Gemini): `venv/Scripts/python -m evaluation.noise_robustness`. Writes `evaluation/results/noise_robustness.md`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Server refuses to start: "database … is out of date" | The table layout changed. Stop the server, delete `data/docblendai.db` and the contents of `data/chroma_db/` (keep `.gitkeep`), restart, re-upload. |
| Upload returns **503** mentioning Tesseract | Install Tesseract (setup step 3), or set `TESSERACT_CMD` in `.env` to the full path of `tesseract.exe`. |
| Upload returns **503** mentioning the HTR model | First run needs internet to download TrOCR; also rerun `pip install -r requirements.txt` (needs `sentencepiece`, `torchvision`). |
| Handwritten notebook page reads as gibberish | Check `REMOVE_RULED_LINES=true` and `HTR_SEGMENTER=doctr` (the defaults). For hard handwriting try `HTR_MODEL=microsoft/trocr-base-handwritten` (more accurate, ~4x slower). |
| `/ask` returns **502** | Both `LLM_MODEL` and `LLM_FALLBACK_MODEL` failed: Gemini is busy, rate-limited, or out of free quota; the server log shows which (`RESOURCE_EXHAUSTED` = quota). Free keys have **daily** per-model limits, so each member should use their **own** key. The default answer model is `gemini-3.5-flash-lite` for its larger free quota; `LLM_MODEL=gemini-3.5-flash` in `.env` needs a paid key for more than 20 answers/day. |
| `/ask` returns **409** | That `query_id` was used before; send a new one (the web UI does this for you). |
| `/ask` returns **503** "Gemini is not configured" / **429** "quota ... used up" | Set `GEMINI_API_KEY` in `.env` and restart; for 429 wait for the daily reset or change `LLM_MODEL` / `LLM_FALLBACK_MODEL`. Documents then show "Not refined: Gemini unavailable" with a Retry link. |
| PaddleOCR fails with `ConvertPirAttribute2RuntimeAttribute ... Unimplemented` | Keep `PADDLE_ENABLE_MKLDNN=false` (the default; paddle 3.3's oneDNN kernels fail on Windows). |
| `WinError 127 ... shm.dll` when TrOCR / docTR loads | paddle was imported before torch in that process; the app imports torch first. In your own scripts, `import torch` before `paddleocr`. |
| Document parsing crashes the server (segfault) on an 8 GB laptop | Keep the default `PADDLE_FORMULA_MODEL=PP-FormulaNet_plus-S`, or set `INGEST_PIPELINE=ocr`. |
| A document shows "Interrupted" | The server restarted while reading it (e.g. `--reload` after saving a `.py` file): upload it again. |
| `gh` not found after installing GitHub CLI | Open a new terminal, or call `"C:\Program Files\GitHub CLI\gh.exe"`. |
