# DocBlendAI

Confidence-aware multi-format document QA assistant: upload typed, scanned, or
handwritten academic documents and ask questions. Every answer carries a reliability
tier (Certain / Moderate / Uncertain / Unreadable).

**Supported uploads:** PDF, images (JPG, PNG, TIFF incl. multi-page, BMP, WEBP;
phone photos are turned upright automatically), Word (.docx, including tables),
PowerPoint (.pptx, one page per slide), and plain text (.txt). See `CLAUDE.md` for full
project context.

**Team members:** start with [CONTRIBUTING.md](CONTRIBUTING.md) for setup, the Git workflow, and troubleshooting.

## Setup

1. Python 3.11 virtualenv in `venv/`, then:
   ```bash
   venv/Scripts/python -m pip install -r requirements.txt
   ```
2. Tesseract OCR (for scanned PDFs):
   ```bash
   winget install UB-Mannheim.TesseractOCR
   ```
   The default install location is found automatically; otherwise set `TESSERACT_CMD` in `.env`.
3. Copy `.env.example` to `.env` and set `GEMINI_API_KEY`. Never put the key in `.env.example`.

The TrOCR handwriting model (~250 MB) downloads automatically on the first handwritten upload.

## Running

```bash
venv/Scripts/python -m uvicorn app.main:app --reload
```

- **App:** http://127.0.0.1:8000/ to upload PDFs, ask questions, and see reliability labels and sources
- **API docs:** http://127.0.0.1:8000/docs

If the server refuses to start with "database is out of date", delete `data/docblendai.db`
and the contents of `data/chroma_db/` (local dev data), restart, and re-upload.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/upload` | Upload a PDF, image, .docx, .pptx, or .txt (optional `format_hint`: typed / scanned / handwritten) |
| GET | `/documents` | List uploaded documents |
| GET | `/documents/{id}/file` | The original uploaded file, inline (document viewer) |
| DELETE | `/documents/{id}` | Remove a document (chunks, record, and file) |
| POST | `/ask` | Ask a question → answer + reliability label (optional `doc_ids`: only use these documents) |
| POST | `/ask/suggest` | "Did you mean": spelling-corrected question from the selected documents' words (`question_text`, optional `doc_ids`) |
| GET | `/answer/{id}` | Fetch a stored answer |
| GET | `/answer/{id}/sources` | Chunks behind an answer, with similarity, confidence, content type |

## Testing and evaluation

```bash
venv/Scripts/python -m pytest
venv/Scripts/python -m evaluation.make_dataset
venv/Scripts/python -m evaluation.run_eval --fit-calibration
```

Tests fake Tesseract, TrOCR, and Gemini, so they run offline. The evaluation uses the
real engines: it generates a synthetic dataset (typed, degraded scans, handwriting-style
fonts), measures CER/WER, fits confidence calibration into `data/calibration.json`,
and scores question answering (Exact Match, Semantic Match, retrieval hits, accuracy per
reliability tier). The latest report is in [evaluation/results/report.md](evaluation/results/report.md).

## Folder structure

```
app/
├── main.py                  FastAPI app, router registration, /health, serves the UI
├── config.py                Settings from .env (pydantic-settings)
├── static/index.html        Demo UI
├── routers/
│   ├── upload.py            Module 1: POST /upload, GET/DELETE /documents, GET /documents/{id}/file
│   └── query.py             POST /ask, POST /ask/suggest, GET /answer/{id}[/sources] (Modules 5-7)
├── modules/
│   ├── format_detection.py  Module 2: detect format, route to an extractor
│   ├── text_parser.py       Module 2: direct parsing of typed PDFs
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
│   └── llm_answer.py        Module 7: Gemini answer generation
├── db/
│   ├── database.py          SQLAlchemy engine/session (SQLite), schema check
│   └── models.py            ORM: Document, Query, Answer, RetrievalResult
└── models/
    └── schemas.py           Pydantic entities shared by all modules
data/
├── uploads/                 Uploaded PDFs (git-ignored)
├── chroma_db/               ChromaDB persistence (git-ignored)
└── calibration.json         Fitted confidence calibration (committed, shared)
evaluation/
├── make_dataset.py          Synthetic typed / scanned / handwritten dataset
├── run_eval.py              Recognition, calibration, and QA evaluation
└── results/report.md        Latest evaluation report
tests/
├── test_*.py                Unit and API tests (offline)
└── eval_metrics.py          Exact Match, Semantic Match, CER, WER
```
