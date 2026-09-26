# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# DocBlendAI

Final-year B.Tech project (Group C4) — a confidence-aware multi-format document
QA assistant. Guide: Prof. R. A. Tiwari, Dept. of CSE, PRMIT&R Badnera.

## What it does

Students, researchers, and faculty work with academic material in mixed formats
(typed, scanned/printed, handwritten). DocBlendAI detects the format of an
uploaded document, extracts text with the right method for that format, and
answers user questions through a confidence-aware Retrieval-Augmented
Generation (RAG) pipeline — always labeling how trustworthy each answer is.

Reliability tiers: **Certain / Moderate / Uncertain / Unreadable**, derived from
jointly weighing recognition confidence and retrieval relevance.

## Scope

**In scope:** typed, scanned/printed, and handwritten academic documents,
uploaded as PDF, images (JPG/PNG/TIFF/BMP/WEBP, incl. phone photos),
Word (.docx), PowerPoint (.pptx), or text (.txt); automatic format detection
and page-orientation correction; content-type identification
(table/paragraph/image); confidence-aware retrieval; four-tier reliability labeling.
(The approved synopsis said "academic PDFs"; image/Office/text uploads were
added at the team's request. Supported extensions live in `app/modules/file_types.py`.)

**Out of scope:** live classroom transcription, multilingual scripts,
real-time video content.

## 7 core modules (per approved synopsis — keep names/boundaries in sync with this)

1. **Document Upload**
2. **Format Detection & Text Extraction** — routes to direct parsing / OCR / HTR
3. **Confidence Capture & Calibration**
4. **Content-Type Identification** — table / paragraph / image
5. **Confidence-Aware Retrieval (RAG)** — combined_score = similarity + calibrated confidence
6. **Reliability Tier Classification**
7. **LLM Answer Generation**

## Architecture decisions (locked — see reasoning before changing)

- **Pattern: single FastAPI monolith**, not microservices. Modules are separated
  by Python file/folder boundaries, not network boundaries — keeps deployment
  and demo simple while still letting 4 people work on separate modules in
  parallel.
- **Storage: ChromaDB + SQLite**, not ChromaDB alone.
  - ChromaDB: `RecognizedChunk.vector` + similarity search only
  - SQLite (via SQLAlchemy): `Document`, `Query`, `Answer`, `RetrievalResult` —
    anything relational/filterable (format_type, reliability_label, user_id, etc.)
  - SQLite chosen over MySQL/Postgres: zero server setup across 4 laptops,
    single file, plenty for this scale.

## Entities (from synopsis Problem Analysis table — schemas.py must match exactly)

- `Document`: doc_id, file_path, format_type, page_count
- `RecognizedChunk`: chunk_id, text, content_type, raw_conf, calibrated_conf, vector
- `Query`: query_id, question_text, user_id
- `RetrievalResult`: chunk_id, similarity, confidence, combined_score
- `Answer`: answer_id, answer_text, reliability_label

## Stack

FastAPI, ChromaDB, SQLite + SQLAlchemy, Gemini API (LLM + embeddings),
pytesseract (OCR), TrOCR/transformers (HTR), jiwer (CER/WER eval)

Gemini is called through the `google-genai` SDK (the older `google-generativeai`
is deprecated): `gemini-embedding-001` for embeddings, `gemini-3.5-flash-lite` for
answers (both overridable in `.env`; the free tier allows only 20 `gemini-3.5-flash`
answers per day per project, so flash needs a paid key). pytesseract also requires the Tesseract
binary installed on the system (`winget install UB-Mannheim.TesseractOCR`).
See CONTRIBUTING.md for team setup and workflow.

### Open-source integrations (synopsis-named and additional)

| Repo | Where | Role |
|---|---|---|
| Temperature scaling, Ayllon et al. ICDAR 2024 (synopsis ref [3]) | `htr_extractor.py`, `confidence_capture.fit_temperature`, `evaluation/fit_htr_temperature.py` | Softens TrOCR's overconfident token probabilities (`HTR_TEMPERATURE`) |
| OHRBench, opendatalab/OHR-Bench (synopsis ref [2]) | `evaluation/noise_robustness.py` | Idea only (graded injected OCR noise), not its data or code: checks that labels drop as recognition noise rises |
| mindee/doctr | `line_segmentation.detect_lines`, `ocr_extractor._doctr_ocr` | DBNet text detection finds handwritten lines for TrOCR; full docTR OCR replaces Tesseract when it is not installed (`OCR_ENGINE=auto`) |
| opencv/opencv | `line_segmentation.remove_ruled_lines` | Erases notebook rules and margin lines before HTR (the cause of gibberish on real notebook pages) |

docTR downloads its weights (~165 MB) to `~/.cache/doctr` on first use. Tests never
load docTR or TrOCR, or call Gemini (see `tests/conftest.py`).

Answers use Gemini only (the synopsis's Qwen2.5 option was dropped at the team's
request). When `LLM_MODEL` is out of quota (429) or overloaded (503), `llm_answer`
retries on `LLM_FALLBACK_MODEL`: free-tier quotas are counted per model.

## Environment & commands

- Python 3.11 with a local virtualenv in `venv/` (git-ignored).
- The venv was created at a pre-OneDrive path (`C:\Users\USER\Desktop\Projects\DocBlendAI`).
  Call tools via the interpreter (`venv/Scripts/python -m pip ...`) rather than
  the `venv/Scripts/*.exe` launchers, which may embed the old path.
- Secrets load from `.env` via python-dotenv; `.env.example` lists required keys
  (currently `GEMINI_API_KEY`).

```bash
venv/Scripts/python -m pip install -r requirements.txt
```

## Evaluation metrics

Exact Match, Semantic Match, Character Error Rate (CER), Word Error Rate (WER)
— across all three document formats.

## Git workflow

`main → feature branch → development → commit → push → Pull Request → review → merge`

Never commit directly to `main`. Each member works on their own feature branch
for their owned module(s).

## Build order (do not skip ahead)

1. Typed-text path only (upload → parse → chunk) — no OCR/HTR, no confidence
2. Embedding + ChromaDB storage
3. Retrieval + Gemini answer generation (typed-only) — first working end-to-end checkpoint
4. Add OCR (pytesseract) + HTR (TrOCR) + confidence capture/calibration
5. Content-type identification + confidence-aware combined_score + reliability labeling
6. Evaluation metrics + frontend + polish

Do not build confidence.py, ocr_extractor.py, or reliability.py before step 3's
typed-only pipeline works end-to-end — parallel half-finished modules stall
integration.
