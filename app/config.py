"""Application settings, loaded from .env via pydantic-settings.

Shared by all modules; not tied to a single module number. Import the
module-level `settings` object rather than reading os.environ directly.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    # Empty default keeps the app bootable without a key; Gemini calls will fail until it is set.
    gemini_api_key: str = ""
    # gemini-embedding-001 (not -2): -2 merges a list of texts into ONE vector instead of one per text.
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768
    # Lite by default: the free tier allows only 20 gemini-3.5-flash answers per day per project,
    # too few for a demo or one evaluation run. Set LLM_MODEL=gemini-3.5-flash in .env on a paid key.
    llm_model: str = "gemini-3.5-flash-lite"
    # Second Gemini model for when llm_model is out of quota (429) or overloaded (503).
    # Free-tier quotas are per model, so this keeps answers coming. Empty = no fallback.
    llm_fallback_model: str = "gemini-3.5-flash"
    top_k: int = 5  # chunks retrieved per question
    # combined_score = w * similarity + (1 - w) * calibrated confidence. Relevance dominates:
    # a clearly-read but off-topic chunk should not outrank a relevant, less legible one.
    similarity_weight: float = 0.7
    # Re-ranking pool: fetch top_k * this many by similarity, then re-sort by combined_score.
    candidate_multiplier: int = 3
    # Chunks whose similarity is within this of the best match count as relevant, and relevant
    # documents share the top_k slots (see retrieval.py), so a large clearly-typed document
    # cannot crowd out a less legible one that matches the question better.
    relevance_margin: float = 0.05

    # OCR / HTR (build step 4)
    tesseract_cmd: str = ""  # path to tesseract.exe if it is not on PATH
    ocr_dpi: int = 300
    # Mean Tesseract confidence (0-1) on a page with no text layer: at or above -> printed scan,
    # below -> treated as handwriting. First guess; tune on real samples.
    scanned_min_ocr_conf: float = 0.6
    # Small model by default: runs on CPU laptops. trocr-base-handwritten is more accurate but ~4x slower.
    htr_model: str = "microsoft/trocr-small-handwritten"
    # How HTR finds text lines: "doctr" (docTR's DBNet word detector, grouped into lines) or
    # "projection" (ink-row profile; no model download). doctr falls back to projection if unavailable.
    htr_segmenter: str = "doctr"
    # Erase notebook rules and margin lines (OpenCV) before finding lines: ruled paper otherwise
    # looks like one full-width line of ink per rule.
    remove_ruled_lines: bool = True
    # Temperature scaling of TrOCR token probabilities (Ayllon et al., ICDAR 2024: raw HTR
    # confidence is overconfident). 1.0 = off; fit with `python -m evaluation.fit_htr_temperature`.
    htr_temperature: float = 1.0
    # Printed-text OCR engine: "tesseract", "doctr" (mindee/doctr, no system install), or "auto"
    # (Tesseract when installed, otherwise docTR).
    ocr_engine: str = "auto"
    calibration_file: Path = DATA_DIR / "calibration.json"
    # Load the OCR/HTR models in the background when the server starts, so the first
    # scanned or handwritten upload does not wait ~15 s for them.
    preload_models: bool = True

    database_url: str = f"sqlite:///{(DATA_DIR / 'docblendai.db').as_posix()}"
    chroma_dir: Path = DATA_DIR / "chroma_db"
    upload_dir: Path = DATA_DIR / "uploads"


settings = Settings()
