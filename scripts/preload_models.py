"""Download every model DocBlendAI loads on first use, so the first OCR run downloads nothing.

Used by the Docker build (`--build-arg PRELOAD_MODELS=true`), and works locally too:

    venv/Scripts/python scripts/preload_models.py

Downloads (into the usual caches: PADDLE_PDX_CACHE_HOME, HF_HOME, DOCTR_CACHE_DIR):
- PaddleOCR text OCR and PP-StructureV3 with the app's configured models (app/config.py:
  PP-OCRv5 mobile det/rec, PP-FormulaNet_plus-S; layout and table models of PP-StructureV3)
- TrOCR (settings.htr_model, handwriting)
- docTR's text detector and recogniser (handwriting line finder; OCR without Tesseract)

Only app/config.py is imported (not the rest of the app), so in the Dockerfile this step can run
before the application code is copied and stays cached when the code changes. Models are only
constructed, never run, so this works on any CPU.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from app.config import settings  # noqa: E402  (after the path / env setup above)


def step(name, fn):
    started = time.perf_counter()
    print(f"-> {name}", flush=True)
    fn()
    print(f"   done in {time.perf_counter() - started:.0f} s", flush=True)


def paddle_models() -> None:
    import torch  # noqa: F401  (load before paddle, like the app does: shared OpenMP/MKL libraries)
    from paddleocr import PaddleOCR, PPStructureV3

    common = {"device": "cpu", "enable_mkldnn": False}  # construction only: no inference
    names = {"text_detection_model_name": settings.paddle_det_model,
             "text_recognition_model_name": settings.paddle_rec_model}
    step("PaddleOCR text OCR", lambda: PaddleOCR(
        lang="en", ocr_version=settings.paddle_ocr_version, use_doc_orientation_classify=False,
        use_doc_unwarping=False, use_textline_orientation=False, return_word_box=True, **names, **common))
    step("PP-StructureV3 (layout, tables, formulas)", lambda: PPStructureV3(
        lang="en", use_doc_orientation_classify=False, use_doc_unwarping=False, use_seal_recognition=False,
        use_chart_recognition=False, formula_recognition_model_name=settings.paddle_formula_model,
        **names, **common))


def trocr() -> None:
    from huggingface_hub import snapshot_download

    step(f"TrOCR ({settings.htr_model})", lambda: snapshot_download(settings.htr_model))


def doctr_models() -> None:
    from doctr.models import detection_predictor, ocr_predictor

    step("docTR detector + recogniser", lambda: (
        detection_predictor("db_resnet50", pretrained=True, assume_straight_pages=True),
        ocr_predictor("db_resnet50", "crnn_vgg16_bn", pretrained=True, assume_straight_pages=True),
    ))


if __name__ == "__main__":
    for part in (paddle_models, trocr, doctr_models):
        part()
    print("All models are cached.")
