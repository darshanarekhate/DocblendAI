# DocBlendAI: full pipeline (PaddleOCR + PP-StructureV3, Tesseract, TrOCR/docTR, Gemini) on CPU.
#
#   docker compose up --build        then open http://localhost:8000
#
# Model weights (PaddleOCR ~1 GB, TrOCR, docTR) download on first use into the
# model-cache volume declared in docker-compose.yml, so they survive rebuilds.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Model caches live under /models (a volume): paddlex, Hugging Face (TrOCR), docTR.
    PADDLE_PDX_CACHE_HOME=/models/paddlex \
    HF_HOME=/models/huggingface \
    DOCTR_CACHE_DIR=/models/doctr \
    # Skip paddlex's per-start connectivity check of model hosters.
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# libgl1/libglib2.0-0: OpenCV; libgomp1: paddle/torch OpenMP; tesseract-ocr: the existing scanned-PDF path.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 tesseract-ocr curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so code edits do not reinstall them. The CPU wheel index keeps torch at
# ~200 MB instead of the multi-GB CUDA build; paddlepaddle/paddleocr/PyMuPDF are pinned in requirements.txt.
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY app ./app
COPY evaluation ./evaluation
COPY scripts ./scripts

RUN mkdir -p data/uploads data/chroma_db /models

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fs http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
