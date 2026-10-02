# syntax=docker/dockerfile:1.7
# DocBlendAI: QA page + Experience Center (PaddleOCR / PP-StructureV3, TrOCR, docTR, Tesseract,
# Gemini) on CPU, in one container.
#
#   docker compose up -d --build                        build and start (see docker-compose.yml)
#   docker build -t docblendai .                        image only, models download on first use
#   docker build --build-arg PRELOAD_MODELS=true -t docblendai:models .
#                                                       ~2 GB larger, the first OCR run downloads nothing
#
# Stage 1 (builder) installs the Python packages into /opt/venv with a BuildKit pip cache, so a
# rebuild after editing requirements.txt only downloads what changed. Stage 2 (runtime) has no
# compilers: system libraries + /opt/venv + the app, run as a non-root user.
# Layer order: system packages -> Python packages -> (optional) models -> app code, so editing the
# code rebuilds in seconds.

ARG PYTHON_VERSION=3.11

# ---------------------------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Compilers only for the rare package without a Linux wheel; they stay in this stage.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# CPU-only PyTorch first, from the PyTorch CPU index: ~200 MB of wheels instead of the multi-GB
# CUDA build that PyPI's torch pulls in. Same versions as the project's venv.
ARG TORCH_VERSION=2.14.0
ARG TORCHVISION_VERSION=0.29.0
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"

# Everything else; paddlepaddle, paddleocr[doc-parser,doc2md], paddlex, PyMuPDF and the OpenCV
# wheels are pinned in requirements.txt. torch/torchvision are already satisfied (CPU builds).
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt

# ---------------------------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

# libgl1 / libglib2.0-0: OpenCV; libgomp1: OpenMP for paddle and torch; tesseract-ocr: the classic
# OCR path and page-orientation correction; curl: the health check.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 tesseract-ocr curl \
    && rm -rf /var/lib/apt/lists/*

ARG APP_UID=1000
RUN useradd --create-home --uid "${APP_UID}" app \
    && mkdir -p /app/data /models \
    && chown -R app:app /app /models

COPY --from=builder /opt/venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Model caches live under /models (a named volume in docker-compose.yml): PaddleOCR (paddlex),
    # Hugging Face (TrOCR), docTR.
    PADDLE_PDX_CACHE_HOME=/models/paddlex \
    HF_HOME=/models/huggingface \
    DOCTR_CACHE_DIR=/models/doctr \
    # Skip paddlex's connectivity check of the model hosters on every start.
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    # oneDNN (MKL-DNN) is off: paddle 3.3's CPU oneDNN kernels fail with "ConvertPirAttribute2Runtime
    # Attribute ... Unimplemented" on Windows; set true in .env to try it on Linux (README: Docker).
    PADDLE_ENABLE_MKLDNN=false

WORKDIR /app
USER app

# Optional: bake the models into the image (the app's configured models; scripts/preload_models.py
# only needs app/config.py, so this layer survives code changes).
ARG PRELOAD_MODELS=false
COPY --chown=app:app app/__init__.py app/config.py /app/app/
COPY --chown=app:app scripts/preload_models.py /app/scripts/
RUN if [ "$PRELOAD_MODELS" = "true" ]; then python scripts/preload_models.py; \
    else echo "PRELOAD_MODELS=false: models download on first use (into the model-cache volume)"; fi

# Application code last: editing it only rebuilds from here.
COPY --chown=app:app app ./app
COPY --chown=app:app evaluation ./evaluation
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app docs ./docs

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
