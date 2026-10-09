# DocBlendAI in one container: FastAPI app + Tesseract + TrOCR/docTR models (CPU only).
#
#   docker compose up --build        then open http://localhost:8000/
#
# The Gemini key comes from .env at run time (see docker-compose.yml); it is never copied into the image.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/models/huggingface \
    DOCTR_CACHE_DIR=/opt/models/doctr

# Tesseract (printed-text OCR + page orientation); libGL/glib for OpenCV, which docTR pulls in.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU builds of torch/torchvision (the default wheels bundle ~4 GB of CUDA libraries a laptop does not use).
COPY constraints.txt requirements.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 torchvision==0.29.0 \
    && pip install -r requirements.txt -c constraints.txt

# Download the handwriting model and the docTR text detectors into the image, so the first
# scanned or handwritten upload works without waiting for (or needing) a download.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/trocr-small-handwritten')" \
    && python -c "from doctr.models import ocr_predictor, detection_predictor; detection_predictor('db_resnet50', pretrained=True); ocr_predictor('db_resnet50', 'crnn_vgg16_bn', pretrained=True)"

COPY app ./app
COPY evaluation ./evaluation
COPY data/calibration.json /opt/defaults/calibration.json

# Uploads, the SQLite database and ChromaDB live in /app/data (a volume, so they survive restarts);
# the fitted confidence calibration ships with the image.
ENV CALIBRATION_FILE=/opt/defaults/calibration.json
# The models above are already in the image: load them from there, without contacting the HuggingFace Hub
ENV HF_HUB_OFFLINE=1
RUN mkdir -p /app/data/uploads /app/data/chroma_db

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
