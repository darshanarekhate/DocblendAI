"""Shared test fixtures: an in-memory PDF builder, fake embeddings, and an isolated API client."""

import re
import textwrap
import zlib
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.db import models  # noqa: F401  (registers tables on Base.metadata)
from app.db.database import Base, get_db
from app.main import app
from app.modules import confidence_capture, embedder, htr_extractor, line_segmentation, ocr_extractor, vector_store


def make_pdf(pages: list[str]) -> bytes:
    """Build a minimal typed PDF (Helvetica text layer), one string per page.

    An empty string gives a page with no text, i.e. what a scan looks like
    to a text-layer parser.
    """
    n = len(pages)
    page_ids = [4 + 2 * i for i in range(n)]
    kids = " ".join(f"{p} 0 R" for p in page_ids)
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for page_id, text in zip(page_ids, pages):
        lines = [w for line in text.splitlines() for w in (textwrap.wrap(line, 80) or [""])]
        escaped = [line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for line in lines]
        stream = ("BT /F1 11 Tf 14 TL 72 740 Td " + " ".join(f"({line}) Tj T*" for line in escaped) + " ET").encode("latin-1")
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_id + 1} 0 R >>".encode()
        )
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")

    out = b"%PDF-1.4\n"
    offsets = []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{off:010d} 00000 n \n".encode() for off in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


TYPED_PAGE = (
    "Retrieval-Augmented Generation combines a retriever with a language model.\n"
    "The retriever selects relevant passages and the model answers from them."
)


@pytest.fixture
def pdf_file(tmp_path: Path):
    """Write a PDF with the given pages to tmp_path and return its path."""

    def _write(pages: list[str], name: str = "doc.pdf") -> Path:
        path = tmp_path / name
        path.write_bytes(make_pdf(pages))
        return path

    return _write


@pytest.fixture
def db_session_factory(tmp_path: Path) -> sessionmaker[Session]:
    engine = create_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


FAKE_DIM = 64


def fake_embed(texts: list[str], task_type: str) -> list[list[float]]:
    """Offline stand-in for Gemini: hashed bag-of-words, so shared words mean higher similarity."""
    vectors = []
    for text in texts:
        v = [0.0] * FAKE_DIM
        for word in re.findall(r"[a-z]+", text.lower()):
            v[zlib.crc32(word.encode()) % FAKE_DIM] += 1.0
        vectors.append(v if any(v) else [1.0] + [0.0] * (FAKE_DIM - 1))
    return vectors


@pytest.fixture
def fake_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route all embedding calls to fake_embed (no network, no API key needed)."""
    monkeypatch.setattr(embedder, "_embed", fake_embed)


@pytest.fixture(autouse=True)
def no_real_htr_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must never load TrOCR (slow, downloads weights): fake htr_image instead."""

    def _refuse(model_name):
        pytest.fail("a test tried to load the real TrOCR model; use the fake_htr fixture")

    monkeypatch.setattr(htr_extractor, "_load", _refuse)


@pytest.fixture(autouse=True)
def no_real_paddle_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Tests must never construct a real PaddleOCR model or import paddle (slow, GBs of RAM).

    Fake models by monkeypatching paddleocr_service._construct_model (see tests/test_paddleocr_*.py).
    Also keeps the Experience Center off data/: page images and calibration files go to tmp_path,
    and the job manager's history goes to a throwaway SQLite file (created on first use; tests
    can call job_manager.configure(db_session_factory) to share the client's database).
    """
    from app.modules import ocr_jobs, paddleocr_service

    def _refuse_model(pipeline, lang):
        pytest.fail("a test tried to construct a real Paddle model; fake _construct_model")

    def _refuse_import():
        pytest.fail("a test tried to import paddleocr; fake the engine")

    monkeypatch.setattr(paddleocr_service, "_construct_model", _refuse_model)
    monkeypatch.setattr(paddleocr_service, "_import_paddleocr", _refuse_import)
    monkeypatch.setattr(settings, "paddle_runs_dir", tmp_path / "paddle_runs")
    monkeypatch.setattr(settings, "paddle_calibration_dir", tmp_path / "paddle_calibration")

    engines = []

    def _lazy_session() -> Session:
        if not engines:
            engine = create_engine(
                f"sqlite:///{(tmp_path / 'paddle_runs.db').as_posix()}", connect_args={"check_same_thread": False}
            )
            Base.metadata.create_all(bind=engine)
            engines.append(engine)
        return sessionmaker(bind=engines[0], autoflush=False, autocommit=False)()

    ocr_jobs.job_manager.configure(_lazy_session)
    paddleocr_service.paddleocr_service.unload()
    yield
    ocr_jobs.job_manager.reset()
    ocr_jobs.job_manager.configure(None)
    paddleocr_service.paddleocr_service.unload()
    for engine in engines:
        engine.dispose()


@pytest.fixture(autouse=True)
def offline_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the engine settings tests assume, and refuse to load docTR models (slow, downloads weights).

    Tests use Tesseract's code path (faked), the projection line splitter, and
    no fallback answer model; test_open_source_integrations covers the alternatives with fakes.
    """
    monkeypatch.setattr(settings, "ocr_engine", "tesseract")
    monkeypatch.setattr(settings, "htr_segmenter", "projection")
    monkeypatch.setattr(settings, "htr_temperature", 1.0)
    monkeypatch.setattr(settings, "llm_fallback_model", "")

    def _refuse():
        pytest.fail("a test tried to load a real docTR model; fake it")

    monkeypatch.setattr(line_segmentation, "_detector", _refuse)
    monkeypatch.setattr(ocr_extractor, "_doctr_predictor", _refuse)


@pytest.fixture(autouse=True)
def no_orientation_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip Tesseract OSD in tests (keeps them offline); test_file_formats tests the real logic."""
    monkeypatch.setattr(ocr_extractor, "auto_orient", lambda page: page)


@pytest.fixture(autouse=True)
def isolated_calibration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Never read or write the real data/calibration.json."""
    monkeypatch.setattr(settings, "calibration_file", tmp_path / "calibration.json")
    confidence_capture._tables.cache_clear()
    yield
    confidence_capture._tables.cache_clear()


@pytest.fixture
def fake_ocr(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Replace Tesseract: every page image OCRs to (fake.text, fake.conf)."""
    fake = SimpleNamespace(text="", conf=0.0, calls=0)

    def _ocr_image(image):
        fake.calls += 1
        return fake.text, fake.conf

    monkeypatch.setattr(ocr_extractor, "ocr_image", _ocr_image)
    return fake


@pytest.fixture
def fake_htr(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Replace TrOCR: every page image reads as (fake.text, fake.conf)."""
    fake = SimpleNamespace(text="", conf=0.0)
    monkeypatch.setattr(htr_extractor, "htr_image", lambda image: (fake.text, fake.conf))
    return fake


@pytest.fixture
def chroma(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point vector_store at a throwaway ChromaDB directory."""
    monkeypatch.setattr(settings, "chroma_dir", tmp_path / "chroma")
    yield
    vector_store._collection.cache_clear()


@pytest.fixture
def client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, db_session_factory, fake_embeddings, chroma
) -> Iterator[TestClient]:
    """API client using a throwaway SQLite file, upload dir, and ChromaDB (never touches data/)."""
    monkeypatch.setattr(settings, "upload_dir", tmp_path / "uploads")

    def _get_test_db() -> Iterator[Session]:
        db = db_session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _get_test_db
    # Not used as a context manager, so the app lifespan (which creates the real DB) does not run.
    yield TestClient(app)
    app.dependency_overrides.clear()
