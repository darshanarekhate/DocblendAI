"""Lazy PaddleOCR 3.x integration for the OCR experience center."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import settings


class PaddleOCRUnavailableError(RuntimeError):
    """Raised when PaddleOCR is not installed or cannot load its models."""


@dataclass(frozen=True)
class OCRLine:
    text: str
    polygon: list[list[float]]
    raw_confidence: float


@dataclass(frozen=True)
class OCRPage:
    page_index: int
    source: str
    lines: list[OCRLine]


class PaddleOCRService:
    """Singleton-style, lazy-loaded PaddleOCR service.

    Model construction and inference are protected because PaddleOCR pipelines
    are expensive and are not guaranteed to be safe for concurrent access.
    """

    _instance: PaddleOCRService | None = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> PaddleOCRService:
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_state()
            return cls._instance

    def _init_state(self) -> None:
        self._model_lock = threading.RLock()
        self._ocr_models: dict[str, Any] = {}
        self._parse_model: Any | None = None
        self._vl_model: Any | None = None
        self._vl_error: str | None = None

    def _make_ocr(self, lang: str) -> Any:
        try:
            from paddleocr import PaddleOCR
        except Exception as exc:  # pragma: no cover - depends on local install
            raise PaddleOCRUnavailableError(f"PaddleOCR is unavailable: {exc}") from exc

        options = {
            "lang": lang,
            "device": settings.paddle_device,
            "use_doc_orientation_classify": settings.paddle_use_orientation,
            "use_doc_unwarping": settings.paddle_use_unwarping,
            "use_textline_orientation": settings.paddle_use_textline_orientation,
        }
        try:
            return PaddleOCR(**options)
        except (TypeError, ValueError) as exc:
            if "ocr_version" not in str(exc).lower():
                raise PaddleOCRUnavailableError(
                    f"Unable to initialize PaddleOCR: {exc}"
                ) from exc
            options["ocr_version"] = "PP-OCRv5"
            try:
                return PaddleOCR(**options)
            except Exception as fallback_exc:
                raise PaddleOCRUnavailableError(
                    f"PP-OCRv6 and PP-OCRv5 initialization failed: {fallback_exc}"
                ) from fallback_exc
        except Exception as exc:
            raise PaddleOCRUnavailableError(
                f"Unable to initialize PaddleOCR: {exc}"
            ) from exc

    def _ocr(self, lang: str) -> Any:
        with self._model_lock:
            if lang not in self._ocr_models:
                self._ocr_models[lang] = self._make_ocr(lang)
            return self._ocr_models[lang]

    @staticmethod
    def _value(result: Any, name: str, default: Any = None) -> Any:
        if isinstance(result, dict):
            return result.get(name, default)
        try:
            return result[name]
        except (KeyError, IndexError, TypeError):
            return getattr(result, name, default)

    @classmethod
    def _normalise_result(cls, result: Any) -> dict[str, Any]:
        payload = cls._value(result, "res", result)
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = {}
        if not isinstance(payload, dict):
            payload = {}
        return payload

    @classmethod
    def _lines_from_result(cls, result: Any) -> list[OCRLine]:
        payload = cls._normalise_result(result)
        texts = cls._value(payload, "rec_texts", []) or []
        scores = cls._value(payload, "rec_scores", []) or []
        polygons = (
            cls._value(payload, "dt_polys", [])
            or cls._value(payload, "rec_polys", [])
            or []
        )
        lines: list[OCRLine] = []
        for index, text in enumerate(texts):
            polygon = polygons[index] if index < len(polygons) else []
            score = scores[index] if index < len(scores) else 0.0
            lines.append(
                OCRLine(
                    text=str(text),
                    polygon=[[float(point[0]), float(point[1])] for point in polygon],
                    raw_confidence=max(0.0, min(1.0, float(score))),
                )
            )
        return lines

    @staticmethod
    def _input_pages(path: Path) -> Iterator[tuple[int, str]]:
        if path.suffix.lower() != ".pdf":
            yield 0, str(path)
            return
        try:
            import fitz
        except Exception as exc:  # pragma: no cover - dependency is pinned
            raise PaddleOCRUnavailableError(f"PyMuPDF is unavailable: {exc}") from exc
        with fitz.open(path) as document:
            for page_index, page in enumerate(document):
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                image_path = (
                    settings.upload_dir
                    / f"paddle-page-{threading.get_ident()}-{page_index}.png"
                )
                pixmap.save(image_path)
                yield page_index, str(image_path)
                image_path.unlink(missing_ok=True)

    def ocr_file(self, path: str | Path, lang: str = "en") -> list[OCRPage]:
        model = self._ocr(lang)
        pages: list[OCRPage] = []
        with self._model_lock:
            for page_index, source in self._input_pages(Path(path)):
                try:
                    results = model.predict(source)
                    result = next(iter(results), None)
                    pages.append(
                        OCRPage(
                            page_index,
                            source,
                            self._lines_from_result(result) if result else [],
                        )
                    )
                except Exception as exc:
                    raise PaddleOCRUnavailableError(
                        f"OCR failed on page {page_index + 1}: {exc}"
                    ) from exc
        return pages

    def parse_file(self, path: str | Path, lang: str = "en") -> dict[str, Any]:
        with self._model_lock:
            if self._parse_model is None:
                try:
                    from paddleocr import PPStructureV3

                    self._parse_model = PPStructureV3(
                        lang=lang,
                        device=settings.paddle_device,
                        use_doc_orientation_classify=settings.paddle_use_orientation,
                        use_doc_unwarping=settings.paddle_use_unwarping,
                    )
                except (
                    Exception
                ) as exc:  # pragma: no cover - model availability is environment-dependent
                    raise PaddleOCRUnavailableError(
                        f"PP-StructureV3 is unavailable: {exc}"
                    ) from exc
            try:
                results = self._parse_model.predict(str(path))
                serialised = []
                for result in results:
                    payload = self._normalise_result(result)
                    serialised.append(payload)
                return {"pages": serialised}
            except Exception as exc:
                raise PaddleOCRUnavailableError(
                    f"Document parsing failed: {exc}"
                ) from exc

    def vl_status(self) -> dict[str, Any]:
        return {
            "available": self._vl_model is not None,
            "message": self._vl_error
            or "PaddleOCR-VL is optional and has not been loaded.",
        }


paddleocr_service = PaddleOCRService()
