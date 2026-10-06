"""Fake PaddleOCR models built from real recorded results (tests/fixtures/*.json)."""

import copy
import json
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).parent / "fixtures"
# Real `result.json["res"]` of PaddleOCR (PP-OCRv5 mobile, return_word_box=True) and of
# PP-StructureV3 on tests/fixtures/ppstructurev3_page.png (1000 x 700 px), recorded on a dev laptop.
TEXT_RES: dict[str, Any] = json.loads((FIXTURES / "paddleocr_text_result.json").read_text(encoding="utf-8"))
STRUCTURE_RES: dict[str, Any] = json.loads((FIXTURES / "ppstructurev3_result.json").read_text(encoding="utf-8"))
STRUCTURE_MARKDOWN = (
    "## Calibration Report\n\nThis page compares raw and calibrated OcR confidence on the sample set."
    "Lower expected calibration error means better calibrated scores.\n\n"
    + STRUCTURE_RES["parsing_res_list"][2]["block_content"]
)


class FakeResult:
    """Mimics a PaddleX result: `.json["res"]` and `.markdown["markdown_texts"]`."""

    def __init__(self, res: dict[str, Any], markdown: str | None = None) -> None:
        self.json = {"res": copy.deepcopy(res)}
        self.markdown = {"markdown_texts": markdown} if markdown is not None else {}


class FakeModel:
    def __init__(self, res: dict[str, Any], markdown: str | None = None) -> None:
        self.res = res
        self.markdown = markdown
        self.calls: list[str] = []

    def predict(self, path: str):
        self.calls.append(path)
        yield FakeResult(self.res, self.markdown)


class FakeFactory:
    """Stand-in for paddleocr_service._construct_model: counts constructions per (pipeline, lang)."""

    def __init__(self) -> None:
        self.built: list[tuple[str, str]] = []
        self.models: dict[tuple[str, str], FakeModel] = {}
        self.fail: Exception | None = None

    def __call__(self, pipeline: str, lang: str) -> FakeModel:
        if self.fail is not None:
            raise self.fail
        self.built.append((pipeline, lang))
        model = FakeModel(TEXT_RES) if pipeline == "ocr" else FakeModel(STRUCTURE_RES, STRUCTURE_MARKDOWN)
        self.models[(pipeline, lang)] = model
        return model
