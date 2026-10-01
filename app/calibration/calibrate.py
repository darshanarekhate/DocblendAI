"""STUB for run_calibration (see docs/experience_center_contract.md §1)."""

from collections.abc import Callable
from pathlib import Path
from typing import Any


def run_calibration(
    dataset: Path,
    ocr_fn: Callable[[Path], list[tuple[str, float]]],
    out_dir: Path | None = None,
    match: str = "cer",
    cer_threshold: float | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    raise NotImplementedError("calibration is not implemented yet")
