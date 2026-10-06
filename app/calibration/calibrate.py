"""Supports Module 3 — fit and save the PaddleOCR confidence calibrator.

run_calibration() OCRs every image of a labelled dataset, labels each line
correct/incorrect against the ground truth (labeling.py), fits and selects a
calibrator (selection.py), and writes to out_dir:

- calibrator.json          the chosen calibrator (Calibrator.to_dict()), read by load_active()
- report.json              metrics before/after, bins, candidates (contract §1)
- reliability_diagram.png  before vs after reliability diagram

Dataset: a folder holding labels.csv, or the path of such a CSV. Columns
`image,text`; image paths are relative to the CSV; text is the ground-truth
lines joined by newlines.

Run:  venv/Scripts/python -m app.calibration.make_sample        (once; the sample is committed)
      venv/Scripts/python -m app.calibration.calibrate [DATASET] [--match exact|cer]
                                                       [--cer-threshold 0.1] [--out DIR] [--lang en]
"""

import argparse
import csv
import json
import os
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.calibration.active import CALIBRATOR_FILE, clear_active_cache
from app.calibration.diagram import draw_reliability_diagram
from app.calibration.labeling import MATCH_MODES, label_lines, split_truth
from app.calibration.selection import fit_and_select
from app.config import BASE_DIR, settings

REPORT_FILE = "report.json"
DIAGRAM_FILE = "reliability_diagram.png"
DEFAULT_DATASET = BASE_DIR / "evaluation" / "calibration_sample"
MIN_LINES = 10


def read_labels(dataset: Path) -> tuple[Path, list[tuple[Path, list[str]]]]:
    """(csv path, [(image path, ground-truth lines)]) from a dataset folder or labels CSV."""
    dataset = Path(dataset)
    csv_path = dataset / "labels.csv" if dataset.is_dir() else dataset
    if not csv_path.is_file():
        raise ValueError(f"no labels file found at {csv_path} (expected a folder with labels.csv, or a CSV path)")
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = {(name or "").strip().lower(): name for name in reader.fieldnames or []}
        if "image" not in fields or "text" not in fields:
            raise ValueError(f"{csv_path.name} must have the columns image,text (found: {', '.join(fields) or 'none'})")
        rows = []
        for n, row in enumerate(reader, start=2):
            name = (row.get(fields["image"]) or "").strip()
            if not name:
                continue
            image = (csv_path.parent / name).resolve()
            if not image.is_file():
                raise ValueError(f"{csv_path.name} line {n}: image not found: {name}")
            rows.append((image, split_truth(row.get(fields["text"]) or "")))
    if not rows:
        raise ValueError(f"{csv_path.name} lists no images")
    return csv_path, rows


def _display_path(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(BASE_DIR).as_posix()
    except ValueError:
        return str(path)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Write via a temporary file and rename, so a reader never sees half a file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def run_calibration(
    dataset: Path,
    ocr_fn: Callable[[Path], list[tuple[str, float]]],
    out_dir: Path | None = None,
    match: str = "cer",
    cer_threshold: float | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """OCR + label the dataset, fit/select a calibrator, save it with its report and diagram.

    Raises ValueError when the dataset is missing/malformed or the labelled
    lines cannot support a fit (fewer than 10, or all correct / all incorrect).
    Returns the report dict (also written to out_dir/report.json).
    """
    if match not in MATCH_MODES:
        raise ValueError(f"match must be one of {', '.join(MATCH_MODES)}, got {match!r}")
    threshold = settings.calibration_cer_threshold if cer_threshold is None else float(cer_threshold)
    if threshold < 0:
        raise ValueError("cer_threshold must be >= 0")
    out = Path(out_dir) if out_dir is not None else Path(settings.paddle_calibration_dir)

    csv_path, rows = read_labels(Path(dataset))
    confidences: list[float] = []
    labels: list[int] = []
    for done, (image, truth) in enumerate(rows, start=1):
        lines = [(str(text), min(max(float(conf), 0.0), 1.0)) for text, conf in ocr_fn(image)]
        for line in label_lines(lines, truth, match=match, cer_threshold=threshold):
            confidences.append(line.confidence)
            labels.append(line.correct)
        if progress is not None:
            progress(done, len(rows))

    n_correct = sum(labels)
    if len(labels) < MIN_LINES:
        raise ValueError(
            f"only {len(labels)} OCR lines were read from {len(rows)} images; at least {MIN_LINES} are needed"
        )
    if n_correct in (0, len(labels)):
        kind = "correct" if n_correct else "incorrect"
        raise ValueError(
            f"all {len(labels)} OCR lines were {kind}; calibration needs both correct and incorrect lines "
            "(add harder or easier images, or change the match mode / CER threshold)"
        )

    calibrator, fit_report = fit_and_select(confidences, labels)
    fitted_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    report: dict[str, Any] = {
        "method": fit_report["method"],
        "selected_by": fit_report["selected_by"],
        "fitted_at": fitted_at,
        "dataset": _display_path(csv_path.parent if csv_path.name == "labels.csv" else csv_path),
        "match": match,
        "cer_threshold": threshold,
        **{k: fit_report[k] for k in ("n_lines", "n_correct", "n_train", "n_val", "candidates", "metrics", "bins")},
        "diagram": DIAGRAM_FILE,
    }

    out.mkdir(parents=True, exist_ok=True)
    draw_reliability_diagram(
        report["bins"]["before"],
        report["bins"]["after"],
        report["metrics"]["before"]["ece"],
        report["metrics"]["after"]["ece"],
        report["method"],
        out / DIAGRAM_FILE,
    )
    _write_json(out / REPORT_FILE, report)
    _write_json(out / CALIBRATOR_FILE, {**calibrator.to_dict(), "fitted_at": fitted_at})
    clear_active_cache()
    return report


def format_summary(report: dict[str, Any]) -> str:
    """Short before/after table for the console."""
    before, after = report["metrics"]["before"], report["metrics"]["after"]
    rows = [
        f"Lines: {report['n_lines']} ({report['n_correct']} correct), train {report['n_train']} / val {report['n_val']}",
        "Candidates (validation ECE): "
        + ", ".join(f"{name} {c['val_ece']:.4f}" for name, c in report["candidates"].items()),
        f"Chosen: {report['method']} (by {report['selected_by']})",
        "",
        f"{'metric':<8}{'before':>10}{'after':>10}",
    ]
    rows += [f"{name:<8}{before[name]:>10.4f}{after[name]:>10.4f}" for name in ("ece", "mce", "brier", "nll")]
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit the PaddleOCR confidence calibrator on a labelled dataset.")
    parser.add_argument("dataset", nargs="?", default=str(DEFAULT_DATASET), help="folder with labels.csv, or a CSV")
    parser.add_argument("--match", choices=MATCH_MODES, default="cer")
    parser.add_argument("--cer-threshold", type=float, default=None, help="default: CALIBRATION_CER_THRESHOLD")
    parser.add_argument("--out", default=None, help="default: settings.paddle_calibration_dir")
    parser.add_argument("--lang", default="en", help="PaddleOCR language code")
    args = parser.parse_args(argv)

    from app.modules.paddleocr_service import paddleocr_service

    def ocr_fn(path: Path) -> list[tuple[str, float]]:
        return paddleocr_service.recognize_lines(path, args.lang)

    def progress(done: int, total: int) -> None:
        print(f"\r  OCR {done}/{total}", end="" if done < total else "\n", flush=True)

    try:
        report = run_calibration(
            Path(args.dataset),
            ocr_fn,
            out_dir=Path(args.out) if args.out else None,
            match=args.match,
            cer_threshold=args.cer_threshold,
            progress=progress,
        )
    except ValueError as exc:
        sys.exit(f"Calibration failed: {exc}")
    print(format_summary(report))
    print(f"Saved to {Path(args.out) if args.out else settings.paddle_calibration_dir}")


if __name__ == "__main__":
    main()
