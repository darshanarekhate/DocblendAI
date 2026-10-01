"""Supports Module 3 — confidence calibration for the PaddleOCR Experience Center.

Maps PaddleOCR's raw line confidence to the probability that the line was
read correctly, using a calibrator fitted on labelled sample images
(`python -m app.calibration.calibrate`). Interface: docs/experience_center_contract.md §1.

- calibrators.py  TemperatureScaler, PlattScaler, IsotonicCalibrator (pure numpy)
- metrics.py      ECE, MCE, Brier, NLL, reliability bins
- selection.py    fit_and_select: pick the method with the lowest validation ECE
- labeling.py     OCR line vs ground truth -> correct / incorrect
- active.py       the saved calibrator applied to new results
- calibrate.py    run_calibration + CLI; diagram.py draws the reliability diagram
- make_sample.py  generates the bundled sample dataset (evaluation/calibration_sample/)
"""

from app.calibration.active import active_method, calibrate_confidences, clear_active_cache, load_active
from app.calibration.calibrators import Calibrator, IsotonicCalibrator, PlattScaler, TemperatureScaler
from app.calibration.selection import fit_and_select

__all__ = [
    "Calibrator",
    "TemperatureScaler",
    "PlattScaler",
    "IsotonicCalibrator",
    "fit_and_select",
    "load_active",
    "clear_active_cache",
    "calibrate_confidences",
    "active_method",
]
