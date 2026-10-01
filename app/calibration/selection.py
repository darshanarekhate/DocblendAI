"""Supports Module 3 — choosing the calibration method for PaddleOCR line confidences.

fit_and_select() splits the labelled lines into train and validation sets
(stratified by label, seeded, so the same data always gives the same split),
fits every calibrator on train, and keeps the one with the lowest validation
ECE (ties broken by Brier score, then by simplicity: temperature, platt,
isotonic). The report's metrics and reliability bins are measured on the
validation set, before (raw confidences) and after (the chosen calibrator).
The calibrator that is returned is then refitted on all the data, so none
of it is wasted once the method is chosen.
"""

from typing import Any

import numpy as np

from app.calibration.calibrators import CALIBRATORS, Calibrator
from app.calibration.metrics import all_metrics, brier, ece, reliability_bins

TIE_TOLERANCE = 1e-9


def stratified_split(labels: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(train indices, validation indices), each class split in the same proportion.

    A class with at least two members always puts one in each set, so the
    validation set can measure both kinds of line whenever possible.
    """
    rng = np.random.default_rng(seed)
    train, val = [], []
    for cls in (0.0, 1.0):
        idx = np.flatnonzero(labels == cls)
        if idx.size == 0:
            continue
        idx = rng.permutation(idx)
        n_val = int(round(idx.size * val_fraction))
        if idx.size >= 2:
            n_val = min(max(n_val, 1), idx.size - 1)
        val.extend(idx[:n_val].tolist())
        train.extend(idx[n_val:].tolist())
    return np.array(sorted(train), dtype=int), np.array(sorted(val), dtype=int)


def _rounded(metrics: dict[str, float]) -> dict[str, float]:
    return {k: round(v, 6) for k, v in metrics.items()}


def _rounded_bins(bins: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {k: (round(v, 6) if isinstance(v, float) and k not in ("lo", "hi") else v) for k, v in b.items()}
        for b in bins
    ]


def fit_and_select(
    confidences: Any, labels: Any, val_fraction: float = 0.3, seed: int = 0
) -> tuple[Calibrator, dict[str, Any]]:
    """Fit temperature, Platt and isotonic calibrators, pick one by validation ECE.

    Returns (calibrator refitted on all data, report). The report holds
    method, selected_by, n_lines, n_correct, n_train, n_val, candidates
    ({method: {val_ece, val_brier}}), metrics.before/after and bins.before/after
    (docs/experience_center_contract.md §1).
    """
    p = np.clip(np.asarray(confidences, dtype=float).ravel(), 0.0, 1.0)
    y = np.asarray(labels, dtype=float).ravel()
    if p.size != y.size:
        raise ValueError(f"{p.size} confidences but {y.size} labels")
    if p.size < 2:
        raise ValueError("need at least 2 labelled lines to fit and validate a calibrator")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    train, val = stratified_split(y, val_fraction, seed)
    if train.size == 0 or val.size == 0:  # only possible with 2-3 single-class lines
        train, val = np.arange(p.size - 1), np.array([p.size - 1])

    candidates: dict[str, dict[str, float]] = {}
    scores = []
    for order, (name, cls) in enumerate(CALIBRATORS.items()):
        model = cls().fit(p[train], y[train])
        val_pred = model.predict(p[val])
        val_ece, val_brier = ece(val_pred, y[val]), brier(val_pred, y[val])
        candidates[name] = {"val_ece": round(val_ece, 6), "val_brier": round(val_brier, 6)}
        scores.append((val_ece, val_brier, order, name, val_pred))

    best = scores[0]
    for s in scores[1:]:
        if s[0] < best[0] - TIE_TOLERANCE or (abs(s[0] - best[0]) <= TIE_TOLERANCE and s[1] < best[1] - TIE_TOLERANCE):
            best = s
    method, val_pred = best[3], best[4]

    final = CALIBRATORS[method]().fit(p, y)
    report = {
        "method": method,
        "selected_by": "validation ECE",
        "n_lines": int(p.size),
        "n_correct": int(y.sum()),
        "n_train": int(train.size),
        "n_val": int(val.size),
        "candidates": candidates,
        "metrics": {
            "before": _rounded(all_metrics(p[val], y[val])),
            "after": _rounded(all_metrics(val_pred, y[val])),
        },
        "bins": {
            "before": _rounded_bins(reliability_bins(p[val], y[val])),
            "after": _rounded_bins(reliability_bins(val_pred, y[val])),
        },
    }
    return final, report
