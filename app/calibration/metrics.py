"""Supports Module 3 — calibration metrics for line confidences.

Inputs: confidences p_i in [0, 1] and labels y_i (1 = line read correctly).
Binning uses n_bins equal-width bins over [0, 1]; bin k holds the p with
k / n_bins <= p < (k + 1) / n_bins, and p = 1.0 falls in the last bin.

- ECE (expected calibration error) = sum_k (n_k / n) * |acc_k - conf_k|, where
  acc_k is the share of correct lines in bin k and conf_k their mean confidence.
- MCE (maximum calibration error) = max_k |acc_k - conf_k| over non-empty bins.
- Brier score = mean((p_i - y_i)^2).
- NLL (negative log-likelihood, log-loss) = -mean(y_i log p_i + (1 - y_i) log(1 - p_i)),
  with p clipped to [EPS, 1 - EPS] (EPS = 1e-6) so a confident miss costs ~13.8, not infinity.

All of them are 0 for a perfect predictor; lower is better. ECE/MCE measure
calibration only, Brier/NLL also reward sharp (discriminating) confidences.
"""

from typing import Any

import numpy as np

from app.calibration.calibrators import EPS

N_BINS = 10


def _arrays(confidences: Any, labels: Any) -> tuple[np.ndarray, np.ndarray]:
    p = np.clip(np.atleast_1d(np.asarray(confidences, dtype=float)), 0.0, 1.0)
    y = np.atleast_1d(np.asarray(labels, dtype=float))
    if p.shape != y.shape:
        raise ValueError(f"{p.size} confidences but {y.size} labels")
    return p, y


def _bin_index(p: np.ndarray, n_bins: int) -> np.ndarray:
    return np.minimum((p * n_bins).astype(int), n_bins - 1)


def reliability_bins(confidences: Any, labels: Any, n_bins: int = N_BINS) -> list[dict[str, Any]]:
    """Per-bin {lo, hi, count, mean_confidence, accuracy}; the last two are None for empty bins."""
    p, y = _arrays(confidences, labels)
    idx = _bin_index(p, n_bins)
    bins = []
    for k in range(n_bins):
        mask = idx == k
        count = int(mask.sum())
        bins.append(
            {
                "lo": round(k / n_bins, 10),
                "hi": round((k + 1) / n_bins, 10),
                "count": count,
                "mean_confidence": float(p[mask].mean()) if count else None,
                "accuracy": float(y[mask].mean()) if count else None,
            }
        )
    return bins


def _gaps(confidences: Any, labels: Any, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    bins = [b for b in reliability_bins(confidences, labels, n_bins) if b["count"]]
    counts = np.array([b["count"] for b in bins], dtype=float)
    gaps = np.array([abs(b["accuracy"] - b["mean_confidence"]) for b in bins])
    return counts, gaps


def ece(confidences: Any, labels: Any, n_bins: int = N_BINS) -> float:
    """Expected calibration error (count-weighted mean |accuracy - confidence| over bins)."""
    counts, gaps = _gaps(confidences, labels, n_bins)
    return float(np.sum(counts * gaps) / counts.sum()) if counts.size else 0.0


def mce(confidences: Any, labels: Any, n_bins: int = N_BINS) -> float:
    """Maximum calibration error (worst |accuracy - confidence| over non-empty bins)."""
    _, gaps = _gaps(confidences, labels, n_bins)
    return float(gaps.max()) if gaps.size else 0.0


def brier(confidences: Any, labels: Any) -> float:
    """Brier score: mean squared difference between confidence and outcome."""
    p, y = _arrays(confidences, labels)
    return float(np.mean((p - y) ** 2)) if p.size else 0.0


def nll(confidences: Any, labels: Any) -> float:
    """Negative log-likelihood (mean log-loss), confidences clipped to [EPS, 1 - EPS]."""
    p, y = _arrays(confidences, labels)
    if not p.size:
        return 0.0
    p = np.clip(p, EPS, 1.0 - EPS)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def all_metrics(confidences: Any, labels: Any, n_bins: int = N_BINS) -> dict[str, float]:
    """{"ece", "mce", "brier", "nll"} for one set of confidences."""
    return {
        "ece": ece(confidences, labels, n_bins),
        "mce": mce(confidences, labels, n_bins),
        "brier": brier(confidences, labels),
        "nll": nll(confidences, labels),
    }
