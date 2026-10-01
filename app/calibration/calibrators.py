"""Supports Module 3 — post-hoc confidence calibrators for PaddleOCR line confidences.

A calibrator maps a raw recognition confidence p in [0, 1] to the probability
that the line was read correctly. Three standard methods, all pure numpy:

- TemperatureScaler: p' = sigmoid(logit(p) / T). One parameter; keeps the
  ranking of lines and only sharpens (T < 1) or softens (T > 1) them.
- PlattScaler: p' = sigmoid(a * logit(p) + b) (Platt, 1999). Two parameters;
  can also shift every confidence up or down.
- IsotonicCalibrator: the best monotone non-decreasing step function
  (pool adjacent violators), interpolated between steps. Most flexible;
  needs the most data.

Confidences are clipped to [EPS, 1 - EPS] before taking the logit, so raw
scores of exactly 0 or 1 are safe. Every calibrator accepts degenerate data
(one class, one distinct confidence, a handful of points) without failing.

This package is independent of confidence_capture.py (TrOCR temperature and
per-format knot tables); it calibrates only the Experience Center's PaddleOCR lines.
"""

from typing import Any, ClassVar

import numpy as np

EPS = 1e-6


def _as_array(confidences: Any) -> np.ndarray:
    return np.clip(np.atleast_1d(np.asarray(confidences, dtype=float)), 0.0, 1.0)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))  # overflow-free form of 1 / (1 + exp(-z))


def _check(confidences: Any, labels: Any) -> tuple[np.ndarray, np.ndarray]:
    p = _as_array(confidences)
    y = np.atleast_1d(np.asarray(labels, dtype=float))
    if p.shape != y.shape:
        raise ValueError(f"{p.size} confidences but {y.size} labels")
    if p.size == 0:
        raise ValueError("cannot fit a calibrator on zero samples")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("labels must be 0 (incorrect) or 1 (correct)")
    return p, y


class Calibrator:
    """Base class: fit(conf, labels) -> self, predict(conf) -> ndarray, to_dict / from_dict."""

    method: ClassVar[str] = "base"

    def fit(self, confidences: Any, labels: Any) -> "Calibrator":
        raise NotImplementedError

    def predict(self, confidences: Any) -> np.ndarray:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def _from_params(cls, data: dict[str, Any]) -> "Calibrator":
        raise NotImplementedError

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Calibrator":
        """Rebuild a calibrator saved with to_dict(); dispatches on data["method"]."""
        method = data.get("method") if isinstance(data, dict) else None
        cls = CALIBRATORS.get(method)  # type: ignore[arg-type]
        if cls is None:
            raise ValueError(f"unknown calibration method: {method!r}")
        return cls._from_params(data)

    def __repr__(self) -> str:
        params = {k: v for k, v in self.to_dict().items() if k != "method"}
        return f"{type(self).__name__}({params})"


class TemperatureScaler(Calibrator):
    """p' = sigmoid(logit(p) / T), T chosen to minimise the negative log-likelihood.

    The search runs on log T in [log 0.05, log 20] (a coarse grid, then golden-section
    refinement), so it is bounded even when the data has one class and the
    likelihood keeps improving towards T -> 0.
    """

    method = "temperature"
    LOG_T_BOUNDS = (float(np.log(0.05)), float(np.log(20.0)))

    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    @staticmethod
    def _nll(log_t: float, z: np.ndarray, y: np.ndarray) -> float:
        q = np.clip(_sigmoid(z / np.exp(log_t)), EPS, 1.0 - EPS)
        return float(-np.mean(y * np.log(q) + (1.0 - y) * np.log1p(-q)))

    def fit(self, confidences: Any, labels: Any) -> "TemperatureScaler":
        p, y = _check(confidences, labels)
        z = _logit(p)
        lo, hi = self.LOG_T_BOUNDS
        grid = np.linspace(lo, hi, 61)
        losses = [self._nll(t, z, y) for t in grid]
        best = int(np.argmin(losses))
        a, b = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]
        ratio = (np.sqrt(5.0) - 1.0) / 2.0
        c, d = b - ratio * (b - a), a + ratio * (b - a)
        fc, fd = self._nll(c, z, y), self._nll(d, z, y)
        for _ in range(60):
            if fc <= fd:
                b, d, fd = d, c, fc
                c = b - ratio * (b - a)
                fc = self._nll(c, z, y)
            else:
                a, c, fc = c, d, fd
                d = a + ratio * (b - a)
                fd = self._nll(d, z, y)
        log_t = (a + b) / 2.0
        # Keep the grid point if refinement did not beat it (flat losses, e.g. all p = 0.5).
        if self._nll(log_t, z, y) > losses[best]:
            log_t = grid[best]
        self.temperature = float(np.exp(log_t))
        return self

    def predict(self, confidences: Any) -> np.ndarray:
        return _sigmoid(_logit(_as_array(confidences)) / self.temperature)

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "temperature": self.temperature}

    @classmethod
    def _from_params(cls, data: dict[str, Any]) -> "TemperatureScaler":
        return cls(float(data["temperature"]))


class PlattScaler(Calibrator):
    """p' = sigmoid(a * logit(p) + b), fitted by Newton's method (IRLS) on the log-loss.

    Uses Platt's smoothed targets, (N+ + 1) / (N+ + 2) for correct lines and
    1 / (N- + 2) for incorrect ones, so the optimum stays finite even for
    perfectly separable or single-class data. A tiny ridge on the Hessian
    handles a constant input (all confidences equal).
    """

    method = "platt"

    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a = float(a)
        self.b = float(b)

    def fit(self, confidences: Any, labels: Any) -> "PlattScaler":
        p, y = _check(confidences, labels)
        x = _logit(p)
        n_pos, n_neg = float(y.sum()), float(y.size - y.sum())
        t = np.where(y == 1, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))
        X = np.column_stack([x, np.ones_like(x)])

        def loss(w: np.ndarray) -> float:
            q = np.clip(_sigmoid(X @ w), 1e-12, 1.0 - 1e-12)
            return float(-np.sum(t * np.log(q) + (1.0 - t) * np.log1p(-q)))

        w = np.array([0.0, float(np.log((n_pos + 1.0) / (n_neg + 1.0)))])  # Platt's starting point
        current = loss(w)
        for _ in range(100):
            q = _sigmoid(X @ w)
            grad = X.T @ (q - t)
            hess = X.T @ (X * (q * (1.0 - q))[:, None]) + 1e-8 * np.eye(2)
            step = np.linalg.solve(hess, grad)
            scale = 1.0
            while scale > 1e-10:
                candidate = w - scale * step
                new = loss(candidate)
                if new <= current:
                    break
                scale /= 2.0
            else:
                break
            converged = abs(current - new) < 1e-12 * max(1.0, abs(current))
            w, current = candidate, new
            if converged or np.max(np.abs(scale * step)) < 1e-10:
                break
        self.a, self.b = float(w[0]), float(w[1])
        return self

    def predict(self, confidences: Any) -> np.ndarray:
        return _sigmoid(self.a * _logit(_as_array(confidences)) + self.b)

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "a": self.a, "b": self.b}

    @classmethod
    def _from_params(cls, data: dict[str, Any]) -> "PlattScaler":
        return cls(float(data["a"]), float(data["b"]))


class IsotonicCalibrator(Calibrator):
    """Monotone non-decreasing map fitted by pool adjacent violators (PAV).

    Equal confidences are first merged into one weighted point; PAV then
    pools neighbouring points while their accuracy does not increase. Each pooled
    block is kept as two knots (its lowest and highest confidence, same
    value), and predict() interpolates linearly between knots and holds the
    end values outside them. Output is clipped to [EPS, 1 - EPS] so a step
    of accuracy 0 or 1 never claims certainty.
    """

    method = "isotonic"

    def __init__(self, x: list[float] | None = None, y: list[float] | None = None):
        self.x = [0.0, 1.0] if x is None else [float(v) for v in x]
        self.y = [0.0, 1.0] if y is None else [float(v) for v in y]
        if len(self.x) != len(self.y) or not self.x:
            raise ValueError("isotonic knots need equal, non-zero numbers of x and y values")

    def fit(self, confidences: Any, labels: Any) -> "IsotonicCalibrator":
        p, y = _check(confidences, labels)
        xs, inverse = np.unique(p, return_inverse=True)
        weights = np.bincount(inverse).astype(float)
        means = np.bincount(inverse, weights=y) / weights

        # Blocks: [value, weight, first index into xs, last index into xs].
        blocks: list[list[float]] = []
        for i, (v, w) in enumerate(zip(means, weights)):
            blocks.append([float(v), float(w), i, i])
            # >= also pools equal neighbours: same fit, fewer knots.
            while len(blocks) > 1 and blocks[-2][0] >= blocks[-1][0]:
                v2, w2, _, last = blocks.pop()
                v1, w1, first, _ = blocks.pop()
                blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, first, last])

        knot_x: list[float] = []
        knot_y: list[float] = []
        for value, _, first, last in blocks:
            for idx in sorted({int(first), int(last)}):
                knot_x.append(float(xs[idx]))
                knot_y.append(float(value))
        self.x, self.y = knot_x, knot_y
        return self

    def predict(self, confidences: Any) -> np.ndarray:
        values = np.interp(_as_array(confidences), self.x, self.y)
        return np.clip(values, EPS, 1.0 - EPS)

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "x": list(self.x), "y": list(self.y)}

    @classmethod
    def _from_params(cls, data: dict[str, Any]) -> "IsotonicCalibrator":
        return cls(list(data["x"]), list(data["y"]))


CALIBRATORS: dict[str, type[Calibrator]] = {
    TemperatureScaler.method: TemperatureScaler,
    PlattScaler.method: PlattScaler,
    IsotonicCalibrator.method: IsotonicCalibrator,
}
