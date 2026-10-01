"""Supports Module 3 — the active (saved) calibrator used by the Experience Center.

`python -m app.calibration.calibrate` (or POST /api/calibrate) writes
settings.paddle_calibration_dir / "calibrator.json". load_active() reads it
once and caches it per path; clear_active_cache() forces a re-read after a
new calibrator is written. Without a usable file every confidence passes
through unchanged and is reported as "uncalibrated".
"""

import json
import logging
import threading
from pathlib import Path

from app.calibration.calibrators import Calibrator
from app.config import settings

logger = logging.getLogger(__name__)

CALIBRATOR_FILE = "calibrator.json"

_lock = threading.Lock()
_cache: dict[str, Calibrator | None] = {}


def active_path() -> Path:
    return Path(settings.paddle_calibration_dir) / CALIBRATOR_FILE


def load_active() -> Calibrator | None:
    """The saved calibrator, or None when there is none (or it cannot be read; logged once)."""
    path = active_path()
    key = str(path)
    with _lock:
        if key in _cache:
            return _cache[key]
        calibrator: Calibrator | None = None
        if path.exists():
            try:
                calibrator = Calibrator.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                logger.warning("Ignoring unreadable calibrator %s: %s", path, exc)
        _cache[key] = calibrator
        return calibrator


def clear_active_cache() -> None:
    """Forget the cached calibrator; the next load_active() reads the file again."""
    with _lock:
        _cache.clear()


def active_method() -> str | None:
    """Method name of the active calibrator ("temperature" | "platt" | "isotonic"), or None."""
    calibrator = load_active()
    return calibrator.method if calibrator else None


def calibrate_confidences(confidences: list[float]) -> tuple[list[float], str]:
    """(calibrated confidences, "calibrated"), or the raw values and "uncalibrated" if none is active."""
    calibrator = load_active()
    if calibrator is None:
        return [float(c) for c in confidences], "uncalibrated"
    if not confidences:
        return [], "calibrated"
    return [float(v) for v in calibrator.predict(confidences)], "calibrated"
