"""Supports Module 3 — confidence calibration for the PaddleOCR Experience Center.

STUB: identity behaviour with the final signatures (see docs/experience_center_contract.md §1),
so the service/API can be built in parallel. Replaced by the real calibrators.
"""

from typing import Any


class Calibrator:
    method = "identity"

    def fit(self, confidences, labels) -> "Calibrator":
        return self

    def predict(self, confidences):
        import numpy as np

        return np.asarray(confidences, dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Calibrator":
        return Calibrator()


def load_active() -> Calibrator | None:
    return None


def clear_active_cache() -> None:
    return None


def calibrate_confidences(confidences: list[float]) -> tuple[list[float], str]:
    """(calibrated confidences, "calibrated" | "uncalibrated")."""
    return [float(c) for c in confidences], "uncalibrated"
