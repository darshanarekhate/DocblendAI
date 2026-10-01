"""Supports Module 2 — optional image clean-up before PaddleOCR (Experience Center).

Responsibility: apply the user-chosen preprocessing steps to a page image, in a
fixed order that suits phone photos and photocopies:
- deskew: rotate small scan tilts (up to MAX_SKEW_DEGREES) back to level
- denoise: non-local-means smoothing; removes sensor noise and speckle, keeps strokes
- contrast: CLAHE (local histogram equalisation) on the lightness channel
- binarize: adaptive Gaussian threshold to pure black text on white

Every step is off by default: PP-OCRv5 is trained on natural images and usually
reads the original best. The steps help on faint, tilted or grainy scans, and
the before/after preview shows the user whether they do.

Uses from schemas.py: nothing; works on OpenCV BGR arrays.
"""

import cv2
import numpy as np

STEPS = ("deskew", "denoise", "contrast", "binarize")
# Larger detected angles are almost always text-free pages or tables, not tilt.
MAX_SKEW_DEGREES = 15.0


class PreprocessOptionsError(ValueError):
    """The preprocess options are not an object of known step names to booleans."""


def parse_options(options: dict | None) -> dict[str, bool]:
    """Validate {"deskew": true, ...}; unknown keys or non-boolean values are rejected."""
    if options is None:
        return {step: False for step in STEPS}
    if not isinstance(options, dict):
        raise PreprocessOptionsError("preprocess must be a JSON object, e.g. {\"deskew\": true}")
    unknown = sorted(set(options) - set(STEPS))
    if unknown:
        raise PreprocessOptionsError(f"unknown preprocess step(s): {', '.join(unknown)}; use {', '.join(STEPS)}")
    if not all(isinstance(v, bool) for v in options.values()):
        raise PreprocessOptionsError("preprocess values must be true or false")
    return {step: bool(options.get(step, False)) for step in STEPS}


def any_enabled(options: dict[str, bool] | None) -> bool:
    return bool(options) and any(options.values())


def _gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def skew_angle(image: np.ndarray) -> float:
    """Degrees to rotate (counter-clockwise) to level the text; 0.0 if no clear skew.

    The minimum-area rectangle around all ink pixels tilts with the text block.
    """
    gray = _gray(image)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    points = cv2.findNonZero(ink)
    if points is None or len(points) < 50:
        return 0.0
    (_, _), (w, h), angle = cv2.minAreaRect(points)
    # OpenCV >= 4.5 reports angles in (0, 90]; map to the smallest rotation that levels the box.
    if w < h:
        angle = angle - 90
    if abs(angle) > MAX_SKEW_DEGREES or abs(angle) < 0.1:
        return 0.0
    return float(angle)


def deskew(image: np.ndarray) -> np.ndarray:
    angle = skew_angle(image)
    if angle == 0.0:
        return image
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    # Expand the canvas so rotated corners are not cut off; fill with white paper.
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    matrix[0, 2] += new_w / 2 - w / 2
    matrix[1, 2] += new_h / 2 - h / 2
    border = 255 if image.ndim == 2 else (255, 255, 255)
    return cv2.warpAffine(image, matrix, (new_w, new_h), flags=cv2.INTER_CUBIC, borderValue=border)


def denoise(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.fastNlMeansDenoising(image, None, h=10)
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if image.ndim == 2:
        return clahe.apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def binarize(image: np.ndarray) -> np.ndarray:
    gray = _gray(image)
    # Block size ~1/30 of the short side (odd, >= 15) adapts to uneven lighting on photos.
    block = max(15, (min(gray.shape) // 30) | 1)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 15)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def apply(image: np.ndarray, options: dict[str, bool] | None) -> np.ndarray:
    """Run the enabled steps in STEPS order; returns a BGR image (input unchanged if none enabled)."""
    if not any_enabled(options):
        return image
    out = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if options.get("deskew"):
        out = deskew(out)
    if options.get("denoise"):
        out = denoise(out)
    if options.get("contrast"):
        out = enhance_contrast(out)
    if options.get("binarize"):
        out = binarize(out)
    return out
