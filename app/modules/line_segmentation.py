"""Supports Module 2 — find handwritten text lines on real notebook pages for HTR.

Responsibility: prepare a page for TrOCR, which reads one text line at a time.

Real student notes are written on ruled notebook paper. The printed rules
defeat a plain ink-projection splitter: every rule is a full-width "ink" row,
so the whole page looks like one huge line (or many 2-pixel lines), and TrOCR
then reads gibberish. Two open-source tools fix this:

1. remove_ruled_lines() — OpenCV morphology (opencv/opencv). Long, thin,
   horizontal runs of ink (rules) and long vertical ones (the margin) are
   masked and inpainted, so a pen stroke that crosses a rule stays joined.
2. detect_lines() — docTR's DBNet text detector (mindee/doctr,
   db_resnet50) finds word boxes on the cleaned page, and
   group_into_lines() joins them into text lines, top to bottom.

If docTR is not installed or its weights cannot be downloaded, the caller
falls back to the projection splitter (htr_extractor.segment_lines), which
also works far better on a page whose rules have been removed.

Uses from schemas.py: nothing; returns PIL images to htr_extractor.
"""

import logging
from functools import lru_cache

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# A rule must span at least this share of the page width (vertical margin rule: page height).
# Handwriting rarely has a straight horizontal run this long; a long underline may be removed too.
MIN_RULE_LENGTH = 1 / 12
# Rules are thin: ~0.3 mm, 2-3 px at 200 DPI. Thicker horizontal ink (e.g. a heavy
# underline or a filled box) is kept as content.
MAX_RULE_THICKNESS = 0.003  # share of page height

# docTR word boxes whose vertical overlap is at least this share of the shorter box are on one line.
SAME_LINE_OVERLAP = 0.5
# A horizontal gap wider than this many line-heights splits a line into two (e.g. two columns).
MAX_WORD_GAP = 4.0
LINE_PAD = 0.25  # padding around each line crop, as a share of the line height
MIN_WORD_SCORE = 0.3  # drop detections the model itself doubts (specks, paper texture)
# Inside a line crop, an ink shape is kept only if its centre lies within the line's band
# (median word top..bottom) widened by this share of the line height; shapes centred outside
# are pieces of the lines above or below.
BAND_SLACK = 0.15
# A shape wider than this many line heights and thinner than RULE_MAX_HEIGHT line heights is a
# leftover notebook rule or an underline, not writing.
RULE_MIN_WIDTH = 2.5
RULE_MAX_HEIGHT = 0.25
# whiten_background: pixels lighter than ink_core + WHITEN_MIX * (otsu - ink_core) become paper.
WHITEN_MIX = 0.5


class LineDetectorUnavailableError(RuntimeError):
    """docTR (or its detection weights) could not be loaded."""


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """Binary ink mask (255 = ink) that copes with uneven lighting in phone photos."""
    import cv2

    h, w = gray.shape
    block = max(15, (min(h, w) // 40) | 1)  # odd window, ~2.5% of the page
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 15)


def _thin_runs(ink: np.ndarray, kernel_shape: tuple[int, int], max_thickness: int, axis: int) -> np.ndarray:
    """Mask of long straight runs (per kernel_shape) no thicker than max_thickness across `axis`."""
    import cv2

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_shape)
    runs = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(runs, connectivity=8)
    # Mean thickness = area / length. Not the bounding box: a rule on a slightly tilted scan
    # climbs a few pixels across the page, so its box is tall although the line is thin.
    length = stats[:, cv2.CC_STAT_WIDTH if axis == 0 else cv2.CC_STAT_HEIGHT]
    thickness = stats[:, cv2.CC_STAT_AREA] / np.maximum(length, 1)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = thickness[1:] <= max_thickness  # label 0 is the background
    return keep[labels]


def remove_ruled_lines(page: Image.Image) -> Image.Image:
    """Return a grayscale copy of the page with notebook rules and margin lines erased."""
    import cv2

    gray = np.asarray(page.convert("L"))
    h, w = gray.shape
    ink = _ink_mask(gray)
    thickness = max(2, round(MAX_RULE_THICKNESS * h))
    rules = _thin_runs(ink, (max(20, int(w * MIN_RULE_LENGTH)), 1), thickness, axis=0)
    rules |= _thin_runs(ink, (1, max(20, int(h * MIN_RULE_LENGTH))), thickness, axis=1)
    if not rules.any():
        return page.convert("L")

    # Grow the mask by a pixel so anti-aliased rule edges go too, then fill it from the
    # surroundings: paper above and below a rule -> paper; a stroke crossing it -> stroke.
    mask = cv2.dilate(rules.astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
    cleaned = cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)
    return Image.fromarray(cleaned)


def whiten_background(page: Image.Image) -> Image.Image:
    """Turn everything clearly lighter than the page's own ink into white paper.

    Thin notebook paper shows the writing on its back as faint grey mirrored text,
    and rule removal leaves light traces; docTR detects both as extra "lines" and
    TrOCR reads them as invented words. The cutoff sits halfway between the ink's
    typical darkness (25th percentile of the pixels darker than Otsu's threshold)
    and Otsu's threshold, so it adapts to the scan: ~85 for dark pen on a phone
    scan, ~180 for pale pencil (whose strokes are kept).
    """
    import cv2

    gray = np.asarray(page.convert("L"))
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = gray[gray < otsu]
    if ink.size == 0:
        return page.convert("L")
    core = float(np.percentile(ink, 25))
    cutoff = core + WHITEN_MIX * (otsu - core)
    return Image.fromarray(np.where(gray > cutoff, 255, gray).astype(np.uint8))


@lru_cache
def _detector():
    try:
        from doctr.models import detection_predictor

        return detection_predictor("db_resnet50", pretrained=True, assume_straight_pages=True)
    # ImportError: docTR not installed. OSError / RuntimeError: weights not downloadable or corrupt.
    except (ImportError, OSError, RuntimeError) as e:
        raise LineDetectorUnavailableError(f"docTR text detector unavailable: {e}") from e


Box = tuple[float, float, float, float]  # left, top, right, bottom in pixels


def group_into_lines(boxes: list[Box]) -> list[Box]:
    """Join word boxes into text-line boxes, top to bottom (see group_words)."""
    return [_union(words) for words in group_words(boxes)]


def group_words(boxes: list[Box]) -> list[list[Box]]:
    """Group word boxes into text lines, top to bottom; each line is its words, left to right.

    A word joins the line it overlaps vertically the most (by at least
    SAME_LINE_OVERLAP of the shorter height); a gap wider than MAX_WORD_GAP
    line-heights starts a new line (two columns side by side stay apart).
    """
    lines: list[list[Box]] = []
    for box in sorted(boxes, key=lambda b: (b[1] + b[3]) / 2):
        best, best_overlap = None, SAME_LINE_OVERLAP
        for line in lines:
            # The line's typical band (median word top/bottom), not its full extent: with the
            # extent, one tall word (a descender, a loop) stretches the line until it swallows
            # the lines below, and TrOCR gets a crop of several lines, which it cannot read.
            top, bottom = float(np.median([b[1] for b in line])), float(np.median([b[3] for b in line]))
            overlap = min(bottom, box[3]) - max(top, box[1])
            share = overlap / max(1e-6, min(bottom - top, box[3] - box[1]))
            if share >= best_overlap:
                best, best_overlap = line, share
        if best is None:
            lines.append([box])
        else:
            best.append(box)

    out: list[list[Box]] = []
    for line in lines:
        words = sorted(line, key=lambda b: b[0])
        height = np.median([b[3] - b[1] for b in words])
        segment = [words[0]]
        for word in words[1:]:
            if word[0] - max(b[2] for b in segment) > MAX_WORD_GAP * height:
                out.append(segment)
                segment = []
            segment.append(word)
        out.append(segment)
    return out  # lines were created top to bottom, and each line's segments left to right


def _union(boxes: list[Box]) -> Box:
    return min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)


def detect_lines(page: Image.Image) -> list[Image.Image]:
    """Crop the page into text-line images using docTR word detection.

    Raises LineDetectorUnavailableError if docTR cannot be loaded.
    """
    rgb = np.asarray(page.convert("RGB"))
    h, w = rgb.shape[:2]
    words = _detector()([rgb])[0]["words"]
    boxes = [
        (x0 * w, y0 * h, x1 * w, y1 * h) for x0, y0, x1, y1, score in np.asarray(words).tolist() if score >= MIN_WORD_SCORE
    ]
    import cv2

    gray = np.asarray(page.convert("L"))
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [_line_crop(gray, otsu, words_in_line) for words_in_line in group_words(boxes)]


def _line_crop(gray: np.ndarray, otsu: float, words_in_line: list[Box]) -> Image.Image:
    """One line's image with everything that is not this line's writing whitened.

    docTR's word boxes on dense handwriting reach into the neighbouring lines, so cropping
    (even word by word) brings in the tops and bottoms of the lines above and below, plus
    rule and underline traces; TrOCR reads all of that as invented words. Each connected ink
    shape is kept only if its centre is inside this line's band and it is not rule-like.
    """
    import cv2

    h, w = gray.shape
    left, top, right, bottom = _union(words_in_line)
    band_top = float(np.median([b[1] for b in words_in_line]))
    band_bottom = float(np.median([b[3] for b in words_in_line]))
    line_h = max(1.0, band_bottom - band_top)
    pad = LINE_PAD * (bottom - top)
    x0, y0, x1, y1 = max(0, int(left - pad)), max(0, int(top - pad)), min(w, int(right + pad)), min(h, int(bottom + pad))
    region = gray[y0:y1, x0:x1]
    count, labels, stats, centroids = cv2.connectedComponentsWithStats((region <= otsu).astype(np.uint8), connectivity=8)
    keep = np.zeros(count, dtype=bool)
    for i in range(1, count):
        centre_y = centroids[i][1] + y0
        inside = band_top - BAND_SLACK * line_h <= centre_y <= band_bottom + BAND_SLACK * line_h
        rule_like = (stats[i, cv2.CC_STAT_WIDTH] > RULE_MIN_WIDTH * line_h
                     and stats[i, cv2.CC_STAT_HEIGHT] < RULE_MAX_HEIGHT * line_h)
        keep[i] = inside and not rule_like
    cleaned = np.where((labels > 0) & ~keep[labels], 255, region)
    return Image.fromarray(cleaned.astype(np.uint8))
