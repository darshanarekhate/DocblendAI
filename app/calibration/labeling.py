"""Supports Module 3 — labelling OCR lines as correct or incorrect against ground truth.

Calibration needs, for every line PaddleOCR returns, a 0/1 label: was it
read correctly? Each image has ground-truth lines; OCR lines are paired with
them by a greedy global assignment: all (OCR line, GT line) pairs are sorted
by character error rate and taken lowest first, each line used at most once.
This copes with OCR returning lines in a different order, missing a line, or
finding an extra one. OCR lines left without a partner are incorrect.

A matched line is correct when
- match="exact": its normalised text equals the ground truth, or
- match="cer":   its CER against the ground truth is <= cer_threshold.

Normalisation strips the ends and collapses runs of whitespace; case and
punctuation are kept (a misread "l" for "1" is a real error).
"""

import re
from dataclasses import dataclass

MATCH_MODES = ("exact", "cer")


def normalize(text: str) -> str:
    """Strip the ends and collapse internal whitespace to single spaces."""
    return re.sub(r"\s+", " ", text or "").strip()


def levenshtein(a: str, b: str) -> int:
    """Edit distance (insertions, deletions, substitutions) between two strings."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def cer(hypothesis: str, reference: str) -> float:
    """Character error rate of hypothesis against reference, both normalised.

    edit distance / len(reference); an empty reference gives 0.0 for an empty
    hypothesis and 1.0 otherwise. Can exceed 1 when the hypothesis is much longer.
    """
    hyp, ref = normalize(hypothesis), normalize(reference)
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(hyp, ref) / len(ref)


@dataclass
class LineLabel:
    """One OCR line with its matched ground-truth line (None when unmatched) and 0/1 label."""

    text: str
    confidence: float
    truth: str | None
    cer: float | None
    correct: int


def split_truth(text: str) -> list[str]:
    """Ground-truth lines from a labels.csv cell: newline-separated (a literal "\\n" also works)."""
    text = (text or "").replace("\r\n", "\n").replace("\\n", "\n")
    return [normalize(line) for line in text.split("\n") if normalize(line)]


def label_lines(
    ocr_lines: list[tuple[str, float]],
    truth_lines: list[str],
    match: str = "cer",
    cer_threshold: float = 0.1,
) -> list[LineLabel]:
    """Label every OCR line of one image; returned in the same order as ocr_lines."""
    if match not in MATCH_MODES:
        raise ValueError(f"match must be one of {MATCH_MODES}, got {match!r}")
    truths = [normalize(t) for t in truth_lines if normalize(t)]
    pairs = sorted(
        (cer(text, truth), i, j) for i, (text, _) in enumerate(ocr_lines) for j, truth in enumerate(truths)
    )
    partner: dict[int, tuple[int, float]] = {}
    used_truth: set[int] = set()
    for error, i, j in pairs:
        if i in partner or j in used_truth:
            continue
        partner[i] = (j, error)
        used_truth.add(j)

    labels = []
    for i, (text, confidence) in enumerate(ocr_lines):
        if i not in partner:
            labels.append(LineLabel(text, float(confidence), None, None, 0))
            continue
        j, error = partner[i]
        if match == "exact":
            correct = normalize(text) == truths[j]
        else:
            correct = error <= cer_threshold + 1e-12
        labels.append(LineLabel(text, float(confidence), truths[j], error, int(correct)))
    return labels
