"""Supports Module 7 — word-level diff between recognised text and an LLM refinement of it.

Responsibility: tell, word by word, what a refinement changed, so the review view
can highlight it and the safety rule can reject over-eager rewrites:
- equal: the word was kept
- replace: a recognised word was corrected (the review view shows "original -> refined")
- insert: a word the model inferred from context (nothing was recognised there)
- delete: a recognised word the model dropped (e.g. OCR noise)

A refined line's overall status is "unchanged", "corrected" (only replacements or
deletions) or "inferred" (at least one inserted word: text that was not on the page
as recognised, so it is the least trustworthy kind of change).

change_ratio is the character edit distance over the original's length: the share of
the recognised characters that the refinement changed. Lines above a limit (0.4 for
manual refinement, 0.3 for the automatic pass) keep their original text.

Uses from schemas.py: nothing; works on plain strings.
"""

from difflib import SequenceMatcher
from typing import Any

from app.calibration.labeling import levenshtein

UNCHANGED, CORRECTED, INFERRED, REJECTED = "unchanged", "corrected", "inferred", "rejected"


def word_diff(original: str, refined: str) -> list[dict[str, Any]]:
    """[{op, original, refined}] over whitespace-separated words, in reading order.

    A replacement of n words by m words is split so that the first min(n, m) pairs are
    "replace" and the extra refined words are "insert" (or extra original words "delete"):
    "teh cat" -> "the black cat" reads as teh->the, +black, cat.
    """
    a, b = original.split(), refined.split()
    ops: list[dict[str, Any]] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            ops.extend({"op": "equal", "original": w, "refined": w} for w in a[i1:i2])
        elif tag == "delete":
            ops.extend({"op": "delete", "original": w, "refined": ""} for w in a[i1:i2])
        elif tag == "insert":
            ops.extend({"op": "insert", "original": "", "refined": w} for w in b[j1:j2])
        else:  # replace
            old, new = a[i1:i2], b[j1:j2]
            if "".join(old) == "".join(new):
                # Only the spacing changed: a merged word split ("submissions:lose" -> "submissions: lose")
                # or a broken word joined. A correction, not an inferred word.
                ops.append({"op": "replace", "original": " ".join(old), "refined": " ".join(new)})
                continue
            paired = min(len(old), len(new))
            ops.extend({"op": "replace", "original": o, "refined": n} for o, n in zip(old[:paired], new[:paired]))
            ops.extend({"op": "insert", "original": "", "refined": w} for w in new[paired:])
            ops.extend({"op": "delete", "original": w, "refined": ""} for w in old[paired:])
    return ops


def line_status(ops: list[dict[str, Any]]) -> str:
    kinds = {op["op"] for op in ops}
    if "insert" in kinds:
        return INFERRED
    if kinds & {"replace", "delete"}:
        return CORRECTED
    return UNCHANGED


def change_ratio(original: str, refined: str) -> float:
    """Character edit distance / original length (an all-new line counts as 1.0 or more)."""
    if original == refined:
        return 0.0
    return levenshtein(original, refined) / max(1, len(original))


def compare(original: str, refined: str, max_change: float) -> dict[str, Any]:
    """Diff + status for one line; status "rejected" when change_ratio exceeds max_change."""
    ops = word_diff(original, refined)
    ratio = change_ratio(original, refined)
    status = line_status(ops)
    out = {"status": status, "change_ratio": round(ratio, 4), "diff": ops, "reason": None}
    if status != UNCHANGED and ratio > max_change:
        out["status"] = REJECTED
        out["reason"] = (
            f"The refinement changes {ratio:.0%} of the characters (limit {max_change:.0%}), "
            "so the recognised text was kept."
        )
    return out
