"""Evaluation metrics: Exact Match, Semantic Match, CER, WER.

Used to evaluate answers (EM, Semantic Match) and recognition quality
(CER, WER via jiwer) across typed, scanned, and handwritten documents.
Not a test module itself (no test_ prefix), so pytest does not collect it.
"""

import re
import string

import jiwer
import numpy as np

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """SQuAD-style normalization: lowercase, drop punctuation and articles, squeeze spaces."""
    text = _ARTICLES.sub(" ", text.lower().translate(_PUNCT))
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> bool:
    """True if the normalized prediction equals the normalized reference."""
    return normalize_answer(prediction) == normalize_answer(reference)


def contains_match(prediction: str, reference: str) -> bool:
    """True if the normalized reference appears inside the normalized prediction.

    Complements Exact Match for LLM answers, which are often full sentences
    ("The unit of force is the newton.") around a correct short answer.
    """
    ref = normalize_answer(reference)
    return bool(ref) and f" {ref} " in f" {normalize_answer(prediction)} "


def cosine(a: list[float], b: list[float]) -> float:
    a_, b_ = np.asarray(a), np.asarray(b)
    return float(a_ @ b_ / (np.linalg.norm(a_) * np.linalg.norm(b_)))


def semantic_match(prediction: str, reference: str, threshold: float = 0.8, embed=None) -> bool:
    """True if the embedding cosine similarity is at least threshold.

    embed: function list[str] -> list[vector]; defaults to the app's Gemini embedder.
    """
    if embed is None:
        from app.modules.embedder import embed_texts as embed
    pred_vec, ref_vec = embed([prediction, reference])
    return cosine(pred_vec, ref_vec) >= threshold


def _norm_text(text: str) -> str:
    # Recognition metrics ignore layout differences (line breaks, repeated spaces) but keep case/punctuation.
    return " ".join(text.split())


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate (jiwer.cer); 1.0 if nothing was recognized."""
    reference, hypothesis = _norm_text(reference), _norm_text(hypothesis)
    if not hypothesis:
        return 1.0
    return float(jiwer.cer(reference, hypothesis))


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate (jiwer.wer); 1.0 if nothing was recognized."""
    reference, hypothesis = _norm_text(reference), _norm_text(hypothesis)
    if not hypothesis:
        return 1.0
    return float(jiwer.wer(reference, hypothesis))
