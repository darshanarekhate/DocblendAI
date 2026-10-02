"""Module 6 — Reliability Tier Classification.

Responsibility: label an answer Certain / Moderate / Uncertain / Unreadable by
jointly weighing retrieval relevance (similarity) and recognition confidence
(calibrated) of the evidence it rests on: the top-ranked retrieval result.

    Unreadable  confidence < UNREADABLE_MAX_CONF (the relevant text could not
                be read reliably, whatever its similarity)
    Certain     similarity >= CERTAIN_MIN_SIM and confidence >= CERTAIN_MIN_CONF
    Moderate    similarity >= MODERATE_MIN_SIM and confidence >= MODERATE_MIN_CONF
    Uncertain   everything else, including no evidence at all

Using two thresholds instead of one on combined_score keeps each label
explainable: "Moderate because the handwriting was only partly legible" vs
"Moderate because the match was loose".

Uses from schemas.py: RetrievalResult, ReliabilityLabel.
"""

from app.models.schemas import ReliabilityLabel, RetrievalResult

# Similarity cut-offs for gemini-embedding-001 at 768 dims, set from the step 6
# evaluation (evaluation/results/report.md): the top chunk behind every correct
# answer scored 0.66-0.77, every unanswerable or unreadable question 0.50-0.52.
# Moderate sits midway across that gap. Retune with `run_eval --rescore` as the
# dataset grows (18 synthetic questions is a small sample).
CERTAIN_MIN_SIM = 0.65
MODERATE_MIN_SIM = 0.58
# Calibrated confidence ~ expected character accuracy (see confidence_capture).
CERTAIN_MIN_CONF = 0.85
MODERATE_MIN_CONF = 0.60
UNREADABLE_MAX_CONF = 0.35
# Text an LLM corrected or inferred ("Refine with LLM", automatic correction) was not read off
# the page, so it never raises reliability: its confidence is capped just below
# CERTAIN_MIN_CONF, i.e. evidence resting on it is at most Moderate (qa_index.py applies this).
LLM_TEXT_MAX_CONF = round(CERTAIN_MIN_CONF - 0.01, 2)
LLM_SOURCES = frozenset({"llm", "llm-auto"})

# Best to worst; used to combine labels ("the worse of two").
TIER_ORDER = [
    ReliabilityLabel.CERTAIN,
    ReliabilityLabel.MODERATE,
    ReliabilityLabel.UNCERTAIN,
    ReliabilityLabel.UNREADABLE,
]


def classify_reliability(results: list[RetrievalResult]) -> ReliabilityLabel:
    """Return the reliability tier for an answer built from these results."""
    if not results:
        return ReliabilityLabel.UNCERTAIN
    top = max(results, key=lambda r: r.combined_score)

    if top.confidence < UNREADABLE_MAX_CONF:
        return ReliabilityLabel.UNREADABLE
    if top.similarity >= CERTAIN_MIN_SIM and top.confidence >= CERTAIN_MIN_CONF:
        return ReliabilityLabel.CERTAIN
    if top.similarity >= MODERATE_MIN_SIM and top.confidence >= MODERATE_MIN_CONF:
        return ReliabilityLabel.MODERATE
    return ReliabilityLabel.UNCERTAIN


def worse_of(a: ReliabilityLabel, b: ReliabilityLabel) -> ReliabilityLabel:
    return max(a, b, key=TIER_ORDER.index)
