"""Module 5 — Question spelling correction ("Did you mean").

Responsibility: suggest a corrected question when it contains words that do
not occur in the selected documents (e.g. "retrival" for "retrieval"), since a
misspelled key term weakens retrieval similarity.

Two steps:
1. Fuzzy: each question word missing from the documents' vocabulary (skipping
   stopwords, numbers, and very short words) is matched with difflib to the
   closest vocabulary word above FUZZY_CUTOFF.
2. Gemini (only if GEMINI_API_KEY is set): asks the model to fix the question
   using only the documents' terms. The rewrite is kept only if every content
   word comes from the documents or the original question; otherwise, or on
   any Gemini error, the fuzzy result stands.

The vocabulary of each document set is cached; upload and delete call
invalidate().

Uses from schemas.py: nothing (works on question text and stored chunk text).
"""

import difflib
import logging
import re
import threading
from dataclasses import dataclass, field

from app.config import settings
from app.modules import llm_answer, vector_store

logger = logging.getLogger(__name__)

WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
MIN_WORD_LEN = 4  # shorter words are too ambiguous to correct ("tha" -> "the"? "that"?)
FUZZY_CUTOFF = 0.8  # difflib ratio; "retrival"/"retrieval" = 0.94, "explain"/"plain" = 0.83
CANDIDATES_PER_WORD = 5  # close vocabulary words shown to Gemini for each unknown word
MAX_CACHED_SETS = 32
SUFFIXES = ("s", "es", "d", "ed", "ing", "ly")  # word forms that count as the same term

# Common English and question words: never "corrected" towards a document term.
STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be because been before being
    below between both but by can could did do does doing done down during each either else even every
    few for from further had has have having he her here hers him his how however i if in into is it
    its itself just like many may me might more most much must my neither no nor not now of off on once
    only or other our ours out over own same shall she should since so some such than that the their
    theirs them then there these they this those though through to too under until up upon us very was
    we were what when where whether which while who whom whose why will with within without would yet
    you your yours
    answer answers brief briefly compare compared comparison define defined definition describe described
    detail details difference differences different differ discuss example examples explain explained
    give given help kind kinds list main mean meaning meant mention mentioned name named note notes please
    show summarise summarize summary tell term terms type types used using uses use work works
    """.split()
)

REWRITE_INSTRUCTION = (
    "You correct spelling mistakes in a question about some documents. Fix only misspelled "
    "or mistyped words, using the document terms provided; do not rephrase, answer, or add "
    "anything. Reply with only the corrected question, or exactly NO_CHANGE if it needs no correction."
)
NO_CHANGE = "NO_CHANGE"


@dataclass
class Suggestion:
    """A spelling suggestion for one question. suggestion is None when nothing needs correcting."""

    original: str
    suggestion: str | None = None
    corrections: list[tuple[str, str]] = field(default_factory=list)  # (word, replacement)
    source: str | None = None  # "fuzzy" | "gemini" | None


_cache: dict[frozenset[str] | None, tuple[frozenset[str], list[str]]] = {}
_cache_lock = threading.Lock()


def invalidate() -> None:
    """Forget every cached vocabulary (a document was added or removed)."""
    with _cache_lock:
        _cache.clear()


def vocabulary(doc_ids: list[str] | None = None) -> frozenset[str]:
    """Lowercased words of the given documents' chunks (None or empty = all documents)."""
    return _vocabulary(doc_ids)[0]


def _vocabulary(doc_ids: list[str] | None) -> tuple[frozenset[str], list[str]]:
    key = frozenset(doc_ids) if doc_ids else None
    with _cache_lock:
        if key in _cache:
            return _cache[key]
    words = frozenset(w.lower() for text in vector_store.get_texts(doc_ids) for w in WORD.findall(text))
    entry = (words, sorted(words))
    with _cache_lock:
        if len(_cache) >= MAX_CACHED_SETS:
            _cache.clear()
        _cache[key] = entry
    return entry


def _checkable(word: str) -> bool:
    """Worth checking against the vocabulary: long enough, not a number, not a stopword."""
    return len(word) >= MIN_WORD_LEN and not any(ch.isdigit() for ch in word) and word.lower() not in STOPWORDS


def _known(word: str, vocab: frozenset[str]) -> bool:
    """In the vocabulary, or another form of a vocabulary word ("select" when the notes say "selects")."""
    word = word.lower()
    return (
        word in vocab
        or any(word + s in vocab for s in SUFFIXES)
        or any(word.endswith(s) and word[: -len(s)] in vocab for s in SUFFIXES)
    )


def _match_case(replacement: str, original: str) -> str:
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def _only_inflection(a: str, b: str) -> bool:
    """"explain" vs "explains": a correctly spelled word in another form, not a typo."""
    short, long_ = sorted((a, b), key=len)
    return long_.startswith(short) and len(long_) - len(short) <= 3


def _fuzzy(question: str, vocab: frozenset[str], vocab_list: list[str]) -> tuple[str, list[tuple[str, str]]]:
    """Replace each unknown word by its closest vocabulary word, keeping punctuation and case."""
    corrections: list[tuple[str, str]] = []

    def _fix(m: re.Match) -> str:
        word = m.group(0)
        lower = word.lower()
        if not _checkable(word) or _known(word, vocab):
            return word
        close = difflib.get_close_matches(lower, vocab_list, n=1, cutoff=FUZZY_CUTOFF)
        if not close or _only_inflection(lower, close[0]):
            return word
        replacement = _match_case(close[0], word)
        corrections.append((word, replacement))
        return replacement

    return WORD.sub(_fix, question), corrections


def _unknown_words(question: str, vocab: frozenset[str]) -> list[str]:
    return [w for w in WORD.findall(question) if _checkable(w) and not _known(w, vocab)]


def _diff(original: str, rewrite: str) -> list[tuple[str, str]]:
    """Word-level replacements turning original into rewrite."""
    a, b = WORD.findall(original), WORD.findall(rewrite)
    matcher = difflib.SequenceMatcher(a=[w.lower() for w in a], b=[w.lower() for w in b], autojunk=False)
    return [
        (" ".join(a[i1:i2]), " ".join(b[j1:j2]))
        for op, i1, i2, j1, j2 in matcher.get_opcodes()
        if op == "replace"
    ]


def _gemini_rewrite(question: str, fuzzy: str, unknown: list[str], vocab: frozenset[str], vocab_list: list[str]) -> str | None:
    """Gemini's corrected question if it passes validation, else None. Never raises."""
    candidates = sorted(
        {c for w in unknown for c in difflib.get_close_matches(w.lower(), vocab_list, n=CANDIDATES_PER_WORD, cutoff=0.6)}
    )
    prompt = (
        f"Question: {question}\n"
        f"Words not found in the documents: {', '.join(unknown)}\n"
        f"Similar document terms: {', '.join(candidates) or '(none)'}\n"
        f"Automatic guess: {fuzzy}"
    )
    try:
        text = llm_answer._generate(prompt, system_instruction=REWRITE_INSTRUCTION)
    except Exception as e:  # any Gemini failure: the fuzzy result is good enough
        logger.warning("Gemini spelling rewrite failed, using fuzzy matching: %s", e)
        return None

    rewrite = text.strip().strip("\"'`*").strip()
    if not rewrite or rewrite.strip(" .").upper() == NO_CHANGE or "\n" in rewrite:
        return None
    allowed = vocab | {w.lower() for w in WORD.findall(question)}
    foreign = [w for w in WORD.findall(rewrite) if _checkable(w) and not _known(w, allowed)]
    if foreign:
        logger.info("Discarded Gemini spelling rewrite (words not in the documents: %s)", ", ".join(foreign))
        return None
    return rewrite


def _same(a: str, b: str) -> bool:
    return [w.lower() for w in WORD.findall(a)] == [w.lower() for w in WORD.findall(b)]


def suggest(question: str, doc_ids: list[str] | None = None) -> Suggestion:
    """Suggest a corrected question using the vocabulary of doc_ids (None or empty = all documents)."""
    result = Suggestion(original=question)
    vocab, vocab_list = _vocabulary(doc_ids)
    unknown = _unknown_words(question, vocab)
    if not vocab or not unknown:
        return result

    fuzzy, corrections = _fuzzy(question, vocab, vocab_list)
    if settings.gemini_api_key:
        rewrite = _gemini_rewrite(question, fuzzy, unknown, vocab, vocab_list)
        if rewrite is not None and not _same(rewrite, question):
            result.suggestion, result.corrections, result.source = rewrite, _diff(question, rewrite), "gemini"
            return result
    if corrections:
        result.suggestion, result.corrections, result.source = fuzzy, corrections, "fuzzy"
    return result
