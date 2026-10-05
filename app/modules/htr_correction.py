"""Module 2 (HTR path) — correct handwriting-recognition errors with Gemini.

TrOCR misreads handwriting into wrong letters or wrong real words ("Reasoning" ->
"Reaswings", "missing information" -> "envisioning information"). Retrieval and
answers then work on the wrong words. After HTR, each page's text is sent to
Gemini once with strict instructions: restore the words the writer most likely
wrote, from the surrounding words, and nothing else: no added, removed or
reworded content. A guard rejects any line that changed too much, so a line the
model rewrote keeps its recognized text.

The recognition confidence (raw_conf) is not changed: it still measures how well
the handwriting itself could be read, which is what the reliability labels report.

On the project's real notebook scan: page 1 CER 0.12 -> 0.05, WER 0.27 -> 0.07.

Any Gemini failure (no key, quota, network) leaves the recognized text as it is;
an upload never fails because of this step.
"""

import difflib
import logging
import re

from google.genai import errors, types

from app.config import settings
from app.modules import llm_answer

logger = logging.getLogger(__name__)

PROMPT = """Below is machine-read text of one page of a student's handwritten notes. The handwriting \
recognizer makes spelling mistakes and sometimes reads a word as a different real word.

Each line starts with its number and "|". Correct only recognition errors: restore the word the \
student most likely wrote, using the surrounding words and the topic of the page. Rules:
- Output every line with its same number and "|", in the same order; never merge or split lines.
- Do not add, remove, summarize or reword anything; do not add explanations or facts.
- Remove stray symbols that are clearly recognizer noise (such as "#", '"', "'" scattered in a line).
- If a line is unreadable noise, output it unchanged.
- Output only the corrected lines.

Text:
"""


def _changed(a: str, b: str) -> float:
    """Share of characters that differ between two strings (0 = same, 1 = nothing in common)."""
    return 1 - difflib.SequenceMatcher(None, a, b).ratio()


_NUMBERED = re.compile(r"^\s*(\d+)\s*\|\s?(.*)$")


def numbered(text: str) -> str:
    """The page with each line prefixed by its number, so replies can be matched line by line."""
    return "\n".join(f"{i}| {line}" for i, line in enumerate(text.splitlines(), 1))


def guard(original: str, corrected: str) -> str:
    """Rebuild the page line by line from the model's numbered reply.

    A line keeps its correction only if it changed at most settings.htr_correction_max_change
    of its characters; a line the model rewrote, dropped or merged keeps the recognized text.
    """
    limit = settings.htr_correction_max_change
    replies: dict[int, str] = {}
    for line in corrected.splitlines():
        m = _NUMBERED.match(line)
        if m:
            replies.setdefault(int(m.group(1)), m.group(2).strip())
    out = []
    for i, old in enumerate(original.splitlines(), 1):
        new = replies.get(i)
        out.append(new if new is not None and _changed(old, new) <= limit else old)
    return "\n".join(out)


def _ask(text: str) -> str | None:
    if not settings.gemini_api_key:
        return None
    config = types.GenerateContentConfig(
        temperature=0.0, automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
    )
    models = [settings.llm_model] + ([settings.llm_fallback_model] if settings.llm_fallback_model else [])
    for model in models:
        try:
            resp = llm_answer._client(settings.gemini_api_key).models.generate_content(
                model=model, contents=PROMPT + numbered(text), config=config
            )
            return (resp.text or "").strip() or None
        except errors.APIError as e:
            logger.warning("Handwriting correction with %s failed (%s)", model, e.code)
    return None


def correct_page(text: str) -> str:
    """The page text with recognition errors corrected; the original text if correction is off or fails."""
    if not settings.htr_llm_correction or not text.strip():
        return text
    try:
        corrected = _ask(text)
    except Exception as e:  # noqa: BLE001 - a correction failure must never fail the upload
        logger.warning("Handwriting correction failed: %s", e)
        return text
    return guard(text, corrected) if corrected else text


def correct_pages(pages: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """correct_page for every (text, raw_conf) page; confidences are kept as recognized."""
    return [(correct_page(text), conf) for text, conf in pages]
