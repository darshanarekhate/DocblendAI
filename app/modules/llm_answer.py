"""Module 7 — LLM Answer Generation.

Responsibility: prompt Gemini with the question and retrieved chunk text,
and return the answer tagged with its reliability label.

Uses from schemas.py: Query, RecognizedChunk, ReliabilityLabel, Answer.
"""

import logging
import time
import uuid
from functools import lru_cache

from google import genai
from google.genai import errors, types

from app.config import settings
from app.models.schemas import Answer, Query, RecognizedChunk, ReliabilityLabel
from app.modules.reliability import worse_of

logger = logging.getLogger(__name__)

NOT_FOUND = "NOT_FOUND"
NOT_FOUND_ANSWER = "I couldn't find the answer to that in the uploaded documents."
NO_DOCUMENTS_ANSWER = "No documents have been uploaded yet, so there is nothing to answer from."

SYSTEM_INSTRUCTION = (
    "You answer questions about academic documents. Use only the numbered context "
    "passages provided; never use outside knowledge. Be concise and precise. The user "
    "cannot see the passages, so answer directly without mentioning them, the context, "
    "or passage numbers. Passages marked as recognized text may contain OCR or "
    "handwriting errors: do not guess at garbled words. If the "
    f"passages do not contain the answer, reply with exactly {NOT_FOUND}."
)

# Conversation memory for follow-up questions (chat): how much of the past goes into the prompt.
MAX_HISTORY_TURNS = 3
MAX_HISTORY_ANSWER_CHARS = 600

# Passages below this confidence are flagged to the model as possibly misrecognized.
LOW_CONFIDENCE = 0.95

# Gemini returns these under load (503) or rate limiting (429); worth a short retry.
RETRYABLE_CODES = {429, 500, 503}
MAX_ATTEMPTS = 3


class LLMError(RuntimeError):
    """Answer generation failed: missing API key or a Gemini API error."""


@lru_cache
def _client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def _passage_header(i: int, chunk: RecognizedChunk) -> str:
    notes = [chunk.content_type.value]
    conf = chunk.calibrated_conf if chunk.calibrated_conf is not None else chunk.raw_conf
    if conf < LOW_CONFIDENCE:
        notes.append(f"recognized text, confidence {conf:.2f}: may contain recognition errors")
    return f"[{i}] ({'; '.join(notes)})"


def build_prompt(question: str, chunks: list[RecognizedChunk], history: list[tuple[str, str]] | None = None) -> str:
    context = "\n\n".join(f"{_passage_header(i, c)}\n{c.text}" for i, c in enumerate(chunks, 1))
    prompt = f"Context passages:\n{context}\n\nQuestion: {question}"
    if history:
        # Only the last few turns, trimmed: enough to resolve "it" / "that one", not a second source.
        turns = "\n".join(
            f"Q: {q}\nA: {a if len(a) <= MAX_HISTORY_ANSWER_CHARS else a[:MAX_HISTORY_ANSWER_CHARS] + '…'}"
            for q, a in history[-MAX_HISTORY_TURNS:]
        )
        prompt = (
            "Earlier in this conversation (use it only to understand what the new question refers to; "
            f"take facts from the context passages, not from here):\n{turns}\n\n{prompt}"
        )
    return prompt


def _generate(prompt: str) -> str:
    """Answer with settings.llm_model, or settings.llm_fallback_model if the first stays busy.

    Free-tier Gemini quotas are counted per model, so when the default model is
    out of quota (429) or overloaded (503) a second model can usually still answer.
    Other errors (a bad request, a missing key) are not retried on another model.
    """
    if not settings.gemini_api_key:
        raise LLMError("GEMINI_API_KEY is not set (see .env.example)")

    models = [settings.llm_model]
    if settings.llm_fallback_model and settings.llm_fallback_model != settings.llm_model:
        models.append(settings.llm_fallback_model)
    for i, model in enumerate(models):
        try:
            return _generate_with(model, prompt)
        except errors.APIError as e:
            if e.code in RETRYABLE_CODES and i < len(models) - 1:
                logger.warning("%s unavailable (%s); answering with %s instead", model, e.code, models[i + 1])
                continue
            raise LLMError(f"Gemini generation failed ({model}): {e}") from e
    raise AssertionError("unreachable")


def _generate_with(model: str, prompt: str) -> str:
    """One Gemini model, with short retries on overload/rate limits. Raises the last APIError."""
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.2,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = _client(settings.gemini_api_key).models.generate_content(model=model, contents=prompt, config=config)
            break
        except errors.APIError as e:
            if e.code in RETRYABLE_CODES and attempt < MAX_ATTEMPTS - 1:
                time.sleep(2**attempt)
                continue
            raise

    if not resp.text:
        raise LLMError("Gemini returned an empty response (possibly blocked)")
    return resp.text.strip()


def generate_answer(
    query: Query,
    chunks: list[RecognizedChunk],
    reliability_label: ReliabilityLabel,
    history: list[tuple[str, str]] | None = None,
) -> Answer:
    """Generate a grounded answer from the retrieved chunks.

    history: earlier (question, answer) turns of the same chat, so follow-ups
    like "explain that in more detail" can be understood.

    If the model finds no answer in the chunks, the label is lowered to at
    least Uncertain, whatever retrieval scores suggested.
    """
    answer_id = uuid.uuid4().hex
    if not chunks:
        return Answer(answer_id=answer_id, answer_text=NO_DOCUMENTS_ANSWER, reliability_label=reliability_label)

    text = _generate(build_prompt(query.question_text, chunks, history))
    # Lite models sometimes wrap the marker in punctuation, quotes, or bold.
    if text.strip(" .\"'`*").upper() == NOT_FOUND:
        text = NOT_FOUND_ANSWER
        reliability_label = worse_of(reliability_label, ReliabilityLabel.UNCERTAIN)
    return Answer(answer_id=answer_id, answer_text=text, reliability_label=reliability_label)
