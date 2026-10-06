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
    """Answer generation failed: missing API key or a Gemini API error.

    status is the HTTP status the API should answer with: 503 (no key), 429 (quota used
    up), 502 (any other Gemini failure). The message is written for the user.
    """

    def __init__(self, message: str, status: int = 502) -> None:
        super().__init__(message)
        self.status = status


NO_KEY_MESSAGE = "Gemini is not configured: set GEMINI_API_KEY in .env (see .env.example) and restart the server."


def describe_gemini_error(code: int | None, model: str, detail: object) -> tuple[str, int]:
    """(message for the user, HTTP status) for a failed Gemini call.

    Shared by answering, spelling, embeddings and the Experience Center plugins, so a used-up
    free-tier quota reads the same everywhere instead of as a bare stack-trace string.
    """
    text = str(detail)
    if code == 429 or "RESOURCE_EXHAUSTED" in text:
        return (
            f"Gemini's quota for {model} is used up for now (429). Free-tier limits reset daily: "
            "try again later, or set LLM_MODEL / LLM_FALLBACK_MODEL in .env to another model.",
            429,
        )
    if code in (500, 503) or "UNAVAILABLE" in text:
        return f"Gemini ({model}) is overloaded or unavailable right now ({code}); try again in a minute.", 503
    if code in (400, 401, 403) and ("API key" in text or "API_KEY" in text or "PERMISSION_DENIED" in text):
        return f"Gemini rejected the API key ({code}): check GEMINI_API_KEY in .env.", 503
    return f"Gemini call failed ({model}, {code}): {text}", 502


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
            f"take facts from the context passages, not from here):\n{turns}\n\n"
            "If the new question refers back to something (e.g. 'they', 'it', 'that'), answer about "
            "that same subject only. If the context passages do not cover that subject, reply with "
            f"exactly {NOT_FOUND}; do not answer about a different subject the passages happen to cover.\n\n"
            f"{prompt}"
        )
    return prompt


def _generate(prompt: str, system_instruction: str = SYSTEM_INSTRUCTION) -> str:
    """Answer with settings.llm_model, or settings.llm_fallback_model if the first stays busy.

    system_instruction defaults to the answering rules; spelling.py passes its own
    to reuse the same models, retries, and fallback.

    Free-tier Gemini quotas are counted per model, so when the default model is
    out of quota (429) or overloaded (503) a second model can usually still answer.
    Other errors (a bad request, a missing key) are not retried on another model.
    """
    if not settings.gemini_api_key:
        raise LLMError(NO_KEY_MESSAGE, 503)

    models = [settings.llm_model]
    if settings.llm_fallback_model and settings.llm_fallback_model != settings.llm_model:
        models.append(settings.llm_fallback_model)
    for i, model in enumerate(models):
        try:
            return _generate_with(model, prompt, system_instruction)
        except errors.APIError as e:
            if e.code in RETRYABLE_CODES and i < len(models) - 1:
                logger.warning("%s unavailable (%s); answering with %s instead", model, e.code, models[i + 1])
                continue
            raise LLMError(*describe_gemini_error(e.code, model, e)) from e
    raise AssertionError("unreachable")


def _generate_with(model: str, prompt: str, system_instruction: str = SYSTEM_INSTRUCTION) -> str:
    """One Gemini model, with short retries on overload/rate limits. Raises the last APIError."""
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
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
