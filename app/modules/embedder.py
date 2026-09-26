"""Supports Module 5 — Gemini embeddings.

Responsibility: turn chunk text and questions into vectors via the Gemini
embedding API (google-genai SDK, key from config.settings.gemini_api_key).

Uses from schemas.py: RecognizedChunk (fills its vector field).
"""

from functools import lru_cache

from google import genai
from google.genai import errors, types

from app.config import settings
from app.models.schemas import RecognizedChunk

# Gemini rejects embedding batches larger than this.
MAX_BATCH = 100


class EmbeddingError(RuntimeError):
    """Embedding failed: missing API key or a Gemini API error."""


@lru_cache
def _client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def _embed(texts: list[str], task_type: str) -> list[list[float]]:
    if not settings.gemini_api_key:
        raise EmbeddingError("GEMINI_API_KEY is not set (see .env.example)")

    config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=settings.embedding_dim)
    vectors: list[list[float]] = []
    try:
        for i in range(0, len(texts), MAX_BATCH):
            resp = _client(settings.gemini_api_key).models.embed_content(
                model=settings.embedding_model, contents=texts[i : i + MAX_BATCH], config=config
            )
            vectors.extend(e.values for e in resp.embeddings)
    except errors.APIError as e:
        raise EmbeddingError(f"Gemini embedding call failed: {e}") from e
    return vectors


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of document texts."""
    if not texts:
        return []
    return _embed(texts, "RETRIEVAL_DOCUMENT")


def embed_query(text: str) -> list[float]:
    """Embed a single question (query-side task type)."""
    return _embed([text], "RETRIEVAL_QUERY")[0]


def embed_chunks(chunks: list[RecognizedChunk]) -> list[RecognizedChunk]:
    """Return copies of the chunks with vector set."""
    vectors = embed_texts([c.text for c in chunks])
    return [c.model_copy(update={"vector": v}) for c, v in zip(chunks, vectors)]
