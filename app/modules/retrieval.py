"""Module 5 — Confidence-Aware Retrieval (RAG).

Responsibility: embed the question, search the vector store, and rank chunks
by combined_score, a weighted sum of retrieval similarity and calibrated
recognition confidence:

    combined_score = w * similarity + (1 - w) * confidence    (w = settings.similarity_weight)

A wider candidate pool is fetched by similarity alone, then re-ranked by
combined_score, so a relevant chunk can win over a slightly more similar one
that was poorly recognized.

Uses from schemas.py: Query, RecognizedChunk, RetrievalResult.
"""

from app.config import settings
from app.models.schemas import Query, RecognizedChunk, RetrievalResult
from app.modules import embedder, vector_store


MAX_POOL = 200  # upper bound on candidates fetched while skipping duplicates


def _text_key(chunk: RecognizedChunk) -> str:
    """Chunks with the same text, ignoring whitespace and case, are the same passage."""
    return " ".join(chunk.text.split()).lower()


def combined_score(similarity: float, confidence: float) -> float:
    """Blend retrieval similarity with calibrated recognition confidence."""
    w = settings.similarity_weight
    return w * similarity + (1 - w) * confidence


def retrieve(query: Query, top_k: int = 5) -> list[tuple[RecognizedChunk, RetrievalResult]]:
    """Return the top_k chunks and their scores, best combined_score first.

    Raises embedder.EmbeddingError if the question cannot be embedded.
    """
    query_vector = embedder.embed_query(query.question_text)
    wanted = top_k * settings.candidate_multiplier
    pool = wanted
    while True:
        hits = vector_store.search(query_vector, pool)
        # Identical chunks (e.g. one file uploaded twice) would fill every slot with the same
        # passage; widen the pool until it holds enough different ones or the store runs out.
        if len({_text_key(c) for c, _ in hits}) >= wanted or len(hits) < pool or pool >= MAX_POOL:
            break
        pool *= 2

    scored = []
    for chunk, similarity in hits:
        confidence = chunk.calibrated_conf if chunk.calibrated_conf is not None else chunk.raw_conf
        result = RetrievalResult(
            chunk_id=chunk.chunk_id,
            similarity=similarity,
            confidence=confidence,
            combined_score=combined_score(similarity, confidence),
        )
        scored.append((chunk, result))
    scored.sort(key=lambda pair: pair[1].combined_score, reverse=True)

    # Keep each passage once: its best-ranked copy, so a clearly read copy beats an illegible one.
    seen: set[str] = set()
    distinct = []
    for chunk, result in scored:
        if _text_key(chunk) not in seen:
            seen.add(_text_key(chunk))
            distinct.append((chunk, result))
    return distinct[:top_k]
