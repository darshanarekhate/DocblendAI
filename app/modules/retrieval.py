"""Module 5 — Confidence-Aware Retrieval (RAG).

Responsibility: embed the question, search the vector store, and rank chunks
by combined_score, a weighted sum of retrieval similarity and calibrated
recognition confidence:

    combined_score = w * similarity + (1 - w) * confidence    (w = settings.similarity_weight)

A wider candidate pool is fetched by similarity alone, then re-ranked by
combined_score, so a relevant chunk can win over a slightly more similar one
that was poorly recognized.

Confidence must not let an off-topic document crowd out the relevant one: a
long, perfectly typed file can otherwise fill every slot with loosely related
passages (confidence 1.0) and push out a handwritten page that matches the
question best. So the slots are filled from the *relevant* candidates
(similarity within RELEVANCE_MARGIN of the best match) by taking turns across
documents, each document's passages in combined_score order; only then are
remaining slots filled by combined_score. The result is returned best
combined_score first, as before.

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


def retrieve(
    query: Query, top_k: int = 5, doc_ids: list[str] | None = None
) -> list[tuple[RecognizedChunk, RetrievalResult]]:
    """Return the top_k chunks and their scores, best combined_score first.

    doc_ids: answer only from these documents (the ones the user selected); None = all.

    Raises embedder.EmbeddingError if the question cannot be embedded.
    """
    query_vector = embedder.embed_query(query.question_text)
    wanted = top_k * settings.candidate_multiplier
    pool = wanted
    while True:
        hits = vector_store.search(query_vector, pool, doc_ids)
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
    return _share_slots(distinct, top_k)


def _doc_of(chunk: RecognizedChunk) -> str:
    return chunk.chunk_id.rsplit(":", 1)[0]


def _share_slots(ranked: list[tuple[RecognizedChunk, RetrievalResult]], top_k: int):
    """Pick top_k from candidates already sorted by combined_score (see module docstring)."""
    if not ranked:
        return []
    best = max(result.similarity for _, result in ranked)
    relevant = [pair for pair in ranked if pair[1].similarity >= best - settings.relevance_margin]

    # Documents take turns, the one holding the best match first; each gives its passages
    # in combined_score order (ranked is already in that order).
    queues: dict[str, list] = {}
    for pair in sorted(relevant, key=lambda p: p[1].similarity, reverse=True):
        queues.setdefault(_doc_of(pair[0]), [])
    for pair in relevant:
        queues[_doc_of(pair[0])].append(pair)
    picked: list = []
    while len(picked) < top_k and any(queues.values()):
        for queue in queues.values():
            if queue and len(picked) < top_k:
                picked.append(queue.pop(0))

    chosen = {id(pair) for pair in picked}
    picked += [pair for pair in ranked if id(pair) not in chosen][: top_k - len(picked)]
    picked.sort(key=lambda pair: pair[1].combined_score, reverse=True)
    return picked
