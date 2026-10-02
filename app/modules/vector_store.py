"""Module 5 — ChromaDB client wrapper.

Responsibility: persist RecognizedChunk vectors (plus doc_id, content_type,
and confidences as metadata) in ChromaDB at settings.chroma_dir, and run
similarity search. Nothing relational is stored here (that is SQLite).

Uses from schemas.py: RecognizedChunk, ContentType.
"""

from functools import lru_cache

import chromadb
from chromadb.api.models.Collection import Collection

from app.config import settings
from app.models.schemas import ContentType, RecognizedChunk
from app.modules.chunker import Location

COLLECTION = "chunks"


@lru_cache
def _collection(path: str) -> Collection:
    client = chromadb.PersistentClient(path=path)
    # embedding_function=None: vectors always come from embedder.py, so Chroma
    # must never download or run its own default model.
    return client.get_or_create_collection(
        COLLECTION, embedding_function=None, configuration={"hnsw": {"space": "cosine"}}
    )


def _get() -> Collection:
    return _collection(str(settings.chroma_dir))


def add_chunks(doc_id: str, chunks: list[RecognizedChunk], locations: dict[str, Location] | None = None) -> None:
    """Upsert embedded chunks for one document (same chunk_id replaces, not duplicates).

    locations maps chunk_id -> page and line range, stored as metadata for source citations.
    """
    if not chunks:
        return
    if any(c.vector is None for c in chunks):
        raise ValueError("every chunk needs a vector before storing (run embedder.embed_chunks)")

    metadatas = []
    for c in chunks:
        meta = {"doc_id": doc_id, "content_type": c.content_type.value, "raw_conf": c.raw_conf}
        if c.calibrated_conf is not None:  # Chroma metadata cannot hold None
            meta["calibrated_conf"] = c.calibrated_conf
        if locations and c.chunk_id in locations:
            loc = locations[c.chunk_id]
            meta.update(page=loc.page, first_line=loc.first_line, last_line=loc.last_line)
        metadatas.append(meta)

    _get().upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=[c.vector for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=metadatas,
    )


def _where(doc_ids: list[str] | None) -> dict | None:
    """Chroma metadata filter limiting a lookup to some documents; None/empty = all documents."""
    return {"doc_id": {"$in": list(doc_ids)}} if doc_ids else None


def search(
    query_vector: list[float], top_k: int = 5, doc_ids: list[str] | None = None
) -> list[tuple[RecognizedChunk, float]]:
    """Return the top_k nearest chunks with cosine similarity (1 = identical), best first.

    doc_ids: search only these documents (None = all). An empty list finds nothing.
    """
    if doc_ids is not None and not doc_ids:
        return []
    res = _get().query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where={"doc_id": {"$in": list(doc_ids)}} if doc_ids is not None else None,
        include=["documents", "metadatas", "distances", "embeddings"],
    )
    results = []
    for chunk_id, text, meta, dist, vec in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0], res["embeddings"][0]
    ):
        chunk = RecognizedChunk(
            chunk_id=chunk_id,
            text=text,
            content_type=ContentType(meta["content_type"]),
            raw_conf=meta["raw_conf"],
            calibrated_conf=meta.get("calibrated_conf"),
            vector=[float(x) for x in vec],
        )
        results.append((chunk, 1.0 - dist))
    return results


def get_chunks(chunk_ids: list[str]) -> dict[str, RecognizedChunk]:
    """Fetch stored chunks by id (without vectors). Unknown ids are left out."""
    if not chunk_ids:
        return {}
    res = _get().get(ids=chunk_ids, include=["documents", "metadatas"])
    return {
        chunk_id: RecognizedChunk(
            chunk_id=chunk_id,
            text=text,
            content_type=ContentType(meta["content_type"]),
            raw_conf=meta["raw_conf"],
            calibrated_conf=meta.get("calibrated_conf"),
        )
        for chunk_id, text, meta in zip(res["ids"], res["documents"], res["metadatas"])
    }


def get_locations(chunk_ids: list[str]) -> dict[str, dict[str, int | None]]:
    """Page and line range of each stored chunk: {"page", "first_line", "last_line"}.

    Chunks stored before locations were recorded are left out; chunks stored
    before line numbers were recorded have first_line/last_line None.
    """
    if not chunk_ids:
        return {}
    res = _get().get(ids=chunk_ids, include=["metadatas"])
    return {
        cid: {"page": meta["page"], "first_line": meta.get("first_line"), "last_line": meta.get("last_line")}
        for cid, meta in zip(res["ids"], res["metadatas"])
        if "page" in meta
    }


def get_texts(doc_ids: list[str] | None = None) -> list[str]:
    """Text of every stored chunk of the given documents (None or empty = all documents)."""
    return _get().get(where=_where(doc_ids), include=["documents"])["documents"]


def delete_document(doc_id: str) -> None:
    """Remove all chunks of one document."""
    _get().delete(where={"doc_id": doc_id})
