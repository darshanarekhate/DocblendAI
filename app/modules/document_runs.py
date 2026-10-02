"""Supports Module 1 — one document for both pages: a QA document and its Experience Center run.

Responsibility: keep the QA page (/) and the Experience Center (/studio) on the same documents.
An upload from either page is one ingest job (ocr_jobs.submit_ingest) that saves the file in
settings.upload_dir, extracts it once (document_extract.py), stores the Experience Center result,
indexes it for questions (qa_index.py) and records the Document row; document_runs links the two.

- library(): the documents both pages list (with processing / QA / refinement state)
- delete_document(): removes the QA side (chunks, row, file) and the run (result, versions, images)
- on_run_change(): registered with refinement.py, so any change to a run's text (LLM refinement
  retried, revert to the original, refined text used again, manual edit, restore) re-chunks and
  re-embeds the QA document; the next questions are answered from the new text.

Uses from schemas.py: Document (via DocumentORM rows).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.config import settings
from app.db.models import DocumentORM, DocumentRunORM, OCRRunORM
from app.models.schemas import FormatType
from app.modules import embedder, qa_index, refinement, spelling, vector_store

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def display_name(doc_id: str, file_path: str) -> str:
    """The upload's original name: stored files are "<doc_id>_<name>.<ext>"."""
    name = Path(file_path).name
    return name[len(doc_id) + 1:] if name.startswith(f"{doc_id}_") else name


def new_upload_path(filename: str) -> tuple[str, Path]:
    """(doc_id, where to store the upload): "<doc_id>_<safe original name>.<ext>" in settings.upload_dir.

    The original name is kept in the stored path so both pages can show it (the synopsis
    Document entity has no name field).
    """
    doc_id = uuid.uuid4().hex
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).stem)[:80] or "document"
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    return doc_id, settings.upload_dir / f"{doc_id}_{safe_name}{Path(filename).suffix.lower()}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def find_duplicate(db, path: Path, format_hint: FormatType | None) -> DocumentORM | None:
    """An earlier Document with byte-identical content (and a compatible format), if any.

    A format_hint that differs from the stored format is a request to re-read the
    file another way, so it is not treated as a duplicate.
    """
    size, digest = path.stat().st_size, None
    for row in db.query(DocumentORM).all():
        other = Path(row.file_path)
        if other == path or not other.is_file() or other.stat().st_size != size:
            continue
        if format_hint is not None and row.format_type != format_hint:
            continue
        digest = digest or _sha256(path)
        if _sha256(other) == digest:
            return row
    return None


def run_for_doc(db, doc_id: str) -> str | None:
    row = db.get(DocumentRunORM, doc_id)
    return row.run_id if row else None


def doc_for_run(db, run_id: str) -> str | None:
    return db.scalar(select(DocumentRunORM.doc_id).where(DocumentRunORM.run_id == run_id))


def link(db, doc_id: str, run_id: str) -> None:
    """Point a document at its run (one run per document: a new run replaces an older link)."""
    row = db.get(DocumentRunORM, doc_id)
    if row is None:
        db.add(DocumentRunORM(doc_id=doc_id, run_id=run_id))
    else:
        row.run_id = run_id


def _set_qa_state(run_id: str, qa: dict[str, Any]) -> None:
    from app.modules.ocr_jobs import job_manager

    with job_manager._session() as db:
        row = db.get(OCRRunORM, run_id)
        if row is None:
            return
        result = json.loads(row.result_json or "{}")
        result["qa"] = qa
        row.result_json = json.dumps(result)
        db.commit()


def reindex(run_id: str, result: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the QA chunks of a run's document from its current text; returns the QA state."""
    doc_id = result.get("doc_id")
    if not doc_id:
        return {"status": "none"}
    try:
        chunks = qa_index.index_document(doc_id, result)
        qa = {"status": "ready", "chunks": len(chunks), "message": None, "indexed_at": _now()}
    except (embedder.EmbeddingError, qa_index.NoTextError) as exc:
        logger.error("re-indexing %s failed: %s", doc_id, exc)
        qa = {"status": "stale", "chunks": None, "indexed_at": _now(),
              "message": f"Questions still use the previous text: {exc}"}
    _set_qa_state(run_id, qa)
    return qa


def on_run_change(run_id: str, result: dict[str, Any], reason: str) -> None:
    if result.get("doc_id"):
        logger.info("run %s changed (%s): re-indexing document %s", run_id, reason, result["doc_id"])
        reindex(run_id, result)


refinement.add_change_hook(on_run_change)


def delete_document(doc_id: str) -> bool:
    """Delete the QA document (chunks, row, file) and its Experience Center run. False if unknown."""
    from app.modules.ocr_jobs import job_manager

    with job_manager._session() as db:
        doc = db.get(DocumentORM, doc_id)
        run_id = run_for_doc(db, doc_id)
        if doc is None and run_id is None:
            return False
        file_path = Path(doc.file_path) if doc else None
        if doc is not None:
            db.delete(doc)
        link_row = db.get(DocumentRunORM, doc_id)
        if link_row is not None:
            db.delete(link_row)
        db.commit()
    vector_store.delete_document(doc_id)
    spelling.invalidate()
    if run_id:
        job_manager.delete(run_id, cascade=False)
    if file_path is not None:
        file_path.unlink(missing_ok=True)
    logger.info("Deleted document %s (run %s)", doc_id, run_id)
    return True


def _run_item(run: OCRRunORM | None) -> dict[str, Any]:
    if run is None:
        return {}
    result = json.loads(run.result_json or "{}")
    return {
        "run_id": run.id, "pipeline": run.pipeline, "status": run.status, "created_at": run.created_at,
        "page_count": run.page_count, "mean_confidence": run.mean_confidence,
        "format_type": result.get("format_type"), "engine": result.get("engine"),
        "qa": result.get("qa"), "refinement": result.get("refinement"), "error": result.get("error"),
        "_text": run.full_text or "",
    }


def library(q: str | None = None) -> list[dict[str, Any]]:
    """Documents for both pages, newest first.

    Each: {doc_id, run_id, name, format_type, page_count, status: processing|ready|error,
    progress, message, error, qa, refinement (refined | unavailable | reverted | off | empty, with
    a message), refine (a running retry), pipeline, engine, created_at, mean_confidence, qa_ready}. QA documents uploaded before the
    Experience Center existed have run_id null; Experience Center runs from before documents
    were shared have doc_id null.
    """
    from app.modules.ocr_jobs import job_manager

    items: list[dict[str, Any]] = []
    with job_manager._session() as db:
        links = {row.doc_id: row.run_id for row in db.scalars(select(DocumentRunORM)).all()}
        linked_runs = set(links.values())
        for doc in db.scalars(select(DocumentORM)).all():
            run = db.get(OCRRunORM, links[doc.doc_id]) if doc.doc_id in links else None
            item = {
                "doc_id": doc.doc_id, "name": display_name(doc.doc_id, doc.file_path),
                "format_type": doc.format_type.value, "page_count": doc.page_count, "qa_ready": True,
                "status": "ready", "progress": 1.0, "message": None, "run_id": None, "created_at": None,
            }
            item.update({k: v for k, v in _run_item(run).items() if v is not None or k not in item})
            item["status"] = "ready"
            job = job_manager.get_job(run.id) if run is not None else None
            if job is not None and job.status in ("queued", "running"):  # an old upload being re-read
                item.update({"status": "processing", "progress": job.progress, "message": job.message})
            if item.get("created_at") is None:
                path = Path(doc.file_path)
                item["created_at"] = (
                    datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
                    if path.exists() else ""
                )
            items.append(item)
        known_docs = {i["doc_id"] for i in items}
        for run in db.scalars(select(OCRRunORM).order_by(OCRRunORM.created_at.desc())).all():
            if run.id in linked_runs:
                continue
            result = json.loads(run.result_json or "{}")
            doc_id = result.get("doc_id")
            if doc_id in known_docs:
                continue
            item = {"doc_id": doc_id, "name": run.filename, "qa_ready": False, "progress": None, "message": None}
            item.update(_run_item(run))
            job = job_manager.get_job(run.id)
            if job is not None and job.status in ("queued", "running"):
                item.update({"status": "processing", "progress": job.progress, "message": job.message})
            elif run.status == "done":
                item["status"] = "ready"  # an Experience Center run without a QA document (older run)
            else:
                item["status"] = "error"
                item["error"] = item.get("error") or result.get("error") or "This run did not finish."
            items.append(item)
    for item in items:
        run_id = item.get("run_id")
        active = job_manager.active_refine(run_id) if run_id else None
        item["refine"] = job_manager.get_result(active.id) if active is not None else None
        text = item.pop("_text", "")
        item["_search"] = f"{item.get('name') or ''}\n{text}".lower()
    if q:
        needle = q.lower()
        items = [i for i in items if needle in i["_search"]]
    for item in items:
        item.pop("_search", None)
    items.sort(key=lambda i: i.get("created_at") or "", reverse=True)
    return items
