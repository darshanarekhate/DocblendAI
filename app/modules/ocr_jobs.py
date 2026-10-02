"""Experience Center job manager: queued OCR/parse/calibration jobs with progress, history in SQLite.

Supports Modules 2-3 for the PaddleOCR Experience Center (docs/experience_center_contract.md §3-§4).
One worker thread runs jobs in submission order: the Paddle models are shared and CPU-bound, so
running two at once would only make both slower and double the memory. Live progress is kept in
memory; finished results (the contract §4 object) are stored in the `ocr_runs` table
(`OCRRunORM`), page images under settings.paddle_runs_dir/<id>/.

Calibration jobs are in memory only (their output, report.json, is written by run_calibration).

Every upload (from either page) is one run: "Extracting page 2 of 5…" (document_extract.py:
exact text for typed files, PaddleOCR / PP-StructureV3 for scans, TrOCR in Paddle's line boxes
for handwriting) -> "Refining with Gemini…" (refinement.py: every line, changes applied at once,
the original extraction kept as a version) -> "Indexing for questions…" (qa_index.py) -> "Ready".
If Gemini is unavailable the unrefined text is used and the run says so; QA is never blocked.
With a doc_id the file stays in settings.upload_dir as the QA document's file, and the run
records the Document row and links it (document_runs.py). A "refine" job retries the refinement
of a run's original extraction (one at a time per run).
"""

from __future__ import annotations

import json
import logging
import queue
import shutil
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.db.models import OCRRunORM
from app.models.schemas import FormatType
from app.modules import document_extract, refinement
from app.modules import document_runs  # noqa: F401  (registers the QA re-index change hook)
from app.modules import paddleocr_service as svc

logger = logging.getLogger(__name__)

INTERRUPTED_MESSAGE = "Interrupted: the server stopped before this run finished. Run it again."
MAX_EDIT_TEXT = 10_000


class JobBusyError(RuntimeError):
    """The job is running and cannot be changed or deleted yet."""


class EditError(ValueError):
    """A PATCH /lines edit refers to a page/line that does not exist, or is malformed."""


@dataclass
class Job:
    id: str
    kind: str  # "run" (OCR/parse) | "calibration"
    pipeline: str
    language: str
    filename: str
    created_at: str
    status: str = "queued"
    progress: float = 0.0
    message: str = "Queued"
    error: str | None = None
    unavailable: bool = False  # failed because the pipeline cannot run here (-> HTTP 503)
    cancelled: bool = False
    processing_time_ms: float | None = None
    report: dict[str, Any] | None = None  # calibration jobs
    run_id: str | None = None  # refine jobs: the run being refined
    doc_id: str | None = None  # ingest runs: the QA document being created (or re-read)
    existing_doc: bool = False  # ingest runs: the Document row already exists (re-reading an old upload)
    keep_failed: bool = True  # ingest runs: keep a failed run in the history (UI) or remove all traces (API)
    http_status: int | None = None  # failed ingest: the HTTP status a waiting /upload answers with
    done: threading.Event = field(default_factory=threading.Event)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def page_image_path(run_id: str, index: int) -> Path:
    return settings.paddle_runs_dir / run_id / f"page-{index}.png"


def _image_url(run_id: str) -> Callable[[int], str]:
    return lambda index: f"/api/results/{run_id}/pages/{index}/image"


def _extracting(message: str) -> str:
    """Extraction progress for the UI: "Page 2 of 5" -> "Extracting page 2 of 5…"."""
    if message.startswith("Page "):
        return f"Extracting page {message[5:]}…"
    return message if message.endswith("…") else f"{message}…"


def _hint(value: str | None) -> FormatType | None:
    return FormatType(value) if value else None


class JobManager:
    def __init__(self, session_factory: sessionmaker[Session] | Callable[[], Session] | None = None) -> None:
        self._session_factory = session_factory
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[tuple[Job, Callable[[Job], None]]] = queue.Queue()
        self._worker: threading.Thread | None = None

    # -- plumbing ------------------------------------------------------------------------------

    def configure(self, session_factory: sessionmaker[Session] | Callable[[], Session] | None) -> None:
        """Point history storage at another database (tests use their temp SQLite)."""
        self._session_factory = session_factory

    @contextmanager
    def _session(self) -> Iterator[Session]:
        factory = self._session_factory
        if factory is None:
            from app.db.database import SessionLocal

            factory = SessionLocal
        db = factory()
        try:
            yield db
        finally:
            db.close()

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._work, name="paddle-jobs", daemon=True)
                self._worker.start()

    def _work(self) -> None:
        while True:
            job, fn = self._queue.get()
            try:
                fn(job)  # each job fn checks job.cancelled itself (it owns temp files to clean)
            except Exception:  # fn records its own errors; this is a last-resort guard
                logger.exception("job %s crashed", job.id)
                job.status, job.error = "error", job.error or "internal error"
            finally:
                job.done.set()
                self._queue.task_done()

    def _enqueue(self, job: Job, fn: Callable[[Job], None]) -> Job:
        with self._lock:
            self._jobs[job.id] = job
        self._ensure_worker()
        self._queue.put((job, fn))
        return job

    def get_job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def wait(self, job: Job, timeout: float | None = None) -> bool:
        return job.done.wait(timeout)

    def reset(self, timeout: float = 30.0) -> None:
        """Tests: wait for queued work to finish, then forget in-memory jobs."""
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.01)
        with self._lock:
            self._jobs.clear()

    # -- OCR / parse runs ----------------------------------------------------------------------

    def submit_run(
        self, upload_path: Path, filename: str, pipeline: str, language: str, review_threshold: float,
        preprocess: dict[str, bool] | None = None, format_hint: str | None = None,
        doc_id: str | None = None, existing_doc: bool = False, keep_failed: bool = True,
    ) -> Job:
        """Queue one document. Takes ownership of upload_path.

        With doc_id it is the QA document's stored file: the run also indexes it for questions and
        creates (or, existing_doc, updates) the Document row. Without, it is deleted when read.
        """
        job = Job(
            id=uuid.uuid4().hex, kind="run", pipeline=pipeline, language=language,
            filename=filename, created_at=_now(), doc_id=doc_id, existing_doc=existing_doc,
            keep_failed=keep_failed,
        )
        with self._session() as db:
            db.add(OCRRunORM(
                id=job.id, filename=filename, pipeline=pipeline, language=language, status="queued",
                created_at=job.created_at, full_text="", result_json=json.dumps(self._base(job)),
            ))
            db.commit()
        return self._enqueue(
            job, lambda j: self._run_document(j, upload_path, review_threshold, preprocess, format_hint)
        )

    def _base(self, job: Job) -> dict[str, Any]:
        return {
            "id": job.id, "status": job.status, "progress": round(job.progress, 4), "message": job.message,
            "error": job.error, "pipeline": job.pipeline, "language": job.language,
            "filename": job.filename, "created_at": job.created_at,
            "processing_time_ms": job.processing_time_ms, "calibration": None, "summary": None,
            "markdown": "", "pages": [], "doc_id": job.doc_id,
        }

    def _set_status(self, job_id: str, status: str) -> None:
        with self._session() as db:
            row = db.get(OCRRunORM, job_id)
            if row is not None:
                row.status = status
                db.commit()

    def _run_document(
        self, job: Job, upload_path: Path, review_threshold: float, preprocess: dict[str, bool] | None = None,
        format_hint: str | None = None,
    ) -> None:
        if job.cancelled:  # deleted while queued
            upload_path.unlink(missing_ok=True)
            return
        started = time.perf_counter()
        job.status, job.message = "running", "Starting"
        self._set_status(job.id, "running")
        pages_dir = settings.paddle_runs_dir / job.id
        logger.info(
            "job %s started: %s %s (%s)", job.id, job.pipeline, job.filename, job.language,
            extra={"job_id": job.id, "pipeline": job.pipeline, "language": job.language},
        )

        def progress(fraction: float, message: str) -> None:
            job.progress = max(job.progress, min(0.6, 0.6 * max(0.0, fraction)))
            job.message = _extracting(message)

        from app.modules import embedder, qa_index

        result: dict[str, Any] | None = None
        try:
            pages_dir.mkdir(parents=True, exist_ok=True)
            extraction = document_extract.extract(
                upload_path, pages_dir=pages_dir, format_hint=_hint(format_hint), pipeline=job.pipeline,
                lang=job.language, image_url=_image_url(job.id), progress=progress, preprocess=preprocess,
            )
            result = self._finished(job, extraction, review_threshold, preprocess)
            with self._session() as db:
                original = refinement.record(db, job.id, "extraction", "Original extraction", result)
                result = self._refine(db, job, result, original.id)
                db.commit()
            if job.doc_id:
                job.message, job.progress = "Indexing for questions…", 0.95
                chunks = qa_index.index_document(job.doc_id, result)
                result["qa"] = {"status": "ready", "chunks": len(chunks), "message": None, "indexed_at": _now()}
                self._save_document(job, upload_path, extraction, result)
            job.processing_time_ms = round((time.perf_counter() - started) * 1000, 1)
            result["processing_time_ms"] = job.processing_time_ms
            job.status, job.progress, job.message = "done", 1.0, " ".join(["Ready"] + extraction.notes)
            result["message"] = job.message
        except svc.PaddleOCRUnavailableError as exc:
            job.unavailable, job.error, job.http_status = True, str(exc), 503
        except (svc.UnreadableInputError, document_extract.ExtractionError) as exc:
            job.error, job.http_status = f"Unreadable file: {exc}", 422
            job.message = f"File is not a readable {upload_path.suffix.lower()} file (damaged, or not what its extension says)"
        except qa_index.NoTextError as exc:
            job.error, job.http_status = str(exc), 422
        except embedder.EmbeddingError as exc:
            logger.error("Embedding failed for %s: %s", job.doc_id, exc)
            job.error, job.http_status = str(exc), getattr(exc, "status", 502)
        except Exception as exc:  # inference failures: report, keep the worker alive
            if type(exc).__name__ in ("OCRUnavailableError", "HTRUnavailableError"):
                job.unavailable, job.error, job.http_status = True, str(exc), 503
            else:
                logger.exception("job %s failed", job.id)
                job.error, job.http_status = f"{type(exc).__name__}: {exc}", 500
        finally:
            if not job.doc_id:
                upload_path.unlink(missing_ok=True)

        if job.error:
            result = None
            self._clean_failed_ingest(job, upload_path)
        if result is None:
            job.processing_time_ms = round((time.perf_counter() - started) * 1000, 1)
            job.status = "error"
            job.message = job.message if job.http_status == 422 and job.message.startswith("File is") else "Failed"
            shutil.rmtree(pages_dir, ignore_errors=True)
            result = self._base(job)
        if result["status"] == "error" and job.doc_id and not job.keep_failed:
            with self._session() as db:  # an API upload that failed leaves nothing behind
                row = db.get(OCRRunORM, job.id)
                if row is not None:
                    db.delete(row)
                    db.commit()
        else:
            self._store(result)
        pages_done = (result.get("summary") or {}).get("page_count", 0)
        logger.info(
            "job %s %s in %.0f ms (%s, %s page(s))", job.id, job.status, job.processing_time_ms,
            job.pipeline, pages_done,
            extra={"job_id": job.id, "pipeline": job.pipeline, "status": job.status,
                   "duration_ms": job.processing_time_ms, "page_count": pages_done},
        )

    def _save_document(
        self, job: Job, source: Path, extraction: document_extract.Extraction, result: dict[str, Any]
    ) -> None:
        """Create (or update) the QA Document row and link it to this run."""
        from app.db.models import DocumentORM
        from app.modules import document_runs

        with self._session() as db:
            row = db.get(DocumentORM, job.doc_id)
            if row is None:
                db.add(DocumentORM(
                    doc_id=job.doc_id, file_path=str(source), format_type=extraction.format_type,
                    page_count=extraction.page_count,
                ))
            else:
                row.format_type, row.page_count = extraction.format_type, extraction.page_count
            document_runs.link(db, job.doc_id, job.id)
            db.commit()

    def _clean_failed_ingest(self, job: Job, upload_path: Path) -> None:
        """A failed ingest of a new upload leaves no file, chunks or Document row behind."""
        with self._session() as db:
            refinement.delete_versions(db, job.id)
            db.commit()
        if not job.doc_id or job.existing_doc:
            return
        from app.modules import vector_store

        try:
            vector_store.delete_document(job.doc_id)
        except Exception:
            logger.exception("could not remove the chunks of failed upload %s", job.doc_id)
        upload_path.unlink(missing_ok=True)

    def wait_until_ready(self, job: Job, timeout: float | None = None) -> bool:
        """For /upload: the run is done (text indexed for questions) or failed."""
        return job.done.wait(timeout)

    def _refine(self, db: Session, job: Job, result: dict[str, Any], original_version: str) -> dict[str, Any]:
        """Refine the extracted text with Gemini and use it; keep the unrefined text if Gemini is unavailable."""
        from app.plugins import PluginError

        if not settings.llm_refine:
            result["refinement"] = {"status": "off", "message": "LLM refinement is turned off (LLM_REFINE=false)."}
            return result
        job.message, job.progress = "Refining with Gemini…", 0.62

        def parts(done: int, total: int) -> None:
            job.progress = 0.62 + 0.3 * done / max(1, total)
            if total > 1 and done < total:
                job.message = f"Refining with Gemini… (part {done + 1} of {total})"

        try:
            return refinement.refine_document(db, job.id, result, original_version, parts)
        except PluginError as exc:  # no key, quota, overload, unusable reply: QA goes on unrefined
            logger.warning("run %s not refined: %s", job.id, exc)
            return refinement.unavailable(result, str(exc))

    def _finished(
        self, job: Job, extraction: document_extract.Extraction, review_threshold: float,
        preprocess: dict[str, bool] | None,
    ) -> dict[str, Any]:
        """Contract result object for a finished extraction."""
        pages = extraction.pages
        calibration = svc.review_flags(pages, review_threshold)
        result = self._base(job)
        result.update({
            "status": "done", "progress": 1.0, "message": "Ready",
            "calibration": calibration, "summary": svc.summarize(pages),
            "markdown": svc.join_markdown(pages), "pages": pages,
            "preprocess": sorted(k for k, v in (preprocess or {}).items() if v),
            "format_type": extraction.format_type.value, "engine": extraction.engine,
            "notes": extraction.notes,
        })
        return result

    def _store(self, result: dict[str, Any]) -> None:
        summary = result.get("summary") or {}
        with self._session() as db:
            row = db.get(OCRRunORM, result["id"])
            if row is None:  # deleted while running? keep nothing
                return
            row.status = result["status"]
            row.page_count = summary.get("page_count")
            row.mean_confidence = summary.get("mean_calibrated_confidence")
            row.processing_time_ms = result.get("processing_time_ms")
            row.full_text = svc.full_text(result.get("pages") or [])
            row.result_json = json.dumps(result)
            db.commit()

    def get_result(self, job_id: str) -> dict[str, Any] | None:
        """Contract §4 object (or a calibration / refine job object); None if unknown."""
        job = self._jobs.get(job_id)
        if job is not None and job.kind == "calibration":
            return self._calibration_view(job)
        if job is not None and job.kind == "refine":
            return self._refine_view(job)
        if job is not None and job.status in ("queued", "running"):
            return self._base(job)
        with self._session() as db:
            row = db.get(OCRRunORM, job_id)
            if row is None:
                return None
            result = json.loads(row.result_json or "{}")
        if result.get("status") in ("queued", "running") and job is None:
            result.update({"status": "error", "error": INTERRUPTED_MESSAGE, "message": "Failed"})
        active = self.active_refine(job_id)
        result["active_job"] = self._refine_view(active) if active is not None else None
        return result

    def history(self, q: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        stmt = select(OCRRunORM).order_by(OCRRunORM.created_at.desc(), OCRRunORM.id).limit(limit)
        if q:
            stmt = stmt.where(or_(
                OCRRunORM.filename.contains(q, autoescape=True),
                OCRRunORM.full_text.contains(q, autoescape=True),
            ))
        with self._session() as db:
            rows = db.scalars(stmt).all()
            items = []
            for row in rows:
                status = row.status
                if status in ("queued", "running") and row.id not in self._jobs:
                    status = "error"
                items.append({
                    "id": row.id, "filename": row.filename, "pipeline": row.pipeline,
                    "language": row.language, "status": status, "created_at": row.created_at,
                    "page_count": row.page_count, "mean_confidence": row.mean_confidence,
                    "processing_time_ms": row.processing_time_ms,
                })
        return items

    def delete(self, job_id: str, cascade: bool = True) -> bool:
        """Delete a run (result, versions, images). cascade: also its QA document, if linked."""
        if cascade:
            from app.modules import document_runs

            with self._session() as db:
                doc_id = document_runs.doc_for_run(db, job_id)
            if doc_id is not None:
                if self.active_refine(job_id) is not None:
                    raise JobBusyError("This document is being refined; delete it when that has finished.")
                return document_runs.delete_document(doc_id)
        job = self._jobs.get(job_id)
        if job is not None and job.status == "running":
            raise JobBusyError("This run is still in progress; delete it when it has finished.")
        if self.active_refine(job_id) is not None:
            raise JobBusyError("This document is being refined; delete it when that has finished.")
        if job is not None and job.status == "queued":
            job.cancelled = True
        with self._session() as db:
            row = db.get(OCRRunORM, job_id)
            if row is None and job is None:
                return False
            if row is not None:
                db.delete(row)
            refinement.delete_versions(db, job_id)
            db.commit()
        with self._lock:
            self._jobs.pop(job_id, None)
        shutil.rmtree(settings.paddle_runs_dir / job_id, ignore_errors=True)
        return True

    def edit_lines(self, job_id: str, edits: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Apply human corrections [{page, line, text}] to a finished run; returns the updated result."""
        job = self._jobs.get(job_id)
        if job is not None and job.status in ("queued", "running"):
            raise JobBusyError("This run has not finished yet.")
        if self.active_refine(job_id) is not None:
            raise JobBusyError("This document is being re-extracted and refined; edit it when that has finished.")
        with self._session() as db:
            row = db.get(OCRRunORM, job_id)
            if row is None:
                return None
            result = json.loads(row.result_json or "{}")
            if result.get("status") != "done":
                raise EditError("Only finished runs can be edited.")
            refinement.ensure_initial(db, job_id, result)
            pages = result.get("pages") or []
            for edit in edits:
                page_i, line_i, text = edit.get("page"), edit.get("line"), edit.get("text")
                if not isinstance(page_i, int) or not isinstance(line_i, int) or not isinstance(text, str):
                    raise EditError("Each edit needs integer 'page' and 'line' and a string 'text'.")
                if len(text) > MAX_EDIT_TEXT:
                    raise EditError(f"Edited text is longer than {MAX_EDIT_TEXT} characters.")
                if not 0 <= page_i < len(pages) or not 0 <= line_i < len(pages[page_i]["lines"]):
                    raise EditError(f"No line {line_i} on page {page_i}.")
                line = pages[page_i]["lines"][line_i]
                line.setdefault("ocr_text", line["text"])
                refinement.set_line_text(pages[page_i], line, text)
                line["edited"] = True
                line["source"] = "human"
                line["needs_review"] = False  # a human has checked it
                for stale in ("llm_flag", "llm_diff", "llm_status"):  # the LLM's diff no longer describes this text
                    line.pop(stale, None)
            refinement.refresh(result)
            refinement.save_current(db, row, result)
            n = len(edits)
            refinement.record(db, job_id, "edit", f"Manual edit ({n} line{'s' if n != 1 else ''})", result)
            db.commit()
        refinement.notify_change(job_id, result, "manual edit")
        return self.get_result(job_id)

    # -- refine with LLM -----------------------------------------------------------------------

    def active_refine(self, run_id: str) -> Job | None:
        """The queued or running refine job of a run, if any."""
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            if job.kind == "refine" and job.run_id == run_id and job.status in ("queued", "running"):
                return job
        return None

    def submit_refine(self, run_id: str) -> Job:
        """Queue a retry of the LLM refinement of a run's original extraction (JobBusyError if one runs)."""
        with self._session() as db:
            row = db.get(OCRRunORM, run_id)
            if row is None:
                raise refinement.RefinementError(f"No result with id {run_id}")
            result = json.loads(row.result_json or "{}")
        if result.get("status") != "done":
            raise JobBusyError("This document has not finished extracting yet.")
        with self._lock:
            for other in self._jobs.values():
                if other.kind == "refine" and other.run_id == run_id and other.status in ("queued", "running"):
                    raise JobBusyError("This document is already being refined.")
            job = Job(
                id=uuid.uuid4().hex, kind="refine", pipeline="refine", language=result.get("language") or "en",
                filename=result.get("filename") or "", created_at=_now(), run_id=run_id,
            )
            self._jobs[job.id] = job
        self._ensure_worker()
        self._queue.put((job, self._run_refine))
        return job

    def _run_refine(self, job: Job) -> None:
        from app.plugins import PluginError

        started = time.perf_counter()
        run_id = job.run_id
        job.status, job.progress, job.message = "running", 0.05, "Refining with Gemini…"

        def parts(done: int, total: int) -> None:
            job.progress = 0.05 + 0.85 * done / max(1, total)
            if total > 1 and done < total:
                job.message = f"Refining with Gemini… (part {done + 1} of {total})"

        result = None
        try:
            with self._session() as db:
                row, current = refinement.load_current(db, run_id)
                refinement.ensure_initial(db, run_id, current)
                original = refinement.original_version(db, run_id)
                base = json.loads(original.data_json)
                base["status"] = "done"
                try:
                    result = refinement.refine_document(db, run_id, base, original.id, parts)
                except PluginError as exc:
                    current = refinement.unavailable(current, str(exc))
                    refinement.save_current(db, row, current)
                    db.commit()
                    job.error = current["refinement"]["message"]
                else:
                    result["qa"] = current.get("qa")
                    refinement.save_current(db, row, result)
                    db.commit()
            if result is not None:
                job.message, job.progress = "Indexing for questions…", 0.95
                refinement.notify_change(run_id, result, "refined again")
                job.status, job.progress, job.message = "done", 1.0, "Ready"
        except refinement.RefinementError as exc:
            job.error = str(exc)
        except Exception as exc:
            logger.exception("refine job %s failed", job.id)
            job.error = f"{type(exc).__name__}: {exc}"
        job.processing_time_ms = round((time.perf_counter() - started) * 1000, 1)
        if job.error:
            job.status, job.message = "error", "Failed"
        logger.info(
            "refine job %s (run %s) %s in %.0f ms", job.id, run_id, job.status, job.processing_time_ms,
            extra={"job_id": job.id, "run_id": run_id, "pipeline": "refine", "status": job.status,
                   "duration_ms": job.processing_time_ms},
        )

    def _refine_view(self, job: Job) -> dict[str, Any]:
        return {
            "id": job.id, "status": job.status, "progress": round(job.progress, 4), "message": job.message,
            "error": job.error, "pipeline": "refine", "run_id": job.run_id,
            "language": job.language, "filename": job.filename, "created_at": job.created_at,
            "processing_time_ms": job.processing_time_ms,
        }

    # -- calibration ---------------------------------------------------------------------------

    def submit_calibration(
        self, dataset: Path, match: str, cer_threshold: float, cleanup: Path | None = None,
        label: str = "calibration",
    ) -> Job:
        """Queue run_calibration on a dataset folder; cleanup (an extracted zip) is removed after."""
        job = Job(
            id=uuid.uuid4().hex, kind="calibration", pipeline="calibration", language="en",
            filename=label, created_at=_now(),
        )
        return self._enqueue(job, lambda j: self._run_calibration(j, dataset, match, cer_threshold, cleanup))

    def _run_calibration(
        self, job: Job, dataset: Path, match: str, cer_threshold: float, cleanup: Path | None
    ) -> None:
        from app.calibration import calibrate as calibrate_module

        if job.cancelled:
            if cleanup is not None:
                shutil.rmtree(cleanup, ignore_errors=True)
            return
        started = time.perf_counter()
        job.status, job.message = "running", "Starting calibration"
        logger.info("calibration job %s started on %s (match=%s)", job.id, dataset, match)

        def progress(done: int, total: int) -> None:
            job.progress = min(0.99, done / total) if total else 0.0
            job.message = f"Image {done} of {total}"

        try:
            job.report = calibrate_module.run_calibration(
                dataset, ocr_fn=svc.paddleocr_service.recognize_lines, match=match,
                cer_threshold=cer_threshold, progress=progress,
            )
            job.status, job.progress, job.message = "done", 1.0, "Done"
        except svc.PaddleOCRUnavailableError as exc:
            job.unavailable, job.error = True, str(exc)
        except Exception as exc:
            logger.exception("calibration job %s failed", job.id)
            job.error = f"{type(exc).__name__}: {exc}"
        finally:
            if cleanup is not None:
                shutil.rmtree(cleanup, ignore_errors=True)
            job.processing_time_ms = round((time.perf_counter() - started) * 1000, 1)
        if job.error:
            job.status, job.message = "error", "Failed"
        logger.info(
            "calibration job %s %s in %.0f ms", job.id, job.status, job.processing_time_ms,
            extra={"job_id": job.id, "pipeline": "calibration", "status": job.status,
                   "duration_ms": job.processing_time_ms},
        )

    def _calibration_view(self, job: Job) -> dict[str, Any]:
        return {
            "id": job.id, "status": job.status, "progress": round(job.progress, 4), "message": job.message,
            "error": job.error, "pipeline": "calibration", "language": job.language,
            "filename": job.filename, "created_at": job.created_at,
            "processing_time_ms": job.processing_time_ms, "report": job.report,
        }


job_manager = JobManager()
