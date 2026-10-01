"""Experience Center job manager: queued OCR/parse/calibration jobs with progress, history in SQLite.

Supports Modules 2-3 for the PaddleOCR Experience Center (docs/experience_center_contract.md §3-§4).
One worker thread runs jobs in submission order: the Paddle models are shared and CPU-bound, so
running two at once would only make both slower and double the memory. Live progress is kept in
memory; finished results (the contract §4 object) are stored in the `ocr_runs` table
(`OCRRunORM`), page images under settings.paddle_runs_dir/<id>/.

Calibration jobs are in memory only (their output, report.json, is written by run_calibration).
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
    done: threading.Event = field(default_factory=threading.Event)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def page_image_path(run_id: str, index: int) -> Path:
    return settings.paddle_runs_dir / run_id / f"page-{index}.png"


def _image_url(run_id: str) -> Callable[[int], str]:
    return lambda index: f"/api/results/{run_id}/pages/{index}/image"


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
        preprocess: dict[str, bool] | None = None,
    ) -> Job:
        """Queue one document. Takes ownership of upload_path (deleted when the job ends)."""
        job = Job(
            id=uuid.uuid4().hex, kind="run", pipeline=pipeline, language=language,
            filename=filename, created_at=_now(),
        )
        with self._session() as db:
            db.add(OCRRunORM(
                id=job.id, filename=filename, pipeline=pipeline, language=language, status="queued",
                created_at=job.created_at, full_text="", result_json=json.dumps(self._base(job)),
            ))
            db.commit()
        return self._enqueue(job, lambda j: self._run_document(j, upload_path, review_threshold, preprocess))

    def _base(self, job: Job) -> dict[str, Any]:
        return {
            "id": job.id, "status": job.status, "progress": round(job.progress, 4), "message": job.message,
            "error": job.error, "pipeline": job.pipeline, "language": job.language,
            "filename": job.filename, "created_at": job.created_at,
            "processing_time_ms": job.processing_time_ms, "calibration": None, "summary": None,
            "markdown": "", "pages": [],
        }

    def _set_status(self, job_id: str, status: str) -> None:
        with self._session() as db:
            row = db.get(OCRRunORM, job_id)
            if row is not None:
                row.status = status
                db.commit()

    def _run_document(
        self, job: Job, upload_path: Path, review_threshold: float, preprocess: dict[str, bool] | None = None
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
            job.progress = max(job.progress, min(0.99, max(0.0, fraction)))
            job.message = message

        result: dict[str, Any] | None = None
        try:
            pages, note = svc.paddleocr_service.process(
                upload_path, job.pipeline, job.language, pages_dir, _image_url(job.id), progress, preprocess
            )
            calibration = svc.apply_calibration(pages, review_threshold)
            job.processing_time_ms = round((time.perf_counter() - started) * 1000, 1)
            job.status, job.progress, job.message = "done", 1.0, note or "Done"
            result = self._base(job)
            result.update({
                "calibration": calibration, "summary": svc.summarize(pages),
                "markdown": svc.join_markdown(pages), "pages": pages,
                "preprocess": sorted(k for k, v in (preprocess or {}).items() if v),
            })
        except svc.PaddleOCRUnavailableError as exc:
            job.unavailable = True
            job.error = str(exc)
        except svc.UnreadableInputError as exc:
            job.error = f"Unreadable file: {exc}"
        except Exception as exc:  # inference failures: report, keep the worker alive
            logger.exception("job %s failed", job.id)
            job.error = f"{type(exc).__name__}: {exc}"
        finally:
            upload_path.unlink(missing_ok=True)

        if result is None:
            job.processing_time_ms = round((time.perf_counter() - started) * 1000, 1)
            job.status, job.message = "error", "Failed"
            shutil.rmtree(pages_dir, ignore_errors=True)
            result = self._base(job)
        self._store(result)
        pages_done = (result.get("summary") or {}).get("page_count", 0)
        logger.info(
            "job %s %s in %.0f ms (%s, %s page(s))", job.id, job.status, job.processing_time_ms,
            job.pipeline, pages_done,
            extra={"job_id": job.id, "pipeline": job.pipeline, "status": job.status,
                   "duration_ms": job.processing_time_ms, "page_count": pages_done},
        )

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
        """Contract §4 object (or a calibration job object); None if unknown."""
        job = self._jobs.get(job_id)
        if job is not None and job.kind == "calibration":
            return self._calibration_view(job)
        if job is not None and job.status in ("queued", "running"):
            return self._base(job)
        with self._session() as db:
            row = db.get(OCRRunORM, job_id)
            if row is None:
                return None
            result = json.loads(row.result_json or "{}")
        if result.get("status") in ("queued", "running") and job is None:
            result.update({"status": "error", "error": INTERRUPTED_MESSAGE, "message": "Failed"})
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

    def delete(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is not None and job.status == "running":
            raise JobBusyError("This run is still in progress; delete it when it has finished.")
        if job is not None and job.status == "queued":
            job.cancelled = True
        with self._session() as db:
            row = db.get(OCRRunORM, job_id)
            if row is None and job is None:
                return False
            if row is not None:
                db.delete(row)
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
        with self._session() as db:
            row = db.get(OCRRunORM, job_id)
            if row is None:
                return None
            result = json.loads(row.result_json or "{}")
            if result.get("status") != "done":
                raise EditError("Only finished runs can be edited.")
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
                line["text"] = text
                line["edited"] = True
                line["needs_review"] = False  # a human has checked it
            if result.get("pipeline") == "ocr":
                for page in pages:
                    page["markdown"] = svc.lines_markdown(page["lines"])
                result["markdown"] = svc.join_markdown(pages)
            result["summary"] = svc.summarize(pages)
            row.full_text = svc.full_text(pages)
            row.result_json = json.dumps(result)
            db.commit()
        return result

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
