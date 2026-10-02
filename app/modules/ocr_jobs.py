"""Experience Center job manager: queued OCR/parse/calibration jobs with progress, history in SQLite.

Supports Modules 2-3 for the PaddleOCR Experience Center (docs/experience_center_contract.md §3-§4).
One worker thread runs jobs in submission order: the Paddle models are shared and CPU-bound, so
running two at once would only make both slower and double the memory. Live progress is kept in
memory; finished results (the contract §4 object) are stored in the `ocr_runs` table
(`OCRRunORM`), page images under settings.paddle_runs_dir/<id>/.

Calibration jobs are in memory only (their output, report.json, is written by run_calibration).

Runs read their file with document_extract.extract (exact text for typed files, PaddleOCR /
PP-StructureV3 for scans, TrOCR in Paddle's line boxes for handwriting) and keep the original
file (source_file) so "Refine with LLM" can re-extract it. A refine job re-extracts, sends the
fresh lines to the `refine` plugin and stores the proposal as a version for review
(refinement.py); one refine at a time per run.
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
    version_id: str | None = None  # refine jobs: the stored proposal
    done: threading.Event = field(default_factory=threading.Event)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def page_image_path(run_id: str, index: int, extraction: int | None = None) -> Path:
    """Page image of a run: the first extraction's in the run folder, re-extraction n's in x<n>/."""
    folder = settings.paddle_runs_dir / run_id
    if extraction is not None:
        folder = folder / f"x{extraction}"
    return folder / f"page-{index}.png"


def page_image_for(run_id: str, page: dict[str, Any]) -> Path:
    """The image file a result page shows (its image_url may name a re-extraction: ?x=<n>)."""
    url = str(page.get("image_url") or "")
    extraction = None
    if "?x=" in url:
        try:
            extraction = int(url.rsplit("?x=", 1)[1])
        except ValueError:
            extraction = None
    return page_image_path(run_id, int(page.get("index") or 0), extraction)


def _image_url(run_id: str, extraction: int | None = None) -> Callable[[int], str]:
    suffix = f"?x={extraction}" if extraction is not None else ""
    return lambda index: f"/api/results/{run_id}/pages/{index}/image{suffix}"


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
    ) -> Job:
        """Queue one document. Takes ownership of upload_path (kept as the run's source file)."""
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
        return self._enqueue(
            job, lambda j: self._run_document(j, upload_path, review_threshold, preprocess, format_hint)
        )

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
            job.progress = max(job.progress, min(0.99, max(0.0, fraction)))
            job.message = message

        result: dict[str, Any] | None = None
        try:
            # Keep the original next to the page images: re-extraction ("Refine with LLM") reads it again.
            pages_dir.mkdir(parents=True, exist_ok=True)
            source = pages_dir / f"source{upload_path.suffix.lower()}"
            shutil.move(str(upload_path), source)
            extraction = document_extract.extract(
                source, pages_dir=pages_dir, format_hint=_hint(format_hint), pipeline=job.pipeline,
                lang=job.language, image_url=_image_url(job.id), progress=progress, preprocess=preprocess,
            )
            result = self._finished(job, extraction, review_threshold, preprocess, started)
            result["source_file"] = source.name
        except svc.PaddleOCRUnavailableError as exc:
            job.unavailable = True
            job.error = str(exc)
        except (svc.UnreadableInputError, document_extract.ExtractionError) as exc:
            job.error = f"Unreadable file: {exc}"
        except Exception as exc:  # inference failures: report, keep the worker alive
            if type(exc).__name__ in ("OCRUnavailableError", "HTRUnavailableError"):
                job.unavailable = True
                job.error = str(exc)
            else:
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
        if result["status"] == "done":
            with self._session() as db:
                refinement.record(db, job.id, "extraction", "Original extraction", result)
                db.commit()
        pages_done = (result.get("summary") or {}).get("page_count", 0)
        logger.info(
            "job %s %s in %.0f ms (%s, %s page(s))", job.id, job.status, job.processing_time_ms,
            job.pipeline, pages_done,
            extra={"job_id": job.id, "pipeline": job.pipeline, "status": job.status,
                   "duration_ms": job.processing_time_ms, "page_count": pages_done},
        )

    def _finished(
        self, job: Job, extraction: document_extract.Extraction, review_threshold: float,
        preprocess: dict[str, bool] | None, started: float,
    ) -> dict[str, Any]:
        """Contract result object for a finished extraction (job marked done)."""
        pages = extraction.pages
        calibration = svc.review_flags(pages, review_threshold)
        job.processing_time_ms = round((time.perf_counter() - started) * 1000, 1)
        job.status, job.progress, job.message = "done", 1.0, " ".join(extraction.notes) or "Done"
        result = self._base(job)
        result.update({
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

    def delete(self, job_id: str) -> bool:
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
        return result

    # -- refine with LLM -----------------------------------------------------------------------

    def active_refine(self, run_id: str) -> Job | None:
        """The queued or running refine job of a run, if any."""
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            if job.kind == "refine" and job.run_id == run_id and job.status in ("queued", "running"):
                return job
        return None

    def submit_refine(
        self, run_id: str, preprocess: dict[str, bool] | None = None, pipeline: str | None = None,
        format_hint: str | None = None,
    ) -> Job:
        """Queue re-extract -> Gemini refine -> proposal. JobBusyError if one is already queued/running."""
        with self._session() as db:
            row = db.get(OCRRunORM, run_id)
            if row is None:
                raise refinement.RefinementError(f"No result with id {run_id}")
            result = json.loads(row.result_json or "{}")
        if result.get("status") != "done":
            raise JobBusyError("This run has not finished yet.")
        with self._lock:
            for other in self._jobs.values():
                if other.kind == "refine" and other.run_id == run_id and other.status in ("queued", "running"):
                    raise JobBusyError("A refinement of this document is already running.")
            job = Job(
                id=uuid.uuid4().hex, kind="refine", pipeline="refine", language=result.get("language") or "en",
                filename=result.get("filename") or "", created_at=_now(), run_id=run_id,
            )
            self._jobs[job.id] = job
        self._ensure_worker()
        self._queue.put((job, lambda j: self._run_refine(j, preprocess, pipeline, format_hint)))
        return job

    def source_path(self, run_id: str, result: dict[str, Any]) -> Path | None:
        """The original uploaded file of a run, if it was kept."""
        name = result.get("source_file")
        if name:
            path = settings.paddle_runs_dir / run_id / Path(str(name)).name
            if path.is_file():
                return path
        return None

    def _run_refine(
        self, job: Job, preprocess: dict[str, bool] | None, pipeline: str | None, format_hint: str | None
    ) -> None:
        from app.plugins import PLUGINS, PluginError

        started = time.perf_counter()
        run_id = job.run_id
        job.status, job.progress, job.message = "running", 0.02, "Starting"
        re_extracted = False
        try:
            with self._session() as db:
                _, result = refinement.load_current(db, run_id)
                refinement.ensure_initial(db, run_id, result)
                db.commit()
            source = self.source_path(run_id, result)
            notes: list[str] = []
            if source is not None:
                result = self._re_extract(job, run_id, result, source, preprocess, pipeline, format_hint)
                re_extracted = True
            else:
                notes.append("The original file of this run was not kept, so its current text was refined.")

            job.progress, job.message = 0.55, "Refining with Gemini…"
            lines, context = refinement.refine_input(result)
            if not lines:
                raise PluginError("There is no recognised text to refine.")

            def part_progress(done: int, total: int) -> None:
                job.progress = 0.55 + 0.4 * (done / max(1, total))
                if done < total and total > 1:
                    job.message = f"Refining with Gemini… (part {done + 1} of {total})"

            output = PLUGINS["refine"].run("", {
                "lines": lines, "context": context, "max_change": settings.refine_max_change,
                "progress": part_progress,
            })
            with self._session() as db:
                base = refinement.list_versions(db, run_id)[-1]["id"]
                proposal = refinement.make_proposal(run_id, output, "manual", base)
                counts = proposal["counts"]
                changed = counts.get("corrected", 0) + counts.get("inferred", 0)
                label = f"LLM refinement: {changed} change{'s' if changed != 1 else ''} proposed"
                if counts.get("rejected"):
                    label += f", {counts['rejected']} rejected"
                version = refinement.record(db, run_id, "refinement", label, proposal, status="pending")
                db.commit()
                job.version_id = version.id
            job.status, job.progress = "done", 1.0
            job.message = " ".join(notes + ["Ready for review"])
        except PluginError as exc:
            job.error = f"Re-extracted the document, but refining failed: {exc}" if re_extracted else str(exc)
        except refinement.RefinementError as exc:
            job.error = str(exc)
        except svc.PaddleOCRUnavailableError as exc:
            job.unavailable, job.error = True, f"Re-extraction failed: {exc}"
        except (svc.UnreadableInputError, document_extract.ExtractionError) as exc:
            job.error = f"Re-extraction failed: unreadable file: {exc}"
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

    def _re_extract(
        self, job: Job, run_id: str, result: dict[str, Any], source: Path, preprocess: dict[str, bool] | None,
        pipeline: str | None, format_hint: str | None,
    ) -> dict[str, Any]:
        """Step 1 of a refine: read the original file again; the fresh result becomes current."""
        started = time.perf_counter()
        with self._session() as db:
            seq = refinement._next_seq(db, run_id)

        def progress(fraction: float, message: str) -> None:
            job.progress = 0.05 + 0.45 * min(1.0, max(0.0, fraction))
            job.message = f"Re-extracting {message[:1].lower()}{message[1:]}…"

        job.message = "Re-extracting…"
        hint = format_hint or result.get("format_type")
        current = result.get("pipeline")
        use_pipeline = pipeline or (current if current in ("ocr", "structure", "vl") else "ocr")
        extraction = document_extract.extract(
            source, pages_dir=settings.paddle_runs_dir / run_id / f"x{seq}", format_hint=_hint(hint),
            pipeline=use_pipeline, lang=result.get("language") or "en", image_url=_image_url(run_id, seq),
            progress=progress, preprocess=preprocess,
        )
        threshold = (result.get("calibration") or {}).get("review_threshold", settings.review_threshold)
        pages = extraction.pages
        fresh = dict(result)
        fresh.update({
            "pages": pages, "calibration": svc.review_flags(pages, threshold), "summary": svc.summarize(pages),
            "markdown": svc.join_markdown(pages), "format_type": extraction.format_type.value,
            "engine": extraction.engine, "notes": extraction.notes,
            "preprocess": sorted(k for k, v in (preprocess or {}).items() if v),
            "processing_time_ms": round((time.perf_counter() - started) * 1000, 1),
            "pipeline": "office" if current == "office" else use_pipeline,
        })
        fresh.pop("active_job", None)
        steps = ", ".join(fresh["preprocess"])
        with self._session() as db:
            row = db.get(OCRRunORM, run_id)
            refinement.save_current(db, row, fresh)
            refinement.record(db, run_id, "extraction", "Re-extraction" + (f" ({steps})" if steps else ""), fresh)
            db.commit()
        refinement.notify_change(run_id, fresh, "re-extracted")
        return fresh

    def _refine_view(self, job: Job) -> dict[str, Any]:
        return {
            "id": job.id, "status": job.status, "progress": round(job.progress, 4), "message": job.message,
            "error": job.error, "pipeline": "refine", "run_id": job.run_id, "version_id": job.version_id,
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
