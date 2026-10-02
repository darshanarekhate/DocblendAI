"""Refine with LLM and text versions (prefix /api): docs/experience_center_contract.md §7.

Supports Modules 2 and 7 for the Experience Center (/studio) and the QA page: start a
re-extract -> Gemini refine job, review its proposal (accept all / per line, reject, discard),
and list, compare or restore earlier versions of a document's text. The work itself is in
app/modules/ocr_jobs.py (the job) and app/modules/refinement.py (proposals and versions).
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Body, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.config import settings
from app.modules import refinement
from app.modules.ocr_jobs import JobBusyError, job_manager
from app.modules.preprocess import PreprocessOptionsError, parse_options
from app.plugins import PLUGINS

router = APIRouter(prefix="/api", tags=["refine"])


class RefineRequest(BaseModel):
    """POST /api/results/{id}/refine body (all optional): how to re-extract before refining."""

    preprocess: dict[str, Any] | None = Field(default=None, description="Clean-up steps, e.g. {\"deskew\": true}")
    pipeline: str | None = Field(default=None, description="ocr | structure (scanned pages); default: the run's")
    format_hint: str | None = Field(default=None, description="typed | scanned | handwritten; default: the run's")


class LineSelection(BaseModel):
    line_ids: list[str] | None = Field(default=None, description="Lines to act on; omitted = all pending lines")


def _llm_or_503() -> None:
    plugin = PLUGINS["refine"]
    if not plugin.enabled():
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, plugin.info()["message"])


def _not_busy(run_id: str) -> None:
    if job_manager.active_refine(run_id) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "This document is being re-extracted and refined; wait until it has finished.")


def _version_payload(row) -> dict[str, Any]:
    return {**refinement.version_meta(row), "data": json.loads(row.data_json or "{}")}


@router.post("/results/{run_id}/refine", status_code=status.HTTP_202_ACCEPTED)
def start_refine(run_id: str, request: RefineRequest | None = Body(default=None)) -> dict[str, Any]:
    """Re-extract the original file, refine its lines with Gemini, and store the proposal for review."""
    _llm_or_503()
    request = request or RefineRequest()
    try:
        preprocess = parse_options(request.preprocess) if request.preprocess is not None else None
    except PreprocessOptionsError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    if request.pipeline not in (None, "ocr", "structure", "vl"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "pipeline must be ocr, structure or vl")
    if request.format_hint not in (None, "typed", "scanned", "handwritten"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "format_hint must be typed, scanned or handwritten")
    try:
        job = job_manager.submit_refine(run_id, preprocess, request.pipeline, request.format_hint)
    except refinement.RefinementError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except JobBusyError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return {"job_id": job.id, "run_id": run_id, "status": job.status}


@router.get("/results/{run_id}/versions")
def list_versions(run_id: str) -> list[dict[str, Any]]:
    """Every extraction, edit, restore and refinement of a run, oldest first."""
    if job_manager.get_result(run_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such result")
    with job_manager._session() as db:
        return refinement.list_versions(db, run_id)


@router.get("/results/{run_id}/versions/{version_id}")
def get_version(run_id: str, version_id: str) -> dict[str, Any]:
    with job_manager._session() as db:
        try:
            return _version_payload(refinement.get_version(db, run_id, version_id))
        except refinement.RefinementError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


@router.get("/results/{run_id}/versions/{version_id}/compare")
def compare_version(run_id: str, version_id: str) -> dict[str, Any]:
    """Per-line word diff from this version's text to the current text."""
    with job_manager._session() as db:
        try:
            return refinement.compare(db, run_id, version_id)
        except refinement.RefinementError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


@router.post("/results/{run_id}/versions/{version_id}/restore")
def restore_version(run_id: str, version_id: str) -> dict[str, Any]:
    """Make an earlier version current again (recorded as a new version; nothing is lost)."""
    _not_busy(run_id)
    with job_manager._session() as db:
        try:
            return refinement.restore(db, run_id, version_id)
        except refinement.RefinementError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


@router.get("/results/{run_id}/refinement")
def latest_refinement(run_id: str) -> dict[str, Any]:
    """The newest refinement still awaiting review (404 if there is none)."""
    with job_manager._session() as db:
        row = refinement.latest_pending(db, run_id)
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "No refinement is waiting for review")
        return _version_payload(row)


def _decide(run_id: str, version_id: str, action: str, line_ids: list[str] | None) -> dict[str, Any]:
    _not_busy(run_id)
    with job_manager._session() as db:
        try:
            if action == "accept":
                return refinement.accept(db, run_id, version_id, line_ids)
            return refinement.reject(db, run_id, version_id, line_ids)
        except refinement.RefinementError as exc:
            code = status.HTTP_404_NOT_FOUND if "No " in str(exc)[:3] else status.HTTP_409_CONFLICT
            raise HTTPException(code, str(exc)) from exc


@router.post("/results/{run_id}/refinements/{version_id}/accept")
async def accept_refinement(run_id: str, version_id: str, selection: LineSelection | None = Body(default=None)) -> dict[str, Any]:
    """Accept every pending change (no body) or the given lines: they become the current text
    (edited, source "llm"), and the QA index is rebuilt from it."""
    line_ids = selection.line_ids if selection else None
    return await run_in_threadpool(_decide, run_id, version_id, "accept", line_ids)


@router.post("/results/{run_id}/refinements/{version_id}/reject")
async def reject_refinement(run_id: str, version_id: str, selection: LineSelection | None = Body(default=None)) -> dict[str, Any]:
    """Reject the given lines (or all pending ones); the current text is not changed."""
    line_ids = selection.line_ids if selection else None
    return await run_in_threadpool(_decide, run_id, version_id, "reject", line_ids)


@router.post("/results/{run_id}/refinements/{version_id}/discard")
async def discard_refinement(run_id: str, version_id: str) -> dict[str, Any]:
    """Reject every pending change of this proposal."""
    return await run_in_threadpool(_decide, run_id, version_id, "reject", None)


def llm_status() -> dict[str, Any]:
    """For /api/health and the UI: is Gemini usable for refinement here?"""
    plugin = PLUGINS["refine"]
    return {"enabled": plugin.enabled(), "model": settings.llm_model,
            "fallback_model": settings.llm_fallback_model or None, "message": plugin.info()["message"]}
