"""LLM refinement and text versions (prefix /api): docs/experience_center_contract.md §7.

Supports Modules 2 and 7 for both pages. Every upload is refined automatically (ocr_jobs.py);
these routes show what changed and let the user go back:
- GET  /results/{id}/changes   the lines Gemini changed (word diff), read-only
- POST /results/{id}/revert    use the original extraction again (QA re-embedded)
- POST /results/{id}/reapply   use the LLM-refined text again (no new Gemini call)
- POST /results/{id}/refine    retry the refinement (e.g. after "Gemini unavailable")
- versions: list, view, compare with the current text, restore
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from app.config import settings
from app.modules import refinement
from app.modules.ocr_jobs import JobBusyError, job_manager
from app.plugins import PLUGINS

router = APIRouter(prefix="/api", tags=["refine"])


def _not_busy(run_id: str) -> None:
    if job_manager.active_refine(run_id) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "This document is being refined; wait until it has finished.")


def _version_payload(row) -> dict[str, Any]:
    return {**refinement.version_meta(row), "data": json.loads(row.data_json or "{}")}


def _run(fn, run_id: str, *args) -> dict[str, Any]:
    with job_manager._session() as db:
        try:
            fn(db, run_id, *args)
        except refinement.RefinementError as exc:
            code = status.HTTP_404_NOT_FOUND if str(exc).startswith("No ") else status.HTTP_409_CONFLICT
            raise HTTPException(code, str(exc)) from exc
    return job_manager.get_result(run_id)  # re-read: the QA re-index updated its "qa" state


@router.post("/results/{run_id}/refine", status_code=status.HTTP_202_ACCEPTED)
def retry_refinement(run_id: str) -> dict[str, Any]:
    """Refine the original extraction with Gemini again (202 {job_id}); poll /api/results/{job_id}."""
    plugin = PLUGINS["refine"]
    if not plugin.enabled():
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, plugin.info()["message"])
    try:
        job = job_manager.submit_refine(run_id)
    except refinement.RefinementError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except JobBusyError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return {"job_id": job.id, "run_id": run_id, "status": job.status}


@router.post("/results/{run_id}/revert")
async def revert_to_original(run_id: str) -> dict[str, Any]:
    """Make the original extraction the current text again; QA answers use it from now on."""
    _not_busy(run_id)
    return await run_in_threadpool(_run, refinement.revert, run_id)


@router.post("/results/{run_id}/reapply")
async def use_refined_text(run_id: str) -> dict[str, Any]:
    """Use the latest LLM-refined text again after a revert (no new Gemini call)."""
    _not_busy(run_id)
    return await run_in_threadpool(_run, refinement.reapply, run_id)


@router.get("/results/{run_id}/changes")
def refinement_changes(run_id: str) -> dict[str, Any]:
    """What Gemini proposed for each line (unchanged / corrected / inferred / rejected, word diff)."""
    result = job_manager.get_result(run_id)
    if result is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such result")
    version_id = (result.get("refinement") or {}).get("proposal_version")
    if not version_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "This document has not been refined by the LLM")
    with job_manager._session() as db:
        try:
            return _version_payload(refinement.get_version(db, run_id, version_id))
        except refinement.RefinementError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


@router.get("/results/{run_id}/versions")
def list_versions(run_id: str) -> list[dict[str, Any]]:
    """The original extraction, LLM proposals and refined text, edits, reverts and restores, oldest first."""
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
async def restore_version(run_id: str, version_id: str) -> dict[str, Any]:
    """Make an earlier version current again (recorded as a new version; nothing is lost)."""
    _not_busy(run_id)
    return await run_in_threadpool(_run, refinement.restore, run_id, version_id)


def llm_status() -> dict[str, Any]:
    """For /api/health and the UI: is Gemini usable for refinement here?"""
    plugin = PLUGINS["refine"]
    return {"enabled": plugin.enabled() and settings.llm_refine, "refine_uploads": settings.llm_refine,
            "model": settings.llm_model, "fallback_model": settings.llm_fallback_model or None,
            "message": plugin.info()["message"] if not plugin.enabled() else
            (None if settings.llm_refine else "LLM refinement is turned off (LLM_REFINE=false).")}
