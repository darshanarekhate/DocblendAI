"""Supports Modules 2 and 7 — versions of a run's text, and LLM refinement proposals for review.

Responsibility: keep every state of an Experience Center run's text so nothing is overwritten,
and turn a `refine` plugin answer into a reviewable proposal:

- a run's current result is ocr_runs.result_json; every change is also recorded as a version
  (run_versions): "extraction" (first read or re-extraction), "edit" (manual corrections,
  accepted refinements), "restore" (roll back to an earlier version). Each holds the full result,
  so any two versions can be compared and any one restored.
- a "refinement" version is an LLM proposal: per line the recognised text, the refined text,
  a word diff (unchanged / corrected / inferred) and the user's decision. Accepting writes the
  refined text into the current result (edited: true, source: "llm" or "llm-auto", the
  recognised text kept as ocr_text); lines the safety rule rejected keep their text and are
  flagged for review.

Every change to the current result calls the registered change hooks (the QA index re-chunks
and re-embeds the document, see document_runs.py).

Uses from schemas.py: nothing; works on contract result dicts (docs/experience_center_contract.md).
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import OCRRunORM, RunVersionORM
from app.modules import paddleocr_service as svc
from app.modules import text_diff

logger = logging.getLogger(__name__)

CHANGEABLE = (text_diff.CORRECTED, text_diff.INFERRED)
ChangeHook = Callable[[str, dict[str, Any], str], None]  # (run_id, result, reason)
_change_hooks: list[ChangeHook] = []


class RefinementError(ValueError):
    """A proposal/version operation is not possible (unknown id, already decided, ...)."""


def add_change_hook(hook: ChangeHook) -> None:
    if hook not in _change_hooks:
        _change_hooks.append(hook)


def notify_change(run_id: str, result: dict[str, Any], reason: str) -> None:
    for hook in list(_change_hooks):
        try:
            hook(run_id, result, reason)
        except Exception:  # a hook failure (e.g. embedding quota) must not undo the text change
            logger.exception("change hook failed for run %s", run_id)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# -- text changes inside a result -----------------------------------------------------------------


def find_line(result: dict[str, Any], line_id: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
    for page in result.get("pages") or []:
        for line in page.get("lines") or []:
            if line.get("id") == line_id:
                return page, line
    return None


def _replace_once(text: str | None, old: str, new: str) -> str | None:
    if not text or not old or old not in text:
        return text
    return text.replace(old, new, 1)


def set_line_text(page: dict[str, Any], line: dict[str, Any], new_text: str) -> None:
    """Change one line's text and the same words in the page's Markdown, blocks and tables,
    so exports (Markdown, DOCX, CSV) show the change too, not only the line list."""
    old = line["text"]
    line["text"] = new_text
    if old == new_text:
        return
    page["markdown"] = _replace_once(page.get("markdown"), old, new_text) or page.get("markdown", "")
    for block in page.get("blocks") or []:
        if old in str(block.get("content") or ""):
            block["content"] = _replace_once(block["content"], old, new_text)
            break
    for table in page.get("tables") or []:
        if table.get("html") and old in table["html"]:
            table["html"] = _replace_once(table["html"], old, new_text)
        for row in table.get("rows") or []:
            for i, cell in enumerate(row):
                if cell == old:
                    row[i] = new_text


def refresh(result: dict[str, Any]) -> None:
    """Recompute what depends on line texts: OCR-pipeline Markdown, document Markdown, summary."""
    pages = result.get("pages") or []
    for page in pages:
        if not page.get("blocks") and not page.get("tables") and page.get("lines") and page.get("text") is None:
            page["markdown"] = svc.lines_markdown(page["lines"])
    result["markdown"] = svc.join_markdown(pages)
    result["summary"] = svc.summarize(pages)


# -- version store --------------------------------------------------------------------------------


def _next_seq(db: Session, run_id: str) -> int:
    return (db.scalar(select(func.max(RunVersionORM.seq)).where(RunVersionORM.run_id == run_id)) or 0) + 1


def record(db: Session, run_id: str, kind: str, label: str, data: dict[str, Any], status: str = "") -> RunVersionORM:
    row = RunVersionORM(
        id=uuid.uuid4().hex, run_id=run_id, seq=_next_seq(db, run_id), kind=kind, status=status,
        label=label, created_at=_now(), data_json=json.dumps(data),
    )
    db.add(row)
    db.flush()
    return row


def ensure_initial(db: Session, run_id: str, result: dict[str, Any]) -> None:
    """Runs made before versions existed: record their current state as version 1."""
    if db.scalar(select(func.count()).select_from(RunVersionORM).where(RunVersionORM.run_id == run_id)) == 0:
        record(db, run_id, "extraction", "Original extraction", result)


def save_current(db: Session, row: OCRRunORM, result: dict[str, Any]) -> None:
    """Write a run's current result (and the columns derived from it); caller commits."""
    summary = result.get("summary") or {}
    row.status = result.get("status", row.status)
    row.page_count = summary.get("page_count")
    row.mean_confidence = summary.get("mean_calibrated_confidence")
    row.processing_time_ms = result.get("processing_time_ms")
    row.full_text = svc.full_text(result.get("pages") or [])
    row.result_json = json.dumps(result)


def load_current(db: Session, run_id: str) -> tuple[OCRRunORM, dict[str, Any]]:
    row = db.get(OCRRunORM, run_id)
    if row is None:
        raise RefinementError(f"No result with id {run_id}")
    result = json.loads(row.result_json or "{}")
    if result.get("status") != "done":
        raise RefinementError("Only finished runs have text to refine or version.")
    return row, result


def version_meta(row: RunVersionORM) -> dict[str, Any]:
    meta = {"id": row.id, "seq": row.seq, "kind": row.kind, "status": row.status, "label": row.label,
            "created_at": row.created_at}
    if row.kind == "refinement":
        data = json.loads(row.data_json or "{}")
        meta["counts"] = data.get("counts")
        meta["mode"] = data.get("mode")
    return meta


def list_versions(db: Session, run_id: str) -> list[dict[str, Any]]:
    rows = db.scalars(select(RunVersionORM).where(RunVersionORM.run_id == run_id).order_by(RunVersionORM.seq)).all()
    return [version_meta(r) for r in rows]


def get_version(db: Session, run_id: str, version_id: str) -> RunVersionORM:
    row = db.get(RunVersionORM, version_id)
    if row is None or row.run_id != run_id:
        raise RefinementError(f"No version {version_id} for run {run_id}")
    return row


def delete_versions(db: Session, run_id: str) -> None:
    for row in db.scalars(select(RunVersionORM).where(RunVersionORM.run_id == run_id)).all():
        db.delete(row)


def latest_pending(db: Session, run_id: str) -> RunVersionORM | None:
    return db.scalars(
        select(RunVersionORM)
        .where(RunVersionORM.run_id == run_id, RunVersionORM.kind == "refinement", RunVersionORM.status.in_(("pending", "partial")))
        .order_by(RunVersionORM.seq.desc())
    ).first()


# -- proposals ------------------------------------------------------------------------------------


def refine_input(result: dict[str, Any], only_below: float | None = None) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Plugin input from a result: lines (optionally only those below a confidence) + page structure."""
    lines, context = [], {}
    for page in result.get("pages") or []:
        index = page.get("index", 0)
        context[str(index)] = page.get("markdown") or svc.lines_markdown(page.get("lines") or [])
        for line in page.get("lines") or []:
            if not line.get("text", "").strip():
                continue
            if only_below is not None and (line.get("edited") or line.get("calibrated_confidence", 1.0) >= only_below):
                continue
            lines.append({"line_id": line["id"], "text": line["text"],
                          "confidence": line.get("calibrated_confidence"), "page": index})
    return lines, context


def make_proposal(run_id: str, plugin_output: dict[str, Any], mode: str, base_version: str | None) -> dict[str, Any]:
    lines = []
    for p in plugin_output.get("lines") or []:
        decision = "pending" if p["status"] in CHANGEABLE else "n/a"
        lines.append({**p, "decision": decision})
    counts: dict[str, int] = {}
    for p in lines:
        counts[p["status"]] = counts.get(p["status"], 0) + 1
    return {
        "run_id": run_id, "mode": mode, "base_version": base_version, "model": settings.llm_model,
        "max_change": plugin_output.get("max_change"), "lines": lines, "counts": counts, "created_at": _now(),
    }


def _proposal_status(proposal: dict[str, Any]) -> str:
    decisions = [l["decision"] for l in proposal["lines"] if l["status"] in CHANGEABLE]
    if any(d == "pending" for d in decisions):
        return "partial" if any(d != "pending" for d in decisions) else "pending"
    if decisions and all(d in ("rejected", "conflict") for d in decisions):
        return "discarded"
    return "accepted"


def apply_proposal(
    result: dict[str, Any], proposal: dict[str, Any], line_ids: set[str] | None, source: str
) -> int:
    """Write accepted refinements into result (in place). Returns how many lines changed.

    line_ids None = every pending change; guard-rejected lines are then flagged for review.
    A line whose current text no longer matches the proposal's original is a "conflict"
    (it changed after the proposal was made) and is left alone.
    """
    changed = 0
    for p in proposal["lines"]:
        hit = find_line(result, p["line_id"])
        if p["status"] == text_diff.REJECTED and line_ids is None and hit is not None:
            page, line = hit
            line["llm_flag"] = p.get("reason")
            line["needs_review"] = True
            continue
        if p["status"] not in CHANGEABLE or p["decision"] != "pending":
            continue
        if line_ids is not None and p["line_id"] not in line_ids:
            continue
        if hit is None or hit[1]["text"] != p["original"]:
            p["decision"] = "conflict"
            continue
        page, line = hit
        line.setdefault("ocr_text", p["original"])
        set_line_text(page, line, p["refined"])
        line.update({"edited": True, "source": source, "needs_review": False, "llm_diff": p["diff"],
                     "llm_status": p["status"]})
        line.pop("llm_flag", None)
        p["decision"] = "accepted"
        changed += 1
    proposal["status"] = _proposal_status(proposal)
    return changed


def accept(db: Session, run_id: str, version_id: str, line_ids: list[str] | None) -> dict[str, Any]:
    """Accept all pending changes (line_ids None) or some lines; returns {result, proposal}."""
    version = get_version(db, run_id, version_id)
    if version.kind != "refinement" or version.status not in ("pending", "partial"):
        raise RefinementError("This refinement has already been reviewed.")
    run, result = load_current(db, run_id)
    ensure_initial(db, run_id, result)
    proposal = json.loads(version.data_json)
    source = "llm-auto" if proposal.get("mode") == "auto" else "llm"
    changed = apply_proposal(result, proposal, set(line_ids) if line_ids is not None else None, source)
    version.status = proposal["status"]
    version.data_json = json.dumps(proposal)
    if changed or line_ids is None:
        refresh(result)
        save_current(db, run, result)
        record(db, run_id, "edit", f"Accepted {changed} LLM correction{'s' if changed != 1 else ''}", result)
    db.commit()
    if changed:
        notify_change(run_id, result, "refinement accepted")
    return {"result": result, "proposal": {**version_meta(version), **proposal}}


def reject(db: Session, run_id: str, version_id: str, line_ids: list[str] | None) -> dict[str, Any]:
    """Reject some lines (or all pending ones: discard). The current text is not touched."""
    version = get_version(db, run_id, version_id)
    if version.kind != "refinement" or version.status not in ("pending", "partial"):
        raise RefinementError("This refinement has already been reviewed.")
    proposal = json.loads(version.data_json)
    wanted = set(line_ids) if line_ids is not None else None
    for p in proposal["lines"]:
        if p["decision"] == "pending" and (wanted is None or p["line_id"] in wanted):
            p["decision"] = "rejected"
    proposal["status"] = _proposal_status(proposal)
    version.status = proposal["status"]
    version.data_json = json.dumps(proposal)
    db.commit()
    return {"proposal": {**version_meta(version), **proposal}}


def restore(db: Session, run_id: str, version_id: str) -> dict[str, Any]:
    version = get_version(db, run_id, version_id)
    if version.kind == "refinement":
        raise RefinementError("A refinement proposal is not a text version; accept its lines instead.")
    run, current = load_current(db, run_id)
    ensure_initial(db, run_id, current)
    result = json.loads(version.data_json)
    result["status"] = "done"
    save_current(db, run, result)
    record(db, run_id, "restore", f"Restored version {version.seq}", result)
    db.commit()
    notify_change(run_id, result, "version restored")
    return result


def compare(db: Session, run_id: str, version_id: str) -> dict[str, Any]:
    """Per-line word diff from a version's text to the current text (matched by line id)."""
    version = get_version(db, run_id, version_id)
    if version.kind == "refinement":
        raise RefinementError("Open a refinement in the review view instead.")
    _, current = load_current(db, run_id)
    old = json.loads(version.data_json)
    old_lines = {l["id"]: l["text"] for p in old.get("pages") or [] for l in p.get("lines") or []}
    rows = []
    for page in current.get("pages") or []:
        for line in page.get("lines") or []:
            before = old_lines.pop(line["id"], None)
            if before is None:
                rows.append({"line_id": line["id"], "page": page.get("index"), "before": None, "after": line["text"], "status": "added"})
            elif before != line["text"]:
                cmp = text_diff.compare(before, line["text"], 1.0)
                rows.append({"line_id": line["id"], "page": page.get("index"), "before": before, "after": line["text"],
                             "status": cmp["status"], "diff": cmp["diff"]})
    rows.extend({"line_id": lid, "page": None, "before": text, "after": None, "status": "removed"} for lid, text in old_lines.items())
    return {"version": version_meta(version), "changes": rows}


def record_edit(db: Session, run_id: str, result: dict[str, Any], label: str) -> None:
    """Manual corrections (PATCH /lines): keep the previous state and the new one as versions."""
    ensure_initial(db, run_id, json.loads(db.get(OCRRunORM, run_id).result_json or "{}"))
    record(db, run_id, "edit", label, result)
