"""Supports Modules 2 and 7 — automatic LLM refinement of a document's text, and its versions.

Responsibility: refine every upload's extracted text with Gemini and use the result, while
never losing what was actually read:

- refine_document(): the `refine` plugin proposes a corrected text for every line (misread words
  fixed, logically implied missing words restored, layout kept, no new facts); every change it
  is allowed to make is applied at once (no approval step). Changed lines get source "llm", the
  recognised text in ocr_text, and the word diff (llm_diff: corrected / inferred words) so both
  pages can highlight them. A line whose refinement changes more than refine_max_change of its
  characters keeps its recognised text and is flagged (llm_flag, needs_review).
- versions (run_versions table): the original extraction, the refinement proposal (what Gemini
  suggested, line by line), the refined text, edits, reverts and restores are all kept in full,
  so the user can show the original, compare, revert to it, or use the refined text again.
- result["refinement"] says what happened: refined | unavailable (no key, quota, Gemini error:
  the unrefined text is used and QA is never blocked) | reverted | off | empty.

Every change to the current text calls the registered change hooks (document_runs.py re-chunks
and re-embeds the QA document, so answers follow the text shown).

Uses from schemas.py: nothing; works on contract result dicts (docs/experience_center_contract.md).
"""

from __future__ import annotations

import copy
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
REFINED_LABEL = "Refined by Gemini"
ChangeHook = Callable[[str, dict[str, Any], str], None]  # (run_id, result, reason)
ProgressFn = Callable[[int, int], None]
_change_hooks: list[ChangeHook] = []


class RefinementError(ValueError):
    """A version operation is not possible (unknown run or version, nothing to revert to, ...)."""


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
    so exports (Markdown, DOCX, CSV) and the layout view show the change too."""
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
        meta["counts"] = json.loads(row.data_json or "{}").get("counts")
    return meta


def list_versions(db: Session, run_id: str) -> list[dict[str, Any]]:
    rows = db.scalars(select(RunVersionORM).where(RunVersionORM.run_id == run_id).order_by(RunVersionORM.seq)).all()
    return [version_meta(r) for r in rows]


def get_version(db: Session, run_id: str, version_id: str) -> RunVersionORM:
    row = db.get(RunVersionORM, version_id)
    if row is None or row.run_id != run_id:
        raise RefinementError(f"No version {version_id} for run {run_id}")
    return row


def _latest(db: Session, run_id: str, kind: str, label_prefix: str | None = None) -> RunVersionORM | None:
    stmt = select(RunVersionORM).where(RunVersionORM.run_id == run_id, RunVersionORM.kind == kind)
    if label_prefix:
        stmt = stmt.where(RunVersionORM.label.startswith(label_prefix))
    return db.scalars(stmt.order_by(RunVersionORM.seq.desc())).first()


def original_version(db: Session, run_id: str) -> RunVersionORM | None:
    """The newest extraction: the text as read, before any LLM refinement or edit."""
    return _latest(db, run_id, "extraction")


def delete_versions(db: Session, run_id: str) -> None:
    for row in db.scalars(select(RunVersionORM).where(RunVersionORM.run_id == run_id)).all():
        db.delete(row)


# -- refinement -----------------------------------------------------------------------------------


def refine_input(result: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Plugin input from a result: every line with its calibrated confidence + each page's structure."""
    lines, context = [], {}
    for page in result.get("pages") or []:
        index = page.get("index", 0)
        context[str(index)] = page.get("markdown") or svc.lines_markdown(page.get("lines") or [])
        for line in page.get("lines") or []:
            if line.get("text", "").strip():
                lines.append({"line_id": line["id"], "text": line["text"],
                              "confidence": line.get("calibrated_confidence"), "page": index})
    return lines, context


def make_proposal(run_id: str, plugin_output: dict[str, Any], base_version: str | None) -> dict[str, Any]:
    lines = [{**p, "decision": "pending" if p["status"] in CHANGEABLE else "n/a"} for p in plugin_output.get("lines") or []]
    counts: dict[str, int] = {}
    for p in lines:
        counts[p["status"]] = counts.get(p["status"], 0) + 1
    return {
        "run_id": run_id, "base_version": base_version, "model": settings.llm_model,
        "max_change": plugin_output.get("max_change"), "lines": lines, "counts": counts, "created_at": _now(),
    }


def apply_proposal(result: dict[str, Any], proposal: dict[str, Any], source: str = "llm") -> int:
    """Write every allowed change into result (in place); flag the lines the safety rule rejected.

    Returns how many lines changed. A line whose text no longer matches the proposal's original
    is left alone ("conflict").
    """
    changed = 0
    for p in proposal["lines"]:
        hit = find_line(result, p["line_id"])
        if hit is None:
            continue
        page, line = hit
        if p["status"] == text_diff.REJECTED:
            line["llm_flag"] = p.get("reason")
            line["needs_review"] = True
            continue
        if p["status"] not in CHANGEABLE:
            continue
        if line["text"] != p["original"]:
            p["decision"] = "conflict"
            continue
        line.setdefault("ocr_text", p["original"])
        set_line_text(page, line, p["refined"])
        line.update({"edited": True, "source": source, "llm_diff": p["diff"], "llm_status": p["status"]})
        line.pop("llm_flag", None)
        p["decision"] = "accepted"
        changed += 1
    return changed


def refine_document(
    db: Session, run_id: str, original: dict[str, Any], base_version: str | None, progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """Refine a result's lines with Gemini and apply the changes: returns the refined result.

    Records the proposal and the refined text as versions (caller commits). Raises the plugin's
    PluginError (incl. PluginModelError / PluginUnavailableError) when Gemini cannot refine:
    callers keep the unrefined text then.
    """
    from app.plugins import PLUGINS

    lines, context = refine_input(original)
    refined = copy.deepcopy(original)
    if not lines:
        refined["refinement"] = {"status": "empty", "message": "There is no text to refine.", "at": _now()}
        return refined
    output = PLUGINS["refine"].run("", {
        "lines": lines, "context": context, "max_change": settings.refine_max_change, "progress": progress,
    })
    proposal = make_proposal(run_id, output, base_version)
    changed = apply_proposal(refined, proposal)
    refresh(refined)
    counts = proposal["counts"]
    proposal_row = record(db, run_id, "refinement", f"Gemini's proposal ({changed} change{'s' if changed != 1 else ''})",
                          proposal, status="applied")
    rejected = counts.get("rejected", 0)
    message = (f"Refined by LLM: {changed} line{'s' if changed != 1 else ''} corrected"
               + (f", {rejected} change{'s' if rejected != 1 else ''} rejected as too large" if rejected else "") + ".")
    refined["refinement"] = {
        "status": "refined", "message": message, "model": settings.llm_model, "corrected": changed,
        "counts": counts, "proposal_version": proposal_row.id, "original_version": base_version, "at": _now(),
    }
    record(db, run_id, "edit", f"{REFINED_LABEL} ({changed} correction{'s' if changed != 1 else ''})", refined)
    return refined


def unavailable(result: dict[str, Any], reason: str) -> dict[str, Any]:
    """Mark a result as not refined (Gemini missing / out of quota / failed); its text is unchanged."""
    result["refinement"] = {"status": "unavailable", "message": f"Not refined: Gemini unavailable. {reason}", "at": _now()}
    return result


def revert(db: Session, run_id: str) -> dict[str, Any]:
    """Make the original extraction current again (the refined text stays in the history)."""
    run, current = load_current(db, run_id)
    ensure_initial(db, run_id, current)
    original = original_version(db, run_id)
    if original is None:
        raise RefinementError("There is no original extraction to revert to.")
    result = json.loads(original.data_json)
    result.update({"status": "done", "qa": current.get("qa")})
    previous = current.get("refinement") or {}
    result["refinement"] = {**previous, "status": "reverted", "at": _now(),
                            "message": "Reverted to the original extraction; the LLM refinement is kept in the history."}
    save_current(db, run, result)
    record(db, run_id, "restore", "Reverted to the original extraction", result)
    db.commit()
    notify_change(run_id, result, "reverted to the original")
    return result


def reapply(db: Session, run_id: str) -> dict[str, Any]:
    """Use the latest LLM-refined text again (no new Gemini call)."""
    run, current = load_current(db, run_id)
    refined = _latest(db, run_id, "edit", REFINED_LABEL)
    if refined is None:
        raise RefinementError("This document has not been refined yet: use Retry to refine it.")
    result = json.loads(refined.data_json)
    result.update({"status": "done", "qa": current.get("qa")})
    result["refinement"] = {**(result.get("refinement") or {}), "status": "refined", "at": _now()}
    save_current(db, run, result)
    record(db, run_id, "restore", "Using the LLM-refined text again", result)
    db.commit()
    notify_change(run_id, result, "refined text restored")
    return result


def restore(db: Session, run_id: str, version_id: str) -> dict[str, Any]:
    version = get_version(db, run_id, version_id)
    if version.kind == "refinement":
        raise RefinementError("A refinement proposal is not a text version.")
    run, current = load_current(db, run_id)
    ensure_initial(db, run_id, current)
    result = json.loads(version.data_json)
    result.update({"status": "done", "qa": current.get("qa")})
    save_current(db, run, result)
    record(db, run_id, "restore", f"Restored version {version.seq}", result)
    db.commit()
    notify_change(run_id, result, "version restored")
    return result


def compare(db: Session, run_id: str, version_id: str) -> dict[str, Any]:
    """Per-line word diff from a version's text to the current text (matched by line id)."""
    version = get_version(db, run_id, version_id)
    if version.kind == "refinement":
        raise RefinementError("A refinement proposal is not a text version.")
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
