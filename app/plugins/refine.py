"""Refine recognised text with Gemini: fix misread words, restore words the context implies.

Input is line-based so every change stays attached to the line (and box) it came from:
options["lines"] = [{"line_id", "text", "confidence"?, "page"?}], plus the page structure
(Markdown with tables) as context. Gemini answers with one {line_id, refined_text} per line.
Long documents are sent in parts of at most MAX_PART_CHARS of line text.

Gemini is told to keep each line's layout, spacing and meaning, and never to add facts or
sentences. The answer is still checked here (app/modules/text_diff.py): a refined line that
changes more than max_change of its characters (default 0.4; the automatic pass uses 0.3)
is rejected and the recognised text kept. Only spacing changes count as unchanged.
"""

from collections.abc import Callable
from typing import Any

from app.modules import text_diff
from app.plugins.base import MAX_INPUT_CHARS, Plugin, PluginError, PluginModelError, gemini, parse_json

DEFAULT_MAX_CHANGE = 0.4
MAX_LINES = 20_000
MAX_LINE_CHARS = 2_000
# Line text per Gemini call: small enough for a reliable JSON reply, large enough for context.
MAX_PART_CHARS = 6_000
MAX_PART_LINES = 150
MAX_CONTEXT_CHARS = 6_000

SYSTEM = (
    "You proofread text produced by OCR or handwriting recognition of an academic document. "
    "You receive the document's structure (Markdown, with tables) for context and a JSON list of "
    "recognised lines, each with the recogniser's confidence (0-1; low means it is more likely "
    "misread). For every line: fix misrecognised words and characters (e.g. 'rn' read for 'm', "
    "'0' for 'O', broken or merged words), and restore missing or garbled words only when the "
    "surrounding text makes them clearly implied. Keep everything else exactly: the line's "
    "wording, order, numbers, symbols, capitalisation style, spacing, and table cell content. "
    "Never add new facts, sentences, explanations or words that are not implied, never merge or "
    "split lines, never translate. If a line is already correct or you are unsure, return it "
    "unchanged. Reply with JSON only: {\"lines\": [{\"line_id\": \"...\", \"refined_text\": \"...\"}]} "
    "with exactly one entry for each input line, in the same order."
)

ProgressFn = Callable[[int, int], None]


def _validated_lines(options: dict[str, Any], text: str) -> list[dict[str, Any]]:
    lines = options.get("lines")
    if lines is None:  # plain text: one entry per non-empty line
        lines = [{"line_id": f"l{i}", "text": t} for i, t in enumerate(text.splitlines()) if t.strip()]
    if not isinstance(lines, list) or len(lines) > MAX_LINES:
        raise PluginError(f"lines must be a list of at most {MAX_LINES} entries")
    seen: set[str] = set()
    out = []
    for line in lines:
        if not isinstance(line, dict) or not isinstance(line.get("line_id"), str) or not isinstance(line.get("text"), str):
            raise PluginError("each line needs a string 'line_id' and a string 'text'")
        if line["line_id"] in seen:
            raise PluginError(f"duplicate line_id {line['line_id']!r}")
        seen.add(line["line_id"])
        conf = line.get("confidence")
        if conf is not None and not isinstance(conf, (int, float)):
            raise PluginError("confidence must be a number")
        out.append({
            "line_id": line["line_id"],
            "text": line["text"][:MAX_LINE_CHARS],
            "confidence": None if conf is None else round(float(conf), 3),
            "page": line.get("page"),
        })
    return out


def _parts(lines: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    parts: list[list[dict[str, Any]]] = []
    size = 0
    for line in lines:
        if not parts or size + len(line["text"]) > MAX_PART_CHARS or len(parts[-1]) >= MAX_PART_LINES:
            parts.append([])
            size = 0
        parts[-1].append(line)
        size += len(line["text"])
    return parts


def _context_for(part: list[dict[str, Any]], context: Any) -> str:
    """The structure of the pages this part's lines are on (a str applies to every part)."""
    if isinstance(context, str):
        return context[:MAX_CONTEXT_CHARS]
    if isinstance(context, dict):
        pages = []
        for line in part:
            key = str(line.get("page"))
            if key in context and key not in pages:
                pages.append(key)
        return "\n\n".join(str(context[k]) for k in pages)[:MAX_CONTEXT_CHARS]
    return ""


def _prompt(part: list[dict[str, Any]], context: str) -> str:
    import json

    items = [{"line_id": l["line_id"], "text": l["text"], "confidence": l["confidence"]} for l in part]
    structure = context.strip() or "(no structure available)"
    return (
        f"Document structure, for context only:\n<<<\n{structure}\n>>>\n\n"
        f"Lines to proofread:\n{json.dumps(items, ensure_ascii=False)}"
    )


def _ask(part: list[dict[str, Any]], context: str) -> dict[str, str]:
    """line_id -> refined text for one part (one retry on an unusable JSON reply)."""
    last_error: PluginModelError | None = None
    for _ in range(2):
        try:
            data = parse_json(gemini(_prompt(part, context), SYSTEM, as_json=True))
        except PluginModelError as exc:
            if exc.status != 502:  # quota / overload / key: retrying the same call will not help
                raise
            last_error = exc
            continue
        entries = data.get("lines") if isinstance(data, dict) else data if isinstance(data, list) else None
        if not isinstance(entries, list):
            last_error = PluginModelError("The model's reply has no 'lines' list")
            continue
        refined = {}
        for entry in entries:
            if isinstance(entry, dict) and isinstance(entry.get("line_id"), str) and isinstance(entry.get("refined_text"), str):
                refined[entry["line_id"]] = entry["refined_text"]
        return refined
    raise last_error or PluginModelError("The model did not return usable JSON")


def refine_lines(
    lines: list[dict[str, Any]], context: Any = None, max_change: float = DEFAULT_MAX_CHANGE,
    progress: ProgressFn | None = None,
) -> list[dict[str, Any]]:
    """Proposals for validated lines: [{line_id, page, original, refined, status, change_ratio, diff, reason}]."""
    proposals = []
    parts = _parts([l for l in lines if l["text"].strip()])
    refined_by_id: dict[str, str] = {}
    for i, part in enumerate(parts):
        if progress:
            progress(i, len(parts))
        refined_by_id.update(_ask(part, _context_for(part, context)))
    if progress:
        progress(len(parts), len(parts))
    for line in lines:
        original = line["text"]
        refined = refined_by_id.get(line["line_id"], original)
        refined = " ".join(refined.split("\n")).rstrip() if refined.strip() else original
        result = text_diff.compare(original, refined, max_change)
        if result["status"] == text_diff.UNCHANGED:
            refined = original  # spacing-only edits are not changes: keep the recognised layout
        proposals.append({
            "line_id": line["line_id"], "page": line.get("page"), "original": original,
            "refined": refined, "confidence": line.get("confidence"), **result,
        })
    return proposals


class RefineText(Plugin):
    name = "refine"
    title = "Refine with LLM"
    description = (
        "Fix misrecognised words and restore words the context clearly implies, line by line, "
        "keeping layout and meaning; every change is shown for review."
    )

    def run(self, text: str, options: dict[str, Any]) -> dict[str, Any]:
        lines = _validated_lines(options, text)
        max_change = options.get("max_change", DEFAULT_MAX_CHANGE)
        if not isinstance(max_change, (int, float)) or not 0 < max_change <= 1:
            raise PluginError("max_change must be a number in (0, 1]")
        if not any(l["text"].strip() for l in lines):
            raise PluginError("There is no recognised text to refine.")
        context = options.get("context", text[:MAX_INPUT_CHARS])
        progress = options.get("progress") if callable(options.get("progress")) else None
        proposals = refine_lines(lines, context, float(max_change), progress)
        counts: dict[str, int] = {}
        for p in proposals:
            counts[p["status"]] = counts.get(p["status"], 0) + 1
        return {"lines": proposals, "counts": counts, "max_change": float(max_change)}
