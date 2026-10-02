"""Key-information extraction: pull named fields (title, author, date, totals, ...) out of OCR text.

Every returned value must appear in the recognised text: values the model
invents are dropped, so the output stays as trustworthy as the OCR it came from.
"""

from typing import Any

from app.plugins.base import MAX_INPUT_CHARS, Plugin, PluginError, PluginModelError, gemini, parse_json

DEFAULT_FIELDS = ["title", "author", "date", "course or subject", "institution", "key terms"]

SYSTEM = (
    "You extract key information from OCR text of a document. Return a JSON object whose keys are "
    "exactly the requested field names. Each value is copied verbatim from the text (a string, or a "
    "list of strings for plural fields), or null if the text does not contain it. Never invent values."
)


def _normalise(s: str) -> str:
    return " ".join(s.lower().split())


def _grounded(value: Any, haystack: str) -> Any:
    """Keep only values (or list items) that occur in the source text."""
    if isinstance(value, list):
        kept = [v for v in value if isinstance(v, str) and _normalise(v) in haystack]
        return kept or None
    if isinstance(value, str) and value.strip() and _normalise(value) in haystack:
        return value.strip()
    return None


class KeyInfoExtraction(Plugin):
    name = "kie"
    title = "Key information"
    description = "Extract named fields (title, author, date, ...) from the recognised text with Gemini."

    def run(self, text: str, options: dict[str, Any]) -> dict[str, Any]:
        fields = options.get("fields") or DEFAULT_FIELDS
        if not isinstance(fields, list) or not all(isinstance(f, str) and f.strip() for f in fields) or len(fields) > 30:
            raise PluginError("fields must be a list of up to 30 non-empty strings")
        if not text.strip():
            raise PluginError("There is no recognised text to extract from.")
        prompt = f"Fields: {', '.join(fields)}\n\nText:\n{text[:MAX_INPUT_CHARS]}"
        data = parse_json(gemini(prompt, SYSTEM, as_json=True))
        if not isinstance(data, dict):
            raise PluginModelError("The model did not return a JSON object")
        haystack = _normalise(text)
        return {"fields": {f: _grounded(data.get(f), haystack) for f in fields}}
