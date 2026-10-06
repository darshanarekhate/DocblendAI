"""Document translation: translate the parsed Markdown (or OCR text) with Gemini, keeping its structure."""

from typing import Any

from app.plugins.base import MAX_INPUT_CHARS, Plugin, PluginError, gemini

SYSTEM = (
    "You translate documents. Translate the user's Markdown into the requested language. Keep the "
    "Markdown structure (headings, lists, tables, HTML tables, formulas) exactly; translate only "
    "human-language text. Output only the translated Markdown."
)


class Translation(Plugin):
    name = "translate"
    title = "Translate"
    description = "Translate the parsed document into another language with Gemini, keeping its layout."

    def run(self, text: str, options: dict[str, Any]) -> dict[str, Any]:
        target = options.get("target_language")
        if not isinstance(target, str) or not target.strip() or len(target) > 40:
            raise PluginError("target_language is required, e.g. \"Hindi\" or \"French\"")
        if not text.strip():
            raise PluginError("There is no text to translate.")
        truncated = len(text) > MAX_INPUT_CHARS
        translated = gemini(f"Target language: {target.strip()}\n\n{text[:MAX_INPUT_CHARS]}", SYSTEM)
        return {"target_language": target.strip(), "markdown": translated, "truncated": truncated}
