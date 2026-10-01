"""Plugin base class and the shared Gemini call used by the built-in plugins."""

import json
import re
from typing import Any

from app.config import settings

# OCR text sent to the model is capped: plugins work on a page or a short document, not a book.
MAX_INPUT_CHARS = 30_000


class PluginError(RuntimeError):
    """The plugin is disabled, got bad options, or its model call failed."""


class Plugin:
    name = ""
    title = ""
    description = ""
    requires = "GEMINI_API_KEY"

    def enabled(self) -> bool:
        return bool(settings.gemini_api_key)

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "enabled": self.enabled(),
            "message": None if self.enabled() else f"Disabled: set {self.requires} in .env to enable.",
        }

    def run(self, text: str, options: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


def gemini(prompt: str, system: str, as_json: bool = False) -> str:
    """One Gemini call with settings.llm_model (falls back to llm_fallback_model on 429/503)."""
    from google.genai import errors, types

    from app.modules.llm_answer import RETRYABLE_CODES, _client

    if not settings.gemini_api_key:
        raise PluginError("GEMINI_API_KEY is not set (see .env.example)")
    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=0.1,
        response_mime_type="application/json" if as_json else None,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    models = [settings.llm_model]
    if settings.llm_fallback_model and settings.llm_fallback_model != settings.llm_model:
        models.append(settings.llm_fallback_model)
    for i, model in enumerate(models):
        try:
            resp = _client(settings.gemini_api_key).models.generate_content(model=model, contents=prompt, config=config)
        except errors.APIError as e:
            if e.code in RETRYABLE_CODES and i < len(models) - 1:
                continue
            raise PluginError(f"Gemini call failed ({model}): {e}") from e
        if not resp.text:
            raise PluginError("Gemini returned an empty response (possibly blocked)")
        return resp.text.strip()
    raise AssertionError("unreachable")


def parse_json(text: str) -> Any:
    """Parse a model's JSON reply, tolerating a ```json fence around it."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise PluginError(f"The model did not return valid JSON: {e}") from e
