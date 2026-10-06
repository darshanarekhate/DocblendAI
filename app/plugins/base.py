"""Plugin base class and the shared Gemini call used by the built-in plugins."""

import json
import re
from typing import Any

from app.config import settings

# OCR text sent to the model is capped: plugins work on a page or a short document, not a book.
MAX_INPUT_CHARS = 30_000


class PluginError(RuntimeError):
    """The plugin got bad options or empty input (the caller's fault)."""


class PluginModelError(PluginError):
    """The model call failed or returned something unusable (not the caller's fault).

    status: HTTP status for the API (429 quota used up, 503 overloaded/bad key, 502 otherwise).
    """

    def __init__(self, message: str, status: int = 502) -> None:
        super().__init__(message)
        self.status = status


class PluginUnavailableError(PluginError):
    """The plugin cannot run here (its API key is not set)."""


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

    from app.modules.llm_answer import NO_KEY_MESSAGE, RETRYABLE_CODES, _client, describe_gemini_error

    if not settings.gemini_api_key:
        raise PluginUnavailableError(NO_KEY_MESSAGE)
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
            message, status = describe_gemini_error(e.code, model, e)
            raise PluginModelError(message, status) from e
        if not resp.text:
            raise PluginModelError("Gemini returned an empty response (possibly blocked)")
        return resp.text.strip()
    raise AssertionError("unreachable")


def parse_json(text: str) -> Any:
    """Parse a model's JSON reply, tolerating a ```json fence around it."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise PluginModelError(f"The model did not return valid JSON: {e}") from e
