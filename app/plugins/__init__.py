"""Pluggable post-processing for Experience Center results (key information, translation, LLM refinement).

Each plugin is a small class with a `name`, a `title`, `enabled()` (only when the API
key it needs is set in .env) and `run(text, options) -> dict`. The router lists them
at GET /api/plugins and runs one at POST /api/results/{id}/plugins/{name}. Adding a
plugin = one module defining a Plugin subclass + one line in PLUGINS below.
"""

from app.plugins.base import Plugin, PluginError, PluginModelError, PluginUnavailableError
from app.plugins.kie import KeyInfoExtraction
from app.plugins.refine import RefineText
from app.plugins.translate import Translation

PLUGINS: dict[str, Plugin] = {p.name: p for p in (RefineText(), KeyInfoExtraction(), Translation())}

__all__ = ["PLUGINS", "Plugin", "PluginError", "PluginModelError", "PluginUnavailableError"]
