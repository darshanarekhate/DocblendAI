"""A fake Gemini for the `refine` plugin: answers with per-line rewrites, no network."""

import json
import re
import threading

LINES_RE = re.compile(r"Lines to proofread:\n(\[.*\])\s*$", re.S)


class FakeGemini:
    """Replace app.plugins.refine.gemini. `rewrites` maps original line text -> refined text.

    Lines not in rewrites come back unchanged. `calls` counts Gemini calls; `gate`, when set,
    blocks every call until released (to test "a refine already running").
    """

    def __init__(self, rewrites=None):
        self.rewrites = dict(rewrites or {})
        self.calls = 0
        self.prompts: list[str] = []
        self.gate: threading.Event | None = None
        self.error: Exception | None = None

    def __call__(self, prompt, system, as_json=False):
        if self.gate is not None:
            self.gate.wait(10)
        self.calls += 1
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        lines = json.loads(LINES_RE.search(prompt).group(1))
        return json.dumps({"lines": [
            {"line_id": l["line_id"], "refined_text": self.rewrites.get(l["text"], l["text"])} for l in lines
        ]})
