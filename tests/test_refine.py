"""Refine with LLM: diff, safety rule, plugin, re-extract -> refine -> review flow, versions.

Gemini (tests/refine_fakes.py), PaddleOCR (tests/paddle_fakes.py) and TrOCR are faked.
"""

import io
import json
import threading
import time
from pathlib import Path

import pytest

from app.config import settings
from app.models.schemas import ContentType, ReliabilityLabel, RetrievalResult
from app.modules import htr_extractor, qa_index, reliability, text_diff
from app.plugins import PLUGINS, PluginError, PluginModelError
from tests.refine_fakes import FakeGemini
from tests.test_paddleocr_api import ec, post_ocr  # noqa: F401  (ec is a fixture)

FIXTURE_PNG = Path(__file__).parent / "fixtures" / "ppstructurev3_page.png"

REWRITES = {
    # corrected: one misread word replaced
    "This page compares raw and calibrated OcR confidence on the sample set.":
        "This page compares raw and calibrated OCR confidence on the sample set.",
    # inferred: a word restored from context
    "Lower expected calibration error means better calibrated scores.":
        "Lower expected calibration error means better calibrated confidence scores.",
    # rejected: rewrites far more than 40% of the line
    "Calibration Report": "A completely different and much longer heading",
}


@pytest.fixture
def gemini(monkeypatch):
    fake = FakeGemini(REWRITES)
    monkeypatch.setattr("app.plugins.refine.gemini", fake)
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    return fake


def wait_job(client, job_id, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/results/{job_id}").json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def run_png(client, **form):
    resp = post_ocr(client, data=FIXTURE_PNG.read_bytes(), **form)
    assert resp.status_code == 200, resp.text
    return resp.json()


def refine(client, run_id, body=None):
    resp = client.post(f"/api/results/{run_id}/refine", json=body)
    assert resp.status_code == 202, resp.text
    return wait_job(client, resp.json()["job_id"])


# --- diff and safety rule ---


def test_word_diff_ops_and_status():
    ops = text_diff.word_diff("teh cat sat", "the black cat sat")
    assert [(o["op"], o["original"], o["refined"]) for o in ops] == [
        ("replace", "teh", "the"), ("insert", "", "black"), ("equal", "cat", "cat"), ("equal", "sat", "sat"),
    ]
    assert text_diff.line_status(ops) == "inferred"
    assert text_diff.line_status(text_diff.word_diff("OcR score", "OCR score")) == "corrected"
    assert text_diff.line_status(text_diff.word_diff("a  b", "a b")) == "unchanged"
    assert [o["op"] for o in text_diff.word_diff("x noise y", "x y")] == ["equal", "delete", "equal"]
    # Splitting a merged word (or joining a broken one) is a correction, not an inferred word.
    split = text_diff.word_diff("Late submissions:lose 10", "Late submissions: lose 10")
    assert [(o["op"], o["original"], o["refined"]) for o in split if o["op"] != "equal"] == [
        ("replace", "submissions:lose", "submissions: lose")]
    assert text_diff.line_status(split) == "corrected"
    assert text_diff.line_status(text_diff.word_diff("calib ration error", "calibration error")) == "corrected"


def test_change_ratio_and_rejection():
    assert text_diff.change_ratio("abc", "abc") == 0
    assert text_diff.change_ratio("calibration", "calibratlon") == pytest.approx(1 / 11)
    ok = text_diff.compare("calibrated OcR confidence", "calibrated OCR confidence", 0.4)
    assert ok["status"] == "corrected" and ok["reason"] is None
    bad = text_diff.compare("Report", "A completely different heading", 0.4)
    assert bad["status"] == "rejected" and "limit 40%" in bad["reason"]
    assert text_diff.compare("", "invented sentence", 0.4)["status"] == "rejected"


# --- plugin ---


def test_refine_plugin_proposals(gemini):
    lines = [
        {"line_id": "a", "text": "This page compares raw and calibrated OcR confidence on the sample set.", "confidence": 0.4},
        {"line_id": "b", "text": "Calibration Report", "confidence": 0.9},
        {"line_id": "c", "text": "unchanged   spacing", "confidence": 0.99},
    ]
    gemini.rewrites["unchanged   spacing"] = "unchanged spacing"
    out = PLUGINS["refine"].run("", {"lines": lines, "context": "## Calibration Report"})
    by_id = {p["line_id"]: p for p in out["lines"]}
    assert by_id["a"]["status"] == "corrected" and by_id["a"]["refined"].endswith("OCR confidence on the sample set.")
    assert by_id["b"]["status"] == "rejected" and by_id["b"]["change_ratio"] > 0.4
    assert by_id["c"]["status"] == "unchanged" and by_id["c"]["refined"] == "unchanged   spacing"  # layout kept
    assert out["counts"] == {"corrected": 1, "rejected": 1, "unchanged": 1}
    prompt = gemini.prompts[0]
    assert "## Calibration Report" in prompt and '"confidence": 0.4' in prompt


def test_refine_plugin_splits_long_documents(gemini, monkeypatch):
    monkeypatch.setattr("app.plugins.refine.MAX_PART_LINES", 2)
    lines = [{"line_id": f"l{i}", "text": f"line {i}"} for i in range(5)]
    seen = []
    out = PLUGINS["refine"].run("", {"lines": lines, "progress": lambda done, total: seen.append((done, total))})
    assert gemini.calls == 3 and len(out["lines"]) == 5
    assert seen == [(0, 3), (1, 3), (2, 3), (3, 3)]


def test_refine_plugin_validation_and_model_errors(gemini):
    with pytest.raises(PluginError):
        PLUGINS["refine"].run("", {"lines": [{"line_id": "a"}]})
    with pytest.raises(PluginError):
        PLUGINS["refine"].run("", {"lines": [{"line_id": "a", "text": "x"}, {"line_id": "a", "text": "y"}]})
    with pytest.raises(PluginError):
        PLUGINS["refine"].run("", {"lines": [{"line_id": "a", "text": "x"}], "max_change": 2})
    gemini.error = PluginModelError("Gemini's quota for m is used up for now (429).", 429)
    with pytest.raises(PluginModelError) as exc:
        PLUGINS["refine"].run("", {"lines": [{"line_id": "a", "text": "x"}]})
    assert exc.value.status == 429 and gemini.calls == 1  # quota errors are not retried


def test_refine_plugin_retries_unusable_json_once(monkeypatch):
    replies = iter(["not json", '{"lines": [{"line_id": "a", "refined_text": "fixed"}]}'])
    monkeypatch.setattr("app.plugins.refine.gemini", lambda *a, **k: next(replies))
    out = PLUGINS["refine"].run("", {"lines": [{"line_id": "a", "text": "fixd"}]})
    assert out["lines"][0]["refined"] == "fixed"


# --- re-extract -> refine -> review ---


def test_refine_flow_reextracts_then_proposes(ec, gemini):
    run = run_png(ec)
    built_before = len(ec.factory.models[("ocr", "en")].calls)
    messages = []
    resp = ec.post(f"/api/results/{run['id']}/refine")
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    deadline = time.monotonic() + 15
    while True:
        job = ec.get(f"/api/results/{job_id}").json()
        messages.append(job["message"])
        if job["status"] in ("done", "error") or time.monotonic() > deadline:
            break
        time.sleep(0.005)
    assert job["status"] == "done", job
    assert job["message"] == "Ready for review" and job["pipeline"] == "refine" and job["run_id"] == run["id"]
    assert len(ec.factory.models[("ocr", "en")].calls) == built_before + 1  # re-extracted from the original

    versions = ec.get(f"/api/results/{run['id']}/versions").json()
    assert [(v["kind"], v["label"]) for v in versions][:2] == [("extraction", "Original extraction"), ("extraction", "Re-extraction")]
    assert versions[-1]["kind"] == "refinement" and versions[-1]["status"] == "pending"
    assert versions[-1]["counts"] == {"rejected": 1, "corrected": 1, "inferred": 1, "unchanged": 12}

    proposal = ec.get(f"/api/results/{run['id']}/refinement").json()
    lines = {l["line_id"]: l for l in proposal["data"]["lines"]}
    assert lines["p0-l1"]["status"] == "corrected" and lines["p0-l1"]["decision"] == "pending"
    assert [o for o in lines["p0-l1"]["diff"] if o["op"] != "equal"] == [{"op": "replace", "original": "OcR", "refined": "OCR"}]
    assert lines["p0-l2"]["status"] == "inferred"
    assert [o["refined"] for o in lines["p0-l2"]["diff"] if o["op"] == "insert"] == ["confidence"]
    assert lines["p0-l0"]["status"] == "rejected" and lines["p0-l0"]["decision"] == "n/a"
    assert ec.get(f"/api/results/{run['id']}").json()["active_job"] is None


def test_accept_per_line_then_reject_then_exports_and_history(ec, gemini):
    run = run_png(ec)
    refine(ec, run["id"])
    vid = ec.get(f"/api/results/{run['id']}/refinement").json()["id"]

    resp = ec.post(f"/api/results/{run['id']}/refinements/{vid}/accept", json={"line_ids": ["p0-l1"]})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    line = body["result"]["pages"][0]["lines"][1]
    assert line["text"] == "This page compares raw and calibrated OCR confidence on the sample set."
    assert line["edited"] is True and line["source"] == "llm" and line["ocr_text"].startswith("This page compares raw and calibrated OcR")
    assert body["proposal"]["status"] == "partial"
    untouched = body["result"]["pages"][0]["lines"][2]
    assert untouched["edited"] is False and "confidence scores" not in untouched["text"]

    resp = ec.post(f"/api/results/{run['id']}/refinements/{vid}/reject", json={"line_ids": ["p0-l2"]})
    assert resp.json()["proposal"]["status"] == "accepted"  # every change decided
    again = ec.post(f"/api/results/{run['id']}/refinements/{vid}/accept")
    assert again.status_code == 409

    txt = ec.get(f"/api/results/{run['id']}/export/txt").text
    assert "calibrated OCR confidence" in txt and "OcR" not in txt  # exports use accepted text

    versions = ec.get(f"/api/results/{run['id']}/versions").json()
    assert versions[-1]["kind"] == "edit" and versions[-1]["label"] == "Accepted 1 LLM correction"
    original = versions[0]["id"]
    changes = ec.get(f"/api/results/{run['id']}/versions/{original}/compare").json()["changes"]
    assert [(c["line_id"], c["status"]) for c in changes] == [("p0-l1", "corrected")]

    restored = ec.post(f"/api/results/{run['id']}/versions/{original}/restore").json()
    assert "OcR" in restored["pages"][0]["lines"][1]["text"]
    assert ec.get(f"/api/results/{run['id']}/versions").json()[-1]["label"] == "Restored version 1"


def test_accept_all_flags_rejected_lines(ec, gemini):
    run = run_png(ec)
    refine(ec, run["id"])
    vid = ec.get(f"/api/results/{run['id']}/refinement").json()["id"]
    body = ec.post(f"/api/results/{run['id']}/refinements/{vid}/accept").json()
    lines = body["result"]["pages"][0]["lines"]
    assert lines[0]["text"] == "Calibration Report" and lines[0]["needs_review"] is True
    assert "limit 40%" in lines[0]["llm_flag"]
    assert lines[1]["source"] == "llm" and lines[2]["source"] == "llm" and lines[2]["llm_status"] == "inferred"
    assert body["proposal"]["status"] == "accepted"
    assert ec.get(f"/api/results/{run['id']}/refinement").status_code == 404


def test_discard_keeps_current_text(ec, gemini):
    run = run_png(ec)
    refine(ec, run["id"])
    vid = ec.get(f"/api/results/{run['id']}/refinement").json()["id"]
    before = ec.get(f"/api/results/{run['id']}").json()["pages"][0]["lines"]
    body = ec.post(f"/api/results/{run['id']}/refinements/{vid}/discard").json()
    assert body["proposal"]["status"] == "discarded"
    assert ec.get(f"/api/results/{run['id']}").json()["pages"][0]["lines"] == before


def test_second_refine_is_blocked_while_one_runs(ec, gemini):
    run = run_png(ec)
    gemini.gate = threading.Event()
    first = ec.post(f"/api/results/{run['id']}/refine")
    assert first.status_code == 202
    second = ec.post(f"/api/results/{run['id']}/refine")
    assert second.status_code == 409 and "already running" in second.json()["detail"]
    assert ec.get(f"/api/results/{run['id']}").json()["active_job"]["id"] == first.json()["job_id"]
    assert ec.patch(f"/api/results/{run['id']}/lines", json=[{"page": 0, "line": 0, "text": "x"}]).status_code == 409
    gemini.gate.set()
    assert wait_job(ec, first.json()["job_id"])["status"] == "done"


def test_refine_disabled_without_key(ec, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "")
    run = run_png(ec)
    resp = ec.post(f"/api/results/{run['id']}/refine")
    assert resp.status_code == 503 and "GEMINI_API_KEY" in resp.json()["detail"]
    llm = ec.get("/api/health").json()["llm"]
    assert llm["enabled"] is False and "GEMINI_API_KEY" in llm["message"]


def test_quota_error_is_reported_on_the_job(ec, gemini):
    run = run_png(ec)
    gemini.error = PluginModelError("Gemini's quota for gemini-x is used up for now (429).", 429)
    job = refine(ec, run["id"])
    assert job["status"] == "error"
    assert job["error"].startswith("Re-extracted the document, but refining failed: Gemini's quota")


def test_refine_unknown_run_and_bad_options(ec, gemini):
    assert ec.post("/api/results/nope/refine").status_code == 404
    run = run_png(ec)
    assert ec.post(f"/api/results/{run['id']}/refine", json={"preprocess": {"x": True}}).status_code == 400
    assert ec.post(f"/api/results/{run['id']}/refine", json={"pipeline": "magic"}).status_code == 400


def test_reextraction_applies_cleanup_into_its_own_image_folder(ec, gemini):
    run = run_png(ec)
    refine(ec, run["id"], {"preprocess": {"binarize": True}})
    current = ec.get(f"/api/results/{run['id']}").json()
    url = current["pages"][0]["image_url"]
    assert url.endswith("?x=2") and current["preprocess"] == ["binarize"]
    assert ec.get(url).status_code == 200
    assert (settings.paddle_runs_dir / run["id"] / "x2" / "page-0.png").is_file()
    assert (settings.paddle_runs_dir / run["id"] / "page-0.png").is_file()  # version 1's image is kept


def test_typed_docx_reextracts_exact_text(ec, gemini, monkeypatch):
    from docx import Document as WordDocument

    from app.modules import paddleocr_service as svc

    monkeypatch.setattr(svc.PaddleOCRService, "convert_office", lambda self, p: "# Notes")
    doc = WordDocument()
    doc.add_paragraph("Second normal form removes partial dependncies.")
    buf = io.BytesIO()
    doc.save(buf)
    run = post_ocr(ec, "notes.docx", buf.getvalue()).json()
    gemini.rewrites["Second normal form removes partial dependncies."] = "Second normal form removes partial dependencies."
    assert refine(ec, run["id"])["status"] == "done"
    assert ("ocr", "en") not in ec.factory.models  # exact text: no OCR at all
    proposal = ec.get(f"/api/results/{run['id']}/refinement").json()["data"]
    assert proposal["lines"][0]["status"] == "corrected"


def test_handwritten_hint_reads_paddle_boxes_with_trocr(ec, gemini, monkeypatch):
    seen = []

    def fake_trocr(crops):
        seen.append(len(crops))
        return [(f"hand line {i}", 0.5) for i in range(len(crops))]

    monkeypatch.setattr(htr_extractor, "recognize_lines", fake_trocr)
    run = run_png(ec, format_hint="handwritten")
    assert run["format_type"] == "handwritten" and run["engine"] == "paddleocr+trocr"
    lines = run["pages"][0]["lines"]
    assert seen == [15] and lines[0]["text"] == "hand line 0" and lines[0]["source"] == "trocr"
    assert lines[0]["paddle_text"] == "Calibration Report" and lines[0]["box"]  # Paddle's line box kept
    job = refine(ec, run["id"])
    assert job["status"] == "done" and seen == [15, 15]  # re-extraction read handwriting again


def test_runs_without_original_file_refine_current_text(ec, gemini, db_session_factory):
    from app.db.models import OCRRunORM

    run = run_png(ec)
    with db_session_factory() as db:
        row = db.get(OCRRunORM, run["id"])
        result = json.loads(row.result_json)
        result.pop("source_file")
        row.result_json = json.dumps(result)
        db.commit()
    job = refine(ec, run["id"])
    assert job["status"] == "done" and job["message"].startswith("The original file of this run was not kept")


# --- reliability: LLM text is at most Moderate ---


def _result(lines, text=None, blocks=None, tables=None):
    return {"pages": [{"index": 0, "lines": lines, "text": text, "blocks": blocks or [], "tables": tables or [], "markdown": ""}]}


def _line(text, conf, source="paddleocr", box=None, edited=False):
    return {"id": text, "text": text, "raw_confidence": conf, "calibrated_confidence": conf, "source": source,
            "box": box, "edited": edited}


def test_llm_lines_never_raise_confidence_above_moderate():
    clean = qa_index.build_chunks("d", _result([_line("Normal forms remove redundancy", 0.97)]))
    assert clean[0][1].calibrated_conf == pytest.approx(0.97)
    refined = qa_index.build_chunks("d", _result([
        _line("Normal forms remove redundancy", 0.97), _line("second normal form", 0.40, "llm", edited=True),
    ]))
    chunk = refined[0][1]
    assert chunk.calibrated_conf <= reliability.LLM_TEXT_MAX_CONF < reliability.CERTAIN_MIN_CONF
    label = reliability.classify_reliability([RetrievalResult(chunk_id="d:0", similarity=0.9,
                                                              confidence=chunk.calibrated_conf, combined_score=0.9)])
    assert label is not ReliabilityLabel.CERTAIN
    # A high-confidence line the LLM touched is capped too.
    only_llm = qa_index.build_chunks("d", _result([_line("Normal forms", 0.99, "llm-auto", edited=True)]))
    assert only_llm[0][1].calibrated_conf == reliability.LLM_TEXT_MAX_CONF


def test_exact_page_text_is_used_until_a_line_is_edited():
    lines = [_line("Method  ECE", 1.0, "text"), _line("Raw  0.081", 1.0, "text")]
    chunks = qa_index.build_chunks("d", _result(lines, text="Method  ECE\nRaw  0.081\n\nfooter"))
    assert chunks[0][1].text == "Method  ECE\nRaw  0.081\n\nfooter"
    lines[1]["edited"] = True
    lines[1]["text"] = "Raw  0.080"
    assert qa_index.build_chunks("d", _result(lines, text="ignored now"))[0][1].text == "Method  ECE\nRaw  0.080"


def test_structure_blocks_give_content_types():
    from tests.paddle_fakes import STRUCTURE_MARKDOWN, STRUCTURE_RES
    from app.modules import paddleocr_service as svc

    page = svc.normalise_structure_page(STRUCTURE_RES, STRUCTURE_MARKDOWN, 0)
    page["text"] = None
    chunks = qa_index.build_chunks("d", {"pages": [page]})
    types = [c.content_type for _, c in chunks]
    assert types == [ContentType.PARAGRAPH, ContentType.TABLE]
    assert chunks[0][1].text.startswith("Calibration Report")
    assert chunks[1][1].text.splitlines()[0] == "Method  ECE  Brier"
