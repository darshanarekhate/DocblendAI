"""Automatic LLM refinement: diff, safety rule, plugin, upload -> extract -> refine -> QA,
fallback when Gemini is unavailable, retry, revert, versions, and the reliability cap.

Gemini (tests/refine_fakes.py), PaddleOCR (tests/paddle_fakes.py), TrOCR and embeddings are faked.
"""

import io
import json
import threading
import time
from pathlib import Path

import pytest

from app.config import settings
from app.models.schemas import ContentType, ReliabilityLabel, RetrievalResult
from app.modules import htr_extractor, qa_index, reliability, text_diff, vector_store
from app.plugins import PLUGINS, PluginError, PluginModelError
from tests.conftest import fake_embed
from tests.refine_fakes import FakeGemini
from tests.test_paddleocr_api import ec, post_ocr  # noqa: F401  (ec is a fixture)

FIXTURE_PNG = Path(__file__).parent / "fixtures" / "ppstructurev3_page.png"
TYPED_TEXT = "Retrieval-Augmented Generation combines a retriever with a language model."

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
    """Fake Gemini, with automatic refinement of uploads switched on (conftest turns it off)."""
    fake = FakeGemini(REWRITES)
    monkeypatch.setattr("app.plugins.refine.gemini", fake)
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_refine", True)
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


# --- automatic refinement of every upload ---


def lines_of(result, page=0):
    return result["pages"][page]["lines"]


def test_upload_is_refined_and_used_automatically(ec, gemini):
    run = run_png(ec)
    assert run["status"] == "done" and run["message"] == "Ready"
    ref = run["refinement"]
    assert ref["status"] == "refined" and ref["corrected"] == 2 and ref["message"].startswith("Refined by LLM: 2 lines corrected")
    lines = lines_of(run)
    corrected, inferred, rejected = lines[1], lines[2], lines[0]
    assert corrected["text"] == "This page compares raw and calibrated OCR confidence on the sample set."
    assert corrected["source"] == "llm" and corrected["llm_status"] == "corrected" and "OcR" in corrected["ocr_text"]
    assert [o for o in corrected["llm_diff"] if o["op"] != "equal"] == [{"op": "replace", "original": "OcR", "refined": "OCR"}]
    assert inferred["llm_status"] == "inferred" and "confidence scores" in inferred["text"]
    # Changing > 40% of the characters: rejected, recognised text kept and flagged.
    assert rejected["text"] == "Calibration Report" and rejected["needs_review"] is True and "limit 40%" in rejected["llm_flag"]
    assert lines[3]["source"] == "paddleocr" and "ocr_text" not in lines[3]
    # The refined text is what QA, exports and the Markdown use; the original is a stored version.
    texts = " ".join(vector_store.get_texts([run["doc_id"]]))
    assert "calibrated OCR confidence" in texts and "OcR" not in texts
    assert "calibrated OCR confidence" in ec.get(f"/api/results/{run['id']}/export/txt").text
    assert "OCR confidence" in run["markdown"]
    versions = ec.get(f"/api/results/{run['id']}/versions").json()
    assert [(v["kind"], v["label"]) for v in versions] == [
        ("extraction", "Original extraction"), ("refinement", "Gemini's proposal (2 changes)"),
        ("edit", "Refined by Gemini (2 corrections)"),
    ]
    original = ec.get(f"/api/results/{run['id']}/versions/{versions[0]['id']}").json()["data"]
    assert "OcR" in lines_of(original)[1]["text"]  # never overwritten


def test_progress_extracting_refining_ready(ec, gemini):
    gemini.gate = threading.Event()
    resp = post_ocr(ec, data=FIXTURE_PNG.read_bytes(), wait="false")
    job_id = resp.json()["id"]
    seen = []
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = ec.get(f"/api/results/{job_id}").json()
        if not seen or seen[-1] != job["message"]:
            seen.append(job["message"])
        if job["message"] == "Refining with Gemini…":
            break
        time.sleep(0.005)
    gemini.gate.set()
    final = wait_job(ec, job_id)
    assert "Refining with Gemini…" in seen and final["message"] == "Ready"
    assert all(m.startswith(("Queued", "Starting", "Extracting page", "Loading", "Preparing", "Refining")) for m in seen)


def test_gemini_unavailable_keeps_unrefined_text_and_qa(ec, monkeypatch):
    monkeypatch.setattr(settings, "llm_refine", True)
    monkeypatch.setattr(settings, "gemini_api_key", "")
    run = run_png(ec)
    ref = run["refinement"]
    assert ref["status"] == "unavailable" and ref["message"].startswith("Not refined: Gemini unavailable.")
    assert "GEMINI_API_KEY" in ref["message"]
    assert "OcR" in lines_of(run)[1]["text"] and run["qa"]["status"] == "ready"  # QA not blocked
    assert vector_store.get_texts([run["doc_id"]])
    assert ec.get(f"/api/results/{run['id']}/changes").status_code == 404


def test_quota_error_falls_back_then_retry_refines(ec, gemini):
    gemini.error = PluginModelError("Gemini's quota for gemini-x is used up for now (429).", 429)
    run = run_png(ec)
    assert run["refinement"]["status"] == "unavailable" and "quota" in run["refinement"]["message"]
    assert run["qa"]["status"] == "ready"

    gemini.error = None
    resp = ec.post(f"/api/results/{run['id']}/refine")
    assert resp.status_code == 202
    job = wait_job(ec, resp.json()["job_id"])
    assert job["status"] == "done" and job["pipeline"] == "refine"
    current = ec.get(f"/api/results/{run['id']}").json()
    assert current["refinement"]["status"] == "refined" and lines_of(current)[1]["source"] == "llm"
    assert current["qa"]["status"] == "ready"
    assert "OCR confidence" in " ".join(vector_store.get_texts([run["doc_id"]]))  # re-embedded


def test_retry_disabled_without_key_and_blocked_while_running(ec, gemini, monkeypatch):
    run = run_png(ec)
    gemini.gate = threading.Event()
    first = ec.post(f"/api/results/{run['id']}/refine")
    second = ec.post(f"/api/results/{run['id']}/refine")
    assert first.status_code == 202 and second.status_code == 409
    assert ec.get(f"/api/results/{run['id']}").json()["active_job"]["id"] == first.json()["job_id"]
    assert ec.post(f"/api/results/{run['id']}/revert").status_code == 409
    gemini.gate.set()
    assert wait_job(ec, first.json()["job_id"])["status"] == "done"
    monkeypatch.setattr(settings, "gemini_api_key", "")
    resp = ec.post(f"/api/results/{run['id']}/refine")
    assert resp.status_code == 503 and "GEMINI_API_KEY" in resp.json()["detail"]
    assert ec.post("/api/results/nope/refine").status_code in (404, 503)


def test_revert_reembeds_original_and_reapply(ec, gemini):
    run = run_png(ec)
    reverted = ec.post(f"/api/results/{run['id']}/revert").json()
    assert reverted["refinement"]["status"] == "reverted"
    assert "OcR" in lines_of(reverted)[1]["text"] and lines_of(reverted)[1]["source"] == "paddleocr"
    assert reverted["qa"]["status"] == "ready"
    texts = " ".join(vector_store.get_texts([run["doc_id"]]))
    assert "OcR confidence" in texts and "OCR confidence" not in texts  # QA answers from the original now

    again = ec.post(f"/api/results/{run['id']}/reapply").json()
    assert again["refinement"]["status"] == "refined" and lines_of(again)[1]["source"] == "llm"
    assert "OCR confidence" in " ".join(vector_store.get_texts([run["doc_id"]]))
    labels = [v["label"] for v in ec.get(f"/api/results/{run['id']}/versions").json()]
    assert labels[-2:] == ["Reverted to the original extraction", "Using the LLM-refined text again"]


def test_changes_lists_what_gemini_did(ec, gemini):
    run = run_png(ec)
    changes = ec.get(f"/api/results/{run['id']}/changes").json()
    by_id = {l["line_id"]: l for l in changes["data"]["lines"]}
    assert by_id["p0-l1"]["status"] == "corrected" and by_id["p0-l1"]["decision"] == "accepted"
    assert by_id["p0-l2"]["status"] == "inferred"
    assert by_id["p0-l0"]["status"] == "rejected" and by_id["p0-l0"]["decision"] == "n/a"
    assert changes["data"]["counts"] == {"rejected": 1, "corrected": 1, "inferred": 1, "unchanged": 12}


def test_compare_and_restore_versions(ec, gemini):
    run = run_png(ec)
    versions = ec.get(f"/api/results/{run['id']}/versions").json()
    changes = ec.get(f"/api/results/{run['id']}/versions/{versions[0]['id']}/compare").json()["changes"]
    assert [(c["line_id"], c["status"]) for c in changes] == [("p0-l1", "corrected"), ("p0-l2", "inferred")]
    restored = ec.post(f"/api/results/{run['id']}/versions/{versions[0]['id']}/restore").json()
    assert "OcR" in lines_of(restored)[1]["text"]
    assert ec.post(f"/api/results/{run['id']}/versions/{versions[1]['id']}/restore").status_code == 409


def test_manual_edit_after_refinement_drops_llm_marks(ec, gemini):
    run = run_png(ec)
    body = ec.patch(f"/api/results/{run['id']}/lines", json=[{"page": 0, "line": 1, "text": "Hand-checked line."}]).json()
    line = lines_of(body)[1]
    assert line["source"] == "human" and "llm_diff" not in line and line["ocr_text"].startswith("This page")
    assert "Hand-checked line." in " ".join(vector_store.get_texts([run["doc_id"]]))


def test_typed_docx_is_refined_too(ec, gemini, monkeypatch):
    from docx import Document as WordDocument

    from app.modules import paddleocr_service as svc

    monkeypatch.setattr(svc.PaddleOCRService, "convert_office", lambda self, p: "# Notes")
    doc = WordDocument()
    doc.add_paragraph("Second normal form removes partial dependncies.")
    buf = io.BytesIO()
    doc.save(buf)
    gemini.rewrites["Second normal form removes partial dependncies."] = "Second normal form removes partial dependencies."
    run = post_ocr(ec, "notes.docx", buf.getvalue()).json()
    assert ("ocr", "en") not in ec.factory.models  # exact text: no OCR at all
    line = lines_of(run)[0]
    assert line["source"] == "llm" and line["text"].endswith("dependencies.") and line["ocr_text"].endswith("dependncies.")


def test_handwritten_hint_reads_paddle_boxes_with_trocr(ec, monkeypatch):
    seen = []

    def fake_trocr(crops):
        seen.append(len(crops))
        return [(f"hand line {i}", 0.5) for i in range(len(crops))]

    monkeypatch.setattr(htr_extractor, "recognize_lines", fake_trocr)
    run = run_png(ec, format_hint="handwritten")
    assert run["format_type"] == "handwritten" and run["engine"] == "paddleocr+trocr"
    lines = lines_of(run)
    assert seen == [15] and lines[0]["text"] == "hand line 0" and lines[0]["source"] == "trocr"
    assert lines[0]["paddle_text"] == "Calibration Report" and lines[0]["box"]  # Paddle's line box kept


# --- the QA page: the same refined text, deletes on both sides ---


def test_qa_upload_is_refined_and_capped_at_moderate(client, pdf_file, gemini):
    page = "Normal forms remove redundancy from relations.\nSecond normal form removes partial dependncies."
    gemini.rewrites["Second normal form removes partial dependncies."] = "Second normal form removes partial dependencies."
    with pdf_file([page]).open("rb") as f:
        resp = client.post("/upload", files={"file": ("notes.pdf", f, "application/pdf")})
    assert resp.status_code == 201, resp.text
    doc_id = resp.json()["doc_id"]
    hits = vector_store.search(fake_embed(["partial dependencies"], "q")[0], 5, [doc_id])
    chunk = hits[0][0]
    assert "partial dependencies" in chunk.text  # answers come from the refined text
    assert chunk.calibrated_conf == reliability.LLM_TEXT_MAX_CONF  # typed (1.0) but LLM-touched: Moderate at most
    item = next(i for i in client.get("/documents/library").json() if i["doc_id"] == doc_id)
    assert item["refinement"]["status"] == "refined" and item["run_id"] and item["status"] == "ready"


def test_delete_from_either_page_removes_both(client, pdf_file):
    with pdf_file([TYPED_TEXT]).open("rb") as f:
        doc = client.post("/upload", files={"file": ("a.pdf", f, "application/pdf")}).json()
    item = next(i for i in client.get("/documents/library").json() if i["doc_id"] == doc["doc_id"])
    assert client.delete(f"/documents/{doc['doc_id']}").status_code == 204
    assert client.get(f"/api/results/{item['run_id']}").status_code == 404
    assert not vector_store.get_texts([doc["doc_id"]])

    with pdf_file([TYPED_TEXT], name="b.pdf").open("rb") as f:
        doc = client.post("/upload", files={"file": ("b.pdf", f, "application/pdf")}).json()
    run_id = next(i for i in client.get("/documents/library").json() if i["doc_id"] == doc["doc_id"])["run_id"]
    assert client.delete(f"/api/results/{run_id}").status_code == 200
    assert all(d["doc_id"] != doc["doc_id"] for d in client.get("/documents").json())


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
