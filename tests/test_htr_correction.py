"""Gemini correction of handwriting-recognition errors (htr_correction.py), with a fake Gemini."""

import pytest

from app.config import settings
from app.modules import htr_correction


@pytest.fixture
def gemini(monkeypatch):
    """Turn correction on and answer with `reply` instead of calling Gemini."""
    monkeypatch.setattr(settings, "htr_llm_correction", True)
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    state = {"reply": "", "calls": 0}

    def fake_ask(text):
        state["calls"] += 1
        return state["reply"]

    monkeypatch.setattr(htr_correction, "_ask", fake_ask)
    return state


def test_misread_words_are_corrected(gemini) -> None:
    gemini["reply"] = "1| Key concepts in Probabilistic Reasoning\n2| Bayesian Networks"
    assert htr_correction.correct_page("Key concepts in Pobaldlisric Reaswings\nbayeslan networks") == (
        "Key concepts in Probabilistic Reasoning\nBayesian Networks"
    )


def test_a_rewritten_line_keeps_the_recognized_text(gemini) -> None:
    """The model must not add its own content: a line changed beyond the limit is rejected."""
    gemini["reply"] = "1| a graphical model\n2| It is widely used in modern artificial intelligence research today."
    assert htr_correction.correct_page("a graphicai model\nused for search") == "a graphical model\nused for search"


def test_dropped_or_merged_lines_keep_the_recognized_text(gemini) -> None:
    gemini["reply"] = "1| line one\n3| line three fixed"  # the model merged line 2 away
    assert htr_correction.correct_page("line one\nline twoo\nline three fixd") == "line one\nline twoo\nline three fixed"


def test_unnumbered_reply_changes_nothing(gemini) -> None:
    gemini["reply"] = "completely different invented text"
    assert htr_correction.correct_page("line one\nline two") == "line one\nline two"


def test_lines_are_sent_numbered() -> None:
    assert htr_correction.numbered("first\nsecond") == "1| first\n2| second"


def test_failure_keeps_the_recognized_text(monkeypatch) -> None:
    monkeypatch.setattr(settings, "htr_llm_correction", True)

    def broken(text):
        raise RuntimeError("network down")

    monkeypatch.setattr(htr_correction, "_ask", broken)
    assert htr_correction.correct_page("some text") == "some text"


def test_switched_off_or_empty_page_does_not_call_gemini(gemini, monkeypatch) -> None:
    assert htr_correction.correct_page("   ") == "   "
    monkeypatch.setattr(settings, "htr_llm_correction", False)
    assert htr_correction.correct_page("text") == "text"
    assert gemini["calls"] == 0


def test_confidences_are_kept(gemini) -> None:
    gemini["reply"] = "1| fixed"
    assert htr_correction.correct_pages([("fixd", 0.42)]) == [("fixed", 0.42)]


def test_handwritten_upload_is_corrected_and_typed_is_not(client, pdf_file, fake_htr, gemini) -> None:
    from tests.test_multipage import _upload

    fake_htr.text, fake_htr.conf = "markov modeis are effective", 0.6
    gemini["reply"] = "1| markov models are effective"
    _upload(client, pdf_file([""], name="notes.pdf"), format_hint="handwritten")
    assert gemini["calls"] == 1
    _upload(client, pdf_file(["Typed text about TCP."], name="typed.pdf"))
    assert gemini["calls"] == 1  # typed documents are not sent to Gemini
