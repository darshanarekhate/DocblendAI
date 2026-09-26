"""Tests for the evaluation metrics (build step 6)."""

import pytest

from tests.eval_metrics import contains_match, cer, cosine, exact_match, normalize_answer, semantic_match, wer


def test_normalize_answer_drops_case_punctuation_articles() -> None:
    assert normalize_answer("  The Newton! ") == "newton"


@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [
        ("Partial dependencies.", "partial dependencies", True),
        ("the newton", "Newton", True),
        ("443", "443", True),
        ("It removes partial dependencies.", "partial dependencies", False),
    ],
)
def test_exact_match(prediction, reference, expected) -> None:
    assert exact_match(prediction, reference) is expected


def test_contains_match_finds_short_answer_in_sentence() -> None:
    assert contains_match("The unit of force is the newton.", "the newton")
    assert not contains_match("The unit of force is the newton.", "joule")
    assert not contains_match("anything", "")  # empty reference never matches
    assert not contains_match("port 4430", "443")  # whole words only


def test_cer_and_wer() -> None:
    assert cer("hello world", "hello world") == 0.0
    assert cer("abcd", "abce") == pytest.approx(0.25)
    assert wer("the cat sat", "the cat sit") == pytest.approx(1 / 3)


def test_recognition_metrics_ignore_layout_whitespace() -> None:
    assert cer("line one\nline two", "line one line  two") == 0.0
    assert wer("line one\nline two", "line one\n\nline two") == 0.0


def test_nothing_recognized_is_full_error() -> None:
    assert cer("some text", "") == 1.0
    assert wer("some text", "   ") == 1.0


def test_semantic_match_uses_embedding_cosine() -> None:
    vectors = {"four layers": [1.0, 0.0], "4 layers": [0.9, 0.1], "port 443": [0.0, 1.0]}
    embed = lambda texts: [vectors[t] for t in texts]  # noqa: E731

    assert semantic_match("4 layers", "four layers", embed=embed)
    assert not semantic_match("port 443", "four layers", embed=embed)
    assert cosine([1, 0], [0, 1]) == 0.0
