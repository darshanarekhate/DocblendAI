"""Build step 5 tests: content types (Module 4), combined_score ranking (Module 5),
reliability tiers (Module 6), and confidence-aware prompting (Module 7)."""

import pytest

from app.models.schemas import ContentType, Query, RecognizedChunk, ReliabilityLabel, RetrievalResult
from app.modules import content_type, llm_answer, reliability, retrieval, vector_store
from tests.conftest import fake_embed

# --- Module 4: content-type identification ------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Name  Roll No  Marks\nAsha  12  91\nRavi  13  84\nMeena  14  77",  # wide-gap columns
        "Name | Marks | Grade\nAsha | 91 | A\nRavi | 84 | B",  # pipe-separated
        "Epoch Loss Accuracy\n1 0.92 61.2\n2 0.54 74.8\n3 0.31 83.5",  # parsed PDF: single spaces, numeric
        "Name Marks\nAsha 91\nRavi 84\nMeena 77",  # parsed two-column table: label + value rows
    ],
)
def test_module4_detects_tables(text) -> None:
    assert content_type.classify(text) is ContentType.TABLE


@pytest.mark.parametrize(
    "text",
    [
        "Figure 3: Accuracy of the model across epochs.",
        "Fig. 2 Architecture of the retrieval pipeline",
        "Input\nEncoder\nAttention\nDecoder\nOutput",  # diagram labels
    ],
)
def test_module4_detects_figure_text_as_image(text) -> None:
    assert content_type.classify(text) is ContentType.IMAGE


@pytest.mark.parametrize(
    "text",
    [
        "Retrieval-augmented generation combines a retriever with a language model.\n"
        "The retriever selects relevant passages and the model answers from them.\n"
        "This grounds the answer in the source documents.",
        "A single short line.",
        "In 2023, 45 students scored above 90 in the exam.",  # numbers inside prose
        "Key points:\nRetrieval\nGeneration\nEvaluation is done with CER and WER.",  # short lines, but prose
    ],
)
def test_module4_prose_is_paragraph(text) -> None:
    assert content_type.classify(text) is ContentType.PARAGRAPH


def test_module4_label_chunks_sets_content_type() -> None:
    chunks = [RecognizedChunk(chunk_id="d:0", text="Figure 1: Overview", raw_conf=1.0)]
    assert content_type.label_chunks(chunks)[0].content_type is ContentType.IMAGE
    assert chunks[0].content_type is ContentType.PARAGRAPH  # original untouched


def test_module4_upload_stores_content_type(client, pdf_file) -> None:
    table = "Name  Marks\nAsha  91\nRavi  84\nMeena  77"
    with pdf_file([table]).open("rb") as f:
        assert client.post("/upload", files={"file": ("t.pdf", f, "application/pdf")}).status_code == 201

    [(chunk, _)] = vector_store.search(fake_embed(["name marks asha"], "")[0], top_k=5)
    assert chunk.content_type is ContentType.TABLE


# --- Module 5: confidence-aware retrieval --------------------------------------------


def test_module5_combined_score_is_weighted_sum(monkeypatch) -> None:
    monkeypatch.setattr(retrieval.settings, "similarity_weight", 0.7)
    assert retrieval.combined_score(0.8, 0.5) == pytest.approx(0.7 * 0.8 + 0.3 * 0.5)


def _stored(chunk_id: str, text: str, conf: float, vector_text: str | None = None) -> RecognizedChunk:
    return RecognizedChunk(
        chunk_id=chunk_id,
        text=text,
        raw_conf=conf,
        calibrated_conf=conf,
        vector=fake_embed([vector_text or text], "")[0],
    )


def test_module5_confident_chunk_outranks_slightly_more_similar_illegible_one(chroma, fake_embeddings) -> None:
    # The same passage read twice: the illegible (garbled) copy has one extra matching word -> higher similarity.
    vector_store.add_chunks(
        "doc",
        [
            _stored("doc:0", "exarn dates rnarch", 0.30, vector_text="exam dates march april"),
            _stored("doc:1", "exam dates march", 0.95),
        ],
    )

    hits = retrieval.retrieve(Query(query_id="q", question_text="exam dates march april", user_id="u"), top_k=2)

    (first, r1), (_, r2) = hits
    assert r2.similarity > r1.similarity  # the illegible one matched better...
    assert first.chunk_id == "doc:1"  # ...but the legible one ranks first
    assert r1.combined_score > r2.combined_score


def test_module5_retrieve_reranks_a_wider_candidate_pool(chroma, fake_embeddings, monkeypatch) -> None:
    monkeypatch.setattr(retrieval.settings, "candidate_multiplier", 3)
    calls = []
    original = vector_store.search
    monkeypatch.setattr(vector_store, "search", lambda vec, k, *rest: calls.append(k) or original(vec, k, *rest))

    retrieval.retrieve(Query(query_id="q", question_text="anything", user_id="u"), top_k=4)

    assert calls == [12]


# --- Module 6: reliability tiers ------------------------------------------------------


def _result(similarity: float, confidence: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id="c",
        similarity=similarity,
        confidence=confidence,
        combined_score=retrieval.combined_score(similarity, confidence),
    )


@pytest.mark.parametrize(
    ("similarity", "confidence", "expected"),
    [
        (0.80, 1.00, ReliabilityLabel.CERTAIN),  # typed, strong match
        (0.80, 0.70, ReliabilityLabel.MODERATE),  # strong match, partly legible
        (0.62, 1.00, ReliabilityLabel.MODERATE),  # typed, loose match
        (0.80, 0.50, ReliabilityLabel.UNCERTAIN),  # strong match, poorly legible
        (0.55, 1.00, ReliabilityLabel.UNCERTAIN),  # typed, off-topic
        (0.90, 0.20, ReliabilityLabel.UNREADABLE),  # relevant but illegible
    ],
)
def test_module6_tier_weighs_relevance_and_legibility(similarity, confidence, expected) -> None:
    assert reliability.classify_reliability([_result(similarity, confidence)]) is expected


def test_module6_judges_the_top_ranked_result() -> None:
    results = [_result(0.55, 1.0), _result(0.85, 0.95)]  # second one ranks higher
    assert reliability.classify_reliability(results) is ReliabilityLabel.CERTAIN


def test_module6_thresholds_are_inclusive() -> None:
    at_edge = _result(reliability.CERTAIN_MIN_SIM, reliability.CERTAIN_MIN_CONF)
    assert reliability.classify_reliability([at_edge]) is ReliabilityLabel.CERTAIN


# --- Module 7: confidence-aware prompt ------------------------------------------------


def test_module7_prompt_flags_low_confidence_and_content_type() -> None:
    chunks = [
        RecognizedChunk(chunk_id="a", text="Typed prose.", raw_conf=1.0, calibrated_conf=1.0),
        RecognizedChunk(
            chunk_id="b", text="Asha 91", content_type=ContentType.TABLE, raw_conf=0.6, calibrated_conf=0.62
        ),
    ]

    prompt = llm_answer.build_prompt("Marks?", chunks)

    assert "[1] (paragraph)\nTyped prose." in prompt
    assert "[2] (table; recognized text, confidence 0.62: may contain recognition errors)\nAsha 91" in prompt
