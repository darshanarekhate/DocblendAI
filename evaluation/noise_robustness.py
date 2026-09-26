"""Noise-robustness evaluation, modeled on OHRBench (Zhang et al., ICCV 2025; synopsis ref [2]).

OHRBench showed that OCR noise cascades through RAG and can halve answer
quality. DocBlendAI's claim is that its reliability labels warn the user
when that happens. This script tests the claim directly:

1. Take the QA documents' ground-truth text (so the only errors are the ones
   injected here) and add OCR-like character noise at increasing levels
   (clean / mild / moderate / severe): look-alike substitutions (rn -> m,
   l -> 1, O -> 0 ...), dropped and doubled characters, merged words.
2. Give every chunk the confidence a perfectly calibrated recognizer would
   report: its measured character accuracy, 1 - CER against the clean text.
3. Run the real Modules 4-7 (content type, embedding, confidence-aware
   retrieval, reliability tier, LLM answer) and score every answerable question.

Good behavior: accuracy falls as noise rises, AND the labels fall with it
(fewer Certain, more Moderate/Uncertain/Unreadable), while the answers that
are still labeled Certain stay correct.

This reuses OHRBench's idea (graded, injected recognition noise), not its
dataset or code. Calls Gemini (embeddings + one answer per question per level).

Run:  venv/Scripts/python -m evaluation.make_dataset   (once)
      venv/Scripts/python -m evaluation.noise_robustness [--levels clean,severe]
"""

import argparse
import json
import random
import shutil
import sys
import uuid
from collections import Counter
from pathlib import Path

EVAL_DIR = Path(__file__).parent
DATASET_DIR = EVAL_DIR / "dataset"
RESULTS_DIR = EVAL_DIR / "results"
WORK_DIR = RESULTS_DIR / "work" / "noise"

# Share of characters perturbed at each level (roughly the CER that results).
LEVELS = {"clean": 0.0, "mild": 0.05, "moderate": 0.15, "severe": 0.30}

# Look-alike confusions typical of OCR/HTR output (both directions).
_CONFUSIONS = [("rn", "m"), ("cl", "d"), ("l", "1"), ("I", "l"), ("O", "0"), ("S", "5"), ("e", "c"),
               ("a", "o"), ("u", "v"), ("h", "b"), ("B", "8"), ("g", "q"), ("t", "f"), ("i", "j")]
_CONFUSE = {}
for a, b in _CONFUSIONS:
    _CONFUSE.setdefault(a, []).append(b)
    _CONFUSE.setdefault(b, []).append(a)


def inject_ocr_noise(text: str, rate: float, rng: random.Random) -> str:
    """Return text with about `rate` of its characters corrupted the way OCR/HTR corrupts them."""
    if rate <= 0:
        return text
    out: list[str] = []
    i = 0
    while i < len(text):
        pair = text[i : i + 2]
        if rng.random() >= rate:
            out.append(text[i])
            i += 1
            continue
        if pair in _CONFUSE:  # two-character look-alike (rn -> m, cl -> d)
            out.append(rng.choice(_CONFUSE[pair]))
            i += 2
            continue
        ch = text[i]
        roll = rng.random()
        if ch == " ":
            out.append("" if roll < 0.7 else "  ")  # merged words, or a split
        elif ch in _CONFUSE and roll < 0.6:
            out.append(rng.choice(_CONFUSE[ch]))
        elif roll < 0.8:
            pass  # dropped character
        else:
            out.append(ch + ch)  # doubled character
        i += 1
    return "".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--levels", default=",".join(LEVELS), help="comma-separated subset of " + ",".join(LEVELS))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    levels = [lv.strip() for lv in args.levels.split(",")]
    if unknown := set(levels) - set(LEVELS):
        sys.exit(f"unknown level(s): {', '.join(sorted(unknown))}")

    manifest_path = DATASET_DIR / "dataset.json"
    if not manifest_path.exists():
        sys.exit("No dataset: run `python -m evaluation.make_dataset` first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    from app.config import settings
    from app.models.schemas import Query
    from app.modules import (
        chunker, content_type, embedder, llm_answer, reliability, retrieval, vector_store,
    )
    from tests.eval_metrics import cer, contains_match, cosine

    def similarity(a: str, b: str) -> float:
        va, vb = embedder._embed([a, b], "SEMANTIC_SIMILARITY")
        return cosine(va, vb)

    rows = []
    for level in levels:
        rate = LEVELS[level]
        rng = random.Random(args.seed)
        # Fresh vector store per level: only this level's noisy text can be retrieved.
        shutil.rmtree(WORK_DIR / level, ignore_errors=True)
        settings.chroma_dir = WORK_DIR / level
        vector_store._collection.cache_clear()

        cers = []
        for entry in manifest["qa"]:
            doc_id = f"{entry['name']}-{level}"
            pages = []
            for truth in entry["ground_truth"]:
                noisy = inject_ocr_noise(truth, rate, rng)
                page_cer = min(cer(truth, noisy), 1.0)
                cers.append(page_cer)
                pages.append((noisy, 1.0 - page_cer))
            chunks = chunker.chunk_pages(doc_id, pages)
            # A perfectly calibrated recognizer: confidence = measured character accuracy.
            chunks = [c.model_copy(update={"calibrated_conf": c.raw_conf}) for c in chunks]
            vector_store.add_chunks(doc_id, embedder.embed_chunks(content_type.label_chunks(chunks)))
        mean_cer = sum(cers) / len(cers)
        print(f"\n[{level}] injected noise rate {rate:.2f}, measured mean CER {mean_cer:.3f}", flush=True)

        for entry in manifest["qa"]:
            for qa in entry["qa"]:
                query = Query(query_id=uuid.uuid4().hex, question_text=qa["question"], user_id="noise-eval")
                try:
                    hits = retrieval.retrieve(query, settings.top_k)
                    label = reliability.classify_reliability([r for _, r in hits])
                    answer = llm_answer.generate_answer(query, [c for c, _ in hits], label)
                except (embedder.EmbeddingError, llm_answer.LLMError) as e:
                    print(f"  [ERROR] {qa['question']}: {e}", flush=True)
                    continue
                sem = similarity(answer.answer_text, qa["answer"])
                correct = contains_match(answer.answer_text, qa["answer"]) or sem >= 0.8
                rows.append({
                    "level": level, "rate": rate, "cer": mean_cer, "doc": entry["name"],
                    "question": qa["question"], "reference": qa["answer"], "answer": answer.answer_text,
                    "label": answer.reliability_label.value, "correct": correct,
                })
                print(f"  [{answer.reliability_label.value:<10}] {'OK ' if correct else 'BAD'} {qa['question']}", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "noise_robustness.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    report = write_report(rows, levels)
    (RESULTS_DIR / "noise_robustness.md").write_text(report, encoding="utf-8")
    print("\n" + report)


def write_report(rows: list[dict], levels: list[str]) -> str:
    tiers = ["Certain", "Moderate", "Uncertain", "Unreadable"]
    lines = [
        "# Noise robustness (OHRBench-style)",
        "",
        "Generated by `python -m evaluation.noise_robustness`. Ground-truth text of the QA documents with injected "
        "OCR-like noise; chunk confidence = measured character accuracy. *Correct* = contains-match or "
        "Semantic Match (cosine >= 0.8).",
        "",
        "| Level | Mean CER | Questions | Correct | " + " | ".join(tiers) + " | Certain answers correct |",
        "|---|---|---|---|" + "---|" * len(tiers) + "---|",
    ]
    for level in levels:
        rs = [r for r in rows if r["level"] == level]
        if not rs:
            continue
        counts = Counter(r["label"] for r in rs)
        certain = [r for r in rs if r["label"] == "Certain"]
        certain_ok = f"{sum(r['correct'] for r in certain)}/{len(certain)}" if certain else "-"
        lines.append(
            f"| {level} | {rs[0]['cer']:.3f} | {len(rs)} | {100 * sum(r['correct'] for r in rs) / len(rs):.0f}% | "
            + " | ".join(str(counts.get(t, 0)) for t in tiers)
            + f" | {certain_ok} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
