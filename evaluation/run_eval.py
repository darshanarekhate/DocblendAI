"""Run the DocBlendAI evaluation (build step 6).

Phases:
1. Recognition: format-detection accuracy and per-page CER / WER against the
   ground truth, for the QA and calibration documents.
2. Calibration: fit a per-format (raw_conf -> 1 - CER) table on the
   calibration set only, then check it on the held-out QA pages.
   With --fit-calibration the tables are saved to settings.calibration_file
   (data/calibration.json) so the app uses them.
3. Question answering: upload the QA documents through the real API (real
   Gemini, real OCR/HTR), ask every question, and score Exact Match,
   contains-match, Semantic Match, retrieval hits, and reliability tiers.

Everything runs against a throwaway database / ChromaDB / upload folder under
evaluation/results/work, never data/. Results go to evaluation/results/.

Run:  venv/Scripts/python -m evaluation.run_eval [--fit-calibration]
"""

import argparse
import json
import os
import shutil
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

EVAL_DIR = Path(__file__).parent
DATASET_DIR = EVAL_DIR / "dataset"
RESULTS_DIR = EVAL_DIR / "results"
WORK_DIR = RESULTS_DIR / "work"


def _isolate_storage() -> None:
    """Point the app at throwaway storage. Must run before any app import (engine is built at import)."""
    shutil.rmtree(WORK_DIR, ignore_errors=True)
    WORK_DIR.mkdir(parents=True)
    os.environ["DATABASE_URL"] = f"sqlite:///{(WORK_DIR / 'eval.db').as_posix()}"
    os.environ["CHROMA_DIR"] = str(WORK_DIR / "chroma")
    os.environ["UPLOAD_DIR"] = str(WORK_DIR / "uploads")


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def _pct(xs) -> str:
    xs = list(xs)
    return f"{100 * sum(xs) / len(xs):.0f}%" if xs else "-"


# --- phase 1: recognition ------------------------------------------------------------------


def evaluate_recognition(entries: list[dict], use_true_format: bool) -> list[dict]:
    from app.models.schemas import Document, FormatType
    from app.modules import confidence_capture, format_detection
    from tests.eval_metrics import cer, wer

    rows = []
    for entry in entries:
        path = str(DATASET_DIR / entry["file"])
        true_fmt = FormatType(entry["format_type"])
        start = time.time()
        detected = format_detection.detect_format(path)
        # QA docs follow the real pipeline (detected format); calibration docs use the true
        # extractor, since each table calibrates one extractor.
        used = true_fmt if use_true_format else detected
        pages = format_detection.extract(Document(doc_id="x", file_path=path, format_type=used, page_count=0))
        elapsed = time.time() - start
        for i, ((text, raw_conf), truth) in enumerate(zip(pages, entry["ground_truth"])):
            rows.append(
                {
                    "doc": entry["name"],
                    "page": i + 1,
                    "format": true_fmt.value,
                    "detected": detected.value,
                    "extractor": used.value,
                    "cer": cer(truth, text),
                    "wer": wer(truth, text),
                    "raw_conf": raw_conf,
                    "calibrated_conf": confidence_capture.calibrate(raw_conf, used),
                    "seconds": elapsed / len(pages),
                    "text": text,
                }
            )
        print(f"  {entry['name']:<30} detected={detected.value:<11} {elapsed:5.1f}s", flush=True)
    return rows


# --- phase 2: calibration --------------------------------------------------------------------


def fit_and_check_calibration(cal_rows: list[dict], qa_rows: list[dict], save: bool) -> dict:
    from app.models.schemas import FormatType
    from app.modules import confidence_capture

    report = {}
    for fmt in (FormatType.SCANNED, FormatType.HANDWRITTEN):
        samples = [(r["raw_conf"], 1 - min(r["cer"], 1.0)) for r in cal_rows if r["extractor"] == fmt.value]
        if not samples:
            continue
        knots = confidence_capture.fit_calibration(samples, bins=5)
        xs, ys = zip(*knots)

        def calibrated(raw, xs=xs, ys=ys):
            import numpy as np

            return float(np.interp(raw, xs, ys))

        held_out = [r for r in qa_rows if r["extractor"] == fmt.value]
        report[fmt.value] = {
            "knots": knots,
            "n_fit_pages": len(samples),
            "n_heldout_pages": len(held_out),
            # Mean |confidence - actual accuracy| on pages NOT used for fitting: lower is better calibrated.
            "heldout_error_raw": _mean(abs(r["raw_conf"] - (1 - min(r["cer"], 1))) for r in held_out),
            "heldout_error_calibrated": _mean(
                abs(calibrated(r["raw_conf"]) - (1 - min(r["cer"], 1))) for r in held_out
            ),
        }
        if save:
            confidence_capture.save_calibration(fmt, knots)
    return report


# --- phase 3: question answering ---------------------------------------------------------


def evaluate_qa(manifest: dict, errors_seen: list[dict]) -> list[dict]:
    from fastapi.testclient import TestClient

    from app.main import app
    from app.modules import embedder, llm_answer
    from tests.eval_metrics import contains_match, cosine, exact_match

    def similarity(a: str, b: str) -> float:
        va, vb = embedder._embed([a, b], "SEMANTIC_SIMILARITY")
        return cosine(va, vb)

    rows = []
    with TestClient(app) as client:  # runs the app lifespan: creates the isolated DB
        doc_ids = {}
        for entry in manifest["qa"]:
            with (DATASET_DIR / entry["file"]).open("rb") as f:
                resp = client.post("/upload", files={"file": (entry["name"] + ".pdf", f, "application/pdf")})
            resp.raise_for_status()
            doc_ids[entry["name"]] = resp.json()["doc_id"]
            print(f"  uploaded {entry['name']} as {resp.json()['format_type']}", flush=True)

        questions = [(e["name"], e["format_type"], q) for e in manifest["qa"] for q in e["qa"]]
        questions += [(None, None, q) for q in manifest["unanswerable"]]
        for doc_name, fmt, qa in questions:
            body = {"query_id": uuid.uuid4().hex, "question_text": qa["question"], "user_id": "eval"}
            resp = client.post("/ask", json=body)
            if resp.status_code != 200:
                # e.g. 502 when Gemini's quota runs out: record it and keep the rest of the run.
                detail = resp.json().get("detail", resp.text)
                print(f"  [ERROR {resp.status_code}] {qa['question']}: {detail}", flush=True)
                errors_seen.append({"question": qa["question"], "status": resp.status_code, "detail": detail})
                continue
            answer = resp.json()
            sources = client.get(f"/answer/{answer['answer_id']}/sources").json()
            top = sources[0] if sources else None

            not_found = answer["answer_text"] in (llm_answer.NOT_FOUND_ANSWER, llm_answer.NO_DOCUMENTS_ANSWER)
            if qa["answer"]:
                sem = similarity(answer["answer_text"], qa["answer"])
                row = {
                    "exact_match": exact_match(answer["answer_text"], qa["answer"]),
                    "contains_match": contains_match(answer["answer_text"], qa["answer"]),
                    "semantic_similarity": sem,
                    "semantic_match": sem >= 0.8,
                    "retrieval_hit": bool(top) and top["doc_id"] == doc_ids[doc_name],
                }
                row["correct"] = row["contains_match"] or row["semantic_match"]
            else:
                row = {"correct": not_found, "exact_match": None, "contains_match": None,
                       "semantic_similarity": None, "semantic_match": None, "retrieval_hit": None}
            rows.append(
                {
                    "doc": doc_name or "(unanswerable)",
                    "format": fmt or "-",
                    "question": qa["question"],
                    "reference": qa["answer"] or "(not in documents)",
                    "answer": answer["answer_text"],
                    "label": answer["reliability_label"],
                    "top_similarity": top["similarity"] if top else None,
                    "top_confidence": top["confidence"] if top else None,
                    "top_content_type": top["content_type"] if top else None,
                    **row,
                }
            )
            print(f"  [{answer['reliability_label']:<10}] {'OK ' if row['correct'] else 'BAD'} {qa['question']}", flush=True)
    return rows


# --- report ------------------------------------------------------------------------------------


def write_report(qa_rec, cal_rec, calibration, qa_rows, fitted: bool) -> str:
    lines = [
        "# DocBlendAI evaluation report",
        "",
        "Generated by `python -m evaluation.run_eval`. Dataset: `evaluation/make_dataset.py` "
        "(**synthetic** scans and handwriting-style fonts, fixed seed). Treat these numbers as a "
        "pipeline check; real student notes will be harder.",
        "",
        "## 1. Recognition (Module 2)",
        "",
        "| Format | Pages | Detection accuracy | Mean CER | Mean WER | Mean raw conf | Mean calibrated conf |",
        "|---|---|---|---|---|---|---|",
    ]
    by_fmt = defaultdict(list)
    for r in qa_rec + cal_rec:
        by_fmt[r["format"]].append(r)
    for fmt in ("typed", "scanned", "handwritten"):
        rs = by_fmt.get(fmt, [])
        if rs:
            lines.append(
                f"| {fmt} | {len(rs)} | {_pct(r['detected'] == fmt for r in rs)} | {_mean(r['cer'] for r in rs):.3f} "
                f"| {_mean(r['wer'] for r in rs):.3f} | {_mean(r['raw_conf'] for r in rs):.2f} "
                f"| {_mean(r['calibrated_conf'] for r in rs):.2f} |"
            )
    lines += ["", "Per document:", "", "| Document | Format | Detected as | CER | WER | Raw conf | s/page |", "|---|---|---|---|---|---|---|"]
    docs = defaultdict(list)
    for r in qa_rec + cal_rec:
        docs[r["doc"]].append(r)
    for name, rs in docs.items():
        lines.append(
            f"| {name} | {rs[0]['format']} | {rs[0]['detected']} | {_mean(r['cer'] for r in rs):.3f} "
            f"| {_mean(r['wer'] for r in rs):.3f} | {_mean(r['raw_conf'] for r in rs):.2f} | {_mean(r['seconds'] for r in rs):.1f} |"
        )

    lines += ["", "## 2. Confidence calibration (Module 3)", ""]
    lines.append(
        "Fitted on the calibration set only; error = mean |confidence − actual character accuracy| on the "
        "held-out QA pages (lower is better)."
        + (" Tables saved to `data/calibration.json`." if fitted else " (Not saved: run with `--fit-calibration`.)")
    )
    lines += ["", "| Format | Fit pages | Held-out pages | Error before | Error after | Knots (raw → calibrated) |", "|---|---|---|---|---|---|"]
    for fmt, c in calibration.items():
        knots = ", ".join(f"{x:.2f}→{y:.2f}" for x, y in c["knots"])
        lines.append(
            f"| {fmt} | {c['n_fit_pages']} | {c['n_heldout_pages']} | {c['heldout_error_raw']:.3f} "
            f"| {c['heldout_error_calibrated']:.3f} | {knots} |"
        )

    answerable = [r for r in qa_rows if r["exact_match"] is not None]
    unanswerable = [r for r in qa_rows if r["exact_match"] is None]
    lines += [
        "",
        "## 3. Question answering (Modules 5-7)",
        "",
        "*Correct* = contains-match or Semantic Match (cosine ≥ 0.8). Unanswerable questions are correct "
        "when the system says it could not find the answer.",
        "",
        "| Format | Questions | Exact Match | Contains | Semantic Match | Correct | Retrieval hit |",
        "|---|---|---|---|---|---|---|",
    ]
    qa_by_fmt = defaultdict(list)
    for r in answerable:
        qa_by_fmt[r["format"]].append(r)
    for fmt in ("typed", "scanned", "handwritten"):
        rs = qa_by_fmt.get(fmt, [])
        if rs:
            lines.append(
                f"| {fmt} | {len(rs)} | {_pct(r['exact_match'] for r in rs)} | {_pct(r['contains_match'] for r in rs)} "
                f"| {_pct(r['semantic_match'] for r in rs)} | {_pct(r['correct'] for r in rs)} | {_pct(r['retrieval_hit'] for r in rs)} |"
            )
    lines.append(
        f"| **all answerable** | {len(answerable)} | {_pct(r['exact_match'] for r in answerable)} "
        f"| {_pct(r['contains_match'] for r in answerable)} | {_pct(r['semantic_match'] for r in answerable)} "
        f"| {_pct(r['correct'] for r in answerable)} | {_pct(r['retrieval_hit'] for r in answerable)} |"
    )
    lines.append(f"| unanswerable | {len(unanswerable)} | - | - | - | {_pct(r['correct'] for r in unanswerable)} | - |")

    lines += [
        "",
        "## 4. Reliability tiers (Module 6)",
        "",
        "A useful label is one where **Certain answers are right more often than Moderate, and so on down**.",
        "",
        "| Label | Answers | Correct |",
        "|---|---|---|",
    ]
    for label in ("Certain", "Moderate", "Uncertain", "Unreadable"):
        rs = [r for r in qa_rows if r["label"] == label]
        lines.append(f"| {label} | {len(rs)} | {_pct(r['correct'] for r in rs)} |")

    lines += [
        "",
        "## 5. Every question",
        "",
        "| Doc | Question | Reference | Answer | Label | Correct | Top sim | Top conf |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in qa_rows:
        sim = f"{r['top_similarity']:.2f}" if r["top_similarity"] is not None else "-"
        conf = f"{r['top_confidence']:.2f}" if r["top_confidence"] is not None else "-"
        answer = r["answer"].replace("|", "/").replace("\n", " ")
        lines.append(
            f"| {r['doc']} | {r['question']} | {r['reference']} | {answer} | {r['label']} "
            f"| {'✓' if r['correct'] else '✗'} | {sim} | {conf} |"
        )
    return "\n".join(lines) + "\n"


def rescore() -> None:
    """Recompute reliability labels from the saved results with the current thresholds.

    For tuning reliability.py without re-running OCR/HTR or spending Gemini quota:
    each answer's label depends only on its top source's scores, plus the
    "couldn't find it" downgrade.
    """
    from app.models.schemas import ReliabilityLabel, RetrievalResult
    from app.modules import llm_answer, reliability, retrieval

    saved = json.loads((RESULTS_DIR / "results.json").read_text(encoding="utf-8"))
    for row in saved["qa"]:
        if row["top_similarity"] is None:
            label = ReliabilityLabel.UNCERTAIN
        else:
            sim, conf = row["top_similarity"], row["top_confidence"]
            label = reliability.classify_reliability(
                [RetrievalResult(chunk_id="-", similarity=sim, confidence=conf, combined_score=retrieval.combined_score(sim, conf))]
            )
        if row["answer"] in (llm_answer.NOT_FOUND_ANSWER, llm_answer.NO_DOCUMENTS_ANSWER):
            label = reliability.worse_of(label, ReliabilityLabel.UNCERTAIN)
        row["label"] = label.value

    rec = saved["recognition"]
    qa_rec = [r for r in rec if not r["doc"].startswith("cal_")]
    cal_rec = [r for r in rec if r["doc"].startswith("cal_")]
    (RESULTS_DIR / "results.json").write_text(json.dumps(saved, indent=2), encoding="utf-8")
    report = write_report(qa_rec, cal_rec, saved["calibration"], saved["qa"], fitted=True)
    (RESULTS_DIR / "report.md").write_text(report, encoding="utf-8")
    print(f"Rescored {len(saved['qa'])} answers. Report: {RESULTS_DIR / 'report.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fit-calibration", action="store_true", help="save fitted tables to data/calibration.json")
    parser.add_argument("--skip-qa", action="store_true", help="recognition + calibration only (no Gemini calls)")
    parser.add_argument(
        "--rescore", action="store_true", help="relabel saved results with current reliability thresholds (no API calls)"
    )
    args = parser.parse_args()
    if args.rescore:
        rescore()
        return

    manifest_path = DATASET_DIR / "dataset.json"
    if not manifest_path.is_file():
        sys.exit("No dataset found. Run: venv/Scripts/python -m evaluation.make_dataset")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    _isolate_storage()
    print("Phase 1: recognition (calibration set)", flush=True)
    cal_rec = evaluate_recognition(manifest["calibration"], use_true_format=True)
    print("Phase 1: recognition (QA set)", flush=True)
    qa_rec = evaluate_recognition(manifest["qa"], use_true_format=False)

    print("Phase 2: calibration", flush=True)
    calibration = fit_and_check_calibration(cal_rec, qa_rec, save=args.fit_calibration)
    if args.fit_calibration:  # re-score QA pages with the saved tables
        from app.models.schemas import FormatType
        from app.modules import confidence_capture

        for r in qa_rec:
            r["calibrated_conf"] = confidence_capture.calibrate(r["raw_conf"], FormatType(r["extractor"]))

    qa_rows, qa_errors = [], []
    if not args.skip_qa:
        print("Phase 3: question answering", flush=True)
        qa_rows = evaluate_qa(manifest, qa_errors)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "results.json").write_text(
        json.dumps(
            {"recognition": qa_rec + cal_rec, "calibration": calibration, "qa": qa_rows, "qa_errors": qa_errors},
            indent=2,
        ),
        encoding="utf-8",
    )
    report = write_report(qa_rec, cal_rec, calibration, qa_rows, args.fit_calibration)
    if qa_errors:
        report += "\n## Questions that failed\n\n" + "\n".join(
            f"- HTTP {e['status']}: {e['question']} ({e['detail']})" for e in qa_errors
        ) + "\n"
    (RESULTS_DIR / "report.md").write_text(report, encoding="utf-8")
    print(f"Report: {RESULTS_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
