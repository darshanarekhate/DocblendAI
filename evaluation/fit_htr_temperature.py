"""Fit TrOCR's softmax temperature (Module 3, temperature scaling).

Ayllon, Castellanos & Calvo-Zaragoza (ICDAR 2024, synopsis ref [3]) showed that
raw HTR confidence is overconfident and that temperature scaling fixes it.
This script fits that temperature on the handwritten calibration pages:

1. Each page is split into lines with the app's own segmenter (find_lines).
   A page is used only when it yields exactly as many lines as its ground
   truth, so every line image can be paired with its true text.
2. TrOCR scores the true text of every line (teacher forcing), and the
   temperature that minimizes the negative log-likelihood of the true tokens
   is chosen (confidence_capture.fit_temperature).
3. The script reports how far line confidence is from line accuracy
   (1 - CER) before and after, so the effect is visible.

The temperature is not written anywhere automatically: put the printed
HTR_TEMPERATURE line in .env, then refit the per-format tables with
`python -m evaluation.run_eval --fit-calibration` (they sit on top of it).

Run:  venv/Scripts/python -m evaluation.make_dataset   (once)
      venv/Scripts/python -m evaluation.fit_htr_temperature
"""

import json
import sys
from pathlib import Path

DATASET_DIR = Path(__file__).parent / "dataset"


def main() -> None:
    from app.config import settings
    from app.modules import confidence_capture, htr_extractor
    from app.modules.pdf_render import denoise, render_pages
    from tests.eval_metrics import cer

    manifest_path = DATASET_DIR / "dataset.json"
    if not manifest_path.exists():
        sys.exit("No dataset: run `python -m evaluation.make_dataset` first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    images, truths = [], []
    for entry in manifest["calibration"] + manifest["qa"]:
        if entry["format_type"] != "handwritten":
            continue
        pages = render_pages(str(DATASET_DIR / entry["file"]), htr_extractor.HTR_DPI)
        for page, truth in zip(pages, entry["ground_truth"]):
            lines = htr_extractor.find_lines(denoise(page))
            true_lines = [t for t in truth.split("\n") if t.strip()]
            if len(lines) == len(true_lines):
                images += lines
                truths += true_lines
            else:
                print(f"  skip {entry['name']}: {len(lines)} lines found, {len(true_lines)} expected")
    if not images:
        sys.exit("No page segmented into the expected number of lines; cannot pair lines with text.")
    print(f"Fitting on {len(images)} lines", flush=True)

    samples = htr_extractor.teacher_forced_logits(images, truths)
    temperature = confidence_capture.fit_temperature(samples)

    def calibration_error(t: float) -> float:
        settings.htr_temperature = t
        read = htr_extractor.recognize_lines(images)
        return sum(abs(conf - (1 - min(cer(truth, text), 1))) for (text, conf), truth in zip(read, truths)) / len(truths)

    before, after = calibration_error(1.0), calibration_error(temperature)
    print(f"NLL of true text: T=1.00 -> {confidence_capture._nll(samples, 1.0):.3f}, "
          f"T={temperature:.2f} -> {confidence_capture._nll(samples, temperature):.3f}")
    print(f"Mean |line confidence - line accuracy|: T=1.00 -> {before:.3f}, T={temperature:.2f} -> {after:.3f}")
    print(f"\nAdd to .env:\nHTR_TEMPERATURE={temperature:.2f}")


if __name__ == "__main__":
    main()
