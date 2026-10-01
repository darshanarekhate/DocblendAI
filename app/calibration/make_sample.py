"""Supports Module 3 — generate the bundled calibration sample (evaluation/calibration_sample/).

Writes ~40 small PNGs, each with 2-4 lines of academic-sounding printed text
in varied fonts and sizes, plus labels.csv (image,text). Images get graded
degradations (blur, noise, low contrast, downscaling, slight rotation, JPEG
artefacts) so PaddleOCR misreads a share of the lines; a calibration set
needs both correct and incorrect lines. Everything is seeded, so the output
is reproducible on a machine with the same fonts.

Run:  venv/Scripts/python -m app.calibration.make_sample [--out DIR] [--count 40]
"""

import argparse
import csv
import io
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from app.config import BASE_DIR

OUT_DIR = BASE_DIR / "evaluation" / "calibration_sample"
SEED = 2026
FONT_FILES = ("arial.ttf", "times.ttf", "georgia.ttf", "verdana.ttf", "cour.ttf", "DejaVuSans.ttf", "DejaVuSerif.ttf")

SUBJECTS = [
    "The gradient descent update", "Each convolution layer", "The retrieval module", "Our baseline model",
    "The attention mechanism", "A hash table lookup", "The sorting algorithm", "Dynamic programming",
    "The proposed method", "The training loss", "Binary search", "The OCR engine", "The query encoder",
    "Stochastic sampling", "The confidence score", "A balanced binary tree", "The kernel function",
    "The database index", "Batch normalization", "The language model",
]
VERBS = [
    "reduces", "improves", "depends on", "converges to", "is bounded by", "estimates", "approximates",
    "requires", "minimizes", "outperforms", "is sensitive to", "preserves",
]
OBJECTS = [
    "the validation error", "the expected cost", "a local minimum", "the learning rate", "the cache misses",
    "the word error rate", "the eigenvalues", "the search space", "the recall at 10", "each memory access",
    "the prior distribution", "the input sequence", "the time complexity", "the noisy labels",
    "the character accuracy", "the matrix rank",
]
FACTS = [
    "Table {a} reports an F1 score of 0.{b}.", "See Section {a}.{c} for the proof.",
    "Accuracy rose from {b}% to {d}% in {e} epochs.", "Lemma {a} holds for all n > {c}.",
    "The dataset has {e},{b}0 labelled pages.", "Figure {a} plots ECE against bin count.",
    "Runtime is O(n log n) for n = {e}00.", "Experiment {a} used a batch size of {f}.",
]


def make_line(rng: random.Random) -> str:
    if rng.random() < 0.3:
        return rng.choice(FACTS).format(
            a=rng.randint(1, 9), b=rng.randint(10, 99), c=rng.randint(1, 9), d=rng.randint(50, 99),
            e=rng.randint(2, 40), f=rng.choice([16, 32, 64, 128]),
        )
    return f"{rng.choice(SUBJECTS)} {rng.choice(VERBS)} {rng.choice(OBJECTS)}."


def available_fonts() -> list[str]:
    found = []
    for name in FONT_FILES:
        try:
            ImageFont.truetype(name, 12)
            found.append(name)
        except OSError:
            continue
    return found


def load_font(name: str | None, size: int):
    if name:
        return ImageFont.truetype(name, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def render(lines: list[str], font_name: str | None, sizes: list[int]) -> Image.Image:
    fonts = [load_font(font_name, s) for s in sizes]
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    boxes = [probe.textbbox((0, 0), text, font=f) for text, f in zip(lines, fonts)]
    margin = 24
    width = max(b[2] for b in boxes) + 2 * margin
    height = sum(int(s * 1.7) for s in sizes) + 2 * margin
    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    y = margin
    for text, font, size in zip(lines, fonts, sizes):
        draw.text((margin, y), text, font=font, fill=0)
        y += int(size * 1.7)
    return image


def degrade(image: Image.Image, kinds: list[str], s: float, rng: random.Random) -> Image.Image:
    """Apply the named degradations at severity s in [0, 1]."""
    for kind in kinds:
        if kind == "contrast":  # faded ink: black text becomes light grey
            ink = int(40 + 175 * s)
            image = image.point(lambda v, ink=ink: ink + (255 - ink) * v // 255)
        elif kind == "rotate":
            angle = rng.choice([-1, 1]) * (0.5 + 3.5 * s)
            image = image.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=255)
        elif kind == "downscale":  # low-resolution scan, upsampled back
            factor = max(0.2, 1.0 - 0.8 * s)
            w, h = image.size
            small = image.resize((max(1, int(w * factor)), max(1, int(h * factor))), Image.BILINEAR)
            image = small.resize((w, h), Image.BILINEAR)
        elif kind == "blur":
            image = image.filter(ImageFilter.GaussianBlur(0.4 + 2.2 * s))
        elif kind == "noise":
            nprng = np.random.default_rng(rng.randint(0, 2**31))
            arr = np.asarray(image, dtype=float) + nprng.normal(0, 8 + 70 * s, (image.height, image.width))
            image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        elif kind == "jpeg":
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=max(4, int(60 - 56 * s)))
            image = Image.open(io.BytesIO(buffer.getvalue())).convert("L")
    return image


KINDS = ["blur", "noise", "contrast", "downscale", "rotate", "jpeg"]
ORDER = {"contrast": 0, "rotate": 1, "downscale": 2, "blur": 3, "noise": 4, "jpeg": 5}


def generate(out_dir: Path = OUT_DIR, count: int = 40, seed: int = SEED) -> list[tuple[str, str]]:
    """Write count PNGs + labels.csv to out_dir; returns the (image, text) rows."""
    rng = random.Random(seed)
    fonts = available_fonts() or [None]
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("sample_*.png"):
        old.unlink()
    rows = []
    for i in range(count):
        n_lines = rng.choice([2, 3, 3, 4])
        lines = [make_line(rng) for _ in range(n_lines)]
        sizes = [rng.choice([13, 15, 17, 20, 24, 28]) for _ in lines]
        image = render(lines, fonts[i % len(fonts)], sizes)
        severity = (i % 8) / 7  # graded: 0, 1/7, ..., 1 repeating
        kinds = sorted(rng.sample(KINDS, rng.choice([1, 2, 2, 3])), key=ORDER.get)
        image = degrade(image, kinds, severity, rng)
        name = f"sample_{i + 1:03d}.png"
        image.save(out_dir / name, format="PNG", optimize=True)
        rows.append((name, "\n".join(lines)))
    with open(out_dir / "labels.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "text"])
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the calibration sample dataset.")
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--count", type=int, default=40)
    args = parser.parse_args()
    rows = generate(Path(args.out), args.count)
    n_lines = sum(len(text.split("\n")) for _, text in rows)
    print(f"Wrote {len(rows)} images ({n_lines} lines) + labels.csv to {args.out}")


if __name__ == "__main__":
    main()
