"""Generate the Experience Center sample images in app/static/samples/.

All three are drawn from scratch with Pillow (its bundled default font), so the repo
ships no third-party assets:

- lecture_notes.png   a printed lecture-notes page (headings, paragraphs, bullets)
- report_table.png    a title, a paragraph and a ruled table (exercises PP-StructureV3)
- noisy_scan.png      a slightly rotated, speckled, low-contrast "scan" (low-confidence lines)

Run:  venv/Scripts/python scripts/make_studio_samples.py
The output is deterministic (fixed random seed), so re-running gives the same files.
"""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT_DIR = Path(__file__).resolve().parents[1] / "app" / "static" / "samples"
MAX_BYTES = 200 * 1024  # keep each sample small: they are served to every visitor
INK = (28, 28, 30)
PAPER = (252, 251, 247)


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, width: int) -> list[str]:
    """Greedy word wrap to a pixel width."""
    lines, current = [], ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=fnt) <= width:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def paragraph(draw, x, y, text, fnt, width, leading=1.45, fill=INK) -> int:
    """Draw a wrapped paragraph; return the y below it."""
    step = int(fnt.size * leading)
    for line in wrap(draw, text, fnt, width):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += step
    return y


def lecture_notes() -> Image.Image:
    img = Image.new("RGB", (900, 1100), PAPER)
    d = ImageDraw.Draw(img)
    x, w = 70, 760
    d.text((x, 60), "Unit 3: Database Normalisation", font=font(34), fill=INK)
    d.text((x, 108), "Lecture 7 - Functional dependencies and normal forms", font=font(18), fill=(90, 90, 90))
    d.line((x, 140, x + w, 140), fill=(180, 175, 165), width=2)
    y = 165
    body = font(19)
    y = paragraph(d, x, y, (
        "A functional dependency X -> Y holds when every pair of rows that agree on the "
        "attributes in X also agree on Y. Normal forms use these dependencies to remove "
        "redundancy, which in turn prevents update, insert and delete anomalies."), body, w)
    y += 18
    d.text((x, y), "1. First normal form (1NF)", font=font(24), fill=INK)
    y += 40
    y = paragraph(d, x, y, (
        "Every attribute holds a single atomic value and every row is unique. Repeating "
        "groups such as phone1, phone2, phone3 are moved into a separate relation."), body, w)
    y += 18
    d.text((x, y), "2. Second normal form (2NF)", font=font(24), fill=INK)
    y += 40
    y = paragraph(d, x, y, (
        "The relation is in 1NF and no non-key attribute depends on only part of a "
        "composite key. Partial dependencies are split into their own tables."), body, w)
    y += 18
    d.text((x, y), "3. Third normal form (3NF)", font=font(24), fill=INK)
    y += 40
    y = paragraph(d, x, y, (
        "The relation is in 2NF and no non-key attribute depends on another non-key "
        "attribute (no transitive dependencies)."), body, w)
    y += 22
    d.text((x, y), "Key points to remember", font=font(22), fill=INK)
    y += 38
    for point in (
        "Normalise to reduce redundancy, then denormalise only for measured performance needs.",
        "BCNF is stricter than 3NF: every determinant must be a candidate key.",
        "Always list the functional dependencies before decomposing a relation.",
    ):
        d.ellipse((x + 4, y + 9, x + 12, y + 17), fill=INK)
        y = paragraph(d, x + 26, y, point, body, w - 26) + 6
    d.text((x, 1040), "Page 1 of 1", font=font(15), fill=(120, 120, 120))
    return img


def report_table() -> Image.Image:
    img = Image.new("RGB", (900, 760), PAPER)
    d = ImageDraw.Draw(img)
    x, w = 60, 780
    d.text((x, 50), "Experiment 4: Sorting Benchmarks", font=font(32), fill=INK)
    y = paragraph(d, x, 115, (
        "Each algorithm sorted one million random integers five times on the lab machine. "
        "The table lists the mean running time and the extra memory used. Merge sort is "
        "stable but needs a buffer the size of the input."), font(19), w)
    # Ruled table: header row + 4 data rows.
    rows = [
        ["Algorithm", "Mean time (ms)", "Extra memory", "Stable"],
        ["Quick sort", "84.2", "O(log n)", "No"],
        ["Merge sort", "97.5", "O(n)", "Yes"],
        ["Heap sort", "131.8", "O(1)", "No"],
        ["Insertion sort", "41250.0", "O(1)", "Yes"],
    ]
    cols = [x, x + 230, x + 430, x + 620, x + w]
    top, row_h = y + 40, 54
    for r, row in enumerate(rows):
        y0 = top + r * row_h
        if r == 0:
            d.rectangle((cols[0], y0, cols[-1], y0 + row_h), fill=(232, 239, 236))
        for c, text in enumerate(row):
            d.text((cols[c] + 14, y0 + 16), text, font=font(19) if r else font(20), fill=INK)
    bottom = top + len(rows) * row_h
    for r in range(len(rows) + 1):
        d.line((cols[0], top + r * row_h, cols[-1], top + r * row_h), fill=(90, 90, 90), width=2)
    for cx in cols:
        d.line((cx, top, cx, bottom), fill=(90, 90, 90), width=2)
    y = bottom + 30
    d.text((x, y), "Table 1. Sorting one million integers (mean of five runs).", font=font(16), fill=(90, 90, 90))
    y += 40
    paragraph(d, x, y, (
        "Conclusion: quick sort was fastest on random data, while insertion sort is only "
        "practical for small or nearly sorted inputs."), font(19), w)
    return img


def noisy_scan() -> Image.Image:
    rng = random.Random(7)
    img = Image.new("RGB", (900, 700), (246, 243, 234))
    d = ImageDraw.Draw(img)
    x, w = 70, 760
    d.text((x, 60), "Assignment 2 - Operating Systems", font=font(30), fill=(60, 58, 55))
    y = 125
    for text in (
        "Q1. Explain the difference between a process and a thread with one example.",
        "Q2. Draw the state diagram of a process and label every transition.",
        "Q3. What is a race condition? How does a mutex prevent it?",
        "Q4. Compare FCFS, SJF and Round Robin scheduling for the jobs below.",
        "Q5. Define deadlock and state the four Coffman conditions.",
    ):
        y = paragraph(d, x, y, text, font(20), w, fill=(70, 68, 64)) + 16
    paragraph(d, x, y + 10, "Submit by Friday, 5 pm. Late submissions lose 10 percent per day.",
              font(17), w, fill=(110, 106, 100))
    # Scanner artefacts: speckles, a faint fold line, slight blur and a small skew.
    for _ in range(2500):
        px, py = rng.randrange(900), rng.randrange(700)
        g = rng.randrange(90, 200)
        d.point((px, py), fill=(g, g, g))
    d.line((0, 420, 900, 432), fill=(222, 218, 208), width=3)
    img = img.filter(ImageFilter.GaussianBlur(0.8))
    img = img.rotate(-2.2, resample=Image.BICUBIC, expand=True, fillcolor=(210, 207, 200))
    return img.convert("L")  # greyscale scans compress well and look the part


def save(img: Image.Image, name: str) -> None:
    path = OUT_DIR / name
    img.save(path, optimize=True)
    size = path.stat().st_size
    if size > MAX_BYTES:  # fall back to a 64-colour palette if a sample grew too large
        img.convert("RGB").quantize(64).save(path, optimize=True)
        size = path.stat().st_size
    print(f"{path.relative_to(OUT_DIR.parents[2])}: {img.width}x{img.height}, {size / 1024:.0f} KB")
    if size > MAX_BYTES:
        raise SystemExit(f"{name} is larger than {MAX_BYTES // 1024} KB")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save(lecture_notes(), "lecture_notes.png")
    save(report_table(), "report_table.png")
    save(noisy_scan(), "noisy_scan.png")


if __name__ == "__main__":
    main()
