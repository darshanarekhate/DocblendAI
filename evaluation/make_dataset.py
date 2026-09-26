"""Generate the synthetic evaluation dataset (typed, scanned, handwritten PDFs).

Every page's ground-truth text is known, so CER/WER can be measured exactly.
- typed: real text layer (same builder the tests use)
- scanned: printed font rendered to an image, then degraded (noise, blur, skew)
- handwritten: Windows handwriting-style fonts with per-line wobble

Synthetic handwriting is cleaner and more regular than real student notes;
treat results on it as a pipeline check, and add real scans to
evaluation/dataset/ for the final numbers.

Two sets are written to evaluation/dataset/:
- qa/: documents plus question/answer pairs (retrieval + answer quality)
- calibration/: extra scanned/handwritten pages at graded difficulty, used
  only to fit confidence calibration (kept separate from the QA set)

Run:  venv/Scripts/python -m evaluation.make_dataset
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from tests.conftest import make_pdf

DATASET_DIR = Path(__file__).parent / "dataset"
FONTS = Path("C:/Windows/Fonts")
DPI = 150
PAGE_W, PAGE_H = int(8.27 * DPI), int(11.69 * DPI)  # A4
MARGIN_X, MARGIN_TOP = 110, 140


@dataclass
class QA:
    question: str
    answer: str  # short reference answer; "" means the documents do not contain it


@dataclass
class Doc:
    name: str
    format_type: str
    pages: list[list[str]]  # page -> lines (the ground truth)
    style: dict = field(default_factory=dict)
    qa: list[QA] = field(default_factory=list)


# --- rendering ------------------------------------------------------------------------


def _degrade(page: Image.Image, level: float, rng: np.random.Generator) -> Image.Image:
    """Simulate a photocopy/scan: skew, blur, faded ink, speckle noise."""
    if level <= 0:
        return page
    page = page.rotate(rng.uniform(-0.6, 0.6) * level, fillcolor=255, resample=Image.BICUBIC)
    page = page.filter(ImageFilter.GaussianBlur(0.35 * level))
    arr = np.asarray(page).astype(float)
    arr = 255 - (255 - arr) * (1 - 0.12 * level)  # fade ink
    arr += rng.normal(0, 9 * level, arr.shape)
    speckles = rng.random(arr.shape) < 0.002 * level
    arr[speckles] = rng.uniform(0, 120, speckles.sum())
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def render_page(lines: list[str], style: dict, rng: np.random.Generator) -> Image.Image:
    font = ImageFont.truetype(str(FONTS / style["font"]), style["size"])
    page = Image.new("L", (PAGE_W, PAGE_H), 255)
    y = MARGIN_TOP
    wobble = style.get("wobble", 0.0)
    for line in lines:
        # Draw each line on its own strip so it can tilt independently, like handwriting.
        left, top, right, bottom = font.getbbox(line)
        strip = Image.new("L", (right + 40, bottom + 40), 255)
        ImageDraw.Draw(strip).text((20, 20 - top), line, font=font, fill=int(rng.integers(0, 40)))
        if wobble:
            strip = strip.rotate(rng.uniform(-1.2, 1.2) * wobble, expand=True, fillcolor=255, resample=Image.BICUBIC)
        x = MARGIN_X + int(rng.uniform(-12, 12) * wobble)
        page.paste(strip, (x, y + int(rng.uniform(-6, 6) * wobble)))
        y += int(style["size"] * style.get("spacing", 1.9))
    return _degrade(page, style.get("noise", 0.0), rng)


def write_doc(doc: Doc, out_dir: Path, rng: np.random.Generator) -> Path:
    path = out_dir / f"{doc.name}.pdf"
    if doc.format_type == "typed":
        path.write_bytes(make_pdf(["\n".join(lines) for lines in doc.pages]))
    else:
        images = [render_page(lines, doc.style, rng) for lines in doc.pages]
        images[0].save(path, save_all=True, append_images=images[1:], resolution=DPI)
    return path


# --- content -------------------------------------------------------------------------

PRINTED = {"font": "times.ttf", "size": 34, "spacing": 1.7}
NEAT_HAND = {"font": "Inkfree.ttf", "size": 44, "spacing": 2.1, "wobble": 0.6}

QA_DOCS = [
    Doc(
        "dbms_notes",
        "typed",
        [
            [
                "Unit 3: Database Normalization",
                "Normalization organizes a relational database to reduce redundancy",
                "and to avoid insertion, update, and deletion anomalies.",
                "First normal form requires every attribute to hold atomic values.",
                "Second normal form removes partial dependencies on a composite key.",
                "Third normal form removes transitive dependencies.",
            ],
            [
                "Boyce-Codd normal form is stricter than third normal form:",
                "every determinant must be a candidate key.",
                "Normal form Removes",
                "1NF 0 repeating groups",
                "2NF 1 partial dependencies",
                "3NF 2 transitive dependencies",
                "Denormalization is sometimes used to speed up read-heavy workloads.",
            ],
        ],
        qa=[
            QA("What does second normal form remove?", "partial dependencies"),
            QA("Which normal form requires every determinant to be a candidate key?", "Boyce-Codd normal form"),
            QA("Why is denormalization sometimes used?", "to speed up read-heavy workloads"),
        ],
    ),
    Doc(
        "os_scheduling",
        "typed",
        [
            [
                "Unit 2: CPU Scheduling",
                "First-Come, First-Served scheduling runs processes in arrival order.",
                "Shortest Job First minimises the average waiting time.",
                "Round Robin gives each process a fixed time quantum of CPU time.",
                "A very small time quantum causes too many context switches.",
            ],
            [
                "Example burst times in milliseconds",
                "Process Burst",
                "P1 24",
                "P2 3",
                "P3 3",
                "With FCFS the average waiting time for this example is 17 ms.",
            ],
        ],
        qa=[
            QA("Which scheduling algorithm minimises the average waiting time?", "Shortest Job First"),
            QA("What is the burst time of process P1?", "24 ms"),
            QA("What happens if the Round Robin time quantum is very small?", "too many context switches"),
        ],
    ),
    Doc(
        "computer_networks",
        "scanned",
        [
            [
                "Chapter 4: The TCP/IP Model",
                "The TCP/IP model has four layers: link, internet, transport",
                "and application.",
                "IP provides best-effort delivery of packets between hosts.",
                "TCP offers reliable, ordered delivery using acknowledgements.",
            ],
            [
                "UDP is connectionless and has lower overhead than TCP.",
                "DNS translates domain names into IP addresses.",
                "HTTP uses port 80 while HTTPS uses port 443.",
                "The three-way handshake uses SYN, SYN-ACK and ACK segments.",
            ],
        ],
        style={**PRINTED, "noise": 0.8},
        qa=[
            QA("How many layers does the TCP/IP model have?", "four"),
            QA("Which port does HTTPS use?", "443"),
            QA("What does DNS do?", "translates domain names into IP addresses"),
        ],
    ),
    Doc(
        "machine_learning_photocopy",
        "scanned",
        [
            [
                "Lecture 6: Overfitting and Regularization",
                "A model overfits when it memorises noise in the training data.",
                "Regularization adds a penalty on large weights to the loss.",
                "L1 regularization can drive some weights exactly to zero.",
            ],
            [
                "Dropout randomly disables neurons during training.",
                "Early stopping halts training when validation loss rises.",
                "Cross-validation estimates how well a model generalises.",
            ],
        ],
        style={**PRINTED, "size": 30, "noise": 3.2},  # a poor photocopy
        qa=[
            QA("What can L1 regularization do to weights?", "drive some weights exactly to zero"),
            QA("When does early stopping halt training?", "when validation loss rises"),
        ],
    ),
    Doc(
        "biology_notes",
        "handwritten",
        [
            [
                "Cell Biology - Lecture 1",
                "The mitochondria makes energy as ATP",
                "Ribosomes build proteins from amino acids",
                "The nucleus stores the genetic material",
            ],
            [
                "Plant cells have a cell wall made of cellulose",
                "Chloroplasts carry out photosynthesis",
                "Animal cells do not have chloroplasts",
            ],
        ],
        style=NEAT_HAND,
        qa=[
            QA("Which organelle builds proteins?", "ribosomes"),
            QA("What is the plant cell wall made of?", "cellulose"),
            QA("Where does photosynthesis take place?", "chloroplasts"),
        ],
    ),
    Doc(
        "physics_rough_notes",
        "handwritten",
        [
            [
                "Newton laws - rough notes",
                "First law an object keeps its velocity",
                "unless a net force acts on it",
                "Second law force equals mass times acceleration",
            ],
            [
                "Third law every action has an equal",
                "and opposite reaction",
                "Unit of force is the newton",
            ],
        ],
        style={"font": "BRUSHSCI.TTF", "size": 46, "spacing": 2.2, "wobble": 1.6, "noise": 1.2},  # messy
        qa=[
            QA("What does Newton's second law state?", "force equals mass times acceleration"),
            QA("What is the unit of force?", "the newton"),
        ],
    ),
]

UNANSWERABLE = [
    QA("Who is the principal of the college?", ""),
    QA("What is the capital of Australia?", ""),
]

_CALIBRATION_TEXT = [
    [
        "Sorting algorithms arrange elements in order",
        "Merge sort divides the list and merges halves",
        "Quick sort picks a pivot and partitions the list",
        "Heap sort builds a binary heap first",
        "Insertion sort is fast for nearly sorted data",
    ],
    [
        "A stack follows last in first out order",
        "A queue follows first in first out order",
        "Binary search needs a sorted array",
        "Hash tables give constant average lookup time",
        "Graphs model networks of connected nodes",
    ],
]

HAND_FONTS = ["Inkfree.ttf", "segoepr.ttf", "LHANDW.TTF", "BRADHITC.TTF", "segoesc.ttf", "BRUSHSCI.TTF"]


def calibration_docs() -> list[Doc]:
    """Graded-difficulty pages (no QA) for fitting calibration."""
    docs = []
    for i, noise in enumerate([0.0, 1.0, 2.0, 2.8, 3.5]):
        docs.append(Doc(f"cal_scanned_{i}", "scanned", _CALIBRATION_TEXT, style={**PRINTED, "noise": noise}))
    for i, font in enumerate(HAND_FONTS):
        wobble, noise = 0.5 + 0.25 * i, 0.3 * i
        style = {"font": font, "size": 44, "spacing": 2.1, "wobble": wobble, "noise": noise}
        docs.append(Doc(f"cal_handwritten_{i}", "handwritten", _CALIBRATION_TEXT, style=style))
    return docs


def main() -> None:
    rng = np.random.default_rng(42)  # fixed seed: the dataset is reproducible
    manifest = {"qa": [], "calibration": [], "unanswerable": [asdict(q) for q in UNANSWERABLE]}
    for subset, docs in (("qa", QA_DOCS), ("calibration", calibration_docs())):
        out_dir = DATASET_DIR / subset
        out_dir.mkdir(parents=True, exist_ok=True)
        for doc in docs:
            path = write_doc(doc, out_dir, rng)
            entry = asdict(doc)
            entry["file"] = str(path.relative_to(DATASET_DIR)).replace("\\", "/")
            entry["ground_truth"] = ["\n".join(lines) for lines in entry.pop("pages")]
            manifest[subset].append(entry)
    (DATASET_DIR / "dataset.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    n_qa = sum(len(d.qa) for d in QA_DOCS) + len(UNANSWERABLE)
    print(f"Wrote {len(manifest['qa'])} QA docs ({n_qa} questions) and {len(manifest['calibration'])} calibration docs")


if __name__ == "__main__":
    main()
