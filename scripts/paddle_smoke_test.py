"""Smoke test for the PaddleOCR install: imports, model construction on CPU, one OCR run.

Run with:  venv/Scripts/python scripts/paddle_smoke_test.py
Exit code 0 = text OCR and PP-StructureV3 both work on this machine.
"""

import sys
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Paddle 3.3 on Windows CPU crashes in its oneDNN kernels (ConvertPirAttribute2RuntimeAttribute
# "Unimplemented"); the plain CPU kernels work, so MKL-DNN stays off.
COMMON = {"device": "cpu", "enable_mkldnn": False}
LINES = ["DocBlendAI smoke test", "Confidence calibration 0.042"]


def make_image(path: Path) -> None:
    image = Image.new("RGB", (720, 180), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 34)
    except OSError:
        font = ImageFont.load_default(size=34)
    for i, line in enumerate(LINES):
        draw.text((30, 30 + 70 * i), line, font=font, fill="black")
    image.save(path)


def step(name, fn):
    started = time.perf_counter()
    try:
        value = fn()
    except Exception as exc:
        print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
        return None, False
    print(f"ok    {name} ({time.perf_counter() - started:.1f}s)")
    return value, True


def main() -> int:
    # torch must load before paddle on Windows: both ship Intel OpenMP/MKL DLLs, and once paddle's
    # are loaded torch fails with "WinError 127 ... shm.dll". The reverse order works.
    import torch  # noqa: F401
    import cv2
    import pymupdf
    import paddle
    import paddleocr

    print(f"python {sys.version.split()[0]} | paddle {paddle.__version__} | paddleocr {paddleocr.__version__} "
          f"| cv2 {cv2.__version__} | PyMuPDF {pymupdf.VersionBind}")
    from paddleocr import PaddleOCR, PPStructureV3

    ok = True
    ocr, good = step("construct PaddleOCR(lang='en')", lambda: PaddleOCR(
        lang="en", use_doc_orientation_classify=False, use_doc_unwarping=False,
        use_textline_orientation=False, **COMMON))
    ok &= good
    with tempfile.TemporaryDirectory() as tmp:
        image = Path(tmp) / "smoke.png"
        make_image(image)
        if ocr is not None:
            result, good = step("OCR one image", lambda: next(iter(ocr.predict(str(image)))))
            ok &= good
            if result is not None:
                for text, score in zip(result["rec_texts"], result["rec_scores"]):
                    print(f"        {score:.4f}  {text}")
                ok &= [t.strip() for t in result["rec_texts"]] == LINES
        structure, good = step("construct PPStructureV3()", lambda: PPStructureV3(
            lang="en", use_doc_orientation_classify=False, use_doc_unwarping=False, **COMMON))
        ok &= good
        if structure is not None:
            result, good = step("PP-StructureV3 parse", lambda: next(iter(structure.predict(str(image)))))
            ok &= good
            if result is not None:
                print("        markdown:", result.markdown["markdown_texts"].replace("\n", " | ")[:200])
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
