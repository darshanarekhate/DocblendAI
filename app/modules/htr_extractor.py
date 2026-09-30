"""Module 2 — HTR path (handwritten PDFs and images).

Responsibility: split each page into text lines and run TrOCR (transformers)
on them, returning page text plus a confidence (0-1) as raw_conf for Module 3.

TrOCR reads one line at a time, so pages are segmented into lines first
(find_lines): notebook rules are erased, then docTR's text detector finds the
lines (see line_segmentation.py), with a horizontal ink-projection profile
(segment_lines) as the fallback. A line's confidence is the geometric mean of
its generated tokens' probabilities, temperature-scaled
(settings.htr_temperature); the page's raw_conf is the mean of its lines'
confidences weighted by line length.

Uses from schemas.py: nothing directly; output feeds chunker.py -> RecognizedChunk.
"""

import logging
import math
from contextlib import closing
from functools import lru_cache

import numpy as np
from PIL import Image

from app.config import settings
from app.modules import line_segmentation, ocr_extractor
from app.modules.pdf_render import denoise, render_pages

logger = logging.getLogger(__name__)

HTR_DPI = 200  # TrOCR resizes each line to 384x384 anyway; higher DPI only slows segmentation
BATCH_SIZE = 8
MAX_LINE_TOKENS = 64

# Line segmentation, as fractions of page height / width.
MIN_INK_ROW_FRACTION = 0.005  # a row counts as text if this share of its pixels is ink
MIN_LINE_HEIGHT = 0.008
MERGE_GAP = 0.004  # bands closer than this are one line (ascenders/descenders, i-dots)
PAD = 0.006


class HTRUnavailableError(RuntimeError):
    """The TrOCR model could not be loaded (e.g. not downloaded and no internet)."""


def _otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    total = gray.size
    cum_count = np.cumsum(hist)
    cum_sum = np.cumsum(hist * np.arange(256))
    best_t, best_var = 0, -1.0
    for t in range(255):
        w0 = cum_count[t]
        w1 = total - w0
        if w0 == 0 or w1 == 0:
            continue
        m0 = cum_sum[t] / w0
        m1 = (cum_sum[-1] - cum_sum[t]) / w1
        var = w0 * w1 * (m0 - m1) ** 2
        if var > best_var:
            best_t, best_var = t, var
    return best_t


def segment_lines(page: Image.Image) -> list[Image.Image]:
    """Crop a page image into text-line images, top to bottom."""
    gray = np.asarray(page.convert("L"))
    h, w = gray.shape
    ink = gray <= _otsu_threshold(gray)
    if ink.mean() > 0.5:  # mostly "ink" means a blank or inverted page
        return []

    text_rows = ink.sum(axis=1) >= max(1, MIN_INK_ROW_FRACTION * w)
    bands: list[list[int]] = []
    for y in np.flatnonzero(text_rows):
        if bands and y - bands[-1][1] <= max(1, MERGE_GAP * h):
            bands[-1][1] = y
        else:
            bands.append([y, y])

    pad_y, pad_x = int(PAD * h), int(PAD * w)
    lines = []
    for top, bottom in bands:
        if bottom - top + 1 < MIN_LINE_HEIGHT * h:
            continue
        cols = np.flatnonzero(ink[top : bottom + 1].any(axis=0))
        left, right = cols[0], cols[-1]
        box = (max(0, left - pad_x), max(0, top - pad_y), min(w, right + pad_x + 1), min(h, bottom + pad_y + 1))
        lines.append(page.crop(box))
    return lines


def find_lines(page: Image.Image) -> list[Image.Image]:
    """Crop a page into text-line images, top to bottom, with the configured segmenter."""
    if settings.remove_ruled_lines:
        page = line_segmentation.remove_ruled_lines(page)
    if settings.htr_segmenter == "doctr":
        try:
            return line_segmentation.detect_lines(page)
        except line_segmentation.LineDetectorUnavailableError as e:
            logger.warning("%s; falling back to projection line segmentation", e)
    return segment_lines(page)


def _matching_tokenizer(model_name: str, vocab_size: int, classes):
    for cls in classes:
        try:
            tokenizer = cls.from_pretrained(model_name)
        except (OSError, ValueError, TypeError):
            continue
        if len(tokenizer) >= vocab_size * 0.9:  # a mismatched class loads only a handful of tokens
            return tokenizer
    raise ValueError(f"no tokenizer class matches {model_name!r} (decoder vocabulary {vocab_size})")


@lru_cache
def _load(model_name: str):
    try:
        from transformers import (
            AutoImageProcessor,
            AutoTokenizer,
            RobertaTokenizer,
            TrOCRProcessor,
            VisionEncoderDecoderModel,
            XLMRobertaTokenizer,
        )

        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except ValueError:
            # transformers 5 cannot auto-build the TrOCR checkpoints' legacy tokenizer configs,
            # but the named classes load them: SentencePiece (XLM-R) for trocr-small, BPE
            # (RoBERTa) for trocr-base/large. Pick the one whose vocabulary fits the decoder;
            # the wrong one loads without error but decodes every line to "".
            tokenizer = _matching_tokenizer(model_name, model.config.decoder.vocab_size,
                                            (XLMRobertaTokenizer, RobertaTokenizer))
        processor = TrOCRProcessor(image_processor=AutoImageProcessor.from_pretrained(model_name), tokenizer=tokenizer)
    # OSError: not downloaded / no internet. ValueError, ImportError: a tokenizer dependency
    # (e.g. sentencepiece for trocr-small) is missing.
    except (OSError, ValueError, ImportError) as e:
        raise HTRUnavailableError(f"Could not load HTR model {model_name!r}: {e}") from e
    model.eval()
    return processor, model


def recognize_lines(lines: list[Image.Image]) -> list[tuple[str, float]]:
    """Run TrOCR on line images. Returns (text, confidence) per line."""
    import torch  # imported lazily: torch is slow to import and only this path needs it

    processor, model = _load(settings.htr_model)
    pad_id = model.generation_config.pad_token_id
    results: list[tuple[str, float]] = []
    for i in range(0, len(lines), BATCH_SIZE):
        batch = [img.convert("RGB") for img in lines[i : i + BATCH_SIZE]]
        pixel_values = processor(images=batch, return_tensors="pt").pixel_values
        with torch.no_grad():
            out = model.generate(
                pixel_values,
                max_new_tokens=MAX_LINE_TOKENS,
                num_beams=1,
                output_scores=True,
                return_dict_in_generate=True,
            )
        # Temperature scaling: divide the logits by T before the softmax. T > 1 softens the
        # overconfident raw probabilities; the chosen tokens (the text) do not change.
        scores = tuple(s / settings.htr_temperature for s in out.scores)
        logprobs = model.compute_transition_scores(out.sequences, scores, normalize_logits=True)
        generated = out.sequences[:, 1:]  # drop the decoder start token
        texts = processor.batch_decode(out.sequences, skip_special_tokens=True)
        for text, lp, tokens in zip(texts, logprobs, generated):
            # Ignore padding after EOS in shorter sequences.
            real = tokens != pad_id if pad_id is not None else torch.ones_like(tokens, dtype=torch.bool)
            conf = math.exp(lp[real].mean().item()) if real.any() else 0.0
            results.append((text.strip(), conf))
    return results


def htr_image(page: Image.Image) -> tuple[str, float]:
    """HTR one page image. Returns (text, raw_conf); a page with no lines gives ("", 0.0)."""
    recognized = [(t, c) for t, c in recognize_lines(find_lines(denoise(page))) if t]
    chars = sum(len(t) for t, _ in recognized)
    if not chars:
        return "", 0.0
    return "\n".join(t for t, _ in recognized), sum(c * len(t) for t, c in recognized) / chars


def teacher_forced_logits(lines: list[Image.Image], texts: list[str]) -> list[tuple[np.ndarray, np.ndarray]]:
    """TrOCR's logits for the TRUE text of each line image: (logits [tokens, vocab], target ids [tokens]).

    Used only to fit the temperature (confidence_capture.fit_temperature): the
    temperature that best predicts the true tokens is the one that makes
    TrOCR's probabilities honest.
    """
    import torch

    processor, model = _load(settings.htr_model)
    out = []
    for img, text in zip(lines, texts):
        pixel_values = processor(images=[img.convert("RGB")], return_tensors="pt").pixel_values
        labels = processor.tokenizer(text, return_tensors="pt").input_ids
        with torch.no_grad():
            logits = model(pixel_values=pixel_values, labels=labels).logits[0]
        out.append((logits.numpy(), labels[0].numpy()))
    return out


def htr_file(file_path: str, max_pages: int | None = None, first_page: int = 0) -> list[tuple[str, float]]:
    """Return (page_text, raw_conf) for each page of a PDF or image (or the first max_pages),
    starting at page index first_page."""
    # closing(): release the PDF immediately if HTR fails (see ocr_extractor.ocr_file).
    with closing(render_pages(file_path, HTR_DPI, max_pages, first_page)) as pages:
        # Orientation (photos only) uses Tesseract OSD; it degrades to a no-op if Tesseract is missing.
        orient = ocr_extractor.orient_for(file_path)
        return [htr_image(orient(img)) for img in pages]
