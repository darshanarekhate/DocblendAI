"""Supports Module 3 — reliability diagram (before vs after calibration) drawn with PIL.

Each panel plots, per confidence bin, the share of lines read correctly (bar)
against the diagonal of perfect calibration. The shaded strip between a bar's
top and the bin's mean confidence (tick) is the calibration gap: above the bar
= over-confident, below = under-confident. ECE is in each panel's title.
"""

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ACCENT = "#2f5d50"
MODERATE = "#9a5b00"
ERROR = "#b3261e"
TEXT = "#1f1e1b"
MUTED = "#6b6862"
GRID = "#e6e3dc"
BACKGROUND = "#ffffff"

WIDTH, HEIGHT = 900, 450


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _rgba(hex_colour: str, alpha: int) -> tuple[int, int, int, int]:
    h = hex_colour.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha


def _text_centered(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font, fill: str) -> None:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - (right - left) / 2, xy[1] - (bottom - top) / 2 - top), text, font=font, fill=fill)


def _hatch(image: Image.Image, box: tuple[float, float, float, float]) -> None:
    """Diagonal light-red stripes inside box (drawn straight onto image)."""
    left, top, right, bottom = (int(round(v)) for v in box)
    w, h = max(right - left, 1), max(bottom - top, 1)
    tile = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)
    for offset in range(-h, w, 7):
        draw.line([(offset, h), (offset + h, 0)], fill="#ecbdb9", width=2)
    image.alpha_composite(tile, (left, top))


def _panel(
    image: Image.Image, box: tuple[int, int, int, int], bins: list[dict[str, Any]], title: str, fonts: dict
) -> None:
    """Draw one reliability panel; box = (left, top, right, bottom) of the plot area."""
    x0, y0, x1, y1 = box
    draw = ImageDraw.Draw(image)

    def px(v: float) -> float:
        return x0 + v * (x1 - x0)

    def py(v: float) -> float:
        return y1 - v * (y1 - y0)

    _text_centered(draw, ((x0 + x1) / 2, y0 - 24), title, fonts["title"], TEXT)
    for t in range(6):
        v = t / 5
        draw.line([(px(v), y0), (px(v), y1)], fill=GRID)
        draw.line([(x0, py(v)), (x1, py(v))], fill=GRID)
        _text_centered(draw, (px(v), y1 + 12), f"{v:.1f}", fonts["tick"], MUTED)
        label = f"{v:.1f}"
        w = draw.textlength(label, font=fonts["tick"])
        _text_centered(draw, (x0 - 8 - w / 2, py(v)), label, fonts["tick"], MUTED)

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    shade = ImageDraw.Draw(overlay)
    for b in bins:
        if not b.get("count") or b.get("accuracy") is None:
            continue
        left, right = px(b["lo"]) + 2, px(b["hi"]) - 2
        draw.rectangle([left, py(b["accuracy"]), right, y1], fill=ACCENT)
        conf = b["mean_confidence"]
        top, bottom = sorted((py(b["accuracy"]), py(conf)))
        if bottom - top >= 1:
            if conf > b["accuracy"]:  # over-confident: gap above the bar, plain tint
                shade.rectangle([left, top, right, bottom], fill=_rgba(ERROR, 90), outline=_rgba(ERROR, 220))
            else:  # under-confident: gap inside the bar, hatched so it stays readable on the dark bar
                _hatch(image, (left, top, right, bottom))
                shade.rectangle([left, top, right, bottom], outline=_rgba(ERROR, 255))
        _text_centered(draw, ((left + right) / 2, py(max(b["accuracy"], conf)) - 9), str(b["count"]), fonts["small"], MUTED)
    image.alpha_composite(overlay)

    draw = ImageDraw.Draw(image)
    draw.line([(px(0), py(0)), (px(1), py(1))], fill=TEXT, width=1)
    for b in bins:
        if b.get("count") and b.get("mean_confidence") is not None:
            y = py(b["mean_confidence"])
            draw.line([(px(b["lo"]) + 2, y), (px(b["hi"]) - 2, y)], fill=MODERATE, width=2)
    draw.rectangle([x0, y0, x1, y1], outline=TEXT)

    _text_centered(draw, ((x0 + x1) / 2, y1 + 32), "Confidence", fonts["label"], TEXT)
    label = Image.new("RGBA", (120, 20), (0, 0, 0, 0))
    _text_centered(ImageDraw.Draw(label), (60, 10), "Accuracy", fonts["label"], TEXT)
    rotated = label.rotate(90, expand=True)
    image.alpha_composite(rotated, (int(x0 - 58), int((y0 + y1) / 2 - rotated.height / 2)))


def draw_reliability_diagram(
    bins_before: list[dict[str, Any]],
    bins_after: list[dict[str, Any]],
    ece_before: float,
    ece_after: float,
    method: str,
    path: Path,
) -> Path:
    """Write the two-panel PNG (raw vs calibrated confidences) to path and return it."""
    fonts = {"title": _font(16), "label": _font(13), "tick": _font(11), "small": _font(10)}
    image = Image.new("RGBA", (WIDTH, HEIGHT), BACKGROUND)
    top, bottom = 52, HEIGHT - 92
    size = bottom - top
    _panel(image, (90, top, 90 + size, bottom), bins_before, f"Before: raw confidence (ECE {ece_before:.3f})", fonts)
    _panel(image, (530, top, 530 + size, bottom), bins_after, f"After: {method} (ECE {ece_after:.3f})", fonts)

    draw = ImageDraw.Draw(image)
    legend = [
        ("bar", ACCENT, "Accuracy per bin"),
        ("tick", MODERATE, "Mean confidence"),
        ("shade", ERROR, "Gap"),
        ("line", TEXT, "Perfect calibration"),
    ]
    x, y = 150, HEIGHT - 26
    for kind, colour, text in legend:
        if kind == "bar":
            draw.rectangle([x, y - 6, x + 16, y + 6], fill=colour)
        elif kind == "tick":
            draw.line([(x, y), (x + 16, y)], fill=colour, width=3)
        elif kind == "shade":
            draw.rectangle([x, y - 6, x + 16, y + 6], fill="#ecbdb9", outline=colour)
        else:
            draw.line([(x, y + 6), (x + 16, y - 6)], fill=colour, width=1)
        draw.text((x + 22, y - 7), text, font=fonts["label"], fill=TEXT)
        x += 30 + int(draw.textlength(text, font=fonts["label"])) + 28

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="PNG", optimize=True)
    return path
