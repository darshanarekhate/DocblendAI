"""Experience Center tools (prefix /api): exports, preprocessing preview, plugins.

Supports Modules 2 and 7 for the Experience Center UI (/studio); see
docs/experience_center_contract.md §5-6. OCR/parse jobs and history live in
app/routers/paddleocr.py; this router only reads finished results.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from app.config import settings
from app.modules import exporters
from app.modules import preprocess as preprocess_steps
from app.modules.ocr_jobs import job_manager, page_image_path
from app.plugins import PLUGINS, PluginError, PluginModelError, PluginUnavailableError
from app.routers.paddleocr import _preprocess, _save_upload, _upload_kind

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["experience-center"])

# Preview images are for a side-by-side glance, not for OCR: keep the JSON small.
PREVIEW_MAX_SIDE = 1400


def _finished_result(run_id: str) -> dict[str, Any]:
    result = job_manager.get_result(run_id)
    if result is None or result.get("pipeline") == "calibration":
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No result with id {run_id}")
    if result.get("status") != "done":
        raise HTTPException(status.HTTP_409_CONFLICT, f"Result {run_id} is {result.get('status')}, not finished")
    return result


# -- export -------------------------------------------------------------------------------------


@router.get("/results/{run_id}/export/{fmt}")
def export_result(run_id: str, fmt: str) -> Response:
    """Download a finished result (with review edits) as txt, json, md, pdf, docx or csv."""
    fmt = fmt.lower()
    if fmt not in exporters.FORMATS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Unknown format {fmt!r}; use one of {', '.join(exporters.FORMATS)}")
    result = _finished_result(run_id)
    page_images = [page_image_path(run_id, i) for i in range(len(result.get("pages") or []))]
    try:
        content, media_type, ext = exporters.export(
            result, fmt, [p if p.exists() else None for p in page_images], settings.paddle_pdf_dpi
        )
    except exporters.ExportError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    stem = Path(result.get("filename") or "result").stem or "result"
    name = f"{stem}.{ext}"
    ascii_name = name.encode("ascii", "replace").decode().replace("?", "_").replace('"', "_")
    headers = {"Content-Disposition": f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(name)}"}
    return Response(content, media_type=media_type, headers=headers)


# -- preprocessing preview ----------------------------------------------------------------------


def _first_page_bgr(path: Path, kind: str):
    """First page (image frame or PDF page) as an OpenCV BGR array."""
    import cv2
    import numpy as np
    from PIL import Image, ImageOps

    if kind == "pdf":
        import pymupdf

        with pymupdf.open(path, filetype="pdf") as doc:
            if doc.page_count == 0:
                raise ValueError("the PDF has no pages")
            pix = doc[0].get_pixmap(dpi=settings.paddle_pdf_dpi, alpha=False)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    else:
        with Image.open(path) as img:
            image = ImageOps.exif_transpose(img.copy()).convert("RGB")
    scale = PREVIEW_MAX_SIDE / max(image.size)
    if scale < 1:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _data_url(image) -> str:
    import cv2

    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("could not encode the preview image")
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _preview(path: Path, kind: str, options: dict[str, bool]) -> dict[str, Any]:
    before = _first_page_bgr(path, kind)
    after = preprocess_steps.apply(before, options)
    return {
        "before": _data_url(before),
        "after": _data_url(after),
        "steps": [s for s in preprocess_steps.STEPS if options.get(s)],
        "skew_angle": round(preprocess_steps.skew_angle(before), 2) if options.get("deskew") else None,
    }


@router.post("/preprocess")
async def preview_preprocess(file: UploadFile = File(...), preprocess: str | None = Form(None)) -> dict[str, Any]:
    """Before/after preview of the clean-up steps on the first page of an image or PDF."""
    suffix, kind = _upload_kind(file.filename)
    if kind not in ("image", "pdf"):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Preview needs an image or a PDF")
    options = _preprocess(preprocess)
    if not preprocess_steps.any_enabled(options):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Choose at least one clean-up step to preview")
    path = await _save_upload(file, suffix)
    try:
        return await run_in_threadpool(_preview, path, kind, options)
    except HTTPException:
        raise
    except Exception as exc:  # damaged image/PDF
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"{file.filename}: could not read it ({exc})") from exc
    finally:
        path.unlink(missing_ok=True)


# -- plugins ------------------------------------------------------------------------------------


@router.get("/plugins")
def list_plugins() -> list[dict[str, Any]]:
    return [plugin.info() for plugin in PLUGINS.values()]


@router.post("/results/{run_id}/plugins/{name}")
async def run_plugin(run_id: str, name: str, options: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    plugin = PLUGINS.get(name)
    if plugin is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown plugin {name!r}; available: {', '.join(PLUGINS)}")
    if not plugin.enabled():
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, plugin.info()["message"])
    result = _finished_result(run_id)
    text = exporters.result_markdown(result)
    if name == "refine" and "lines" not in options:  # refine the result's own lines (ids match)
        from app.modules.refinement import refine_input

        lines, context = refine_input(result)
        options = {**options, "lines": lines, "context": context}
    try:
        output = await run_in_threadpool(plugin.run, text, options)
    except PluginUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except PluginModelError as exc:
        logger.warning("plugin %s failed on %s: %s", name, run_id, exc)
        raise HTTPException(exc.status, str(exc)) from exc
    except PluginError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return {"plugin": name, "result_id": run_id, **output}
