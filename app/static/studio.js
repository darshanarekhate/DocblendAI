/* DocBlendAI Experience Center (served at /studio).
 *
 * Upload an image, PDF or Office file, pick a pipeline (Text OCR / PP-StructureV3 / PaddleOCR-VL),
 * and see the page with a confidence-coloured box around each line next to the parsed result.
 * Every upload is refined by Gemini automatically and is also a document on the question page:
 * the refined text is shown with corrected / inferred words marked (hover: the original), with
 * Show original, Revert to original, Retry, a Layout view and the document's versions.
 * Talks only to the /api endpoints in docs/experience_center_contract.md (§3, result object §4,
 * calibration report §1). Plain JavaScript, no build step, no external libraries.
 *
 * Safety: nothing from the server or a document is inserted as HTML except (a) Markdown that this
 * file renders itself after escaping, and (b) table HTML after it has been rebuilt from an
 * allow-list (table/thead/tbody/tfoot/tr/th/td with numeric colspan/rowspan only).
 */
"use strict";

// ---------------------------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------------------------

const SAMPLES = [
  { file: "lecture_notes.png", label: "Lecture notes" },
  { file: "report_table.png", label: "Report with a table" },
  { file: "noisy_scan.png", label: "Noisy, tilted scan" },
];
const ACCEPTED = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".pdf", ".docx", ".xlsx", ".pptx", ".txt"];
const OFFICE = [".docx", ".xlsx", ".pptx", ".txt"];
const PIPELINE_LABEL = { ocr: "Text OCR", structure: "Document parsing", vl: "PaddleOCR-VL", office: "Office conversion" };
// Used only when /api/health does not list languages (PaddleOCR language codes).
const FALLBACK_LANGUAGES = [
  { code: "en", name: "English" }, { code: "ch", name: "Chinese and English" }, { code: "fr", name: "French" },
  { code: "german", name: "German" }, { code: "es", name: "Spanish" }, { code: "pt", name: "Portuguese" },
  { code: "it", name: "Italian" }, { code: "hi", name: "Hindi" },
];
const POLL_MS = 1000;           // job polling interval
const MAX_POLL_FAILURES = 5;    // consecutive network failures before a job is given up
const HIGH_BAND = 0.15;         // green ("confident") starts this far above the review threshold
const THEME_KEY = "docblend-theme";

// ---------------------------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------------------------

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const pct = (v) => (v == null || Number.isNaN(v) ? "–" : `${Math.round(v * 100)}%`);
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const extOf = (name) => { const m = /\.[^.]+$/.exec(name || ""); return m ? m[0].toLowerCase() : ""; };
const baseName = (name) => (name || "result").replace(/\.[^.]+$/, "") || "result";

/** Create an element: el("button", {class: "x", type: "button"}, "text", childNode). */
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children) if (c != null) node.append(c);
  return node;
}
const SVG_NS = "http://www.w3.org/2000/svg";
function svg(tag, attrs = {}) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
}

/** fetch + JSON; throws Error(detail) on a non-2xx answer, with .status set. */
async function api(path, options, withStatus = false) {
  let resp;
  try {
    resp = await fetch(path, options);
  } catch {
    const err = new Error("Could not reach the server. Is it running?");
    err.status = 0;
    throw err;
  }
  const body = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    const d = body.detail;
    const msg = typeof d === "string" ? d
      : Array.isArray(d) ? d.map((x) => x.msg || JSON.stringify(x)).join("; ")  // FastAPI 422 list
      : `Request failed (${resp.status})`;
    const err = new Error(msg);
    err.status = resp.status;
    throw err;
  }
  return withStatus ? { body, status: resp.status } : body;
}

function setStatus(id, text, kind = "") {
  const s = $(id);
  s.className = `status ${kind}`.trim();
  s.textContent = text;
}

function downloadText(text, filename, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = el("a", { href: url, download: filename });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function copyText(text, button) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    // Older browsers / insecure origins: fall back to a temporary textarea.
    const ta = el("textarea", { class: "visually-hidden" });
    ta.value = text;
    document.body.append(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
  }
  const old = button.textContent;
  button.textContent = "Copied";
  setTimeout(() => { button.textContent = old; }, 1500);
}

// ---------------------------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------------------------

const state = {
  health: null,
  jobs: [],              // {key, file, name, pipeline, language, status, progress, message, id, result, error}
  running: false,
  current: null,         // result object (§4) on display
  currentKey: null,      // which job it came from
  page: 0,
  mode: "calibrated",    // which confidence drives colours, badges and filters
  minConf: 0,            // hide lines below this
  threshold: 0.7,        // review threshold (client-side)
  thresholdTouched: false,
  view: { z: 1, x: 0, y: 0, fitted: true },
  selected: null,        // line id
  hovered: null,
  mdRaw: false,
  calibration: null,
  calibrationLoaded: false,
  calibrating: false,
};
let jobSeq = 0;

// ---------------------------------------------------------------------------------------------
// Theme: system / light / dark, remembered per browser
// ---------------------------------------------------------------------------------------------

function applyTheme(choice) {
  if (choice === "light" || choice === "dark") document.documentElement.dataset.theme = choice;
  else delete document.documentElement.dataset.theme;
  try {
    if (choice === "light" || choice === "dark") localStorage.setItem(THEME_KEY, choice);
    else localStorage.removeItem(THEME_KEY);
  } catch { /* storage blocked: the choice just lasts for this visit */ }
}
(function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem(THEME_KEY); } catch { /* ignore */ }
  $("theme").value = saved === "light" || saved === "dark" ? saved : "system";
  $("theme").addEventListener("change", (e) => applyTheme(e.target.value));
})();

// ---------------------------------------------------------------------------------------------
// Engine status (GET /api/health): pill, pipeline availability, languages
// ---------------------------------------------------------------------------------------------

async function loadHealth() {
  const box = $("engine"), label = $("engine-label"), panel = $("engine-panel");
  try {
    const h = await api("/api/health");
    state.health = h;
    const pipes = h.pipelines || {};
    const ocrOk = pipes.ocr?.available !== false;
    // VL is optional (off by default), so only a missing OCR/parsing/Office pipeline turns the pill amber.
    const anyDown = Object.entries(pipes).some(([name, p]) => name !== "vl" && p && p.available === false);
    const cal = h.calibration || {};
    box.className = `engine ${!ocrOk || h.status !== "ok" ? "down" : anyDown ? "warn" : "ok"}`;
    const calText = cal.status === "calibrated" ? `calibrated${cal.method ? ` (${cal.method})` : ""}` : "uncalibrated";
    label.textContent = `${ocrOk && h.status === "ok" ? "Engine ready" : "Engine unavailable"} · ${calText}`;

    const eng = typeof h.engine === "object" && h.engine ? h.engine : { paddleocr: h.engine };
    const rows = [
      ["PaddleOCR", eng.paddleocr], ["PaddlePaddle", eng.paddlepaddle], ["Model", eng.ocr_version],
      ["Device", eng.device || h.device], ["Calibration", calText],
      ["Review below", h.review_threshold != null ? pct(h.review_threshold) : null],
      ["Max upload", h.max_upload_mb ? `${h.max_upload_mb} MB` : null],
      ["LLM refinement", h.llm ? (h.llm.enabled ? `${h.llm.model}, every upload` : h.llm.message || "off") : null],
    ].filter(([, v]) => v != null && v !== "");
    const dl = el("dl");
    for (const [k, v] of rows) dl.append(el("dt", { text: k }), el("dd", { text: String(v) }));
    const list = el("ul");
    for (const [name, p] of Object.entries(pipes)) {
      const ok = p?.available !== false;
      list.append(el("li", {}, `${PIPELINE_LABEL[name] || name}: `,
        el("span", { class: ok ? "ok-text" : "bad-text", text: ok ? "available" : "unavailable" }),
        !ok && p?.message ? el("span", { class: "hint", text: ` (${p.message})` }) : null));
    }
    panel.replaceChildren(dl, Object.keys(pipes).length ? el("h4", { text: "Pipelines" }) : null, list);

    applyPipelineAvailability(pipes);
    fillLanguages(h.languages);
    renderRefineBar();
    if (!state.thresholdTouched && !state.current && typeof h.review_threshold === "number") setThreshold(h.review_threshold, false);
  } catch (err) {
    box.className = "engine down";
    label.textContent = "Server unreachable";
    panel.replaceChildren(el("p", { class: "hint", text: err.message }));
    fillLanguages(null);
  }
}

/** Disable pipelines the server reports as unavailable, with its message as a note and tooltip. */
function applyPipelineAvailability(pipes) {
  for (const input of document.querySelectorAll('input[name="pipeline"]')) {
    const info = pipes[input.value];
    const down = info && info.available === false;
    input.disabled = !!down;
    const choice = input.closest(".choice");
    choice.title = down ? info.message || "This pipeline is not available on this server." : "";
    if (input.value === "vl") {
      const note = $("vl-note");
      note.hidden = !down;
      note.textContent = down ? `Unavailable${info.message ? `: ${info.message}` : ""}` : "";
    }
    if (down && input.checked) document.querySelector('input[name="pipeline"][value="ocr"]').checked = true;
  }
}

function fillLanguages(languages) {
  const select = $("language");
  const prev = select.value || "en";
  const list = Array.isArray(languages) && languages.length ? languages : FALLBACK_LANGUAGES;
  select.replaceChildren(...list.map((l) => el("option", { value: l.code, text: l.name || l.code })));
  select.value = list.some((l) => l.code === prev) ? prev : list.some((l) => l.code === "en") ? "en" : list[0].code;
}

// ---------------------------------------------------------------------------------------------
// File queue: add (drop / pick / sample), run one job at a time, poll each to completion
// ---------------------------------------------------------------------------------------------

function addFiles(files) {
  const maxMb = state.health?.max_upload_mb || 25;
  let added = 0;
  for (const file of files) {
    const job = { key: ++jobSeq, file, name: file.name, status: "ready", progress: 0, message: "Ready", id: null, result: null, error: null };
    if (!ACCEPTED.includes(extOf(file.name))) {
      job.status = "error";
      job.error = `Unsupported file type. Use ${ACCEPTED.join(", ")}.`;
    } else if (file.size > maxMb * 1024 * 1024) {
      job.status = "error";
      job.error = `File is larger than the ${maxMb} MB limit.`;
    } else {
      added++;
    }
    state.jobs.push(job);
  }
  renderQueue();
  if (added) setStatus("job-status", `${added} file${added === 1 ? "" : "s"} ready. Press Run.`);
}

function renderQueue() {
  const ul = $("queue");
  ul.replaceChildren();
  for (const job of state.jobs) {
    const done = job.status === "done";
    const busy = ["uploading", "queued", "running"].includes(job.status);
    const nameNode = done
      ? el("button", { type: "button", class: "job-name", title: `Show ${job.name}`, text: job.name })
      : el("span", { class: "job-name", title: job.name, text: job.name });
    if (done) nameNode.addEventListener("click", () => showJob(job));
    const remove = busy ? null : el("button", { type: "button", class: "job-x", "aria-label": `Remove ${job.name} from the list`, title: "Remove" }, "×");
    remove?.addEventListener("click", () => { state.jobs = state.jobs.filter((j) => j !== job); renderQueue(); });
    const meta = job.status === "error" ? job.error
      : done ? `${PIPELINE_LABEL[job.result?.pipeline] || ""} · ${job.result?.summary?.line_count ?? countLines(job.result)} lines`
      : job.status === "ready" ? "Waiting for Run"
      : job.message || job.status;
    const li = el("li", { class: `job ${job.status}${job.key === state.currentKey ? " current" : ""}` },
      el("div", { class: "job-top" }, nameNode, remove),
      el("span", { class: "job-meta", text: meta }));
    if (busy) {
      // No value = indeterminate bar while uploading or before the server reports progress.
      const bar = el("progress", { max: "1", "aria-label": `Progress for ${job.name}` });
      if (job.status !== "uploading" && job.progress > 0) bar.value = job.progress;
      li.append(bar);
    }
    ul.append(li);
  }
  const ready = state.jobs.some((j) => j.status === "ready");
  $("run").disabled = state.running || !ready;
  $("run").textContent = state.running ? "Running…" : "Run";
  $("clear-done").hidden = !state.jobs.some((j) => j.status === "done" || j.status === "error");
}

const countLines = (r) => (r?.pages || []).reduce((n, p) => n + (p.lines?.length || 0), 0);

async function runQueue() {
  if (state.running) return;
  const pipeline = document.querySelector('input[name="pipeline"]:checked').value;
  const language = $("language").value || "en";
  const batch = state.jobs.filter((j) => j.status === "ready");
  if (!batch.length) return;
  state.running = true;
  for (const job of batch) { job.pipeline = pipeline; job.language = language; }
  renderQueue();
  for (const job of batch) {
    if (!state.jobs.includes(job)) continue;  // removed meanwhile
    await runJob(job);
  }
  state.running = false;
  renderQueue();
  const failed = batch.filter((j) => j.status === "error").length;
  setStatus("job-status", failed ? `Finished with ${failed} error${failed === 1 ? "" : "s"}. See the list below.` : "All done.", failed ? "error" : "ok");
}

async function runJob(job) {
  const office = OFFICE.includes(extOf(job.name));
  const form = new FormData();
  form.append("file", job.file, job.name);
  form.append("language", job.language);
  form.append("wait", "false");
  form.append("review_threshold", String(state.threshold));
  if (anyPreprocess() && !office) form.append("preprocess", JSON.stringify(preprocessOptions()));
  if ($("format-hint").value && !office) form.append("format_hint", $("format-hint").value);
  // Office files run the "office" pipeline on either endpoint (contract §3).
  const url = job.pipeline === "ocr" ? "/api/ocr" : "/api/parse";
  if (url === "/api/parse") form.append("pipeline", job.pipeline);
  job.status = "uploading";
  job.message = "Uploading…";
  renderQueue();
  setStatus("job-status", `Uploading ${job.name}…`);
  try {
    const { body } = await api(url, { method: "POST", body: form }, true);
    if (body.status === "done" && body.pages) return finishJob(job, body);  // server answered synchronously
    job.id = body.id;
    job.status = body.status || "queued";
    job.message = office ? "Converting…" : "Queued";
    renderQueue();
    let failures = 0, lastMsg = "";
    for (;;) {
      await sleep(POLL_MS);
      let r;
      try {
        r = await api(`/api/results/${encodeURIComponent(job.id)}`);
        failures = 0;
      } catch (err) {
        if (err.status === 404 || ++failures >= MAX_POLL_FAILURES) throw err;
        continue;
      }
      job.status = r.status || job.status;
      job.progress = typeof r.progress === "number" ? r.progress : job.progress;
      job.message = r.message || (job.status === "queued" ? "Waiting for the engine…" : "Reading…");
      if (job.status === "done") return finishJob(job, r);
      if (job.status === "error") throw new Error(r.error || r.message || "The job failed.");
      renderQueue();
      const msg = `${job.name}: ${job.message}${job.progress ? ` (${pct(job.progress)})` : ""}`;
      if (msg !== lastMsg) { setStatus("job-status", msg); lastMsg = msg; }  // announce changes only
    }
  } catch (err) {
    job.status = "error";
    job.error = err.message;
    renderQueue();
    setStatus("job-status", `${job.name}: ${err.message}`, "error");
  }
}

function finishJob(job, result) {
  job.status = "done";
  job.result = result;
  job.progress = 1;
  setStatus("job-status", `${job.name}: done in ${formatMs(result.processing_time_ms)}.`, "ok");
  loadHistory();
  // Show the newest finished result, unless the user is in the middle of correcting a line.
  if (!document.querySelector(".ln-edit")) showJob(job);
  else renderQueue();
}

function showJob(job) {
  state.currentKey = job.key;
  setResult(job.result);
  renderQueue();
}

function formatMs(ms) {
  if (ms == null) return "–";
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(ms < 10000 ? 1 : 0)} s`;
}

// Upload widgets: click to pick, or drop files anywhere on the page.
const fileInput = $("file"), drop = $("drop");
fileInput.addEventListener("change", () => { addFiles([...fileInput.files]); fileInput.value = ""; });
const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
window.addEventListener("dragover", (e) => { if (hasFiles(e)) { e.preventDefault(); drop.classList.add("dragging"); } });
window.addEventListener("dragleave", (e) => { if (!e.relatedTarget) drop.classList.remove("dragging"); });
window.addEventListener("drop", (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  drop.classList.remove("dragging");
  if (e.dataTransfer.files.length) addFiles([...e.dataTransfer.files]);
});
$("run").addEventListener("click", runQueue);
$("clear-done").addEventListener("click", () => {
  state.jobs = state.jobs.filter((j) => !["done", "error"].includes(j.status));
  renderQueue();
});

// Samples are ordinary files served from /static/samples/; fetch them and queue them like an upload.
for (const s of SAMPLES) {
  const b = el("button", { type: "button", class: "sample", text: s.label, title: `Add ${s.file}` });
  b.addEventListener("click", async () => {
    try {
      const resp = await fetch(`/static/samples/${s.file}`);
      if (!resp.ok) throw new Error(`Sample not found (${resp.status})`);
      const blob = await resp.blob();
      addFiles([new File([blob], s.file, { type: blob.type || "image/png" })]);
    } catch (err) {
      setStatus("job-status", err.message, "error");
    }
  });
  $("samples").append(b);
}

// ---------------------------------------------------------------------------------------------
// Confidence: which score to use, tiers, review flag
// ---------------------------------------------------------------------------------------------

const isUncalibrated = () => (state.current ? state.current.calibration?.status !== "calibrated" : false);
const effectiveMode = () => (isUncalibrated() ? "raw" : state.mode);
function confOf(line) {
  const raw = line.raw_confidence, cal = line.calibrated_confidence;
  const v = effectiveMode() === "raw" ? raw ?? cal : cal ?? raw;
  return typeof v === "number" ? v : null;
}
/** certain (green) / moderate (yellow) / unreadable (red), relative to the review threshold. */
function tierOf(c) {
  if (c == null) return "moderate";
  if (c < state.threshold) return "unreadable";
  return c >= Math.min(0.99, state.threshold + HIGH_BAND) ? "certain" : "moderate";
}
const needsReview = (c) => c != null && c < state.threshold;
const lineId = (line, pi, li) => line.id || `p${pi}-l${li}`;
const visible = (line) => { const c = confOf(line); return c == null || c >= state.minConf; };

// ---------------------------------------------------------------------------------------------
// Showing a result
// ---------------------------------------------------------------------------------------------

function setResult(result) {
  state.current = result;
  state.page = 0;
  state.selected = null;
  state.hovered = null;
  if (!state.thresholdTouched && typeof result?.calibration?.review_threshold === "number") {
    setThreshold(result.calibration.review_threshold, false);
  }
  renderGlobal();
  renderPage(true);
  renderLines();
  renderMarkdown();
  renderJson();
  renderTables();
  $("export").hidden = !result?.id;
  setStatus("export-status", "");
  if (state.plugins) renderPlugins();  // enables Run, clears outputs of the previous result
  onResultShown(result);
}

/** Re-render everything that depends on mode / thresholds (not Markdown, JSON or tables). */
function refreshConfidenceViews() {
  renderGlobal();
  renderOverlay();
  renderLines();
}

const currentPage = () => state.current?.pages?.[state.page] || null;

// --- global controls: confidence mode, filters, summary chips ---

function renderGlobal() {
  const r = state.current;
  const uncal = isUncalibrated();
  $("uncal-badge").hidden = !uncal;
  $("mode-calibrated").disabled = $("mode-raw").disabled = uncal;
  const mode = effectiveMode();
  $("mode-calibrated").setAttribute("aria-pressed", String(mode === "calibrated"));
  $("mode-raw").setAttribute("aria-pressed", String(mode === "raw"));
  const note = $("mode-note");
  note.hidden = !uncal;
  note.textContent = uncal
    ? "No calibrator was active when this ran, so only raw engine confidence exists. Fit one in the Calibration tab and run the file again to compare."
    : "";

  const chips = $("chips");
  chips.replaceChildren();
  if (!r) return;
  const lines = (r.pages || []).flatMap((p) => p.lines || []);
  const confs = lines.map(confOf).filter((c) => c != null);
  const mean = confs.length ? confs.reduce((a, b) => a + b, 0) / confs.length : null;
  const review = confs.filter(needsReview).length;
  const chip = (label, value, cls) => chips.append(el("li", { class: cls }, `${label} `, el("b", { text: value })));
  chip("File", r.filename || "–");
  chip("Pipeline", PIPELINE_LABEL[r.pipeline] || r.pipeline || "–");
  chip("Pages", String(r.summary?.page_count ?? (r.pages || []).length));
  chip("Lines", String(lines.length));
  chip(`Mean ${mode}`, mean == null ? "–" : pct(mean));
  chip("Needs review", String(review), review ? "warn" : "");
  chip("Time", formatMs(r.processing_time_ms));
}

function setThreshold(v, touched = true) {
  state.threshold = clamp(Number(v), 0, 1);
  if (touched) state.thresholdTouched = true;
  for (const [input, out] of [["threshold", "threshold-out"], ["threshold2", "threshold2-out"]]) {
    $(input).value = String(state.threshold);
    $(out).textContent = pct(state.threshold);
  }
}

$("mode-calibrated").addEventListener("click", () => { state.mode = "calibrated"; refreshConfidenceViews(); });
$("mode-raw").addEventListener("click", () => { state.mode = "raw"; refreshConfidenceViews(); });
$("min-conf").addEventListener("input", (e) => {
  state.minConf = Number(e.target.value);
  $("min-conf-out").textContent = pct(state.minConf);
  renderOverlay();
  renderLines();
});
for (const id of ["threshold", "threshold2"]) {
  $(id).addEventListener("input", (e) => { setThreshold(e.target.value); refreshConfidenceViews(); });
}

// ---------------------------------------------------------------------------------------------
// Page viewer: image + SVG overlay, zoom (buttons, Ctrl/Cmd+wheel, keys), drag to pan, pages
// ---------------------------------------------------------------------------------------------

const viewer = $("viewer"), stage = $("stage"), img = $("page-img"), overlay = $("overlay");

function pageSize(page) {
  return { w: page?.width || img.naturalWidth || 1000, h: page?.height || img.naturalHeight || 1400 };
}

function showPlaceholder(title, text) {
  stage.hidden = true;
  const ph = $("placeholder");
  ph.hidden = false;
  ph.replaceChildren(el("strong", { text: title }), el("span", { text }));
}

function renderPage(refit) {
  const r = state.current, page = currentPage();
  const n = r?.pages?.length || 0;
  $("page-label").textContent = n ? `Page ${state.page + 1} of ${n}` : "No page";
  $("page-prev").disabled = state.page <= 0;
  $("page-next").disabled = state.page >= n - 1;
  const hasImage = !!page?.image_url;
  for (const id of ["zoom-in", "zoom-out", "zoom-fit"]) $(id).disabled = !hasImage;
  if (!hasImage) $("zoom-val").textContent = "–";

  if (!r) {
    showPlaceholder("No document yet", "Add a file or pick a sample, then press Run. The page appears here with a box around every line it read.");
    return;
  }
  if (!page) {
    showPlaceholder("Nothing to show", "This result has no pages.");
    return;
  }
  if (!hasImage) {
    const office = r.pipeline === "office";
    showPlaceholder(office ? "No page image for Office files" : "No page image",
      office ? "Word, Excel and PowerPoint files are converted straight to Markdown without OCR. See the Markdown and Tables tabs."
             : "The server did not return an image for this page.");
    return;
  }
  $("placeholder").hidden = true;
  stage.hidden = false;
  const sizeAndDraw = () => {
    const { w, h } = pageSize(page);
    stage.style.width = `${w}px`;
    stage.style.height = `${h}px`;
    overlay.setAttribute("viewBox", `0 0 ${w} ${h}`);
    renderOverlay();
    if (refit || state.view.fitted) fitView();
    else applyView();
  };
  img.alt = `Page ${state.page + 1} of ${r.filename || "the document"}`;
  if (img.getAttribute("src") !== page.image_url) {
    img.onload = sizeAndDraw;
    img.onerror = () => showPlaceholder("Could not load the page image", "The result may have been deleted on the server. Run the file again.");
    img.setAttribute("src", page.image_url);
    if (page.width && page.height) sizeAndDraw();  // boxes can be drawn before the pixels arrive
  } else {
    sizeAndDraw();
  }
}

function renderOverlay() {
  overlay.replaceChildren();
  const page = currentPage();
  if (!page?.image_url) return;
  (page.lines || []).forEach((line, i) => {
    if (!visible(line)) return;
    const pts = polygonOf(line);
    if (!pts) return;
    const c = confOf(line), id = lineId(line, state.page, i);
    const poly = svg("polygon", { points: pts.map((p) => p.join(",")).join(" "), class: tierOf(c), "data-id": id });
    if (id === state.selected) poly.classList.add("sel");
    const title = svg("title");
    title.textContent = `${line.text} (${pct(c)}${needsReview(c) ? ", needs review" : ""})`;
    poly.append(title);
    overlay.append(poly);
  });
}

/** Line outline as [[x, y], ...]: the polygon if given, else the axis-aligned box. */
function polygonOf(line) {
  if (Array.isArray(line.polygon) && line.polygon.length >= 3) return line.polygon;
  if (Array.isArray(line.box) && line.box.length === 4) {
    const [x0, y0, x1, y1] = line.box;
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
  }
  return null;
}

function applyView() {
  const v = state.view;
  stage.style.transform = `translate(${v.x}px, ${v.y}px) scale(${v.z})`;
  $("zoom-val").textContent = `${Math.round(v.z * 100)}%`;
}

function fitView() {
  const page = currentPage();
  if (!page?.image_url) return;
  const { w, h } = pageSize(page);
  const r = viewer.getBoundingClientRect();
  const pad = 12;
  const z = clamp(Math.min((r.width - pad * 2) / w, (r.height - pad * 2) / h), 0.02, 8);
  Object.assign(state.view, { z, x: (r.width - w * z) / 2, y: (r.height - h * z) / 2, fitted: true });
  applyView();
}

/** Zoom by `factor`, keeping the point (cx, cy) in viewer pixels fixed. Defaults to the centre. */
function zoomBy(factor, cx, cy) {
  const v = state.view, r = viewer.getBoundingClientRect();
  cx ??= r.width / 2;
  cy ??= r.height / 2;
  const z = clamp(v.z * factor, 0.02, 8);
  v.x = cx - (cx - v.x) * (z / v.z);
  v.y = cy - (cy - v.y) * (z / v.z);
  v.z = z;
  v.fitted = false;
  applyView();
}

function goPage(delta) {
  const n = state.current?.pages?.length || 0;
  const next = clamp(state.page + delta, 0, Math.max(0, n - 1));
  if (next === state.page) return;
  state.page = next;
  state.selected = null;
  renderPage(true);
  renderLines();
  renderLayout();
}

$("zoom-in").addEventListener("click", () => zoomBy(1.25));
$("zoom-out").addEventListener("click", () => zoomBy(1 / 1.25));
$("zoom-fit").addEventListener("click", fitView);
$("page-prev").addEventListener("click", () => goPage(-1));
$("page-next").addEventListener("click", () => goPage(1));

// Ctrl/Cmd + wheel (and trackpad pinch, which arrives as ctrl+wheel) zooms at the cursor.
viewer.addEventListener("wheel", (e) => {
  if (!(e.ctrlKey || e.metaKey) || stage.hidden) return;
  e.preventDefault();
  const r = viewer.getBoundingClientRect();
  zoomBy(Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0018)), e.clientX - r.left, e.clientY - r.top);
}, { passive: false });

// Drag to pan. A press that does not move is a click: on a box it selects that line.
let drag = null;
viewer.addEventListener("pointerdown", (e) => {
  if (e.button !== 0 || stage.hidden) return;
  drag = { x: e.clientX, y: e.clientY, vx: state.view.x, vy: state.view.y, moved: false, target: e.target };
  try { viewer.setPointerCapture(e.pointerId); } catch { /* pointer already gone */ }
});
viewer.addEventListener("pointermove", (e) => {
  if (drag) {
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (!drag.moved && Math.hypot(dx, dy) < 4) return;
    drag.moved = true;
    viewer.classList.add("panning");
    state.view.x = drag.vx + dx;
    state.view.y = drag.vy + dy;
    state.view.fitted = false;
    applyView();
    return;
  }
  // Hover a box: highlight its line in the list (pointer capture is off while not dragging).
  const id = e.target instanceof SVGPolygonElement ? e.target.dataset.id : null;
  if (id !== state.hovered) highlight(id, "box");
});
viewer.addEventListener("pointerleave", () => { if (!drag) highlight(null, "box"); });
const endDrag = (e) => {
  if (!drag) return;
  const { moved, target } = drag;
  drag = null;
  viewer.classList.remove("panning");
  if (viewer.hasPointerCapture(e.pointerId)) viewer.releasePointerCapture(e.pointerId);
  if (!moved && target instanceof SVGPolygonElement) selectLine(target.dataset.id, { edit: true, fromBox: true });
};
viewer.addEventListener("pointerup", endDrag);
viewer.addEventListener("pointercancel", endDrag);

viewer.addEventListener("keydown", (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const step = 60;
  const actions = {
    ArrowLeft: () => goPage(-1), PageUp: () => goPage(-1),
    ArrowRight: () => goPage(1), PageDown: () => goPage(1),
    ArrowUp: () => { state.view.y += step; state.view.fitted = false; applyView(); },
    ArrowDown: () => { state.view.y -= step; state.view.fitted = false; applyView(); },
    "+": () => zoomBy(1.25), "=": () => zoomBy(1.25), "-": () => zoomBy(1 / 1.25), "0": fitView,
  };
  if (actions[e.key]) { e.preventDefault(); actions[e.key](); }
});

// Keep the page fitted when the window (and so the viewer) changes size.
new ResizeObserver(() => { if (state.view.fitted && !stage.hidden) fitView(); }).observe(viewer);

/** Pan so a line's box is inside the viewer (no-op when it is already fully visible). */
function ensureBoxVisible(id) {
  const poly = overlay.querySelector(`polygon[data-id="${CSS.escape(id)}"]`);
  if (!poly || stage.hidden) return;
  const b = poly.getBBox(), v = state.view, r = viewer.getBoundingClientRect();
  const x0 = v.x + b.x * v.z, y0 = v.y + b.y * v.z, x1 = x0 + b.width * v.z, y1 = y0 + b.height * v.z;
  if (x0 >= 0 && y0 >= 0 && x1 <= r.width && y1 <= r.height) return;
  v.x = r.width / 2 - (b.x + b.width / 2) * v.z;
  v.y = r.height / 2 - (b.y + b.height / 2) * v.z;
  v.fitted = false;
  applyView();
}

// ---------------------------------------------------------------------------------------------
// Text tab: line list with confidence badges, review flags and inline correction
// ---------------------------------------------------------------------------------------------

const linesEl = $("lines"), textPanel = $("panel-text");

function renderLines() {
  const scroll = textPanel.scrollTop;
  linesEl.replaceChildren();
  const r = state.current, page = currentPage();
  $("text-hint").hidden = !page?.lines?.length;
  if (!r) { linesEl.append(el("li", { class: "empty", text: "Run a document to see its lines here." })); return; }
  if (!page) { linesEl.append(el("li", { class: "empty", text: "This result has no pages." })); return; }
  const n = r.pages.length;
  const all = page.lines || [];
  if (!all.length) {
    linesEl.append(el("li", { class: "empty",
      text: "No text lines were found on this page." }));
    return;
  }
  const shown = all.filter(visible).length;
  linesEl.append(el("li", { class: "page-head", "aria-hidden": "true",
    text: `${n > 1 ? `Page ${state.page + 1} of ${n} · ` : ""}${shown} of ${all.length} lines` }));
  all.forEach((line, i) => {
    if (!visible(line)) return;
    const id = lineId(line, state.page, i), c = confOf(line), tier = tierOf(c);
    const llm = isLlmLine(line);
    const shown = state.showOriginal ? DocLayout.originalText(line) : line.text;
    const textBtn = el("button", { type: "button", class: `ln-text${line.text ? "" : " empty-text"}`,
      "aria-label": `Line ${i + 1}: ${shown || "empty"}${llm && !state.showOriginal ? `, refined by the LLM; original: ${line.ocr_text}` : ""}. Confidence ${pct(c)}${needsReview(c) ? ", needs review" : ""}. Press to edit.` },
      shown ? DocLayout.lineContent(line, state.showOriginal ? "original" : "refined") : "(empty)");
    textBtn.addEventListener("click", () => selectLine(id, { edit: true }));
    const li = el("li", { class: `ln${id === state.selected ? " sel" : ""}`, "data-id": id, "data-index": String(i) },
      el("span", { class: "ln-no", "aria-hidden": "true", text: String(i + 1) }),
      textBtn,
      el("span", { class: "ln-flags", "aria-hidden": "true" },
        llm ? el("span", { class: "flag llm", title: `Refined by the LLM. Original: ${line.ocr_text ?? ""}`, text: line.llm_status === "inferred" ? "LLM: inferred" : "LLM" })
          : line.edited ? el("span", { class: "flag edited", text: "edited" }) : null,
        line.llm_flag ? el("span", { class: "flag review", title: line.llm_flag, text: "LLM change rejected" }) : null,
        needsReview(c) ? el("span", { class: "flag review", text: "needs review" }) : null,
        el("span", { class: `conf ${tier}`, title: confTitle(line), text: pct(c) })));
    li.addEventListener("mouseenter", () => highlight(id, "list"));
    li.addEventListener("mouseleave", () => highlight(null, "list"));
    li.addEventListener("focusin", () => highlight(id, "list"));
    linesEl.append(li);
  });
  const hidden = all.length - shown;
  if (hidden) linesEl.append(el("li", { class: "filtered-note", text: `${hidden} line${hidden === 1 ? "" : "s"} hidden by the "Hide below ${pct(state.minConf)}" filter.` }));
  textPanel.scrollTop = scroll;
}

function confTitle(line) {
  const raw = line.raw_confidence, cal = line.calibrated_confidence;
  return isUncalibrated() ? `Raw confidence ${pct(raw)} (uncalibrated)` : `Calibrated ${pct(cal)} · raw ${pct(raw)}`;
}

/** Highlight one line in both the list and the overlay (id null clears). */
function highlight(id, from) {
  state.hovered = id;
  for (const n of document.querySelectorAll(".ln.hl, polygon.hl, .lt-line.hl, .rv.hl")) n.classList.remove("hl");
  if (!id) return;
  const sel = `[data-id="${CSS.escape(id)}"]`;
  const li = linesEl.querySelector(`li${sel}`), poly = overlay.querySelector(`polygon${sel}`);
  li?.classList.add("hl");
  poly?.classList.add("hl");
  document.querySelector(`#layout-view .lt-line${sel}`)?.classList.add("hl");
  if (from === "list" || from === "layout" || from === "changes") ensureBoxVisible(id);
  if (from === "box" && li && !textPanel.hidden) scrollWithin(textPanel, li);
}

/** Scroll `child` into view inside the scrolling `container` only (never the whole page). */
function scrollWithin(container, child) {
  const c = container.getBoundingClientRect(), r = child.getBoundingClientRect();
  if (r.top < c.top + 40) container.scrollTop -= c.top + 40 - r.top;
  else if (r.bottom > c.bottom - 8) container.scrollTop += r.bottom - (c.bottom - 8);
}

function selectLine(id, { edit = false, fromBox = false } = {}) {
  state.selected = id;
  for (const n of document.querySelectorAll(".ln.sel, polygon.sel")) n.classList.remove("sel");
  const sel = `[data-id="${CSS.escape(id)}"]`;
  overlay.querySelector(`polygon${sel}`)?.classList.add("sel");
  if (fromBox) activateTab("tab-text", false);
  const li = linesEl.querySelector(`li${sel}`);
  if (!li) return;
  li.classList.add("sel");
  scrollWithin(textPanel, li);
  if (fromBox && window.matchMedia("(max-width: 899px)").matches) li.scrollIntoView({ block: "center", behavior: "smooth" });
  if (edit) startEdit(li);
}

/** Swap a line's text for an input. Enter or leaving the field saves, Esc cancels. */
function startEdit(li) {
  if (li.querySelector(".ln-edit")) return;
  const i = Number(li.dataset.index), page = state.page;
  const line = currentPage().lines[i];
  const btn = li.querySelector(".ln-text");
  const input = el("input", { type: "text", class: "ln-edit", "aria-label": `Correct line ${i + 1}` });
  input.value = line.text || "";
  btn.replaceWith(input);
  input.focus();
  input.select();
  let finished = false;
  const finish = (save) => {
    if (finished) return;
    finished = true;
    const text = input.value;
    if (!save || text === (line.text || "")) {
      input.replaceWith(btn);
      if (!save) btn.focus();
      return;
    }
    saveLine(page, i, text, li, input);
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); finish(true); }
    else if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); finish(false); }
  });
  input.addEventListener("blur", () => finish(true));
}

async function saveLine(pageIndex, lineIndex, text, li, input) {
  const r = state.current;
  input.disabled = true;
  const msg = el("span", { class: "ln-msg saving", role: "status", text: "Saving…" });
  li.append(msg);
  try {
    const updated = await api(`/api/results/${encodeURIComponent(r.id)}/lines`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify([{ page: pageIndex, line: lineIndex, text }]),
    });
    if (updated && Array.isArray(updated.pages)) {
      // Keep the view (page, zoom, selection) and swap in the server's copy.
      state.current = updated;
      const job = state.jobs.find((j) => j.key === state.currentKey);
      if (job) job.result = updated;
    } else {
      const line = r.pages[pageIndex].lines[lineIndex];
      line.text = text;
      line.edited = true;
    }
    renderLines();
    renderOverlay();
    renderMarkdown();
    renderJson();
    renderTables();
    setStatus("job-status", `Line ${lineIndex + 1} saved.`, "ok");
  } catch (err) {
    renderLines();
    const again = linesEl.querySelector(`li[data-index="${lineIndex}"]`);
    again?.append(el("span", { class: "ln-msg", role: "alert", text: `Not saved: ${err.message}` }));
  }
}

const { markdownToHtml, sanitizeTable, diffNodes, isLlmLine } = window.DocLayout;  // static/layout.js

// ---------------------------------------------------------------------------------------------
// Markdown: a small renderer that escapes everything first (headings, lists, emphasis, code,
// quotes, pipe tables). HTML <table> blocks from the parser are sanitised and kept; figures
// become a short note; any other HTML is shown as text.
// ---------------------------------------------------------------------------------------------

function renderMarkdown() {
  let md = state.current?.markdown ?? (state.current?.pages || []).map((p) => p.markdown || "").join("\n\n");
  const allLines = (state.current?.pages || []).flatMap((p) => p.lines || []);
  if (state.showOriginal) md = DocLayout.originalMarkdown(md, allLines);
  const view = $("md-view"), source = $("md-source");
  source.textContent = md || "";
  if (!state.current) view.innerHTML = '<p class="empty">Run a document to see its Markdown here.</p>';
  else if (!md) view.innerHTML = '<p class="empty">This result has no Markdown.</p>';
  else {
    view.innerHTML = markdownToHtml(md);  // safe: markdownToHtml escapes all text itself
    if (!state.showOriginal) DocLayout.highlightIn(view, allLines);
  }
  view.hidden = state.mdRaw;
  source.hidden = !state.mdRaw;
  $("md-rendered").setAttribute("aria-pressed", String(!state.mdRaw));
  $("md-raw").setAttribute("aria-pressed", String(state.mdRaw));
  for (const id of ["md-copy", "md-download"]) $(id).disabled = !md;
}
$("md-rendered").addEventListener("click", () => { state.mdRaw = false; renderMarkdown(); });
$("md-raw").addEventListener("click", () => { state.mdRaw = true; renderMarkdown(); });
$("md-copy").addEventListener("click", (e) => copyText($("md-source").textContent, e.currentTarget));
$("md-download").addEventListener("click", () =>
  downloadText($("md-source").textContent, `${baseName(state.current?.filename)}.md`, "text/markdown;charset=utf-8"));

// ---------------------------------------------------------------------------------------------
// JSON tab
// ---------------------------------------------------------------------------------------------

function renderJson() {
  const text = state.current ? JSON.stringify(state.current, null, 2) : "";
  $("json-view").textContent = text || "Run a document to see the raw result here.";
  $("json-copy").disabled = $("json-download").disabled = !text;
}
$("json-copy").addEventListener("click", (e) => copyText(JSON.stringify(state.current, null, 2), e.currentTarget));
$("json-download").addEventListener("click", () =>
  downloadText(JSON.stringify(state.current, null, 2), `${baseName(state.current?.filename)}.json`, "application/json"));

// ---------------------------------------------------------------------------------------------
// Tables tab: every table as HTML, each with a client-side CSV download
// ---------------------------------------------------------------------------------------------

function tableFromRows(rows) {
  const wrap = el("div", { class: "table-wrap" });
  const table = el("table");
  rows.forEach((row, r) => {
    const tr = el("tr");
    for (const cell of row) tr.append(el(r === 0 ? "th" : "td", { text: cell == null ? "" : String(cell) }));
    table.append(tr);
  });
  wrap.append(table);
  return wrap;
}

function toCsv(rows) {
  const cell = (v) => { const s = v == null ? "" : String(v); return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
  return "﻿" + rows.map((r) => r.map(cell).join(",")).join("\r\n");  // BOM so Excel reads UTF-8
}

function renderTables() {
  const view = $("tables-view");
  view.replaceChildren();
  const r = state.current;
  if (!r) { view.append(el("p", { class: "empty", text: "Run a document to see its tables here." })); return; }
  let count = 0;
  (r.pages || []).forEach((page, pi) => {
    (page.tables || []).forEach((t) => {
      count++;
      const node = t.html ? sanitizeTable(t.html) : tableFromRows(t.rows || []);
      const domTable = node.querySelector("table");
      const rows = Array.isArray(t.rows) && t.rows.length ? t.rows
        : domTable ? [...domTable.rows].map((tr) => [...tr.cells].map((c) => c.textContent.trim())) : [];
      const csv = el("button", { type: "button", class: "btn ghost", text: "Download CSV" });
      const name = `${baseName(r.filename)}-table-${count}.csv`;
      csv.addEventListener("click", () => downloadText(toCsv(rows), name, "text/csv;charset=utf-8"));
      csv.disabled = !rows.length;
      view.append(el("section", { class: "tbl", "aria-label": `Table ${count}` },
        el("div", { class: "tbl-head" },
          el("h4", { text: `Table ${count}${r.pages.length > 1 ? ` · page ${pi + 1}` : ""}` }), csv),
        node));
    });
  });
  if (!count) {
    view.append(el("p", { class: "empty",
      text: r.pipeline === "ocr"
        ? "Text OCR returns lines only. Run Document parsing (PP-StructureV3) to detect tables."
        : "No tables were found in this document." }));
  }
}

// ---------------------------------------------------------------------------------------------
// Calibration tab: report, metrics before -> after, reliability diagram, refit
// ---------------------------------------------------------------------------------------------

async function loadCalibration() {
  state.calibrationLoaded = true;
  try {
    state.calibration = await api("/api/calibration");
  } catch (err) {
    state.calibration = { status: "error", error: err.message };
  }
  renderCalibration();
}

function renderCalibration() {
  const view = $("calib-view"), c = state.calibration;
  view.replaceChildren();
  if (!c) { view.append(el("p", { class: "hint", text: "Loading the calibration report…" })); return; }
  if (c.status === "error") { view.append(el("p", { class: "status error", text: c.error })); return; }
  if (!c.method) {
    view.append(el("p", {}, el("strong", { text: "No calibrator fitted yet. " }),
      "The confidence shown is the engine's raw score, which is usually over-confident. ",
      "Fit a calibrator on the bundled sample data, or on your own labelled pages, to map it to how often lines are actually read correctly."));
    return;
  }
  const grid = el("div", { class: "calib-grid" });
  const meta = el("dl", { class: "calib-meta" });
  const when = c.fitted_at ? new Date(c.fitted_at) : null;
  const rows = [
    ["Method", `${c.method}${c.selected_by ? ` (chosen by ${c.selected_by})` : ""}`],
    ["Fitted", when && !Number.isNaN(when.getTime()) ? when.toLocaleString() : c.fitted_at],
    ["Data", c.dataset],
    ["Correct when", c.match === "exact" ? "text matches exactly" : c.match ? `CER ≤ ${c.cer_threshold ?? "?"}` : null],
    ["Lines", c.n_lines != null ? `${c.n_lines} (${c.n_correct ?? "?"} correct; ${c.n_train ?? "?"} to fit, ${c.n_val ?? "?"} to check)` : null],
  ];
  for (const [k, v] of rows) if (v != null && v !== "") meta.append(el("dt", { text: k }), el("dd", { text: String(v) }));
  if (c.candidates && typeof c.candidates === "object") {
    const txt = Object.entries(c.candidates)
      .map(([m, v]) => `${m} ${v?.val_ece != null ? v.val_ece.toFixed(3) : "–"}${m === c.method ? " (chosen)" : ""}`).join(" · ");
    meta.append(el("dt", { text: "Validation ECE" }), el("dd", { text: txt }));
  }
  grid.append(meta);

  const m = c.metrics || {};
  if (m.before || m.after) {
    const table = el("table", { class: "metrics" },
      el("caption", { class: "visually-hidden", text: "Calibration metrics before and after, lower is better" }),
      el("thead", {}, el("tr", {}, el("th", { scope: "col", text: "Metric (lower is better)" }),
        el("th", { scope: "col", text: "Before" }), el("th", { scope: "col", text: "After" }))));
    const tbody = el("tbody");
    for (const [key, label] of [["ece", "ECE"], ["mce", "MCE"], ["brier", "Brier"], ["nll", "NLL"]]) {
      const b = m.before?.[key], a = m.after?.[key];
      const cls = a == null || b == null ? "" : a < b ? "better" : a > b ? "worse" : "";
      tbody.append(el("tr", {}, el("th", { scope: "row", text: label }),
        el("td", { text: b == null ? "–" : b.toFixed(3) }), el("td", { class: cls, text: a == null ? "–" : a.toFixed(3) })));
    }
    table.append(tbody);
    grid.append(table);
  }
  if (c.bins?.before || c.bins?.after) {
    grid.append(el("div", {}, reliabilityDiagram(c.bins),
      el("ul", { class: "diagram-legend" },
        el("li", {}, el("i", { style: "background: var(--box-moderate); opacity: .6" }), "before (raw)"),
        el("li", {}, el("i", { style: "background: var(--accent)" }), "after (calibrated)"),
        el("li", {}, el("i", { style: "border-top: 2px dashed var(--faint); height: 0; border-radius: 0" }), "perfect calibration")),
      el("p", { class: "hint" }, "Each bar is the share of lines read correctly among lines with that confidence. Bars on the dashed line mean the confidence can be taken at face value. ",
        el("a", { href: "/api/calibration/diagram.png", target: "_blank", rel: "noopener" }, "Open the server's diagram (PNG)"))));
  }
  view.append(grid);
}

/** Reliability diagram as inline SVG: accuracy per confidence bin, before vs after, colours from CSS. */
function reliabilityDiagram(bins) {
  const W = 360, H = 290, L = 44, T = 14, PW = 300, PH = 230;
  const X = (v) => L + v * PW, Y = (v) => T + (1 - v) * PH;
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "diagram", role: "img",
    "aria-label": "Reliability diagram: accuracy against confidence, before and after calibration" });
  for (let k = 0; k <= 5; k++) {
    const v = k / 5;
    root.append(svg("line", { x1: X(0), x2: X(1), y1: Y(v), y2: Y(v), class: "grid" }));
    const ty = svg("text", { x: L - 6, y: Y(v) + 4, "text-anchor": "end" }); ty.textContent = v.toFixed(1); root.append(ty);
    const tx = svg("text", { x: X(v), y: T + PH + 16, "text-anchor": "middle" }); tx.textContent = v.toFixed(1); root.append(tx);
  }
  const series = [["before", "bar-before", 0.08], ["after", "bar-after", 0.5]];  // offsets within each bin
  for (const [key, cls, offset] of series) {
    for (const b of bins[key] || []) {
      if (b.accuracy == null || !b.count) continue;
      const w = (b.hi - b.lo) * PW;
      const x = X(b.lo) + w * offset, bw = w * 0.42;
      const rect = svg("rect", { x, y: Y(b.accuracy), width: bw, height: Math.max(0, Y(0) - Y(b.accuracy)), class: cls });
      const t = svg("title");
      t.textContent = `${key}: confidence ${b.lo.toFixed(1)}–${b.hi.toFixed(1)}, ${b.count} lines, ${pct(b.accuracy)} correct` +
        (b.mean_confidence != null ? `, mean confidence ${pct(b.mean_confidence)}` : "");
      rect.append(t);
      root.append(rect);
    }
  }
  root.append(svg("line", { x1: X(0), y1: Y(0), x2: X(1), y2: Y(1), class: "ideal" }));
  root.append(svg("line", { x1: X(0), y1: Y(0), x2: X(1), y2: Y(0), class: "axis" }));
  root.append(svg("line", { x1: X(0), y1: Y(0), x2: X(0), y2: Y(1), class: "axis" }));
  const xl = svg("text", { x: X(0.5), y: H - 4, "text-anchor": "middle" }); xl.textContent = "Confidence";
  const yl = svg("text", { x: 12, y: Y(0.5), "text-anchor": "middle", transform: `rotate(-90 12 ${Y(0.5)})` }); yl.textContent = "Accuracy";
  root.append(xl, yl);
  return root;
}

/** POST /api/calibrate, then poll the job; on success reload the report and the engine pill. */
async function startCalibration(form) {
  if (state.calibrating) return;
  state.calibrating = true;
  $("refit-sample").disabled = $("calib-zip-btn").disabled = true;
  setStatus("calib-status", "Starting calibration…");
  try {
    form.append("match", $("calib-match").value);
    const { body } = await api("/api/calibrate", { method: "POST", body: form }, true);
    if (body.id && body.status !== "done") {
      let last = "";
      for (;;) {
        await sleep(POLL_MS);
        const job = await api(`/api/results/${encodeURIComponent(body.id)}`);
        if (job.status === "done") break;
        if (job.status === "error") throw new Error(job.error || job.message || "Calibration failed.");
        const msg = `Calibrating: ${job.message || "working…"}${job.progress ? ` (${pct(job.progress)})` : ""}`;
        if (msg !== last) { setStatus("calib-status", msg); last = msg; }
      }
    }
    await loadCalibration();
    loadHealth();
    const method = state.calibration?.method;
    setStatus("calib-status", `Calibrator fitted${method ? ` (${method})` : ""}. Run a document again to use it.`, "ok");
  } catch (err) {
    setStatus("calib-status", err.message, "error");
  } finally {
    state.calibrating = false;
    $("refit-sample").disabled = $("calib-zip-btn").disabled = false;
  }
}

$("refit-sample").addEventListener("click", () => {
  const form = new FormData();
  form.append("use_sample", "true");
  startCalibration(form);
});
$("calib-zip-btn").addEventListener("click", () => $("calib-zip").click());
$("calib-zip").addEventListener("change", (e) => {
  const file = e.target.files[0];
  e.target.value = "";
  if (!file) return;
  if (extOf(file.name) !== ".zip") { setStatus("calib-status", "Choose a .zip file.", "error"); return; }
  const form = new FormData();
  form.append("file", file, file.name);
  startCalibration(form);
});

// ---------------------------------------------------------------------------------------------
// Image clean-up (preprocess): options sent with each run, plus a before/after preview.
// POST /api/preprocess may not exist on older servers: a 404 shows a short note instead.
// ---------------------------------------------------------------------------------------------

/** {"deskew": bool, "denoise": bool, "contrast": bool, "binarize": bool} from the checkboxes. */
function preprocessOptions() {
  const opts = {};
  for (const box of document.querySelectorAll('input[name="prep"]')) opts[box.value] = box.checked;
  return opts;
}
const anyPreprocess = () => Object.values(preprocessOptions()).some(Boolean);
const isImageName = (name) => ACCEPTED.includes(extOf(name)) && extOf(name) !== ".pdf" && !OFFICE.includes(extOf(name));
const safeDataImage = (s) => typeof s === "string" && /^data:image\/(png|jpeg|webp);base64,[A-Za-z0-9+/=]+$/.test(s);

$("prep-preview").addEventListener("click", async () => {
  const view = $("prep-view");
  view.hidden = false;
  const job = state.jobs.find((j) => j.file && isImageName(j.name));
  const say = (text, kind = "") => view.replaceChildren(el("p", { class: `status ${kind}`.trim(), text }));
  if (!job) { say("Add an image (not a PDF or Office file) to the list first.", "error"); return; }
  if (!anyPreprocess()) { say("Tick at least one clean-up option.", "error"); return; }
  say(`Cleaning up ${job.name}…`);
  const form = new FormData();
  form.append("file", job.file, job.name);
  form.append("preprocess", JSON.stringify(preprocessOptions()));
  try {
    const r = await api("/api/preprocess", { method: "POST", body: form });
    if (!safeDataImage(r.before) || !safeDataImage(r.after)) throw new Error("The server returned no preview images.");
    const fig = (src, cap) => el("figure", {}, el("img", { src, alt: `${cap}: ${job.name}` }), el("figcaption", { text: cap }));
    view.replaceChildren(fig(r.before, "Before"), fig(r.after, "After"));
  } catch (err) {
    say(err.status === 404 || err.status === 405 ? "Preview is not available on this server yet." : err.message, "error");
  }
});

// ---------------------------------------------------------------------------------------------
// Export: GET /api/results/{id}/export/{fmt} (txt, md, json, csv, docx, pdf)
// ---------------------------------------------------------------------------------------------

for (const b of document.querySelectorAll("#export [data-fmt]")) {
  b.addEventListener("click", async () => {
    const r = state.current, fmt = b.dataset.fmt;
    if (!r?.id) return;
    b.disabled = true;
    setStatus("export-status", "");
    try {
      const resp = await fetch(`/api/results/${encodeURIComponent(r.id)}/export/${fmt}`);
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}));
        // FastAPI answers an unknown route with 404 "Not Found": the export endpoint is not deployed yet.
        const unknownRoute = resp.status === 404 && (body.detail === "Not Found" || typeof body.detail !== "string");
        throw new Error(unknownRoute ? "Export is not available on this server yet."
          : typeof body.detail === "string" ? body.detail : `Export failed (${resp.status})`);
      }
      const blob = await resp.blob();
      // Prefer the server's file name (Content-Disposition), else "<original name>.<fmt>".
      const cd = resp.headers.get("Content-Disposition") || "";
      const m = /filename\*=UTF-8''([^;]+)|filename="?([^";]+)"?/i.exec(cd);
      const name = m ? decodeURIComponent(m[1] || m[2]) : `${baseName(r.filename)}.${fmt}`;
      const url = URL.createObjectURL(blob);
      const a = el("a", { href: url, download: name });
      document.body.append(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (err) {
      setStatus("export-status", err.status === 0 ? "Could not reach the server." : err.message, "error");
    } finally {
      b.disabled = false;
    }
  });
}

// ---------------------------------------------------------------------------------------------
// Documents: GET /documents/library?q= (the same documents as the question page, newest first).
// Click to open; documents still being read show their progress (polled); × deletes everywhere.
// ---------------------------------------------------------------------------------------------

let historyTimer = null, libraryPoll = null;
const REFINE_TAG = { refined: "refined by LLM", reverted: "original text", unavailable: "not refined", off: "", empty: "" };

async function loadHistory() {
  const ul = $("history");
  const q = $("history-q").value.trim();
  clearTimeout(libraryPoll);
  try {
    const rows = await api(`/documents/library${q ? `?q=${encodeURIComponent(q)}` : ""}`);
    ul.replaceChildren();
    if (!Array.isArray(rows) || !rows.length) {
      ul.append(el("li", { class: "empty", text: q ? "No documents match." : "No documents yet." }));
      return;
    }
    for (const row of rows) ul.append(libraryItem(row));
    // Keep progress live while anything is being read or refined.
    if (rows.some((r) => r.status === "processing" || r.refine)) libraryPoll = setTimeout(loadHistory, 2000);
  } catch (err) {
    ul.replaceChildren(el("li", { class: "empty", text: err.message }));
  }
}

function libraryItem(row) {
  const when = row.created_at ? new Date(row.created_at) : null;
  const ref = row.refinement?.status;
  const meta = [row.format_type, PIPELINE_LABEL[row.pipeline] || null,
    row.page_count != null ? `${row.page_count} p.` : null,
    row.status === "processing" ? row.message || "Reading…" : null,
    row.status === "error" ? "failed" : null,
    row.refine ? row.refine.message : REFINE_TAG[ref] || null,
    !row.doc_id && row.status === "ready" ? "not on the question page" : null,
    when && !Number.isNaN(when.getTime()) ? when.toLocaleString([], { dateStyle: "short", timeStyle: "short" }) : null,
  ].filter(Boolean).join(" · ");
  const open = el("button", { type: "button", class: "h-open", title: row.error || `Open ${row.name}` },
    el("span", { class: "h-name", text: row.name || row.run_id }),
    el("span", { class: `h-meta${row.status === "error" || ref === "unavailable" ? " bad-text" : ""}`, text: meta }));
  if (row.run_id && row.status === "ready") open.addEventListener("click", () => openHistory(row.run_id));
  else if (!row.run_id && row.doc_id) {
    open.title = "Uploaded before the two pages were shared: read it here to see its layout";
    open.addEventListener("click", () => readIntoStudio(row));
  } else open.disabled = row.status === "processing";
  const li = el("li", { class: row.status === "processing" ? "processing" : "" }, open);
  if (row.status === "processing" && typeof row.progress === "number") {
    const bar = el("progress", { max: "1", "aria-label": `Progress for ${row.name}` });
    bar.value = row.progress;
    li.append(bar);
  }
  if (row.status !== "processing") {
    const del = el("button", { type: "button", class: "job-x", "aria-label": `Delete ${row.name}`, title: "Delete from both pages" }, "×");
    del.addEventListener("click", () => deleteHistory(row));
    li.append(del);
  }
  return li;
}

async function openHistory(id, tab) {
  try {
    const r = await api(`/api/results/${encodeURIComponent(id)}`);
    if (r.status !== "done") { setStatus("job-status", `That document is ${r.status}${r.error ? `: ${r.error}` : ""}.`, r.status === "error" ? "error" : ""); return; }
    state.currentKey = null;
    setResult(r);
    renderQueue();
    setStatus("job-status", `Opened ${r.filename || "document"}.`);
    if (tab) activateTab(`tab-${tab}`, false);
  } catch (err) {
    setStatus("job-status", err.message, "error");
  }
}

async function readIntoStudio(row) {
  try {
    await api(`/documents/${encodeURIComponent(row.doc_id)}/experience`, { method: "POST" });
    setStatus("job-status", `Reading ${row.name}…`);
  } catch (err) {
    setStatus("job-status", err.message, "error");
  }
  loadHistory();
}

async function deleteHistory(row) {
  if (!confirm(`Delete "${row.name}"? It is removed from DocBlendAI (both pages); your original file is not touched.`)) return;
  try {
    if (row.doc_id) await api(`/documents/${encodeURIComponent(row.doc_id)}`, { method: "DELETE" }).catch((err) => { if (err.status) throw err; });
    else await api(`/api/results/${encodeURIComponent(row.run_id)}`, { method: "DELETE" });
    if (state.current && (state.current.id === row.run_id)) { state.current = null; setResult(null); }
    setStatus("job-status", `Deleted ${row.name}.`);
  } catch (err) {
    setStatus("job-status", err.message, "error");
  }
  loadHistory();
}

/** Links from the question page: /studio?run=<run id>[&tab=layout|changes] or ?doc=<doc id>. */
async function openFromUrl() {
  const params = new URLSearchParams(location.search);
  const run = params.get("run"), doc = params.get("doc"), tab = params.get("tab");
  if (run) return openHistory(run, tab);
  if (doc) {
    const rows = await api("/documents/library").catch(() => []);
    const row = rows.find((r) => r.doc_id === doc);
    if (row?.run_id) return openHistory(row.run_id, tab);
    if (row) setStatus("job-status", `${row.name} has not been read here yet: click it in Documents to read it.`);
  }
}

$("history-refresh").addEventListener("click", loadHistory);
$("history-q").addEventListener("input", () => { clearTimeout(historyTimer); historyTimer = setTimeout(loadHistory, 300); });

// ---------------------------------------------------------------------------------------------
// Plugins tab: GET /api/plugins, run one with POST /api/results/{id}/plugins/{name}
// ---------------------------------------------------------------------------------------------

const PLUGIN_OPTIONS = {  // extra inputs for plugins that take options
  kie: { label: "Fields to extract (comma separated)", placeholder: "e.g. name, date, total", key: "fields", list: true },
  translate: { label: "Translate into", placeholder: "e.g. French", key: "target_language", value: "French" },
};
state.plugins = null;

async function loadPlugins() {
  try {
    state.plugins = await api("/api/plugins");
  } catch (err) {
    state.plugins = { error: err.status === 404 || err.status === 405 ? "Plugins are not available on this server yet." : err.message };
  }
  renderPlugins();
}

function renderPlugins() {
  const view = $("plugins-view"), p = state.plugins;
  view.replaceChildren();
  if (!p) { view.append(el("p", { class: "hint", text: "Loading plugins…" })); return; }
  if (p.error) { view.append(el("p", { class: "empty", text: p.error })); return; }
  if (!Array.isArray(p) || !p.length) { view.append(el("p", { class: "empty", text: "No plugins are installed." })); return; }
  for (const plugin of p) {
    if (plugin.name === "refine") continue;  // has its own "Refine with LLM" button and Review tab
    const on = plugin.enabled !== false;
    const card = el("section", { class: `plugin${on ? "" : " off"}`, "aria-label": plugin.title || plugin.name },
      el("h4", { text: plugin.title || plugin.name }),
      el("p", { text: plugin.description || "" }));
    if (!on) card.append(el("p", { class: "off-msg", text: plugin.message || "This plugin is turned off on the server." }));
    const form = el("form", { class: "plugin-form" });
    const opt = PLUGIN_OPTIONS[plugin.name];
    let input = null;
    if (opt) {
      input = el("input", { type: "text", placeholder: opt.placeholder, disabled: !on });
      input.value = opt.value || "";
      form.append(el("label", {}, opt.label, input));
    }
    const run = el("button", { type: "submit", class: "btn ghost", disabled: !on || !state.current?.id, text: "Run" });
    if (on && !state.current?.id) run.title = "Run a document first.";
    form.append(run);
    const out = el("pre", { class: "code", hidden: true });
    const status = el("p", { class: "status", role: "status" });
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      if (!state.current?.id) return;
      const options = {};
      if (opt) {
        const v = input.value.trim();
        options[opt.key] = opt.list ? v.split(",").map((s) => s.trim()).filter(Boolean) : v;
      }
      run.disabled = true;
      status.className = "status";
      status.textContent = "Running…";
      try {
        const res = await api(`/api/results/${encodeURIComponent(state.current.id)}/plugins/${encodeURIComponent(plugin.name)}`, {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(options),
        });
        out.textContent = typeof res === "string" ? res : JSON.stringify(res, null, 2);
        out.hidden = false;
        status.textContent = "";
      } catch (err) {
        status.className = "status error";
        status.textContent = err.message;
      } finally {
        run.disabled = false;
      }
    });
    card.append(form, status, out);
    view.append(card);
  }
}

// ---------------------------------------------------------------------------------------------
// LLM refinement (automatic on every upload): status bar, Show original, Revert / Use refined
// text, Retry; Changes tab (what Gemini changed, read-only) and versions; Layout tab.
// Contract §7: /api/results/{id}/changes, /revert, /reapply, /refine (retry), /versions.
// ---------------------------------------------------------------------------------------------

state.showOriginal = false;
state.refineJob = null;     // a retry being polled
state.layoutScale = "fit";  // number (px per page px) or "fit"
const STATUS_LABEL = { corrected: "corrected", inferred: "inferred", rejected: "rejected", unchanged: "unchanged" };

function renderRefineBar() {
  const r = state.current, bar = $("refine-bar");
  bar.hidden = !r?.id || r.status !== "done";
  if (bar.hidden) return;
  const ref = r.refinement || {};
  const busy = !!state.refineJob || !!r.active_job;
  const badge = $("llm-badge"), msg = $("refine-msg");
  const label = { refined: "Refined by LLM", reverted: "Original text", unavailable: "Not refined", off: "Not refined", empty: "Nothing to refine" }[ref.status] || "Not refined";
  badge.textContent = busy ? "Refining…" : label;
  badge.className = `llm-badge ${busy ? "busy" : ref.status || "none"}`;
  badge.title = ref.message || "";
  msg.textContent = busy ? "" : ref.status === "unavailable" ? ref.message.replace(/^Not refined: /, "")
    : ref.status === "refined" ? `${ref.corrected ?? 0} line${ref.corrected === 1 ? "" : "s"} corrected by ${ref.model || "Gemini"}.`
    : ref.status === "reverted" ? "Showing the original extraction; the LLM refinement is kept in the history."
    : ref.message || "";
  const llmLines = (r.pages || []).some((p) => (p.lines || []).some(isLlmLine));
  const llmOn = state.health?.llm?.enabled !== false;
  $("show-original").hidden = !llmLines;
  $("show-original").setAttribute("aria-pressed", String(state.showOriginal));
  $("show-original").textContent = state.showOriginal ? "Show refined text" : "Show original";
  $("revert-btn").hidden = ref.status !== "refined" || busy;
  $("reapply-btn").hidden = ref.status !== "reverted" || busy;
  $("retry-btn").hidden = busy || !["unavailable", "off", "empty"].includes(ref.status || "unavailable") || !llmOn;
  $("retry-btn").textContent = ref.status === "unavailable" ? "Retry" : "Refine with Gemini";
  const ask = $("ask-link");
  ask.hidden = !r.doc_id;
  if (r.doc_id) ask.href = `/?doc=${encodeURIComponent(r.doc_id)}`;
}

function onResultShown(result) {
  state.showOriginal = false;
  state.changes = null;
  $("version-compare").replaceChildren();
  renderRefineBar();
  renderLayout();
  renderChanges();
  if (!result?.id || result.status !== "done") return;
  if (result.active_job && state.refineJob !== result.active_job.id) pollRefine(result.active_job.id, result.id);
  loadChanges(result.id);
  if (!$("panel-changes").hidden) loadVersions();
}

/** Swap in a new copy of the current result, keeping the page, zoom and tab. */
function replaceCurrent(result) {
  const page = state.page, view = { ...state.view };
  const job = state.jobs.find((j) => j.key === state.currentKey);
  if (job) job.result = result;
  state.current = result;
  state.page = clamp(page, 0, Math.max(0, (result.pages || []).length - 1));
  state.view = view;
  renderGlobal();
  renderPage(false);
  renderLines();
  renderMarkdown();
  renderJson();
  renderTables();
  renderLayout();
  renderRefineBar();
}

function setShowOriginal(on) {
  state.showOriginal = on;
  renderRefineBar();
  renderLines();
  renderMarkdown();
  renderLayout();
}
$("show-original").addEventListener("click", () => setShowOriginal(!state.showOriginal));

async function textAction(path, confirmText, done) {
  const r = state.current;
  if (!r?.id || (confirmText && !confirm(confirmText))) return;
  setStatus("refine-status", "Working… (the document is re-embedded for questions)");
  try {
    const result = await api(`/api/results/${encodeURIComponent(r.id)}/${path}`, { method: "POST" });
    state.showOriginal = false;
    replaceCurrent(result);
    setStatus("refine-status", done, "ok");
    loadChanges(r.id);
    if (!$("panel-changes").hidden) loadVersions();
    loadHistory();
  } catch (err) {
    setStatus("refine-status", err.message, "error");
  }
}
$("revert-btn").addEventListener("click", () => textAction("revert",
  "Use the original extraction instead of the LLM-refined text? Questions will be answered from the original; you can switch back.",
  "Reverted to the original extraction."));
$("reapply-btn").addEventListener("click", () => textAction("reapply", null, "Using the LLM-refined text again."));

async function retryRefine() {
  const r = state.current;
  if (!r?.id) return;
  setStatus("refine-status", "Starting…");
  try {
    const job = await api(`/api/results/${encodeURIComponent(r.id)}/refine`, { method: "POST" });
    await pollRefine(job.job_id, r.id);
  } catch (err) {
    setStatus("refine-status", err.message, "error");
    renderRefineBar();
  }
}
$("retry-btn").addEventListener("click", retryRefine);

async function pollRefine(jobId, runId) {
  state.refineJob = jobId;
  renderRefineBar();
  const bar = $("refine-progress");
  bar.hidden = false;
  let failures = 0, last = "";
  try {
    for (;;) {
      let j;
      try {
        j = await api(`/api/results/${encodeURIComponent(jobId)}`);
        failures = 0;
      } catch (err) {
        if (err.status === 404 || ++failures >= MAX_POLL_FAILURES) throw err;
        await sleep(POLL_MS);
        continue;
      }
      bar.value = j.progress || 0;
      if (j.message && j.message !== last) { setStatus("refine-status", j.message); last = j.message; }
      if (j.status === "done" || j.status === "error") {
        state.refineJob = null;
        const fresh = await api(`/api/results/${encodeURIComponent(runId)}`).catch(() => null);
        if (fresh && state.current?.id === runId) replaceCurrent(fresh);
        if (j.status === "error") setStatus("refine-status", j.error || "Refinement failed.", "error");
        else setStatus("refine-status", "Ready: the refined text is used for questions and exports.", "ok");
        if (state.current?.id === runId) loadChanges(runId);
        loadHistory();
        break;
      }
      await sleep(POLL_MS);
    }
  } catch (err) {
    setStatus("refine-status", err.message, "error");
  } finally {
    state.refineJob = null;
    bar.hidden = true;
    renderRefineBar();
  }
}

// --- Changes tab: what Gemini changed, line by line (read-only) ------------------------------

async function loadChanges(runId) {
  try {
    state.changes = await api(`/api/results/${encodeURIComponent(runId)}/changes`);
  } catch {
    state.changes = null;  // 404: not refined
  }
  if (state.current?.id === runId) renderChanges();
}

function renderChanges() {
  const view = $("changes-view"), badge = $("changes-badge");
  view.replaceChildren();
  const r = state.current, v = state.changes;
  const lines = v?.data?.lines || [];
  const changed = lines.filter((l) => l.status !== "unchanged");
  badge.hidden = !changed.length;
  badge.textContent = String(changed.filter((l) => l.status !== "rejected").length);
  if (!r?.id) { view.append(el("p", { class: "empty", text: "Run a document first." })); return; }
  if (!v) {
    view.append(el("p", { class: "hint", text: r.refinement?.message || "This document has not been refined by the LLM." }));
    return;
  }
  const counts = v.data.counts || {};
  view.append(el("div", { class: "review-head" },
    el("h4", { text: `What ${v.data.model || "Gemini"} changed` }),
    el("ul", { class: "chips" }, ...["corrected", "inferred", "rejected", "unchanged"].filter((k) => counts[k])
      .map((k) => el("li", { class: k === "rejected" ? "warn" : "" }, `${STATUS_LABEL[k]} `, el("b", { text: String(counts[k]) })))),
    el("p", { class: "hint legend-words" }, "Applied automatically. ", el("span", { class: "w-corr", text: "corrected word" }), " ",
      el("span", { class: "w-inf", text: "inferred word" }), " ", el("span", { class: "w-old", text: "replaced or removed" }),
      ` · a line changing more than ${pct(v.data.max_change ?? 0.4)} of its characters keeps its recognised text and is flagged`),
    r.refinement?.status === "reverted" ? el("p", { class: "status", text: "These changes are not in use: the document was reverted to its original text." }) : null));
  if (!changed.length) { view.append(el("p", { class: "empty", text: "The LLM found nothing to correct." })); return; }
  const ol = el("ol", { class: "review-list" });
  for (const l of changed) ol.append(changeRow(l));
  view.append(ol);
  const same = lines.length - changed.length;
  if (same) view.append(el("p", { class: "hint", text: `${same} line${same === 1 ? "" : "s"} unchanged.` }));
}

function changeRow(l) {
  const m = /^p(\d+)-l(\d+)$/.exec(l.line_id);
  const where = m ? `p. ${Number(m[1]) + 1}, line ${Number(m[2]) + 1}` : l.line_id;
  const li = el("li", { class: `rv ${l.status}`, "data-id": l.line_id },
    el("div", { class: "rv-top" },
      el("button", { type: "button", class: "link-btn rv-where", title: "Show this line on the page", text: where }),
      el("span", { class: `flag st-${l.status}`, text: STATUS_LABEL[l.status] || l.status }),
      el("span", { class: "hint", text: `${pct(l.change_ratio)} of the characters changed${l.confidence != null ? ` · confidence ${pct(l.confidence)}` : ""}` })),
    el("div", { class: "rv-cols" },
      el("div", { class: "rv-col" }, el("span", { class: "sub-label", text: "Original" }), el("p", {}, diffNodes(l.diff, "original"))),
      el("div", { class: "rv-col" }, el("span", { class: "sub-label", text: l.status === "rejected" ? "LLM suggested (not used)" : "Refined" }), el("p", {}, diffNodes(l.diff, "refined")))));
  if (l.reason) li.append(el("p", { class: "rv-reason", text: l.reason }));
  if (l.decision === "conflict") li.append(el("p", { class: "rv-reason", text: "Not applied: the line had changed." }));
  li.querySelector(".rv-where").addEventListener("click", () => showLine(l.line_id));
  li.addEventListener("mouseenter", () => highlight(l.line_id, "changes"));
  li.addEventListener("mouseleave", () => highlight(null, "changes"));
  return li;
}

/** Go to a line's page and select it (ids are "p<page>-l<line>"). */
function showLine(id) {
  const m = /^p(\d+)-l\d+$/.exec(id);
  if (m && Number(m[1]) !== state.page) goPage(Number(m[1]) - state.page);
  selectLine(id);
  ensureBoxVisible(id);
}

// --- versions -------------------------------------------------------------------------------

async function loadVersions() {
  const ol = $("versions"), r = state.current;
  if (!r?.id) { ol.replaceChildren(); return; }
  try {
    const rows = await api(`/api/results/${encodeURIComponent(r.id)}/versions`);
    ol.replaceChildren();
    if (!rows.length) { ol.append(el("li", { class: "empty", text: "No versions yet." })); return; }
    for (const v of [...rows].reverse()) {
      const when = new Date(v.created_at);
      const li = el("li", { class: `ver ${v.kind}` },
        el("span", { class: "ver-n", text: `v${v.seq}` }),
        el("span", { class: "ver-label", text: v.label }),
        el("span", { class: "hint", text: Number.isNaN(when.getTime()) ? "" : when.toLocaleString([], { dateStyle: "short", timeStyle: "short" }) }));
      if (v.kind !== "refinement") {
        const cmp = el("button", { type: "button", class: "link-btn", text: "Compare" });
        const back = el("button", { type: "button", class: "link-btn", text: "Restore" });
        cmp.addEventListener("click", () => compareVersion(v));
        back.addEventListener("click", () => restoreVersion(v));
        li.append(cmp, back);
      }
      ol.append(li);
    }
  } catch (err) {
    ol.replaceChildren(el("li", { class: "empty", text: err.message }));
  }
}

async function compareVersion(v) {
  const box = $("version-compare"), r = state.current;
  box.replaceChildren(el("p", { class: "hint", text: "Comparing…" }));
  try {
    const res = await api(`/api/results/${encodeURIComponent(r.id)}/versions/${encodeURIComponent(v.id)}/compare`);
    box.replaceChildren(el("h4", { text: `v${v.seq} → current: ${res.changes.length} changed line${res.changes.length === 1 ? "" : "s"}` }));
    const ol = el("ol", { class: "review-list" });
    for (const c of res.changes) {
      ol.append(el("li", { class: `rv ${c.status}`, "data-id": c.line_id },
        el("div", { class: "rv-top" }, el("span", { class: "rv-where", text: c.line_id }), el("span", { class: "flag", text: c.status })),
        el("div", { class: "rv-cols" },
          el("div", { class: "rv-col" }, el("span", { class: "sub-label", text: `v${v.seq}` }), el("p", {}, c.diff ? diffNodes(c.diff, "original") : c.before ?? "—")),
          el("div", { class: "rv-col" }, el("span", { class: "sub-label", text: "Current" }), el("p", {}, c.diff ? diffNodes(c.diff, "refined") : c.after ?? "—")))));
    }
    box.append(ol);
  } catch (err) {
    box.replaceChildren(el("p", { class: "status error", text: err.message }));
  }
}

async function restoreVersion(v) {
  await textAction(`versions/${encodeURIComponent(v.id)}/restore`,
    `Make version ${v.seq} ("${v.label}") the current text? The current text stays in the history.`,
    `Restored version ${v.seq}.`);
}
$("versions-refresh").addEventListener("click", loadVersions);

// --- Layout tab: the page rebuilt at its real geometry (static/layout.js) ---------------------

function renderLayout() {
  const view = $("layout-view");
  if ($("panel-layout").hidden) return;  // rendered when the tab is opened (it needs its width)
  const r = state.current;
  if (!r) { view.replaceChildren(el("p", { class: "empty", text: "Run a document to see its layout here." })); return; }
  const page = currentPage();
  const scale = DocLayout.render(view, r, state.page, {
    mode: state.showOriginal ? "original" : "refined",
    scale: state.layoutScale,
    onHover: (id) => highlight(id, "layout"),
    onSelect: (id) => selectLine(id),
  });
  const geometry = DocLayout.hasGeometry(page);
  $("layout-zoom").textContent = geometry ? `${Math.round(scale * 100)}%` : "–";
  for (const id of ["layout-in", "layout-out", "layout-fit"]) $(id).disabled = !geometry;
  $("layout-note").textContent = geometry
    ? `Page ${state.page + 1}${r.pages.length > 1 ? ` of ${r.pages.length}` : ""} rebuilt from its layout${state.showOriginal ? " (original text)" : ""}. Hover a line to find it on the page.`
    : "This file has no page geometry: its structure (headings, lists, tables) is shown instead.";
  state.layoutLastScale = scale;
}
function zoomLayout(factor) {
  const base = typeof state.layoutScale === "number" ? state.layoutScale : state.layoutLastScale || 1;
  state.layoutScale = clamp(base * factor, 0.2, 4);
  renderLayout();
}
$("layout-in").addEventListener("click", () => zoomLayout(1.25));
$("layout-out").addEventListener("click", () => zoomLayout(1 / 1.25));
$("layout-fit").addEventListener("click", () => { state.layoutScale = "fit"; renderLayout(); });
// "Fit" follows the panel's width (window resized, layout changed between one and two columns).
let layoutWidth = 0;
new ResizeObserver(([entry]) => {
  const w = Math.round(entry.contentRect.width);
  if (w === layoutWidth || w < 120) return;
  layoutWidth = w;
  if (state.layoutScale === "fit") requestAnimationFrame(renderLayout);
}).observe($("layout-view"));

// ---------------------------------------------------------------------------------------------
// Tabs (WAI-ARIA tabs pattern: arrow keys move between tabs)
// ---------------------------------------------------------------------------------------------

const tabs = [...document.querySelectorAll('#tablist [role="tab"]')];
function activateTab(id, focus = true) {
  for (const t of tabs) {
    const on = t.id === id;
    t.setAttribute("aria-selected", String(on));
    t.tabIndex = on ? 0 : -1;
    $(t.getAttribute("aria-controls")).hidden = !on;
    if (on && focus) t.focus();
  }
  if (id === "tab-calibration" && !state.calibrationLoaded) loadCalibration();
  if (id === "tab-plugins" && !state.plugins) loadPlugins();
  if (id === "tab-changes" && state.current?.id) { renderChanges(); loadVersions(); }
  if (id === "tab-layout") renderLayout();
}
for (const t of tabs) {
  t.addEventListener("click", () => activateTab(t.id));
  t.addEventListener("keydown", (e) => {
    const i = tabs.indexOf(t);
    const next = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: tabs.length - 1 }[e.key];
    if (next == null) return;
    e.preventDefault();
    activateTab(tabs[(next + tabs.length) % tabs.length].id);
  });
}

// ---------------------------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------------------------

setThreshold(state.threshold, false);
renderQueue();
renderGlobal();
renderPage(false);
renderLines();
renderMarkdown();
renderJson();
renderTables();
loadHealth();
loadHistory();
openFromUrl();
