/* DocBlendAI shared page rendering, used by the QA page (/) and the Experience Center (/studio).
 *
 * - layout view: a page rebuilt at its real geometry from an Experience Center result
 *   (docs/experience_center_contract.md §4): lines and words at their boxes (word gaps,
 *   indentation and column spacing as on the page), tables as real HTML tables (merged cells
 *   kept) at their block positions, figures and formulas at theirs; zoomable, with hover / click
 *   callbacks so the page image and the layout highlight together. Pages without geometry
 *   (Word, PowerPoint, Excel, text) are shown as structured Markdown and tables, .txt files
 *   with their own spacing.
 * - LLM refinement marks: corrected words highlighted, inferred words marked differently,
 *   the recognised text on hover; "original" mode shows the text as recognised.
 * - a small Markdown renderer and a table sanitiser (shared so both pages render the same way).
 *
 * Safety: text is always inserted as text; HTML comes only from markdownToHtml (escapes first)
 * and sanitizeTable (rebuilt from an allow-list: table elements, numeric colspan/rowspan).
 */
(function () {
  "use strict";

  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
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

  // ------------------------------------------------------------------------------------------
  // Markdown (escapes everything first) and table sanitising
  // ------------------------------------------------------------------------------------------

  function renderInline(s) {
    const codes = [];
    let out = esc(s).replace(/`([^`]+)`/g, (_, c) => { codes.push(c); return `\u0001${codes.length - 1}\u0001`; });
    out = out
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/__(.+?)__/g, "<strong>$1</strong>")
      .replace(/(^|[^\w*])\*(\S(?:.*?\S)?)\*(?=[^\w*]|$)/g, "$1<em>$2</em>")
      .replace(/(^|[^\w])_(\S(?:.*?\S)?)_(?=[^\w]|$)/g, "$1<em>$2</em>")
      .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    return out.replace(/\u0001(\d+)\u0001/g, (_, i) => `<code>${codes[Number(i)]}</code>`);
  }

  function markdownToHtml(md) {
    const blocks = [];  // pre-rendered safe HTML chunks, referenced by placeholder lines
    const hold = (html) => `\n\u0000${blocks.push(html) - 1}\u0000\n`;
    let src = String(md || "").replace(/\r\n?/g, "\n");
    src = src.replace(/<table[\s\S]*?<\/table>/gi, (m) => hold(sanitizeTable(m).outerHTML));
    src = src.replace(/<img\b[^>]*>/gi, () => hold('<span class="figure-note">[figure]</span>'));
    src = src.replace(/<\/?(?:div|html|body|center|span|p|br)\b[^>]*>/gi, "\n");  // layout wrappers only

    const lines = src.split("\n");
    let html = "", para = [], list = null;
    const flushPara = () => { if (para.length) { html += `<p>${renderInline(para.join(" "))}</p>`; para = []; } };
    const closeList = () => { if (list) { html += `</${list}>`; list = null; } };
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i], t = line.trim();
      const held = /^\u0000(\d+)\u0000$/.exec(t);
      if (held) { flushPara(); closeList(); html += blocks[Number(held[1])]; continue; }
      if (/^```/.test(t)) {  // fenced code
        flushPara(); closeList();
        const code = [];
        while (++i < lines.length && !/^```/.test(lines[i].trim())) code.push(lines[i]);
        html += `<pre><code>${esc(code.join("\n"))}</code></pre>`;
        continue;
      }
      if (t.startsWith("$$")) {  // display formula: show its LaTeX source
        flushPara(); closeList();
        const f = [t];
        const closedOnSameLine = t.length > 2 && t.endsWith("$$");
        if (!closedOnSameLine) while (++i < lines.length) { f.push(lines[i]); if (lines[i].trim().endsWith("$$")) break; }
        html += `<pre><code>${esc(f.join("\n").replace(/\$\$/g, "").trim())}</code></pre>`;
        continue;
      }
      if (!t) { flushPara(); closeList(); continue; }
      const h = /^(#{1,6})\s+(.*)$/.exec(t);
      if (h) { flushPara(); closeList(); const lv = Math.min(4, h[1].length); html += `<h${lv}>${renderInline(h[2])}</h${lv}>`; continue; }
      if (/^(-{3,}|\*{3,}|_{3,})$/.test(t)) { flushPara(); closeList(); html += "<hr>"; continue; }
      if (t.startsWith(">")) { flushPara(); closeList(); html += `<blockquote>${renderInline(t.replace(/^>\s?/, ""))}</blockquote>`; continue; }
      // pipe table: header row, then a |---|---| separator
      if (t.includes("|") && i + 1 < lines.length && /^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$/.test(lines[i + 1].trim())) {
        flushPara(); closeList();
        const cells = (row) => row.trim().replace(/^\||\|$/g, "").split("|").map((c) => renderInline(c.trim()));
        let t2 = `<div class="table-wrap"><table><thead><tr>${cells(t).map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>`;
        i++;
        while (i + 1 < lines.length && lines[i + 1].includes("|") && lines[i + 1].trim()) {
          t2 += `<tr>${cells(lines[++i]).map((c) => `<td>${c}</td>`).join("")}</tr>`;
        }
        html += `${t2}</tbody></table></div>`;
        continue;
      }
      const bullet = /^[-*+•]\s+(.*)$/.exec(t), num = /^\d+[.)]\s+(.*)$/.exec(t);
      if (bullet || num) {
        flushPara();
        const kind = bullet ? "ul" : "ol";
        if (list !== kind) { closeList(); html += `<${kind}>`; list = kind; }
        html += `<li>${renderInline((bullet || num)[1])}</li>`;
        continue;
      }
      closeList();
      para.push(t);
    }
    flushPara();
    closeList();
    return html;
  }

  /** Rebuild table HTML from an allow-list. Returns a <div class="table-wrap"> holding the table(s). */
  function sanitizeTable(html) {
    const ALLOWED = new Set(["TABLE", "THEAD", "TBODY", "TFOOT", "TR", "TH", "TD"]);
    const DROP = new Set(["SCRIPT", "STYLE", "TEMPLATE", "IFRAME", "OBJECT", "EMBED", "NOSCRIPT", "SVG", "MATH"]);
    const doc = new DOMParser().parseFromString(String(html), "text/html");  // inert: nothing runs or loads
    const wrap = el("div", { class: "table-wrap" });
    const copy = (from, to) => {
      for (const n of from.childNodes) {
        if (n.nodeType === Node.TEXT_NODE) { to.append(n.textContent); continue; }
        if (n.nodeType !== Node.ELEMENT_NODE || DROP.has(n.tagName)) continue;
        if (ALLOWED.has(n.tagName)) {
          const e = document.createElement(n.tagName.toLowerCase());
          for (const a of ["colspan", "rowspan"]) {
            const v = n.getAttribute(a);
            if (v && /^\d{1,3}$/.test(v)) e.setAttribute(a, v);
          }
          copy(n, e);
          to.append(e);
        } else {
          copy(n, to);  // unknown wrapper (b, span, div…): keep its text, drop the tag
        }
      }
    };
    copy(doc.body, wrap);
    return wrap;
  }

  // ------------------------------------------------------------------------------------------
  // LLM refinement marks: corrected words highlighted, inferred words marked, original on hover
  // ------------------------------------------------------------------------------------------

  const isLlmLine = (line) => !!line && (line.source === "llm" || line.source === "llm-auto");
  /** The text as recognised: an LLM-refined line keeps it in ocr_text. */
  const originalText = (line) => (isLlmLine(line) && line.ocr_text != null ? line.ocr_text : line.text);

  /** Words of a word diff for one side: "original" (replaced / removed words struck through) or
   *  "refined" (corrected words highlighted, inferred words marked; hover shows the original). */
  function diffNodes(ops, side) {
    const span = el("span", { class: "diff" });
    for (const op of ops || []) {
      let node;
      if (side === "original") {
        if (op.op === "insert") continue;
        const cls = op.op === "replace" ? "w-old" : op.op === "delete" ? "w-del" : null;
        node = cls ? el("span", { class: cls, title: op.op === "replace" ? `corrected by the LLM to "${op.refined}"` : "removed by the LLM", text: op.original })
          : document.createTextNode(op.original);
      } else {
        if (op.op === "delete") continue;
        const cls = op.op === "replace" ? "w-corr" : op.op === "insert" ? "w-inf" : null;
        node = cls ? el("span", { class: cls,
          title: op.op === "replace" ? `Corrected by the LLM. Original: "${op.original}"` : "Inferred by the LLM from context (not on the page as read)",
          text: op.refined })
          : document.createTextNode(op.refined);
      }
      if (span.childNodes.length) span.append(" ");
      span.append(node);
    }
    return span;
  }

  /** A line's content for a view mode: "refined" (LLM marks) or "original" (as recognised). */
  function lineContent(line, mode) {
    if (mode === "original") return document.createTextNode(originalText(line) || "");
    if (isLlmLine(line) && line.llm_diff) {
      const d = diffNodes(line.llm_diff, "refined");
      d.title = `Original: ${line.ocr_text ?? ""}`;
      return d;
    }
    return document.createTextNode(line.text || "");
  }

  /** Plain text with the refined lines it contains marked (QA source passages, Markdown views). */
  function highlightText(text, lines) {
    const frag = document.createDocumentFragment();
    const hits = [];
    for (const line of lines || []) {
      if (!isLlmLine(line) || !line.llm_diff || !line.text) continue;
      let from = 0, at;
      while ((at = text.indexOf(line.text, from)) !== -1) {
        if (!hits.some((h) => at < h.end && at + line.text.length > h.start)) hits.push({ start: at, end: at + line.text.length, line });
        from = at + line.text.length;
      }
    }
    hits.sort((a, b) => a.start - b.start);
    let pos = 0;
    for (const h of hits) {
      if (h.start > pos) frag.append(text.slice(pos, h.start));
      frag.append(lineContent(h.line, "refined"));
      pos = h.end;
    }
    if (pos < text.length) frag.append(text.slice(pos));
    return frag;
  }

  /** Mark refined lines inside already-rendered HTML (text nodes only; markup is untouched). */
  function highlightIn(root, lines) {
    const wanted = (lines || []).filter((l) => isLlmLine(l) && l.llm_diff && l.text);
    if (!wanted.length) return;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    for (const node of nodes) {
      if (!wanted.some((l) => node.textContent.includes(l.text))) continue;
      node.replaceWith(highlightText(node.textContent, wanted));
    }
  }

  /** Markdown with the LLM's changes undone (refined line text -> recognised text). */
  function originalMarkdown(md, lines) {
    let out = String(md || "");
    for (const l of lines || []) if (isLlmLine(l) && l.ocr_text != null && l.text) out = out.replace(l.text, l.ocr_text);
    return out;
  }

  // ------------------------------------------------------------------------------------------
  // Layout view: the page rebuilt at its real geometry
  // ------------------------------------------------------------------------------------------

  const TITLE_BLOCKS = new Set(["doc_title", "paragraph_title", "title", "figure_title", "table_title", "chart_title"]);
  const FIGURE_BLOCKS = new Set(["image", "figure", "chart", "seal"]);
  const FORMULA_BLOCKS = new Set(["formula", "equation", "display_formula"]);

  const centerIn = (line, box) => {
    if (!box || !line.box) return false;
    const cx = (line.box[0] + line.box[2]) / 2, cy = (line.box[1] + line.box[3]) / 2;
    return cx >= box[0] && cx <= box[2] && cy >= box[1] && cy <= box[3];
  };
  const hasGeometry = (page) => !!(page && page.width && page.height && (page.lines || []).some((l) => l.box));

  /** Pages without geometry (Word, PowerPoint, Excel, text): structured Markdown and tables,
   *  or the file's own spacing for .txt. */
  function renderFlow(container, page, mode) {
    const lines = page.lines || [];
    if (page.preformatted) {
      const text = mode === "original" ? (page.text ?? lines.map(originalText).join("\n")) : lines.map((l) => l.text).join("\n");
      const pre = el("pre", { class: "lt-pre" });
      if (mode === "original") pre.textContent = text;
      else pre.append(highlightText(text, lines));
      container.append(pre);
      return;
    }
    const md = mode === "original" ? originalMarkdown(page.markdown, lines) : page.markdown;
    const div = el("div", { class: "lt-flow md" });
    div.innerHTML = md ? markdownToHtml(md) : "";  // safe: markdownToHtml escapes all text itself
    if (!md) div.append(el("p", { class: "empty", text: lines.map((l) => (mode === "original" ? originalText(l) : l.text)).join("\n") || "No text on this page." }));
    if (mode !== "original") highlightIn(div, lines);
    container.append(div);
  }

  /** Squeeze or stretch each positioned line horizontally to its box width (fonts differ from the page). */
  function fitLines(sheet) {
    for (const node of sheet.querySelectorAll(".lt-line[data-w]")) {
      const target = Number(node.dataset.w), natural = node.scrollWidth;
      if (natural > 0 && target > 0) node.style.transform = `scaleX(${Math.max(0.45, Math.min(1.8, target / natural))})`;
    }
  }

  /**
   * Render page `pageIndex` of `result` into `container`.
   * opts: { mode: "refined" | "original", scale: number (px per page px) or "fit",
   *         onHover(id|null), onSelect(id) }
   * Returns the scale used.
   */
  function render(container, result, pageIndex, opts = {}) {
    const mode = opts.mode || "refined";
    container.replaceChildren();
    const page = result?.pages?.[pageIndex];
    if (!page) { container.append(el("p", { class: "empty", text: "No page to show." })); return 1; }
    if (!hasGeometry(page)) { renderFlow(container, page, mode); return 1; }

    // A hidden or not-yet-laid-out container has no width: assume a typical one (callers re-fit on resize).
    const avail = container.clientWidth > 120 ? container.clientWidth - 24 : 800;
    const scale = opts.scale === "fit" || !opts.scale ? avail / page.width : opts.scale;
    const sheet = el("div", { class: "lt-sheet", role: "document", "aria-label": `Page ${pageIndex + 1}, rebuilt from its layout` });
    sheet.style.width = `${page.width * scale}px`;
    sheet.style.height = `${page.height * scale}px`;
    const place = (node, box) => {
      node.style.left = `${box[0] * scale}px`;
      node.style.top = `${box[1] * scale}px`;
      node.style.width = `${(box[2] - box[0]) * scale}px`;
      node.style.height = `${(box[3] - box[1]) * scale}px`;
    };

    const blocks = page.blocks || [];
    const tableBoxes = [];
    for (const table of page.tables || []) {
      if (!table.box || !(table.html || (table.rows || []).length)) continue;
      const wrap = el("div", { class: "lt-table" });
      const rows = Math.max(1, (table.rows || []).length);
      wrap.style.fontSize = `${Math.max(5, Math.min(40, ((table.box[3] - table.box[1]) / rows) * 0.42 * scale))}px`;
      let source = table.html;
      if (mode === "original" && source) source = originalMarkdown(source, page.lines);
      const tableNode = source ? sanitizeTable(source) : null;
      if (tableNode && tableNode.querySelector("table")) wrap.append(tableNode);
      else {
        const t = el("table");
        for (const row of table.rows || []) t.append(el("tr", {}, ...row.map((c) => el("td", { text: c }))));
        wrap.append(t);
      }
      if (mode !== "original") highlightIn(wrap, page.lines);
      place(wrap, table.box);
      sheet.append(wrap);
      tableBoxes.push(table.box);
    }
    const formulaBoxes = [];
    for (const block of blocks) {
      if (!block.box) continue;
      if (FIGURE_BLOCKS.has(block.type)) {
        const fig = el("div", { class: "lt-figure", title: block.type }, el("span", { text: block.type === "chart" ? "Chart" : "Figure" }));
        place(fig, block.box);
        sheet.append(fig);
      } else if (FORMULA_BLOCKS.has(block.type) && block.content) {
        const f = el("div", { class: "lt-formula", title: "Formula (LaTeX)", text: block.content });
        f.style.fontSize = `${Math.max(6, Math.min(28, (block.box[3] - block.box[1]) * 0.45 * scale))}px`;
        place(f, block.box);
        sheet.append(f);
        formulaBoxes.push(block.box);
      }
    }
    const titleBoxes = blocks.filter((b) => b.box && TITLE_BLOCKS.has(b.type)).map((b) => b.box);

    for (const line of page.lines || []) {
      if (!line.box || !line.text) continue;
      if (tableBoxes.some((b) => centerIn(line, b)) || formulaBoxes.some((b) => centerIn(line, b))) continue;
      const [x0, y0, x1, y1] = line.box;
      const h = (y1 - y0) * scale;
      const node = el("div", { class: `lt-line${titleBoxes.some((b) => centerIn(line, b)) ? " lt-title" : ""}${isLlmLine(line) && mode !== "original" ? " lt-llm" : ""}`,
        "data-id": line.id || "" });
      node.style.left = `${x0 * scale}px`;
      node.style.top = `${y0 * scale}px`;
      node.style.height = `${h}px`;
      node.style.fontSize = `${Math.max(4, h * 0.78)}px`;
      node.style.lineHeight = `${h}px`;
      const words = (line.words || []).filter((w) => w.box && w.text);
      const plain = mode === "original" || !isLlmLine(line);
      if (plain && words.length > 1 && words.map((w) => w.text).join("") === originalText(line).replace(/\s+/g, "")) {
        // Word boxes reproduce word gaps, indentation and column spacing exactly.
        node.style.width = `${(x1 - x0) * scale}px`;
        for (const w of words) {
          const span = el("span", { class: "lt-word", text: w.text });
          span.style.left = `${(w.box[0] - x0) * scale}px`;
          span.dataset.w = String((w.box[2] - w.box[0]) * scale);
          node.append(span);
        }
        node.classList.add("lt-words");
      } else {
        node.dataset.w = String((x1 - x0) * scale);
        node.append(lineContent(line, mode));
      }
      if (opts.onHover) {
        node.addEventListener("mouseenter", () => opts.onHover(line.id));
        node.addEventListener("mouseleave", () => opts.onHover(null));
      }
      if (opts.onSelect) node.addEventListener("click", () => opts.onSelect(line.id));
      sheet.append(node);
    }
    container.append(sheet);
    // Fit after layout: measuring needs the nodes in the document.
    requestAnimationFrame(() => {
      fitLines(sheet);
      for (const w of sheet.querySelectorAll(".lt-word[data-w]")) {
        const natural = w.scrollWidth, target = Number(w.dataset.w);
        if (natural > 0 && target > 0) w.style.transform = `scaleX(${Math.max(0.45, Math.min(1.8, target / natural))})`;
      }
    });
    return scale;
  }

  window.DocLayout = {
    render, hasGeometry, diffNodes, lineContent, highlightText, highlightIn, originalText, originalMarkdown,
    isLlmLine, markdownToHtml, sanitizeTable, renderInline,
  };
})();
