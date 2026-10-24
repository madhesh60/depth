/* DEPTH studio — For judges: the evaluation guide (JUDGES.md, served by GET /api/judges) rendered beside live
   facts from this server (GET /api/health), with shortcuts into the demonstration.
   Zero-build: index.html loads webui/js/*.js in order as classic scripts sharing one global scope;
   the boot runs on DOMContentLoaded (core.js), after every file has loaded. API: src/dashboard/routes/. */
"use strict";

/* ------------------------------------------------------------------ FOR JUDGES */
const Judges = {
  loaded: false,
  async load() {
    if (!this.loaded) this.wire();
    let g, h;
    try {
      [g, h] = await Promise.all([fetch(`${API}/api/judges`).then(r => r.json()), fetch(`${API}/api/health`).then(r => r.json())]);
    } catch (e) {
      $("#jgDoc").innerHTML = `<p class="empty-hint">The guide could not be loaded (${escapeHtml(e.message)}). It is also in the repository: JUDGES.md.</p>`;
      return;
    }
    $("#jgFacts").innerHTML = this.facts(g.server || {}, h || {}, g.repo || "");
    if (g.repo) $("#jgRepo").href = g.repo;
    const doc = $("#jgDoc");
    doc.innerHTML = Markdown.render(g.markdown || "", g.docs_base || "");
    $("#jgToc").innerHTML = $$("h2[id]", doc).filter(x => x.id !== "contents")
      .map(x => `<a href="#${x.id}" data-anchor="${x.id}">${escapeHtml(x.textContent)}</a>`).join("");
    this.loaded = true;
  },
  facts(s, h, repo) {
    const pct = x => x == null ? "—" : `${Math.round(x * 100)}%`;
    const modes = (h.modes || []).map(m => `≥ ${pct(m.promise)} ${escapeHtml(m.label)}`).join(" · ");
    const rt = h.runtime || {}, host = s.host || {};
    const cool = s.is_cool_path || h.is_cool_path;
    const chip = (k, v, cls = "") => `<div class="jg-fact ${cls}"><span>${k}</span><b>${v}</b></div>`;
    return [
      chip("Model", escapeHtml(h.model || s.model || "—") + (h.model_loaded ? "" : " · warming up")),
      chip("Recall promise (95%)", modes || "—"),
      chip("OpenCV", `${escapeHtml(h.opencv || s.opencv || "—")} · ${cool ? "COOL" : "stock (not COOL)"}`, cool ? "is-ok" : ""),
      chip("DNN engine", escapeHtml(rt.engine || "loads with the model")),
      chip("Host", escapeHtml([host.instance_type, host.machine, host.vcpus ? `${host.vcpus} vCPU` : ""].filter(Boolean).join(" · ") || "—")),
      chip("Code", s.code_commit && repo && /^[0-9a-f]{7,40}$/.test(s.code_commit)
        ? `<a href="${escapeHtml(repo)}/commit/${s.code_commit}" target="_blank" rel="noopener">${s.code_commit} ↗</a>`
        : escapeHtml(s.code_commit || "—")),
      chip("Approvals", h.approver_pin_required ? "a person, with the approver PIN" : "a named person (no PIN on this server)"),
    ].join("");
  },
  wire() {
    $("#jgToc").addEventListener("click", e => this.jump(e));
    $("#jgDoc").addEventListener("click", e => this.jump(e));
    $$("[data-jg]").forEach(b => b.onclick = () => this.go(b.dataset.jg));
  },
  jump(e) {                                    // in-page anchors scroll the guide, not the window
    const a = e.target.closest("a[data-anchor]"); if (!a) return;
    const t = document.getElementById(a.dataset.anchor); if (!t) return;
    e.preventDefault();
    const box = $("#judgesPage");
    box.scrollTo({ top: t.getBoundingClientRect().top - box.getBoundingClientRect().top + box.scrollTop - 16, behavior: "smooth" });
  },
  async go(what) {
    if (what === "demo") { if (!Demo.on) Demo.run(); return; }
    if (what === "connect") { setMode("connect"); return; }
    if (what === "analyze") {
      setMode("analyze");
      const card = $(".sample-card"); if (!card) return;
      card.click(); await runAnalyze(); return;
    }
    if (what === "recording") {
      setMode("survey");
      const opt = $("#gpsRecording"), sel = $("#gpsMode");
      sel.value = opt && !opt.hidden ? "recording" : "synthetic";
      await runSurvey(); Tabs.go("Agent");
    }
  },
};

/* A small Markdown renderer for the guide: headings, paragraphs, lists, tables, fenced code, inline code, bold,
   italics and links. Every piece of text is escaped first, HTML comments are dropped, and a link is kept only if it
   is http(s), an in-page anchor, or a path relative to the guide (resolved against `base`, the docs on GitHub). */
const Markdown = {
  slug: s => s.trim().toLowerCase().replace(/[^\w\- ]/g, "").replace(/ /g, "-"),
  href(url, base) {
    if (url.startsWith("#")) return { anchor: url.slice(1) };
    if (/^https?:\/\//i.test(url)) return { url };
    if (/^[a-z][a-z0-9+.-]*:/i.test(url)) return null;          // javascript:, data: ... never
    try { return { url: new URL(url, base || location.href).href }; } catch { return null; }
  },
  inline(text, base) {
    const code = [];
    let s = String(text).replace(/`([^`]+)`/g, (m, c) => `\u0000${code.push(c) - 1}\u0000`);
    s = escapeHtml(s);
    s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (m, label, url) => {
      const h = this.href(url.replace(/&amp;/g, "&"), base);
      if (!h) return label;
      if (h.anchor != null) return `<a href="#${escapeHtml(h.anchor)}" data-anchor="${escapeHtml(this.slug(h.anchor))}">${label}</a>`;
      return `<a href="${escapeHtml(h.url)}" target="_blank" rel="noopener">${label}</a>`;
    });
    s = s.replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>").replace(/(^|[^\w*])\*([^*\s][^*]*?)\*(?![\w*])/g, "$1<i>$2</i>");
    return s.replace(/\u0000(\d+)\u0000/g, (m, i) => `<code>${escapeHtml(code[+i])}</code>`);
  },
  cells: row => row.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(c => c.trim()),
  render(md, base) {
    const lines = md.replace(/<!--[\s\S]*?-->/g, "").replace(/\r/g, "").split("\n");
    const out = [];
    let i = 0;
    const para = [];
    const flush = () => { if (para.length) { out.push(`<p>${this.inline(para.join(" "), base)}</p>`); para.length = 0; } };
    while (i < lines.length) {
      const line = lines[i];
      if (/^```/.test(line)) {                                   // fenced code
        flush();
        const body = [];
        for (i++; i < lines.length && !/^```/.test(lines[i]); i++) body.push(lines[i]);
        out.push(`<pre><code>${escapeHtml(body.join("\n"))}</code></pre>`); i++; continue;
      }
      const hd = /^(#{1,6})\s+(.*)$/.exec(line);
      if (hd) {
        flush();
        const n = hd[1].length;                                    // the guide's sections (h2) feed the side contents
        out.push(`<h${n} id="${escapeHtml(this.slug(hd[2]))}"${n === 1 ? ' class="jg-title"' : ""}>${this.inline(hd[2], base)}</h${n}>`);
        i++; continue;
      }
      if (/^\s*\|/.test(line) && /^\s*\|?\s*:?-{3,}/.test(lines[i + 1] || "")) {   // a table
        flush();
        const head = this.cells(line);
        const rows = [];
        for (i += 2; i < lines.length && /^\s*\|/.test(lines[i]); i++) rows.push(this.cells(lines[i]));
        out.push(`<div class="jg-table"><table><thead><tr>${head.map(c => `<th>${this.inline(c, base)}</th>`).join("")}</tr></thead>`
          + `<tbody>${rows.map(r => `<tr>${r.map(c => `<td>${this.inline(c, base)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`);
        continue;
      }
      const li = /^(\s*)([-*]|\d+\.)\s+(.*)$/.exec(line);
      if (li && !li[1]) {                                          // a list; indented lines continue an item
        flush();
        const ordered = /\d/.test(li[2]);
        const items = [];
        while (i < lines.length) {
          const m = /^([-*]|\d+\.)\s+(.*)$/.exec(lines[i]);
          if (m && (/\d/.test(m[1]) === ordered)) { items.push(m[2]); i++; continue; }
          if (/^\s{2,}\S/.test(lines[i]) && items.length) { items[items.length - 1] += " " + lines[i].trim(); i++; continue; }
          break;
        }
        const tag = ordered ? "ol" : "ul";
        out.push(`<${tag}>${items.map(x => `<li>${this.inline(x, base)}</li>`).join("")}</${tag}>`);
        continue;
      }
      if (!line.trim()) { flush(); i++; continue; }
      para.push(line.trim()); i++;
    }
    flush();
    return out.join("\n");
  },
};
