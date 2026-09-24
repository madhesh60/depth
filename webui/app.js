/* DEPTH studio — drives the See → Prove → Decide → Act loop over the FastAPI backend.
   Zero-build vanilla JS. All API shapes match src/dashboard/app.py. */
"use strict";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const API = ""; // same-origin (served by FastAPI)
const SVGNS = "http://www.w3.org/2000/svg";

const VERDICTS = ["confirmed", "review", "rejected"];
const VCOLOR = { confirmed: "#2ea043", review: "#d9a441", rejected: "#5f6f7d" };
const CLASS_SW = ["#1fb6d5", "#d9a441", "#8b7ff0", "#4fb477", "#e5789b"];

const state = {
  samples: [], selected: null, survey: null, map: null, mapLayers: [],
  _analyze: null, _boxes: [], _cursor: -1, gate: 0.10,
  hiddenClasses: new Set(),
  _ready: false,               // an analyze result is on screen (keyboard guard)
  _tour: false, _tourOptOut: false,
};

/* ------------------------------------------------------------------ boot */
window.addEventListener("DOMContentLoaded", async () => {
  wireModes();
  wireViewerControls();
  wireInputs();
  wireKeyboard();
  Viewer.init();
  await Promise.all([loadHealth(), loadSamples()]);
});

async function loadHealth() {
  try {
    const h = await fetch(`${API}/api/health`).then(r => r.json());
    const cv = $("#pillCv"), md = $("#pillModel");
    cv.textContent = `OpenCV ${h.opencv}`;
    cv.className = "pill " + (String(h.opencv).startsWith("5") ? "pill-ok" : "pill");
    md.textContent = h.model_loaded ? "model ready" : "model lazy";
    md.className = "pill " + (h.model_loaded ? "pill-ok" : "pill");
    const rc = $("#regCv"), rm = $("#regModel");
    if (rc) rc.textContent = h.opencv;
    if (rm) rm.textContent = h.model_loaded ? "loaded" : "lazy (loads on first run)";
  } catch (e) {
    $("#pillCv").textContent = "backend offline";
    $("#pillCv").className = "pill pill-bad";
  }
}

async function loadSamples() {
  const grid = $("#sampleGrid");
  try {
    const j = await fetch(`${API}/api/samples`).then(r => r.json());
    state.samples = j.samples || [];
  } catch { state.samples = []; }
  const cnt = $("#sampleCount"); if (cnt) cnt.textContent = state.samples.length ? `${state.samples.length} frames` : "";
  if (!state.samples.length) {
    grid.innerHTML = `<p class="empty-hint" style="grid-column:1/-1">No local samples — drop your own sonar frame below.</p>`;
    return;
  }
  grid.innerHTML = "";
  state.samples.forEach(s => {
    const card = document.createElement("button");
    card.className = "sample-card"; card.dataset.id = s.id;
    const img = document.createElement("img");
    img.alt = s.name; img.loading = "lazy";
    img.src = `${API}/api/sample_thumb/${encodeURIComponent(s.id)}`;
    img.onerror = () => img.replaceWith(sonarGlyph());
    const meta = document.createElement("div");
    meta.className = "sc-meta";
    meta.innerHTML = `<div class="sc-name">${escapeHtml(s.name)}</div><div class="sc-kind">${escapeHtml(s.kind)}</div>`;
    card.append(img, meta);
    card.onclick = () => selectSample(s.id, card);
    grid.appendChild(card);
  });
}

function sonarGlyph() {
  const d = document.createElement("div");
  d.style.cssText = "aspect-ratio:1;display:grid;place-items:center;background:radial-gradient(circle at 50% 35%,#123049,#0a1626);color:#1fb6d5";
  d.innerHTML = `<svg viewBox="0 0 24 24" width="28" height="28" opacity=".7"><path d="M12 3a9 9 0 1 0 9 9h-9V3Z" fill="none" stroke="currentColor" stroke-width="1.4"/><circle cx="12" cy="12" r="2" fill="currentColor"/></svg>`;
  return d;
}

function selectSample(id, card) {
  state.selected = { id };
  $$(".sample-card").forEach(c => c.classList.toggle("is-sel", c === card));
  const s = state.samples.find(x => x.id === id);
  $("#analyzeBtn").disabled = false;
  $("#analyzeHint").textContent = `${s.name} — ${s.kind}. Ready to run.`;
}

/* ------------------------------------------------------------------ modes + inputs */
function wireModes() {
  $$(".mode-btn[data-mode]").forEach(b => b.onclick = () => {
    cancelTour();
    $$(".mode-btn").forEach(x => x.classList.toggle("is-active", x === b));
    $("#workspace").dataset.mode = b.dataset.mode;
    if (b.dataset.mode === "survey" && state.map) setTimeout(() => state.map.invalidateSize(), 80);
  });
}

function wireViewerControls() {
  $$(".seg-btn").forEach(b => b.onclick = () => {
    $$(".seg-btn").forEach(x => x.classList.toggle("is-active", x === b));
    $("#viewer").classList.toggle("boxes-off", b.dataset.view === "frame");
  });
  $("#zoomIn").onclick = () => { cancelTour(); Viewer.zoomBy(1.3); };
  $("#zoomOut").onclick = () => { cancelTour(); Viewer.zoomBy(1 / 1.3); };
  $("#zoomFit").onclick = () => { cancelTour(); Viewer.fit(); };
  $("#tourBtn").onclick = () => (state._tour ? cancelTour() : playGazeTour(true));
  const gate = $("#confGate");
  gate.oninput = () => { state.gate = parseFloat(gate.value); $("#confGateVal").textContent = state.gate.toFixed(2); applyFilters(); };
}

function wireInputs() {
  const dz = $("#dropzone"), fi = $("#fileInput");
  fi.onchange = () => fi.files[0] && pickFile(fi.files[0]);
  ["dragover", "dragenter"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", e => e.dataTransfer.files[0] && pickFile(e.dataTransfer.files[0]));
  $("#analyzeBtn").onclick = runAnalyze;
  $("#surveyBtn").onclick = runSurvey;
}

function wireKeyboard() {
  window.addEventListener("keydown", e => {
    if (!state._ready || $("#workspace").dataset.mode !== "analyze") return;
    const tag = (e.target.tagName || "").toLowerCase();
    if (tag === "input" || tag === "textarea" || tag === "select") return;
    if (e.key === "+" || e.key === "=") { Viewer.zoomBy(1.3); e.preventDefault(); }
    else if (e.key === "-" || e.key === "_") { Viewer.zoomBy(1 / 1.3); e.preventDefault(); }
    else if (e.key === "0") { Viewer.fit(); e.preventDefault(); }
    else if (e.key === "ArrowRight") { cycleCandidate(1); e.preventDefault(); }
    else if (e.key === "ArrowLeft") { cycleCandidate(-1); e.preventDefault(); }
    else if (e.key.toLowerCase() === "g" && state._cursor >= 0) { gazeTo(state._cursor); e.preventDefault(); }
  });
}

function pickFile(file) {
  state.selected = { file };
  $$(".sample-card").forEach(c => c.classList.remove("is-sel"));
  $("#analyzeBtn").disabled = false;
  $("#analyzeHint").textContent = `Uploaded ${file.name}. Ready to run.`;
}

/* ------------------------------------------------------------------ interactive viewer */
const Viewer = (() => {
  let scale = 1, tx = 0, ty = 0, natW = 0, natH = 0, drag = null, wheelClear = null;
  const canvas = () => $("#viewerCanvas");
  const stage = () => $("#viewerStage");
  const svg = () => $("#boxLayer");

  function apply() {
    stage().style.transform = `translate(${tx}px,${ty}px) scale(${scale})`;
    $("#zoomRead").textContent = Math.round(scale * 100) + "%";
    svg().style.setProperty("--inv", (1 / scale).toFixed(4));
  }
  const clamp = s => Math.min(16, Math.max(0.05, s));

  function fit() {
    const c = canvas().getBoundingClientRect();
    if (!natW || !natH || !c.width) return;
    const s = Math.min(c.width / natW, c.height / natH);
    scale = s; tx = (c.width - natW * s) / 2; ty = (c.height - natH * s) / 2;
    stage().style.transition = "transform .35s ease"; apply();
    setTimeout(() => stage().style.transition = "", 360);
  }
  function zoomAtPoint(px, py, factor) {
    const ns = clamp(scale * factor);
    tx = px - (px - tx) * (ns / scale); ty = py - (py - ty) * (ns / scale);
    scale = ns; apply();
  }
  function zoomBy(factor) { const c = canvas().getBoundingClientRect(); zoomAtPoint(c.width / 2, c.height / 2, factor); }
  function focusBox(bbox, opts = {}) {
    const [x1, y1, x2, y2] = bbox;
    const bw = Math.max(4, x2 - x1), bh = Math.max(4, y2 - y1), cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
    const c = canvas().getBoundingClientRect();
    let target = Math.min(c.width / (bw * 3.0), c.height / (bh * 3.0));
    if (opts.scale) target = Math.max(target, opts.scale * 1.4);
    scale = clamp(target); tx = c.width / 2 - cx * scale; ty = c.height / 2 - cy * scale;
    stage().style.transition = "transform .95s cubic-bezier(.34,.08,.18,1)"; apply();
    setTimeout(() => stage().style.transition = "", 980);
  }
  function render(d, boxes) {
    natW = d.width; natH = d.height;
    const im = $("#stageImg"), s = svg();
    s.setAttribute("viewBox", `0 0 ${natW} ${natH}`);
    s.style.width = natW + "px"; s.style.height = natH + "px";
    stage().style.width = natW + "px"; stage().style.height = natH + "px";
    drawBoxes(boxes);
    im.onload = () => fit();
    im.src = d.frame_png;
    if (im.complete && im.naturalWidth) fit();
  }
  function drawBoxes(boxes) {
    const s = svg(); s.innerHTML = "";
    const proof = document.createElementNS(SVGNS, "g"); proof.id = "proofLayer"; s.appendChild(proof);
    boxes.forEach(b => {
      const [x1, y1, x2, y2] = b.bbox;
      const g = document.createElementNS(SVGNS, "g");
      g.setAttribute("class", `bx bx-${b.verdict}`); g.dataset.id = b.id;
      const r = document.createElementNS(SVGNS, "rect");
      r.setAttribute("x", x1); r.setAttribute("y", y1);
      r.setAttribute("width", x2 - x1); r.setAttribute("height", y2 - y1);
      g.appendChild(r);
      const t = document.createElementNS(SVGNS, "text");
      t.setAttribute("x", x1 + 2); t.setAttribute("y", y1 - 3);
      t.setAttribute("class", "bx-label"); t.textContent = b.conf.toFixed(2);
      g.appendChild(t);
      g.addEventListener("click", ev => { ev.stopPropagation(); selectCandidate(b.id); });
      s.appendChild(g);
    });
  }
  function drawProof(cand) {
    const p = $("#proofLayer"); if (!p) return; p.innerHTML = "";
    const sh = (cand.evidence || {}).shadow || {};
    if (Array.isArray(sh.strip)) {
      const [x1, y1, x2, y2] = sh.strip;
      const r = document.createElementNS(SVGNS, "rect");
      r.setAttribute("x", x1); r.setAttribute("y", y1); r.setAttribute("width", x2 - x1); r.setAttribute("height", y2 - y1);
      r.setAttribute("class", "proof-strip"); p.appendChild(r);
    }
    if (Array.isArray(sh.echo_xy)) {
      const [ex, ey] = sh.echo_xy;
      const c = document.createElementNS(SVGNS, "circle");
      c.setAttribute("cx", ex); c.setAttribute("cy", ey); c.setAttribute("r", 5); c.setAttribute("class", "proof-echo"); p.appendChild(c);
    }
    p.classList.remove("flash"); void p.offsetWidth; p.classList.add("flash");
  }
  function highlight(id, on) { const g = svg().querySelector(`.bx[data-id="${id}"]`); if (g) g.classList.toggle("bx-hi", on); }
  function pulse(id) {
    const g = svg().querySelector(`.bx[data-id="${id}"]`); if (!g) return;
    g.classList.remove("bx-pulse"); void g.getBBox; g.classList.add("bx-pulse");
    setTimeout(() => g.classList.remove("bx-pulse"), 1200);
  }
  function init() {
    const cv = canvas();
    cv.addEventListener("wheel", e => {
      e.preventDefault(); cancelTour();
      stage().style.transition = "transform .10s ease-out";
      const rect = cv.getBoundingClientRect();
      zoomAtPoint(e.clientX - rect.left, e.clientY - rect.top, e.deltaY < 0 ? 1.14 : 1 / 1.14);
      clearTimeout(wheelClear); wheelClear = setTimeout(() => stage().style.transition = "", 150);
    }, { passive: false });
    cv.addEventListener("pointerdown", e => {
      if (e.target.closest(".bx")) return;
      cancelTour(); stage().style.transition = "";
      drag = { x: e.clientX, y: e.clientY, tx, ty }; cv.setPointerCapture(e.pointerId); cv.classList.add("grabbing");
    });
    cv.addEventListener("pointermove", e => { if (!drag) return; tx = drag.tx + (e.clientX - drag.x); ty = drag.ty + (e.clientY - drag.y); apply(); });
    const end = () => { drag = null; cv.classList.remove("grabbing"); };
    cv.addEventListener("pointerup", end); cv.addEventListener("pointercancel", end);
    cv.addEventListener("dblclick", () => { cancelTour(); fit(); });
    window.addEventListener("resize", () => { if (natW) fit(); });
  }
  return { init, render, fit, zoomBy, focusBox, drawProof, highlight, pulse };
})();

/* ------------------------------------------------------------------ pipeline dock: stepper + live agent log */
let stepperTimer = null;
function stepperRun(stages) {
  stepperReset();
  let i = 0;
  const tick = () => {
    $$(".dstep").forEach(s => s.classList.remove("is-active"));
    stages.slice(0, i).forEach(st => setStage(st, "done"));
    if (i < stages.length) setStage(stages[i], "active");
    i = Math.min(i + 1, stages.length);
  };
  tick(); stepperTimer = setInterval(tick, 520);
}
function stepperFinish(stages) { clearInterval(stepperTimer); stepperTimer = null; $$(".dstep").forEach(s => s.classList.remove("is-active")); stages.forEach(st => setStage(st, "done")); }
function stepperReset() { clearInterval(stepperTimer); stepperTimer = null; $$(".dstep").forEach(s => s.classList.remove("is-active", "is-done")); }
function setStage(name, mode) {
  const el = $(`.dstep[data-stage="${name}"]`); if (!el) return;
  el.classList.toggle("is-active", mode === "active"); el.classList.toggle("is-done", mode === "done");
}

const Log = (() => {
  const el = () => $("#agentLog");
  let queue = [], timer = null;
  function reset() { clearTimeout(timer); timer = null; queue = []; el().innerHTML = ""; }
  function idle() { el().innerHTML = `<li class="log-idle">idle — run the loop to stream the agent's tool calls</li>`; }
  function push(entries) { queue.push(...entries); if (!timer) drain(); }
  function now(entry) {
    const li = document.createElement("li");
    li.className = "log-line" + (entry.stage ? " is-stage" : "") + (entry.verdict ? ` v-${entry.verdict}` : "");
    li.innerHTML = `<span class="log-t">${escapeHtml(entry.t || "")}</span><span class="log-tool">${escapeHtml(entry.tool || "")}</span><span class="log-msg">${escapeHtml(entry.msg || "")}</span>`;
    el().appendChild(li); el().scrollTop = el().scrollHeight;
  }
  function drain() { if (!queue.length) { timer = null; return; } now(queue.shift()); timer = setTimeout(drain, 95); }
  return { reset, idle, push, now };
})();

function toast(txt) { const t = $("#runToast"); $("#runToastTxt").textContent = txt; t.hidden = false; }
function toastHide() { $("#runToast").hidden = true; }

/* ------------------------------------------------------------------ ANALYZE */
async function runAnalyze() {
  if (!state.selected) return;
  $("#analyzeBtn").disabled = true;
  $("#analyzePlaceholder").hidden = true;
  toast("See → Prove → Decide …");
  stepperRun(["see", "prove", "decide"]);
  Log.reset(); Log.now({ stage: true, tool: "SEE", msg: "perceive — full-frame YOLO11 via cv2.dnn …", t: "run" });

  try {
    let url = `${API}/api/analyze`, opts = { method: "POST" };
    if (state.selected.file) { const fd = new FormData(); fd.append("file", state.selected.file); opts.body = fd; }
    else url += `?sample=${encodeURIComponent(state.selected.id)}`;
    const d = await fetch(url, opts).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); });
    state._analyze = d;
    renderAnalyze(d);
    stepperFinish(["see", "prove", "decide"]);
  } catch (e) {
    stepperReset();
    $("#analyzePlaceholder").hidden = false;
    $("#analyzePlaceholder").querySelector("p").innerHTML =
      `<b style="color:var(--danger)">Analyze failed (${e.message}).</b><br/>Is the backend running and the model present?`;
  } finally { toastHide(); $("#analyzeBtn").disabled = false; }
}

function renderAnalyze(d) {
  cancelTour(); hideAgentEye(); state._tourOptOut = false; state._ready = true;
  $("#analyzePlaceholder").hidden = true;
  $$(".seg-btn").forEach(x => x.classList.toggle("is-active", x.dataset.view === "overlay"));
  $("#viewer").classList.remove("boxes-off");
  $("#stageCap").textContent = `${d.frame_id} · ${d.width}×${d.height}px · nadir=${d.nadir} · ${d.candidates.length} candidate(s)`;
  $("#verdictCounts").innerHTML = VERDICTS.map(v => `<span class="vc vc-${v}"><span class="n">${d.counts[v] || 0}</span>${v}</span>`).join("");
  const sm = d.stage_ms || {};
  $("#latency").textContent = `⏱ ${d.wall_ms}ms · see ${fmt(sm.see)} · prove+decide ${fmt(sm.prove_decide)}`;
  const dl = $("#dockLatency"); if (dl) dl.textContent = `wall ${d.wall_ms}ms`;

  state._boxes = d.candidates.map((c, i) => ({ id: i, bbox: c.bbox, verdict: c.verdict, conf: c.conf, cand: c }));
  state._cursor = -1;
  Viewer.render(d, state._boxes);
  $("#viewerEmpty").hidden = d.candidates.length > 0;

  const grid = $("#evidenceGrid"); grid.innerHTML = "";
  if (!d.candidates.length) {
    grid.innerHTML = `<p class="empty-hint">No candidates on this frame — the detector found nothing to prove. (For the "no labelled pot" samples, that is the correct, honest result.)</p>`;
    buildClassLegend([]); applyFilters(); streamLog(d, []); return;
  }
  const order = { confirmed: 0, review: 1, rejected: 2 };
  const ordered = state._boxes.slice().sort((a, b) =>
    (order[a.verdict] - order[b.verdict]) || ((b.cand.evidence?.evidence_score || 0) - (a.cand.evidence?.evidence_score || 0)));
  ordered.forEach(b => grid.appendChild(evidenceCard(b.cand, b.id)));
  buildClassLegend(ordered);
  applyFilters();
  streamLog(d, ordered);
  maybeAutoTour();
}

/* class cut/toggle legend */
function buildClassLegend(ordered) {
  const leg = $("#classLegend"); if (!leg) return;
  const counts = {};
  ordered.forEach(b => { counts[b.cand.cls_name] = (counts[b.cand.cls_name] || 0) + 1; });
  const names = Object.keys(counts);
  state.hiddenClasses.forEach(c => { if (!names.includes(c)) state.hiddenClasses.delete(c); });
  leg.innerHTML = "";
  names.forEach((nm, i) => {
    const chip = document.createElement("button");
    chip.className = "cls-chip" + (state.hiddenClasses.has(nm) ? " is-off" : "");
    chip.innerHTML = `<span class="sw" style="background:${CLASS_SW[i % CLASS_SW.length]}"></span>${escapeHtml(nm)}<span class="cnt">${counts[nm]}</span>`;
    chip.title = "show / cut this class";
    chip.onclick = () => {
      if (state.hiddenClasses.has(nm)) state.hiddenClasses.delete(nm); else state.hiddenClasses.add(nm);
      chip.classList.toggle("is-off");
      applyFilters();
    };
    leg.appendChild(chip);
  });
}

/* gate + class filter → dim boxes & cards */
function applyFilters() {
  const g = state.gate; let shown = 0;
  state._boxes.forEach(b => {
    const on = b.conf >= g && !state.hiddenClasses.has(b.cand.cls_name);
    if (on) shown++;
    const box = $(`#boxLayer .bx[data-id="${b.id}"]`); if (box) box.classList.toggle("bx-under", !on);
    const card = $(`.ev-card[data-id="${b.id}"]`); if (card) card.classList.toggle("ev-under", !on);
  });
  const el = $("#gateShowing"); if (el) el.textContent = state._boxes.length ? `${shown}/${state._boxes.length} shown` : "";
}

/* stream the trace into the dock agent log */
function streamLog(d, ordered) {
  const sm = d.stage_ms || {};
  const entries = [{ stage: true, tool: "SEE", msg: `perceive → ${d.candidates.length} candidate(s)`, t: fmt(sm.see) + "ms" }];
  ordered.forEach(b => {
    const c = b.cand;
    entries.push({ stage: true, tool: `cand #${b.id}`, msg: `${c.cls_name} · det ${c.conf.toFixed(2)}`, t: "" });
    (c.trace || []).forEach(s => entries.push({ tool: s.tool, msg: s.rationale, t: fmt(s.latency_ms) + "ms" }));
    entries.push({ verdict: c.verdict, tool: "→ " + c.verdict, msg: verdictReason(c), t: "" });
  });
  Log.push(entries);
}
function verdictReason(c) {
  const rl = (c.evidence || {}).relook || {};
  if (c.verdict === "confirmed") return rl.found && rl.conf >= 0.4 ? "re-look persisted" : "high detector confidence";
  if (c.verdict === "review") return "human review — evidence retained, ranked";
  return "low evidence — retained for audit, not surfaced";
}

function selectCandidate(id) {
  state._cursor = id;
  Viewer.focusBox(state._boxes.find(b => b.id === id).bbox);
  Viewer.pulse(id);
  $$(".ev-card").forEach(c => c.classList.toggle("is-linked", +c.dataset.id === id));
  const card = $(`.ev-card[data-id="${id}"]`); if (card) card.scrollIntoView({ behavior: "smooth", block: "nearest" });
}
function cycleCandidate(dir) {
  const visible = $$(".ev-card:not(.ev-under)").map(c => +c.dataset.id);
  if (!visible.length) return;
  let idx = visible.indexOf(state._cursor);
  idx = state._cursor < 0 ? (dir > 0 ? 0 : visible.length - 1) : (idx + dir + visible.length) % visible.length;
  selectCandidate(visible[idx]);
}

/* the agent's-eye picture-in-picture */
function gazeTo(id) {
  const b = state._boxes.find(x => x.id === id); if (!b) return;
  const c = b.cand, rl = (c.evidence || {}).relook || {};
  state._cursor = id;
  Viewer.focusBox(c.bbox, { scale: rl.scale });
  Viewer.drawProof(c); Viewer.pulse(id); showAgentEye(c);
  $$(".ev-card").forEach(x => x.classList.toggle("is-linked", +x.dataset.id === id));
  const card = $(`.ev-card[data-id="${id}"]`);
  if (card && !state._tour) card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  const msg = rl.found
    ? `Agent zoomed ${rl.scale ? rl.scale.toFixed(1) + "×" : ""} here → re-fired at ${(rl.conf || 0).toFixed(2)} (${(rl.gain || 0) >= 0 ? "+" : ""}${(rl.gain || 0).toFixed(2)})`
    : `Agent zoomed in here → did not re-fire (speckle, not a solid object)`;
  toast(msg); if (!state._tour) setTimeout(toastHide, 2600);
}
function showAgentEye(c) {
  const ae = $("#agentEye"); if (!ae) return;
  const rv = c.relook_view, src = rv ? (rv.enhanced_png || rv.zoom_png) : c.crop_png;
  if (!src) { hideAgentEye(); return; }
  $("#aeImg").src = src;
  const rl = (c.evidence || {}).relook || {};
  $("#aeCap").textContent = `agent's eye${rv && rv.scale ? " · " + rv.scale.toFixed(1) + "× · CLAHE" : ""}` + (rl.found ? ` · re-fire ${(rl.conf || 0).toFixed(2)}` : " · no re-fire");
  ae.hidden = false; ae.classList.remove("pop"); void ae.offsetWidth; ae.classList.add("pop");
}
function hideAgentEye() { const ae = $("#agentEye"); if (ae) ae.hidden = true; }

/* cinematic auto-tour */
let _tourTimer = null;
function maybeAutoTour() { if (state._boxes.length && !state._tourOptOut) setTimeout(() => { if (!state._tourOptOut && !state._tour) playGazeTour(false); }, 750); }
function playGazeTour() {
  endTour();
  const ids = $$(".ev-card:not(.ev-under)").map(c => +c.dataset.id);
  if (!ids.length) return;
  state._tour = true; state._tourOptOut = false;
  const btn = $("#tourBtn"); if (btn) { btn.classList.add("is-on"); btn.innerHTML = "⏸ stop"; }
  let i = 0;
  const step = () => {
    if (!state._tour) return;
    if (i >= ids.length) { endTour(); setTimeout(() => Viewer.fit(), 400); setTimeout(hideAgentEye, 1300); return; }
    gazeTo(ids[i]); i++; _tourTimer = setTimeout(step, 2100);
  };
  step();
}
function endTour() { state._tour = false; clearTimeout(_tourTimer); _tourTimer = null; const btn = $("#tourBtn"); if (btn) { btn.classList.remove("is-on"); btn.innerHTML = "▶ agent tour"; } }
function cancelTour() { if (state._tour) state._tourOptOut = true; endTour(); hideAgentEye(); }

function evidenceCard(c, id) {
  const ev = c.evidence || {}, sh = ev.shadow || {}, rl = ev.relook || {}, v = c.verdict;
  const card = document.createElement("div");
  card.className = `ev-card v-${v}`; card.dataset.id = id;
  const rlConf = rl.conf || 0, gain = (rlConf - c.conf);
  const arrowCls = rlConf > c.conf ? "up" : (rlConf < c.conf ? "down" : "");
  const shCls = sh.quality === "clear" ? "chip-clear" : sh.quality === "weak" ? "chip-weak" : "chip-none";
  const height = sh.height_m != null ? `${sh.height_m} m` : "—";
  card.innerHTML = `
    <div class="ev-crop${c.relook_view ? " has-enh" : ""}">
      ${c.crop_png ? `<img class="ev-img raw" alt="evidence crop" src="${c.crop_png}"/>` : `<div style="aspect-ratio:1"></div>`}
      ${c.relook_view ? `<img class="ev-img enh" alt="CLAHE re-look (what the agent saw)" src="${c.relook_view.enhanced_png}"/>` : ""}
      <span class="ev-badge b-${v}">${v}</span>
      ${c.relook_view ? `<button class="cmp-chip" title="toggle raw ⇄ CLAHE re-look">raw ⇄ enhanced</button>` : ""}
      <button class="gaze-btn" title="replay what the agent saw (G)">◉ Replay gaze</button>
    </div>
    <div class="ev-body">
      <div class="conf-flow">
        <div class="conf-chip"><span class="lbl">detector</span><span class="val">${(c.conf).toFixed(2)}</span></div>
        <div class="conf-arrow ${arrowCls}"><div class="track"></div>
          <span class="tag">re-look ${rl.found ? `${rlConf.toFixed(2)} (${gain >= 0 ? "+" : ""}${gain.toFixed(2)})` : "no re-fire"}</span></div>
        <div class="conf-chip"><span class="lbl">${rl.scale ? rl.scale.toFixed(1) + "× zoom" : "zoom"}</span><span class="val ${arrowCls}">${rl.found ? rlConf.toFixed(2) : "—"}</span></div>
      </div>
      <div class="ev-facts">
        <div class="fact"><span class="k">shadow</span><span class="v ${shCls}">${sh.quality || "none"}${sh.contrast ? ` · c${sh.contrast}` : ""}</span></div>
        <div class="fact"><span class="k">est. height</span><span class="v">${height}</span></div>
        <div class="fact"><span class="k">echo × bg</span><span class="v">${(sh.echo_ratio ?? ev.echo_ratio ?? 0).toFixed(1)}×</span></div>
        <div class="fact"><span class="k">evidence</span><span class="v">${(ev.evidence_score ?? 0).toFixed(2)}</span></div>
      </div>
      <ul class="ev-notes">${(ev.notes || []).map(n => `<li>${escapeHtml(n)}</li>`).join("")}</ul>
      ${traceBlock(c.trace || [])}
    </div>`;
  card.addEventListener("mouseenter", () => Viewer.highlight(id, true));
  card.addEventListener("mouseleave", () => Viewer.highlight(id, false));
  card.querySelector(".ev-crop").addEventListener("click", () => selectCandidate(id));
  card.querySelector(".gaze-btn").addEventListener("click", e => { e.stopPropagation(); gazeTo(id); });
  const chip = card.querySelector(".cmp-chip");
  if (chip) chip.addEventListener("click", e => { e.stopPropagation(); card.querySelector(".ev-crop").classList.toggle("show-enh"); });
  return card;
}
function traceBlock(trace) {
  if (!trace.length) return "";
  const steps = trace.map(s => `
    <li class="trace-step ${s.tool === "decide" ? "t-decide" : ""}">
      <span class="trace-tool">${s.tool}</span><span class="trace-ms">${fmt(s.latency_ms)} ms</span>
      <div class="trace-why">${escapeHtml(s.rationale)}</div></li>`).join("");
  return `<details class="trace"><summary>agent trace — ${trace.length} tool calls</summary><ol>${steps}</ol></details>`;
}

/* ------------------------------------------------------------------ SURVEY */
async function runSurvey() {
  const btn = $("#surveyBtn"); btn.disabled = true;
  $("#surveyPlaceholder").hidden = true; $("#missionBanner").hidden = true;
  toast("Running full loop over the survey …");
  stepperRun(["see", "prove", "decide", "act"]);
  Log.reset(); Log.now({ stage: true, tool: "SURVEY", msg: "running See→Prove→Decide→Act over all sample frames …", t: "run" });
  const gps = $("#gpsMode").value;
  try {
    const d = await fetch(`${API}/api/survey?use_samples=1&gps=${gps}`, { method: "POST" }).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); });
    state.survey = d;
    renderSurvey(d);
    stepperFinish(["see", "prove", "decide", "act"]);
  } catch (e) {
    stepperReset();
    $("#surveyPlaceholder").hidden = false;
    $("#surveyPlaceholder").querySelector("p").innerHTML =
      `<b style="color:var(--danger)">Survey failed (${e.message}).</b><br/>The backend needs the model + local samples.`;
  } finally { toastHide(); btn.disabled = false; }
}

function renderSurvey(d) {
  $("#surveyPlaceholder").hidden = true;
  const m = d.mission, cc = m.counts || {};
  const gpsTag = !m.gps_available ? `<span class="mb-tag">no GPS — table only</span>`
    : m.gps_synthetic ? `<span class="mb-tag mb-synthetic">⚠ synthetic demo GPS</span>` : `<span class="mb-tag mb-real">real GPS</span>`;
  const mb = $("#missionBanner"); mb.hidden = false; mb.className = "mission-banner";
  mb.innerHTML = `<b>🧭 Mission plan ready</b>
    <span>${cc.confirmed || 0} confirmed → <b>${m.recovery_route.length}-stop recovery route</b>${m.route_length_m != null ? ` (${m.route_length_m} m)` : ""}</span>
    <span>· ${cc.review || 0} to re-survey</span>${gpsTag}
    <span class="mb-tag" style="border-color:rgba(217,164,65,.5);color:#ffd9a3">✋ human approval required — nothing auto-dispatched</span>`;
  renderMap(d); renderDownloads(d.survey_id, m); renderThumbs(d); renderHazards(d);
  Log.push([
    { stage: true, tool: "ACT", msg: `${cc.confirmed || 0} confirmed · ${cc.review || 0} review · ${cc.rejected || 0} rejected`, t: "" },
    { tool: "plan_route", msg: `${m.recovery_route.length}-stop nearest-neighbour recovery route${m.route_length_m != null ? " · " + m.route_length_m + " m" : ""}`, t: "" },
    { tool: "human_gate", msg: "human approval required — nothing auto-dispatched", t: "" },
  ]);
}

function renderMap(d) {
  const m = d.mission, geoObjs = d.tracked.filter(t => t.lat != null && t.lon != null), note = $("#mapNote");
  if (!m.gps_available || !geoObjs.length) { $("#mapWrap").style.display = "none"; return; }
  $("#mapWrap").style.display = "";
  note.textContent = m.gps_synthetic
    ? "⚠ SYNTHETIC DEMO GPS — not real coordinates. The HF crab-pot frames carry no GPS; this track is generated only to demonstrate the map + route."
    : "Real per-ping GPS.";
  if (!state.map) {
    state.map = L.map("map", { zoomControl: true, attributionControl: true });
    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", { attribution: '&copy; OpenStreetMap &copy; CARTO', maxZoom: 20, subdomains: "abcd" }).addTo(state.map);
  }
  state.mapLayers.forEach(l => state.map.removeLayer(l)); state.mapLayers = [];
  const byId = Object.fromEntries(d.tracked.map(t => [t.oid, t])), latlngs = [];
  geoObjs.forEach(t => {
    const color = VCOLOR[t.verdict] || "#8ba6c2";
    const mk = L.circleMarker([t.lat, t.lon], { radius: t.verdict === "confirmed" ? 8 : 6, color: "#02121d", weight: 1.5, fillColor: color, fillOpacity: .95 })
      .bindPopup(`<b>${t.oid}</b> · ${t.verdict}<br/>${t.cls_name} · conf ${t.conf}<br/>evidence ${t.evidence_score} · ${t.shadow_quality} shadow${t.height_m != null ? ` · h~${t.height_m} m` : ""}<br/>${t.lat.toFixed(5)}, ${t.lon.toFixed(5)} ±${t.geo_error_m ?? "?"} m`);
    mk.addTo(state.map); state.mapLayers.push(mk); latlngs.push([t.lat, t.lon]);
  });
  const routePts = m.recovery_route.map(id => byId[id]).filter(t => t && t.lat != null).map(t => [t.lat, t.lon]);
  if (routePts.length > 1) {
    const line = L.polyline(routePts, { color: "#1fb6d5", weight: 2.5, dashArray: "6 6", opacity: .9 }).addTo(state.map); state.mapLayers.push(line);
    routePts.forEach((p, i) => {
      const badge = L.marker(p, { icon: L.divIcon({ className: "", html: `<div style="background:#1fb6d5;color:#04121a;font:700 10px/18px var(--mono,monospace);width:18px;height:18px;border-radius:50%;text-align:center;border:2px solid #04121a">${i + 1}</div>`, iconSize: [18, 18], iconAnchor: [9, 9] }) }).addTo(state.map);
      state.mapLayers.push(badge);
    });
  }
  state.map.fitBounds(L.latLngBounds(latlngs).pad(0.25)); setTimeout(() => state.map.invalidateSize(), 60);
}

function renderDownloads(surveyId) {
  const fmts = [["geojson", "hazards + route (GIS)"], ["gpx", "waypoints (boat GPS)"], ["kml", "Google Earth"], ["csv", "spreadsheet"], ["json", "full audit trail"]];
  $("#dlGrid").innerHTML = fmts.map(([f, desc]) => `<button class="dl-btn" data-fmt="${f}"><b>.${f}</b><span>${desc}</span></button>`).join("");
  $$("#dlGrid .dl-btn").forEach(b => b.onclick = () => {
    const a = document.createElement("a");
    a.href = `${API}/api/report/${b.dataset.fmt}?survey_id=${encodeURIComponent(surveyId)}`;
    a.download = `mission.${b.dataset.fmt}`; document.body.appendChild(a); a.click(); a.remove();
  });
}
function renderThumbs(d) {
  $("#thumbStrip").innerHTML = d.frames.filter(f => f.thumb_png)
    .map(f => `<img class="thumb" title="${f.frame_id} · ${cShort(f.counts)}" src="${f.thumb_png}" alt="${f.frame_id}"/>`).join("");
}
function renderHazards(d) {
  const rows = d.tracked;
  $("#hazCount").textContent = `${rows.length} on map · REJECTED kept for audit`;
  const body = $("#hazBody"); body.innerHTML = "";
  rows.forEach(t => {
    const tr = document.createElement("tr");
    const coords = t.lat != null ? `${t.lat.toFixed(5)}, ${t.lon.toFixed(5)}` : `<span class="muted">no GPS</span>`;
    tr.innerHTML = `<td>${t.oid}</td><td>${t.cls_name}</td>
      <td><span class="v-tag" style="color:${VCOLOR[t.verdict]};background:${VCOLOR[t.verdict]}22">${t.verdict}</span></td>
      <td>${t.conf}</td><td>${t.evidence_score}</td><td>${t.height_m != null ? t.height_m + " m" : "—"}</td>
      <td class="${t.shadow_quality === "clear" ? "chip-clear" : t.shadow_quality === "weak" ? "chip-weak" : "chip-none"}">${t.shadow_quality}</td>
      <td>${coords}</td><td>${t.geo_error_m != null ? "±" + t.geo_error_m + " m" : "—"}</td><td>${reviewCell(t)}</td>`;
    body.appendChild(tr);
  });
  $$("#hazBody .rev-btns button").forEach(b => b.onclick = e => {
    const tr = e.target.closest("tr"); tr.classList.add("rev-done");
    e.target.closest("td").innerHTML = `<span class="muted">${b.classList.contains("ok") ? "✓ approved" : "✕ rejected"}</span>`;
  });
}
function reviewCell(t) {
  if (t.verdict !== "review") return `<span class="muted">—</span>`;
  return `<span class="rev-btns"><button class="ok" title="approve for recovery">✓</button><button class="no" title="dismiss">✕</button></span>`;
}

/* ------------------------------------------------------------------ utils */
function fmt(x) { return x == null ? "—" : (Math.round(x * 10) / 10).toFixed(1); }
function cShort(c) { return `C${c.confirmed || 0} R${c.review || 0} X${c.rejected || 0}`; }
function escapeHtml(s) { return String(s).replace(/[&<>"']/g, m => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m])); }
