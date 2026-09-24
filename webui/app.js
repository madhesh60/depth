/* DEPTH studio — drives the See → Prove → Decide → Act loop over the FastAPI backend.
   Zero-build vanilla JS. All API shapes match src/dashboard/app.py. */
"use strict";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const API = ""; // same-origin (served by FastAPI)
const SVGNS = "http://www.w3.org/2000/svg";

const VERDICTS = ["confirmed", "review", "low_risk"];
const VCOLOR = { confirmed: "#2ea043", review: "#d9a441", low_risk: "#5f6f7d", rejected: "#5f6f7d" };
const VLABEL = { confirmed: "confirmed", review: "review", low_risk: "low-risk", rejected: "low-risk" };
const CLASS_SW = ["#1fb6d5", "#d9a441", "#8b7ff0", "#4fb477", "#e5789b"];

const state = {
  samples: [], selected: null, survey: null, map: null, mapLayers: [],
  _analyze: null, _boxes: [], _cursor: -1, gate: 0.05,
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
  wireMissedTool();
  wireStudy();
  wireAudit();
  await Promise.all([loadHealth(), loadSamples(), loadMetrics(), loadLabelStats()]);
});

async function loadHealth() {
  try {
    const h = await fetch(`${API}/api/health`).then(r => r.json());
    const cv = $("#pillCv"), md = $("#pillModel");
    cv.textContent = `OpenCV ${h.opencv}`;
    cv.className = "pill " + (String(h.opencv).startsWith("5") ? "pill-ok" : "pill");
    md.textContent = h.model_loaded ? "model ready" : h.model_error ? "model error" : "model warming up…";
    md.className = "pill " + (h.model_loaded ? "pill-ok" : h.model_error ? "pill-bad" : "pill");
    md.title = h.model_error || (h.warmup_ms != null ? `warm-up ${h.warmup_ms} ms` : "");
    if (h.is_cool_path) { cv.textContent += " · COOL"; cv.title = h.cv2_file; }
    const rc = $("#regCv"), rm = $("#regModel");
    if (rc) rc.textContent = h.opencv + (h.is_cool_path ? " (COOL /opt/cool)" : "");
    if (rm) rm.textContent = h.model_loaded ? "loaded + warm" : "loading…";
    state.calibration = h.calibration || null;
    const cal = h.calibration || {};
    if ($("#regWeights")) $("#regWeights").textContent = `${cal.model || h.model} · ${cal.imgsz || "?"} px ONNX`;
    if ($("#regClasses")) $("#regClasses").textContent = `${(cal.names || []).length} · ${(cal.names || []).join(", ")}`;
    if ($("#regGateK")) $("#regGateK").textContent = `gate · ${cal.guaranteed_class || "?"}`;
    renderGuarantees(h.calibration);
    if (!h.model_loaded && !h.model_error) setTimeout(loadHealth, 1500);   // re-poll until warm
  } catch (e) {
    $("#pillCv").textContent = "backend offline";
    $("#pillCv").className = "pill pill-bad";
  }
}

/* Live runtime: where this server runs (arch · EC2 type · COOL?) + rolling per-stage p50. */
async function loadMetrics() {
  try {
    const m = await fetch(`${API}/api/metrics`).then(r => r.json());
    const h = m.host || {}, e = m.ec2 || {}, o = m.opencv || {};
    const arch = /aarch64|arm64/i.test(h.machine || "") ? "Arm64 (Graviton)" : (h.machine || "?");
    $("#regHost").textContent = `${arch}${e["instance-type"] ? " · " + e["instance-type"] : ""} · ${h.vcpus || "?"} vCPU · ${o.is_cool_path ? "COOL" : "stock"} OpenCV`;
    const s = m.stages_ms || {};
    if (s.see) $("#regLat").textContent = `stage1 ${s.stage1 ? s.stage1.p50 : "—"} · see ${s.see.p50} · decide ${s.prove_decide ? s.prove_decide.p50 : "—"} ms (n=${s.see.n})`;
  } catch (e) { /* metrics are optional */ }
}

/* The promises the tiers carry — read from models/<MODEL>/calibration.json via /api/health. */
function renderGuarantees(c) {
  const panel = $("#guarPanel"), meta = $("#guarMeta"), gate = $("#regGate"), tiers = $("#regTiers");
  if (!panel) return;
  if (!c || c.tau_review == null) {
    panel.innerHTML = `<p class="panel-note">No calibrated guarantee loaded — tiers use the legacy rules.</p>`;
    if (tiers) tiers.textContent = "legacy rules"; if (gate) gate.textContent = "0.10";
    return;
  }
  const g = c.guarantees || {}, v = g.verified_on_test || {};
  const pct = x => x == null ? "—" : `${Math.round(x * 100)}%`;
  const held = ok => ok == null ? "" : ok ? `<span class="g-held">held on test</span>` : `<span class="g-miss">NOT held on test</span>`;
  meta.textContent = "95% confidence";
  if (gate) gate.textContent = `${(c.detector_floor ?? c.tau_review).toFixed(2)} (floor) · review ≥ ${c.tau_review.toFixed(2)}`;
  if (tiers) tiers.textContent = `${c.method ? "conformal (CP · LTT)" : "calibrated"}${c.policy ? " · " + c.policy : ""}`;
  panel.innerHTML = `
    <div class="g-row"><span class="g-badge g-recall">≥ ${pct(g.recall_promise)}</span>
      <span class="g-txt">of pots reach a human (CONFIRMED + REVIEW)${g.requested_recall_achievable === false ? ` — the requested 90% is <b>not</b> achievable with this detector (proposal ceiling ${pct(g.recall_ceiling)})` : ""}. ${held(v.recall_promise_held)}</span></div>
    <div class="g-row"><span class="g-badge g-prec">${g.precision_promise ? "≥ " + pct(g.precision_promise) : "none"}</span>
      <span class="g-txt">${g.precision_promise ? `of CONFIRMED finds are real (τ_confirm ${c.tau_confirm != null ? c.tau_confirm.toFixed(2) : "—"}). ${held(v.precision_promise_held)}` : "no auto-confirm can be promised with this model — every find goes to a human."}</span></div>
    <p class="panel-note">Fit on held-out validation frames, verified once on test (${escapeHtml((c.fit && c.fit.frames) ? c.fit.frames + " calib. frames" : "")}). <a href="https://github.com/madhesh60/depth/blob/main/${escapeHtml(g.report || "docs/")}" target="_blank" rel="noopener">report</a></p>`;
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
    if (b.dataset.mode === "study") { loadStudySummary(); }
    if (b.dataset.mode === "audit") { loadAuditSummary(); }
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
    else if (e.key.toLowerCase() === "m") { toggleMissed(); e.preventDefault(); }
    else if (e.key === "Escape" && state._drawMissed) { toggleMissed(false); e.preventDefault(); }
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
    drawBoxes(boxes, (d.stage1 || {}).bottom_line_native, (d.stage1 || {}).altitude_px);
    im.onload = () => fit();
    im.src = d.frame_png;
    if (im.complete && im.naturalWidth) fit();
  }
  function drawBoxes(boxes, seabed, alt) {
    const s = svg(); s.innerHTML = "";
    if (Array.isArray(seabed) && seabed.length > 1) {         // Stage 1: tracked first seabed return
      const g = document.createElementNS(SVGNS, "g"); g.setAttribute("class", "seabed");
      const pl = document.createElementNS(SVGNS, "polyline");
      pl.setAttribute("points", seabed.map(p => p.join(",")).join(" "));
      g.appendChild(pl);
      const [lx, ly] = seabed[Math.min(seabed.length - 1, 2)];
      const t = document.createElementNS(SVGNS, "text");
      t.setAttribute("x", lx + 4); t.setAttribute("y", ly - 4); t.setAttribute("class", "seabed-label");
      t.textContent = `seabed · altitude ${alt != null ? alt.toFixed(0) : "?"} px (stage 1)`;
      g.appendChild(t); s.appendChild(g);
    }
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
  function toImage(clientX, clientY) {
    const r = canvas().getBoundingClientRect();
    return [Math.max(0, Math.min(natW, (clientX - r.left - tx) / scale)), Math.max(0, Math.min(natH, (clientY - r.top - ty) / scale))];
  }
  return { init, render, fit, zoomBy, focusBox, drawProof, highlight, pulse, toImage };
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
    loadMetrics();
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
  const s1 = d.stage1 || {};
  $("#stageCap").textContent = `${d.frame_id} · ${d.width}×${d.height}px · nadir=${d.nadir}${d.orientation && d.orientation.source ? ` (${d.orientation.source})` : ""}`
    + ` · seabed ${s1.measured ? `tracked @ ${s1.altitude_px} px` : "not measured"} · ${d.candidates.length} candidate(s)`;
  $("#verdictCounts").innerHTML = VERDICTS.map(v => `<span class="vc vc-${v}"><span class="n">${d.counts[v] || 0}</span>${v}</span>`).join("");
  const sm = d.stage_ms || {};
  $("#latency").textContent = `⏱ ${d.wall_ms}ms · stage1 ${fmt(sm.stage1)} · see ${fmt(sm.see)} · prove+decide ${fmt(sm.prove_decide)}`;
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
  const order = { confirmed: 0, review: 1, low_risk: 2, rejected: 2 };
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
  const s1 = d.stage1 || {}, o = s1.orientation || d.orientation || {};
  const entries = [
    { stage: true, tool: "STAGE 1", msg: `canonicalise → ${s1.palette || "gray"} palette · nadir ${o.nadir || d.nadir} (${o.source || "?"})`
        + ` · ${s1.measured ? `seabed tracked at ${s1.altitude_px} px slant (${Math.round((s1.track_conf || 0) * 100)}% of pings)` : "seabed not measured (no water-column step / unknown orientation)"}`
        + ` · detector input ${s1.detector_input || "raw"}`, t: fmt(sm.stage1) + "ms" },
    { stage: true, tool: "SEE", msg: `perceive → ${d.candidates.length} candidate(s)`, t: fmt(sm.see) + "ms" }];
  ordered.forEach(b => {
    const c = b.cand;
    entries.push({ stage: true, tool: `cand #${b.id}`, msg: `${c.cls_name} · det ${c.conf.toFixed(2)}`, t: "" });
    (c.trace || []).forEach(s => entries.push({ tool: s.tool, msg: s.rationale, t: fmt(s.latency_ms) + "ms" }));
    entries.push({ verdict: c.verdict, tool: "→ " + c.verdict, msg: verdictReason(c), t: "" });
  });
  Log.push(entries);
}
// Did the agent actually spend a re-look on this candidate? (calibrated mode skips it when the
// value of information is zero — the decide step records that choice.)
function relooked(c) {
  const d = (c.trace || []).find(s => s.tool === "decide");
  if (d && d.detail && d.detail.relooked != null) return !!d.detail.relooked;
  return (c.trace || []).some(s => /relook/.test(s.tool));
}
function decideDetail(c) { const d = (c.trace || []).find(s => s.tool === "decide"); return (d && d.detail) || {}; }
function tierPromise(c) {
  const g = (state.calibration && state.calibration.guarantees) || {}, d = decideDetail(c);
  if (d.mode !== "calibrated") return "legacy rule (no calibrated promise)";
  if (c.verdict === "confirmed") return `CONFIRMED tier promise: ≥ ${Math.round((g.precision_promise || 0) * 100)}% are real (95% conf.)`;
  if (c.verdict === "review") return `REVIEW + CONFIRMED together reach ≥ ${Math.round((g.recall_promise || 0) * 100)}% of pots (95% conf.)`;
  if (((state.calibration || {}).non_hazard_classes || []).includes(c.cls_name)) return "natural seabed class — not a hazard, kept for audit";
  return `LOW-RISK tier holds ≤ ${Math.round((1 - (g.recall_promise || 0)) * 100)}% of pots — kept for audit, never deleted`;
}
function verdictReason(c) {
  const d = decideDetail(c), p = (c.evidence || {}).p_pot;
  if (c.verdict === "confirmed") return d.mode === "calibrated" ? "above tau_confirm — carries the precision promise" : "legacy rule";
  if (c.verdict === "review") return `human review${p != null ? ` · P(pot) ~${Math.round(p * 100)}%` : ""} — ordered in the queue`;
  return "LOW-RISK — below tau_review, retained for audit, not surfaced";
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
  const p = (c.evidence || {}).p_pot;
  const msg = !relooked(c)
    ? `${VLABEL[c.verdict] || c.verdict}: det ${c.conf.toFixed(2)}${p != null ? ` · P(pot) ~${Math.round(p * 100)}%` : ""} — no re-look spent (it could not change this tier)`
    : rl.found
      ? `Agent zoomed ${rl.scale ? rl.scale.toFixed(1) + "×" : ""} here → re-fired at ${(rl.conf || 0).toFixed(2)} (${(rl.gain || 0) >= 0 ? "+" : ""}${(rl.gain || 0).toFixed(2)})`
      : `Agent zoomed in here → did not re-fire at higher resolution`;
  toast(msg); if (!state._tour) setTimeout(toastHide, 2600);
}
function showAgentEye(c) {
  const ae = $("#agentEye"); if (!ae) return;
  const rv = c.relook_view, src = rv ? (rv.enhanced_png || rv.zoom_png) : c.crop_png;
  if (!src) { hideAgentEye(); return; }
  $("#aeImg").src = src;
  const rl = (c.evidence || {}).relook || {};
  $("#aeCap").textContent = `agent's eye${rv && rv.scale ? " · " + rv.scale.toFixed(1) + "× · CLAHE" : ""}` +
    (!relooked(c) ? " · view only (no re-look inference)" : rl.found ? ` · re-fire ${(rl.conf || 0).toFixed(2)}` : " · no re-fire");
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

// Height is RELATIVE to sonar altitude unless the altitude was measured (never an assumed 10 m).
function heightText(m, rel, run) {
  if (m != null) return `${m} m`;
  if (run && rel) return `${Math.round(rel * 100)}% alt`;
  return "—";
}

function evidenceCard(c, id) {
  const ev = c.evidence || {}, sh = ev.shadow || {}, rl = ev.relook || {}, v = c.verdict;
  const card = document.createElement("div");
  card.className = `ev-card v-${v}`; card.dataset.id = id;
  const rlDone = relooked(c), rlConf = rl.conf || 0, gain = (rlConf - c.conf);
  const sc = ev.evidence_score ?? c.conf;
  const arrowCls = sc > c.conf + 1e-6 ? "up" : (sc < c.conf - 1e-6 ? "down" : "");
  const shCls = sh.quality === "clear" ? "chip-clear" : sh.quality === "weak" ? "chip-weak" : "chip-none";
  const height = heightText(sh.height_m, sh.height_rel, sh.run_px);
  card.innerHTML = `
    <div class="ev-crop${c.relook_view ? " has-enh" : ""}">
      ${c.crop_png ? `<img class="ev-img raw" alt="evidence crop" src="${c.crop_png}"/>` : `<div style="aspect-ratio:1"></div>`}
      ${c.relook_view ? `<img class="ev-img enh" alt="CLAHE re-look (what the agent saw)" src="${c.relook_view.enhanced_png}"/>` : ""}
      <span class="ev-badge b-${v}">${v}</span>
      ${c.relook_view ? `<button class="cmp-chip" title="toggle raw ⇄ CLAHE re-look">raw ⇄ enhanced</button>` : ""}
      <button class="gaze-btn" title="replay what the agent saw (G)">◉ Replay gaze</button>
    </div>
    <div class="ev-body">
      <p class="ev-tier">${escapeHtml(tierPromise(c))}</p>
      <div class="conf-flow">
        <div class="conf-chip"><span class="lbl">detector</span><span class="val">${(c.conf).toFixed(2)}</span></div>
        <div class="conf-arrow ${arrowCls}"><div class="track"></div>
          <span class="tag">${!rlDone ? "no re-look (VoI 0)" : `re-look ${rl.found ? `${rlConf.toFixed(2)} (${gain >= 0 ? "+" : ""}${gain.toFixed(2)})` : "no re-fire"}`}</span></div>
        <div class="conf-chip"><span class="lbl">score</span><span class="val ${arrowCls}">${(ev.evidence_score ?? c.conf).toFixed(2)}</span></div>
      </div>
      <div class="ev-facts">
        <div class="fact"><span class="k">shadow</span><span class="v ${shCls}">${sh.quality || "none"}${sh.contrast ? ` · c${sh.contrast}` : ""}</span></div>
        <div class="fact"><span class="k">rel. height</span><span class="v">${height}</span></div>
        <div class="fact"><span class="k">echo × bg</span><span class="v">${(sh.echo_ratio ?? ev.echo_ratio ?? 0).toFixed(1)}×</span></div>
        <div class="fact"><span class="k">position</span><span class="v ${c.in_water_column ? "chip-weak" : ""}">${c.in_water_column == null ? "—" : c.in_water_column ? "water column" : "on seabed"}${c.ground_range_px != null ? ` · ${Math.round(c.ground_range_px)} px gnd` : ""}</span></div>
        <div class="fact"><span class="k">P(pot)</span><span class="v">${ev.p_pot != null ? Math.round(ev.p_pot * 100) + "%" : "—"}</span></div>
      </div>
      <ul class="ev-notes">${(ev.notes || []).map(n => `<li>${escapeHtml(n)}</li>`).join("")}</ul>
      <div class="ev-label"><span class="lbl-k">your label</span>
        <button class="ok" title="a real pot — saved as a training label">✓ real pot</button>
        <button class="no" title="not a pot — saved as a hard negative">✕ not a pot</button></div>
      ${traceBlock(c.trace || [])}
    </div>`;
  card.querySelectorAll(".ev-label button").forEach(b => b.addEventListener("click", async e => {
    e.stopPropagation();
    const ok = b.classList.contains("ok");
    const res = await sendLabel({ bbox: c.bbox, cls_name: c.cls_name, decision: ok ? "confirm" : "reject",
      verdict_before: c.verdict, conf: c.conf, p_pot: ev.p_pot, source: "analyze" });
    if (res) {
      card.querySelectorAll(".ev-label button").forEach(x => x.classList.toggle("on", x === b));
      let sv = card.querySelector(".ev-label .saved");
      if (!sv) { sv = document.createElement("span"); sv.className = "saved"; card.querySelector(".ev-label").appendChild(sv); }
      sv.textContent = "saved ✓";
    }
  }));
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
  const budget = Math.max(1, Math.min(600, parseFloat($("#budgetMin").value) || 5));
  const boat = parseFloat($("#boatMin").value);
  const boatQ = Number.isFinite(boat) && boat > 0 ? `&boat_minutes=${Math.min(1440, boat)}` : "";
  try {
    // Surveys run as background jobs (no proxy timeouts, the server stays responsive); poll progress.
    const job = await fetch(`${API}/api/jobs/survey?use_samples=1&gps=${gps}&budget_minutes=${budget}${boatQ}`, { method: "POST" })
      .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); });
    Log.now({ tool: "queue_job", msg: `${job.job_id} · ${job.frames} frames queued`, t: "" });
    const d = await pollJob(job.job_id);
    state.survey = d;
    renderSurvey(d);
    loadMetrics();
    stepperFinish(["see", "prove", "decide", "act"]);
  } catch (e) {
    stepperReset();
    $("#surveyPlaceholder").hidden = false;
    $("#surveyPlaceholder").querySelector("p").innerHTML =
      `<b style="color:var(--danger)">Survey failed (${e.message}).</b><br/>The backend needs the model + local samples.`;
  } finally { toastHide(); btn.disabled = false; }
}

async function pollJob(jobId) {
  let lastDone = -1;
  for (let i = 0; i < 1800; i++) {                       // ≤ ~15 min at 0.5 s
    const j = await fetch(`${API}/api/jobs/${encodeURIComponent(jobId)}`).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); });
    const p = j.progress || {};
    if (p.done !== lastDone && p.total) {
      lastDone = p.done;
      $("#runToastTxt").textContent = `survey · frame ${Math.min(p.done + (j.status === "running" ? 1 : 0), p.total)}/${p.total}${p.frame ? " · " + p.frame : ""}`;
      if (p.done > 0) Log.now({ tool: "frame_done", msg: `${p.done}/${p.total}${p.frame ? " · " + p.frame : ""}`, t: "" });
    }
    if (j.status === "done") return j.result;
    if (j.status === "error") throw new Error(j.error || "job failed");
    await new Promise(r => setTimeout(r, 500));
  }
  throw new Error("timed out waiting for the survey job");
}

function renderSurvey(d) {
  $("#surveyPlaceholder").hidden = true;
  const m = d.mission, cc = m.counts || {};
  const gpsTag = !m.gps_available ? `<span class="mb-tag">no GPS — table only</span>`
    : m.gps_synthetic ? `<span class="mb-tag mb-synthetic">⚠ synthetic demo GPS</span>` : `<span class="mb-tag mb-real">real GPS</span>`;
  const mb = $("#missionBanner"); mb.hidden = false; mb.className = "mission-banner";
  mb.innerHTML = `<b>🧭 Mission plan ready</b>
    <span>${cc.confirmed || 0} confirmed → <b>${m.recovery_route.length}-stop recovery route</b>${m.route_length_m != null ? ` (${m.route_length_m} m)` : ""}</span>
    <span>· ${cc.review || 0} REVIEW → <b>${(m.inspection_route || []).length}-stop inspection route</b>${m.inspection_length_m != null ? ` (${m.inspection_length_m} m)` : ""}</span>${gpsTag}
    ${(m.resurvey_plan && (m.resurvey_plan.lines || []).length) ? `<span>· <b>${m.resurvey_plan.lines.length} re-survey pass${m.resurvey_plan.lines.length > 1 ? "es" : ""}</b> (${m.resurvey_plan.boat_minutes_planned} min boat) — opposite side, mid-swath</span>` : ""}
    ${m.repeat_merges ? `<span>· ${m.repeat_merges} repeat sighting${m.repeat_merges > 1 ? "s" : ""} merged</span>` : ""}
    <span class="mb-tag" style="border-color:rgba(217,164,65,.5);color:#ffd9a3">✋ human approval required — nothing auto-dispatched</span>`;
  renderMap(d); renderDownloads(d.survey_id, m); renderThumbs(d); renderHazards(d); renderEffort(d);
  Log.push([
    { stage: true, tool: "ACT", msg: `${cc.confirmed || 0} confirmed · ${cc.review || 0} review · ${cc.low_risk || 0} low-risk (kept for audit)`, t: "" },
    ...(m.budget && m.budget.cards_total != null ? [{ tool: "budget_plan", msg: `${m.budget.cards_affordable}/${m.budget.cards_total} review cards fit ${m.budget.minutes} min → ~${m.budget.expected_pots_in_budget} of ~${m.budget.expected_pots_in_queue} expected real pots`, t: "" }] : []),
    { tool: "plan_route", msg: `${m.recovery_route.length}-stop nearest-neighbour recovery route${m.route_length_m != null ? " · " + m.route_length_m + " m" : ""}`, t: "" },
    ...(m.resurvey_plan && m.resurvey_plan.lines ? [{ tool: "plan_resurvey", msg: `${m.resurvey_plan.lines.length} second-look passes · ${m.resurvey_plan.targets_covered}/${m.resurvey_plan.targets_total} uncertain targets · uncertainty Σp(1−p) ${m.resurvey_plan.voi_covered}/${m.resurvey_plan.voi_total} · ${m.resurvey_plan.boat_minutes_planned} min boat${m.resurvey_plan.boat_minutes_budget ? " (budget " + m.resurvey_plan.boat_minutes_budget + ")" : ""} — a real object's shadow must flip`, t: "" }] : []),
    ...(m.repeat_merges != null ? [{ tool: "merge_sightings", msg: `${m.repeat_merges} repeat sighting(s) merged (error circles overlap, different frames)`, t: "" }] : []),
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
    state.map = L.map("map", { zoomControl: true, attributionControl: true, maxZoom: 20 });
    // Keyless basemaps (CARTO dark now needs an API key). Satellite by default, OSM as an option; if
    // tiles can't load (offline / blocked) the map degrades to a plain grid — the hazards, error
    // radii and routes are vector overlays and stay correct without any basemap.
    const sat = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      { attribution: "Imagery &copy; Esri", maxZoom: 20, maxNativeZoom: 19, className: "tiles-dim" });
    const osm = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png",
      { attribution: "&copy; OpenStreetMap contributors", maxZoom: 20, maxNativeZoom: 19, className: "tiles-dim" });
    let tileErrors = 0;
    [sat, osm].forEach(layer => layer.on("tileerror", () => {
      if (++tileErrors === 6) { $("#map").classList.add("map-offline"); $("#mapNote").textContent += "  ·  basemap unavailable — overlays still exact."; }
    }));
    sat.addTo(state.map);
    L.control.layers({ "satellite": sat, "streets (OSM)": osm }, null, { position: "topright" }).addTo(state.map);
    L.control.scale({ imperial: false }).addTo(state.map);
  }
  state.mapLayers.forEach(l => state.map.removeLayer(l)); state.mapLayers = [];
  const byId = Object.fromEntries(d.tracked.map(t => [t.oid, t])), latlngs = [];
  geoObjs.forEach(t => {
    const color = VCOLOR[t.verdict] || "#8ba6c2";
    const mk = L.circleMarker([t.lat, t.lon], { radius: t.verdict === "confirmed" ? 8 : 6, color: "#02121d", weight: 1.5, fillColor: color, fillOpacity: .95 })
      .bindPopup(`<b>${t.oid}</b> · ${t.verdict}<br/>${t.cls_name} · conf ${t.conf}<br/>evidence ${t.evidence_score} · ${t.shadow_quality} shadow${t.height_m != null || t.height_rel ? ` · h ${heightText(t.height_m, t.height_rel, 1)}` : ""}<br/>${t.lat.toFixed(5)}, ${t.lon.toFixed(5)} ±${t.geo_error_m ?? "?"} m`);
    if (t.geo_error_m) {
      const ring = L.circle([t.lat, t.lon], { radius: t.geo_error_m, color, weight: 1, opacity: .55, fillOpacity: .06, dashArray: "3 4", interactive: false });
      ring.addTo(state.map); state.mapLayers.push(ring);
    }
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
  const inspPts = (m.inspection_route || []).map(id => byId[id]).filter(t => t && t.lat != null).map(t => [t.lat, t.lon]);
  if (inspPts.length > 1) {
    const il = L.polyline(inspPts, { color: "#d9a441", weight: 2, dashArray: "2 6", opacity: .85 })
      .bindTooltip("inspection route — REVIEW cards to check first (pending human approval)").addTo(state.map);
    state.mapLayers.push(il);
  }
  ((m.resurvey_plan || {}).lines || []).forEach(Lr => {
    const pts = [Lr.start, Lr.end];
    const line = L.polyline(pts, { color: "#b3a8ff", weight: 2.5, dashArray: "8 5", opacity: .95 })
      .bindPopup(`<b>${Lr.id}</b> · re-survey pass <i>(planned, not executed)</i><br/>heading ${Lr.heading_deg}° · ${Lr.length_m} m · ~${Lr.boat_min} min<br/>`
        + `targets on ${Lr.targets_on}: ${Lr.targets.join(", ")}<br/>uncertainty Σp(1−p) ${Lr.voi} · ${Lr.voi_per_100m}/100 m<br/>`
        + (Lr.predictions || []).slice(0, 3).map(p => `${p.id}: shadow was ${p.shadow_was.toFixed(0)}° → must point ${p.shadow_must_point.toFixed(0)}°`).join("<br/>"))
      .addTo(state.map);
    state.mapLayers.push(line); latlngs.push(Lr.start, Lr.end);
    const arrow = L.marker(Lr.end, { interactive: false, icon: L.divIcon({ className: "", iconSize: [14, 14], iconAnchor: [7, 7],
      html: `<div class="rs-arrow" style="transform:rotate(${Lr.heading_deg - 90}deg)">➤</div>` }) }).addTo(state.map);
    state.mapLayers.push(arrow);
  });
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
  const rank = Object.fromEntries((d.mission.review_queue || []).map((id, i) => [id, i]));
  rows.sort((a, b) => (VERDICTS.indexOf(a.verdict) - VERDICTS.indexOf(b.verdict)) || ((rank[a.oid] ?? 1e9) - (rank[b.oid] ?? 1e9)));
  $("#hazCount").textContent = `${rows.length} hazards · REVIEW ordered by P(pot) · LOW-RISK kept for audit`;
  const body = $("#hazBody"); body.innerHTML = "";
  rows.forEach(t => {
    const tr = document.createElement("tr"); tr.dataset.oid = t.oid;
    const coords = t.lat != null ? `${t.lat.toFixed(5)}, ${t.lon.toFixed(5)}` : `<span class="muted">no GPS</span>`;
    tr.innerHTML = `<td>${t.oid}${t.sightings > 1 ? ` <span class="muted" title="seen on ${t.sightings} frames/passes">×${t.sightings}</span>` : ""}</td><td>${t.cls_name}</td>
      <td><span class="v-tag" style="color:${VCOLOR[t.verdict]};background:${VCOLOR[t.verdict]}22">${VLABEL[t.verdict] || t.verdict}</span></td>
      <td>${t.conf}</td><td>${t.evidence_score}</td><td>${t.p_pot != null ? Math.round(t.p_pot * 100) + "%" : "—"}</td><td>${heightText(t.height_m, t.height_rel, t.height_rel ? 1 : 0)}</td>
      <td class="${t.shadow_quality === "clear" ? "chip-clear" : t.shadow_quality === "weak" ? "chip-weak" : "chip-none"}">${t.shadow_quality}</td>
      <td>${coords}</td><td>${t.geo_error_m != null ? "±" + t.geo_error_m + " m" : "—"}</td><td>${reviewCell(t)}</td>`;
    body.appendChild(tr);
  });
  $$("#hazBody .rev-btns button").forEach(b => b.onclick = async e => {
    const tr = e.target.closest("tr"), t = rows.find(x => x.oid === tr.dataset.oid);
    const ok = b.classList.contains("ok");
    const res = t && await sendLabel({ bbox: t.bbox, cls_name: t.cls_name, decision: ok ? "confirm" : "reject",
      verdict_before: t.verdict, conf: t.conf, p_pot: t.p_pot, source: "survey",
      frame_id: t.frame_id, frame_ref: (d.frame_refs || {})[t.frame_id] });
    tr.classList.add("rev-done");
    e.target.closest("td").innerHTML = `<span class="muted">${ok ? "✓ approved" : "✕ rejected"}${res ? " · label saved" : ""}</span>`;
  });
}
function reviewCell(t) {
  if (t.verdict !== "review") return `<span class="muted">—</span>`;
  return `<span class="rev-btns"><button class="ok" title="approve for recovery">✓</button><button class="no" title="dismiss">✕</button></span>`;
}

/* ------------------------------------------------------------------ human labels (every decision is a training label) */
async function sendLabel(rec) {
  const a = state._analyze || {};
  const body = Object.assign({ frame_id: a.frame_id, frame_ref: a.frame_ref }, rec);
  try {
    const r = await fetch(`${API}/api/feedback`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!r.ok) { toast(`label not saved (${r.status})`); setTimeout(toastHide, 1600); return null; }
    const j = await r.json(); showLabelStats(j.stats); return j;
  } catch { return null; }
}
function showLabelStats(st) {
  const el = $("#regLabels"); if (!el || !st) return;
  el.textContent = `${st.objects} · ✓${st.confirm} ✕${st.reject} ＋${st.missed} · ${st.frames} frames`;
}
async function loadLabelStats() { try { showLabelStats(await fetch(`${API}/api/feedback/stats`).then(r => r.json())); } catch { } }

/* "＋ missed pot": draw a box the detector never proposed → a positive label (fixes recall, not just precision) */
function toggleMissed(on) {
  state._drawMissed = on ?? !state._drawMissed;
  $("#missedBtn").classList.toggle("is-on", state._drawMissed);
  $("#viewer").classList.toggle("draw-missed", state._drawMissed);
  if (state._drawMissed) { cancelTour(); toast("drag a box around the missed pot"); setTimeout(toastHide, 1400); }
}
function wireMissedTool() {
  const btn = $("#missedBtn"); if (!btn) return;
  btn.onclick = () => { if (state._ready) toggleMissed(); };
  const cv = $("#viewerCanvas");
  let start = null, rect = null;
  cv.addEventListener("pointerdown", e => {
    if (!state._drawMissed || e.button !== 0) return;
    e.stopImmediatePropagation(); e.preventDefault();
    start = Viewer.toImage(e.clientX, e.clientY);
    rect = document.createElementNS(SVGNS, "g"); rect.setAttribute("class", "bx-missed");
    rect.innerHTML = `<rect x="${start[0]}" y="${start[1]}" width="1" height="1"/>`;
    $("#boxLayer").appendChild(rect); cv.setPointerCapture(e.pointerId);
  }, true);
  cv.addEventListener("pointermove", e => {
    if (!start || !rect) return;
    const p = Viewer.toImage(e.clientX, e.clientY), r = rect.firstChild;
    r.setAttribute("x", Math.min(start[0], p[0])); r.setAttribute("y", Math.min(start[1], p[1]));
    r.setAttribute("width", Math.abs(p[0] - start[0])); r.setAttribute("height", Math.abs(p[1] - start[1]));
  }, true);
  cv.addEventListener("pointerup", async e => {
    if (!start || !rect) return;
    const p = Viewer.toImage(e.clientX, e.clientY);
    const bbox = [Math.min(start[0], p[0]), Math.min(start[1], p[1]), Math.max(start[0], p[0]), Math.max(start[1], p[1])].map(Math.round);
    const g = rect; start = null; rect = null;
    if (bbox[2] - bbox[0] < 3 || bbox[3] - bbox[1] < 3) { g.remove(); return; }
    const cal = state.calibration || {};
    const res = await sendLabel({ bbox, cls_name: cal.guaranteed_class || "", decision: "missed", verdict_before: null, source: "analyze" });
    if (res) {
      const t = document.createElementNS(SVGNS, "text"); t.setAttribute("x", bbox[0] + 2); t.setAttribute("y", bbox[1] - 3);
      t.textContent = "missed pot · label saved"; g.appendChild(t);
      Log.push([{ tool: "human_label", msg: `missed pot marked at [${bbox.join(", ")}] → positive training label`, t: "" }]);
    } else g.remove();
    toggleMissed(false);
  }, true);
}

/* ------------------------------------------------------------------ STUDY: timed manual review vs DEPTH cards */
const Study = { plan: null, armIdx: 0, i: 0, t0: 0, timer: null, manual: [], cards: [], clicks: [] };
const EFF = { man: "#3987e5", list: "#d95926", depth: "#199e70" };     // validated categorical slots (dark)

function studyShow(id) { ["studyIntro", "studyManual", "studyCards", "studyBreak", "studyDone"].forEach(x => $("#" + x).hidden = x !== id); }
function studyTick(el) { clearInterval(Study.timer); Study.timer = setInterval(() => { $(el).textContent = ((performance.now() - Study.t0) / 1000).toFixed(1) + " s"; }, 100); }

function wireStudy() {
  $("#studyStartBtn").onclick = studyStart;
  $("#stNext").onclick = manualNext;
  $("#stUndo").onclick = () => { Study.clicks.pop(); drawMarks(); };
  $("#stYes").onclick = () => cardAnswer("real");
  $("#stNo").onclick = () => cardAnswer("not");
  $("#stContinue").onclick = () => startArm(1);
  $("#stFrame").addEventListener("click", e => {
    const im = e.currentTarget, r = im.getBoundingClientRect();
    Study.clicks.push([(e.clientX - r.left) * im.naturalWidth / r.width, (e.clientY - r.top) * im.naturalHeight / r.height]);
    drawMarks();
  });
  window.addEventListener("keydown", e => {
    if ($("#workspace").dataset.mode !== "study" || (e.target.tagName || "").toLowerCase() === "input") return;
    if (!$("#studyCards").hidden && (e.key === "y" || e.key === "Y")) { cardAnswer("real"); e.preventDefault(); }
    else if (!$("#studyCards").hidden && (e.key === "n" || e.key === "N")) { cardAnswer("not"); e.preventDefault(); }
    else if (!$("#studyManual").hidden && e.key === "Enter") { manualNext(); e.preventDefault(); }
    else if (!$("#studyManual").hidden && e.key === "Backspace") { Study.clicks.pop(); drawMarks(); e.preventDefault(); }
  });
  ["effSpf", "effSpc"].forEach(id => $("#" + id).addEventListener("input", () => {
    $("#effSpfV").textContent = $("#effSpf").value + " s"; $("#effSpcV").textContent = $("#effSpc").value + " s";
    clearTimeout(Study._eff); Study._eff = setTimeout(() => loadEffort(true), 120);
  }));
}

async function studyStart() {
  const btn = $("#studyStartBtn"); btn.disabled = true; toast("preparing your session — the agent is building its review cards …");
  try {
    const name = ($("#studyName").value || "anon").trim();
    Study.plan = await fetch(`${API}/api/study/plan?participant=${encodeURIComponent(name)}`).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); });
    Study.manual = []; Study.cards = [];
    $("#studyMeta").textContent = `session ${Study.plan.index + 1} · ${Study.plan.order.join(" → ")}`;
    startArm(0);
  } catch (e) { toast(`could not start (${e.message})`); setTimeout(toastHide, 1800); }
  finally { toastHide(); btn.disabled = false; }
}
function startArm(k) {
  Study.armIdx = k; Study.i = 0;
  const arm = Study.plan.order[k];
  if (arm === "manual") showManualFrame(); else showCard();
}
function armDone() {
  clearInterval(Study.timer);
  if (Study.armIdx === 0) {
    const next = Study.plan.order[1];
    $("#stBreakTitle").textContent = "Part 1 done";
    $("#stBreakTxt").innerHTML = next === "manual" ? "Next: <b>manual review</b> — click every pot on each raw frame." : "Next: <b>DEPTH cards</b> — answer Y (real pot) or N (not a pot) for each card.";
    studyShow("studyBreak");
  } else finishStudy();
}

function showManualFrame() {
  const fr = Study.plan.manual_frames[Study.i];
  if (!fr) return armDone();
  studyShow("studyManual"); Study.clicks = []; drawMarks();
  $("#stManProg").textContent = `frame ${Study.i + 1} / ${Study.plan.manual_frames.length}`;
  const im = $("#stFrame");
  im.onload = () => { Study.t0 = performance.now(); studyTick("#stManTimer"); drawMarks(); };
  im.src = `${API}/api/study/frame/${encodeURIComponent(fr.frame_id)}?t=${Date.now()}`;
}
function drawMarks() {
  const im = $("#stFrame"), svg = $("#stMarks");
  $("#stManClicks").textContent = `${Study.clicks.length} mark${Study.clicks.length === 1 ? "" : "s"}`;
  if (!im.naturalWidth) { svg.innerHTML = ""; return; }
  const r = im.getBoundingClientRect(), w = $("#stFrameWrap").getBoundingClientRect();
  svg.style.left = (r.left - w.left) + "px"; svg.style.top = (r.top - w.top) + "px";
  svg.setAttribute("width", r.width); svg.setAttribute("height", r.height);
  const sx = r.width / im.naturalWidth, sy = r.height / im.naturalHeight;
  svg.innerHTML = Study.clicks.map(([x, y]) => `<circle cx="${x * sx}" cy="${y * sy}" r="11"/>`).join("");
}
function manualNext() {
  const fr = Study.plan.manual_frames[Study.i]; if (!fr) return;
  Study.manual.push({ frame_id: fr.frame_id, ms: Math.round(performance.now() - Study.t0), clicks: Study.clicks.map(p => p.map(Math.round)) });
  Study.i++; showManualFrame();
}

function showCard() {
  const c = Study.plan.cards[Study.i];
  if (!c) return armDone();
  studyShow("studyCards");
  $("#stCardProg").textContent = `card ${Study.i + 1} / ${Study.plan.cards.length}`;
  $("#stCardMeta").textContent = `agent: P(pot) ${c.p_pot != null ? Math.round(c.p_pot * 100) + "%" : "—"} · detector ${c.conf.toFixed(2)} · ${c.verdict}`;
  const im = $("#stCard");
  im.onload = () => { Study.t0 = performance.now(); studyTick("#stCardTimer"); };
  im.src = c.crop_png;
}
function cardAnswer(decision) {
  const c = Study.plan && Study.plan.cards[Study.i];
  if (!c || $("#studyCards").hidden) return;
  Study.cards.push({ card_id: c.card_id, ms: Math.round(performance.now() - Study.t0), decision });
  Study.i++; showCard();
}

async function finishStudy() {
  clearInterval(Study.timer); toast("scoring against the labels …");
  try {
    const r = await fetch(`${API}/api/study/result`, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: Study.plan.session_id, manual: Study.manual, cards: Study.cards }) }).then(r => r.json());
    const m = r.session.manual, c = r.session.cards, pct = x => x == null ? "—" : Math.round(x * 100) + "%";
    $("#stDoneCard").innerHTML = `<h2>Thank you — session saved</h2>
      <div class="st-kv"><span class="h"></span><span class="h">manual</span><span class="h">DEPTH cards</span>
        <span>median time</span><span class="v">${m.sec_per_frame ?? "—"} s / frame</span><span class="v">${c.sec_per_card ?? "—"} s / card</span>
        <span>pots found (of the labelled pots)</span><span class="v">${m.found}/${m.pots} · ${pct(m.recall)}</span><span class="v">${c.confirmed_real}/${c.pots} · ${pct(c.recall)}</span>
        <span>false marks / false confirms</span><span class="v">${m.false_clicks}</span><span class="v">${c.confirmed_fake}</span>
        <span>total time</span><span class="v">${m.total_s} s</span><span class="v">${c.total_s} s</span></div>
      <p class="muted">DEPTH-card recall also counts pots the detector never proposed — it is bounded by the model, not by you. Pass the laptop to the next person: order and frame halves rotate automatically.</p>
      <button class="btn btn-primary" id="stAgain">New session</button>`;
    $("#stAgain").onclick = () => { studyShow("studyIntro"); $("#studyName").value = ""; $("#studyName").focus(); };
    studyShow("studyDone");
    renderStudySummary(r.summary); loadStudySummary();            // re-syncs sliders + curves to the measured timings
  } catch (e) { toast("could not save the session"); setTimeout(toastHide, 1800); }
  finally { toastHide(); }
}

async function loadStudySummary() {
  try {
    const j = await fetch(`${API}/api/study/summary`).then(r => r.json());
    renderStudySummary(j.summary);
    $("#effSpf").value = j.effort_params.sec_per_frame; $("#effSpc").value = j.effort_params.sec_per_card;
    $("#effSpfV").textContent = j.effort_params.sec_per_frame + " s"; $("#effSpcV").textContent = j.effort_params.sec_per_card + " s";
    Study._params = j.effort_params; loadEffort();
  } catch { }
}
function renderStudySummary(s) {
  const box = $("#stSummary"); if (!s || !s.sessions) { $("#stSumMeta").textContent = ""; return; }
  const pct = x => x == null ? "—" : Math.round(x * 100) + "%";
  $("#stSumMeta").textContent = `${s.sessions} session${s.sessions > 1 ? "s" : ""} · ${s.participants} people`;
  const ph = s.per_survey_hour_min || {};
  box.innerHTML = `<div class="st-kv"><span class="h"></span><span class="h">manual</span><span class="h">DEPTH</span>
    <span>median time</span><span class="v">${s.manual.sec_per_frame ?? "—"} s/frame</span><span class="v">${s.cards.sec_per_card ?? "—"} s/card</span>
    <span>recall of labelled pots</span><span class="v">${pct(s.manual.recall)}</span><span class="v">${pct(s.cards.recall)}</span>
    <span>min / survey-hour</span><span class="v">${ph.manual ?? "—"}</span><span class="v">${ph.depth ?? "—"}</span></div>
    <p class="panel-note" style="margin-top:8px">${s.speedup ? `Per frame of sonar: DEPTH cards took <b>${(s.cards.cards_per_frame * s.cards.sec_per_card).toFixed(1)} s</b> (${s.cards.cards_per_frame} cards × ${s.cards.sec_per_card} s) vs <b>${s.manual.sec_per_frame} s</b> by hand — <b>${s.speedup >= 1 ? s.speedup + "× faster" : (1 / s.speedup).toFixed(2) + "× slower"}</b>. ` : ""}Reviewers confirmed ${pct(s.cards.accuracy_on_real)} of the real pots on their cards. ${escapeHtml(s.caveat || "")}.</p>`;
}

async function loadEffort(fromSliders) {
  const q = fromSliders ? `?sec_per_frame=${$("#effSpf").value}&sec_per_card=${$("#effSpc").value}` : "";
  let j; try { j = await fetch(`${API}/api/effort${q}`).then(r => { if (!r.ok) throw 0; return r.json(); }); } catch { $("#effChart").innerHTML = `<p class="empty-hint">No effort data for this model yet (python -m src.agentic.effort).</p>`; return; }
  const C = j.curves, W = 300, H = 170, P = { l: 30, r: 8, t: 8, b: 22 };
  $("#effProv").textContent = j.provenance.startsWith("measured") ? "measured" : j.provenance.startsWith("set") ? "what-if" : "assumed";
  const xmax = Math.max(...[C.manual, C.detector_list, C.depth].map(c => c[0][c[0].length - 1] || 1));
  const X = x => P.l + (x / xmax) * (W - P.l - P.r), Y = y => H - P.b - y * (H - P.t - P.b);
  const pl = (c, col, dash) => `<polyline fill="none" stroke="${col}" stroke-width="${dash ? 1.2 : 2}" ${dash ? 'stroke-dasharray="3 3"' : ""} points="${c[0].map((x, i) => X(x).toFixed(1) + "," + Y(c[1][i]).toFixed(1)).join(" ")}"/>`;
  const R = C.promise || 0, mt = C.minutes_to_promise;
  const grid = [0.25, 0.5, 0.75, 1].map(v => `<line class="gr" x1="${P.l}" x2="${W - P.r}" y1="${Y(v)}" y2="${Y(v)}"/><text x="2" y="${Y(v) + 3}">${Math.round(v * 100)}%</text>`).join("");
  const mk = (m, col) => m == null ? "" : `<circle cx="${X(m)}" cy="${Y(R)}" r="4.5" fill="${col}" stroke="#10161d" stroke-width="2"/>`;
  $("#effChart").innerHTML = `<svg class="eff-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="share of pots found versus analyst minutes">
    ${grid}<line class="ax" x1="${P.l}" x2="${W - P.r}" y1="${H - P.b}" y2="${H - P.b}"/>
    <line x1="${P.l}" x2="${W - P.r}" y1="${Y(R)}" y2="${Y(R)}" stroke="#5a6874" stroke-dasharray="2 3"/>
    <text x="${P.l + 3}" y="${Y(R) - 3}">promise ≥ ${Math.round(R * 100)}%</text>
    ${pl(C.detector_list, EFF.list)}${pl(C.manual, EFF.man)}${pl(C.forecast, EFF.depth, true)}${pl(C.depth, EFF.depth)}
    ${mk(mt.manual, EFF.man)}${mk(mt.depth, EFF.depth)}
    <text x="${W - P.r}" y="${H - 6}" text-anchor="end">analyst minutes (${j.frames} frames)</text></svg>
    <div class="eff-leg"><span><i style="background:${EFF.man}"></i>manual</span><span><i style="background:${EFF.list}"></i>detector list</span><span><i style="background:${EFF.depth}"></i>DEPTH queue</span><span><i style="background:none;border-top:1px dashed ${EFF.depth}"></i>DEPTH forecast</span></div>`;
  const ph = C.per_survey_hour_min || {};
  $("#effStats").innerHTML = `At the promise (≥ ${Math.round(R * 100)}%): manual <b>${mt.manual != null ? mt.manual.toFixed(1) : "—"} min</b>, DEPTH <b>${mt.depth != null ? mt.depth.toFixed(1) : "never"}</b>${mt.depth != null ? ` min (${C.cards_to_promise} cards)` : " — card accuracy too low for the promise"} · per survey-hour ${ph.manual ?? "—"} vs ${ph.depth ?? "—"} min.<br/>
    <b>Break-even: ${C.breakeven_sec_per_card ?? "—"} s per card</b> — faster card review than this and DEPTH wins.<br/>
    Forecast: the agent expected <b>${C.forecast_at_promise ?? "—"}</b> real pots at that point; <b>${C.actual_at_promise ?? "—"}</b> were real.`;
}

/* ------------------------------------------------------------------ AUDIT: blinded false-alarm review */
const Aud = { items: [], i: 0, tags: {}, name: "" };
function wireAudit() {
  $("#audStart").onclick = audStart;
  $("#audBack").onclick = () => { if (Aud.i > 0) { Aud.i--; audShow(); } };
  $$(".aud-answer .btn").forEach(b => b.onclick = () => audTag(b.dataset.tag));
  window.addEventListener("keydown", e => {
    if ($("#workspace").dataset.mode !== "audit" || $("#audTask").hidden || (e.target.tagName || "").toLowerCase() === "input") return;
    const k = { "1": "real", "2": "clutter", "3": "noise", "4": "unsure" }[e.key];
    if (k) { audTag(k); e.preventDefault(); }
    else if (e.key === "ArrowLeft" && Aud.i > 0) { Aud.i--; audShow(); e.preventDefault(); }
  });
}
async function audStart() {
  Aud.name = ($("#audName").value || "anon").trim();
  const j = await fetch(`${API}/api/audit/items?annotator=${encodeURIComponent(Aud.name)}`).then(r => r.json());
  Aud.items = j.items || []; Aud.tags = j.tagged || {};
  if (!Aud.items.length) { toast("no audit set built (python -m src.detection.fp_audit build)"); setTimeout(toastHide, 2000); return; }
  Aud.i = Math.max(0, Aud.items.findIndex(it => !Aud.tags[it.id]));
  if (Aud.items.every(it => Aud.tags[it.id])) Aud.i = 0;
  ["audIntro", "audDone"].forEach(x => $("#" + x).hidden = true); $("#audTask").hidden = false;
  audShow();
}
function audShow() {
  const it = Aud.items[Aud.i];
  if (!it) { $("#audTask").hidden = true; $("#audDone").hidden = false; loadAuditSummary(); return; }
  $("#audCrop").src = it.crop; $("#audCtx").src = it.context;
  const n = Object.keys(Aud.tags).filter(k => Aud.items.some(x => x.id === k)).length;
  $("#audProg").textContent = `item ${Aud.i + 1} / ${Aud.items.length} · ${n} tagged`;
  $("#audMeta").textContent = `${Aud.name}`;
  $$(".aud-answer .btn").forEach(b => b.classList.toggle("on", Aud.tags[it.id] === b.dataset.tag));
}
async function audTag(tag) {
  const it = Aud.items[Aud.i]; if (!it) return;
  const r = await fetch(`${API}/api/audit/tag`, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ item: it.id, tag, annotator: Aud.name }) });
  if (!r.ok) { toast("tag not saved"); setTimeout(toastHide, 1200); return; }
  Aud.tags[it.id] = tag; Aud.i++; audShow();
  if (Aud.i % 10 === 0) loadAuditSummary();
}
async function loadAuditSummary() {
  let s; try { s = await fetch(`${API}/api/audit/summary`).then(r => r.json()); } catch { return; }
  const box = $("#audSummary");
  if (!s.built) { box.innerHTML = `<p class="empty-hint">No audit set built for this model.</p>`; return; }
  const pct = x => x == null ? "—" : Math.round(x * 100) + "%";
  $("#audSumMeta").textContent = `${(s.annotators || []).length} auditor${(s.annotators || []).length === 1 ? "" : "s"}`;
  const tax = s.fp_taxonomy || {}, tot = Object.values(tax).reduce((a, b) => a + b, 0) || 1;
  const bar = (k, lbl) => `<div class="aud-bar"><span>${lbl}</span><span class="track"><span class="fill" style="width:${100 * (tax[k] || 0) / tot}%"></span></span><span class="mono">${tax[k] || 0}</span></div>`;
  const per = Object.entries(s.per_annotator || {}).map(([a, p]) => `${escapeHtml(a)}: ${p.tagged} tagged · catch ${p.catch_real}/${p.catch_tagged} real`).join("<br/>");
  box.innerHTML = `<p class="panel-note">Band: every false alarm with confidence ≥ ${s.conf_cut} on the calibration recordings (${s.band_fp} false alarms, ${s.band_tp} true pots). Raw precision <b>${pct(s.band_precision_raw)}</b>.</p>
    ${bar("real", "real object")}${bar("clutter", "clutter / natural")}${bar("noise", "noise / artefact")}${bar("unsure", "unsure")}
    <p class="panel-note" style="margin-top:8px">${s.fp_tagged}/${s.band_fp} false alarms tagged · real objects ${pct(s.fp_real_share)}${s.fp_real_share_ci95 && s.fp_real_share_ci95[0] != null ? ` (95% CI ${pct(s.fp_real_share_ci95[0])}–${pct(s.fp_real_share_ci95[1])})` : ""}.<br/>
    Audited precision: <b>${s.band_precision_audited != null ? pct(s.band_precision_audited) : "after every item is tagged"}</b>${s.kappa_first_two != null ? ` · agreement κ ${s.kappa_first_two}` : ""}.</p>
    ${per ? `<p class="panel-note">${per}</p>` : ""}
    <p class="panel-note">A result counts only if the auditor recognised the catch trials (known pots).</p>`;
}

/* ------------------------------------------------------------------ utils */
function fmt(x) { return x == null ? "—" : (Math.round(x * 10) / 10).toFixed(1); }
function cShort(c) { return `C${c.confirmed || 0} R${c.review || 0} L${c.low_risk || 0}`; }

/* Analyst effort: cumulative expected real pots vs review minutes, REVIEW queue in P(pot) order,
   with the budget line. Seconds-per-card is ASSUMED until the timed user study replaces it. */
function renderEffort(d) {
  const box = $("#effortPanel"), meta = $("#effortMeta"), m = d.mission, b = m.budget || {};
  const byId = Object.fromEntries(d.tracked.map(t => [t.oid, t]));
  const q = (m.review_queue || []).map(id => byId[id]).filter(Boolean);
  const conf = (m.counts || {}).confirmed || 0;
  if (!q.length) { box.innerHTML = `<p class="empty-hint">No REVIEW cards in this survey — ${conf} auto-confirmed under the precision promise.</p>`; meta.textContent = ""; return; }
  const spc = b.sec_per_card || 8, W = 300, H = 120, P = 26;
  const mins = [0], pots = [0];
  q.forEach((t, i) => { mins.push((i + 1) * spc / 60); pots.push(pots[i] + (t.p_pot ?? 0)); });
  const X = x => P + (W - P - 6) * x / Math.max(mins[mins.length - 1], b.minutes || 0, 1e-6);
  const Y = y => H - 18 - (H - 30) * y / Math.max(pots[pots.length - 1], 1e-6);
  const path = mins.map((x, i) => `${i ? "L" : "M"}${X(x).toFixed(1)},${Y(pots[i]).toFixed(1)}`).join("");
  const bx = b.minutes != null ? X(b.minutes) : null;
  meta.textContent = `${spc}s/card · ${b.sec_per_card_source && b.sec_per_card_source.startsWith("ASSUMED") ? "assumed" : "measured"}`;
  box.innerHTML = `
    <svg class="effort-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="expected real pots found versus review minutes">
      <line x1="${P}" y1="${H - 18}" x2="${W - 4}" y2="${H - 18}" class="ax"/><line x1="${P}" y1="8" x2="${P}" y2="${H - 18}" class="ax"/>
      ${bx != null ? `<rect x="${P}" y="8" width="${Math.max(0, bx - P).toFixed(1)}" height="${H - 26}" class="budget-fill"/><line x1="${bx.toFixed(1)}" y1="8" x2="${bx.toFixed(1)}" y2="${H - 18}" class="budget-line"/>` : ""}
      <path d="${path}" class="curve"/>
      <text x="${P}" y="${H - 4}" class="lbl">0</text><text x="${W - 6}" y="${H - 4}" class="lbl" text-anchor="end">${mins[mins.length - 1].toFixed(1)} min</text>
      <text x="${P - 4}" y="${Y(pots[pots.length - 1]) + 3}" class="lbl" text-anchor="end">${pots[pots.length - 1].toFixed(1)}</text>
    </svg>
    <div class="effort-stats">
      <div><b>${conf}</b><span>auto-confirmed (no card)</span></div>
      <div><b>${b.cards_affordable ?? "—"}/${q.length}</b><span>cards in ${b.minutes ?? "—"} min</span></div>
      <div><b>~${b.expected_pots_in_budget ?? "—"}</b><span>expected real pots in budget</span></div>
      <div><b>${b.share_of_expected_pots != null ? Math.round(b.share_of_expected_pots * 100) + "%" : "—"}</b><span>of the queue's expected pots</span></div>
    </div>
    <p class="panel-note">Curve = cumulative Σ P(pot) reviewing cards most-likely-first. ${b.sec_per_card_source ? escapeHtml(b.sec_per_card_source) + "." : ""}</p>`;
}
function escapeHtml(s) { return String(s).replace(/[&<>"']/g, m => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m])); }
