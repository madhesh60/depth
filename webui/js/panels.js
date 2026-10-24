/* DEPTH studio — modes, health, metrics, the calibrated promises, the sample gallery and the inputs.
   Zero-build: index.html loads webui/js/*.js in order as classic scripts sharing one global scope;
   the boot runs on DOMContentLoaded (core.js), after every file has loaded. API: src/dashboard/routes/. */
"use strict";

function setMode(mode) { const b = $(`.mode-btn[data-mode="${mode}"]`); if (b && !b.classList.contains("is-active")) b.click(); }

/* calm panels: a panel's explanation folds behind a small (i); collapsible sections open on click */
function tidyPanels() {
  document.querySelectorAll(".panel").forEach(panel => {
    const head = panel.querySelector(":scope > .panel-head"), title = head && head.querySelector(".panel-title");
    const notes = [...panel.querySelectorAll(":scope > .panel-note:not([id])")];
    if (title && notes.length) {
      notes.forEach(n => n.classList.add("tucked"));
      const b = document.createElement("button");
      b.type = "button"; b.className = "info-btn"; b.textContent = "i";
      b.setAttribute("aria-label", "about " + title.textContent); b.setAttribute("aria-expanded", "false");
      b.onclick = e => { e.stopPropagation(); const open = panel.classList.toggle("notes-open"); b.setAttribute("aria-expanded", String(open)); };
      title.after(b);
    }
    if (panel.classList.contains("collapsible") && head) {
      head.setAttribute("role", "button"); head.tabIndex = 0;
      const toggle = () => panel.classList.toggle("is-collapsed");
      head.onclick = toggle;
      head.onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); } };
    }
  });
}

function setSys(state, text) {
  const b = $("#sysBtn"); if (!b) return;
  b.dataset.state = state; $("#sysTxt").textContent = text;
}

async function loadHealth() {
  try {
    const h = await fetch(`${API}/api/health`).then(r => r.json());
    const cv = $("#pillCv"), md = $("#pillModel");
    if (h.recording && h.recording.available && $("#gpsRecording")) $("#gpsRecording").hidden = false;
    if ($("#apprPin")) {
      $("#apprPin").hidden = !h.approver_pin_required; $("#apprOpen").hidden = !!h.approver_pin_required;
      const hint = $("#apprHint");                    // where a person finds the PIN (never the PIN itself)
      if (hint) { hint.textContent = h.approver_pin_hint || ""; hint.hidden = !(h.approver_pin_required && h.approver_pin_hint); }
      if (h.approver_pin_hint) $("#apprPin").title = h.approver_pin_hint;
    }
    const gr = $("#goalRecall");
    if (gr && h.modes && gr.options.length === 1) {     // short options fit a narrow rail; the cost is in the tooltip
      h.modes.forEach(m => gr.add(new Option(`≥ ${pctS(m.promise)} · ${m.label}`, String(m.promise))));
      gr.title = "Agent decides = the strongest promise the modes and the compute budget allow. " + h.modes.map(m =>
        `${m.label}: ≥ ${pctS(m.promise)}, ${m.passes} detector pass${m.passes > 1 ? "es" : ""} per frame`).join(" · ");
    }
    cv.textContent = `OpenCV ${h.opencv}`;
    cv.className = "pill " + (String(h.opencv).startsWith("5") ? "pill-ok" : "pill");
    md.textContent = h.model_loaded ? "Ready" : h.model_error ? "Error" : "Warming up";
    md.className = "pill " + (h.model_loaded ? "pill-ok" : h.model_error ? "pill-bad" : "pill");
    md.title = h.model_error || (h.warmup_ms != null ? `warm-up ${h.warmup_ms} ms` : "");
    if (h.is_cool_path) { cv.textContent += " · COOL"; cv.title = h.cv2_file; }
    const rc = $("#regCv"), rm = $("#regModel");
    const rt = h.runtime || {}, ch = rt.engine_choice;   // the cv2.dnn engine; `fastest` measured it at load
    if (rc) {
      rc.textContent = h.opencv + (h.is_cool_path ? " (COOL /opt/cool)" : "") + (rt.engine ? ` · dnn ${rt.engine}` : "");
      rc.title = ch ? `cv2.dnn engine measured on this machine at load: ${ch.picked} (${ch.why}). ms per forward, ` +
        `${ch.round_ms.new.length} rounds: new ${ch.round_ms.new.join(" / ")} · classic ${ch.round_ms.classic.join(" / ")}`
        : rt.engine ? `cv2.dnn engine: ${rt.engine} (DEPTH_DNN_ENGINE=${rt.engine_requested || "auto"})` : "";
    }
    if (rm) rm.textContent = h.model_loaded ? "Loaded and warm" : "Loading";
    state.calibration = h.calibration || null;
    const cal = h.calibration || {};
    if ($("#regWeights")) $("#regWeights").textContent = `${cal.model || h.model} · ${cal.imgsz || "?"} px ONNX`;
    if ($("#regBrief")) $("#regBrief").textContent = `${cal.model || h.model || ""}`;
    if ($("#regClasses")) $("#regClasses").textContent = `${(cal.names || []).length} · ${(cal.names || []).map(human).join(", ")}`;
    if ($("#regGateK")) $("#regGateK").textContent = `Gate · ${human(cal.guaranteed_class || "?")}`;
    state.modes = h.modes || [];
    renderGuarantees(h.calibration);
    setSys(h.model_loaded ? "ok" : h.model_error ? "bad" : "busy",
           h.model_loaded ? (h.is_cool_path ? "Ready · COOL" : "Ready") : h.model_error ? "Model error" : "Warming up");
    if (!h.model_loaded && !h.model_error) setTimeout(loadHealth, 1500);   // re-poll until warm
  } catch (e) {
    $("#pillCv").textContent = "backend offline";
    $("#pillCv").className = "pill pill-bad";
    setSys("bad", "Offline");
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
    panel.innerHTML = `<p class="panel-note">No calibrated promise for this model — every find goes to a person (python -m src.agentic.calibrate).</p>`;
    if (tiers) tiers.textContent = "uncalibrated"; if (gate) gate.textContent = "—";
    return;
  }
  const g = c.guarantees || {}, v = g.verified_on_test || {};
  const pct = x => x == null ? "—" : `${Math.round(x * 100)}%`;
  const held = ok => ok == null ? "" : ok ? `<span class="g-held">held on test</span>` : `<span class="g-miss">NOT held on test</span>`;
  // the rail has room for one short line: the one-look promise and, when calibrated, the closer look's
  const lh = (state.modes || []).find(m => m.name !== "standard" && m.promise > (g.recall_promise || 0));
  meta.textContent = `≥ ${pct(g.recall_promise)}${lh ? ` / ${pct(lh.promise)}` : ""}${v.recall_promise_held === true ? " · held" : ""}`;
  meta.title = `at least ${pct(g.recall_promise)} of real pots reach a person with one look`
    + (lh ? `, ${pct(lh.promise)} when the agent looks harder` : "") + (v.recall_promise_held === true ? " — held on test" : "");
  if (gate) gate.textContent = `${(c.detector_floor ?? c.tau_review).toFixed(2)} (floor) · review ≥ ${c.tau_review.toFixed(2)}`;
  if (tiers) tiers.textContent = `${c.method ? "conformal (CP · LTT)" : "calibrated"}${c.policy ? " · " + c.policy : ""}`;
  panel.innerHTML = `
    <div class="g-row" title="${g.requested_recall_achievable === false ? `The requested 90% is not achievable with this detector (proposal ceiling ${pct(g.recall_ceiling)}).` : ""}"><span class="g-badge g-recall">≥ ${pct(g.recall_promise)}</span>
      <span class="g-txt">of real pots reach a person<br>${held(v.recall_promise_held)}</span></div>
    ${(state.modes || []).filter(m => m.name !== "standard").map(m => `<div class="g-row" title="${escapeHtml(m.label)}: ${m.passes} cv2.dnn passes per frame (STUDY-18). The agent uses it when the goal and the compute budget allow."><span class="g-badge g-recall">≥ ${pct(m.promise)}</span>
      <span class="g-txt">with the closer look (${m.passes} passes / frame)<br>${held((m.verified || {}).promise_held)}${(m.verified || {}).recall != null ? ` <span class="muted">${(m.verified.recall * 100).toFixed(1)}%</span>` : ""}</span></div>`).join("")}
    <div class="g-row"><span class="g-badge g-prec">${g.precision_promise ? "≥ " + pct(g.precision_promise) : "none"}</span>
      <span class="g-txt">${g.precision_promise ? `of confirmed finds are real<br>${held(v.precision_promise_held)}` : "no auto-confirm: every find is checked by a person"}</span></div>
    <p class="panel-note">95% confidence · fit on validation, verified once on test · <a href="https://github.com/madhesh60/depth/blob/main/${escapeHtml(g.report || "docs/")}" target="_blank" rel="noopener">report</a></p>`;
}

async function loadSamples() {
  const grid = $("#sampleGrid");
  try {
    const j = await fetch(`${API}/api/samples`).then(r => r.json());
    state.samples = j.samples || [];
  } catch { state.samples = []; }
  const recs = new Set(state.samples.map(x => (String(x.name || "").match(/^Rec\d+/) || [""])[0]));
  const cnt = $("#sampleCount");
  if (cnt) cnt.textContent = state.samples.length ? (recs.size === 1 && !recs.has("") ? `${[...recs][0]} · ` : "") + `${state.samples.length} frames` : "";
  if (!state.samples.length) {
    grid.innerHTML = `<p class="empty-hint" style="grid-column:1/-1">No local samples — drop your own sonar frame below.</p>`;
    return;
  }
  grid.innerHTML = "";
  state.samples.forEach(s => {
    const card = document.createElement("button");
    card.className = "sample-card"; card.dataset.id = s.id;
    const parts = String(s.kind || "").split(" · ");   // "crab-pot sonogram · 6 labelled pots"
    const thumb = document.createElement("div");
    thumb.className = "sc-thumb";
    const img = document.createElement("img");
    img.alt = s.name; img.loading = "lazy"; img.decoding = "async";
    img.src = `${API}/api/sample_thumb/${encodeURIComponent(s.id)}`;
    img.onerror = () => img.replaceWith(sonarGlyph());
    thumb.appendChild(img);
    if (parts[1]) {                                    // pot count → a quiet badge, not a third wrapping line
      const badge = document.createElement("span");
      badge.className = "sc-count";
      badge.textContent = parts[1].replace("labelled ", "");
      thumb.appendChild(badge);
    }
    const meta = document.createElement("div");
    meta.className = "sc-meta";
    const short = String(s.name || "").replace(/^Rec\d+\s+/, "").replace(/^\w/, ch => ch.toUpperCase());   // "Rec6 starboard #24" → "Starboard #24"
    card.title = s.name;
    meta.innerHTML = `<div class="sc-name">${escapeHtml(short || s.name)}</div><div class="sc-kind">${escapeHtml(parts[0] || s.kind)}</div>`;
    card.append(thumb, meta);
    card.onclick = () => selectSample(s.id, card);
    grid.appendChild(card);
  });
}

function sonarGlyph() {
  const d = document.createElement("div");
  d.style.cssText = "aspect-ratio:1;display:grid;place-items:center;background:radial-gradient(circle at 50% 35%,#0d1013,#050607);color:#9ecbff";
  d.innerHTML = `<svg viewBox="0 0 24 24" width="28" height="28" opacity=".7"><path d="M12 3a9 9 0 1 0 9 9h-9V3Z" fill="none" stroke="currentColor" stroke-width="1.4"/><circle cx="12" cy="12" r="2" fill="currentColor"/></svg>`;
  return d;
}

function selectSample(id, card) {
  state.selected = { id };
  $("#nadirField").hidden = true;               // a sample's sonar edge comes from its source rule
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
    $("#workspace").dataset.mode = b.dataset.mode; document.body.dataset.mode = b.dataset.mode;
    Pop.closeAll(); Tabs.render();
    if (b.dataset.mode === "survey" && state.map) setTimeout(refitIfLost, 80);
    if (b.dataset.mode === "study") { loadStudySummary(); }
    if (b.dataset.mode === "audit") { loadAuditSummary(); }
    if (b.dataset.mode === "connect") { Connect.load(); }
    if (b.dataset.mode === "judges") { Judges.load(); }
  });
  const m0 = new URLSearchParams(location.search).get("mode");   // deep link: ?mode=survey
  if (m0) setTimeout(() => setMode(m0), 0);
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
  gate.oninput = () => { state.gate = parseFloat(gate.value); $("#confGateVal").textContent = state.gate.toFixed(2); applyFilters();
    const bv = $("#gateBtnVal"); if (bv) bv.textContent = state.gate.toFixed(2); };
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
  $("#nadirField").hidden = false;              // an upload's sonar edge is the person's to say (never guessed)
  $("#nadirSel").value = "";                     // a new frame: no answer carried over from the last one
  $("#analyzeBtn").disabled = false;
  $("#analyzeHint").textContent = `Uploaded ${file.name}. Pick where the sonar is (above), so range and shadow can be measured; then run.`;
}
