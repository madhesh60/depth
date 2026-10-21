/* DEPTH studio — Analyze: one frame, its evidence cards, the agent's-eye view and the auto-tour.
   Zero-build: index.html loads webui/js/*.js in order as classic scripts sharing one global scope;
   the boot runs on DOMContentLoaded (core.js), after every file has loaded. API: src/dashboard/routes/. */
"use strict";

/* ------------------------------------------------------------------ ANALYZE */
async function runAnalyze(simulate) {
  if (!state.selected) return;
  simulate = typeof simulate === "string" ? simulate : "";
  $("#analyzeBtn").disabled = true;
  $("#analyzePlaceholder").hidden = true;
  Busy.start("Analyzing frame");
  Busy.set(null, Active.on ? "Detect → choose OpenCV tools → decide" : "Detect → OpenCV evidence → decide");
  stepperRun(["stage1", "see", "prove", "decide"]);
  Log.reset(); Log.now({ stage: true, tool: "SEE", msg: "perceive — full-frame YOLO11 via cv2.dnn …", t: "run" });

  try {
    let url = `${API}/api/analyze`, opts = { method: "POST" };
    if (state.selected.file) { const fd = new FormData(); fd.append("file", state.selected.file); opts.body = fd; }
    else url += `?sample=${encodeURIComponent(state.selected.id)}`;
    if (Active.on) url += `${url.includes("?") ? "&" : "?"}active=1`;
    if (simulate) url += `${url.includes("?") ? "&" : "?"}simulate=${simulate}`;
    const d = await apiJSON(url, opts);
    state._analyze = d;
    renderAnalyze(d);
    showProvenance(d.provenance);
    if (Twin.dim === "3d") showFrameTwin();
    loadMetrics();
    stepperFinish(["stage1", "see", "prove", "decide"]);
  } catch (e) {
    stepperReset();
    const inc = e.incident;
    if (inc) Notices.show([inc], "analyze" + Date.now());
    Log.now({ stage: true, tool: "FAILED", msg: inc ? `${inc.title} → ${inc.fallback}` : e.message, t: "" });
    $("#analyzePlaceholder").hidden = false;
    $("#analyzePlaceholder").querySelector("h3").textContent = inc ? inc.title : "Analyze failed";
    $("#analyzePlaceholder").querySelector("p").innerHTML = inc
      ? `${escapeHtml(inc.message)}<br/><span class="muted">${escapeHtml(inc.fallback)}.</span>`
      : `${escapeHtml(e.message)}<br/><span class="muted">Is the backend running and the model present?</span>`;
  } finally { Busy.done(); $("#analyzeBtn").disabled = false; }
}

function renderAnalyze(d) {
  cancelTour(); hideAgentEye(); state._tourOptOut = false; state._ready = true;
  $("#analyzePlaceholder").hidden = true;
  $$(".seg-btn").forEach(x => x.classList.toggle("is-active", x.dataset.view === "overlay"));
  $("#viewer").classList.remove("boxes-off");
  const s1 = d.stage1 || {};
  $("#stageCap").textContent = `${d.frame_id} · ${d.width}×${d.height}px · nadir=${d.nadir}${d.orientation && d.orientation.source ? ` (${d.orientation.source})` : ""}`
    + ` · seabed ${s1.measured ? `tracked @ ${s1.altitude_px} px` : "not measured"} · ${d.candidates.length} candidate(s)`;
  $("#verdictCounts").innerHTML = VERDICTS.map(v => `<span class="vc vc-${v}"><span class="n">${d.counts[v] || 0}</span>${VLABEL[v] || human(v)}</span>`).join("");
  const sm = d.stage_ms || {};
  $("#latency").textContent = `⏱ ${d.wall_ms}ms · stage1 ${fmt(sm.stage1)} · see ${fmt(sm.see)}`
    + (sm.look_harder > 1 ? ` · look harder ${fmt(sm.look_harder)}` : "") + ` · prove+decide ${fmt(sm.prove_decide)}`;
  const dl = $("#dockLatency"); if (dl) dl.textContent = `wall ${d.wall_ms}ms`;

  state._boxes = d.candidates.map((c, i) => ({ id: i, bbox: c.bbox, verdict: c.verdict, conf: c.conf, cand: c }));
  state._cursor = -1;
  Viewer.render(d, state._boxes);
  $("#viewerEmpty").hidden = d.candidates.length > 0;

  const ph = $("#analyzePlaceholder"); ph.querySelector("h3").textContent = "Choose a sonar frame";
  const ac = d.agent_choice, pc = d.perception || {}, fp = $("#framePerception");
  if (fp) {
    fp.hidden = !ac || !ac.mode;
    if (ac && ac.mode) fp.innerHTML = `<b>${escapeHtml(ac.mode.label)}</b> · promise ≥ ${pctS(ac.mode.promise)} — ${escapeHtml(chainText({ rationale: ac.why }))}`
      + (pc.passes ? ` <span class="muted">· the closer look added ${pc.new || 0}, raised ${pc.raised || 0}</span>` : "");
  }
  const incs = d.incidents || [];
  $("#frameIncidents").innerHTML = incs.map(incidentRow).join("");
  if (incs.some(i => i.severity !== "info" || i.simulated)) Notices.show(incs, d.frame_id);
  $("#evMeta").textContent = d.candidates.length ? `${d.candidates.length} find${d.candidates.length > 1 ? "s" : ""}` : "";
  const grid = $("#evidenceGrid"); grid.innerHTML = "";
  if (!d.candidates.length) {
    grid.innerHTML = `<p class="empty-hint">Nothing to prove on this frame. If you can see an object, use <b>Mark missed</b>; it becomes a training label.</p>`;
    buildClassLegend([]); applyFilters(); streamLog(d, []); return;
  }
  const order = { confirmed: 0, review: 1, low_risk: 2, rejected: 2 };
  const ordered = state._boxes.slice().sort((a, b) =>
    (order[a.verdict] - order[b.verdict]) || ((b.cand.evidence?.evidence_score || 0) - (a.cand.evidence?.evidence_score || 0)));
  ordered.forEach(b => grid.appendChild(evidenceCard(b.cand, b.id)));
  const first = grid.querySelector(".ev-card"); if (first) first.classList.add("is-linked");   // the strongest find opens
  buildClassLegend(ordered);
  applyFilters();
  streamLog(d, ordered);
  // no automatic zoom tour after Analyze: it surprised first-time users (audit 2026-10-13 §4.7); the Tour button plays it
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
    chip.innerHTML = `<span class="sw" style="background:${CLASS_SW[i % CLASS_SW.length]}"></span>${escapeHtml(human(nm))}<span class="cnt">${counts[nm]}</span>`;
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
    { stage: true, tool: "SEE", msg: `perceive → ${d.candidates.length} candidate(s)`
        + (sm.look_harder > 1 ? ` (incl. the 2×2 tile look, ${fmt(sm.look_harder)}ms)` : ""),
      t: fmt((sm.see || 0) + (sm.look_harder || 0)) + "ms" }];
  ordered.forEach(b => {
    const c = b.cand;
    entries.push({ stage: true, tool: `cand #${b.id}`, msg: `${c.cls_name} · det ${c.conf.toFixed(2)}`, t: "" });
    (c.trace || []).forEach(s => entries.push({ tool: (s.status && s.status !== "done" ? s.status + " · " : "") + s.tool, msg: s.rationale, t: fmt(s.latency_ms) + "ms" }));
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
  if (d.mode !== "calibrated" && d.mode !== "active") return "Legacy rule (no calibrated promise)";
  if (c.verdict === "confirmed") return `Confirmed tier: at least ${Math.round((g.precision_promise || 0) * 100)}% are real (95% confidence)`;
  if (c.verdict === "review") return `Review and confirmed together reach at least ${Math.round((g.recall_promise || 0) * 100)}% of pots (95% confidence)`;
  if (((state.calibration || {}).non_hazard_classes || []).includes(c.cls_name)) return "Natural seabed class: not a hazard, kept for audit";
  return `Low-risk tier holds at most ${Math.round((1 - (g.recall_promise || 0)) * 100)}% of pots; kept for audit, never deleted`;
}
function verdictReason(c) {
  const p = (c.evidence || {}).p_pot;
  if (c.verdict === "confirmed") return "Above the confirm threshold: carries the precision promise";
  if (c.verdict === "review") return `Human review${p != null ? ` · P(pot) ~${Math.round(p * 100)}%` : ""}, ordered in the queue`;
  return "Low risk: below the review threshold, kept for audit, not surfaced";
}

function selectCandidate(id, fromTwin) {
  state._cursor = id;
  if (!fromTwin && Twin.dim === "3d" && Twin.frame) Twin.frame.select(id, false);
  Viewer.focusBox(state._boxes.find(b => b.id === id).bbox);
  Viewer.pulse(id);
  $$(".ev-card").forEach(c => c.classList.toggle("is-linked", +c.dataset.id === id));
  const card = $(`.ev-card[data-id="${id}"]`);
  if (card) {                                   // scroll ONLY the inspector (scrollIntoView also scrolls the page)
    const box = card.closest(".scrollable");
    if (box) box.scrollTo({ top: card.getBoundingClientRect().top - box.getBoundingClientRect().top + box.scrollTop - 12, behavior: "smooth" });
  }
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
  const how = [rv && rv.scale ? rv.scale.toFixed(1) + "× · CLAHE" : "",
    !relooked(c) ? "view only, no re-look" : rl.found ? `re-fired ${(rl.conf || 0).toFixed(2)}` : "no re-fire"].filter(Boolean).join(" · ");
  $("#aeCap").innerHTML = `<b>Agent's eye</b><small>${escapeHtml(how)}</small>`;
  ae.hidden = false; ae.classList.remove("pop"); void ae.offsetWidth; ae.classList.add("pop");
}
function hideAgentEye() { const ae = $("#agentEye"); if (ae) ae.hidden = true; }

/* cinematic auto-tour */
let _tourTimer = null;
function playGazeTour() {
  endTour();
  const ids = $$(".ev-card:not(.ev-under)").map(c => +c.dataset.id);
  if (!ids.length) return;
  state._tour = true; state._tourOptOut = false;
  const btn = $("#tourBtn"); if (btn) { btn.classList.add("is-on"); btn.innerHTML = "Stop tour"; }
  let i = 0;
  const step = () => {
    if (!state._tour) return;
    if (i >= ids.length) { endTour(); setTimeout(() => Viewer.fit(), 400); setTimeout(hideAgentEye, 1300); return; }
    gazeTo(ids[i]); i++; _tourTimer = setTimeout(step, 2100);
  };
  step();
}
function endTour() { state._tour = false; clearTimeout(_tourTimer); _tourTimer = null; setTimeout(toastHide, 1200); const btn = $("#tourBtn"); if (btn) { btn.classList.remove("is-on"); btn.innerHTML = "Agent tour"; } }
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
  const pp = ev.p_pot != null ? `${Math.round(ev.p_pot * 100)}%` : "—";
  card.innerHTML = `
    <button type="button" class="ev-sum" aria-label="show the evidence for find ${id}">
      ${c.crop_png ? `<img src="${c.crop_png}" alt="" loading="lazy" decoding="async"/>` : `<span class="ev-sum-ph"></span>`}
      <span class="ev-sum-txt"><span class="ev-sum-v">${escapeHtml(VLABEL[v] || v)}</span>
        <span class="ev-sum-m">score ${sc.toFixed(2)} · P(pot) ${pp}</span></span>
    </button>
    <div class="ev-crop${c.relook_view ? " has-enh" : ""}">
      ${c.crop_png ? `<img class="ev-img raw" alt="evidence crop" src="${c.crop_png}" decoding="async"/>` : `<div style="aspect-ratio:1"></div>`}
      ${c.relook_view ? `<img class="ev-img enh" alt="CLAHE re-look (what the agent saw)" src="${c.relook_view.enhanced_png}" loading="lazy" decoding="async"/>` : ""}
      <span class="ev-badge b-${v}">${VLABEL[v] || human(v)}</span>
      ${c.relook_view ? `<button class="cmp-chip" title="toggle raw / CLAHE re-look">Enhanced</button>` : ""}
      <button class="gaze-btn" title="replay what the agent saw (G)">Replay gaze</button>
    </div>
    <div class="ev-body">
      <p class="ev-tier">${escapeHtml(tierPromise(c))}</p>
      ${ev.action ? `<div class="conf-flow">
        <div class="conf-chip"><span class="lbl">Detector P</span><span class="val">${pp}</span></div>
        <div class="conf-arrow ${ev.p_evidence > ev.p_pot + 1e-6 ? "up" : ev.p_evidence < ev.p_pot - 1e-6 ? "down" : ""}"><div class="track"></div>
          <span class="tag">after OpenCV evidence</span></div>
        <div class="conf-chip"><span class="lbl">Evidence P</span><span class="val">${Math.round(ev.p_evidence * 100)}%</span></div>
      </div>` : `<div class="conf-flow">
        <div class="conf-chip"><span class="lbl">Detector</span><span class="val">${(c.conf).toFixed(2)}</span></div>
        <div class="conf-arrow ${arrowCls}"><div class="track"></div>
          <span class="tag">${!rlDone ? "No re-look (VoI 0)" : `re-look ${rl.found ? `${rlConf.toFixed(2)} (${gain >= 0 ? "+" : ""}${gain.toFixed(2)})` : "no re-fire"}`}</span></div>
        <div class="conf-chip"><span class="lbl">Score</span><span class="val ${arrowCls}">${(ev.evidence_score ?? c.conf).toFixed(2)}</span></div>
      </div>`}
      <div class="ev-facts">
        <div class="fact"><span class="k">Shadow</span><span class="v ${shCls}">${human(sh.quality || "none")}${sh.contrast ? ` · c${sh.contrast}` : ""}</span></div>
        <div class="fact"><span class="k">Relative height</span><span class="v">${height}</span></div>
        <div class="fact"><span class="k">Echo vs background</span><span class="v">${(sh.echo_ratio ?? ev.echo_ratio ?? 0).toFixed(1)}×</span></div>
        <div class="fact"><span class="k">Position</span><span class="v ${c.in_water_column ? "chip-weak" : ""}">${c.in_water_column == null ? "—" : c.in_water_column ? "Water column" : "On seabed"}${c.ground_range_px != null ? ` · ${Math.round(c.ground_range_px)} px gnd` : ""}</span></div>
        <div class="fact"><span class="k">P(pot)</span><span class="v">${ev.p_pot != null ? Math.round(ev.p_pot * 100) + "%" : "—"}</span></div>
      </div>
      ${agentTrace(c)}
      ${(c.evidence || {}).action ? `<details class="ev-why ev-steps"><summary>Every step (${(c.trace || []).length})</summary>${chainBlock(c.trace || [])}</details>`
        : `<div class="ev-chain"><div class="ev-chain-h">Agent decision<span>${(c.trace || []).filter(s => s.status === "skipped").length} skipped · ${(c.trace || []).filter(s => s.status === "failed").length} failed</span></div>
        ${chainBlock(c.trace || [])}</div>`}
      ${(ev.notes || []).length ? `<details class="ev-why"><summary>Evidence notes</summary><ul class="ev-notes">${ev.notes.map(n => `<li>${escapeHtml(n)}</li>`).join("")}</ul></details>` : ""}
      <div class="ev-label"><span class="lbl-k">Your call</span>
        <button class="ok" data-a="confirm" title="a real object - saved as a positive training label">Approve</button>
        <button class="no" data-a="reject" title="not an object - saved as a hard negative">Reject</button>
        <span class="ov-wrap"><button class="ov" data-a="override" aria-haspopup="menu" title="override the agent's tier">Override</button>
          <span class="ov-menu" role="menu" hidden>${VERDICTS.filter(x => x !== v).map(x => `<button role="menuitem" data-to="${x}">${VLABEL[x]}</button>`).join("")}</span></span></div>
      <p class="ev-saved" hidden></p>
    </div>`;
  const saved = (txt) => { const el = card.querySelector(".ev-saved"); el.hidden = false; el.textContent = txt; };
  const label = async (rec, btn) => {
    const res = await sendLabel(Object.assign({ bbox: c.bbox, cls_name: c.cls_name, verdict_before: c.verdict, conf: c.conf,
      p_pot: ev.p_pot, source: "analyze", annotator: whoAmI() || "anon" }, rec));
    if (!res) return;
    card.querySelectorAll(".ev-label button").forEach(x => x.classList.toggle("on", x === btn));
    const who = whoAmI();
    saved(rec.decision === "override"
      ? `Override recorded: ${VLABEL[c.verdict]} → ${VLABEL[rec.override_to]}${who ? " by " + who : ""}. The agent's verdict is kept beside it.`
      : `${rec.decision === "confirm" ? "Approved" : "Rejected"}${who ? " by " + who : ""} · saved as a training label`);
    Log.push([{ tool: "human_label", msg: `${rec.decision}${rec.override_to ? " → " + rec.override_to : ""} on ${c.cls_name} ${c.conf.toFixed(2)}`, t: "" }]);
  };
  card.querySelectorAll(".ev-label [data-a=confirm], .ev-label [data-a=reject]").forEach(b => b.addEventListener("click", e => {
    e.stopPropagation(); label({ decision: b.dataset.a }, b);
  }));
  const ovBtn = card.querySelector(".ov"), ovMenu = card.querySelector(".ov-menu");
  ovBtn.addEventListener("click", e => { e.stopPropagation(); ovMenu.hidden = !ovMenu.hidden; });
  ovMenu.addEventListener("click", e => {
    e.stopPropagation(); const b = e.target.closest("[data-to]"); if (!b) return;
    ovMenu.hidden = true; label({ decision: "override", override_to: b.dataset.to }, ovBtn);
  });
  card.addEventListener("mouseenter", () => Viewer.highlight(id, true));
  card.addEventListener("mouseleave", () => Viewer.highlight(id, false));
  card.querySelector(".ev-crop").addEventListener("click", () => selectCandidate(id));
  card.querySelector(".ev-sum").addEventListener("click", () => selectCandidate(id));
  card.querySelector(".gaze-btn").addEventListener("click", e => { e.stopPropagation(); gazeTo(id); });
  const chip = card.querySelector(".cmp-chip");
  if (chip) chip.addEventListener("click", e => { e.stopPropagation(); card.querySelector(".ev-crop").classList.toggle("show-enh"); });
  return card;
}
