/* DEPTH studio — measured with people: the timed user study, the effort curve and the blinded audit.
   Zero-build: index.html loads webui/js/*.js in order as classic scripts sharing one global scope;
   the boot runs on DOMContentLoaded (core.js), after every file has loaded. API: src/dashboard/routes/. */
"use strict";

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
  $$("#effMode button").forEach(b => b.addEventListener("click", () => {
    $$("#effMode button").forEach(x => x.classList.toggle("on", x === b)); loadEffort(true);
  }));
  ["effSpf", "effSpc"].forEach(id => $("#" + id).addEventListener("input", () => {
    $("#effSpfV").textContent = $("#effSpf").value + " s"; $("#effSpcV").textContent = $("#effSpc").value + " s";
    clearTimeout(Study._eff); Study._eff = setTimeout(() => loadEffort(true), 120);
  }));
}

async function studyStart() {
  const btn = $("#studyStartBtn"); btn.disabled = true; Busy.start("Preparing your session");
  try {
    const name = ($("#studyName").value || "anon").trim();
    Study.plan = await fetch(`${API}/api/study/plan?participant=${encodeURIComponent(name)}`).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); });
    Study.manual = []; Study.cards = [];
    $("#studyMeta").textContent = `session ${Study.plan.index + 1} · ${Study.plan.order.join(" → ")}`;
    startArm(0);
  } catch (e) { toast(`could not start (${e.message})`); setTimeout(toastHide, 1800); }
  finally { Busy.done(); btn.disabled = false; }
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
  clearInterval(Study.timer); Busy.start("Scoring against the labels");
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
  finally { Busy.done(); }
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
  const mb = $("#effMode button.on"), md = mb ? mb.dataset.mode : "standard";
  const q = `?mode=${md}` + (fromSliders ? `&sec_per_frame=${$("#effSpf").value}&sec_per_card=${$("#effSpc").value}` : "");
  let j; try { j = await fetch(`${API}/api/effort${q}`).then(r => { if (!r.ok) throw 0; return r.json(); }); } catch { $("#effChart").innerHTML = `<p class="empty-hint">No effort data for this model yet (python -m src.agentic.effort).</p>`; return; }
  const C = j.curves, W = 300, H = 170, P = { l: 30, r: 8, t: 8, b: 22 };
  $("#effProv").textContent = j.provenance.startsWith("measured") ? "measured" : j.provenance.startsWith("set") ? "what-if" : "assumed";
  const xmax = Math.max(...[C.manual, C.detector_list, C.depth].map(c => c[0][c[0].length - 1] || 1));
  const X = x => P.l + (x / xmax) * (W - P.l - P.r), Y = y => H - P.b - y * (H - P.t - P.b);
  const pl = (c, col, dash) => `<polyline fill="none" stroke="${col}" stroke-width="${dash ? 1.2 : 2}" ${dash ? 'stroke-dasharray="3 3"' : ""} points="${c[0].map((x, i) => X(x).toFixed(1) + "," + Y(c[1][i]).toFixed(1)).join(" ")}"/>`;
  const R = C.promise || 0, mt = C.minutes_to_promise;
  const grid = [0.25, 0.5, 0.75, 1].map(v => `<line class="gr" x1="${P.l}" x2="${W - P.r}" y1="${Y(v)}" y2="${Y(v)}"/><text x="2" y="${Y(v) + 3}">${Math.round(v * 100)}%</text>`).join("");
  const mk = (m, col) => m == null ? "" : `<circle cx="${X(m)}" cy="${Y(R)}" r="4.5" fill="${col}" stroke="#0e141b" stroke-width="2"/>`;
  $("#effChart").innerHTML = `<svg class="eff-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="share of pots found versus analyst minutes">
    ${grid}<line class="ax" x1="${P.l}" x2="${W - P.r}" y1="${H - P.b}" y2="${H - P.b}"/>
    <line x1="${P.l}" x2="${W - P.r}" y1="${Y(R)}" y2="${Y(R)}" stroke="#6a7885" stroke-dasharray="2 3"/>
    <text x="${P.l + 3}" y="${Y(R) - 3}">promise ≥ ${Math.round(R * 100)}%</text>
    ${pl(C.detector_list, EFF.list)}${pl(C.manual, EFF.man)}${pl(C.forecast, EFF.depth, true)}${pl(C.depth, EFF.depth)}
    ${mk(mt.manual, EFF.man)}${mk(mt.depth, EFF.depth)}
    <text x="${W - P.r}" y="${H - 6}" text-anchor="end">analyst minutes (${j.frames} frames)</text></svg>
    <div class="eff-leg"><span><i style="background:${EFF.man}"></i>manual</span><span><i style="background:${EFF.list}"></i>detector list</span><span><i style="background:${EFF.depth}"></i>DEPTH queue</span><span><i style="background:none;border-top:1px dashed ${EFF.depth}"></i>DEPTH forecast</span></div>`;
  const ph = C.per_survey_hour_min || {};
  const lhBtn = $('#effMode button[data-mode="look_harder"]'); if (lhBtn) lhBtn.hidden = !(j.modes || []).includes("look_harder");
  $("#effStats").innerHTML = `${(j.cards || {}).one_per_object ? "One card per object (STUDY-23). " : ""}At the promise (≥ ${Math.round(R * 100)}%): manual <b>${mt.manual != null ? mt.manual.toFixed(1) : "—"} min</b>, DEPTH <b>${mt.depth != null ? mt.depth.toFixed(1) : "never"}</b>${mt.depth != null ? ` min (${C.cards_to_promise} cards)` : " — card accuracy too low for the promise"} · per survey-hour ${ph.manual ?? "—"} vs ${ph.depth ?? "—"} min.<br/>
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
