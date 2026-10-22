/* DEPTH studio — people decide: the approvals inbox, the hazard table, person decisions and training labels.
   Zero-build: index.html loads webui/js/*.js in order as classic scripts sharing one global scope;
   the boot runs on DOMContentLoaded (core.js), after every file has loaded. API: src/dashboard/routes/. */
"use strict";

/* approvals inbox: requests from this studio, MCP agents and partner consoles; a NAMED person decides */
const Appr = {
  timer: null,
  async load() {
    try {
      const r = await fetch(`${API}/api/approvals`).then(x => x.json());
      const c = r.counts || {}, list = r.requests || [];
      $("#apprMeta").textContent = `${c.pending || 0} pending · ${c.approved || 0} approved · ${c.declined || 0} declined${c.withdrawn ? ` · ${c.withdrawn} withdrawn` : ""}`;
      if (!list.length) return;
      $("#apprList").innerHTML = list.slice(0, 30).map(q => `<div class="appr" data-id="${escapeHtml(q.id)}">
        <div class="appr-top"><span class="act">${escapeHtml(q.action)}</span><span>${escapeHtml(q.targets.slice(0, 6).join(", "))}${q.targets.length > 6 ? " …" : ""}</span><span class="st ${q.status}">${q.status}</span></div>
        <div class="appr-why">${escapeHtml(q.rationale)}</div>
        <div class="appr-meta">${escapeHtml(q.id)} · asked by ${escapeHtml(q.requested_by)} via ${escapeHtml(q.channel)} · ${new Date(q.t * 1000).toLocaleString()}${q.decision ? ` · ${q.status} by <b>${escapeHtml(q.decision.by)}</b>${q.decision.note ? `: ${escapeHtml(q.decision.note)}` : ""}` : ""}${q.status === "approved" ? " · signed work order sent to consoles (DEPTH never dispatches)" : ""}</div>
        ${q.status === "pending" ? `<div class="appr-btns"><button class="ok" data-d="approved">Approve</button><button class="no" data-d="declined">Decline</button></div>` : ""}
      </div>`).join("");
      $$("#apprList .appr-btns button").forEach(b => b.onclick = () => this.decide(b.closest(".appr").dataset.id, b.dataset.d));
    } catch (e) { $("#apprMeta").textContent = "unavailable"; }
  },
  async decide(id, decision) {
    const by = ($("#apprName").value || "").trim();
    if (!by) { $("#apprName").focus(); $("#apprName").placeholder = "name required to decide"; return; }
    localStorage.setItem("depth.approver", by);
    const card = $(`#apprList .appr[data-id="${CSS.escape(id)}"]`);
    if (card) card.querySelectorAll(".appr-btns button").forEach(b => b.disabled = true);   // one decision per request
    let r, body;
    try {
      r = await fetch(`${API}/api/approvals/${encodeURIComponent(id)}/decide`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ decision, by, pin: approverPin() }) });
      body = await r.json().catch(() => ({}));
    } catch (e) { r = { ok: false, status: 0 }; body = { detail: "the server did not answer - try again" }; }
    if (r.status === 403) { $("#apprPin").hidden = false; $("#apprPin").focus(); }
    if (r.status === 409) { toast(`Not recorded: ${body.detail}.`); setTimeout(toastHide, 5200); }   // someone decided first
    Log.push([{ tool: "human_gate", msg: r.ok ? `${id} ${decision} by ${by}` : `${id}: ${body.detail || r.status}`, t: "" }]);
    if (r.ok && body.dispatch && state.survey && body.survey_id === state.survey.survey_id) {
      state.survey.dispatch = body.dispatch;
      if (body.mission) {                                   // a rejected pass: the agent changed its plan
        state.survey.tracked = body.tracked; state.survey.mission = body.mission;
        if (body.replan) { toast(chainText({ rationale: body.replan.rationale })); setTimeout(toastHide, 4200); }
        renderSurvey(state.survey, { keepView: true });
      } else {
        if (body.work_order) { toast(`Approved: work order ${body.work_order.request} issued. The plan stands — next: ${body.work_order.next}.`); setTimeout(toastHide, 5200); }
        renderAgentPanel(state.survey); loadAudit(state.survey.survey_id);
      }
    }
    this.load();
  },
  start() {
    $("#apprName").value = localStorage.getItem("depth.approver") || "";
    this.load(); clearInterval(this.timer); this.timer = setInterval(() => { if (!document.hidden) this.load(); }, 8000);
  },
};
/* provenance of the last result: which model, thresholds, OpenCV build (COOL?) and code produced it */
function showProvenance(p) {
  const el = $("#provLine"); if (!el || !p) return;
  const o = p.opencv || {};
  el.innerHTML = `model <b>${escapeHtml(p.model || "?")}</b> ${escapeHtml((p.model_onnx_sha256 || "").slice(0, 10))} · calib ${escapeHtml((p.calibration_sha256 || "").slice(0, 10))} · OpenCV ${escapeHtml(o.version || "?")} <b>${o.is_cool_path ? "COOL" : "stock"}</b> · ${escapeHtml(p.code_commit || "?")}`;
  el.title = JSON.stringify(p, null, 1);
}
function renderThumbs(d) {
  $("#thumbStrip").innerHTML = d.frames.filter(f => f.thumb_png)
    .map(f => `<img class="thumb" title="${f.frame_id} · ${cShort(f.counts)}" src="${f.thumb_png}" alt="${f.frame_id}" loading="lazy" decoding="async"/>`).join("");
}
function renderHazards(d) {
  const rows = d.tracked;
  const rank = Object.fromEntries((d.mission.review_queue || []).map((id, i) => [id, i]));
  rows.sort((a, b) => (VERDICTS.indexOf(a.verdict) - VERDICTS.indexOf(b.verdict)) || ((rank[a.oid] ?? 1e9) - (rank[b.oid] ?? 1e9)));
  const imp = d.mission.impact || {};
  $("#hazCount").textContent = `${rows.length} hazards · review ordered by P(real) × severity` + (imp.decisions
    ? ` · people: ${imp.person_confirmed} confirmed, ${imp.person_rejected} rejected, ${imp.person_recovered} recovered` + (imp.reviewed_precision != null
      ? ` · reviewed precision ${Math.round(imp.reviewed_precision * 100)}% (95% CI ${Math.round(imp.reviewed_precision_ci95[0] * 100)}–${Math.round(imp.reviewed_precision_ci95[1] * 100)}%)` : "")
    : " · low risk kept for audit");
  const body = $("#hazBody"); body.innerHTML = "";
  rows.forEach(t => {
    const tr = document.createElement("tr"); tr.dataset.oid = t.oid;
    const coords = t.lat != null ? `${t.lat.toFixed(5)}, ${t.lon.toFixed(5)}`
      : t.position_withheld ? `<span class="chip-weak" title="${escapeHtml(t.position_withheld)}">withheld</span>` : `<span class="muted">no GPS</span>`;
    tr.innerHTML = `<td>${t.oid}${t.sightings > 1 ? ` <span class="muted" title="seen on ${t.sightings} frames/passes">×${t.sightings}</span>` : ""}</td><td>${human(t.cls_name)}</td>
      <td><span class="v-tag" style="color:${VCOLOR[t.verdict]};background:${VCOLOR[t.verdict]}22">${VLABEL[t.verdict] || t.verdict}</span></td>
      <td>${t.conf}</td><td>${t.evidence_score}</td><td>${t.p_pot != null ? Math.round(t.p_pot * 100) + "%" : "—"}${t.severity > 1 ? ` <span class="chip-weak" title="severity ×${t.severity}: ${escapeHtml((t.severity_why || []).join("; "))} (operator policy)">×${t.severity}</span>` : ""}</td><td>${heightText(t.height_m, t.height_rel, t.height_rel ? 1 : 0)}</td>
      <td class="${t.shadow_quality === "clear" ? "chip-clear" : t.shadow_quality === "weak" ? "chip-weak" : "chip-none"}">${t.shadow_quality}</td>
      <td>${coords}</td><td>${t.geo_error_m != null ? "±" + t.geo_error_m + " m" : "—"}</td><td>${reviewCell(t)}</td>`;
    body.appendChild(tr);
  });
  $$("#hazBody .rev-btns button").forEach(b => b.onclick = async e => {
    const tr = e.target.closest("tr"), t = rows.find(x => x.oid === tr.dataset.oid);
    if (t) await decideHazard(d, t, b.dataset.d);
  });
}
function reviewCell(t) {
  const HUM = { confirmed: "✓ confirmed", rejected: "✕ rejected", recovered: "⚓ recovered", not_found: "○ not found" };
  const onRoute = t.human === "confirmed" || (t.verdict === "confirmed" && !t.human);
  if (t.human && !onRoute) return `<span class="rev-btns"><span class="h-chip h-${t.human}" title="by ${escapeHtml(t.human_by || "")}">${HUM[t.human]}</span><button data-d="undo" title="re-open this card">undo</button></span>`;
  if (onRoute) return `<span class="rev-btns">${t.human ? `<span class="h-chip h-confirmed" title="by ${escapeHtml(t.human_by || "")}">✓</span>` : ""}<button class="ok" data-d="recovered" title="the crew brought it up">recovered</button><button data-d="not_found" title="the crew went - nothing there">not found</button>${t.human ? `<button data-d="undo" title="re-open">undo</button>` : ""}</span>`;
  if (t.verdict !== "review") return `<span class="muted">—</span>`;
  const pri = t.human_priority
    ? `<span class="h-chip h-pri" title="a person's override - the agent keeps this order">${t.human_priority === "high" ? "↑ first" : "↓ last"}</span><button data-d="reset_priority" title="give the order back to the agent">reset</button>`
    : `<button data-d="prioritize" title="override the agent: inspect this first">↑</button><button data-d="deprioritize" title="override the agent: end of the queue">↓</button>`;
  return `<span class="rev-btns"><button class="ok" data-d="confirm" title="a real pot - put it on the recovery route">✓</button><button class="no" data-d="reject" title="not a pot - take it off the queue">✕</button><span class="rev-sep"></span>${pri}</span>`;
}
/* the approver PIN (when the server sets one) lives only in this tab's input - never stored */
function approverPin() { const p = $("#apprPin"); return p && !p.hidden ? p.value : undefined; }
/* a PERSON's decision -> the agent re-plans (recovery route, queue, budget, passes); also a training label */
async function decideHazard(d, t, decision) {
  const by = ($("#apprName").value || "").trim();
  if (!by) { $("#apprName").focus(); $("#apprName").placeholder = "your name — required to decide"; toast("enter your name in Approvals first"); setTimeout(toastHide, 1800); return; }
  localStorage.setItem("depth.approver", by);
  const r = await fetch(`${API}/api/survey/${encodeURIComponent(d.survey_id)}/decide`, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ hazard_id: t.oid, decision, by, pin: approverPin() }) });
  if (r.status === 403) { $("#apprPin").hidden = false; $("#apprPin").focus(); }
  const b = await r.json().catch(() => ({}));
  if (!r.ok) { toast(typeof b.detail === "string" ? b.detail : `not saved (${r.status})`); setTimeout(toastHide, 2200); return; }
  state.survey.tracked = b.tracked; state.survey.mission = b.mission; state.survey.dispatch = b.dispatch;
  const m = b.mission, rp = b.replan;
  Log.push([{ tool: "human_gate", msg: `${t.oid} ${decision} by ${by}${b.label ? " · saved as a training label" : ""}`, t: "" },
            { tool: "replan", msg: rp ? rp.rationale : `queue ${m.review_queue.length}`, t: "" }]);
  if (rp) { toast(chainText({ rationale: rp.rationale })); setTimeout(toastHide, 4200); }
  if (b.label) loadLabelStats();
  renderSurvey(state.survey, { keepView: true });
  Appr.load();
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
  if (Twin.dim === "3d") return;                                   // drawing a box is a 2D action
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
