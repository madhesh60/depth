/* DEPTH studio — incidents and the agent's decision text: traces, the core loop's steps, active vision, drills.
   Zero-build: index.html loads webui/js/*.js in order as classic scripts sharing one global scope;
   the boot runs on DOMContentLoaded (core.js), after every file has loaded. API: src/dashboard/routes/. */
"use strict";

/* ------------------------------------------------------------------ incidents: every failure is told to a person
   (src/agentic/incidents.py): what happened, the safe fallback the system took, and what YOU should do */
const Notices = {
  seen: new Set(),
  show(list, ctx = "") {
    const box = $("#notices"); if (!box) return;
    (list || []).forEach(inc => {
      const key = `${ctx}|${inc.code}|${inc.frame_id || ""}|${inc.message}`;
      if (this.seen.has(key)) return; this.seen.add(key);
      const el = document.createElement("div");
      el.className = `notice sev-${inc.severity || "warning"}`;
      el.setAttribute("role", inc.severity === "error" ? "alert" : "status");
      el.innerHTML = `<div class="nt-head"><span class="nt-dot"></span><b>${escapeHtml(inc.title || human(inc.code))}</b>`
        + `${inc.simulated ? `<span class="nt-drill">Drill</span>` : ""}<button class="nt-x" aria-label="dismiss">×</button></div>`
        + `<div class="nt-msg">${escapeHtml(inc.message || "")}</div>`
        + (inc.fallback ? `<div class="nt-row"><span>Fallback</span><p>${escapeHtml(inc.fallback)}</p></div>` : "")
        + (inc.human_action ? `<div class="nt-row"><span>You</span><p>${escapeHtml(inc.human_action)}</p></div>` : "");
      el.querySelector(".nt-x").onclick = () => el.remove();
      box.prepend(el);
      if (inc.severity === "info") setTimeout(() => { el.classList.add("nt-out"); setTimeout(() => el.remove(), 400); }, 9000);
    });
    while (box.children.length > 4) box.lastChild.remove();
  },
};
function incidentRow(inc) {
  return `<div class="inc sev-${inc.severity || "warning"}"><div class="inc-h"><span class="nt-dot"></span><b>${escapeHtml(inc.title || human(inc.code))}</b>`
    + `${inc.simulated ? `<span class="nt-drill">Drill</span>` : ""}</div><div class="inc-m">${escapeHtml(inc.message || "")}</div>`
    + `<div class="inc-f"><span>Fallback</span> ${escapeHtml(inc.fallback || "")}</div>`
    + `<div class="inc-f"><span>You</span> ${escapeHtml(inc.human_action || "")}</div></div>`;
}
/* fetch JSON; a failed call throws with the server's incident attached (never a bare status code) */
async function apiJSON(url, opts) {
  const r = await fetch(url, opts);
  const b = await r.json().catch(() => ({}));
  if (!r.ok) {
    const d = b.detail;
    const e = new Error((d && (d.message || (typeof d === "string" ? d : ""))) || `HTTP ${r.status}`);
    e.incident = d && d.incident; e.status = r.status; throw e;
  }
  return b;
}
function whoAmI() { return ($("#apprName") && $("#apprName").value.trim()) || localStorage.getItem("depth.approver") || ""; }

/* the agent's decision chain: Ask -> Evidence (done / skipped / failed) -> Update -> Decide -> Next */
const STEP = {
  voi_check: ["Ask", "Can another look change it?"], zoom_relook: ["Evidence", "Re-look"],
  mosaic_relook: ["Evidence", "Re-look (batched)"], enhance_relook: ["Evidence", "CLAHE re-look"],
  water_column_check: ["Evidence", "Water column"], shadow_check: ["Evidence", "Acoustic shadow"],
  estimate_height: ["Evidence", "Height"], stitch_boundary: ["Evidence", "Chunk stitch"],
  update_belief: ["Update", "P(pot)"], decide: ["Decide", "Tier"], handoff: ["Next", "Hand-off"],
  assess: ["Detect", "Detector alone"], geometry_check: ["Evidence", "Geometry (Stage 1)"],
  choose_tool: ["Choose", "Next OpenCV tool"], conflict: ["Conflict", "Evidence conflict"],
  request_observation: ["Ask", "Another observation"], look_harder: ["Detect", "Closer look (2×2 tiles)"],
  replan: ["Re-plan", "Compute budget"],
};
/* the core loop (src/agentic/core.py): goal -> inspect -> choose how hard to look -> perceive -> plan */
const CORE_STEP = {
  inspect_data: ["Inspect", "Usable frames"], check_promise: ["Check", "Does the promise apply here?"],
  measure_geometry: ["Sense", "Seabed: image vs sounder"],
  choose_mode: ["Decide", "How hard to look"], perceive: ["Perceive", "OpenCV 5 on every frame"],
  replan: ["Re-plan", "Compute budget"], retier: ["Re-plan", "Re-tier earlier frames"],
  plan: ["Plan", "People and boat"],
};
const pctS = p => (p == null ? "—" : Math.round(p * 100) + "%");
/* "How the agent decided": the goal, every perception mode it weighed, and its steps */
function agentRunBlock(run) {
  if (!run || !run.steps) return "";
  const g = run.goal || {}, m = run.mode || {};
  const ch = (run.steps.find(s => s.act === "choose_mode") || {}).options || [];
  const goal = [g.recall != null ? `recall ≥ ${pctS(g.recall)}` : "recall: strongest available",
    g.compute_s != null ? `compute ≤ ${g.compute_s} s` : "", g.analyst_minutes != null ? `analyst ${g.analyst_minutes} min` : "",
    g.boat_minutes != null ? `boat ${g.boat_minutes} min` : ""].filter(Boolean);
  const steps = run.steps.map(s => ({ tool: s.act, status: s.status,
    rationale: s.why + (s.decided ? " -> " + s.decided : "") }));
  return `<div class="cfx run-x"><div class="gate-h">How the agent decided<span>${escapeHtml(m.label || "uncalibrated")}${m.promise != null ? ` · promise ≥ ${pctS(m.promise)}` : ""}${(run.replans || []).length ? " · re-planned mid-run" : ""}</span></div>
    <div class="mb-tags">${goal.map(t => `<span class="mb-tag">${escapeHtml(t)}</span>`).join("")}</div>
    ${ch.length ? `<table><thead><tr><th>Perception mode</th><th>Promise</th><th>Passes / frame</th><th>Cards (est.)</th><th>Compute (est.)</th><th></th></tr></thead><tbody>${ch.map(o =>
      `<tr class="${o.chosen ? "chg" : ""}"><th>${escapeHtml(o.label)}</th><td>≥ ${pctS(o.promise)}</td><td>${o.passes_per_frame}</td>`
      + `<td>${o.expected_cards ?? "—"}</td><td>${o.expected_compute_s != null ? o.expected_compute_s + " s" : "—"}</td>`
      + `<td>${o.chosen ? "chosen" : escapeHtml((o.why_not || []).join("; ") || "weaker or costlier")}</td></tr>`).join("")}</tbody></table>` : ""}
    ${chainBlock(steps, CORE_STEP)}
    <p class="gate-note">${escapeHtml(chainText({ rationale: run.stop || "" }))}</p></div>`;
}
const PLAN_STEP = {
  triage: ["Triage", "Hazards"], voi_check: ["Ask", "Can more looking help?"], budget_check: ["Budget", "Analyst time"],
  stop_rule: ["Stop", "Where to stop"], human_override: ["Person", "Overrides kept"], route_update: ["Route", "Inspection route"],
  recovery_route: ["Route", "Recovery route"], resurvey_plan: ["Re-survey", "Boat time"], dispatch_gate: ["Gate", "Dispatch"],
  request_approval: ["Ask person", "Approval"], human_declined: ["Person", "Pass declined"],
  resurvey_rank: ["Re-survey", "Most informative first"],
  sensor_crosscheck: ["Sense", "Seabed: image vs sounder"], geometry_update: ["Sense", "Corrected geometry used"],
  geometry_withheld: ["Sense", "Geometry withheld"],
};
/* one chunk of a raw recording: the agent's seabed cross-check (src/agentic/sensor_check.py) */
const SENSE_STEP = {
  bottom_track: ["Sense", "OpenCV bottom track"], sounder_check: ["Sense", "Depth sounder"],
  crosscheck: ["Check", "Image vs sounder"], retrack_guided: ["Re-measure", "Re-track where the sounder points"],
  image_check: ["Check", "Is the image edge coherent?"], retrack_cross_channel: ["Re-measure", "Re-track from the other channel"],
  resolve: ["Decide", "Seabed for this chunk"], estimate_scale: ["Model", "Range scale"],
};
const GEO_LABEL = { agree: "Agree", corrected: "Corrected", recovered: "Recovered", image_trusted: "Image trusted", not_measured: "Not measured" };
/* display text: the question is the step's name, so show only the answer; arrows typeset */
function chainText(s) {
  let t = String(s.rationale || "");
  if (s.tool === "voi_check") t = t.replace(/^.*?change (the decision|these \d+ card\(s\))\? /, "");
  t = t.replace(/ -> /g, " → ").replace(/ <= /g, " ≤ ").replace(/ >= /g, " ≥ ")
    .replace(/^(yes|no) - (\w)/, (m, w, c) => (w === "yes" ? "Yes. " : "No. ") + c.toUpperCase());
  return t ? t[0].toUpperCase() + t.slice(1) : t;
}
function chainBlock(steps, map = STEP) {
  if (!steps || !steps.length) return "";
  let prev = "";
  return `<ol class="chain">${steps.map(s => {
    const [stage, name] = map[s.tool] || ["Step", human(s.tool)], st = s.status || "done";
    const head = stage !== prev; prev = stage;
    return `<li class="ch ch-${st}${head ? " ch-first" : ""}"><span class="ch-stage">${head ? stage : ""}</span>`
      + `<span class="ch-node"></span><div class="ch-body"><div class="ch-name">${escapeHtml(name)}`
      + `${st !== "done" ? `<span class="ch-st">${st}</span>` : ""}${s.latency_ms ? `<span class="ch-ms">${fmt(s.latency_ms)} ms</span>` : ""}</div>`
      + `<div class="ch-why">${escapeHtml(chainText(s))}</div></div></li>`;
  }).join("")}</ol>`;
}

/* the Agent Trace for one find: DETECT -> evidence -> decision -> OpenCV tool -> new evidence ->
   decision changed -> human approval -> survey plan. Built only from the agent's own logged steps. */
const ACT = { accept: "Accept", review: "Review", watch: "Watch" };
function agentTrace(c) {
  const ev = c.evidence || {}, tr = c.trace || [];
  if (!ev.action) return "";
  const pct = p => (p == null ? "—" : Math.round(p * 100) + "%");
  const geo = tr.find(s => s.tool === "geometry_check");
  const chosen = tr.filter(s => s.tool === "choose_tool").map(s => human((s.detail || {}).chosen || ""));
  const upd = tr.filter(s => s.tool === "update_belief");
  const skipped = tr.filter(s => (s.tool === "shadow_check" || s.tool === "zoom_relook") && s.status === "skipped").map(s => human(s.tool));
  const changed = ev.action !== ev.action0;
  const geoTxt = !geo || geo.status === "skipped" ? "Geometry not measured" : (geo.detail || {}).in_water_column ? "Geometry: in the water column" : "Geometry: on the seabed";
  const rows = [
    ["Detect", `YOLO11 via cv2.dnn · confidence ${c.conf.toFixed(2)}`],
    ["Evidence collected", geoTxt + " (Stage 1, free)"],
    ["Agent decision", `${ACT[ev.action0]} · P ${pct(ev.p_pot)} from the detector alone`],
    ["OpenCV tool selected", chosen.length ? chosen.join(" → ") + (skipped.length ? ` · skipped ${skipped.join(", ").toLowerCase()} (could not change it)` : "")
                                          : "None — no tool could change the decision"],
    ["New evidence", upd.length ? upd.map(u => `${human((u.detail || {}).tool || "")} ${(u.detail || {}).result || ""} · P ${pct(u.detail.p_prior)} → ${pct(u.detail.p_post)}`).join("; ") : "—"],
  ];
  if (ev.conflict) rows.push(["Evidence conflict", `${ev.conflict.why} → another observation`, "warn"]);
  rows.push(["Decision changed", changed ? `${ACT[ev.action0]} → ${ACT[ev.action]}` : `No — stays ${ACT[ev.action]}`, changed ? "hi" : ""]);
  rows.push(["Human approval", ev.action === "accept" ? "Dispatch waits for a named person" : "A person reviews the card"]);
  rows.push(["New survey plan", ev.request_resurvey ? "Opposite-side pass requested (approve in Survey)"
    : ev.action === "accept" ? "Priority inspection target" : ev.action === "watch" ? "End of the review queue" : "Review queue"]);
  return `<div class="atrace"><div class="ev-chain-h">Agent trace${((tr.find(s => s.tool === "decide") || {}).detail || {}).active_mode === "experimental" ? `<span class="exp-tag">experimental</span>` : ""}</div><ol>${rows.map(([k, v, cls]) =>
    `<li class="${cls || ""}"><span class="at-k">${k}</span><span class="at-v">${escapeHtml(v)}</span></li>`).join("")}</ol></div>`;
}

/* active vision: off by default (its validation checks failed); the switch runs it, labelled experimental */
const Active = {
  get on() { return localStorage.getItem("depth.active") === "1"; },
  q() { return this.on ? "&active=1" : ""; },
  init() {
    $$(".activeToggle").forEach(x => {
      x.checked = this.on;
      x.onchange = () => { localStorage.setItem("depth.active", x.checked ? "1" : "0"); $$(".activeToggle").forEach(y => y.checked = x.checked); };
    });
  },
};

/* failure drills: trigger a real failure path on purpose and watch the fallback (?simulate=<code>) */
const DRILLS = [
  { code: "model_unavailable", label: "Detector offline", where: "analyze" },
  { code: "corrupt_frame", label: "Corrupt frame", where: "analyze" },
  { code: "tool_failed", label: "Evidence tool fails", where: "analyze" },
  { code: "tool_timeout", label: "Evidence tool hangs", where: "analyze" },
  { code: "no_detection", label: "Empty frame", where: "analyze" },
  { code: "missing_gps", label: "No GPS", where: "survey" },
  { code: "invalid_geometry", label: "Bad GPS fix", where: "survey" },
  { code: "storage_failed", label: "S3 upload fails", where: "survey" },
  { code: "llm_failed", label: "LLM outage", where: "brief" },
];
const Drills = {
  init() {
    const el = $("#drills"); if (!el) return;
    if (!new URLSearchParams(location.search).has("drills")) return;   // presenter-only: open with ?drills
    $("#drillBox").hidden = false;
    el.innerHTML = DRILLS.map(d => `<button class="drill" data-code="${d.code}" data-close="1"><b>${d.label}</b><span>${d.where === "brief" ? "survey brief" : d.where}</span></button>`).join("");
    el.onclick = e => { const b = e.target.closest(".drill"); if (b) this.run(DRILLS.find(d => d.code === b.dataset.code)); };
  },
  run(d) {
    Pop.closeAll();
    if (d.where === "analyze") {
      setMode("analyze");
      if (!state.selected) { const c = $(".sample-card"); if (c) c.click(); }
      if (state.selected) runAnalyze(d.code);
    } else if (d.where === "survey") { setMode("survey"); runSurvey(d.code); }
    else {
      setMode("survey");
      if (state.survey) { Tabs.go("Brief"); loadBrief(state.survey.survey_id, d.code); }
      else Notices.show([{ code: "drill", severity: "info", title: "Run a survey first", message: "The LLM drill rewrites a survey's mission brief.", fallback: "", human_action: "" }]);
    }
  },
};
