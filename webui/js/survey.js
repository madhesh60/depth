/* DEPTH studio — Survey: a run toward a goal, the Agent tab, the map, routes, downloads and the mission brief.
   Zero-build: index.html loads webui/js/*.js in order as classic scripts sharing one global scope;
   the boot runs on DOMContentLoaded (core.js), after every file has loaded. API: src/dashboard/routes/. */
"use strict";

/* ------------------------------------------------------------------ SURVEY */
async function runSurvey(simulate) {
  simulate = typeof simulate === "string" ? simulate : "";
  const btn = $("#surveyBtn"); btn.disabled = true;
  $("#surveyPlaceholder").hidden = true; $("#missionBanner").hidden = true;
  $("#surveyPlaceholder").classList.remove("ph-nogps"); $("#surveyPlaceholder").querySelector("h3").textContent = "Run a survey";
  Busy.start("Running survey");
  stepperRun(["stage1", "see", "prove", "decide", "act"]);
  Log.reset(); Log.now({ stage: true, tool: "SURVEY", msg: "running See→Prove→Decide→Act over all sample frames …", t: "run" });
  const gps = $("#gpsMode").value;
  const budget = Math.max(1, Math.min(600, parseFloat($("#budgetMin").value) || 2));
  const boat = parseFloat($("#boatMin").value);
  const boatQ = Number.isFinite(boat) && boat > 0 ? `&boat_minutes=${Math.min(1440, boat)}` : "";
  const recall = parseFloat(($("#goalRecall") || {}).value), comp = parseFloat(($("#goalCompute") || {}).value);
  const goalQ = (Number.isFinite(recall) ? `&recall=${recall}` : "") + (Number.isFinite(comp) && comp >= 1 ? `&compute_s=${Math.min(3600, comp)}` : "");
  try {
    // Surveys run as background jobs (no proxy timeouts, the server stays responsive); poll progress.
    const job = await apiJSON(`${API}/api/jobs/survey?use_samples=1&gps=${gps}&budget_minutes=${budget}${boatQ}${goalQ}${simulate ? "&simulate=" + simulate : ""}${Active.q()}`, { method: "POST" });
    Log.now({ tool: "queue_job", msg: `${job.job_id} · ${job.frames} frames queued`, t: "" });
    const d = await pollJob(job.job_id);
    state.survey = d;
    renderSurvey(d);
    Notices.show(surveyIncidents(d).filter(i => i.severity !== "info" || i.simulated), d.survey_id);
    showProvenance(d.provenance);
    if (Twin.sv === "3d") showSurveyTwin();
    loadMetrics();
    stepperFinish(["stage1", "see", "prove", "decide", "act"]);
  } catch (e) {
    stepperReset();
    if (e.incident) Notices.show([e.incident], "survey" + Date.now());
    $("#surveyPlaceholder").hidden = false;
    $("#surveyPlaceholder").querySelector("p").innerHTML = e.incident
      ? `<b>${escapeHtml(e.incident.title)}</b><br/>${escapeHtml(e.incident.fallback)}.`
      : `<b style="color:var(--danger)">Survey failed (${escapeHtml(e.message)}).</b><br/>The backend needs the model + local samples.`;
  } finally { Busy.done(); btn.disabled = false; }
}

async function pollJob(jobId) {
  let lastDone = -1;
  for (let i = 0; i < 1800; i++) {                       // ≤ ~15 min at 0.5 s
    const j = await fetch(`${API}/api/jobs/${encodeURIComponent(jobId)}`).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); });
    const p = j.progress || {};
    if (p.done !== lastDone && p.total) {
      lastDone = p.done;
      Busy.progress(p.done, p.total, `Frame ${Math.min(p.done + (j.status === "running" ? 1 : 0), p.total)} of ${p.total}`);
      if (p.done > 0) Log.now({ tool: "frame_done", msg: `${p.done}/${p.total}${p.frame ? " · " + p.frame : ""}`, t: "" });
    }
    if (j.status === "done") return j.result;
    if (j.status === "error") throw new Error(j.error || "job failed");
    await new Promise(r => setTimeout(r, 500));
  }
  throw new Error("timed out waiting for the survey job");
}

function renderSurvey(d, opts = {}) {
  $("#surveyPlaceholder").hidden = true;
  const m = d.mission, cc = m.counts || {};
  const gpsTag = !m.gps_available ? `<span class="mb-tag">No GPS — table only</span>`
    : m.gps_synthetic ? `<span class="mb-tag mb-synthetic">Synthetic demo GPS — not real positions</span>` : `<span class="mb-tag mb-real">Real GPS</span>`;
  const rp = m.resurvey_plan || {}, passes = (rp.lines || []).length, insp = (m.inspection_route || []).length;
  const total = (cc.confirmed || 0) + (cc.review || 0) + (cc.low_risk || 0);
  const tile = (label, value, sub) => `<div class="kpi-t"><div class="kpi-l">${label}</div><div class="kpi-v">${value}</div><div class="kpi-s">${sub}</div></div>`;
  const mb = $("#missionBanner"); mb.hidden = false; mb.className = "mission-banner mission-kpis";
  mb.innerHTML = `<div class="kpi-strip">
      ${tile("Hazards", total, `${cc.review || 0} for review · ${cc.confirmed || 0} confirmed`)}
      ${tile("Inspection stops", insp, m.inspection_length_m != null ? `${m.inspection_length_m} m · budgeted cards` : "no GPS — frame order")}
      ${tile("Re-survey passes", passes, passes ? `${rp.boat_minutes_planned} boat-min · opposite side` : "none planned")}
      ${tile("Recovery stops", m.recovery_route.length, (m.impact && m.impact.decisions)
        ? `${m.impact.person_confirmed} confirmed by people · ${m.impact.person_recovered} recovered`
        : (m.recovery_route.length ? `${m.route_length_m} m · confirmed finds` : "none yet — a person confirms"))}
    </div>
    <div class="mb-tags">${promiseTag(m.guarantees || {})}${gpsTag}${m.repeat_merges ? `<span class="mb-tag">${m.repeat_merges} repeat sighting${m.repeat_merges > 1 ? "s" : ""} merged</span>` : ""}<span class="mb-tag mb-gate">● Human approval required — nothing is dispatched automatically</span></div>`;
  renderMap(d, opts.keepView); renderDownloads(d.survey_id, m); renderThumbs(d); renderHazards(d); renderEffort(d);
  renderAgentPanel(d); loadAudit(d.survey_id);
  if (!opts.keepView) loadBrief(d.survey_id);
  if (!opts.keepView) Log.push([
    { stage: true, tool: "ACT", msg: `${cc.confirmed || 0} confirmed · ${cc.review || 0} review · ${cc.low_risk || 0} low-risk (kept for audit)`, t: "" },
    ...(m.budget && m.budget.cards_total != null ? [{ tool: "budget_plan", msg: `${m.budget.cards_affordable}/${m.budget.cards_total} review cards fit ${m.budget.minutes} min → ~${m.budget.expected_pots_in_budget} of ~${m.budget.expected_pots_in_queue} expected real pots`, t: "" }] : []),
    { tool: "plan_route", msg: `${m.recovery_route.length}-stop nearest-neighbour recovery route${m.route_length_m != null ? " · " + m.route_length_m + " m" : ""}`, t: "" },
    ...(m.resurvey_plan && m.resurvey_plan.lines ? [{ tool: "plan_resurvey", msg: `${m.resurvey_plan.lines.length} second-look passes · ${m.resurvey_plan.targets_covered}/${m.resurvey_plan.targets_total} uncertain targets · uncertainty Σp(1−p) ${m.resurvey_plan.voi_covered}/${m.resurvey_plan.voi_total} · ${m.resurvey_plan.boat_minutes_planned} min boat${m.resurvey_plan.boat_minutes_budget ? " (budget " + m.resurvey_plan.boat_minutes_budget + ")" : ""} — a real object's shadow must flip`, t: "" }] : []),
    ...(m.repeat_merges != null ? [{ tool: "merge_sightings", msg: `${m.repeat_merges} repeat sighting(s) merged (error circles overlap, different frames)`, t: "" }] : []),
    { tool: "human_gate", msg: "human approval required — nothing auto-dispatched", t: "" },
  ]);
}

/* this survey's own promise: the mode the agent chose, or withdrawn when it does not apply here */
function promiseTag(g) {
  if (g.promise_withdrawn) return `<span class="mb-tag mb-synthetic" title="${escapeHtml(g.promise_withdrawn.why || "")}">No recall promise — this recording does not look like the calibration recordings</span>`;
  if (g.recall_promise == null) return "";
  return `<span class="mb-tag mb-real">Promise ≥ ${Math.round(g.recall_promise * 100)}%${g.perception_mode && g.perception_mode !== "standard" ? " · " + human(g.perception_mode) : ""}</span>`;
}
function surveyIncidents(d) {
  return [...((d.mission || {}).incidents || []), ...(d.frames || []).flatMap(f => f.incidents || [])];
}
const ACTION_LABEL = { inspect: "Inspection route", recover: "Recovery route", resurvey: "Re-survey passes", full_review: "Full review" };
/* the Agent tab: dispatch gate, last re-plan, incidents, and the planner's decisions step by step */
function renderAgentPanel(d) {
  const m = d.mission, disp = d.dispatch || {}, incs = surveyIncidents(d);
  const rp = (m.replans || []).slice(-1)[0];
  const gate = ["full_review", "inspect", "recover"].filter(a => disp[a]).map(a => `<div class="gate-row">
      <span class="gate-a">${ACTION_LABEL[a]}</span><span class="gate-n">${disp[a].targets} target${disp[a].targets === 1 ? "" : "s"}</span>
      <span class="st ${disp[a].status}">${disp[a].status}${disp[a].by ? " · " + escapeHtml(disp[a].by) : ""}</span></div>`).join("");
  const pending = Object.values(disp).filter(x => x.status === "pending").length;
  $("#agentMeta").textContent = `${(m.agent_log || []).length} steps${incs.length ? ` · ${incs.length} incident${incs.length > 1 ? "s" : ""}` : ""}`;
  $("#agentPanel").innerHTML = `
    ${agentRunBlock(d.agent_run)}
    ${recoCards(d)}
    ${seabedBlock(d)}
    ${cfTable(d.opencv_counterfactual)}
    <div class="gate">
      <div class="gate-h">Dispatch gate<span>${pending ? `${pending} waiting for a person` : "nothing waiting"}</span></div>
      ${gate || `<p class="empty-hint">Nothing to dispatch.</p>`}
      <p class="gate-note">The agent asks; only a named person approves. The LLM writes the brief and never decides.</p>
      ${pending ? `<button class="mini-btn" id="toApprovals">Review approvals</button>` : ""}
    </div>
    ${rp ? `<div class="replan"><div class="gate-h">Last re-plan<span>${new Date(rp.t * 1000).toLocaleTimeString()}</span></div><p>${escapeHtml(chainText({ rationale: rp.rationale }))}</p>
      ${(rp.requests || []).length ? `<p class="muted">${rp.requests.map(c => c.requested ? `asked again: ${c.action} (${c.requested})` : `withdrew ${c.withdrawn}`).join(" · ")}</p>` : ""}</div>` : ""}
    ${incs.length ? `<div class="inc-list"><div class="gate-h">Incidents<span>${incs.length}</span></div>${incs.map(incidentRow).join("")}</div>` : ""}
    <div class="gate-h plan-h">Plan, step by step</div>${chainBlock(m.agent_log || [], PLAN_STEP)}`;
  const ta = $("#toApprovals"); if (ta) ta.onclick = () => Tabs.go("Approvals");
  const geo = d.recording && d.recording.geometry;
  if (geo) {
    const show = fid => {
      const c = geo.chunks.find(x => x.frame_id === fid); if (!c) return;
      $$("#agentPanel .geo-chip").forEach(b => b.classList.toggle("is-on", b.dataset.f === fid));
      $("#geoDetail").innerHTML = `<figure class="geo-fig"><img src="${API}/api/recording/geometry/${encodeURIComponent(fid)}.jpg" alt="seabed cross-check overlay for ${escapeHtml(fid)}" decoding="async"/>
        <figcaption><span class="lg lg-t"></span>tracker alone <span class="lg lg-s"></span>depth sounder <span class="lg lg-a"></span>accepted by the agent</figcaption></figure>
        <p class="geo-why"><b>${GEO_LABEL[c.status] || human(c.status)}</b>${c.source ? ` · ${escapeHtml(c.source)}` : ""}. ${escapeHtml(chainText({ rationale: c.meaning }))}.</p>
        ${chainBlock(c.steps || [], SENSE_STEP)}`;
    };
    $$("#agentPanel .geo-chip").forEach(b => b.onclick = () => show(b.dataset.f));
    const first = geo.chunks.find(x => x.status !== "agree") || geo.chunks[0];
    if (first) show(first.frame_id);
  }
  $$("#agentPanel .reco [data-d]").forEach(b => b.onclick = () => Appr.decide(b.closest(".reco").dataset.id, b.dataset.d));
}
/* "Seabed cross-check" - per chunk, did the OpenCV seabed track agree with the depth sounder, and what
   did the agent do when it did not (re-track, trust the image, withhold). Chips open the overlay + steps. */
function seabedBlock(d) {
  const g = d.recording && d.recording.geometry;
  if (!g || !(g.chunks || []).length) return "";
  const n = g.counts || {}, it = (g.log || []).map(s => chainText(s)).join(" ");
  const order = ["port", "starboard"];
  const chips = order.map(ch => {
    const cs = g.chunks.filter(c => c.channel === ch).sort((a, b) => a.index - b.index);
    return cs.length ? `<div class="geo-row"><span class="geo-ch">${ch === "port" ? "Port" : "Starboard"}</span>${cs.map(c =>
      `<button class="geo-chip geo-${c.status}" data-f="${escapeHtml(c.frame_id)}" title="${escapeHtml((GEO_LABEL[c.status] || c.status) + " — chunk " + c.index)}">${c.index}</button>`).join("")}</div>` : "";
  }).join("");
  const parts = Object.keys(GEO_LABEL).filter(k => n[k]).map(k => `<span class="geo-k geo-${k}"><i></i>${n[k]} ${GEO_LABEL[k].toLowerCase()}</span>`).join("");
  return `<div class="gate geo">
    <div class="gate-h">Seabed cross-check<span>${g.trusted} of ${g.chunks.length} chunks measured</span></div>
    <p class="geo-lead">Every metre on the map hangs on the sonar's altitude. The agent checks OpenCV's seabed track against the depth sounder,
      re-runs the tracker where they conflict, trusts the image where the sounder lost lock, and withholds what neither supports.</p>
    <div class="geo-legend">${parts}</div>
    ${chips}
    <p class="gate-note">${escapeHtml(it)}</p>
    <div id="geoDetail"></div>
  </div>`;
}
/* "Agent recommends" - one card per opposite-side pass, each approved or rejected on its own */
function recoCards(d) {
  const lines = ((d.mission || {}).resurvey_plan || {}).lines || [], disp = d.dispatch || {};
  if (!lines.length) return "";
  return `<div class="recos">${lines.map(L => {
    const r = disp["resurvey:" + L.targets.join(",")] || {};
    const st = r.status || "planned";
    return `<div class="reco" data-id="${escapeHtml(r.id || "")}">
      <div class="reco-k">Agent recommends</div>
      <div class="reco-t">${escapeHtml(L.recommendation || "Perform opposite-side resurvey")} <span class="mono">${escapeHtml(L.id)}</span></div>
      <div class="reco-r"><span>Reason</span>${escapeHtml(L.reason || "uncertain visual evidence")}</div>
      <div class="reco-m">${L.targets.length} target${L.targets.length === 1 ? "" : "s"} (${escapeHtml(L.targets.slice(0, 5).join(", "))}${L.targets.length > 5 ? "…" : ""}) · ${L.boat_min} boat-min${L.synthetic ? " · synthetic GPS" : ""}</div>
      ${st === "pending" && r.id ? `<div class="appr-btns"><button class="ok" data-d="approved">Approve</button><button class="no" data-d="declined">Reject</button></div>`
        : `<div class="reco-s st ${st}">${st}${r.by ? " · " + escapeHtml(r.by) : ""}${st === "approved" ? " · work order issued; plan unchanged until the pass data is surveyed" : ""}</div>`}
    </div>`;
  }).join("")}</div>`;
}
/* the same survey WITH vs WITHOUT the OpenCV evidence - what changes downstream */
function cfTable(cf) {
  if (!cf || !cf.available) return "";
  return `<div class="cfx"><div class="gate-h">Does OpenCV's measurement change the plan?<span>at planning time · ${cf.active ? "active vision (experimental)" : "active vision off"}</span></div>
    <table><thead><tr><th></th><th title="the same cv2.dnn detections, nothing OpenCV measured afterwards">Detector only</th><th>+ Stage 1 + OpenCV evidence</th></tr></thead><tbody>${cf.rows.map(r =>
      `<tr class="${r.changed ? "chg" : ""}"><th>${escapeHtml(r.what)}</th><td>${escapeHtml(r.without)}</td><td>${escapeHtml(r.with)}<div class="cfx-d">${escapeHtml(r.detail)}</div></td></tr>`).join("")}</tbody></table>
    ${cf.example ? `<p class="gate-note"><b>${escapeHtml(cf.example.id)}</b>: ${escapeHtml(cf.example.conflict || "")} — ${ACT[cf.example.without]} → ${ACT[cf.example.with]}${cf.example.resurvey ? ", opposite-side pass requested" : ""}.</p>` : ""}
    <p class="gate-note">${escapeHtml(cf.note)}</p></div>`;
}
/* the audit log: agent, person, system and LLM entries on one timeline */
async function loadAudit(surveyId) {
  try {
    const j = await apiJSON(`${API}/api/survey/${encodeURIComponent(surveyId)}/log`);
    const E = j.entries || [];
    const cnt = a => E.filter(e => e.actor === a).length;
    $("#auditMeta").textContent = `${E.length} entries`;
    $("#auditLog").innerHTML = `<div class="audit-sum"><span class="who who-agent">Agent ${cnt("agent")}</span><span class="who who-person">Person ${cnt("person")}</span>`
      + `<span class="who who-system">System ${cnt("system")}</span>${cnt("llm") ? `<span class="who who-llm">LLM ${cnt("llm")}</span>` : ""}</div>`
      + `<ol class="tl">${E.slice().reverse().map(e => `<li class="tl-i tl-${e.actor}${e.severity ? " sev-" + e.severity : ""}">
          <div class="tl-h"><span class="who who-${e.actor}">${human(e.actor)}</span><b>${escapeHtml(human(e.title || e.kind))}</b>
          <time>${new Date(e.t * 1000).toLocaleTimeString()}</time></div>
          ${e.text ? `<div class="tl-t">${escapeHtml(chainText({ tool: e.kind, rationale: e.text }))}</div>` : ""}${e.by ? `<div class="tl-by">by ${escapeHtml(e.by)}</div>` : ""}</li>`).join("")}</ol>`
      + `<p class="gate-note">${escapeHtml(j.llm_authority || "")}</p>`;
  } catch (e) { $("#auditLog").innerHTML = `<p class="empty-hint">${escapeHtml(e.message)}</p>`; }
}

function renderMap(d, keepView) {
  const m = d.mission, geoObjs = d.tracked.filter(t => t.lat != null && t.lon != null), note = $("#mapNote");
  const ph = $("#surveyPlaceholder");
  if (!m.gps_available || !geoObjs.length) {
    $("#mapWrap").style.display = "none"; $("#cfBtn").hidden = $("#cfSum").hidden = true; CF.clear();
    ph.hidden = false; ph.classList.add("ph-nogps");                 // say why the map is empty, never a blank void
    ph.querySelector("h3").textContent = "Positions withheld";
    ph.querySelector("p").textContent = !m.gps_available
      ? "This survey has no GPS, so no map, route or re-survey pass is drawn. The hazards are listed in frame order under Hazards."
      : "No position passed the geometry check. The hazards are still in the queue under Hazards.";
    return;
  }
  ph.classList.remove("ph-nogps");
  $("#mapWrap").style.display = "";
  const rc = d.recording && d.recording.available ? d.recording : null;
  note.textContent = m.gps_synthetic
    ? "Synthetic demo track: these crab-pot frames carry no GPS, so positions are illustrative."
    : rc ? `Real per-ping GPS — raw Humminbird recording ${rc.name} (PINGMapper sample, ${rc.duration_s} s, ${rc.track_m} m of track). `
      + (rc.range_scale && rc.range_scale.m_per_sample ? `Range scale measured: ${(rc.range_scale.m_per_sample * 100).toFixed(2)} cm/sample ±${Math.round(rc.range_scale.rel_unc * 100)}% (sonar depth ÷ Stage-1 altitude). ` : "")
      + "No crab pots are known in this river — its cards measure false alarms."
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
    const hc = { confirmed: "#9ecbff", recovered: "#9ecbff", rejected: "#7f8a97", not_found: "#7f8a97" }[t.human];
    const color = hc || VCOLOR[t.verdict] || "#8ba6c2";
    const onRoute = t.human === "confirmed" || (t.verdict === "confirmed" && !t.human);
    const mk = L.circleMarker([t.lat, t.lon], { radius: onRoute ? 8 : (t.human === "rejected" || t.human === "not_found") ? 4 : 6,
        color: onRoute ? "#ffffff" : "#02121d", weight: onRoute ? 2 : 1.5, fillColor: color, fillOpacity: (t.human === "rejected" || t.human === "not_found") ? .5 : .95 })
      .bindPopup(`<b>${t.oid}</b> · ${VLABEL[t.verdict] || human(t.verdict)}<br/>${human(t.cls_name)} · conf ${t.conf}<br/>evidence ${t.evidence_score} · ${t.shadow_quality} shadow${t.height_m != null || t.height_rel ? ` · h ${heightText(t.height_m, t.height_rel, 1)}` : ""}<br/>${t.lat.toFixed(5)}, ${t.lon.toFixed(5)} ±${t.geo_error_m ?? "?"} m${t.human ? `<br/><b>person: ${t.human}</b> (${escapeHtml(t.human_by || "")})` : ""}`);
    if (t.geo_error_m) {
      const ring = L.circle([t.lat, t.lon], { radius: t.geo_error_m, color, weight: 1, opacity: .55, fillOpacity: .06, dashArray: "3 4", interactive: false });
      ring.addTo(state.map); state.mapLayers.push(ring);
    }
    mk.addTo(state.map); state.mapLayers.push(mk); latlngs.push([t.lat, t.lon]);
  });
  const routePts = m.recovery_route.map(id => byId[id]).filter(t => t && t.lat != null).map(t => [t.lat, t.lon]);
  if (routePts.length > 1) {
    const line = L.polyline(routePts, { color: "#9ecbff", weight: 2.2, dashArray: "6 6", opacity: .9 }).addTo(state.map); state.mapLayers.push(line);
    routePts.forEach((p, i) => {
      const badge = L.marker(p, { icon: L.divIcon({ className: "", html: `<div style="background:#9ecbff;color:#061018;font:600 10px/18px var(--sans,sans-serif);width:18px;height:18px;border-radius:50%;text-align:center;border:1.5px solid rgba(7,26,43,.6);box-shadow:0 2px 6px rgba(0,0,0,.4)">${i + 1}</div>`, iconSize: [18, 18], iconAnchor: [9, 9] }) }).addTo(state.map);
      state.mapLayers.push(badge);
    });
  }
  const inspPts = (m.inspection_route || []).map(id => byId[id]).filter(t => t && t.lat != null).map(t => [t.lat, t.lon]);
  if (inspPts.length > 1) {
    const il = L.polyline(inspPts, { color: "#e6c48a", weight: 2, dashArray: "2 6", opacity: .85 })
      .bindTooltip("Inspection route: review cards to check first (pending human approval)").addTo(state.map);
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
  state.mapBounds = L.latLngBounds(latlngs).pad(0.25);
  if (!keepView) { fitMap(); setTimeout(fitMap, 80); }
  const cf = d.stage1_counterfactual;
  $("#cfBtn").hidden = !(cf && cf.available);
  CF.draw(d);
}
/* STUDY-12, live: ghost pins where the boat would go if the agent ignored Stage 1's bottom track */
const CF = {
  on: false, layers: [],
  clear() { this.layers.forEach(l => state.map && state.map.removeLayer(l)); this.layers = []; },
  draw(d) {
    this.clear();
    const cf = d && d.stage1_counterfactual, sum = $("#cfSum");
    $("#cfBtn").classList.toggle("is-on", this.on);
    if (!this.on || !cf || !cf.available || !state.map) { sum.hidden = true; return; }
    const byId = Object.fromEntries(d.tracked.map(t => [t.oid, t]));
    Object.entries(cf.pins).forEach(([oid, p]) => {
      const t = byId[oid]; if (!t || t.lat == null) return;
      const col = p.outside ? "#e5789b" : "#9aa8b5";
      const link = L.polyline([[t.lat, t.lon], [p.lat, p.lon]], { color: col, weight: 1.2, opacity: .8, dashArray: "2 3", interactive: false });
      const ghost = L.circleMarker([p.lat, p.lon], { radius: 5, color: col, weight: 1.5, fill: false, dashArray: "2 2" })
        .bindTooltip(`<b>${oid}</b> without Stage 1: ${p.shift_m} m away${p.outside ? ` — <b>outside</b> its ±${t.geo_error_m} m circle` : ` (inside ±${t.geo_error_m} m)`}`);
      [link, ghost].forEach(l => { l.addTo(state.map); this.layers.push(l); });
    });
    sum.hidden = false;
    sum.innerHTML = `without Stage 1: <b>${cf.outside} of ${cf.n}</b> pins leave their own error circle · median shift <b>${cf.median_shift_m} m</b> (max ${cf.max_shift_m}) · tiers unchanged`;
  },
  toggle() { this.on = !this.on; this.draw(state.survey); },
};
/* size first, then fit: a container measured while hidden/0-px makes Leaflet fit the whole world.
   Read the element, not map.getSize() — Leaflet caches that, and invalidateSize() is a no-op until the
   map has a view, so one early 0-px read would stick. */
function fitMap() {
  if (!state.map || !state.mapBounds || !state.map.getContainer().clientWidth) return;
  state.map.invalidateSize();
  state.map.fitBounds(state.mapBounds);
}
/* re-fit only when the view is lost (never fitted, or world zoom) — a user's own pan/zoom is kept */
function refitIfLost() {
  if (!state.map) return;
  state.map.invalidateSize();
  if (!(state.map.getZoom() >= 4)) fitMap();
}
/* the map element's real size (0 while its mode/pane is hidden) drives the re-fit — window events miss
   layout changes inside embedded panes */
if (window.ResizeObserver) new ResizeObserver(() => refitIfLost()).observe(document.getElementById("map"));

function renderDownloads(surveyId) {
  const fmts = [["brief", "mission brief (markdown)"], ["geojson", "hazards + route (GIS)"], ["gpx", "waypoints + re-survey (boat GPS)"], ["kml", "Google Earth"], ["csv", "spreadsheet"], ["json", "full result + provenance"], ["trace", "agent decision log (JSONL)"]];
  const draw = () => {
    const pub = $("#pubShare").checked;
    $("#dlGrid").innerHTML = fmts.map(([f, desc]) => `<button class="dl-btn${pub && f === "trace" ? " is-off" : ""}" data-fmt="${f}" title="${pub && f === "trace" ? "the decision log carries exact positions — never shared publicly" : ""}"><b>.${({ trace: "trace.jsonl", brief: "brief.md" })[f] || f}</b><span>${desc}${pub && f !== "trace" ? " · public" : ""}</span></button>`).join("");
    $$("#dlGrid .dl-btn").forEach(b => b.onclick = () => {
      const a = document.createElement("a");
      a.href = `${API}/api/report/${b.dataset.fmt}?survey_id=${encodeURIComponent(surveyId)}${pub ? "&public=1" : ""}`;
      a.download = `mission${pub ? ".public" : ""}.${({ trace: "trace.jsonl", brief: "brief.md" })[b.dataset.fmt] || b.dataset.fmt}`; document.body.appendChild(a); a.click(); a.remove();
    });
  };
  $("#pubShare").onchange = () => { draw(); loadBrief(surveyId); }; draw();
}
/* mission brief: the agent's hand-over. Written by the template or (when enabled) Claude on Bedrock —
   either way every number is checked against the survey facts server-side before it is shown. */
function mdLite(md) {
  const inl = s => escapeHtml(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/\b(H\d{2,4}|RS\d{1,3})\b/g, '<span class="hz">$1</span>');
  const out = []; let list = false;
  const close = () => { if (list) { out.push("</ul>"); list = false; } };
  md.split(/\r?\n/).forEach(line => {
    const t = line.trim();
    if (!t) { close(); return; }
    if (t.startsWith("# ")) { close(); out.push(`<h3>${inl(t.slice(2))}</h3>`); }
    else if (t.startsWith("## ")) { close(); out.push(`<h4>${inl(t.slice(3))}</h4>`); }
    else if (/^[-*] /.test(t)) { if (!list) { out.push("<ul>"); list = true; } out.push(`<li>${inl(t.slice(2))}</li>`); }
    else if (/^_.*_$/.test(t)) { close(); out.push(`<div class="brief-sub">${inl(t.slice(1, -1))}</div>`); }
    else { close(); out.push(`<p>${inl(t)}</p>`); }
  });
  close(); return out.join("");
}
async function loadBrief(surveyId, simulate) {
  const pub = $("#pubShare") && $("#pubShare").checked;
  $("#briefMeta").textContent = "writing …";
  try {
    const b = await apiJSON(`${API}/api/brief?survey_id=${encodeURIComponent(surveyId)}${pub ? "&public=1" : ""}${simulate ? "&simulate=" + simulate : ""}`);
    const g = b.grounding || {};
    if (b.incident) { Notices.show([b.incident], surveyId + Date.now()); loadAudit(surveyId); }
    $("#briefBody").innerHTML = mdLite(b.text || "");
    $("#briefMeta").innerHTML = `<span class="writer-tag ${b.writer === "llm" ? "llm" : ""}">${b.writer === "llm" ? "Claude · Bedrock" : "template"}</span>${pub ? " · public" : ""}`;
    $("#briefGround").innerHTML = `<b class="${g.ok ? "ok" : "warn"}">${g.ok ? "✓" : "!"} ${g.numbers_checked} numbers traced to the survey</b> · ${g.words} words`
      + (b.fallback_reason ? ` · <span title="${escapeHtml(b.fallback_reason)}">LLM fell back to the template</span>` : "");
    $("#briefFoot").hidden = false;
    $("#briefCopy").onclick = () => { navigator.clipboard && navigator.clipboard.writeText(b.text); $("#briefCopy").textContent = "copied ✓"; setTimeout(() => $("#briefCopy").textContent = "copy", 1400); };
    Log.push([{ tool: "write_brief", msg: `mission brief by ${b.writer === "llm" ? b.model : "template"} · ${g.numbers_checked} numbers checked against the survey · ${g.ok ? "grounded" : "NOT grounded"}${b.fallback_reason ? " (LLM fell back: " + b.fallback_reason + ")" : ""}`, t: "" }]);
  } catch (e) {
    $("#briefMeta").textContent = "unavailable"; $("#briefBody").innerHTML = `<p class="empty-hint">brief unavailable (${escapeHtml(String(e.message || e))})</p>`;
  }
}
