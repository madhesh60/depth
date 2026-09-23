/* GhostGear Sonar dashboard — drives the See → Prove → Decide → Act loop over the FastAPI backend.
   Zero-build vanilla JS. All API shapes match src/dashboard/app.py. */
"use strict";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const API = ""; // same-origin (served by FastAPI). Set to e.g. "http://localhost:8000" for split dev.

const VERDICTS = ["confirmed", "review", "rejected"];
const VCOLOR = { confirmed: "#2ee06a", review: "#f7a83b", rejected: "#6c8199" };

const state = {
  samples: [],
  selected: null,        // {id} sample or {file}
  survey: null,          // last SurveyResult
  map: null,
  mapLayers: [],
};

/* ------------------------------------------------------------------ boot */
window.addEventListener("DOMContentLoaded", async () => {
  wireTabs();
  wireImageToggle();
  wireInputs();
  await Promise.all([loadHealth(), loadSamples()]);
});

async function loadHealth() {
  try {
    const h = await fetch(`${API}/api/health`).then(r => r.json());
    const cv = $("#pillCv"), md = $("#pillModel");
    cv.textContent = `OpenCV ${h.opencv}`;
    cv.className = "pill " + (String(h.opencv).startsWith("5") ? "pill-ok" : "pill-muted");
    md.textContent = h.model_loaded ? "model ready" : `model lazy · ${h.samples} samples`;
    md.className = "pill " + (h.model_loaded ? "pill-ok" : "pill-muted");
    $("#footVer").textContent = `OpenCV ${h.opencv} · cv2.dnn · ${h.samples} samples · running on the demo server`;
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
  if (!state.samples.length) {
    grid.innerHTML = `<p class="hint" style="grid-column:1/-1">No local samples — drop your own sonar frame below.</p>`;
    return;
  }
  grid.innerHTML = "";
  state.samples.forEach(s => {
    const card = document.createElement("button");
    card.className = "sample-card";
    card.dataset.id = s.id;
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
  d.style.cssText = "aspect-ratio:1;display:grid;place-items:center;background:radial-gradient(circle at 50% 35%,#123049,#0a1626);color:#35d6f2";
  d.innerHTML = `<svg viewBox="0 0 24 24" width="30" height="30" opacity=".7"><path d="M12 3a9 9 0 1 0 9 9h-9V3Z" fill="none" stroke="currentColor" stroke-width="1.4"/><circle cx="12" cy="12" r="2" fill="currentColor"/></svg>`;
  return d;
}

function selectSample(id, card) {
  state.selected = { id };
  $$(".sample-card").forEach(c => c.classList.toggle("is-sel", c === card));
  const s = state.samples.find(x => x.id === id);
  $("#analyzeBtn").disabled = false;
  $("#analyzeHint").textContent = `${s.name} — ${s.kind}. Ready to run.`;
}

/* ------------------------------------------------------------------ tabs + inputs */
function wireTabs() {
  $$(".tab[data-tab]").forEach(t => t.onclick = () => {
    $$(".tab[data-tab]").forEach(x => x.classList.toggle("is-active", x === t));
    $$(".panel-tab").forEach(p => p.classList.toggle("is-active", p.id === `tab-${t.dataset.tab}`));
    if (t.dataset.tab === "survey" && state.map) setTimeout(() => state.map.invalidateSize(), 60);
  });
}

function wireImageToggle() {
  $$(".seg-btn").forEach(b => b.onclick = () => {
    $$(".seg-btn").forEach(x => x.classList.toggle("is-active", x === b));
    const d = state._analyze;
    if (d) $("#stageImg").src = b.dataset.view === "frame" ? d.frame_png : d.overlay_png;
  });
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

function pickFile(file) {
  state.selected = { file };
  $$(".sample-card").forEach(c => c.classList.remove("is-sel"));
  $("#analyzeBtn").disabled = false;
  $("#analyzeHint").textContent = `Uploaded ${file.name}. Ready to run.`;
}

/* ------------------------------------------------------------------ stepper animation */
let stepperTimer = null;
function stepperRun(stages) {
  stepperReset();
  let i = 0;
  const steps = stages;
  const tick = () => {
    $$(".step").forEach(s => s.classList.remove("is-active"));
    steps.slice(0, i).forEach(st => setStage(st, "done"));
    if (i < steps.length) setStage(steps[i], "active");
    i = (i + 1);
    if (i > steps.length) i = steps.length; // hold on last
  };
  tick();
  stepperTimer = setInterval(tick, 550);
}
function stepperFinish(stages) {
  clearInterval(stepperTimer); stepperTimer = null;
  $$(".step").forEach(s => s.classList.remove("is-active"));
  stages.forEach(st => setStage(st, "done"));
}
function stepperReset() {
  clearInterval(stepperTimer); stepperTimer = null;
  $$(".step").forEach(s => s.classList.remove("is-active", "is-done"));
}
function setStage(name, mode) {
  const el = $(`.step[data-stage="${name}"]`);
  if (!el) return;
  if (mode === "active") { el.classList.add("is-active"); el.classList.remove("is-done"); }
  else if (mode === "done") { el.classList.add("is-done"); el.classList.remove("is-active"); }
}

function toast(txt) { const t = $("#runToast"); $("#runToastTxt").textContent = txt; t.hidden = false; }
function toastHide() { $("#runToast").hidden = true; }

/* ------------------------------------------------------------------ ANALYZE */
async function runAnalyze() {
  if (!state.selected) return;
  $("#analyzeBtn").disabled = true;
  $("#analyzePlaceholder").hidden = true;
  $("#analyzeResult").hidden = true;
  toast("See → Prove → Decide …");
  stepperRun(["see", "prove", "decide"]);

  try {
    let url = `${API}/api/analyze`, opts = { method: "POST" };
    if (state.selected.file) {
      const fd = new FormData(); fd.append("file", state.selected.file);
      opts.body = fd;
    } else {
      url += `?sample=${encodeURIComponent(state.selected.id)}`;
    }
    const d = await fetch(url, opts).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); });
    state._analyze = d;
    renderAnalyze(d);
    stepperFinish(["see", "prove", "decide"]);
  } catch (e) {
    stepperReset();
    $("#analyzePlaceholder").hidden = false;
    $("#analyzePlaceholder").querySelector("p").innerHTML =
      `<b style="color:var(--danger)">Analyze failed (${e.message}).</b><br/>Is the backend running and the model present?`;
  } finally {
    toastHide();
    $("#analyzeBtn").disabled = false;
  }
}

function renderAnalyze(d) {
  $("#analyzeResult").hidden = false;
  $$(".seg-btn").forEach(x => x.classList.toggle("is-active", x.dataset.view === "overlay"));
  $("#stageImg").src = d.overlay_png;
  $("#stageCap").textContent =
    `${d.frame_id} · ${d.width}×${d.height}px · nadir=${d.nadir} · ${d.candidates.length} candidate(s)`;

  $("#verdictCounts").innerHTML = VERDICTS.map(v =>
    `<span class="vc vc-${v}"><span class="n">${d.counts[v] || 0}</span>${v}</span>`).join("");

  const sm = d.stage_ms || {};
  $("#latency").textContent =
    `⏱ ${d.wall_ms} ms wall · see ${fmt(sm.see)} ms · prove+decide ${fmt(sm.prove_decide)} ms`;

  const grid = $("#evidenceGrid");
  grid.innerHTML = "";
  if (!d.candidates.length) {
    grid.innerHTML = `<p class="hint">No candidates on this frame — the detector found nothing to prove. (For the "no labelled pot" samples, that is the correct, honest result.)</p>`;
    return;
  }
  // sort: confirmed → review → rejected, then by evidence score
  const order = { confirmed: 0, review: 1, rejected: 2 };
  [...d.candidates]
    .sort((a, b) => (order[a.verdict] - order[b.verdict]) ||
                    ((b.evidence?.evidence_score || 0) - (a.evidence?.evidence_score || 0)))
    .forEach((c, i) => grid.appendChild(evidenceCard(c, i)));
}

function evidenceCard(c, i) {
  const ev = c.evidence || {};
  const sh = ev.shadow || {};
  const rl = ev.relook || {};
  const v = c.verdict;
  const card = document.createElement("div");
  card.className = `ev-card v-${v}`;

  const rlConf = rl.conf || 0, gain = (rlConf - c.conf);
  const arrowCls = rlConf > c.conf ? "up" : (rlConf < c.conf ? "down" : "");
  const shCls = sh.quality === "clear" ? "chip-clear" : sh.quality === "weak" ? "chip-weak" : "chip-none";
  const height = sh.height_m != null ? `${sh.height_m} m` : "—";

  card.innerHTML = `
    <div class="ev-crop">
      ${c.crop_png ? `<img alt="evidence crop" src="${c.crop_png}"/>` : `<div style="aspect-ratio:1"></div>`}
      <span class="ev-badge b-${v}">${v}</span>
    </div>
    <div class="ev-body">
      <div class="conf-flow">
        <div class="conf-chip"><span class="lbl">detector</span><span class="val">${(c.conf).toFixed(2)}</span></div>
        <div class="conf-arrow ${arrowCls}">
          <div class="track"></div>
          <span class="tag">re-look ${rl.found ? `${rlConf.toFixed(2)} (${gain >= 0 ? "+" : ""}${gain.toFixed(2)})` : "no re-fire"}</span>
        </div>
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
  return card;
}

function traceBlock(trace) {
  if (!trace.length) return "";
  const steps = trace.map(s => `
    <li class="trace-step ${s.tool === "decide" ? "t-decide" : ""}">
      <span class="trace-tool">${s.tool}</span><span class="trace-ms">${fmt(s.latency_ms)} ms</span>
      <div class="trace-why">${escapeHtml(s.rationale)}</div>
    </li>`).join("");
  return `<details class="trace"><summary>agent trace — ${trace.length} tool calls</summary><ol>${steps}</ol></details>`;
}

/* ------------------------------------------------------------------ SURVEY */
async function runSurvey() {
  const btn = $("#surveyBtn");
  btn.disabled = true;
  $("#surveyPlaceholder").hidden = true;
  $("#surveyResult").hidden = true;
  toast("Running full loop over the survey …");
  stepperRun(["see", "prove", "decide", "act"]);
  const gps = $("#gpsMode").value;

  try {
    const d = await fetch(`${API}/api/survey?use_samples=1&gps=${gps}`, { method: "POST" })
      .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); });
    state.survey = d;
    renderSurvey(d);
    stepperFinish(["see", "prove", "decide", "act"]);
  } catch (e) {
    stepperReset();
    $("#surveyPlaceholder").hidden = false;
    $("#surveyPlaceholder").querySelector("p").innerHTML =
      `<b style="color:var(--danger)">Survey failed (${e.message}).</b><br/>The backend needs the model + local samples.`;
  } finally {
    toastHide();
    btn.disabled = false;
  }
}

function renderSurvey(d) {
  $("#surveyResult").hidden = false;
  const m = d.mission, cc = m.counts || {};

  // mission banner
  const gpsTag = !m.gps_available ? `<span class="mb-tag">no GPS — table only</span>`
    : m.gps_synthetic ? `<span class="mb-tag mb-synthetic">⚠ synthetic demo GPS</span>`
    : `<span class="mb-tag mb-real">real GPS</span>`;
  $("#missionBanner").className = "mission-banner mb-approve";
  $("#missionBanner").innerHTML = `
    <b>🧭 Mission plan ready</b>
    <span>${cc.confirmed || 0} confirmed → <b>${m.recovery_route.length}-stop recovery route</b>${m.route_length_m != null ? ` (${m.route_length_m} m)` : ""}</span>
    <span>· ${cc.review || 0} to re-survey</span>
    ${gpsTag}
    <span class="mb-tag" style="border-color:rgba(247,168,59,.5);color:#ffd9a3">✋ human approval required — nothing auto-dispatched</span>`;

  renderMap(d);
  renderDownloads(d.survey_id, m);
  renderThumbs(d);
  renderHazards(d);
}

function renderMap(d) {
  const m = d.mission;
  const geoObjs = d.tracked.filter(t => t.lat != null && t.lon != null);
  const note = $("#mapNote");

  if (!m.gps_available || !geoObjs.length) {
    $(".map-wrap").style.display = "none";
    return;
  }
  $(".map-wrap").style.display = "";
  note.textContent = m.gps_synthetic
    ? "⚠ SYNTHETIC DEMO GPS — not real coordinates. The HF crab-pot frames carry no GPS; this track is generated only to demonstrate the map + route."
    : "Real per-ping GPS.";

  if (!state.map) {
    state.map = L.map("map", { zoomControl: true, attributionControl: true });
    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
      attribution: '&copy; OpenStreetMap &copy; CARTO', maxZoom: 20, subdomains: "abcd",
    }).addTo(state.map);
  }
  state.mapLayers.forEach(l => state.map.removeLayer(l));
  state.mapLayers = [];

  const byId = Object.fromEntries(d.tracked.map(t => [t.oid, t]));
  const latlngs = [];
  geoObjs.forEach(t => {
    const color = VCOLOR[t.verdict] || "#8ba6c2";
    const mk = L.circleMarker([t.lat, t.lon], {
      radius: t.verdict === "confirmed" ? 8 : 6, color: "#02121d", weight: 1.5,
      fillColor: color, fillOpacity: .95,
    }).bindPopup(
      `<b>${t.oid}</b> · ${t.verdict}<br/>${t.cls_name} · conf ${t.conf}<br/>` +
      `evidence ${t.evidence_score} · ${t.shadow_quality} shadow` +
      (t.height_m != null ? ` · h~${t.height_m} m` : "") +
      `<br/>${t.lat.toFixed(5)}, ${t.lon.toFixed(5)} ±${t.geo_error_m ?? "?"} m`
    );
    mk.addTo(state.map); state.mapLayers.push(mk); latlngs.push([t.lat, t.lon]);
  });

  // recovery route polyline (confirmed, in order)
  const routePts = m.recovery_route.map(id => byId[id]).filter(t => t && t.lat != null).map(t => [t.lat, t.lon]);
  if (routePts.length > 1) {
    const line = L.polyline(routePts, { color: "#35d6f2", weight: 2.5, dashArray: "6 6", opacity: .9 }).addTo(state.map);
    state.mapLayers.push(line);
    routePts.forEach((p, i) => {
      const badge = L.marker(p, { icon: L.divIcon({
        className: "", html: `<div style="background:#35d6f2;color:#04121a;font:700 10px/18px var(--mono,monospace);width:18px;height:18px;border-radius:50%;text-align:center;border:2px solid #04121a">${i + 1}</div>`,
        iconSize: [18, 18], iconAnchor: [9, 9] }) }).addTo(state.map);
      state.mapLayers.push(badge);
    });
  }
  state.map.fitBounds(L.latLngBounds(latlngs).pad(0.25));
  setTimeout(() => state.map.invalidateSize(), 60);
}

function renderDownloads(surveyId, mission) {
  const fmts = [
    ["geojson", "hazards + route (GIS)"],
    ["gpx", "waypoints (boat GPS)"],
    ["kml", "Google Earth"],
    ["csv", "spreadsheet"],
    ["json", "full audit trail"],
  ];
  $("#dlGrid").innerHTML = fmts.map(([f, desc]) =>
    `<button class="dl-btn" data-fmt="${f}"><b>.${f}</b><span>${desc}</span></button>`).join("");
  $$("#dlGrid .dl-btn").forEach(b => b.onclick = () => {
    const a = document.createElement("a");
    a.href = `${API}/api/report/${b.dataset.fmt}?survey_id=${encodeURIComponent(surveyId)}`;
    a.download = `mission.${b.dataset.fmt}`;
    document.body.appendChild(a); a.click(); a.remove();
  });
}

function renderThumbs(d) {
  $("#thumbStrip").innerHTML = d.frames
    .filter(f => f.thumb_png)
    .map(f => `<img class="thumb" title="${f.frame_id} · ${cShort(f.counts)}" src="${f.thumb_png}" alt="${f.frame_id}"/>`)
    .join("");
}

function renderHazards(d) {
  const rows = d.tracked;
  $("#hazCount").textContent = `— ${rows.length} promoted to the map/route (REJECTED kept for audit, not shown)`;
  const body = $("#hazBody");
  body.innerHTML = "";
  rows.forEach(t => {
    const tr = document.createElement("tr");
    const coords = t.lat != null ? `${t.lat.toFixed(5)}, ${t.lon.toFixed(5)}` : `<span class="muted">no GPS</span>`;
    tr.innerHTML = `
      <td>${t.oid}</td>
      <td>${t.cls_name}</td>
      <td><span class="v-tag" style="color:${VCOLOR[t.verdict]};background:${VCOLOR[t.verdict]}22">${t.verdict}</span></td>
      <td>${t.conf}</td>
      <td>${t.evidence_score}</td>
      <td>${t.height_m != null ? t.height_m + " m" : "—"}</td>
      <td class="${t.shadow_quality === "clear" ? "chip-clear" : t.shadow_quality === "weak" ? "chip-weak" : "chip-none"}">${t.shadow_quality}</td>
      <td>${coords}</td>
      <td>${t.geo_error_m != null ? "±" + t.geo_error_m + " m" : "—"}</td>
      <td>${reviewCell(t)}</td>`;
    body.appendChild(tr);
  });
  $$("#hazBody .rev-btns button").forEach(b => b.onclick = e => {
    const tr = e.target.closest("tr");
    tr.classList.add("rev-done");
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
