/* DEPTH studio — the 3D twin's panels and controls (the three.js scene itself is webui/twin3d.js).
   Zero-build: index.html loads webui/js/*.js in order as classic scripts sharing one global scope;
   the boot runs on DOMContentLoaded (core.js), after every file has loaded. API: src/dashboard/routes/. */
"use strict";

/* ------------------------------------------------------------------ 3D TWIN (webui/twin3d.js, three.js r160) */
const Twin = { dim: "2d", sv: "map", frame: null, survey: null, no3d: null };
function twinReady(cb) { if (window.DepthTwin) cb(); else window.addEventListener("depthtwin-ready", cb, { once: true }); }
function twinPanel(wrap, kind) {
  const survey = kind === "survey";
  wrap.innerHTML = `<div class="twin-bar">
      <div class="tw-seg" data-g="cam"><button data-cam="orbit" class="on" title="orbit (drag)">⟲ orbit</button><button data-cam="fly" title="fly: W A S D + Q / E">✈ fly</button><button data-cam="top" title="top-down">⊤ top</button></div>
      <div class="tw-seg" data-g="render"><button data-r="tex" class="on">textured</button><button data-r="wire">wireframe</button></div>
      ${survey ? "" : `<label class="tw-sl" title="backscatter relief — a visual aid, NOT bathymetry">relief <input type="range" data-s="relief" min="0" max="6" step="0.5" value="0"><output>0</output></label>`}
      <label class="tw-sl" title="vertical exaggeration of heights and sonar altitude">height × <input type="range" data-s="exag" min="1" max="12" step="1" value="${survey ? 6 : 3}"><output>${survey ? 6 : 3}</output></label>
      <span class="tb-spacer"></span>
      ${survey ? `<button class="tw-btn" data-a="replay" title="replay: the boat sweeps its sonar fans; finds appear as it passes them">Replay</button>` : ""}
      <button class="tw-btn" data-a="home" title="reset view">⌂ home</button><button class="tw-btn" data-a="snap" title="save a PNG of this view">⤓ PNG</button>
    </div><div class="twin-canvas"></div>
    <div class="twin-foot"><span class="twin-legend"><span><i style="background:#9ecbff"></i>confirmed</span><span><i style="background:#e6c48a"></i>review</span><span><i style="background:#7f8a97"></i>low-risk</span>${survey ? `<span><i style="background:#9ecbff"></i>track · recovery</span><span><i style="background:#b3a8ff"></i>re-survey</span>` : `<span><i style="background:#9ecbff"></i>sonar at tracked altitude</span><span><i style="background:#000;border:1px solid #445"></i>measured shadow</span>`}</span><span class="tw-note"></span></div>
    <div class="twin-empty" hidden></div>`;
  const ui = { canvas: wrap.querySelector(".twin-canvas"), note: wrap.querySelector(".tw-note"), empty: wrap.querySelector(".twin-empty"), wrap };
  return ui;
}
function wireTwinPanel(ui, viewer) {
  const w = ui.wrap;
  w.querySelectorAll("[data-cam]").forEach(b => b.onclick = () => { w.querySelectorAll("[data-cam]").forEach(x => x.classList.toggle("on", x === b)); viewer.setCameraMode(b.dataset.cam); });
  w.querySelectorAll("[data-r]").forEach(b => b.onclick = () => { w.querySelectorAll("[data-r]").forEach(x => x.classList.toggle("on", x === b)); viewer.setWire(b.dataset.r === "wire"); });
  w.querySelectorAll("[data-s]").forEach(inp => inp.oninput = () => {
    inp.nextElementSibling.textContent = inp.value;
    if (inp.dataset.s === "relief") viewer.setRelief(+inp.value); else viewer.setExag(+inp.value);
  });
  w.querySelector('[data-a="home"]').onclick = () => viewer.goHome();
  w.querySelector('[data-a="snap"]').onclick = () => { const a = document.createElement("a"); a.href = viewer.snapshot(); a.download = "depth-twin.png"; a.click(); };
  const rp = w.querySelector('[data-a="replay"]');
  if (rp) rp.onclick = () => { const on = viewer.replay(); rp.classList.toggle("on", !!on); rp.textContent = on ? "Pause" : "Replay"; };
}
function wireTwin() {
  $$(".dim-btn").forEach(b => b.onclick = () => {
    $$(".dim-btn").forEach(x => x.classList.toggle("is-active", x === b));
    Twin.dim = b.dataset.dim; cancelTour(); toggleMissed(false);
    $("#viewer").classList.toggle("dim-3d", Twin.dim === "3d"); $("#twinFrameWrap").hidden = Twin.dim !== "3d";
    if (Twin.dim === "3d") showFrameTwin();
  });
  $$(".sv-btn").forEach(b => b.onclick = () => {
    $$(".sv-btn").forEach(x => x.classList.toggle("is-active", x === b));
    Twin.sv = b.dataset.sv; $("#mapWrap").style.display = Twin.sv === "3d" ? "none" : ""; $("#twinSurveyWrap").hidden = Twin.sv !== "3d";
    if (Twin.sv === "3d") showSurveyTwin(); else setTimeout(fitMap, 60);
  });
}
/* No WebGL (a VM, a remote desktop, hardware acceleration off): say so once, never throw, never retry. */
const NO_3D = "The 3D view needs WebGL, the browser's 3D graphics, and it is off or not available in this browser "
  + "(common in virtual machines, remote desktops, or with hardware acceleration turned off). Everything else in DEPTH "
  + "works. To see the 3D view, turn on hardware acceleration or open the studio in another browser.";
function makeTwin(ui, opts) {
  if (Twin.no3d) return null;
  if (!window.DepthTwin.supported()) { Twin.no3d = NO_3D; console.warn("3D twin: no WebGL"); return null; }
  try {
    return window.DepthTwin.create(ui.canvas, opts);
  } catch (e) {                                  // WebGL is there, so this is a bug: say so, and log it
    Twin.no3d = `The 3D view failed to start (${String((e && e.message) || e)}). Everything else in DEPTH works.`;
    console.error("3D twin failed:", e);
    return null;
  }
}
function showNo3d(ui) {
  ui.wrap.classList.add("no-3d");
  ui.empty.hidden = false;
  ui.empty.textContent = Twin.no3d || NO_3D;
}
function showFrameTwin() {
  twinReady(() => {
    if (!Twin.frameUi) Twin.frameUi = twinPanel($("#twinFrameWrap"), "frame");
    if (!Twin.frame) {
      Twin.frame = makeTwin(Twin.frameUi, { onPick: id => selectCandidate(id, true) });
      if (!Twin.frame) return showNo3d(Twin.frameUi);
      Twin.frame.setExag(3); wireTwinPanel(Twin.frameUi, Twin.frame);
    }
    const d = state._analyze, ui = Twin.frameUi, tw = d && d.twin;
    ui.empty.hidden = !!(tw && tw.available);
    if (!d) ui.empty.textContent = "Run a frame first — the twin is built from the measured geometry of the analysed frame.";
    else if (!tw || !tw.available) ui.empty.textContent = (tw && tw.reason) || "3D twin unavailable for this frame.";
    else { Twin.frame.showFrame(tw); ui.note.textContent = "heights: shadow × tracked altitude · relief = backscatter, not bathymetry"; }   // opens on the overview
  });
}
function showSurveyTwin() {
  twinReady(() => {
    if (!Twin.surveyUi) Twin.surveyUi = twinPanel($("#twinSurveyWrap"), "survey");
    if (!Twin.survey) {
      Twin.survey = makeTwin(Twin.surveyUi, { onReplayEnd: () => { const b = Twin.surveyUi.wrap.querySelector('[data-a="replay"]'); b.classList.remove("on"); b.textContent = "Replay"; } });
      if (!Twin.survey) return showNo3d(Twin.surveyUi);
      Twin.survey.setExag(6); wireTwinPanel(Twin.surveyUi, Twin.survey);
    }
    const d = state.survey, ui = Twin.surveyUi, tw = d && d.twin;
    ui.empty.hidden = !!(tw && tw.available);
    if (!d) ui.empty.textContent = "Run a survey first — the twin lays every frame's swath along the boat track.";
    else if (!tw || !tw.available) ui.empty.textContent = (tw && tw.reason) || "3D survey twin unavailable.";
    else { Twin.survey.showSurvey(tw); ui.note.textContent = tw.synthetic ? "⚠ synthetic demo track · scale assumed" : "real track"; }
  });
}
