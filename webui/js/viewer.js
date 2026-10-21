/* DEPTH studio — the interactive frame viewer, the pipeline dock (stepper + live agent log), toasts, loading.
   Zero-build: index.html loads webui/js/*.js in order as classic scripts sharing one global scope;
   the boot runs on DOMContentLoaded (core.js), after every file has loaded. API: src/dashboard/routes/. */
"use strict";

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
    const idleLi = el().querySelector(".log-idle"); if (idleLi) idleLi.remove();
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

/* loading: one quiet card in the middle of the workspace - a spinning arc, what is happening in words,
   the elapsed time, and a bar that always moves: an indeterminate sweep until the server reports
   progress, then a fill that creeps toward the end of the current step (never past it) so it never
   looks stalled, snaps to 100% on finish and fades out. Fast calls (< 180 ms) never flash it. */
const Busy = {
  n: 0, t0: 0, shown: 0, floor: 0, ceil: 0, determinate: false, tick: null, showT: null, hideT: null,
  start(txt) {
    this.n++;
    $("#busyTitle").textContent = txt; $("#busySub").textContent = "";
    if (this.n > 1) return;                       // nested: keep the running bar and clock
    clearTimeout(this.hideT); clearTimeout(this.showT); clearInterval(this.tick);
    Object.assign(this, { t0: performance.now(), shown: 0, floor: 0, ceil: 0, determinate: false });
    const c = $("#busyCard"), bar = $("#busyBar");
    bar.classList.add("is-indet"); bar.style.setProperty("--p", 0);
    $("#busyTime").textContent = "";
    c.classList.remove("is-leaving", "is-done");
    document.body.classList.add("is-busy");
    this.showT = setTimeout(() => { c.hidden = false; }, 180);
    this.tick = setInterval(() => this._frame(), 100);
  },
  _frame() {
    const s = (performance.now() - this.t0) / 1000;
    $("#busyTime").textContent = s >= 1 ? `${s.toFixed(1)} s` : "";
    if (!this.determinate) return;
    // ease toward the end of the current step; slows as it approaches, never reaches it
    this.shown = Math.max(this.floor, this.shown + (this.ceil - this.shown) * 0.035);
    $("#busyBar").style.setProperty("--p", this.shown.toFixed(4));
  },
  /* frac: fraction already done (0..1); next: where the step now running ends (defaults to frac). */
  set(frac, txt, next) {
    if (txt) $("#busySub").textContent = txt;
    if (frac == null) return;
    const bar = $("#busyBar");
    if (!this.determinate) { this.determinate = true; bar.classList.remove("is-indet"); }
    this.floor = Math.max(this.floor, Math.min(1, frac));
    this.ceil = Math.max(this.floor, Math.min(.98, next != null ? next : frac));
    this.shown = Math.max(this.shown, this.floor);
    bar.style.setProperty("--p", this.shown.toFixed(4));
  },
  progress(done, total, txt) {
    if (!total) return this.set(null, txt);
    this.set(done / total, txt, (done + .9) / total);
  },
  done() {
    this.n = Math.max(0, this.n - 1); if (this.n) return;
    clearTimeout(this.showT); clearInterval(this.tick);
    const c = $("#busyCard"), bar = $("#busyBar");
    document.body.classList.remove("is-busy");
    if (c.hidden) return;                          // finished before it was ever shown
    bar.classList.remove("is-indet"); bar.style.setProperty("--p", 1); c.classList.add("is-done");
    this.hideT = setTimeout(() => {
      c.classList.add("is-leaving");
      this.hideT = setTimeout(() => { c.hidden = true; c.classList.remove("is-leaving", "is-done"); }, 220);
    }, 260);
  },
};

/* ------------------------------------------------------------------ DOCK: slim status bar <-> agent log */
const Dock = {
  set(min) {
    document.body.classList.toggle("dock-min", min);
    $("#dockToggle").textContent = "Agent log";
    $("#dockToggle").setAttribute("aria-expanded", String(!min));
    localStorage.setItem("depth.dockMin", min ? "1" : "0");
    setTimeout(() => { if (state.map) state.map.invalidateSize(); window.dispatchEvent(new Event("resize")); }, 220);
  },
  init() {
    const saved = localStorage.getItem("depth.dockMin");
    this.set(saved != null ? saved === "1" : true);    // slim by default: the log opens on request
    $("#dockToggle").onclick = () => this.set(!document.body.classList.contains("dock-min"));
  },
};
