/* DEPTH studio — constants, state, the boot sequence, popovers, tabs and small utilities.
   Zero-build: index.html loads webui/js/*.js in order as classic scripts sharing one global scope;
   the boot runs on DOMContentLoaded (core.js), after every file has loaded. API: src/dashboard/routes/. */
"use strict";


const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const API = ""; // same-origin (served by FastAPI)
const SVGNS = "http://www.w3.org/2000/svg";

const VERDICTS = ["confirmed", "review", "low_risk"];
const VCOLOR = { confirmed: "#9ecbff", review: "#e6c48a", low_risk: "#7f8a97", rejected: "#7f8a97" };
const VLABEL = { confirmed: "Confirmed", review: "Review", low_risk: "Low risk", rejected: "Low risk" };
/* display text: sentence case, no snake_case ("fishing_gear" -> "Fishing gear") */
const human = s => { s = String(s ?? "").replace(/_/g, " ").trim(); return s ? s[0].toUpperCase() + s.slice(1) : s; };
const CLASS_SW = ["#9ecbff", "#e6c48a", "#b3a8e6", "#86cfc2", "#e6a6be"];

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
  wireTwin();
  $("#demoBtn").onclick = () => (Demo.on ? Demo.stop() : Demo.run());
  $("#cfBtn").onclick = () => CF.toggle();
  Dock.init();
  Appr.start();
  $("#demoStop").onclick = () => Demo.stop();
  wireAudit();
  tidyPanels();
  Pop.init();
  Tabs.render();
  Drills.init();
  Active.init();
  $("#missedBtn2").onclick = () => { if (state._ready) toggleMissed(); };
  await Promise.all([loadHealth(), loadSamples(), loadMetrics(), loadLabelStats()]);
});

/* small popovers: secondary tools stay out of sight until asked for (click outside or Esc closes) */
const Pop = {
  init() {
    document.addEventListener("click", e => {
      const btn = e.target.closest("[data-pop]");
      if (btn) { this.toggle(btn); return; }
      if (e.target.closest(".popover [data-close]") || !e.target.closest(".popover")) this.closeAll();
    });
    document.addEventListener("keydown", e => { if (e.key === "Escape") this.closeAll(); });
    const mb = $("#missedBtn"); if (mb) mb.dataset.close = "1";       // drawing happens on the frame
  },
  toggle(btn) {
    const pop = document.getElementById(btn.dataset.pop); if (!pop) return;
    const open = pop.hidden;
    this.closeAll();
    if (open) { pop.hidden = false; btn.setAttribute("aria-expanded", "true"); btn.classList.add("is-open"); }
  },
  closeAll() {
    $$(".popover").forEach(pop => {
      if (pop.hidden) return;
      pop.hidden = true;
      const b = $(`[data-pop="${pop.id}"]`);
      if (b) { b.setAttribute("aria-expanded", "false"); b.classList.remove("is-open"); }
    });
  },
};

/* the right column shows one section at a time when a mode has several (labels from data-tab) */
const Tabs = {
  active: {},
  render() {
    const ins = $(".inspector"); if (!ins) return;
    const mode = $("#workspace").dataset.mode;
    const panels = $$(".inspector > .panel").filter(p => !p.dataset.when || p.dataset.when.split(" ").includes(mode));
    const tabbed = panels.filter(p => p.querySelector("[data-tab]"));
    let bar = ins.querySelector(":scope > .tabs");
    if (tabbed.length < 2) {
      if (bar) bar.remove();
      ins.classList.remove("tabbed");
      $$(".inspector > .panel").forEach(p => p.classList.remove("tab-off"));
      return;
    }
    if (!bar) { bar = document.createElement("div"); bar.className = "tabs"; bar.setAttribute("role", "tablist"); ins.prepend(bar); }
    const cur = Math.min(this.active[mode] ?? 0, tabbed.length - 1);
    bar.innerHTML = tabbed.map((p, i) => `<button type="button" role="tab" class="tab${i === cur ? " is-active" : ""}" aria-selected="${i === cur}" data-i="${i}">${escapeHtml(p.querySelector("[data-tab]").dataset.tab)}</button>`).join("");
    tabbed.forEach((p, i) => p.classList.toggle("tab-off", i !== cur));
    ins.classList.add("tabbed");                                 // a long strip wraps: every tab label stays visible
    bar.onclick = e => { const b = e.target.closest(".tab"); if (!b) return; this.active[mode] = +b.dataset.i; this.render(); ins.scrollTop = 0; };
  },
  go(label) {                                          // open a tab by its label ("Approvals", "Log" ...)
    const mode = $("#workspace").dataset.mode;
    const tabbed = $$(".inspector > .panel").filter(p => (!p.dataset.when || p.dataset.when.split(" ").includes(mode)) && p.querySelector("[data-tab]"));
    const i = tabbed.findIndex(p => p.querySelector("[data-tab]").dataset.tab === label);
    if (i >= 0) { this.active[mode] = i; this.render(); }
  },
};

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
  if (!q.length) { box.innerHTML = `<p class="empty-hint">No review cards in this survey: ${conf} auto-confirmed under the precision promise.</p>`; meta.textContent = ""; return; }
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
