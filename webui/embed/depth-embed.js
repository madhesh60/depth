/*
 * depth-embed.js — drop DEPTH's hazard picture into any web console (mission control, a cleanup NGO's
 * dispatch board, a port authority dashboard). No framework, no dependencies, Shadow DOM (the host
 * page's CSS cannot break it, it cannot break the host page).
 *
 *   <script src="https://<depth-host>/embed/depth-embed.js" defer></script>
 *   <depth-hazards api="https://<depth-host>" survey="latest" limit="8" theme="light"></depth-hazards>
 *
 * Attributes: api (DEPTH base URL; default = where this script came from) · survey (id | "latest") ·
 * limit (cards listed, default 8) · theme (light | dark) · public (generalise protected-site positions)
 * · refresh (seconds, default 60; 0 = off).
 * Reads the OGC API – Features endpoint only (/ogc/collections/{hazards,work_orders}/items) — the same
 * data QGIS / ArcGIS see. Cross-origin consoles: add their origin to DEPTH_CORS_ORIGINS on the server.
 * Nothing here can approve or dispatch: approvals are made by a person in the DEPTH studio.
 */
(() => {
  const SCRIPT_ORIGIN = (() => { try { return new URL(document.currentScript.src).origin; } catch (e) { return location.origin; } })();
  const esc = s => String(s ?? "").replace(/[&<>"']/g, m => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m]));
  const TIER = { confirmed: "#1f8a4c", review: "#b7791f", low_risk: "#7a8794" };

  const CSS = `
  :host { display: block; font: 13px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
  .card { --bg: #ffffff; --fg: #17212b; --mut: #5f6c78; --line: #dfe5ea; --soft: #f5f7f9; --acc: #0b6e86;
          background: var(--bg); color: var(--fg); border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
  .card.dark { --bg: #10161d; --fg: #e7eef4; --mut: #93a2b0; --line: #243240; --soft: #161f28; --acc: #1fb6d5; }
  header { display: flex; align-items: baseline; gap: 10px; padding: 12px 14px; border-bottom: 1px solid var(--line); }
  header b { font-size: 13px; letter-spacing: .04em; }
  header .sid { font: 11px ui-monospace, Menlo, Consolas, monospace; color: var(--mut); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  header .tag { margin-left: auto; font: 600 10px ui-monospace, monospace; padding: 2px 7px; border-radius: 4px; border: 1px solid #d9a441; color: #9a6a12; white-space: nowrap; }
  .card.dark header .tag { color: #ffd9a3; }
  .kpis { display: grid; grid-template-columns: repeat(4, 1fr); border-bottom: 1px solid var(--line); }
  .kpi { padding: 9px 14px; border-right: 1px solid var(--line); }
  .kpi:last-child { border-right: 0; }
  .kpi .v { font-size: 18px; font-weight: 650; font-variant-numeric: tabular-nums; }
  .kpi .l { font-size: 10.5px; color: var(--mut); text-transform: uppercase; letter-spacing: .06em; }
  .body { display: grid; grid-template-columns: minmax(180px, 38%) 1fr; }
  svg { display: block; width: 100%; height: 100%; min-height: 200px; background: var(--soft); border-right: 1px solid var(--line); color: var(--mut); }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th { text-align: left; font-weight: 600; font-size: 10.5px; color: var(--mut); text-transform: uppercase; letter-spacing: .05em; padding: 7px 10px; border-bottom: 1px solid var(--line); }
  td { padding: 6px 10px; border-bottom: 1px solid var(--line); font-variant-numeric: tabular-nums; }
  tr:last-child td { border-bottom: 0; }
  td.id { font: 600 11.5px ui-monospace, monospace; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; vertical-align: 1px; }
  .st { font-size: 10.5px; padding: 1px 6px; border-radius: 4px; border: 1px solid var(--line); color: var(--mut); }
  .st.approved { border-color: #1f8a4c; color: #1f8a4c; } .st.pending { border-color: #b7791f; color: #9a6a12; }
  footer { display: flex; gap: 14px; align-items: center; padding: 8px 14px; border-top: 1px solid var(--line); font-size: 11px; color: var(--mut); }
  footer a { color: var(--acc); text-decoration: none; } footer a:hover { text-decoration: underline; }
  footer .gate { margin-left: auto; }
  .msg { padding: 18px 14px; color: var(--mut); }
  @media (max-width: 560px) { .body { grid-template-columns: 1fr; } svg { border-right: 0; border-bottom: 1px solid var(--line); } .kpis { grid-template-columns: repeat(2, 1fr); } }`;

  class DepthHazards extends HTMLElement {
    static get observedAttributes() { return ["api", "survey", "limit", "theme", "public"]; }
    constructor() { super(); this.attachShadow({ mode: "open" }); this._timer = null; }
    connectedCallback() { this.load(); this._arm(); }
    disconnectedCallback() { clearInterval(this._timer); }
    attributeChangedCallback() { if (this.isConnected) this.load(); }
    _arm() {
      clearInterval(this._timer);
      const s = parseFloat(this.getAttribute("refresh") ?? "60");
      if (s > 0) this._timer = setInterval(() => { if (!document.hidden) this.load(); }, Math.max(15, s) * 1000);
    }
    get api() { return (this.getAttribute("api") || SCRIPT_ORIGIN).replace(/\/$/, ""); }
    async load() {
      const survey = this.getAttribute("survey") || "latest", pub = this.hasAttribute("public") ? "&public=true" : "";
      const q = `survey_id=${encodeURIComponent(survey)}&limit=1000${pub}`;
      try {
        const [hz, wo] = await Promise.all([
          fetch(`${this.api}/ogc/collections/hazards/items?${q}`).then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); }),
          fetch(`${this.api}/ogc/collections/work_orders/items?${q}`).then(r => r.ok ? r.json() : { features: [] }),
        ]);
        this.render(hz.features || [], wo.features || [], q);
        this.dispatchEvent(new CustomEvent("depth-loaded", { detail: { hazards: hz.numberMatched, workOrders: (wo.features || []).length } }));
      } catch (e) {
        this.shadowRoot.innerHTML = `<style>${CSS}</style><div class="card ${this.getAttribute("theme") === "dark" ? "dark" : ""}"><div class="msg">DEPTH hazards unavailable (${esc(e.message)}). Is <code>${esc(this.api)}</code> reachable and this origin in DEPTH_CORS_ORIGINS?</div></div>`;
      }
    }
    render(feats, orders, q) {
      const P = f => f.properties, dark = this.getAttribute("theme") === "dark";
      const limit = Math.max(1, parseInt(this.getAttribute("limit") || "8", 10));
      if (!feats.length) { this.shadowRoot.innerHTML = `<style>${CSS}</style><div class="card ${dark ? "dark" : ""}"><div class="msg">No hazards in this survey yet.</div></div>`; return; }
      const sid = P(feats[0]).survey_id, synth = feats.some(f => P(f).gps_synthetic);
      const count = t => feats.filter(f => P(f).tier === t).length;
      const pending = feats.filter(f => P(f).approval && P(f).approval.status === "pending").length;
      const ordered = [...feats].sort((a, b) => (P(a).review_rank ?? 1e9) - (P(b).review_rank ?? 1e9));
      // mini map: equirectangular in the survey's bbox (metres-true aspect), error circles to scale
      const xs = feats.map(f => f.geometry.coordinates[0]), ys = feats.map(f => f.geometry.coordinates[1]);
      const lat0 = (Math.min(...ys) + Math.max(...ys)) / 2, kx = Math.cos(lat0 * Math.PI / 180) * 111320, ky = 110540;
      const W = 300, H = 220, pad = 18;
      const spanX = Math.max(1e-9, (Math.max(...xs) - Math.min(...xs)) * kx), spanY = Math.max(1e-9, (Math.max(...ys) - Math.min(...ys)) * ky);
      const s = Math.min((W - 2 * pad) / spanX, (H - 2 * pad) / spanY);
      const px = lon => pad + (lon - Math.min(...xs)) * kx * s + ((W - 2 * pad) - spanX * s) / 2;
      const py = lat => H - pad - (lat - Math.min(...ys)) * ky * s - ((H - 2 * pad) - spanY * s) / 2;
      const approved = new Set(orders.flatMap(o => (o.properties.targets || []).map(t => `${o.properties.survey_id}:${t}`)));
      const dots = feats.map(f => {
        const [lon, lat] = f.geometry.coordinates, p = P(f), c = TIER[p.tier] || "#7a8794";
        const r = Math.max(1.5, (p.error_m || 3) * s);
        return `<g><title>${esc(p.hazard_id)} · ${esc(p.class)} · P(pot) ${p.p_pot ?? "–"}</title>
          <circle cx="${px(lon).toFixed(1)}" cy="${py(lat).toFixed(1)}" r="${r.toFixed(1)}" fill="${c}" fill-opacity=".08" stroke="${c}" stroke-opacity=".35" stroke-dasharray="2 2"/>
          <circle cx="${px(lon).toFixed(1)}" cy="${py(lat).toFixed(1)}" r="3.2" fill="${c}"/>
          ${approved.has(f.id) ? `<circle cx="${px(lon).toFixed(1)}" cy="${py(lat).toFixed(1)}" r="7" fill="none" stroke="#1f8a4c" stroke-width="1.6"/>` : ""}</g>`;
      }).join("");
      const scaleM = [10, 20, 50, 100, 200, 500, 1000].find(m => m * s > 40) || 1000;
      const rows = ordered.slice(0, limit).map(f => {
        const p = P(f), a = p.approval;
        return `<tr><td class="id"><span class="dot" style="background:${TIER[p.tier] || "#7a8794"}"></span>${esc(p.hazard_id)}</td>
          <td>${esc((p.class || "").replace(/_/g, " "))}</td><td>${p.p_pot != null ? Math.round(p.p_pot * 100) + "%" : "–"}</td>
          <td>${esc(p.shadow || "–")}</td><td>${a ? `<span class="st ${esc(a.status)}">${esc(a.action)} · ${esc(a.status)}</span>` : "–"}</td></tr>`;
      }).join("");
      this.shadowRoot.innerHTML = `<style>${CSS}</style>
      <div class="card ${dark ? "dark" : ""}" part="card">
        <header><b>DEPTH · hazards</b><span class="sid">${esc(sid)}</span>${synth ? `<span class="tag">SYNTHETIC DEMO GPS</span>` : ""}</header>
        <div class="kpis">
          <div class="kpi"><div class="v">${feats.length}</div><div class="l">hazards</div></div>
          <div class="kpi"><div class="v">${count("review")}</div><div class="l">for review</div></div>
          <div class="kpi"><div class="v">${pending}</div><div class="l">approvals pending</div></div>
          <div class="kpi"><div class="v">${orders.length}</div><div class="l">approved orders</div></div>
        </div>
        <div class="body">
          <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" aria-label="hazard positions">${dots}
            <g transform="translate(${pad},${H - 8})"><line x1="0" y1="0" x2="${(scaleM * s).toFixed(1)}" y2="0" stroke="currentColor" stroke-width="1.5" opacity=".6"/>
            <text x="${(scaleM * s + 5).toFixed(1)}" y="3" font-size="9" fill="currentColor" opacity=".7">${scaleM} m</text></g></svg>
          <table><thead><tr><th>ID</th><th>Class</th><th>P(pot)</th><th>Shadow</th><th>Approval</th></tr></thead><tbody>${rows}</tbody></table>
        </div>
        <footer><a href="${this.api}/ogc/collections/hazards/items?${q}" target="_blank" rel="noopener">GeoJSON (OGC API)</a>
          <a href="${this.api}/api/brief?survey_id=${encodeURIComponent(sid)}" target="_blank" rel="noopener">mission brief</a>
          <a href="${this.api}/" target="_blank" rel="noopener">open DEPTH</a>
          <span class="gate">nothing is dispatched without a person's approval</span></footer>
      </div>`;
    }
  }
  if (!customElements.get("depth-hazards")) customElements.define("depth-hazards", DepthHazards);
})();
