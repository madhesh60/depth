/* DEPTH studio — Connect: MCP, OGC API, signed webhooks and the drop-in panel.
   Zero-build: index.html loads webui/js/*.js in order as classic scripts sharing one global scope;
   the boot runs on DOMContentLoaded (core.js), after every file has loaded. API: src/dashboard/routes/. */
"use strict";

/* ------------------------------------------------------------------ CONNECT (integrations) */
const Connect = {
  info: null, lastSecret: null,
  code(txt, multi) {
    const id = "cx" + Math.random().toString(36).slice(2, 8);
    return `<div class="cx-code">${multi ? `<pre id="${id}">${escapeHtml(txt)}</pre>` : `<code id="${id}">${escapeHtml(txt)}</code>`}<button class="cx-copy" data-copy="${id}">Copy</button></div>`;
  },
  wireCopy(root) {
    root.querySelectorAll(".cx-copy[data-copy]").forEach(b => b.onclick = () => {
      const t = document.getElementById(b.dataset.copy).textContent;
      (navigator.clipboard ? navigator.clipboard.writeText(t) : Promise.reject()).then(() => { b.textContent = "copied ✓"; setTimeout(() => b.textContent = "Copy", 1300); }).catch(() => {});
    });
  },
  steps(items) { return `<ol class="cx-steps">${items.map(([t, body]) => `<li><div class="cx-st">${t}</div>${body || ""}</li>`).join("")}</ol>`; },
  more(title, body) { return `<details class="cx-more"><summary>${title}</summary><div class="cx-more-b">${body}</div></details>`; },
  async load() {
    try { this.info = await fetch(`${API}/api/integrations`).then(r => r.json()); } catch (e) { return; }
    const i = this.info, base = i.base_url;
    $("#cxBase").innerHTML = `<span>Server</span><code>${escapeHtml(base)}</code><a href="${escapeHtml(i.openapi)}" target="_blank" rel="noopener">OpenAPI spec ↗</a>`;
    const tools = ["depth_status", "list_samples", "analyze_frame", "frame_overlay", "run_survey", "get_review_queue",
      "get_hazard", "get_resurvey_plan", "get_stage1_counterfactual", "get_mission_brief", "export_report",
      "request_human_approval", "get_approval_status"];
    $("#cxMcpBody").innerHTML = this.steps([
      ["Add DEPTH to Claude Code", this.code(`claude mcp add --transport http depth ${i.mcp.url}` + (i.mcp.token_required ? ` --header "Authorization: Bearer <token>"` : ""))],
      ["Or run it locally, without the server", this.code(`claude mcp add depth -e DEPTH_LAZY_MODEL=1 -- ${i.mcp.stdio}`)],
      ["Ask your agent", `<p class="cx-p">“Run a survey and show me the review queue.”</p>`],
    ]) + `<div class="cx-status ${i.mcp.token_required ? "is-ok" : "is-warn"}">${i.mcp.token_required
        ? "Protected — agents need the bearer token (DEPTH_MCP_TOKEN)."
        : "No token set — fine on this computer; set DEPTH_MCP_TOKEN before exposing the server."}</div>`
      + this.more(`${tools.length} tools the agent can use`, `<div class="cx-chips">${tools.map(t => `<span>${t}</span>`).join("")}</div>
        <p class="cx-p">Agents can run the loop and read every decision. They cannot approve, dispatch, label or change a threshold — <code>request_human_approval</code> only asks.</p>
        <p class="cx-p">Endpoint</p>${this.code(i.mcp.url)}`);
    const cols = await Promise.all(i.ogc.collections.map(c => fetch(`${API}/ogc/collections/${c}/items?survey_id=latest&limit=1`)
      .then(r => r.ok ? r.json() : null).catch(() => null)));
    $("#cxOgcBody").innerHTML = this.steps([
      ["Copy the server address", this.code(i.ogc.landing)],
      ["QGIS: Layer ▸ Add Layer ▸ Add WFS / OGC API – Features Layer", `<p class="cx-p">New connection, paste the address, version “OGC API – Features”.</p>`],
      ["Connect and add a layer", `<div class="cx-chips">${i.ogc.collections.map((c, k) =>
        `<a href="${API}/ogc/collections/${c}/items?survey_id=latest" target="_blank" rel="noopener">${c}${cols[k] ? ` <b>${cols[k].numberMatched}</b>` : ""}</a>`).join("")}</div>`],
    ]) + this.more("ArcGIS Pro and filters", `<p class="cx-p"><b>ArcGIS Pro:</b> Insert ▸ Connections ▸ Server ▸ New OGC API Server ▸ paste the address.</p>
        <p class="cx-p"><b>Filters:</b> <code>survey_id=latest</code>, <code>bbox</code>, <code>limit</code>, <code>offset</code>, <code>public=true</code> (wreck positions generalised${i.ogc.public_forced ? " — forced on this server" : ""}).</p>
        <p class="cx-p">Demo positions are synthetic and carry <code>gps_synthetic: true</code>.</p>`);
    this.loadHooks();
    const snippet = `<script src="${base}/embed/depth-embed.js" defer></script>
<depth-hazards api="${base}" survey="latest" limit="6" theme="light"></depth-hazards>`;
    $("#cxEmbedBody").innerHTML = this.steps([
      ["Paste these two lines into your page", this.code(snippet, true)],
      ["It looks like this", `<div class="cx-embed-preview"><depth-hazards api="${API || location.origin}" survey="latest" limit="4" theme="light" refresh="0"></depth-hazards></div>`],
    ]) + this.more("Options", `<p class="cx-p"><code>survey</code> (id or latest) · <code>limit</code> · <code>theme</code> (light or dark) · <code>public</code> · <code>refresh</code> (seconds)</p>
        <p class="cx-p">A page on another site must be allowed in <code>DEPTH_CORS_ORIGINS</code>${i.cors_origins && i.cors_origins.length ? ` (now: ${escapeHtml(i.cors_origins.join(", "))})` : " (none set)"}.</p>`);
    if (!customElements.get("depth-hazards")) { const s = document.createElement("script"); s.src = "embed/depth-embed.js"; document.head.appendChild(s); }
    ["#cxMcpBody", "#cxOgcBody", "#cxEmbedBody"].forEach(sel => this.wireCopy($(sel)));
  },
  async loadHooks() {
    const r = await fetch(`${API}/api/integrations/webhooks`);
    if (r.status === 403) { $("#cxHooksBody").innerHTML = `<p class="cx-p">Webhook management needs the operator's admin token (DEPTH_ADMIN_TOKEN).</p>`; return; }
    const w = await r.json();
    const verify = [
      "import hmac, hashlib, time",
      "",
      "def verify(secret: str, header: str, body: bytes) -> bool:",
      "    parts = dict(p.split('=', 1) for p in header.split(','))   # X-DEPTH-Signature",
      "    t = int(parts['t'])",
      "    mac = hmac.new(secret.encode(), f'{t}.'.encode() + body, hashlib.sha256).hexdigest()",
      "    return abs(time.time() - t) < 300 and hmac.compare_digest(mac, parts['v1'])"].join("\n");
    const hooks = w.webhooks.length ? w.webhooks.map(h => `<tr><td class="mono">${escapeHtml(h.url)}<div class="cx-note">${escapeHtml(h.id)} · secret ${escapeHtml(h.secret)}</div></td>
        <td class="mono">${escapeHtml(h.events.join(", "))}</td>
        <td style="white-space:nowrap"><button class="cx-copy" data-test="${escapeHtml(h.id)}">send test</button> <button class="cx-copy" data-del="${escapeHtml(h.id)}">remove</button></td></tr>`).join("")
      : `<tr><td colspan="3" class="cx-note">No webhooks yet.</td></tr>`;
    const dels = w.deliveries.length ? w.deliveries.slice(0, 8).map(d => `<tr><td class="mono">${escapeHtml(d.event)} → ${escapeHtml(d.url)}</td>
        <td class="${d.status === "delivered" ? "cx-ok" : "cx-warn"}">${escapeHtml(d.status)}${d.code ? " · " + d.code : ""}${d.error ? `<div class="cx-note">${escapeHtml(d.error)}</div>` : ""}</td>
        <td class="mono">${d.ms ?? "–"}</td></tr>`).join("")
      : `<tr><td colspan="3" class="cx-note">No deliveries yet.</td></tr>`;
    $("#cxHooksBody").innerHTML = this.steps([
      ["Enter the address that should receive events", `<form class="cx-form" id="cxHookForm"><input type="url" id="cxHookUrl" placeholder="https://ops.example.org/depth-events" required />
          <button class="cx-btn" type="submit">Add</button>
          <div class="cx-evs">${w.events.filter(e => e !== "ping").map(e => `<label><input type="checkbox" value="${e}" checked /> ${e}</label>`).join("")}</div></form>`
        + (this.lastSecret ? `<div class="cx-secret">Signing secret for <b>${escapeHtml(this.lastSecret.id)}</b> — shown once, store it now:${this.code(this.lastSecret.secret)}</div>` : "")],
      ["Check the signature on your side", `<p class="cx-p">Every message carries <code>X-DEPTH-Signature</code> (HMAC-SHA256).</p>`],
    ]) + `<table class="cx-table"><thead><tr><th>Webhook</th><th>Events</th><th></th></tr></thead><tbody>${hooks}</tbody></table>`
      + this.more("Verification code (Python)", this.code(verify, true))
      + this.more("Recent deliveries", `<table class="cx-table"><thead><tr><th>Event</th><th>Status</th><th>ms</th></tr></thead><tbody>${dels}</tbody></table>
        <p class="cx-p">Private, loopback and link-local addresses are refused (DEPTH_WEBHOOK_ALLOW_PRIVATE=1 for on-premise consoles). 3 attempts with back-off.</p>`);
    this.wireCopy($("#cxHooksBody"));
    $("#cxHookForm").onsubmit = async e => {
      e.preventDefault();
      const events = [...$$("#cxHookForm .cx-evs input:checked")].map(x => x.value);
      const r2 = await fetch(`${API}/api/integrations/webhooks`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url: $("#cxHookUrl").value, events }) });
      const b = await r2.json();
      if (!r2.ok) { alert(b.detail || r2.status); return; }
      this.lastSecret = { id: b.id, secret: b.secret }; this.loadHooks();
    };
    $$("#cxHooksBody [data-test]").forEach(b => b.onclick = async () => { b.textContent = "sending…"; await fetch(`${API}/api/integrations/webhooks/${b.dataset.test}/test`, { method: "POST" }); this.loadHooks(); });
    $$("#cxHooksBody [data-del]").forEach(b => b.onclick = async () => { await fetch(`${API}/api/integrations/webhooks/${b.dataset.del}`, { method: "DELETE" }); this.lastSecret = null; this.loadHooks(); });
  },
};
