# Integrating DEPTH with your consoles (ocean institutes, cleanup organisations, port authorities)

DEPTH does not ask you to adopt a new application. It speaks the standards your systems already use:

| your system | connect with | what you get |
|---|---|---|
| **GIS** — QGIS, ArcGIS Pro, OpenLayers, GeoServer clients | **OGC API – Features** at `/ogc` | hazards, re-survey passes, routes and **approved work orders** as live layers |
| **Mission control / dispatch / ticketing / ERP** | **signed webhooks** | a push for every finished survey, every approval request and every decision (with positions) |
| **Any web console or intranet page** | **`<depth-hazards>` web component** | a self-contained hazard panel: KPIs, a mini-map to scale, the review queue, approval status |
| **AI agents** — Claude, IDE agents, your own agents | **MCP** at `/mcp` (or stdio) | run the sonar agent, read decision traces, **ask** for approval ([`mcp.md`](mcp.md)) |
| **Anything else** | REST + OpenAPI (`/openapi.json`, `/docs`) | the full API, generated clients in any language |

All of these are listed, with live URLs and copy-ready snippets, in the studio's **Connect** tab.

**The rule that holds across all channels:** every channel can *read*, and agents and consoles can
*ask* (`POST /api/approvals`, or MCP `request_human_approval`). Only a named person, in the DEPTH
studio, approves. An approved request is the only thing DEPTH ever presents as a work order.

---

## 1. OGC API – Features (`/ogc`)

It implements OGC API – Features Part 1 (Core + GeoJSON), with queryables from Part 3. It was checked
with OWSLib, an independent OGC client, against the running server.

| path | |
|---|---|
| `/ogc` | landing page (links to conformance, collections and the OpenAPI definition) |
| `/ogc/conformance` | `core`, `geojson`, `oas30` |
| `/ogc/collections` | `hazards` · `resurvey_passes` · `routes` · `work_orders`, each with its spatial extent |
| `/ogc/collections/{id}/items` | GeoJSON FeatureCollection with `numberMatched`, `numberReturned` and a `next` link |
| `/ogc/collections/{id}/items/{featureId}` | one feature (`<survey_id>:<hazard_id>`) |
| `/ogc/collections/{id}/queryables` | the filterable properties (JSON Schema) |

Parameters: `limit` (≤ 1000), `offset`, `bbox=minLon,minLat,maxLon,maxLat`, `survey_id=<id>|latest`,
`public=true`.

**Hazard properties:**

- identity and status: `survey_id`, `hazard_id`, `class`, `tier` (`review` / `confirmed` / `low_risk`);
- ranking: `p_pot` (calibrated), `confidence`, `review_rank`, `in_analyst_budget`;
- evidence: `shadow`, `relative_height`, `sightings`;
- position: `error_m`, `gps_synthetic`, `protected_site_generalised`;
- workflow: `approval` (`id`, `action`, `status`), `human_approval_required`.

**QGIS:**

1. Layer ▸ Add Layer ▸ Add WFS / OGC API – Features Layer.
2. New connection, with the URL `https://<host>/ogc`.
3. Connect, then add `hazards` and `work_orders`.

**ArcGIS Pro:** Insert ▸ Connections ▸ Server ▸ New OGC API Server.

**Protected sites.** `public=true` generalises wreck positions to about 1.1 km and drops re-survey
passes that would point at them. Setting `DEPTH_OGC_PUBLIC=1` forces this for every request, which
suits a public-facing deployment.

## 2. Signed webhooks

To register a webhook, use the Connect tab, or call:

```bash
curl -X POST https://<host>/api/integrations/webhooks \
  -H "Content-Type: application/json" -H "X-DEPTH-Admin: $DEPTH_ADMIN_TOKEN" \
  -d '{"url": "https://ops.example.org/depth-events", "events": ["survey.completed", "approval.decided"]}'
```

The response contains the signing secret, **once**.

**Events:**

| event | data |
|---|---|
| `survey.completed` | counts, GPS kind (real / synthetic / none), recall promise, top-5 review cards with positions, re-survey summary, links (brief, GeoJSON, OGC items) |
| `approval.requested` | the request (action, targets, rationale, who asked, via which channel), plus the targets' positions |
| `approval.decided` | the same, plus the decision and who made it. **An approved decision is a work order.** |
| `ping` | a test delivery ("send test" in the Connect tab) |

**Signature:** `X-DEPTH-Signature: t=<unix>,v1=<hex>` is HMAC-SHA256 over `f"{t}.{raw body}"`, the same
scheme Stripe and GitHub use. To verify a delivery:

```python
import hmac, hashlib, time
def verify(secret: str, header: str, body: bytes) -> bool:
    parts = dict(p.split("=", 1) for p in header.split(","))
    t = int(parts["t"])
    mac = hmac.new(secret.encode(), f"{t}.".encode() + body, hashlib.sha256).hexdigest()
    return abs(time.time() - t) < 300 and hmac.compare_digest(mac, parts["v1"])
```

**Delivery:**

- 3 attempts with 1 s and 4 s back-off; a `4xx` response (other than 429) is not retried.
- Every delivery is logged, and the Connect tab shows the log.

**Safety:**

- URLs that resolve to private, loopback, link-local or reserved addresses (for example the EC2 metadata service) are refused, both at registration and again at every delivery.
- On-premise consoles can be allowed with `DEPTH_WEBHOOK_ALLOW_PRIVATE=1`.
- Managing webhooks needs `X-DEPTH-Admin` whenever `DEPTH_ADMIN_TOKEN` is set.

## 3. The `<depth-hazards>` web component

```html
<script src="https://<host>/embed/depth-embed.js" defer></script>
<depth-hazards api="https://<host>" survey="latest" limit="6" theme="light"></depth-hazards>
```

The component uses Shadow DOM (the host page's CSS and the panel's CSS cannot interfere) and has no
dependencies. It reads only the OGC endpoint and refreshes every 60 s by default.

- **Attributes:** `survey`, `limit`, `theme` (`light` / `dark`), `public`, `refresh` (in seconds; `0` turns refresh off).
- **Event:** it emits `depth-loaded` with `{hazards, workOrders}`.
- **CORS:** a console on another origin must be listed in `DEPTH_CORS_ORIGINS`.

## 4. Server settings

| variable | purpose |
|---|---|
| `DEPTH_PUBLIC_URL` | base URL used in webhook links (for example the CloudFront domain) |
| `DEPTH_CORS_ORIGINS` | consoles allowed to call the API from a browser (for the embed) |
| `DEPTH_OGC_PUBLIC` | `1` = always generalise protected-site positions on `/ogc` |
| `DEPTH_ADMIN_TOKEN` | required for webhook management |
| `DEPTH_WEBHOOK_ALLOW_PRIVATE` | `1` = allow on-premise (private-address) webhook targets |
| `DEPTH_MCP_TOKEN`, `DEPTH_MCP_ALLOWED_HOSTS`, `DEPTH_APPROVER_PIN` | see [`mcp.md`](mcp.md) |
