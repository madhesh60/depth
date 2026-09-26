# DEPTH over MCP — let any agent run the sonar agent (and ask a person to approve)

DEPTH is a **Model Context Protocol** server. Claude Desktop, Claude Code, IDE agents, agent frameworks
and a partner organisation's own agent can call it locally or over the internet. They run the same
OpenCV 5 loop as the studio, read the evidence behind every tier, and file **approval requests**. A
person approves or declines those requests in the DEPTH studio.

Code: [`src/dashboard/mcp_server.py`](../src/dashboard/mcp_server.py) (official MCP Python SDK,
`mcp==1.28.1`) · tests: [`tests/test_mcp.py`](../tests/test_mcp.py). The tests use the official
client over **stdio** and raw JSON-RPC over **HTTP**; the live server was also driven with the
official Streamable-HTTP client.

## Connect

**Claude Code, local (stdio)** — run from the repo:

```bash
claude mcp add depth -e DEPTH_LAZY_MODEL=1 -- python -m src.dashboard.mcp_server
```

**Claude Desktop, local (stdio)** — in `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "depth": {
      "command": "python",
      "args": ["-m", "src.dashboard.mcp_server"],
      "env": { "PYTHONPATH": "C:\\path\\to\\depth", "DEPTH_LAZY_MODEL": "1" }
    }
  }
}
```

**Remote (Streamable HTTP)** — the DEPTH server exposes `https://<host>/mcp`. It is stateless and
returns JSON, so it works behind CloudFront or a load balancer. Send the bearer token on every request:

```bash
claude mcp add --transport http depth https://<cloudfront-domain>/mcp --header "Authorization: Bearer <DEPTH_MCP_TOKEN>"
```

**Inspect / debug** with any MCP client, for example the MCP Inspector
(`npx @modelcontextprotocol/inspector`): choose transport "Streamable HTTP", URL `http://localhost:8000/mcp`.

## Tools

| tool | what it does | writes? |
|---|---|---|
| `depth_status` | model, calibrated promises, OpenCV build (COOL or stock), limits, approval counts | no |
| `list_samples` | the shipped CC-BY-SA sonar frames | no |
| `analyze_frame` | full Stage 1 → See → Prove → Decide on one frame (a sample id or a base64 image) | no |
| `frame_overlay` | the analysed frame as an image (seabed track, boxes by tier) | no |
| `run_survey` | a whole survey; returns a `survey_id` and the plan summary | runs compute |
| `get_review_queue` | REVIEW cards by calibrated P(pot), with evidence | no |
| `get_hazard` | one hazard in full, including the agent's **decision trace** | no |
| `get_resurvey_plan` | opposite-side passes with shadow-flip predictions | no |
| `get_stage1_counterfactual` | where each pin would land without Stage 1 | no |
| `get_mission_brief` | the one-page brief (numbers checked against the survey) | no |
| `export_report` | geojson · gpx · kml · csv · json · brief · trace (`public=true` generalises wrecks) | no |
| `request_human_approval` | **asks** a person: inspect · recover · resurvey · share_public · other | files a request |
| `get_approval_status` | the request's status and who decided it | no |

There is also a resource (`depth://guarantees`, `depth://survey/{id}/brief`) and a prompt
(`triage_survey`).

**Not exposed, deliberately:** approving, dispatching, labelling, or changing thresholds. An agent can
only ask. Requests with unknown hazard or pass IDs are refused.

## Agents ask, people decide

1. An agent calls `request_human_approval(survey_id, "inspect", ["H041"], "Top card: P(pot) 0.71 …")`.
   The request is stored as `pending` in an append-only event log (`runs/approvals/events.jsonl`).
2. It appears in the studio's **Approvals** panel (Survey mode), showing who asked, through which
   channel, and why.
3. A named person approves or declines it. With `DEPTH_APPROVER_PIN` set, deciding also needs the PIN,
   so a client holding the MCP/API token still cannot approve its own request.
4. The agent reads the outcome with `get_approval_status`. Approved requests are work orders for
   downstream consoles; DEPTH itself never dispatches.

## Security

| setting | effect |
|---|---|
| `DEPTH_MCP_TOKEN` | bearer token required on `/mcp` (401 otherwise). The AWS setup generates one into root-only `/etc/depth/mcp.env`. |
| `DEPTH_MCP_ALLOWED_HOSTS` | extra Host names for DNS-rebinding protection. Localhost is allowed by default; other hosts get 421. `*` disables the check **only if a token is set**. |
| `DEPTH_APPROVER_PIN` | deciding a request needs the PIN |
| `DEPTH_MCP_MAX_FRAMES` | frames per `run_survey` over MCP (default 24; use the jobs API for more) |

The decision log (`trace`) is never exported with `public=true`. Coordinates from the synthetic demo
track are labelled `SYNTHETIC` in every result, and the server instructions tell clients not to
present them as real positions.
