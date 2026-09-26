"""
Tests for DEPTH's MCP server (src/dashboard/mcp_server.py) and the human-approval flow:

* Streamable HTTP at /mcp (JSON-RPC through the app, lifespan running): initialize, tools, a survey,
  an approval REQUEST that a person then decides over REST, bearer-token auth, DNS-rebinding guard;
* stdio with the official MCP client (the Claude Desktop / Claude Code path);
* the approval store: agents ask, people decide, append-only.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fastapi.testclient import TestClient

from src.agentic.approvals import ApprovalError, ApprovalStore
from src.dashboard import app as app_mod
from src.dashboard import mcp_server
from src.detection.infer import DEFAULT_ONNX

H = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json",
     "MCP-Protocol-Version": "2025-06-18"}
_HAVE_MODEL = DEFAULT_ONNX.exists()


def _rpc(c, method, params=None, i=1, headers=None):
    r = c.post("/mcp", headers={**H, **(headers or {})},
               json={"jsonrpc": "2.0", "id": i, "method": method, "params": params or {}})
    return r


def _call(c, name, args, i=9):
    res = _rpc(c, "tools/call", {"name": name, "arguments": args}, i).json()["result"]
    text = res["content"][0]["text"] if res.get("content") else ""
    return res.get("isError", False), (json.loads(text) if text[:1] in "{[" else text)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    app_mod._APPROVALS = ApprovalStore(tmp_path_factory.mktemp("approvals"))
    hosts = mcp_server.mcp.settings.transport_security.allowed_hosts
    if "testserver" not in hosts:
        hosts.append("testserver")
    os.environ["DEPTH_LAZY_MODEL"] = "1"
    with TestClient(app_mod.app) as c:
        yield c


def test_http_initialize_and_tool_surface(client):
    r = _rpc(client, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                    "clientInfo": {"name": "pytest", "version": "0"}})
    assert r.status_code == 200 and r.json()["result"]["serverInfo"]["name"] == "DEPTH"
    tools = {t["name"]: t for t in _rpc(client, "tools/list", i=2).json()["result"]["tools"]}
    for need in ("analyze_frame", "run_survey", "get_review_queue", "get_hazard", "get_mission_brief",
                 "get_resurvey_plan", "export_report", "request_human_approval", "get_approval_status"):
        assert need in tools
    assert not any(w in n for n in tools for w in ("approve", "dispatch", "decide", "label"))   # agents can only ASK
    assert tools["get_review_queue"]["annotations"]["readOnlyHint"] is True
    assert "depth://guarantees" in [x["uri"] for x in _rpc(client, "resources/list", i=3).json()["result"]["resources"]]


def test_dns_rebinding_and_bearer_token(client, monkeypatch):
    assert _rpc(client, "tools/list", headers={"Host": "evil.example"}).status_code == 421
    monkeypatch.setenv("DEPTH_MCP_TOKEN", "s3cret")
    assert _rpc(client, "tools/list").status_code == 401
    assert _rpc(client, "tools/list", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert _rpc(client, "tools/list", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_agent_asks_person_decides(client):
    if not _HAVE_MODEL:
        pytest.skip("model absent")
    err, s = _call(client, "run_survey", {"budget_minutes": 2})
    assert not err and s["human_approval_required"] is True and s["gps"].startswith("SYNTHETIC")
    sid = s["survey_id"]
    err, q = _call(client, "get_review_queue", {"survey_id": sid, "limit": 3})
    q = q["cards"]
    assert not err and len(q) == 3 and q[0]["p_pot"] >= q[-1]["p_pot"]
    err, h = _call(client, "get_hazard", {"survey_id": sid, "hazard_id": q[0]["id"]})
    assert not err and h["trace"] and h["trace"][-1]["tool"] == "decide"
    err, b = _call(client, "get_mission_brief", {"survey_id": sid})
    assert not err and b["grounding"]["ok"]
    err, bad = _call(client, "request_human_approval", {"survey_id": sid, "action": "inspect", "targets": ["H999"],
                                                        "rationale": "made-up target id"})
    assert err                                                           # unknown IDs are refused
    err, r = _call(client, "request_human_approval", {"survey_id": sid, "action": "inspect", "targets": [q[0]["id"]],
                                                      "rationale": "highest P(pot) card with a clear shadow"})
    assert not err and r["status"] == "pending" and r["channel"] == "mcp"
    d = client.post(f"/api/approvals/{r['id']}/decide", json={"decision": "approved", "by": "R. Analyst", "note": "ok"})
    assert d.status_code == 200 and d.json()["status"] == "approved"
    assert client.post(f"/api/approvals/{r['id']}/decide", json={"decision": "declined", "by": "x"}).status_code == 409
    err, st = _call(client, "get_approval_status", {"request_id": r["id"]})
    st = st["requests"]
    assert not err and st[0]["status"] == "approved" and st[0]["decision"]["by"] == "R. Analyst"


def test_approval_store_rules(tmp_path, monkeypatch):
    st = ApprovalStore(tmp_path)
    with pytest.raises(ApprovalError):
        st.request("s", "launch_boat", ["H1"], "a long enough rationale")         # unknown action
    with pytest.raises(ApprovalError):
        st.request("s", "inspect", ["H1"], "short")                               # rationale required
    r = st.request("s", "resurvey", ["RS1"], "two uncertain targets on the port side")
    monkeypatch.setenv("DEPTH_APPROVER_PIN", "4321")
    with pytest.raises(PermissionError):
        st.decide(r["id"], "approved", "A. Person")                               # automated clients lack the PIN
    assert st.decide(r["id"], "declined", "A. Person", pin="4321")["status"] == "declined"
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 2        # append-only log
    assert st.counts() == {"pending": 0, "approved": 0, "declined": 1}


def test_stdio_with_the_official_client():
    """The Claude Desktop / Claude Code path: spawn the server over stdio and talk MCP to it."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def go():
        env = {**os.environ, "DEPTH_LAZY_MODEL": "1", "PYTHONPATH": str(REPO)}
        params = StdioServerParameters(command=sys.executable, args=["-m", "src.dashboard.mcp_server"], cwd=str(REPO), env=env)
        with open(os.devnull, "w") as devnull:
            async with stdio_client(params, errlog=devnull) as (rd, wr):
                async with ClientSession(rd, wr) as s:
                    init = await s.initialize()
                    names = [t.name for t in (await s.list_tools()).tools]
                    samples = await s.call_tool("list_samples", {})
                    first = json.loads(samples.content[0].text)["samples"][0]["id"]
                    frame = await s.call_tool("analyze_frame", {"sample_id": first}) if _HAVE_MODEL else None
                    return init.serverInfo.name, names, samples, frame
    name, names, samples, frame = asyncio.run(asyncio.wait_for(go(), 180))
    assert name == "DEPTH" and "run_survey" in names and not samples.isError
    if frame is not None:                                     # the model loads lazily inside the tool call
        f = json.loads(frame.content[0].text)
        assert not frame.isError and f["candidates"] and f["provenance"]["model"]
