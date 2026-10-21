"""
mcp_session.py — record one end-to-end MCP session: an external agent drives DEPTH, a person decides.

The autonomy split, shown on the real server rather than described:

1. start the real server (uvicorn on a free local port, a fresh approvals log, an approver PIN);
2. a scripted MCP client — standing in for an LLM agent such as Claude — connects to ``/mcp`` (Streamable
   HTTP, the remote-agent path), lists the tools (none of them can decide), runs a survey toward a goal,
   reads the review queue and the top card's evidence and decision trace, and **asks** a person to approve
   an inspection, with a rationale built from that evidence;
3. the same client tries to approve its own request through the HTTP API without the PIN → refused (403);
4. a person approves it — the decision goes to the endpoint the studio's Approvals panel calls, with a
   name and the PIN (scripted here; in the demo a person clicks *Approve*);
5. the agent reads the outcome over MCP: approved, by whom, and the work order is what downstream
   consoles receive.

Writes ``docs/reports/mcp_session.md`` (each call, its arguments and a short result).

    python -m src.dashboard.mcp_session [--recall 0.78] [--timeout 300]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REPORT = REPO / "docs" / "reports" / "mcp_session.md"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _http(method: str, url: str, body: dict | None = None, timeout: float = 30.0) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def _wait_ready(base: str, deadline: float) -> dict:
    while time.time() < deadline:
        try:
            code, h = _http("GET", f"{base}/api/health", timeout=5)
            if code == 200 and h.get("model_loaded"):
                return h
            if h.get("model_error"):
                raise SystemExit(f"the server could not load the model: {h['model_error']}")
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(1.0)
    raise SystemExit("the server did not become ready in time")


async def _drive(base: str, pin: str, recall: float | None) -> list[dict]:
    """The scripted agent. Returns the transcript: one entry per call."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    log: list[dict] = []

    async def call(s, tool: str, args: dict, keep) -> dict:
        t0 = time.perf_counter()
        res = await s.call_tool(tool, args)
        if res.isError:
            raise RuntimeError(f"{tool} failed: {res.content[0].text[:300]}")
        out = json.loads(res.content[0].text)
        log.append({"via": "MCP", "call": tool, "args": args, "result": keep(out),
                    "ms": round((time.perf_counter() - t0) * 1000)})
        return out

    async with streamablehttp_client(f"{base}/mcp") as (rd, wr, _):
        async with ClientSession(rd, wr) as s:
            init = await s.initialize()
            tools = [t for t in (await s.list_tools()).tools]
            names = [t.name for t in tools]
            writes = [t.name for t in tools if t.annotations and t.annotations.readOnlyHint is False]
            log.append({"via": "MCP", "call": "initialize + list_tools", "args": {},
                        "result": {"server": init.serverInfo.name, "tools": len(names),
                                   "not_read_only": writes,
                                   "a_tool_that_decides": [n for n in names if "decide" in n or "approve_" in n]},
                        "ms": None})
            goal = {"use_samples": True, "gps": "synthetic"} | ({"recall": recall} if recall else {})
            sv = await call(s, "run_survey", goal, lambda o: {
                "survey_id": o["survey_id"], "frames": o["frames"], "counts": o["counts"],
                "promise": o["guarantees"]["recall_promise"], "mode": o["agent"]["mode"],
                "why": next((x for x in o["agent"]["steps"] if x.startswith("choose_mode")), ""),
                "gps": o["gps"], "human_approval_required": o["human_approval_required"]})
            sid = sv["survey_id"]
            q = await call(s, "get_review_queue", {"survey_id": sid, "limit": 3}, lambda o: {
                "queue_length": o["queue_length"],
                "top": [{k: c[k] for k in ("rank", "id", "p_pot", "conf", "shadow")} for c in o["cards"]]})
            top = q["cards"][0]
            hz = await call(s, "get_hazard", {"survey_id": sid, "hazard_id": top["id"]}, lambda o: {
                "tier": o.get("verdict"), "p_pot": o.get("p_pot"), "evidence": o.get("evidence_notes", [])[:3],
                "trace": [t["tool"] for t in o.get("trace") or []]})
            notes = hz.get("evidence_notes", [])
            rationale = (f"Top of the review queue: {top['id']}, P(pot) {top['p_pot']:.2f} (detector {top['conf']:.2f}); "
                         + "; ".join(notes[:2]) + ". Send the boat to inspect it.")
            rq = await call(s, "request_human_approval", {"survey_id": sid, "action": "inspect",
                                                          "targets": [top["id"]], "rationale": rationale},
                            lambda o: {k: o[k] for k in ("id", "status", "action", "targets", "channel")})
            rid = rq["id"]

            code, body = _http("POST", f"{base}/api/approvals/{rid}/decide",
                               {"decision": "approved", "by": "mcp-agent"})
            log.append({"via": "HTTP", "call": f"POST /api/approvals/{rid}/decide (the agent, no PIN)",
                        "args": {"decision": "approved", "by": "mcp-agent"},
                        "result": {"status_code": code, "detail": body.get("detail")}, "ms": None})
            code, body = _http("POST", f"{base}/api/approvals/{rid}/decide",
                               {"decision": "approved", "by": "A. Person (studio)", "pin": pin,
                                "note": "evidence card checked"})
            log.append({"via": "HTTP", "call": f"POST /api/approvals/{rid}/decide (a person, studio, with the PIN)",
                        "args": {"decision": "approved", "by": "A. Person (studio)", "pin": "******"},
                        "result": {"status_code": code, "status": body.get("status"),
                                   "decided_by": (body.get("decision") or {}).get("by")}, "ms": None})
            await call(s, "get_approval_status", {"request_id": rid}, lambda o: {
                k: o["requests"][0][k] for k in ("id", "status", "action", "targets")}
                | {"decided_by": (o["requests"][0].get("decision") or {}).get("by")})
    return log


def _fmt(v) -> str:
    s = json.dumps(v, ensure_ascii=False)
    return s if len(s) <= 420 else s[:417] + "…"


def report(log: list[dict], health: dict, wall_s: float) -> str:
    L = ["# An external agent drives DEPTH over MCP; a person decides — a recorded session", "",
         "_`python -m src.dashboard.mcp_session` · the real server (uvicorn) on a local port with a fresh "
         "approvals log and an approver PIN · a scripted MCP client over Streamable HTTP (`/mcp`, the "
         "remote-agent path) stands in for an LLM agent such as Claude · regenerated by the script — do not "
         "edit by hand_", "",
         f"Server: OpenCV {health.get('opencv')} · model {health.get('model')} · cv2.dnn engine "
         f"{(health.get('runtime') or {}).get('engine')} · approver PIN required: "
         f"{health.get('approver_pin_required')} · session {wall_s:.0f} s.", "",
         "| # | via | call | sent | came back |", "|--:|---|---|---|---|"]
    for i, e in enumerate(log, 1):
        L.append(f"| {i} | {e['via']} | `{e['call']}` | {_fmt(e['args']) if e['args'] else '—'} | "
                 f"{_fmt(e['result'])}" + (f" ({e['ms']} ms)" if e.get("ms") else "") + " |")
    L += ["", "## What it shows", "",
          "| the agent decides alone | it asks a person | only a person does |", "|---|---|---|",
          "| how hard to look (the goal and budgets → standard or look-harder), whether its promise applies, "
          "which seabed measurement to trust, where pins go, the queue order, which passes are worth boat time "
          "| inspect · recover · re-survey · full review · public sharing — `request_human_approval`, with a "
          "rationale | approve or decline (name + PIN; five wrong PINs lock the address out), confirm or "
          "reject a card, mark recovered / not found — every decision re-plans |", "",
          "- No MCP tool can decide (row 1): of the three that are not read-only, two compute (`analyze_frame`, "
          "`run_survey`) and the third only files a request.",
          "- The agent cannot approve its own request through the HTTP API either: without the PIN it is "
          "refused (403).",
          "- A person's approval becomes a work order the agent can read back; downstream consoles receive it "
          "as the signed `approval.decided` webhook. DEPTH itself never dispatches.",
          "- Here the person's click is scripted (the same endpoint the studio's Approvals panel calls); in "
          "the demo a person presses *Approve* in the studio.", "",
          "---", "_Regenerated by the script; do not edit by hand._"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Record an end-to-end MCP session against the real DEPTH server.")
    ap.add_argument("--recall", type=float, default=None, help="the goal's recall (default: the strongest promise)")
    ap.add_argument("--timeout", type=float, default=300.0)
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    port, pin = _free_port(), f"{secrets.randbelow(10 ** 6):06d}"
    base = f"http://127.0.0.1:{port}"
    tmp = tempfile.mkdtemp(prefix="depth-mcp-session-")
    env = {**os.environ, "DEPTH_APPROVER_PIN": pin, "DEPTH_APPROVALS_DIR": tmp, "DEPTH_PUBLIC": "0",
           "PYTHONPATH": str(REPO)}
    for k in ("DEPTH_MCP_TOKEN", "DEPTH_ORIGIN_SECRET", "DEPTH_TRUST_PROXY"):
        env.pop(k, None)
    srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.dashboard.app:app", "--host", "127.0.0.1",
                            "--port", str(port)], cwd=REPO, env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    try:
        health = _wait_ready(base, t0 + a.timeout)
        log = asyncio.run(asyncio.wait_for(_drive(base, pin, a.recall), a.timeout))
    finally:
        srv.terminate()
        try:
            srv.wait(15)
        except subprocess.TimeoutExpired:
            srv.kill()
    REPORT.write_text(report(log, health, time.time() - t0), encoding="utf-8")
    for e in log:
        print(f"{e['via']:4} {e['call']}: {_fmt(e['result'])[:160]}")
    print(f"wrote {REPORT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
