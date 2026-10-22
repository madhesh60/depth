"""
llm_session.py — a real LLM agent (Claude) drives DEPTH over MCP; a person decides. A recorded session.

``mcp_session.py`` shows the autonomy split with a *scripted* client. This one hands the same MCP tools to a
real model and lets it choose every call itself:

1. start the real server (uvicorn on a free local port, a fresh approvals log, an approver PIN);
2. connect to ``/mcp`` (Streamable HTTP), give Claude the tool list exactly as the server describes it, and
   the task: survey the shipped frames, look at the evidence, and ask a person for what the boat should do.
   Claude picks the tools and their arguments; every call and its result is logged;
3. a person approves the agent's requests (the endpoint the studio's Approvals panel calls, with the PIN -
   scripted here); Claude is told and reads the outcome back.

No tool can decide, and the server refuses an approval without the PIN, so the model can only ask. Writes
``docs/reports/llm_session.md``.

Needs ``pip install anthropic`` and credentials: the Claude API (``ANTHROPIC_API_KEY``, or an
``ant auth login`` profile) or, with ``--bedrock``, Amazon Bedrock (AWS credentials, e.g. the ``hackathon``
profile).

    python -m src.dashboard.llm_session [--bedrock] [--model claude-opus-5-5] [--max-turns 16]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .mcp_session import _fmt, _free_port, _http, _wait_ready

REPO = Path(__file__).resolve().parents[2]
REPORT = REPO / "docs" / "reports" / "llm_session.md"
MODEL = "claude-opus-5-5"

SYSTEM = (
    "You operate DEPTH, a side-scan sonar system that finds lost crab pots (ghost gear) and wreck debris, "
    "through its MCP tools. You cannot approve or dispatch anything: you can only ask a named person, with "
    "request_human_approval, and the person decides. Base every request on evidence you read with the tools "
    "(the review queue, a hazard's evidence and trace, the recall promise and whether it applies), quote "
    "that evidence in the rationale, and keep to the targets the tools returned. Be brief.")
TASK = (
    "Survey the sonar frames that ship with DEPTH (use_samples true, synthetic GPS). Then decide what the "
    "boat crew should do first: read the review queue and the evidence of the most promising find(s), and "
    "file the approval request(s) you think are justified. Finish with a short summary of what you asked "
    "for and why.")
FOLLOW_UP = ("A person has now decided on your request(s) in the DEPTH studio. Check the outcome with the "
             "tools and say in two sentences what happens next.")


def tool_specs(tools) -> list[dict]:
    """MCP tools -> Messages API tool definitions (the server's own names, descriptions and schemas)."""
    return [{"name": t.name, "description": (t.description or "")[:1024], "input_schema": t.inputSchema}
            for t in tools]


def run_agent(client, model: str, tools: list[dict], messages: list, call_tool: Callable[[str, dict], str],
              log: list, max_turns: int) -> str:
    """The tool-use loop: the model picks every call; each one runs on the real server and is logged.
    Returns the model's final text."""
    for _ in range(max_turns):
        resp = client.messages.create(model=model, max_tokens=16000, system=SYSTEM, tools=tools,
                                      messages=messages, output_config={"effort": "medium"})
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason == "refusal":
            return "(the model declined)"
        uses = [b for b in resp.content if b.type == "tool_use"]
        if resp.stop_reason != "tool_use" or not uses:
            return "".join(b.text for b in resp.content if b.type == "text").strip()
        results = []
        for b in uses:
            t0 = time.perf_counter()
            try:
                out, err = call_tool(b.name, dict(b.input)), False
            except Exception as e:                       # a refused call goes back to the model, as an error
                out, err = f"{type(e).__name__}: {e}", True
            log.append({"via": "MCP (Claude chose)", "call": b.name, "args": dict(b.input),
                        "result": out[:600], "error": err, "ms": round((time.perf_counter() - t0) * 1000)})
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": out, "is_error": err})
        messages.append({"role": "user", "content": results})
    return "(stopped: turn limit)"


def _client(bedrock: bool):
    import anthropic
    if bedrock:
        return anthropic.AnthropicBedrockMantle(aws_region=os.environ.get("AWS_REGION", "us-east-1"))
    return anthropic.Anthropic()


async def _session(base: str, pin: str, client, model: str, max_turns: int) -> tuple[list, list[str]]:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    log: list[dict] = []
    finals: list[str] = []
    async with streamablehttp_client(f"{base}/mcp") as (rd, wr, _):
        async with ClientSession(rd, wr) as s:
            await s.initialize()
            tools = tool_specs((await s.list_tools()).tools)
            loop = asyncio.get_running_loop()

            def call_tool(name: str, args: dict) -> str:
                res = asyncio.run_coroutine_threadsafe(s.call_tool(name, args), loop).result(600)
                text = res.content[0].text if res.content else ""
                if res.isError:
                    raise RuntimeError(text[:400])
                return text

            messages: list[Any] = [{"role": "user", "content": TASK}]
            finals.append(await asyncio.to_thread(run_agent, client, model, tools, messages, call_tool, log, max_turns))
            rids = [json.loads(e["result"])["id"] for e in log if e["call"] == "request_human_approval"
                    and not e["error"] and e["result"].startswith("{")]
            for rid in rids:                             # a person decides (scripted; the studio's endpoint)
                code, body = _http("POST", f"{base}/api/approvals/{rid}/decide",
                                   {"decision": "approved", "by": "A. Person (studio)", "pin": pin,
                                    "note": "evidence card checked"})
                log.append({"via": "HTTP (a person, with the PIN)", "call": f"POST /api/approvals/{rid}/decide",
                            "args": {"decision": "approved", "by": "A. Person (studio)", "pin": "******"},
                            "result": json.dumps({"status_code": code, "status": body.get("status")}),
                            "error": code != 200, "ms": None})
            if rids:
                messages.append({"role": "user", "content": FOLLOW_UP})
                finals.append(await asyncio.to_thread(run_agent, client, model, tools, messages, call_tool, log,
                                                      max_turns))
    return log, finals


def report(log: list, finals: list[str], model: str, health: dict, wall_s: float) -> str:
    asked = [e for e in log if e["call"] == "request_human_approval"]
    L = ["# A real LLM agent drives DEPTH over MCP; a person decides — a recorded session", "",
         f"_`python -m src.dashboard.llm_session` · model `{model}` chose every tool call itself · the real server "
         "on a local port with a fresh approvals log and an approver PIN · regenerated by the script — do not edit "
         "by hand_", "",
         f"Server: OpenCV {health.get('opencv')} · model {health.get('model')} · session {wall_s:.0f} s · "
         f"{sum(1 for e in log if e['via'].startswith('MCP'))} tool calls · {len(asked)} approval request(s).", "",
         "| # | via | call | sent | came back |", "|--:|---|---|---|---|"]
    for i, e in enumerate(log, 1):
        L.append(f"| {i} | {e['via']} | `{e['call']}` | {_fmt(e['args']) if e['args'] else '—'} | "
                 f"{'**error:** ' if e['error'] else ''}{_fmt(e['result'])}"
                 + (f" ({e['ms']} ms)" if e.get("ms") else "") + " |")
    L += ["", "## What the agent said", ""]
    for i, t in enumerate(finals, 1):
        L += [f"**{'After the survey' if i == 1 else 'After the person decided'}:**", "", "> " + t.replace("\n", "\n> "), ""]
    L += ["## What it shows", "",
          "- The model chose which tools to call and with what arguments; DEPTH's OpenCV results (the queue, the "
          "evidence, the promise) shaped its requests.",
          "- It could only ask: no MCP tool decides, and an approval needs a person's PIN.",
          "- The person's click is scripted here (the endpoint the studio's Approvals panel calls).", "",
          "---", "_Regenerated by the script; do not edit by hand._"]
    return "\n".join(L) + "\n"


def main(argv: Optional[list[str]] = None):
    ap = argparse.ArgumentParser(description="Record a real LLM agent driving DEPTH over MCP.")
    ap.add_argument("--bedrock", action="store_true", help="Claude on Amazon Bedrock (AWS credentials)")
    ap.add_argument("--model", default=None, help=f"default {MODEL} (Bedrock: anthropic.{MODEL})")
    ap.add_argument("--max-turns", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=600.0)
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    model = a.model or (f"anthropic.{MODEL}" if a.bedrock else MODEL)
    client = _client(a.bedrock)
    port, pin = _free_port(), f"{secrets.randbelow(10 ** 6):06d}"
    base = f"http://127.0.0.1:{port}"
    env = {**os.environ, "DEPTH_APPROVER_PIN": pin, "DEPTH_APPROVALS_DIR": tempfile.mkdtemp(prefix="depth-llm-"),
           "DEPTH_PUBLIC": "0", "PYTHONPATH": str(REPO)}
    for k in ("DEPTH_MCP_TOKEN", "DEPTH_ORIGIN_SECRET", "DEPTH_TRUST_PROXY"):
        env.pop(k, None)
    srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.dashboard.app:app", "--host", "127.0.0.1",
                            "--port", str(port)], cwd=REPO, env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    try:
        health = _wait_ready(base, t0 + a.timeout)
        log, finals = asyncio.run(asyncio.wait_for(_session(base, pin, client, model, a.max_turns), a.timeout))
    finally:
        srv.terminate()
        try:
            srv.wait(15)
        except subprocess.TimeoutExpired:
            srv.kill()
    REPORT.write_text(report(log, finals, model, health, time.time() - t0), encoding="utf-8")
    print(f"{len(log)} steps -> {REPORT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
