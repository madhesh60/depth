"""
The real-LLM session (src/dashboard/llm_session.py) with a stub in place of Claude: the tool-use loop runs every
call the model picks, feeds refused calls back as errors, and stops at end_turn or the turn limit.
"""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

from src.dashboard import llm_session


class StubClaude:
    """Answers like the Messages API: a scripted list of turns, each a list of tool calls or a final text."""

    def __init__(self, turns):
        self.turns, self.seen = list(turns), []
        self.messages = self

    def create(self, **kw):
        self.seen.append(kw)
        turn = self.turns.pop(0)
        if isinstance(turn, str):
            return NS(stop_reason="end_turn", content=[NS(type="text", text=turn)])
        return NS(stop_reason="tool_use",
                  content=[NS(type="tool_use", id=f"t{i}", name=n, input=a) for i, (n, a) in enumerate(turn)])


def test_the_model_picks_the_calls_and_refused_calls_come_back_as_errors():
    calls = []

    def call_tool(name, args):
        calls.append(name)
        if name == "request_human_approval" and args.get("targets") == ["H999"]:
            raise RuntimeError("unknown target")
        return json.dumps({"id": "AR-1", "status": "pending"} if name == "request_human_approval" else {"ok": True})

    claude = StubClaude([[("run_survey", {"use_samples": True})],
                         [("request_human_approval", {"targets": ["H999"]}),
                          ("request_human_approval", {"targets": ["H001"]})],
                         "Asked to inspect H001."])
    log, msgs = [], [{"role": "user", "content": llm_session.TASK}]
    final = llm_session.run_agent(claude, "m", [], msgs, call_tool, log, max_turns=8)
    assert final == "Asked to inspect H001." and calls == ["run_survey", "request_human_approval", "request_human_approval"]
    assert [e["error"] for e in log] == [False, True, False]
    results = msgs[-2]["content"]                                      # the turn that answered the two requests
    assert results[0]["is_error"] and "unknown target" in results[0]["content"] and not results[1]["is_error"]
    assert all("temperature" not in kw and kw["system"] == llm_session.SYSTEM for kw in claude.seen)


def test_the_loop_stops_at_the_turn_limit_and_on_a_refusal():
    loop = StubClaude([[("get_review_queue", {})]] * 3)
    assert llm_session.run_agent(loop, "m", [], [], lambda n, a: "{}", [], max_turns=3) == "(stopped: turn limit)"
    refuse = StubClaude([])
    refuse.create = lambda **kw: NS(stop_reason="refusal", content=[])
    refuse.messages = refuse
    assert llm_session.run_agent(refuse, "m", [], [], lambda n, a: "{}", [], max_turns=3) == "(the model declined)"


def test_mcp_tools_become_api_tool_definitions():
    t = NS(name="run_survey", description="Run a survey.", inputSchema={"type": "object", "properties": {}})
    assert llm_session.tool_specs([t]) == [{"name": "run_survey", "description": "Run a survey.",
                                             "input_schema": {"type": "object", "properties": {}}}]
