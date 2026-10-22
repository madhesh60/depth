"""
Tests for the approval gate under concurrency and restarts (src/agentic/approvals.py, the two decide routes,
service._withdraw_orphans). A request is decided once: two people deciding it at the same moment get one 200
and one 409 that names who decided first, never two OKs with one decision silently lost (judge audit
2026-10-22, B1). The first decision stands on replay. The agent's requests whose survey left memory (a
restart, the bounded survey memory) are withdrawn with the reason (B3); a console's own requests stay open.
No model needed: the plans are synthetic.
"""
from __future__ import annotations

import json
import threading
from collections import OrderedDict

import pytest
from fastapi.testclient import TestClient

from src.agentic.approvals import ApprovalError, ApprovalStore
from src.agentic.geo import synthetic_track
from src.agentic.mission import plan_mission
from src.agentic.types import SurveyResult, TrackedObject, Verdict
from src.dashboard import app as app_mod
from src.dashboard import service as S

FIDS = [f"Rec6_wcp_ss_port_{i:05d}" for i in range(4, 8)]


def _at_once(*calls):
    """Run the calls in threads released by one barrier; their results (or exceptions), in order."""
    gate, out = threading.Barrier(len(calls)), [None] * len(calls)

    def run(i, f):
        gate.wait()
        try:
            out[i] = f()
        except Exception as e:                                   # the test inspects it
            out[i] = e
    threads = [threading.Thread(target=run, args=(i, f)) for i, f in enumerate(calls)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    return out


def _survey(sid: str) -> SurveyResult:
    """A planned survey with review cards, an inspection route and re-survey passes (as test_human_loop)."""
    track = synthetic_track(FIDS)
    tr = [TrackedObject(f"H{i:03d}", "fishing_gear", Verdict.REVIEW, 0.5, 0.5, FIDS[i % 4], (i, 0, i + 9, 9),
                        lat=track[FIDS[i % 4]].lat + i * 1e-5, lon=track[FIDS[i % 4]].lon, geo_error_m=4.0,
                        p_pot=round(0.9 - i * 0.07, 2)) for i in range(1, 9)]
    g = {"recall_promise": 0.65, "precision_promise": None, "tau_confirm": None}
    args = {"budget_minutes": 0.5, "sec_per_card": 8.0, "boat_minutes": None}
    m = plan_mission(tr, track, g, args["budget_minutes"], args["sec_per_card"], None)
    return SurveyResult(sid, [], tr, m, track=track, plan_args=args)


def _planned(sid: str) -> SurveyResult:
    """The survey as the server holds it: the agent's requests filed, the survey in memory."""
    s = _survey(sid)
    S._sync_agent_requests(s, reason="survey planned")
    S._remember(s)
    return s


@pytest.fixture
def store(tmp_path, monkeypatch):
    st = ApprovalStore(tmp_path / "approvals")
    monkeypatch.setattr(S, "_APPROVALS", st)
    return st


# ---- the store ---------------------------------------------------------------------------------
def test_two_people_deciding_at_once_one_wins(tmp_path):
    st = ApprovalStore(tmp_path)
    for _ in range(20):                                          # the audit: 20 of 20 such races were both accepted
        rid = st.request("s", "resurvey", ["RS1"], "an uncertain target on the port side")["id"]
        a, b = _at_once(lambda: st.decide(rid, "approved", "A. Person"), lambda: st.decide(rid, "declined", "B. Person"))
        won = [x for x in (a, b) if isinstance(x, dict)]
        lost = [x for x in (a, b) if isinstance(x, ApprovalError)]
        assert len(won) == 1 and len(lost) == 1, (a, b)
        assert st.get(rid)["status"] == won[0]["status"]
        assert f"already {won[0]['status']} by {won[0]['decision']['by']}" in str(lost[0])   # who decided first
    events = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert sum(e["type"] == "decision" for e in events) == 20  # one decision per request reached the log


def test_the_agent_withdrawing_while_a_person_decides(tmp_path):
    st = ApprovalStore(tmp_path)
    for _ in range(10):
        rid = st.request("s", "inspect", ["H1"], "inspect H1 first", channel="agent")["id"]
        a, b = _at_once(lambda: st.decide(rid, "approved", "A. Person"), lambda: st.withdraw(rid, "superseded"))
        assert sum(isinstance(x, dict) for x in (a, b)) == 1 and sum(isinstance(x, ApprovalError) for x in (a, b)) == 1


def test_the_first_decision_stands_on_replay(tmp_path):
    st = ApprovalStore(tmp_path)
    rid = st.request("s", "inspect", ["H1"], "inspect H1 first")["id"]
    st.decide(rid, "approved", "A. Person")
    with (tmp_path / "events.jsonl").open("a", encoding="utf-8") as f:   # a late write (another process, an old build)
        f.write(json.dumps({"type": "decision", "id": rid, "t": 9e9, "decision": "declined", "by": "B. Person"}) + "\n")
    r = st.get(rid)
    assert r["status"] == "approved" and r["decision"]["by"] == "A. Person"
    assert st.counts() == {"pending": 0, "approved": 1, "declined": 0, "withdrawn": 0}
    with pytest.raises(ApprovalError, match="already approved by A. Person"):
        st.decide(rid, "declined", "C. Person")


# ---- the routes: one 200 and one 409, and the agent re-plans once ---------------------------------
def test_api_approve_and_decline_at_once(store):
    s = _planned("s-race-pass")
    rid = next(v for k, v in s.plan_args["requests"].items() if k.startswith("resurvey:"))
    c = TestClient(app_mod.app)
    post = lambda d, by: c.post(f"/api/approvals/{rid}/decide", json={"decision": d, "by": by})
    a, b = _at_once(lambda: post("approved", "A. Person"), lambda: post("declined", "B. Person"))
    assert sorted([a.status_code, b.status_code]) == [200, 409], (a.text, b.text)
    won, lost = (a, b) if a.status_code == 200 else (b, a)
    status = won.json()["status"]
    assert store.get(rid)["status"] == status and f"already {status}" in lost.json()["detail"]
    declines = s.plan_args.get("declined_resurvey") or []
    assert len(declines) == (1 if status == "declined" else 0)  # the decline re-plan ran once, not twice
    replans = [r for r in s.mission.replans if r["decision"] == "decline_resurvey"]
    assert len(replans) == len(declines)


def test_api_card_decisions_at_once(store):
    _planned("s-race-card")
    c = TestClient(app_mod.app)
    post = lambda d, by: c.post("/api/survey/s-race-card/decide", json={"hazard_id": "H002", "decision": d, "by": by})
    a, b = _at_once(lambda: post("confirm", "A. Person"), lambda: post("reject", "B. Person"))
    assert sorted([a.status_code, b.status_code]) == [200, 409], (a.text, b.text)
    s = S._SURVEYS["s-race-card"]
    assert len([h for h in s.mission.human_log if h["hazard"] == "H002"]) == 1


# ---- the agent's requests outlive their survey: withdrawn, with the reason --------------------------
def test_a_restart_withdraws_the_agents_orphaned_requests(store, monkeypatch):
    monkeypatch.setenv("DEPTH_LAZY_MODEL", "1")
    old = store.request("survey-before-restart", "resurvey", ["H001"], "a pass the old plan asked for",
                        requested_by=S._AGENT, channel="agent")
    console = store.request("survey-before-restart", "inspect", ["H001"], "a partner console asks too", channel="api")
    live = _planned("s-still-in-memory")
    with TestClient(app_mod.app):                                # the server starts
        pass
    r = store.get(old["id"])
    assert r["status"] == "withdrawn" and r["decision"]["by"] == S._AGENT
    assert "restarted" in r["decision"]["note"] and "re-run the survey" in r["decision"]["note"]
    assert store.get(console["id"])["status"] == "pending"      # a console's request is a work order in its own right
    assert all(store.get(i)["status"] == "pending" for i in live.plan_args["requests"].values())
    c = TestClient(app_mod.app)
    assert c.post(f"/api/approvals/{old['id']}/decide", json={"decision": "approved", "by": "A"}).status_code == 409


def test_a_survey_leaving_memory_withdraws_its_pending_requests(store, monkeypatch):
    monkeypatch.setattr(S, "MAX_SURVEYS", 1)
    monkeypatch.setattr(S, "_SURVEYS", OrderedDict())
    _planned("s-old")
    assert store.list(status="pending", survey_id="s-old")
    _planned("s-new")                                            # the bounded memory drops s-old
    assert not store.list(status="pending", survey_id="s-old")
    assert all("left the server's memory" in r["decision"]["note"] for r in store.list(survey_id="s-old"))
    assert store.list(status="pending", survey_id="s-new")
