"""
Tests for the visible agentic loop: the per-candidate decision chain (voi_check -> tools done /
skipped / failed -> update_belief -> decide -> handoff), the compute-budget stop rule, deliberate
failure handling (incidents.py), the planner's decision log + re-plans, person overrides, the
agent's approval requests, and override labels. The perceptor is stubbed (no ONNX needed) except in
the API section, which runs the real model on the shipped samples.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.infer import Detection
from src.agentic.agent import ReLookAgent, AgentConfig
from src.agentic.pipeline import AgenticPipeline, validate_frame
from src.agentic.geo import synthetic_track
from src.agentic.mission import apply_human
from src.agentic.policy import GuaranteedTiers
from src.agentic.shadow import ShadowProver, ShadowConfig
from src.agentic.types import RelookResult, Verdict

BINS = [{"lo": 0.1, "hi": 0.3, "p_pot": 0.3, "n": 10}, {"lo": 0.3, "hi": 1.0, "p_pot": 0.6, "n": 10}]


class _Stub:
    def __init__(self, dets, relook=None, fail=False):
        self.dets, self.relook, self.fail, self.relooked = dets, relook or {}, fail, []

    def perceive(self, frame):
        return list(self.dets)

    def zoom_relook(self, frame, bbox, cls_name="ghost_gear", enhance=False):
        return RelookResult(found=False, conf=0.0, gain=0.0, scale=2.0)

    def relook_batch(self, frame, items, enhance=False):
        if self.fail:
            raise RuntimeError("relook backend down")
        out = []
        for bbox, _ in items:
            d = next(d for d in self.dets if d.bbox == bbox)
            self.relooked.append(d.conf)
            c = self.relook.get(d.conf, 0.0)
            out.append(RelookResult(found=c > 0, conf=c, gain=0.0, scale=1.0))
        return out


def _tiers(**kw):
    base = dict(tau_review=0.10, tau_confirm=0.50, relook_mode="mosaic", escalate_clahe=False, p_pot_bins=BINS,
                guaranteed_class="fishing_gear", guarantees={"recall_promise": 0.6, "precision_promise": 0.85})
    base.update(kw)
    return GuaranteedTiers(**base)


def _det(conf, box):
    return Detection(bbox=box, cls_id=0, cls_name="fishing_gear", conf=conf)


def _frame():
    return (90 + np.random.default_rng(0).normal(0, 4, (200, 200, 3))).clip(0, 255).astype(np.uint8)


def _agent(stub, **cfg):
    return ReLookAgent(perceptor=stub, shadow_prover=ShadowProver(ShadowConfig(nadir="top")),
                       cfg=AgentConfig(tiers=_tiers(), **cfg))


def _tools(c):
    return [(s.tool, s.status) for s in c.trace]


# ---- the per-candidate decision chain -----------------------------------------------------------
def test_every_candidate_trace_is_one_chain_with_skips_recorded():
    dets = [_det(0.70, (10, 10, 30, 30)), _det(0.30, (60, 60, 80, 80)), _det(0.06, (120, 120, 140, 140))]
    r = _agent(_Stub(dets, {0.30: 0.62})).run_frame(_frame(), frame_id="Rec9_wcp_ss_port_00001")
    for c in r.candidates:
        t = [s.tool for s in c.trace]
        assert t[0] == "voi_check" and t[-2:] == ["decide", "handoff"]
        assert c.trace[0].detail["question"] == "can another observation change the decision?"
    hi, band, low = r.candidates
    assert hi.trace[0].detail["answer"] == "no" and ("zoom_relook", "skipped") in _tools(hi)
    assert band.trace[0].detail["answer"] == "yes" and ("mosaic_relook", "done") in _tools(band)
    assert ("update_belief", "done") in _tools(band)
    ub = next(s for s in band.trace if s.tool == "update_belief")
    assert ub.detail == {"p_prior": 0.6, "p_post": 0.6} or ub.detail["p_post"] >= ub.detail["p_prior"]
    assert ("zoom_relook", "skipped") in _tools(low) and "update_belief" not in [s.tool for s in low.trace]
    assert all(s.rationale for c in r.candidates for s in c.trace)           # every step says why


def test_compute_budget_stops_relooking_and_records_the_skip():
    dets = [_det(0.30, (10, 10, 30, 30)), _det(0.20, (60, 60, 80, 80)), _det(0.25, (120, 120, 140, 140))]
    stub = _Stub(dets)
    r = _agent(stub, max_relooks=1).run_frame(_frame())
    assert stub.relooked == [0.30]                                          # the likeliest one only
    skipped = [s for c in r.candidates for s in c.trace if s.status == "skipped" and s.detail.get("reason") == "compute_budget"]
    assert len(skipped) == 2 and "stop" in skipped[0].rationale


def test_a_failed_tool_falls_back_and_never_auto_confirms():
    dets = [_det(0.30, (60, 60, 80, 80))]
    r = _agent(_Stub(dets, {0.30: 0.9}, fail=True)).run_frame(_frame(), frame_id="f1")
    c = r.candidates[0]
    assert ("mosaic_relook", "failed") in _tools(c) or ("zoom_relook", "failed") in _tools(c)
    assert c.verdict is Verdict.REVIEW and c.evidence.evidence_score == 0.30
    assert "failed" in c.evidence.notes[0]
    assert [i["code"] for i in r.incidents] == ["tool_failed"]


def test_tool_failure_drill_and_no_detection_incident():
    dets = [_det(0.30, (60, 60, 80, 80))]
    r = _agent(_Stub(dets, {0.30: 0.9})).run_frame(_frame(), faults={"tool_failed"})
    inc = r.incidents[0]
    assert inc["code"] == "tool_failed" and inc["simulated"] and inc["fallback"] and inc["human_action"]
    assert r.candidates[0].verdict is not Verdict.CONFIRMED
    empty = _agent(_Stub([])).run_frame(_frame(), frame_id="f2")
    assert empty.incidents[0]["code"] == "no_detection"


def test_validate_frame_catches_corrupt_blank_and_tiny():
    assert validate_frame(None) == "corrupt_frame"
    assert validate_frame(np.zeros((100, 100, 3), np.uint8)) == "invalid_frame"
    assert validate_frame(np.full((10, 10, 3), 7, np.uint8)) == "invalid_frame"
    assert validate_frame(_frame()) is None


# ---- the survey: failure boundaries, geometry, planner log ----------------------------------------
IDS = [f"Rec9_wcp_ss_port_000{i:02d}" for i in range(1, 5)]


def _survey(gps=True, faults=None, dets=None, boat=None, budget=0.05):
    dets = dets or [_det(0.30, (60, 60, 80, 80)), _det(0.20, (120, 30, 140, 50))]
    pipe = AgenticPipeline(agent=_agent(_Stub(dets, {0.30: 0.35, 0.20: 0.0})))
    frames = [(fid, _frame()) for fid in IDS] + [("broken", None)]
    track = synthetic_track(IDS) if gps else None
    return pipe.run_survey(frames, track=track, survey_id="t", budget_minutes=budget, boat_minutes=boat,
                           faults=faults or set())


def test_a_corrupt_frame_is_skipped_and_reported_not_fatal():
    s = _survey()
    assert len(s.frames) == 4
    assert any(i["code"] == "corrupt_frame" and i["frame_id"] == "broken" for i in s.mission.incidents)


def test_invalid_geometry_withholds_the_position_never_guesses():
    s = _survey(faults={"invalid_geometry"})
    fr = s.frames[0]
    assert any(i["code"] == "invalid_geometry" for i in fr.incidents)
    assert all(c.lat is None and c.position_withheld for c in fr.candidates)
    withheld = [t for t in s.tracked if t.frame_id == IDS[0]]
    assert withheld and all(t.lat is None and t.position_withheld for t in withheld)
    assert all(t.lat is not None for t in s.tracked if t.frame_id != IDS[0])


def test_missing_gps_is_an_incident_and_the_route_is_skipped():
    s = _survey(gps=False)
    assert any(i["code"] == "missing_gps" for i in s.mission.incidents)
    route = next(st for st in s.mission.agent_log if st["tool"] == "route_update")
    assert route["status"] == "skipped" and "no GPS" in route["rationale"]


def test_planner_log_reads_budget_stop_route_gate():
    s = _survey(budget=0.05)                                  # 3 s -> 0 cards at 8 s: everything deferred
    tools = [st["tool"] for st in s.mission.agent_log]
    for t in ("triage", "voi_check", "budget_check", "stop_rule", "route_update", "recovery_route", "dispatch_gate"):
        assert t in tools
    assert tools[-1] == "dispatch_gate" and s.mission.agent_log[-1]["detail"]["awaiting_approval"] is True
    stop = next(st for st in s.mission.agent_log if st["tool"] == "stop_rule")
    assert "deferred" in stop["rationale"]


def test_boat_budget_records_the_passes_it_did_not_plan():
    s = _survey(boat=0.01, budget=5)
    rp = s.mission.resurvey_plan
    assert not rp["lines"] and rp["skipped"] and "boat budget" in rp["skipped"][0]["reason"]
    assert any(st["tool"] == "resurvey_plan" and st["status"] == "skipped" for st in s.mission.agent_log)


# ---- people override the agent; the agent re-plans and says what changed ---------------------------
def test_prioritize_override_reorders_queue_and_replan_records_the_diff():
    s = _survey(budget=5)
    last = s.mission.review_queue[-1]
    apply_human(s, last, "prioritize", "RK", "diver saw it")
    assert s.mission.review_queue[0] == last
    assert any(st["tool"] == "human_override" and last in st["detail"]["prioritized"] for st in s.mission.agent_log)
    rp = s.mission.replans[-1]
    assert rp["hazard"] == last and rp["by"] == "RK" and "re-planned" in rp["rationale"]
    first = s.mission.review_queue[0]
    apply_human(s, first, "reject", "RK")
    assert first not in s.mission.review_queue and s.mission.replans[-1]["changed"]["queue"][1] == len(s.mission.review_queue)
    with pytest.raises(ValueError):
        apply_human(s, first, "prioritize", "RK")             # closed by a person: not in the queue
    t = next(x for x in s.tracked if x.oid == first)
    assert t.verdict is Verdict.REVIEW                         # the agent's verdict is never overwritten


def test_reset_priority_and_replans_survive_each_replan():
    s = _survey(budget=5)
    oid = s.mission.review_queue[-1]
    apply_human(s, oid, "deprioritize", "A")
    assert s.mission.review_queue[-1] == oid
    apply_human(s, oid, "reset_priority", "A")
    assert len(s.mission.replans) == 2 and len(s.mission.human_log) == 2
    with pytest.raises(ValueError):
        apply_human(s, oid, "reset_priority", "A")


# ---- approvals: the agent can withdraw its own stale request, never a decided one -------------------
def test_withdraw_only_pending(tmp_path):
    from src.agentic.approvals import ApprovalStore, ApprovalError
    st = ApprovalStore(tmp_path)
    r = st.request("s1", "inspect", ["H1"], "inspect H1 first", requested_by="DEPTH agent", channel="agent")
    assert st.withdraw(r["id"], "superseded")["status"] == "withdrawn"
    r2 = st.request("s1", "inspect", ["H2"], "inspect H2 first")
    st.decide(r2["id"], "approved", "RK")
    with pytest.raises(ApprovalError):
        st.withdraw(r2["id"], "too late")
    assert st.counts()["withdrawn"] == 1


# ---- override labels -----------------------------------------------------------------------------
def test_override_labels_export_as_positive_or_hard_negative(tmp_path, monkeypatch):
    from src.agentic.feedback import FeedbackStore
    from src.dashboard import samples as samples_mod
    fs = FeedbackStore(tmp_path / "fb")
    img = _frame()
    import cv2
    cv2.imwrite(str(tmp_path / "x.jpg"), img)
    monkeypatch.setattr(samples_mod, "sample_path", lambda sid: tmp_path / "x.jpg")
    monkeypatch.setattr(samples_mod, "is_heldout", lambda sid: False, raising=False)
    base = {"frame_id": "x", "frame_ref": {"sample": "x"}, "cls_name": "fishing_gear", "decision": "override"}
    with pytest.raises(ValueError):
        fs.add({**base, "bbox": [1, 1, 20, 20]})                       # needs override_to
    fs.add({**base, "bbox": [1, 1, 20, 20], "override_to": "confirmed"})
    fs.add({**base, "bbox": [100, 100, 130, 130], "override_to": "low_risk"})
    fs.add({**base, "bbox": [50, 150, 70, 170], "override_to": "review"})
    out = fs.export(tmp_path / "ft", ["fishing_gear"])
    assert out["positives"] == 1 and out["hard_negatives"] == 1
    assert fs.stats()["override"] == 3


# ---- the API: drills, the audit log, decisions that re-plan and re-file approvals -----------------
@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from src.dashboard.app import app
    return TestClient(app)


def test_incident_catalogue_and_unknown_drill(client):
    j = client.get("/api/incidents").json()
    codes = {c["code"] for c in j["catalogue"]}
    for c in ("no_detection", "corrupt_frame", "missing_gps", "invalid_geometry", "model_unavailable",
              "tool_failed", "storage_failed", "llm_failed"):
        assert c in codes
    assert all(c["fallback"] and c["human_action"] for c in j["catalogue"])
    from src.dashboard import samples as samples_mod
    sid = samples_mod.list_samples()[0]["id"]
    assert client.post(f"/api/analyze?sample={sid}&simulate=nope").status_code == 400


def test_analyze_drills(client):
    from src.dashboard import samples as samples_mod
    sid = samples_mod.list_samples()[0]["id"]
    r = client.post(f"/api/analyze?sample={sid}&simulate=model_unavailable")
    assert r.status_code == 503 and r.json()["detail"]["incident"]["code"] == "model_unavailable"
    r = client.post(f"/api/analyze?sample={sid}&simulate=corrupt_frame")
    assert r.status_code == 422 and r.json()["detail"]["incident"]["simulated"] is True
    body = client.post(f"/api/analyze?sample={sid}&simulate=no_detection").json()
    assert body["candidates"] == [] and body["incidents"][0]["code"] == "no_detection"


def test_survey_drills_log_and_replan(client):
    r = client.post("/api/survey?use_samples=1&gps=synthetic&budget_minutes=1&boat_minutes=60"
                    "&simulate=corrupt_frame,storage_failed,invalid_geometry")
    assert r.status_code == 200, r.text
    d = r.json()
    codes = {i["code"] for i in d["mission"]["incidents"]} | {i["code"] for f in d["frames"] for i in f["incidents"]}
    assert {"corrupt_frame", "storage_failed", "invalid_geometry"} <= codes
    assert d["mission"]["agent_log"][0]["tool"] == "triage"
    assert any(st["tool"] == "request_approval" for st in d["mission"]["agent_log"])
    assert d["dispatch"] and all(v["status"] == "pending" for v in d["dispatch"].values())
    sid = d["survey_id"]
    before = d["dispatch"].get("inspect")
    target = d["mission"]["review_queue"][-1]
    rr = client.post(f"/api/survey/{sid}/decide", json={"hazard_id": target, "decision": "prioritize", "by": "RK"})
    assert rr.status_code == 200, rr.text
    j = rr.json()
    assert j["replan"]["hazard"] == target and j["mission"]["review_queue"][0] == target
    if before and j["replan"]["changed"].get("inspection_stops") or before and j["dispatch"]["inspect"]["id"] != before["id"]:
        assert any("withdrawn" in c for c in j["replan"]["requests"])
    rj = client.post(f"/api/survey/{sid}/decide", json={"hazard_id": target, "decision": "reject", "by": "RK"}).json()
    assert rj["label"] is not None and rj["label"]["decision"] == "reject" and rj["label"]["source"] == "survey"
    log = client.get(f"/api/survey/{sid}/log").json()
    actors = {e["actor"] for e in log["entries"]}
    assert {"agent", "person", "system"} <= actors
    kinds = {e["kind"] for e in log["entries"]}
    assert {"incident", "approval_request", "decision", "replan", "dispatch_gate"} <= kinds
    b = client.get(f"/api/brief?survey_id={sid}&simulate=llm_failed").json()
    assert b["writer"] == "template" and b["incident"]["code"] == "llm_failed" and "never decides" in b["authority"]
    assert any(e["kind"] == "brief" for e in client.get(f"/api/survey/{sid}/log").json()["entries"])


def test_missing_gps_drill_on_the_api(client):
    d = client.post("/api/survey?use_samples=1&gps=synthetic&simulate=missing_gps").json()
    assert not d["mission"]["gps_available"]
    inc = next(i for i in d["mission"]["incidents"] if i["code"] == "missing_gps")
    assert inc["simulated"] is True
