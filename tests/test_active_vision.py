"""
Tests for active vision: value-of-information tool choice (active.py), evidence conflict and the
request for another observation, the agent's experimental switch (default off while the evidence
model's validation gates fail), information-ranked re-survey passes, a person's declined pass making
the agent re-plan, and the WITH vs WITHOUT OpenCV counterfactual.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agentic import active
from src.agentic.agent import ReLookAgent, AgentConfig
from src.agentic.evidence_model import EvidenceModel, THRESHOLDS
from src.agentic.policy import GuaranteedTiers
from src.agentic.shadow import ShadowProver, ShadowConfig
from src.agentic.types import RelookResult, TrackedObject, Verdict
from src.detection.infer import Detection


def _lr(present_tp, present_fp):
    return {"present": present_tp / present_fp, "absent": (1 - present_tp) / (1 - present_fp),
            "present_tp": present_tp, "present_fp": present_fp,
            "absent_tp": 1 - present_tp, "absent_fp": 1 - present_fp}


def _model(use=False):
    band = {"shadow": _lr(0.55, 0.40), "relook": _lr(0.87, 0.76)}
    weak = {"shadow": _lr(0.50, 0.49), "relook": _lr(0.50, 0.49)}
    high = {"shadow": _lr(0.50, 0.36), "relook": _lr(0.74, 0.69)}     # ~ the measured high band (val)
    return EvidenceModel({"bands": [weak, band, high], "band_edges": [0.15, 0.30], "thresholds": dict(THRESHOLDS),
                          "costs_ms": {"geometry_check": 0, "shadow_check": 1.0, "zoom_relook": 400.0}, "use": use,
                          "model": "TEST", "actions_reason": "test", "belief_reason": "test"})


# ---- the policy ------------------------------------------------------------------------------------
def test_action_thresholds_and_water_column_rule():
    th = THRESHOLDS
    assert active.action_of(0.7, th) == "accept"
    assert active.action_of(0.3, th) == "review"
    assert active.action_of(0.05, th) == "watch"
    assert active.action_of(0.9, th, water_column=True) == "review"        # never accepted off the seabed


def test_no_tool_runs_when_no_outcome_can_change_the_action():
    em = _model()
    tool, opts = active.next_tool(em, 0.05, 0.10, set(), THRESHOLDS)        # weak band, deep in WATCH
    assert tool is None and all(o["p_flip"] == 0 for o in opts)


def test_cheapest_informative_tool_first_and_belief_moves():
    em = _model()
    tool, opts = active.next_tool(em, 0.58, 0.40, set(), THRESHOLDS)        # just under ACCEPT
    assert tool == "shadow_check"                                          # 1 ms beats 400 ms
    assert active.update(em, 0.58, 0.40, "shadow", True) > 0.58
    assert active.update(em, 0.58, 0.40, "shadow", False) < 0.58


def test_conflict_is_detected_and_another_observation_requested():
    em = _model()
    seen = []
    res = active.run_policy(em, 0.40, 0.63, lambda t: seen.append(t) or False)   # detector ACCEPT, no shadow
    assert res["action0"] == "accept" and res["action"] != "accept"
    assert res["conflict"] and "shadow" in res["conflict"]["between"]
    assert seen[0] == "shadow_check"
    assert res["request_resurvey"] and res["resurvey"]["p_flip"] > 0


def test_confident_and_consistent_find_is_accepted_without_the_relook():
    em = _model()
    seen = []
    res = active.run_policy(em, 0.40, 0.63, lambda t: seen.append(t) or True)
    assert res["action"] == "accept" and "zoom_relook" not in seen and res["conflict"] is None


def test_relook_result_can_change_the_decision():
    em = _model()
    outcomes = {"shadow_check": True, "zoom_relook": False}
    res = active.run_policy(em, 0.25, 0.53, lambda t: outcomes[t])
    assert res["tools_run"] == ["shadow_check", "zoom_relook"]
    steps = [s for s in res["steps"] if s.get("tool") == "zoom_relook"]
    assert steps and steps[0]["action_before"] == "accept" and steps[0]["action_after"] == "review"


def test_orientation_unknown_means_no_shadow_tool():
    em = _model()
    seen = []
    active.run_policy(em, 0.25, 0.53, lambda t: seen.append(t) or True, available=("zoom_relook",))
    assert "shadow_check" not in seen


# ---- the agent ---------------------------------------------------------------------------------------
class _Stub:
    def __init__(self, dets):
        self.dets = dets

    def perceive(self, frame):
        return list(self.dets)

    def zoom_relook(self, frame, bbox, cls_name="ghost_gear", enhance=False):
        return RelookResult(found=False, conf=0.0, gain=0.0, scale=2.0)

    def relook_batch(self, frame, items, enhance=False):
        return [RelookResult(found=False, conf=0.0, gain=0.0, scale=1.0) for _ in items]


BINS = [{"lo": 0.06, "hi": 0.2, "p_pot": 0.1, "n": 10}, {"lo": 0.2, "hi": 0.35, "p_pot": 0.53, "n": 10},
        {"lo": 0.35, "hi": 1.0, "p_pot": 0.63, "n": 10}]


def _agent(dets, em):
    tiers = GuaranteedTiers(tau_review=0.06, tau_confirm=None, relook_mode=None, p_pot_bins=BINS,
                            guaranteed_class="ghost_gear", guarantees={"recall_promise": 0.79})
    a = ReLookAgent(perceptor=_Stub(dets), shadow_prover=ShadowProver(ShadowConfig(nadir="top")),
                    cfg=AgentConfig(tiers=tiers))
    a.evidence_model = em
    return a


def _frame():
    return (90 + np.random.default_rng(0).normal(0, 4, (200, 200, 3))).clip(0, 255).astype(np.uint8)


def _dets():
    return [Detection(bbox=(20, 40, 50, 70), cls_id=0, cls_name="ghost_gear", conf=0.50),
            Detection(bbox=(120, 40, 150, 70), cls_id=0, cls_name="ghost_gear", conf=0.08)]


def test_default_is_off_when_the_model_is_not_validated(monkeypatch):
    monkeypatch.delenv("DEPTH_ACTIVE", raising=False)
    r = _agent(_dets(), _model(use=False)).run_frame(_frame(), frame_id="Rec9_wcp_ss_port_00001")
    assert r.active["mode"] == "off" and all(c.evidence.action is None for c in r.candidates)


def test_experimental_switch_runs_the_loop_and_labels_it(monkeypatch):
    monkeypatch.delenv("DEPTH_ACTIVE", raising=False)
    r = _agent(_dets(), _model(use=False)).run_frame(_frame(), frame_id="Rec9_wcp_ss_port_00001",
                                                     faults={"opt:active"})
    assert r.active["mode"] == "experimental"
    hi, lo = r.candidates
    tools = [s.tool for s in hi.trace]
    assert tools[:2] == ["geometry_check", "assess"] or tools[0] == "assess" or "assess" in tools
    assert "choose_tool" in tools and tools[-2:] == ["decide", "handoff"]
    d = next(s for s in hi.trace if s.tool == "decide").detail
    assert d["mode"] == "active" and d["active_mode"] == "experimental" and d["action0"] in ("accept", "review", "watch")
    assert hi.verdict is Verdict.REVIEW and lo.verdict is Verdict.REVIEW       # tiers (recall promise) untouched
    assert any("EXPERIMENTAL" in n for n in hi.evidence.notes)
    # the low-confidence find: no tool can change WATCH -> nothing run, and the skips say why
    assert not any(s.tool in ("shadow_check", "zoom_relook") and s.status == "done" for s in lo.trace)
    assert any(s.tool == "zoom_relook" and s.status == "skipped" for s in lo.trace)


def test_env_off_wins(monkeypatch):
    monkeypatch.setenv("DEPTH_ACTIVE", "off")
    r = _agent(_dets(), _model(use=True)).run_frame(_frame(), frame_id="Rec9_wcp_ss_port_00001", faults={"opt:active"})
    assert r.active["mode"] == "off" and all(c.evidence.action is None for c in r.candidates)


# ---- planning: information-ranked passes, a person's decline -> a new plan -----------------------------
def _survey(n=4):
    from src.agentic.geo import synthetic_track
    from src.agentic.mission import plan_mission
    from src.agentic.types import SurveyResult
    fids = [f"Rec9_wcp_ss_port_{i:05d}" for i in range(n)]
    track = synthetic_track(fids)
    tracked = []
    for i, fid in enumerate(fids):
        fx = track[fid]
        tracked.append(TrackedObject(oid=f"H{i + 1:03d}", cls_name="ghost_gear", verdict=Verdict.REVIEW, conf=0.5,
                                     evidence_score=0.5, frame_id=fid, bbox=(10, 10, 30, 30), lat=fx.lat + 1e-4,
                                     lon=fx.lon, geo_error_m=4.0, p_pot=0.6, p_evidence=0.55 - 0.1 * (i % 2),
                                     action="review", action0="accept", conflict={"why": "x"} if i < 2 else None,
                                     request_resurvey=i < 3, info_bits=[0.30, 0.05, 0.20, 0.40][i]))
    m = plan_mission(tracked, track, {}, None, 8.0, None)
    return SurveyResult(survey_id="s1", frames=[], tracked=tracked, mission=m, track=track, plan_args={})


def test_passes_follow_information_not_confidence():
    s = _survey()
    rp = s.mission.resurvey_plan
    planned = {t for L in rp["lines"] for t in L["targets"]}
    assert "H004" not in planned                                    # info bits but not requested by the agent
    assert rp["info_unit"] == "bits" and rp["lines"][0]["recommendation"] == "Perform opposite-side resurvey"
    assert rp["lines"] == sorted(rp["lines"], key=lambda L: -L["info_per_boat_min"])
    assert any("evidence conflict" in L["reason"] for L in rp["lines"])


def test_declined_pass_changes_the_plan():
    from src.agentic.mission import decline_resurvey
    s = _survey()
    first = s.mission.resurvey_plan["lines"][0]
    entry = decline_resurvey(s, first["targets"], "Test Person", "no boat today")
    left = {t for L in s.mission.resurvey_plan["lines"] for t in L["targets"]}
    assert not (set(first["targets"]) & left)                      # never proposed again
    assert s.mission.review_queue[:len(first["targets"])] == sorted(
        s.mission.review_queue[:len(first["targets"])], key=lambda o: s.mission.review_queue.index(o))
    assert all(t.fallback == "inspect" for t in s.tracked if t.oid in first["targets"])
    assert set(s.mission.review_queue[:len(first["targets"])]) == set(first["targets"])
    assert entry["decision"] == "decline_resurvey" and "passes" in entry["changed"]
    assert any(st["tool"] == "human_declined" for st in s.mission.agent_log)


def test_counterfactual_without_opencv():
    from src.agentic.counterfactual import without_opencv
    s = _survey()
    cf = without_opencv(s, {"available": True, "pins": {}, "n": 0, "outside": 0, "median_shift_m": 0.0})
    assert cf["available"] and cf["active"]
    rows = {r["what"]: r for r in cf["rows"]}
    assert rows["Decision (agent action)"]["changed"] and cf["actions_changed"] == 4
    assert {"Priority (top 10 of the queue)", "Geolocation", "Re-survey route", "Human approval"} <= set(rows)


# ---- the API (real model, shipped samples) -------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from src.dashboard.app import app
    return TestClient(app)


def test_api_default_off_and_experimental_on(client):
    from src.dashboard import samples as samples_mod
    sid = samples_mod.list_samples()[0]["id"]
    off = client.post(f"/api/analyze?sample={sid}").json()
    assert off["active"]["mode"] in ("off", "unavailable", "validated")
    if off["active"]["mode"] == "unavailable":
        pytest.skip("no evidence model for this detector")
    on = client.post(f"/api/analyze?sample={sid}&active=1").json()
    assert on["active"]["mode"] in ("experimental", "validated")
    assert all(c["evidence"]["action"] for c in on["candidates"] if c["cls_name"] == "ghost_gear")
    assert [c["verdict"] for c in on["candidates"]] == [c["verdict"] for c in off["candidates"]]   # tiers unchanged


def test_api_declined_pass_replans(client):
    d = client.post("/api/survey?use_samples=1&gps=synthetic&boat_minutes=60&active=1").json()
    if not (d["mission"]["resurvey_plan"].get("lines")):
        pytest.skip("no pass planned on the samples")
    assert d["opencv_counterfactual"]["available"]
    key = next(k for k in d["dispatch"] if k.startswith("resurvey:"))
    rid = d["dispatch"][key]["id"]
    r = client.post(f"/api/approvals/{rid}/decide", json={"decision": "declined", "by": "Test Person"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["replan"]["decision"] == "decline_resurvey"
    declined = set(key.split(":", 1)[1].split(","))
    left = {t for L in j["mission"]["resurvey_plan"]["lines"] for t in L["targets"]}
    assert not (declined & left)
    assert set(j["mission"]["review_queue"][:len(declined)]) == declined
