"""
Tests for the person-confirmed loop (src/agentic/mission.py apply_human / plan_mission): a person's
decision re-plans routes, the queue, the budget and the re-survey passes, never overwrites the agent's
verdict, and feeds the impact ledger, the exports, the brief and the API.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from src.agentic import brief
from src.agentic.geo import synthetic_track
from src.agentic.mission import apply_human, export, plan_mission
from src.agentic.types import SurveyResult, TrackedObject, Verdict
from src.dashboard import app as app_mod
from src.detection.infer import DEFAULT_ONNX

FIDS = [f"Rec6_wcp_ss_port_{i:05d}" for i in range(4, 8)]


def _survey() -> SurveyResult:
    track = synthetic_track(FIDS)
    tr = [TrackedObject(f"H{i:03d}", "fishing_gear", Verdict.REVIEW, 0.5, 0.5, FIDS[i % 4], (i, 0, i + 9, 9),
                        lat=track[FIDS[i % 4]].lat + i * 1e-5, lon=track[FIDS[i % 4]].lon, geo_error_m=4.0,
                        p_pot=round(0.9 - i * 0.07, 2)) for i in range(1, 9)]
    g = {"recall_promise": 0.65, "precision_promise": None, "tau_confirm": None}
    args = {"budget_minutes": 0.5, "sec_per_card": 8.0, "boat_minutes": None}
    m = plan_mission(tr, track, g, args["budget_minutes"], args["sec_per_card"], None)
    return SurveyResult("s-h", [], tr, m, track=track, plan_args=args)


def test_confirm_puts_it_on_the_route_and_replans():
    s = _survey()
    assert s.mission.recovery_route == [] and len(s.mission.review_queue) == 8
    passes_before = sum(len(L["targets"]) for L in s.mission.resurvey_plan["lines"])
    apply_human(s, "H001", "confirm", "A. Person")
    apply_human(s, "H002", "confirm", "A. Person")
    apply_human(s, "H003", "reject", "A. Person")
    m = s.mission
    assert set(m.recovery_route) == {"H001", "H002"} and m.route_length_m and m.route_length_m > 0
    assert len(m.review_queue) == 5 and not {"H001", "H002", "H003"} & set(m.review_queue)
    assert not {"H001", "H002", "H003"} & set(m.budget["review_ids"])            # budget re-spent on open cards
    targets = {t for L in m.resurvey_plan["lines"] for t in L["targets"]}
    assert not {"H001", "H002", "H003"} & targets and len(targets) == passes_before - 3
    t1 = next(t for t in s.tracked if t.oid == "H001")
    assert t1.verdict is Verdict.REVIEW and t1.human == "confirmed" and t1.human_by == "A. Person"   # agent verdict kept


def test_field_outcomes_rules_and_undo():
    s = _survey()
    with pytest.raises(ValueError):
        apply_human(s, "H004", "recovered", "Crew")                    # not on the route yet
    apply_human(s, "H004", "confirm", "A. Person")
    with pytest.raises(ValueError):
        apply_human(s, "H004", "confirm", "B. Person")                 # already decided
    with pytest.raises(ValueError):
        apply_human(s, "H005", "confirm", "")                          # a named person is required
    with pytest.raises(ValueError):
        apply_human(s, "H005", "launch", "A. Person")
    with pytest.raises(KeyError):
        apply_human(s, "H999", "confirm", "A. Person")
    apply_human(s, "H004", "recovered", "Crew")
    assert "H004" not in s.mission.recovery_route
    apply_human(s, "H004", "undo", "A. Person")
    assert s.mission.review_queue[0] == "H004" or "H004" in s.mission.review_queue
    assert [h["decision"] for h in s.mission.human_log] == ["confirm", "recovered", "undo"]


def test_impact_ledger_measures_reviewed_precision():
    s = _survey()
    for oid in ("H001", "H002", "H003"):
        apply_human(s, oid, "confirm", "A")
    apply_human(s, "H004", "reject", "A")
    apply_human(s, "H001", "recovered", "Crew")
    apply_human(s, "H002", "not_found", "Crew")
    imp = s.mission.impact
    assert imp["person_confirmed"] == 1 and imp["person_recovered"] == 1 and imp["person_not_found"] == 1
    assert imp["reviewed_precision"] == 0.5                              # (1 confirmed + 1 recovered) / 4 judged
    lo, hi = imp["reviewed_precision_ci95"]
    assert lo < 0.5 < hi and imp["on_recovery_route"] == 1 and imp["decisions"] == 6


def test_exports_and_brief_carry_the_decisions():
    s = _survey()
    apply_human(s, "H001", "confirm", "A. Person")
    apply_human(s, "H002", "confirm", "A. Person")
    assert '"person_decision": "confirmed"' in export("geojson", s)
    trace = export("trace", s)
    assert '"type": "human"' in trace and '"by": "A. Person"' in trace
    F = brief.facts(s.to_dict(), {"model": "T", "opencv": {"version": "5.0.0"}})
    text = brief.template_brief(F)
    assert "People have decided 2 times" in text and "2 stops" in text
    assert brief.check(text, F)["ok"]                                    # still every number traced


def test_api_decide_pin_and_replan(monkeypatch):
    if not DEFAULT_ONNX.exists():
        pytest.skip("model absent")
    c = TestClient(app_mod.app)
    s = c.post("/api/survey?use_samples=1&gps=synthetic&budget_minutes=1").json()
    sid, q = s["survey_id"], s["mission"]["review_queue"]
    assert s["mission"]["recovery_route"] == []
    monkeypatch.setenv("DEPTH_APPROVER_PIN", "2468")
    assert c.post(f"/api/survey/{sid}/decide", json={"hazard_id": q[0], "decision": "confirm", "by": "A"}).status_code == 403
    r = c.post(f"/api/survey/{sid}/decide", json={"hazard_id": q[0], "decision": "confirm", "by": "A", "pin": "2468"})
    assert r.status_code == 200 and r.json()["mission"]["recovery_route"] == [q[0]]
    assert c.post(f"/api/survey/{sid}/decide", json={"hazard_id": q[0], "decision": "confirm", "by": "A", "pin": "2468"}).status_code == 409
    assert c.post("/api/survey/nope/decide", json={"hazard_id": "H1", "decision": "confirm", "by": "A", "pin": "2468"}).status_code == 409
    feat = c.get(f"/ogc/collections/hazards/items/{sid}:{q[0]}").json()
    assert feat["properties"]["person_decision"] == "confirmed" and feat["properties"]["on_recovery_route"] is True
