"""
Tests for provenance stamps, the agent decision log (trace export) and protected-site redaction.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agentic.mission import export, redact_public
from src.agentic.types import (AgentStep, Candidate, FrameResult, MissionPlan, SurveyResult, TrackedObject, Verdict)


def _survey():
    wreck = TrackedObject("H1", "structural_fragment", Verdict.REVIEW, 0.4, 0.4, "F1", (0, 0, 9, 9),
                          lat=37.812345, lon=-76.151234, geo_error_m=5.0, p_pot=None)
    pot = TrackedObject("H2", "fishing_gear", Verdict.REVIEW, 0.5, 0.5, "F1", (20, 20, 29, 29),
                        lat=37.801111, lon=-76.149999, geo_error_m=4.0, p_pot=0.5)
    m = MissionPlan([], ["H1", "H2"], None, True, gps_synthetic=True)
    m.resurvey_plan = {"lines": [{"id": "RS1", "targets": ["H1"], "voi": 0.25, "boat_min": 2, "status": "PLANNED",
                                  "start": [37.81, -76.15], "end": [37.82, -76.15], "heading_deg": 0, "length_m": 50,
                                  "targets_on": "port", "voi_per_100m": 0.5, "predictions": [], "why": "x"},
                                 {"id": "RS2", "targets": ["H2"], "voi": 0.25, "boat_min": 2, "status": "PLANNED",
                                  "start": [37.80, -76.15], "end": [37.805, -76.15], "heading_deg": 0, "length_m": 50,
                                  "targets_on": "port", "voi_per_100m": 0.5, "predictions": [], "why": "y"}]}
    c = Candidate((0, 0, 9, 9), 2, "structural_fragment", 0.4, verdict=Verdict.REVIEW,
                  trace=[AgentStep("detect", "found", 1.0), AgentStep("decide", "review", 0.0)])
    fr = FrameResult("F1", 64, 64, [c], stage_ms={"see": 10.0}, stage1={"altitude_px": 12.0})
    return SurveyResult("s1", [fr], [wreck, pot], m)


PROV = {"model": "T", "model_onnx_sha256": "ab" * 32, "calibration_sha256": "cd" * 32, "code_commit": "abc123",
        "opencv": {"version": "5.0.0", "cv2_file": "/opt/cool/x/cv2/__init__.py", "is_cool_path": True},
        "host": {"machine": "aarch64", "vcpus": 4, "instance_type": "c8g.xlarge"}, "created_utc": "2026-01-01T00:00:00Z"}


def test_public_share_generalises_wrecks_and_drops_revealing_passes():
    s = _survey()
    tracked, mission, n = redact_public(s.tracked, s.mission)
    w = next(t for t in tracked if t.oid == "H1"); p = next(t for t in tracked if t.oid == "H2")
    assert n == 1 and (w.lat, w.lon) == (37.81, -76.15) and w.geo_error_m >= 1100
    assert (p.lat, p.lon) == (37.801111, -76.149999)                         # cleanup targets keep precision
    assert [L["id"] for L in mission.resurvey_plan["lines"]] == ["RS2"]      # the pass at the wreck is gone
    assert s.tracked[0].lat == 37.812345                                      # the original is untouched
    g = json.loads(export("geojson", s, public=True, prov=PROV))
    assert "generalised" in g["properties"]["redaction"] and g["properties"]["provenance"]["model"] == "T"


def test_trace_is_a_complete_decision_log_and_never_public():
    s = _survey()
    lines = [json.loads(l) for l in export("trace", s, prov=PROV).strip().splitlines()]
    kinds = [l["type"] for l in lines]
    assert kinds[0] == "provenance" and lines[0]["opencv"]["is_cool_path"] is True
    assert kinds.count("step") == 2 and kinds.count("verdict") == 1 and kinds[-1] == "mission"
    assert lines[-1]["human_approval_required"] is True
    try:
        export("trace", s, public=True, prov=PROV)
        raise AssertionError("public trace must be refused")
    except ValueError:
        pass


def test_every_format_carries_provenance():
    s = _survey()
    assert "abc123" in export("gpx", s, prov=PROV) and "abc123" in export("kml", s, prov=PROV)
    assert json.loads(export("json", s, prov=PROV))["provenance"]["code_commit"] == "abc123"
    assert export("csv", s, prov=PROV).startswith("id,")                   # CSV stays a clean table
