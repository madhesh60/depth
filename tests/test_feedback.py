"""
Tests for human-feedback labels (src/agentic/feedback.py + /api/feedback): validation, latest-wins
dedupe, upload persistence, YOLO export (positives / hard negatives) and reviewer agreement with GT.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agentic.feedback import FeedbackStore


def _rec(decision, bbox, ref, **kw):
    return {"decision": decision, "bbox": bbox, "frame_ref": ref, "frame_id": "f", "cls_name": "fishing_gear", **kw}


def test_validation(tmp_path):
    st = FeedbackStore(tmp_path)
    for bad in (_rec("maybe", [0, 0, 10, 10], {"sample": "s"}), _rec("confirm", [0, 0, 1, 1], {"sample": "s"}),
                _rec("confirm", [0, 0, 10, 10], {})):
        try:
            st.add(bad)
            raise AssertionError("should have raised")
        except ValueError:
            pass


def test_latest_decision_wins_and_export(tmp_path):
    st = FeedbackStore(tmp_path / "fb")
    img = np.full((100, 200, 3), 90, np.uint8)
    ref = {"upload": "abc123"}
    st.add(_rec("reject", [10, 10, 30, 30], ref), img)
    st.add(_rec("confirm", [11, 10, 31, 30], ref), img)            # same object, changed mind → confirm
    st.add(_rec("reject", [100, 50, 120, 70], ref), img)           # a hard negative
    st.add(_rec("missed", [150, 20, 170, 44], ref), img)           # a pot the detector never proposed
    s = st.stats()
    assert s["decisions"] == 4 and s["objects"] == 3 and s["confirm"] == 1 and s["reject"] == 1 and s["missed"] == 1
    assert (tmp_path / "fb" / "frames" / "abc123.jpg").exists()      # upload persisted with the label
    out = st.export(tmp_path / "ds", ["fishing_gear", "pipe"])
    assert out["frames"] == 1 and out["positives"] == 2 and out["hard_negatives"] == 1
    lines = (tmp_path / "ds" / "labels" / "u_abc123.txt").read_text().splitlines()
    assert len(lines) == 2 and all(l.startswith("0 ") for l in lines)
    hn = json.loads((tmp_path / "ds" / "hard_negatives.json").read_text())
    assert hn[0]["bbox"] == [100, 50, 120, 70]


def test_agreement_with_shipped_ground_truth(tmp_path):
    from src.dashboard import samples as samples_mod
    gts = samples_mod.sample_gt("sample-02")
    if not gts:
        return                                                        # samples not shipped in this checkout
    st = FeedbackStore(tmp_path)
    g = list(gts[0])
    st.add(_rec("confirm", g, {"sample": "sample-02"}))                # agrees with GT
    st.add(_rec("confirm", [300, 300, 320, 320], {"sample": "sample-02"}))   # not a labelled pot
    st.add(_rec("reject", list(gts[1]), {"sample": "sample-02"}))      # dismissed a real pot
    a = st.agreement()
    assert a["confirmed_real"] == 1 and a["confirmed_not_in_gt"] == 1 and a["dismissed_real"] == 1
    assert a["reviewer_precision"] == 0.5


def test_api_feedback_roundtrip(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import src.dashboard.app as app_mod
    monkeypatch.setattr(app_mod, "_FEEDBACK", FeedbackStore(tmp_path))
    c = TestClient(app_mod.app)
    r = c.post("/api/feedback", json={"decision": "missed", "bbox": [5, 5, 25, 25], "frame_id": "Rec6_wcp_ss_port_00005",
                                       "cls_name": "fishing_gear"})
    if r.status_code == 400 and "frame_ref" in r.text:
        return                                                        # samples not shipped in this checkout
    assert r.status_code == 200 and r.json()["saved"]["frame_ref"] == {"sample": "sample-02"}
    assert c.get("/api/feedback/stats").json()["missed"] == 1
    assert c.post("/api/feedback", json={"decision": "nope", "bbox": [0, 0, 9, 9], "frame_ref": {"sample": "x"}}).status_code == 400
