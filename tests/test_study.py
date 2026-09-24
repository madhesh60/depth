"""
Tests for the timed user study (src/agentic/study.py) and the analyst-effort model
(src/agentic/effort.py): Latin-square counterbalancing, scoring against ground truth, pooled
summary, and the effort curves / break-even / promise point on a tiny synthetic split.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agentic import study, effort


def _fake_frames(monkeypatch):
    frames = [{"frame_id": f"F{i}", "sample_id": None, "path": "x", "gt": [(10 * i, 10, 10 * i + 8, 18)]} for i in range(4)]
    monkeypatch.setattr(study, "study_frames", lambda: frames)
    return frames


def test_latin_square_rotates_halves_and_order(tmp_path, monkeypatch):
    monkeypatch.setenv("DEPTH_STUDY_RESULTS", str(tmp_path))
    _fake_frames(monkeypatch)
    seen = []
    for k in range(4):
        p = study.plan("p")
        seen.append((tuple(f["frame_id"] for f in p["manual_frames"]), tuple(p["order"])))
        study.save({"session_id": f"s{k}", "participant": "p", "index": k, "order": p["order"],
                    "manual": {"found": 0, "pots": 0, "false_clicks": 0, "frames": 0, "sec_per_frame": None},
                    "cards": {"confirmed_real": 0, "dismissed_real": 0, "pots": 0, "cards": 0, "frames": 0, "sec_per_card": None}})
    halves = {h for h, _ in seen}; orders = {o for _, o in seen}
    assert len(halves) == 2 and len(orders) == 2 and len(set(seen)) == 4      # all four cells of the 2x2


def test_scoring_against_ground_truth(monkeypatch):
    frames = _fake_frames(monkeypatch)
    p = {"session_id": "s", "participant": "p", "index": 0, "order": ["manual", "cards"],
         "manual_frames": [{"frame_id": "F0"}, {"frame_id": "F1"}], "card_frames": [{"frame_id": "F2"}, {"frame_id": "F3"}]}
    cards = {"a": {"frame_id": "F2", "bbox": [20, 10, 28, 18]},             # real (matches F2's pot)
             "b": {"frame_id": "F2", "bbox": [100, 100, 110, 110]},        # not a pot
             "c": {"frame_id": "F3", "bbox": [30, 10, 38, 18]}}            # real
    result = {"manual": [{"frame_id": "F0", "ms": 10000, "clicks": [[4, 14], [200, 200]]},   # 1 hit + 1 false mark
                         {"frame_id": "F1", "ms": 20000, "clicks": []}],                     # missed
              "cards": [{"card_id": "a", "ms": 2000, "decision": "real"}, {"card_id": "b", "ms": 3000, "decision": "real"},
                        {"card_id": "c", "ms": 4000, "decision": "not"}]}
    r = study.score(p, result, cards)
    assert r["manual"]["found"] == 1 and r["manual"]["pots"] == 2 and r["manual"]["false_clicks"] == 1
    assert r["manual"]["sec_per_frame"] == 15.0
    c = r["cards"]
    assert (c["confirmed_real"], c["confirmed_fake"], c["dismissed_real"]) == (1, 1, 1)
    assert c["accuracy_on_real"] == 0.5 and c["recall"] == 0.5 and c["sec_per_card"] == 3.0


def test_card_recall_counts_frames_without_cards(monkeypatch):
    _fake_frames(monkeypatch)
    p = {"session_id": "s", "participant": "p", "index": 0, "order": ["cards", "manual"], "manual_frames": [],
         "card_frames": [{"frame_id": "F2"}, {"frame_id": "F3"}]}
    r = study.score(p, {"cards": [{"card_id": "a", "ms": 1000, "decision": "real"}]},
                    {"a": {"frame_id": "F2", "bbox": [20, 10, 28, 18]}})      # F3 produced no card
    assert r["cards"]["pots"] == 2 and r["cards"]["recall"] == 0.5


def test_effort_curves_breakeven_and_forecast():
    E = {"frames": 4, "pots": 4, "pots_per_frame": [1, 1, 1, 1], "recall_promise": 0.5,
         "detector_list": [[0, 0], [0, 1], [1, 1], [2, 0], [3, 1]],
         "depth_queue": [[1, 0.9], [1, 0.8], [0, 0.3], [1, 0.2]], "frames_per_survey_hour": 168.75}
    C = effort.curves(E, sec_per_frame=30, sec_per_card=6)
    assert C["minutes_to_promise"]["manual"] == 1.0            # 2 frames × 30 s
    assert C["minutes_to_promise"]["depth"] == 0.2             # 2 cards × 6 s
    assert C["cards_to_promise"] == 2 and C["breakeven_sec_per_card"] == 30.0   # 2 frames × 30 s / 2 cards
    assert abs(C["forecast_at_promise"] - 1.7) < 1e-6 and C["actual_at_promise"] == 2.0
    # card accuracy below what the promise needs → DEPTH never reaches it (honest None)
    assert effort.curves(E, 30, 6, card_accuracy=0.4)["minutes_to_promise"]["depth"] is None
