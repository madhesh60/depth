"""
STUDY-20 plumbing: the engine that loaded is recorded, and the parity matcher counts what it should.
The study itself (timings) is a benchmark, not a test; these check its pure parts and the provenance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agentic.policy import GuaranteedTiers
from src.bench.dnn_engine import compare_answers, separation
from src.detection.calibration import load_calibration
from src.detection.infer import DEFAULT_ONNX, ENGINE, Detection, YoloOnnxDetector, faster_engine, runtime_info

_TIERS = GuaranteedTiers.from_calibration(load_calibration())
_G = _TIERS.guaranteed_class


def _d(conf: float, box=(10, 10, 50, 50), cls: str = _G) -> Detection:
    return Detection(tuple(box), 0, cls, conf)


def test_identical_answers_match():
    a = [_d(0.40), _d(0.20, (60, 60, 90, 90))]
    c, bad = compare_answers(a, [_d(0.40), _d(0.20, (60, 60, 90, 90))], _TIERS)
    assert c["match"] == 2 and c["differ"] == c["missing"] == c["extra"] == c["tier_flips"] == 0 and not bad


def test_a_score_gap_and_a_tier_flip_are_counted():
    t = _TIERS.tau_review
    c, bad = compare_answers([_d(0.50), _d(t + 0.002, (60, 60, 90, 90))],
                             [_d(0.52), _d(t - 0.002, (60, 60, 90, 90))], _TIERS)
    assert c["differ"] == 2 and c["tier_flips"] == 1 and c["match"] == 0
    assert sum(b.get("tier_flip", False) for b in bad) == 1


def test_missing_and_extra_detections():
    c, bad = compare_answers([_d(0.4)], [_d(0.4, (200, 200, 240, 240))], _TIERS)
    assert c["missing"] == 1 and c["extra"] == 1 and {b["kind"] for b in bad} == {"missing", "extra"}


def test_a_class_change_is_not_a_match():
    c, _ = compare_answers([_d(0.4)], [_d(0.4, cls="wreck_debris")], _TIERS)
    assert c["missing"] == 1 and c["extra"] == 1


def test_speedup_is_claimed_only_when_rounds_separate():
    assert separation([10, 11, 12], [7, 8, 9]) == "round ranges separate"
    assert separation([10, 11, 12], [9, 10.5, 11]).startswith("round ranges overlap")
    assert separation([7, 8], [10, 11]) == "round ranges separate (slower)"
    assert separation([10], [7]).startswith("single round")


def test_runtime_info_names_the_requested_and_the_loaded_engine():
    r = runtime_info()
    assert {"threads", "engine", "engine_requested", "engine_fallback", "engine_choice"} <= set(r)


def test_fastest_switches_engine_only_when_faster_in_every_round_with_the_same_answers():
    new, classic = [377.0, 376.4, 376.7], [255.4, 255.5, 255.3]          # the c8g.large rounds (STUDY-20)
    assert faster_engine({"new": new, "classic": classic}, True) == ("classic", "classic faster in every round, same answers")
    assert faster_engine({"new": new, "classic": [255.4, 380.0, 255.3]}, True)[0] == "new"   # one round overlaps
    assert faster_engine({"new": new, "classic": classic}, False)[0] == "new"                # the answers differ
    assert faster_engine({"new": classic, "classic": new}, True)[0] == "new"


@pytest.mark.skipif(not DEFAULT_ONNX.exists(), reason="model weights absent (python -m src.detection.fetch_model)")
def test_fastest_measures_both_engines_on_this_machine_and_says_why(monkeypatch):
    monkeypatch.setenv("DEPTH_DNN_ENGINE", "fastest")
    det = YoloOnnxDetector(DEFAULT_ONNX)
    ch = runtime_info()["engine_choice"]
    assert det.engine == ch["picked"] in ("new", "classic") and runtime_info()["engine_requested"] == "fastest"
    assert ch["same_answers"] and ch["max_score_gap"] <= 0.01 and ch["max_box_gap_px"] <= 1.0
    assert len(ch["round_ms"]["new"]) == len(ch["round_ms"]["classic"]) == 3 and ch["why"]
    monkeypatch.setenv("DEPTH_DNN_ENGINE", "auto")
    YoloOnnxDetector(DEFAULT_ONNX)
    assert runtime_info()["engine_choice"] is None                         # only `fastest` measures


@pytest.mark.skipif(not DEFAULT_ONNX.exists(), reason="model weights absent (python -m src.detection.fetch_model)")
def test_the_loaded_engine_is_recorded(monkeypatch):
    monkeypatch.setenv("DEPTH_DNN_ENGINE", "classic")
    assert YoloOnnxDetector(DEFAULT_ONNX).engine == "classic" and ENGINE["requested"] == "classic"
    monkeypatch.setenv("DEPTH_DNN_ENGINE", "auto")
    det = YoloOnnxDetector(DEFAULT_ONNX)
    assert det.engine == "new" and runtime_info()["engine"] == "new" and runtime_info()["engine_requested"] == "auto"
