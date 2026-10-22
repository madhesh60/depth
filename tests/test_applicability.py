"""
Tests for the promise applicability check (src/agentic/applicability.py, STUDY-19) and for what the
core loop does with it: a recording that does not look like the calibration recordings loses its recall
promise, runs the cheapest look, files an incident and asks a person for a full review.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.agentic.applicability import FEATURES, Reference, frame_stats
from src.agentic.core import Goal, SurveyAgent

from test_core import _FakePipe, _frames  # noqa: F401  (the detector stub shared with the core tests)

SAMPLES = Path(__file__).resolve().parents[1] / "webui" / "samples"


def _grey(mean: float, seed: int, n: int = 1) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        g = np.clip(rng.normal(mean, 30, (200, 200)), 0, 255).astype(np.uint8)
        out.append(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
    return out


def _orange(n: int, seed: int = 9) -> list[np.ndarray]:
    out = []
    for im in _grey(110, seed, n):
        im = im.copy()
        im[..., 0] //= 4                                   # blue down -> an orange palette
        out.append(im)
    return out


def _ref() -> Reference:
    return Reference.fit([frame_stats(im) for im in _grey(105, 1, 60)], share_limit=0.2)


def not_sonar() -> np.ndarray:
    """A picture that is plainly not a sonogram: coloured shapes and a word (the judge-audit upload)."""
    rng = np.random.default_rng(0)
    im = np.full((480, 640, 3), (200, 150, 90), np.uint8)
    for _ in range(30):
        colour = tuple(int(x) for x in rng.integers(0, 255, 3))
        cv2.circle(im, (int(rng.integers(0, 640)), int(rng.integers(0, 480))), int(rng.integers(10, 60)), colour, -1)
    cv2.putText(im, "HELLO", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
    return im


def test_frame_stats_measure_the_palette_and_texture():
    g = frame_stats(_grey(100, 2)[0])
    o = frame_stats(_orange(1)[0])
    assert set(FEATURES) - {"hist_dist"} <= set(g) and g["hist"].shape == (32,)
    assert g["saturation"] < 0.01 < o["saturation"]
    assert 0 < g["texture"] and 0.3 < g["mean"] < 0.5


def test_similar_frames_keep_the_promise_and_another_palette_loses_it():
    ref = _ref()
    same = ref.check([frame_stats(im) for im in _grey(105, 3, 20)])
    assert same["applies"] and same["share_lcb"] <= 0.2
    other = ref.check([frame_stats(im) for im in _orange(20)])
    assert not other["applies"] and other["atypical"] == 20 and "does NOT apply" in other["why"]
    assert "saturation" in other["drivers"]


def test_one_odd_frame_never_withdraws_the_promise():
    ref = _ref()
    one = ref.check([frame_stats(im) for im in _orange(1)])
    assert one["atypical"] == 1 and one["applies"]                 # a 1-frame lower bound stays below the limit


def test_the_shipped_reference_accepts_the_demo_frames():
    ref = Reference.load()
    assert ref is not None and ref.share_limit == 0.2 and ref.data["n_cal"] == 163
    stats = [frame_stats(cv2.imread(str(p))) for p in sorted(SAMPLES.glob("*.jpg"))]
    assert len(stats) == 8 and ref.check(stats)["applies"]


def test_one_frame_unlike_every_calibration_frame_loses_the_promise():
    ref = _ref()
    same = ref.check_frame(frame_stats(_grey(105, 5)[0]))
    assert same["applies"] and not same["caution"] and "applies" in same["why"]
    odd = ref.check_frame(frame_stats(_orange(1)[0]))
    assert not odd["applies"] and not odd["caution"] and odd["p_value"] == odd["p_floor"] == round(1 / 61, 4)
    assert "does NOT apply" in odd["why"] and odd["n_cal"] == 60


def test_the_shipped_reference_keeps_every_demo_frame_alone_and_drops_a_non_sonar_picture():
    ref = Reference.load()
    checks = {p.name: ref.check_frame(frame_stats(cv2.imread(str(p)))) for p in sorted(SAMPLES.glob("*.jpg"))}
    assert len(checks) == 8 and all(c["applies"] for c in checks.values())
    # Rec9 port #13 is the demo frame the survey check counts as atypical (p <= 0.05): alone it keeps the
    # promise, with a caution - one frame cannot decide
    assert [n for n, c in checks.items() if c["caution"]] == ["Rec9_wcp_ss_port_00013.jpg"]
    chk = ref.check_frame(frame_stats(not_sonar()))
    assert not chk["applies"] and chk["p_value"] == chk["p_floor"]


def test_the_agent_withdraws_its_promise_and_asks_for_a_full_review():
    ref = _ref()
    pipe = _FakePipe()
    s = SurveyAgent(pipe, reference=ref).run([(f"o{i}", im) for i, im in enumerate(_orange(6))], goal=Goal())
    run = s.agent_run
    acts = [st["act"] for st in run["steps"]]
    assert acts[:3] == ["inspect_data", "check_promise", "choose_mode"]
    assert run["promise_applies"] is False and pipe.modes_run == ["standard"] * 6   # no promise left to buy
    g = s.mission.guarantees
    assert g["recall_promise"] is None and g["promise_withdrawn"]["was"] > 0
    assert any(i["code"] == "promise_not_applicable" for i in s.mission.incidents)
    assert "WITHOUT a recall promise" in run["stop"]


def test_a_matching_recording_keeps_the_promise():
    ref = _ref()
    s = SurveyAgent(_FakePipe(), reference=ref).run([(f"g{i}", im) for i, im in enumerate(_grey(105, 4, 5))],
                                                    goal=Goal())
    assert s.agent_run["promise_applies"] is True and s.mission.guarantees["recall_promise"]
    assert not any(i["code"] == "promise_not_applicable" for i in s.mission.incidents)
