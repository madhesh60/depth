"""
Tests for the Decide-stage agent (src/agentic/agent.py).

The perceptor is stubbed (preset detections + re-look confidences) so the decision procedure,
trace, and triage are tested deterministically without the ONNX model. Runs under pytest, or
standalone: python tests/test_agent.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.infer import Detection
from src.agentic.agent import ReLookAgent, AgentConfig, render
from src.agentic.shadow import ShadowProver, ShadowConfig
from src.agentic.types import RelookResult, Verdict


class _StubPerceptor:
    def __init__(self, dets, relook_normal, relook_enhanced=None):
        self.dets = dets
        self.rn = relook_normal
        self.re = relook_enhanced if relook_enhanced is not None else relook_normal

    def perceive(self, frame):
        return list(self.dets)

    def zoom_relook(self, frame, bbox, cls_name="fishing_gear", enhance=False):
        c = self.re if enhance else self.rn
        return RelookResult(found=c > 0, conf=c, gain=0.0, scale=2.0)


def _agent(dets, relook_normal, relook_enhanced=None, cfg=None):
    return ReLookAgent(
        perceptor=_StubPerceptor(dets, relook_normal, relook_enhanced),
        shadow_prover=ShadowProver(ShadowConfig(nadir="top")),
        cfg=cfg or AgentConfig(),
    )


def _frame():
    return (90 + np.random.default_rng(0).normal(0, 4, (200, 200, 3))).clip(0, 255).astype(np.uint8)


def _det(conf, box=(90, 60, 110, 84)):
    return Detection(bbox=box, cls_id=0, cls_name="fishing_gear", conf=conf)


def test_strong_relook_is_confirmed_with_trace():
    res = _agent([_det(0.30)], relook_normal=0.55).run_frame(_frame())
    c = res.candidates[0]
    assert c.verdict is Verdict.CONFIRMED
    tools = [s.tool for s in c.trace]
    # adaptive controller: re-look is the agent's first probe, decide is last
    assert tools[0] == "zoom_relook" and "shadow_check" in tools and tools[-1] == "decide"
    # strong first look ⇒ no enhanced re-look needed
    assert "enhance_relook" not in tools
    assert c.trace[-1].detail["confirm_path"] == "relook"


def test_high_confidence_path_confirms_and_skips_escalation():
    """A detection the detector is already sure about (conf ≥ 0.60, precision ~0.83 — STUDY-04) is
    auto-confirmed even if it does not re-fire, and the agent skips the expensive enhanced re-look."""
    res = _agent([_det(0.68)], relook_normal=0.0, relook_enhanced=0.9).run_frame(_frame())
    c = res.candidates[0]
    assert c.verdict is Verdict.CONFIRMED
    tools = [s.tool for s in c.trace]
    assert "enhance_relook" not in tools                       # early-stop: no wasted second inference
    assert c.trace[-1].detail["confirm_path"] == "high_confidence"
    assert c.trace[-1].detail["escalated"] is False


def test_enhanced_relook_can_upgrade_decision():
    """A low-confidence candidate that only re-fires after the CLAHE 'try harder' pass gets confirmed,
    and the trace records the enhanced re-look that changed the outcome."""
    res = _agent([_det(0.18)], relook_normal=0.30, relook_enhanced=0.52).run_frame(_frame())
    c = res.candidates[0]
    assert c.verdict is Verdict.CONFIRMED
    assert "enhance_relook" in [s.tool for s in c.trace]
    assert c.evidence.relook.conf == 0.52


def test_review_and_rejected_tiers():
    review = _agent([_det(0.20)], relook_normal=0.22, relook_enhanced=0.24).run_frame(_frame())
    assert review.candidates[0].verdict is Verdict.REVIEW
    rejected = _agent([_det(0.11)], relook_normal=0.0).run_frame(_frame())
    assert rejected.candidates[0].verdict is Verdict.REJECTED


def test_frame_result_counts_and_serialisation():
    dets = [_det(0.30, (90, 60, 110, 84)), _det(0.11, (10, 10, 26, 30))]
    res = _agent(dets, relook_normal=0.0, relook_enhanced=0.0)
    # first det: conf 0.30 no relook -> REVIEW ; second: conf 0.11 no relook -> REJECTED
    r = res.run_frame(_frame())
    assert r.counts["review"] == 1 and r.counts["rejected"] == 1
    d = r.to_dict()
    assert d["candidates"][0]["trace"] and "counts" in d and d["nadir"] == "top"


def test_render_returns_image():
    r = _agent([_det(0.55)], relook_normal=0.55).run_frame(_frame())
    vis = render(_frame(), r)
    assert vis.shape == (200, 200, 3) and vis.dtype == np.uint8


def test_toolbox_estimate_height_and_cross_pass():
    """The two new tools are pure/deterministic — test them directly."""
    from src.agentic.tools import Toolbox
    from src.agentic.types import ShadowProof, ShadowQuality
    tb = Toolbox(_StubPerceptor([], 0.0), ShadowProver(ShadowConfig(nadir="top")))

    proof = ShadowProof(ShadowQuality.CLEAR, 0.4, 8, 0.8, 0.7, 0.2, 3.1, (10, 10), (5, 12, 15, 20))
    h, step = tb.estimate_height(proof)
    assert h == 0.7 and step.tool == "estimate_height" and "0.7" in step.rationale

    # two overlapping passes (same recording+channel, adjacent ping, same across-track position)
    others = [
        {"frame_id": "crabpot_Rec6_ss_port_00010", "cls": "fishing_gear", "cx": 0.5, "cy": 0.4},
        {"frame_id": "crabpot_Rec6_ss_port_00011", "cls": "fishing_gear", "cx": 0.51, "cy": 0.41},
    ]
    matched, mframe, step = tb.match_other_pass("crabpot_Rec6_ss_port_00010", 0.5, 0.4, "fishing_gear", others)
    assert matched and mframe == "crabpot_Rec6_ss_port_00011" and step.detail["matched"] is True
    # a different recording does not corroborate
    solo, _, _ = tb.match_other_pass("crabpot_Rec9_ss_star_00099", 0.5, 0.4, "fishing_gear", others)
    assert solo is False


def test_cross_pass_corroboration_is_noted_never_gated():
    """Survey-level corroboration adds a match_other_pass step + note + score boost, and NEVER
    changes a verdict (the CONFIRMED tier stays exactly the calibrated set)."""
    from src.agentic.pipeline import AgenticPipeline
    agent = _agent([_det(0.30, (90, 60, 110, 84))], relook_normal=0.0)   # conf 0.30, no re-fire -> REVIEW
    pipe = AgenticPipeline(agent=agent)
    frames = [("crabpot_Rec6_ss_port_00010", _frame()), ("crabpot_Rec6_ss_port_00011", _frame())]
    survey = pipe.run_survey(frames, track=None, survey_id="t")
    for fr in survey.frames:
        c = fr.candidates[0]
        assert c.verdict is Verdict.REVIEW                                # unchanged by corroboration
        steps = [s.tool for s in c.trace]
        assert "match_other_pass" in steps and steps.index("decide") < steps.index("match_other_pass")
        assert any("cross-pass" in n for n in c.evidence.notes)
    assert survey.mission.human_approval_required is True


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} agent tests passed")


if __name__ == "__main__":
    _run_all()
