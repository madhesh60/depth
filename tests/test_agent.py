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
    # first det: conf 0.30 no relook -> REVIEW ; second: conf 0.11 no relook -> LOW_RISK
    r = res.run_frame(_frame())
    assert r.counts["review"] == 1 and r.counts["low_risk"] == 1
    d = r.to_dict()
    assert d["candidates"][0]["trace"] and "counts" in d and d["nadir"] == "top"


def test_render_returns_image():
    r = _agent([_det(0.55)], relook_normal=0.55).run_frame(_frame())
    vis = render(_frame(), r)
    assert vis.shape == (200, 200, 3) and vis.dtype == np.uint8


def test_toolbox_estimate_height_is_relative():
    """Height is reported relative to sonar altitude; no metres without a measured altitude."""
    from src.agentic.tools import Toolbox
    from src.agentic.types import ShadowProof, ShadowQuality
    tb = Toolbox(_StubPerceptor([], 0.0), ShadowProver(ShadowConfig(nadir="top")))
    proof = ShadowProof(ShadowQuality.CLEAR, 0.4, 8, 0.8, None, 0.2, 3.1, (10, 10), (5, 12, 15, 20))
    h, step = tb.estimate_height(proof)
    assert h == 0.2 and step.tool == "estimate_height" and "20% of sonar altitude" in step.rationale


def test_orientation_comes_from_source_rule_not_a_guess():
    agent = ReLookAgent(perceptor=_StubPerceptor([_det(0.3)], 0.0), shadow_prover=ShadowProver())
    r = agent.run_frame(_frame(), frame_id="Rec9_wcp_ss_port_00031")
    assert r.nadir == "top" and r.orientation["source"] == "source-rule"
    u = agent.run_frame(_frame(), frame_id="BC_POST_T2_00_00_1_7")
    assert u.nadir == "unknown"
    assert u.candidates[0].evidence.shadow.orientation_known is False


def _boundary_survey(conf_a=0.30, conf_b=0.30):
    """Two adjacent chunks with one object cut by the boundary (same range)."""
    from src.agentic.pipeline import AgenticPipeline

    class _Seq(_StubPerceptor):
        def __init__(self):
            super().__init__([], 0.0)
            self.calls = 0

        def perceive(self, frame):
            self.calls += 1
            if self.calls == 1:     # chunk 10: box touches the trailing (right) edge
                return [Detection((186, 60, 200, 84), 0, "fishing_gear", conf_a)]
            return [Detection((0, 62, 12, 86), 0, "fishing_gear", conf_b)]   # chunk 11: leading edge

    agent = ReLookAgent(perceptor=_Seq(), shadow_prover=ShadowProver(), cfg=AgentConfig())
    frames = [("Rec6_wcp_ss_port_00010", _frame()), ("Rec6_wcp_ss_port_00011", _frame())]
    return AgenticPipeline(agent=agent).run_survey(frames, track=None, survey_id="t")


def test_chunk_boundary_stitching_counts_one_hazard_and_never_changes_verdicts():
    survey = _boundary_survey()
    a, b = survey.frames[0].candidates[0], survey.frames[1].candidates[0]
    assert a.continues_in == "Rec6_wcp_ss_port_00011" and b.continuation_of == "Rec6_wcp_ss_port_00010"
    assert a.verdict is Verdict.REVIEW and b.verdict is Verdict.REVIEW       # verdicts untouched
    assert any(s.tool == "stitch_boundary" for s in a.trace)
    assert len(survey.tracked) == 1 and survey.tracked[0].also_in           # counted once
    assert not any("cross-pass" in n for n in a.evidence.notes)             # old claim is gone


def test_stitching_keeps_the_stronger_sighting():
    survey = _boundary_survey(conf_a=0.11, conf_b=0.70)     # earlier half weak, later half confident
    assert len(survey.tracked) == 1
    t = survey.tracked[0]
    assert t.frame_id == "Rec6_wcp_ss_port_00011" and t.verdict is Verdict.CONFIRMED


# ---- calibrated (guaranteed-tier) mode: the value-of-information agent -------------------------
class _BatchStub(_StubPerceptor):
    """Stub with the batched re-look API; counts how many candidates were re-looked."""
    def __init__(self, dets, relook_by_conf):
        super().__init__(dets, 0.0)
        self.relook_by_conf = relook_by_conf
        self.relooked = []

    def relook_batch(self, frame, items, enhance=False):
        out = []
        for bbox, cls in items:
            d = next(d for d in self.dets if d.bbox == bbox)
            self.relooked.append(d.conf)
            c = self.relook_by_conf.get(d.conf, 0.0)
            out.append(RelookResult(found=c > 0, conf=c, gain=0.0, scale=1.0))
        return out


def _tiers(**kw):
    from src.agentic.policy import GuaranteedTiers
    base = dict(tau_review=0.10, tau_confirm=0.50, relook_mode="mosaic", escalate_clahe=False,
                p_pot_bins=[{"lo": 0.1, "hi": 0.3, "p_pot": 0.3, "n": 10}, {"lo": 0.3, "hi": 1.0, "p_pot": 0.6, "n": 10}],
                guarantees={"recall_promise": 0.6, "precision_promise": 0.85})
    base.update(kw)
    return GuaranteedTiers(**base)


def test_value_of_information_relooks_only_the_uncertain_band():
    dets = [_det(0.70, (10, 10, 30, 30)), _det(0.30, (60, 60, 80, 80)), _det(0.06, (120, 120, 140, 140))]
    stub = _BatchStub(dets, {0.30: 0.62})
    agent = ReLookAgent(perceptor=stub, shadow_prover=ShadowProver(ShadowConfig(nadir="top")),
                        cfg=AgentConfig(tiers=_tiers()))
    r = agent.run_frame(_frame(), frame_id="Rec9_wcp_ss_port_00001")
    v = [c.verdict for c in r.candidates]
    assert v == [Verdict.CONFIRMED, Verdict.CONFIRMED, Verdict.LOW_RISK]
    assert stub.relooked == [0.30]                          # only the band candidate cost compute
    assert r.stage_ms["inferences"] == 2                     # detect + one mosaic pass
    assert "no re-look spent" in r.candidates[0].trace[-1].rationale
    assert "lifted the score" in r.candidates[1].trace[-1].rationale
    assert r.candidates[2].trace[-1].detail["relooked"] is False


def test_review_card_carries_calibrated_p_pot_and_promise_text():
    stub = _BatchStub([_det(0.20)], {0.20: 0.35})
    agent = ReLookAgent(perceptor=stub, shadow_prover=ShadowProver(ShadowConfig(nadir="top")),
                        cfg=AgentConfig(tiers=_tiers()))
    c = agent.run_frame(_frame()).candidates[0]
    assert c.verdict is Verdict.REVIEW and c.evidence.p_pot == 0.6 and c.evidence.evidence_score == 0.35
    assert "REVIEW card" in c.trace[-1].rationale


def test_no_precision_promise_means_nothing_is_auto_confirmed():
    stub = _BatchStub([_det(0.95)], {})
    agent = ReLookAgent(perceptor=stub, shadow_prover=ShadowProver(ShadowConfig(nadir="top")),
                        cfg=AgentConfig(tiers=_tiers(tau_confirm=None)))
    c = agent.run_frame(_frame()).candidates[0]
    assert c.verdict is Verdict.REVIEW                       # even conf 0.95 is never auto-confirmed
    assert stub.relooked == [0.95]                            # re-look spent only to ORDER the queue
    assert "no precision promise" in c.trace[-1].rationale


def test_non_guaranteed_classes_never_auto_confirm():
    dets = [Detection((10, 10, 40, 40), 2, "structural_fragment", 0.9),
            Detection((60, 60, 90, 90), 3, "natural_formation", 0.9)]
    agent = ReLookAgent(perceptor=_BatchStub(dets, {}), shadow_prover=ShadowProver(ShadowConfig(nadir="top")),
                        cfg=AgentConfig(tiers=_tiers()))
    r = agent.run_frame(_frame())
    assert [c.verdict for c in r.candidates] == [Verdict.REVIEW, Verdict.LOW_RISK]


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} agent tests passed")


if __name__ == "__main__":
    _run_all()
