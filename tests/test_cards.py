"""
Tests for one review card per object (src/agentic/cards.py, STUDY-23): the rule, the fold on a frame's
candidates, what a person sees (counts, hazards, study cards), and that no box is ever dropped.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agentic import study
from src.agentic.agent import AgentConfig, TriageAgent
from src.agentic.cards import card_count, fold, group, ios
from src.agentic.pipeline import AgenticPipeline
from src.agentic.shadow import ShadowConfig, ShadowProver
from src.agentic.types import AgentStep, Candidate, FrameResult, Verdict
from src.detection.infer import Detection


def _cand(box, conf, verdict=Verdict.REVIEW, cls="ghost_gear"):
    return Candidate(bbox=box, cls_id=0, cls_name=cls, conf=conf, verdict=verdict)


def test_ios_is_the_share_of_the_smaller_box_inside_the_other():
    assert ios((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert ios((2, 2, 4, 4), (0, 0, 20, 20)) == 1.0                    # a small box inside a large one
    assert abs(ios((0, 0, 10, 10), (5, 0, 15, 10)) - 0.5) < 1e-9       # half of it
    assert ios((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0                 # touching edges do not overlap


def test_group_joins_one_object_and_keeps_classes_and_far_boxes_apart():
    boxes = [(303, 151, 318, 173), (281, 148, 343, 211), (291, 150, 328, 181),   # three boxes on one pot
             (20, 20, 40, 40),                                                    # another object
             (300, 150, 320, 175)]                                                # same place, other class
    classes = ["g", "g", "g", "g", "w"]
    cards = group(boxes, classes, [0.22, 0.05, 0.09, 0.3, 0.4], 0.5)
    assert sorted(map(sorted, cards)) == [[0, 1, 2], [3], [4]]
    assert next(c for c in cards if len(c) == 3)[0] == 0                # the strongest box heads the card


def test_group_is_transitive():
    # a ~ b and b ~ c (half of the smaller box inside) -> one card, although a and c do not touch
    cards = group([(0, 0, 10, 10), (5, 0, 25, 10), (20, 0, 30, 10)], ["g"] * 3, [0.3, 0.2, 0.1], 0.5)
    assert cards == [[0, 1, 2]]


def test_fold_marks_members_counts_cards_and_drops_no_box():
    cands = [_cand((303, 151, 318, 173), 0.22), _cand((281, 148, 343, 211), 0.08),
             _cand((20, 20, 40, 40), 0.30), _cand((300, 150, 322, 176), 0.04, Verdict.LOW_RISK)]
    n = fold(cands, 0.5)
    assert n == 1 and cands[1].card_of == 0 and cands[0].card_members == [1]
    assert cands[2].card_of is None and cands[3].card_of is None        # low-risk boxes are not cards
    assert card_count(cands) == 2 and len(cands) == 4                   # every box is kept
    assert cands[0].trace[-1].tool == "one_card" and cands[1].trace[-1].detail["card_of"] == 0
    c = _cand((0, 0, 10, 10), 0.3)
    c.trace = [AgentStep("decide", "tier", 0.0), AgentStep("handoff", "a person decides", 0.0)]
    pair = [c, _cand((2, 2, 8, 8), 0.1)]
    fold(pair, 0.5)
    assert [s.tool for s in c.trace] == ["decide", "one_card", "handoff"]    # the chain still ends with who acts
    fr = FrameResult(frame_id="f", width=400, height=300, candidates=cands)
    assert fr.counts["review"] == 2 and fr.counts["low_risk"] == 1 and fr.boxes == 4
    assert fr.to_dict()["candidates"][1]["card_of"] == 0
    assert fold(cands, None) == 0 and all(c.card_of is None for c in cands)   # rule off: one card per box


class _Stub:
    def __init__(self, dets):
        self.dets = dets

    def perceive(self, frame):
        return list(self.dets)


def _frame():
    return (90 + np.random.default_rng(0).normal(0, 4, (240, 400, 3))).clip(0, 255).astype(np.uint8)


def test_the_frame_agent_folds_with_the_calibrated_rule_and_hazards_follow_the_cards():
    dets = [Detection((300, 150, 318, 173), 0, "ghost_gear", 0.40), Detection((285, 148, 340, 205), 0, "ghost_gear", 0.07),
            Detection((40, 60, 60, 80), 0, "ghost_gear", 0.30)]
    agent = TriageAgent(perceptor=_Stub(dets), shadow_prover=ShadowProver(ShadowConfig(nadir="top")),
                        cfg=AgentConfig(card_ios=0.5))
    r = agent.run_frame(_frame(), frame_id="Rec9_wcp_ss_port_00001")
    assert [c.verdict for c in r.candidates] == [Verdict.REVIEW] * 3     # uncalibrated: every find to a person
    assert r.perception["folded"] == 1 and r.counts["review"] == 2 and r.boxes == 3
    s = AgenticPipeline(agent=agent).assemble([r], None, "t")
    assert len(s.tracked) == 2 and sorted(t.boxes for t in s.tracked) == [1, 2]


def test_a_study_card_is_real_when_any_of_its_boxes_hits_a_pot(monkeypatch):
    frames = [{"frame_id": "F0", "sample_id": None, "path": "x", "gt": [(100, 100, 116, 112)]}]
    monkeypatch.setattr(study, "study_frames", lambda: frames)
    p = {"session_id": "s", "participant": "p", "index": 0, "order": ["cards", "manual"],
         "manual_frames": [], "card_frames": [{"frame_id": "F0"}]}
    # the head is a loose box around the pot (IoU < 0.3); the tight box on the same card hits it
    card = {"card_id": "c0", "frame_id": "F0", "bbox": [80, 80, 140, 136], "boxes": [[80, 80, 140, 136], [100, 99, 117, 113]]}
    r = study.score(p, {"cards": [{"card_id": "c0", "decision": "real", "ms": 3000}]}, {"c0": card})
    assert r["cards"]["confirmed_real"] == 1 and r["cards"]["recall"] == 1.0
