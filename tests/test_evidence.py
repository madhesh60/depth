"""
Tests for the Prove-stage evidence fusion + the triage policy (no model needed — the perceptor's
re-look is stubbed so the logic is tested deterministically).

Runs under pytest, or standalone: python tests/test_evidence.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.infer import Detection
from src.agentic.evidence import EvidenceGatherer
from src.agentic.shadow import ShadowProver, ShadowConfig
from src.agentic.types import RelookResult, Verdict
from src.agentic.policy import TriageConfig, triage


class _StubPerceptor:
    """Minimal perceptor whose re-look returns a preset confidence."""
    def __init__(self, relook_conf: float):
        self._c = relook_conf

    def zoom_relook(self, frame, bbox, cls_name="fishing_gear") -> RelookResult:
        return RelookResult(found=self._c > 0, conf=self._c, gain=0.0, scale=2.0)


def _frame_and_det():
    img = (90 + np.random.default_rng(0).normal(0, 4, (200, 200))).clip(40, 140).astype(np.uint8)
    det = Detection(bbox=(92, 47, 108, 63), cls_id=0, cls_name="fishing_gear", conf=0.20)
    return img, det


def test_evidence_score_tracks_relook():
    """A strong re-look yields a much higher evidence score than none."""
    img, det = _frame_and_det()
    sp = ShadowProver(ShadowConfig(nadir="top"))
    hi = EvidenceGatherer(_StubPerceptor(0.7), sp).gather(img, det)
    lo = EvidenceGatherer(_StubPerceptor(0.0), sp).gather(img, det)
    assert hi.evidence_score > lo.evidence_score + 0.3
    assert hi.relook.gain == round(0.7 - det.conf, 4)
    assert any("re-look persisted" in n for n in hi.notes)
    assert any("did not re-fire" in n for n in lo.notes)


def test_triage_confirmed_on_strong_relook():
    img, det = _frame_and_det()
    ev = EvidenceGatherer(_StubPerceptor(0.55), ShadowProver(ShadowConfig(nadir="top"))).gather(img, det)
    assert triage(det.conf, ev) is Verdict.CONFIRMED


def test_triage_rejected_only_when_low_conf_and_no_relook():
    img, det = _frame_and_det()          # det.conf = 0.20
    sp = ShadowProver(ShadowConfig(nadir="top"))
    ev = EvidenceGatherer(_StubPerceptor(0.0), sp).gather(img, det)
    # conf 0.20 is above reject_conf 0.15 -> REVIEW, not LOW_RISK (legacy rule, recall-safe)
    assert triage(det.conf, ev) is Verdict.REVIEW
    low = Detection(bbox=det.bbox, cls_id=0, cls_name="fishing_gear", conf=0.11)
    ev_low = EvidenceGatherer(_StubPerceptor(0.0), sp).gather(img, low)
    assert triage(low.conf, ev_low) is Verdict.REJECTED


def test_triage_config_is_respected():
    img, det = _frame_and_det()
    ev = EvidenceGatherer(_StubPerceptor(0.30), ShadowProver(ShadowConfig(nadir="top"))).gather(img, det)
    assert triage(det.conf, ev, TriageConfig(confirm_relook=0.25)) is Verdict.CONFIRMED
    assert triage(det.conf, ev, TriageConfig(confirm_relook=0.50)) is Verdict.REVIEW


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} evidence/policy tests passed")


if __name__ == "__main__":
    _run_all()
