"""
The evidence card's text (src/agentic/evidence.py): what the re-look and the acoustic shadow found, in words
a person reads. (The retired re-look ladder and its fused score were removed with the legacy rules.)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agentic.evidence import evidence_notes
from src.agentic.types import RelookResult, ShadowProof, ShadowQuality


def _proof(quality=ShadowQuality.CLEAR, height_m=None, known=True) -> ShadowProof:
    return ShadowProof(quality, 0.4, 8, 0.8, height_m, 0.2, 3.1, (10, 10), (5, 12, 15, 20), orientation_known=known)


def test_relook_notes_say_what_happened():
    hi = evidence_notes(0.20, "ghost_gear", RelookResult(found=True, conf=0.7, gain=0.5, scale=2.0), _proof())
    lo = evidence_notes(0.20, "ghost_gear", RelookResult(found=True, conf=0.1, gain=-0.1, scale=2.0), _proof())
    no = evidence_notes(0.20, "ghost_gear", RelookResult(found=False, conf=0.0, gain=0.0, scale=1.0), _proof())
    assert hi[0].startswith("re-look persisted") and lo[0].startswith("re-look weakened")
    assert no[0] == "re-look: did not re-fire when zoomed in"


def test_shadow_notes_never_guess():
    rl = RelookResult(found=False, conf=0.0, gain=0.0, scale=1.0)
    assert "not measured" in evidence_notes(0.2, "ghost_gear", rl, _proof(known=False))[1]
    rel = evidence_notes(0.2, "ghost_gear", rl, _proof())[1]
    assert "clear" in rel and "20% of sonar altitude" in rel                # no metres without a measured altitude
    assert "h~0.35 m" in evidence_notes(0.2, "ghost_gear", rl, _proof(height_m=0.35))[1]
    assert "none" in evidence_notes(0.2, "ghost_gear", rl, _proof(ShadowQuality.NONE))[1]


def test_echo_is_reported_not_weighted():
    rl = RelookResult(found=False, conf=0.0, gain=0.0, scale=1.0)
    assert evidence_notes(0.2, "ghost_gear", rl, _proof())[-1] == "echo 3.1x local background"
