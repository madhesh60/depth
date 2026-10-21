"""
evidence.py — the **Prove** stage's card text: what the evidence tools found, in words a person reads.

The agent (``agent.py``) runs the evidence tools itself — the Stage-1 water-column check, the acoustic
**shadow** (``shadow.ShadowProver``) and its height, and, where calibration showed it can change a
decision, a zoomed **re-look** — and :func:`evidence_notes` turns their results into the lines on the
evidence card. None of it is a silent gate: on EXP-003 the re-look did not beat plain detector
confidence (STUDY-07) and the shadow separates true from false finds barely better than chance (AUC
0.563), so they are shown to the person, not folded into the calibrated tiers.
"""
from __future__ import annotations

from .types import RelookResult, ShadowProof, ShadowQuality


def evidence_notes(det_conf: float, cls_name: str, relook: RelookResult, proof: ShadowProof) -> list[str]:
    """Human-readable evidence lines for the card."""
    notes: list[str] = []
    if relook.found:
        verb = "persisted" if relook.conf >= det_conf else "weakened"
        notes.append(f"re-look {verb}: re-fired at {relook.conf:.2f} on {relook.scale:.1f}x zoom")
    else:
        notes.append("re-look: did not re-fire when zoomed in")
    if not proof.orientation_known:
        notes.append("acoustic shadow: not measured (frame orientation unknown)")
    elif proof.quality is ShadowQuality.CLEAR:
        h = (f", h~{proof.height_m:.2f} m" if proof.height_m is not None
             else f", h~{100 * proof.height_rel:.0f}% of sonar altitude")
        notes.append(f"acoustic shadow: clear (contrast {proof.contrast:.2f}, {proof.run_px}px{h})")
    elif proof.quality is ShadowQuality.WEAK:
        notes.append(f"acoustic shadow: weak (contrast {proof.contrast:.2f})")
    else:
        notes.append("acoustic shadow: none (common for small low-relief pots in speckle)")
    notes.append(f"echo {proof.echo_ratio:.1f}x local background")
    return notes
