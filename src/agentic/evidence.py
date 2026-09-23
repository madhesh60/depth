"""
evidence.py — the **Prove** stage: gather physical + consistency evidence for one candidate.

For each detection the gatherer runs the agent's evidence tools and fuses them into an
:class:`~src.agentic.types.Evidence`:

* **re-look persistence** (primary) — does the detection re-fire when zoomed in? This is the signal
  that actually separates true crab-pots from the detector's speckle false-positives
  (``experiments.md`` STUDY-03).
* **acoustic shadow** (secondary) — shown where it genuinely exists; contributes to the score only
  when a shadow is measurable (a minority of larger objects), and yields a height estimate.
* **echo strength** — reported for the evidence card; deliberately NOT weighted into the score
  because it is bright for both true and false detections and so is non-discriminative here.

``evidence_score`` is a 0..1 **ranking** aid (used to order the human review queue). The agent's
CONFIRMED/REVIEW/REJECTED triage uses the validated raw signals directly (see ``agent.py``), not
this score, so the two never drift.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.detection.infer import Detection
from .perception import Perceptor
from .shadow import ShadowProver
from .types import Evidence, RelookResult, ShadowProof, ShadowQuality


def fuse_score(det_conf: float, relook: RelookResult, proof: ShadowProof) -> float:
    """0..1 ranking score: re-look persistence dominates; a real shadow nudges it up; a modest
    original-confidence term breaks ties. Echo is intentionally excluded — non-discriminative.
    Single source of truth shared by :class:`EvidenceGatherer` and the agent (``agent.py``)."""
    score = 0.70 * min(relook.conf, 1.0) + 0.15 * proof.strength + 0.15 * min(det_conf / 0.5, 1.0)
    return float(np.clip(score, 0.0, 1.0))


def evidence_notes(det_conf: float, cls_name: str, relook: RelookResult, proof: ShadowProof) -> list[str]:
    """Human-readable evidence lines for the card. Shared by the gatherer and the agent."""
    notes: list[str] = []
    if relook.found:
        verb = "persisted" if relook.conf >= det_conf else "weakened"
        notes.append(f"re-look {verb}: re-fired at {relook.conf:.2f} on {relook.scale:.1f}x zoom")
    else:
        notes.append("re-look: did not re-fire when zoomed in")
    if proof.quality is ShadowQuality.CLEAR:
        h = f", h~{proof.height_m:.1f} m" if proof.height_m is not None else ""
        notes.append(f"acoustic shadow: clear (contrast {proof.contrast:.2f}, {proof.run_px}px{h})")
    elif proof.quality is ShadowQuality.WEAK:
        notes.append(f"acoustic shadow: weak (contrast {proof.contrast:.2f})")
    else:
        notes.append("acoustic shadow: none (common for small low-relief pots in speckle)")
    notes.append(f"echo {proof.echo_ratio:.1f}x local background")
    return notes


class EvidenceGatherer:
    """Runs re-look + shadow + echo for a candidate and fuses them into an Evidence object."""

    def __init__(self, perceptor: Perceptor, shadow_prover: ShadowProver | None = None):
        self.perceptor = perceptor
        self.shadow = shadow_prover or ShadowProver()

    def gather(self, frame: np.ndarray, det: Detection, nadir: str | None = None) -> Evidence:
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        relook = self.perceptor.zoom_relook(frame, det.bbox, det.cls_name)
        relook.gain = round(relook.conf - det.conf, 4)

        proof = self.shadow.prove(gray, det.bbox, nadir=nadir)

        score = fuse_score(det.conf, relook, proof)
        notes = evidence_notes(det.conf, det.cls_name, relook, proof)
        return Evidence(relook=relook, shadow=proof, echo_ratio=proof.echo_ratio,
                        evidence_score=round(score, 3), notes=notes)
