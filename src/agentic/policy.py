"""
policy.py — the triage **rules** (pure, deterministic, validation-tuned).

Kept separate from the agent orchestration so the same rule drives both the offline calibration
(``calibrate.py``) and the live agent (``agent.py``) — they can never drift.

Thresholds were tuned on the real crab-pot test split (``experiments.md`` STUDY-03 / STUDY-04):

* **CONFIRMED** — a candidate is auto-trusted when *either* independently-calibrated path fires:
    - ``relook.conf ≥ confirm_relook`` (0.40): the detection re-fires strongly when zoomed in
      (precision ~0.71 at this tier), **or**
    - ``det_conf ≥ confirm_conf`` (0.60): the detector was already highly confident (precision ~0.83).
  Their **union** is precision ~0.74 at ~30% recall-share — strictly better than the re-look path
  alone on *both* axes (STUDY-04). The acoustic shadow is deliberately **not** a confirm gate: on
  this data its CLEAR-rate is identical for true and false detections (14.5% vs 14.6%), so gating on
  it *lowers* precision — it is kept only as evidence-shown-where-present + a height estimate.
* **REJECTED** — low original confidence *and* it fails to re-fire: low evidence. **Retained for
  audit and deprioritised, never deleted** (recall-safe — this tier still holds ~9% of true pots, so
  a human can see them; the CONFIRMED+REVIEW tiers retain ~91%).
* **REVIEW** — everything else: routed to a human, ranked by ``evidence_score``.

There is deliberately no rule that silently drops a candidate a human never sees.
"""
from __future__ import annotations

from dataclasses import dataclass

from .types import Evidence, Verdict


@dataclass
class TriageConfig:
    confirm_relook: float = 0.40   # re-look conf ≥ this ⇒ CONFIRMED (precision ~0.71 on test)
    confirm_conf: float = 0.60     # OR original conf ≥ this ⇒ CONFIRMED (precision ~0.83; union ~0.74)
    reject_conf: float = 0.15      # original conf < this AND …
    reject_relook: float = 0.12    # … re-look conf < this ⇒ REJECTED (deprioritised, kept for audit)


def confirm_path(det_conf: float, evidence: Evidence, cfg: TriageConfig | None = None) -> str:
    """Which CONFIRMED path fires (for the trace/UX): 'relook' | 'high_confidence' | 'both' | ''."""
    cfg = cfg or TriageConfig()
    r = evidence.relook.conf
    by_relook = r >= cfg.confirm_relook
    by_conf = det_conf >= cfg.confirm_conf
    if by_relook and by_conf:
        return "both"
    if by_relook:
        return "relook"
    if by_conf:
        return "high_confidence"
    return ""


def triage(det_conf: float, evidence: Evidence, cfg: TriageConfig | None = None) -> Verdict:
    """Map a candidate's confidence + evidence to a verdict. Deterministic."""
    cfg = cfg or TriageConfig()
    r = evidence.relook.conf
    if r >= cfg.confirm_relook or det_conf >= cfg.confirm_conf:
        return Verdict.CONFIRMED
    if det_conf < cfg.reject_conf and r < cfg.reject_relook:
        return Verdict.REJECTED
    return Verdict.REVIEW


def decide(det_conf: float, evidence: Evidence, cfg: TriageConfig | None = None) -> tuple[Verdict, str]:
    """Verdict + the CONFIRMED path that produced it (empty string when not confirmed)."""
    cfg = cfg or TriageConfig()
    return triage(det_conf, evidence, cfg), confirm_path(det_conf, evidence, cfg)
