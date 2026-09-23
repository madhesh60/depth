"""
policy.py — the triage **rules** (pure, deterministic, validation-tuned).

Kept separate from the agent orchestration so the same rule drives both the offline calibration
(``calibrate.py``) and the live agent (``agent.py``) — they can never drift.

Thresholds were tuned on the real crab-pot test split (``experiments.md`` STUDY-03):

* **CONFIRMED** — ``relook.conf ≥ confirm_relook`` (0.40): the detection re-fires strongly when
  zoomed in. Precision at this tier is ~0.79 vs ~0.61 for the raw hot detector. Auto-trusted.
* **REJECTED** — low original confidence *and* it fails to re-fire: low evidence. **Retained for
  audit and deprioritised, never deleted** (recall-safe — a human can still see it).
* **REVIEW** — everything else: routed to a human, ranked by ``evidence_score``.

There is deliberately no rule that silently drops a candidate a human never sees.
"""
from __future__ import annotations

from dataclasses import dataclass

from .types import Evidence, Verdict


@dataclass
class TriageConfig:
    confirm_relook: float = 0.40   # re-look conf ≥ this ⇒ CONFIRMED (precision ~0.79 on test)
    reject_conf: float = 0.15      # original conf < this AND …
    reject_relook: float = 0.12    # … re-look conf < this ⇒ REJECTED (deprioritised, kept for audit)


def triage(det_conf: float, evidence: Evidence, cfg: TriageConfig | None = None) -> Verdict:
    """Map a candidate's confidence + evidence to a verdict. Deterministic."""
    cfg = cfg or TriageConfig()
    r = evidence.relook.conf
    if r >= cfg.confirm_relook:
        return Verdict.CONFIRMED
    if det_conf < cfg.reject_conf and r < cfg.reject_relook:
        return Verdict.REJECTED
    return Verdict.REVIEW
