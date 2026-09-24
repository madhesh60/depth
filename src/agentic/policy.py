"""
policy.py — the triage rules (pure, deterministic).

Two modes:

* :class:`GuaranteedTiers` (**product default when ``calibration.json`` is fitted**) — thresholds fit
  on VALIDATION by ``calibrate.py`` with Clopper-Pearson / Learn-Then-Test so each tier carries a
  statistical promise (recall of pots reaching a human; precision of CONFIRMED). See
  ``guarantees.py`` and ``docs/calibration_exp001.md``.
* :class:`TriageConfig` / :func:`triage` — the LEGACY rules below, kept for tests and as the fallback
  when no calibration exists. Their thresholds were tuned on the test split (a flaw the review
  called out); the guaranteed tiers replace them in the product.

Legacy notes follow.

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

from dataclasses import dataclass, field
from typing import Optional

from .types import Evidence, Verdict


@dataclass
class GuaranteedTiers:
    """Calibrated tier thresholds + the promises they carry (from ``calibration.json``)."""
    tau_review: float
    tau_confirm: Optional[float]                  # None ⇒ no auto-confirm can be promised
    relook_mode: Optional[str] = "mosaic"         # "single" | "mosaic" | None (confidence only)
    escalate_clahe: bool = False
    combine: str = "lift"                         # "lift" = max(conf, re-look) | "demote" = veto only
    guaranteed_class: str = "fishing_gear"
    non_hazard_classes: tuple = ("natural_formation",)
    p_pot_bins: list = field(default_factory=list)
    guarantees: dict = field(default_factory=dict)
    policy: str = ""

    @classmethod
    def from_calibration(cls, cal) -> Optional["GuaranteedTiers"]:
        t = cal.tiers
        if t.get("tau_review") is None:
            return None
        return cls(tau_review=float(t["tau_review"]),
                   tau_confirm=None if t.get("tau_confirm") is None else float(t["tau_confirm"]),
                   relook_mode=t.get("relook_mode"), escalate_clahe=bool(t.get("escalate_clahe")),
                   combine=t.get("combine", "lift"),
                   guaranteed_class=t.get("guaranteed_class", "fishing_gear"),
                   non_hazard_classes=tuple(t.get("non_hazard_classes") or ()),
                   p_pot_bins=list(t.get("p_pot_bins") or []), guarantees=dict(t.get("guarantees") or {}),
                   policy=t.get("policy", ""))

    # -- value of information: can more looking change this candidate's outcome? -------------
    def needs_relook(self, conf: float) -> bool:
        """Re-look only where it can change something: the tier, or (when the calibrated score
        depends on the re-look) the candidate's place in the human REVIEW queue."""
        if self.relook_mode is None or conf < self.tau_review:
            return False                       # score is confidence-only / LOW-RISK regardless
        if self.tau_confirm is None:
            return True                        # no auto-confirm: the re-look orders the REVIEW queue
        if self.combine == "demote":
            return conf * 0.5 < self.tau_confirm   # a veto could still drop it below tau_confirm
        return conf < self.tau_confirm         # lift: above tau_confirm it is CONFIRMED anyway

    in_band = needs_relook                     # back-compat name

    def score(self, conf: float, relook_conf: Optional[float]) -> float:
        """The calibrated agent score (must match ``calibrate.agent_score`` for the chosen policy)."""
        if relook_conf is None or self.relook_mode is None:
            return conf
        if self.combine == "demote":
            return conf * (0.5 + 0.5 * float(relook_conf > 0))
        return max(conf, relook_conf)

    def tier(self, conf: float, score: float) -> Verdict:
        if self.tau_confirm is not None and score >= self.tau_confirm and conf >= self.tau_review:
            return Verdict.CONFIRMED
        if conf >= self.tau_review:
            return Verdict.REVIEW
        return Verdict.LOW_RISK

    def p_pot(self, score: float) -> Optional[float]:
        if not self.p_pot_bins:
            return None
        for b in self.p_pot_bins:
            if score <= b["hi"]:
                return float(b["p_pot"])
        return float(self.p_pot_bins[-1]["p_pot"])

    @property
    def recall_promise(self) -> Optional[float]:
        return self.guarantees.get("recall_promise")

    @property
    def precision_promise(self) -> Optional[float]:
        return self.guarantees.get("precision_promise")


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
