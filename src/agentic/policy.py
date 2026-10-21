"""
policy.py — the triage tiers (pure, deterministic).

:class:`GuaranteedTiers` holds the thresholds ``calibrate.py`` fits on the calibration recordings with
Clopper-Pearson / Learn-Then-Test (``guarantees.py``), and the promises they carry: the share of pots
that reach a person (REVIEW or CONFIRMED), and — only when calibration can support one — the precision
of CONFIRMED. A recall promise must also pass STUDY-21's block bootstrap for correlated frames. The same
rules drive the offline calibration and the live agent (``agent.py``), so the two can never drift. A
model without fitted tiers has no policy here: the agent sends every find to a person.

(The retired "legacy" rules — re-look ≥ 0.40 or confidence ≥ 0.60 ⇒ CONFIRMED, tuned on the test split
— are no longer in the product; ``calibrate.legacy_rules`` re-scores them on unseen data for the record.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .types import Verdict


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
    def from_calibration(cls, cal, mode: str = "standard") -> Optional["GuaranteedTiers"]:
        """The tiers of one perception mode: ``standard`` = ``tiers``; any other name = its entry in
        ``modes`` (fit and verified for that mode only — a promise is valid only for the mode it was
        calibrated on). None when the mode is not calibrated."""
        t = cal.tiers
        if t.get("tau_review") is None:
            return None
        if mode != "standard":
            m = cal.modes.get(mode)
            if not m or m.get("tau_review") is None:
                return None
            vt = m.get("verified_on_test") or {}
            return cls(tau_review=float(m["tau_review"]), tau_confirm=None, relook_mode=None,
                       guaranteed_class=cal.guaranteed_class,
                       non_hazard_classes=tuple(t.get("non_hazard_classes") or ()),
                       p_pot_bins=list(m.get("p_pot_bins") or []),
                       guarantees={"recall_promise": m.get("recall_promise"), "precision_promise": None,
                                   "recall_ceiling": m.get("recall_ceiling"), "requested_recall_achievable": False,
                                   "verified_on_test": {"recall": vt.get("recall"),
                                                        "recall_promise_held": vt.get("promise_held")},
                                   "report": m.get("report")},
                       policy=mode)
        return cls(tau_review=float(t["tau_review"]),
                   tau_confirm=None if t.get("tau_confirm") is None else float(t["tau_confirm"]),
                   relook_mode=t.get("relook_mode"), escalate_clahe=bool(t.get("escalate_clahe")),
                   combine=t.get("combine", "lift"),
                   guaranteed_class=cal.guaranteed_class,
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
