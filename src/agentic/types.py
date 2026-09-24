"""
types.py — serialisable data structures shared across the See → Prove → Decide → Act loop.

Every structure is a plain ``@dataclass`` with a ``to_dict()`` that yields JSON-safe primitives
(tuples → lists, enums → their value), so the same objects flow unchanged from the Python
pipeline into the FastAPI responses and the dashboard.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


def _json(obj):
    """Recursively coerce dataclasses/enums/tuples into JSON-safe primitives."""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json(v) for v in obj]
    return obj


class Verdict(str, Enum):
    """The agent's decision for a candidate. No candidate is ever deleted.

    With a fitted ``calibration.json`` the tiers are statistical PROMISES (``guarantees.py``):
    CONFIRMED ⇒ "≥ target precision are real"; CONFIRMED+REVIEW ⇒ "≥ promised share of pots reach a
    human"; LOW_RISK ⇒ "holds at most the complementary share of pots" — retained for audit,
    deprioritised, never deleted. (LOW_RISK was called REJECTED; the old name is kept as an alias.)"""
    CONFIRMED = "confirmed"   # auto-trusted under the precision promise; eligible for the route
    REVIEW = "review"         # routed to a human; ordered by P(pot) (the recall promise lives here)
    LOW_RISK = "low_risk"     # below τ_review; kept for audit, not surfaced as a hazard
    REJECTED = "low_risk"     # legacy alias of LOW_RISK


class ShadowQuality(str, Enum):
    CLEAR = "clear"
    WEAK = "weak"
    NONE = "none"


@dataclass
class ShadowProof:
    """Result of the acoustic-shadow measurement for one candidate (Prove — physical evidence).

    On this data a shadow is only measurable on a minority of objects; ``quality == NONE`` is a
    common, honest outcome and does NOT by itself reject a candidate.
    """
    quality: ShadowQuality
    contrast: float                      # (background − shadow) / background, ∈ (−∞, 1]
    run_px: int                          # shadow length along the range axis, pixels
    strength: float                      # 0..1 soft score (contrast × length plausibility)
    height_m: Optional[float]            # height in metres — ONLY when the altitude was measured
    height_rel: float                    # height as a fraction of sonar altitude, h/H = Ls/(R+Ls)
    echo_ratio: float                    # echo brightness / local background
    echo_xy: tuple[int, int]             # echo location in original pixels (for the overlay)
    strip: tuple[int, int, int, int]     # shadow search box x1,y1,x2,y2 (for the overlay)
    orientation_known: bool = True       # False ⇒ frame orientation unknown, nothing measured

    @property
    def has_shadow(self) -> bool:
        return self.quality is not ShadowQuality.NONE

    def to_dict(self) -> dict:
        return _json(asdict(self))


@dataclass
class RelookResult:
    """Result of a zoom-in re-detection (the agent's core 'squint and look again' action)."""
    found: bool
    conf: float                          # best re-fire confidence over the zoomed crop (0 if none)
    gain: float                          # conf − original detector confidence
    scale: float                         # zoom factor applied to the crop

    def to_dict(self) -> dict:
        return _json(asdict(self))


@dataclass
class Evidence:
    """All physical + consistency evidence gathered for one candidate (the output of Prove)."""
    relook: RelookResult
    shadow: ShadowProof
    echo_ratio: float
    evidence_score: float                # 0..1 the agent score the tiers are calibrated on
    notes: list[str] = field(default_factory=list)
    p_pot: Optional[float] = None        # calibrated P(real pot | score) — budget-mode ordering

    def to_dict(self) -> dict:
        return {
            "relook": self.relook.to_dict(),
            "shadow": self.shadow.to_dict(),
            "echo_ratio": self.echo_ratio,
            "evidence_score": self.evidence_score,
            "notes": list(self.notes),
            "p_pot": self.p_pot,
        }


@dataclass
class AgentStep:
    """One entry in the agent's decision trace — the Agentic-Vision audit record."""
    tool: str                            # e.g. "detect", "zoom_relook", "shadow_check", "decide"
    rationale: str                       # human-readable why
    latency_ms: float
    conf_before: Optional[float] = None
    conf_after: Optional[float] = None
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return _json(asdict(self))


@dataclass
class Candidate:
    """A single detection as it moves through Prove → Decide, carrying its full audit trail."""
    bbox: tuple[int, int, int, int]      # x1, y1, x2, y2 in ORIGINAL pixels
    cls_id: int
    cls_name: str
    conf: float                          # original detector confidence
    evidence: Optional[Evidence] = None
    verdict: Optional[Verdict] = None
    trace: list[AgentStep] = field(default_factory=list)
    # geo (filled by Act; None ⇒ no GPS available for this frame)
    lat: Optional[float] = None
    lon: Optional[float] = None
    geo_error_m: Optional[float] = None
    # chunk-boundary stitching (survey-level): the same object split across two adjacent chunks
    continues_in: Optional[str] = None   # frame id where this object continues (kept as primary)
    continuation_of: Optional[str] = None  # frame id of the primary sighting (this one is merged)
    # Stage-1 geometry (canonical view: nadir at top): which ping the object is on, its across-track
    # GROUND range (slant corrected with the tracked altitude) and whether it floats in the water column
    ping_px: Optional[float] = None
    n_pings: Optional[int] = None
    ground_range_px: Optional[float] = None
    in_water_column: Optional[bool] = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "cls_id": self.cls_id,
            "cls_name": self.cls_name,
            "conf": self.conf,
            "evidence": self.evidence.to_dict() if self.evidence else None,
            "verdict": self.verdict.value if self.verdict else None,
            "trace": [s.to_dict() for s in self.trace],
            "lat": self.lat,
            "lon": self.lon,
            "geo_error_m": self.geo_error_m,
            "continues_in": self.continues_in,
            "continuation_of": self.continuation_of,
            "ground_range_px": None if self.ground_range_px is None else round(self.ground_range_px, 1),
            "in_water_column": self.in_water_column,
        }


@dataclass
class FrameResult:
    """Everything produced for one sonar frame by the See → Prove → Decide pipeline."""
    frame_id: str
    width: int
    height: int
    candidates: list[Candidate]
    stage_ms: dict[str, float] = field(default_factory=dict)   # see/prove/decide latency
    nadir: str = "unknown"                                     # resolved nadir edge (or "unknown")
    orientation: dict = field(default_factory=dict)            # {nadir, rule, source} provenance
    stage1: dict = field(default_factory=dict)                 # Stage-1 canonicalisation record

    def by_verdict(self, v: Verdict) -> list[Candidate]:
        return [c for c in self.candidates if c.verdict is v]

    @property
    def counts(self) -> dict[str, int]:
        return {v.value: len(self.by_verdict(v)) for v in Verdict}

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "width": self.width,
            "height": self.height,
            "nadir": self.nadir,
            "orientation": dict(self.orientation),
            "stage1": dict(self.stage1),
            "counts": self.counts,
            "stage_ms": {k: round(v, 2) for k, v in self.stage_ms.items()},
            "candidates": [c.to_dict() for c in self.candidates],
        }


@dataclass
class TrackedObject:
    """A survey-level hazard: one candidate promoted to the map/route, with geo + evidence summary."""
    oid: str
    cls_name: str
    verdict: Verdict
    conf: float
    evidence_score: float
    frame_id: str
    bbox: tuple[int, int, int, int]
    lat: Optional[float] = None
    lon: Optional[float] = None
    geo_error_m: Optional[float] = None
    height_m: Optional[float] = None
    height_rel: float = 0.0
    shadow_quality: str = "none"
    also_in: list[str] = field(default_factory=list)   # other chunks showing the same object
    p_pot: Optional[float] = None                      # calibrated P(real pot) for REVIEW ordering

    def to_dict(self) -> dict:
        return _json(asdict(self))


@dataclass
class MissionPlan:
    """The **Act** output: a human-approved cleanup route + resurvey list over a survey's hazards."""
    recovery_route: list[str]                        # ordered TrackedObject ids (nearest-neighbour)
    resurvey: list[str]                              # REVIEW object ids a human should revisit
    route_length_m: Optional[float]                 # None when no GPS
    gps_available: bool
    gps_synthetic: bool = False                     # True ⇒ demo track, NOT real coordinates
    human_approval_required: bool = True            # nothing is ever auto-dispatched
    counts: dict[str, int] = field(default_factory=dict)
    review_queue: list[str] = field(default_factory=list)   # REVIEW ids, most-likely-pot first
    inspection_route: list[str] = field(default_factory=list)   # REVIEW ids to inspect (budget), NN order
    inspection_length_m: Optional[float] = None
    budget: dict = field(default_factory=dict)             # budget-mode plan (analyst minutes)
    guarantees: dict = field(default_factory=dict)         # the promises the tiers carry

    def to_dict(self) -> dict:
        return _json(asdict(self))


@dataclass
class SurveyResult:
    """Everything produced for a multi-frame survey (frames + hazards + the mission plan)."""
    survey_id: str
    frames: list[FrameResult]
    tracked: list[TrackedObject]
    mission: MissionPlan

    def to_dict(self) -> dict:
        return {
            "survey_id": self.survey_id,
            "frames": [f.to_dict() for f in self.frames],
            "tracked": [t.to_dict() for t in self.tracked],
            "mission": self.mission.to_dict(),
        }
