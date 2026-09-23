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
    """The agent's decision for a candidate. No candidate is ever deleted — REJECTED means
    'low evidence, retained for audit and deprioritised', not discarded (recall-safe)."""
    CONFIRMED = "confirmed"   # auto-trusted; high precision; eligible for the cleanup route
    REVIEW = "review"         # routed to a human; ranked by evidence (recall preserved here)
    REJECTED = "rejected"     # low evidence; kept for audit, not surfaced as a hazard


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
    height_m: Optional[float]            # estimated object height, metres (None if scale unknown)
    height_rel: float                    # scale-free height proxy = run / ground_range
    echo_ratio: float                    # echo brightness / local background
    echo_xy: tuple[int, int]             # echo location in original pixels (for the overlay)
    strip: tuple[int, int, int, int]     # shadow search box x1,y1,x2,y2 (for the overlay)

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
    evidence_score: float                # 0..1 combined trust score (drives ranking + triage)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "relook": self.relook.to_dict(),
            "shadow": self.shadow.to_dict(),
            "echo_ratio": self.echo_ratio,
            "evidence_score": self.evidence_score,
            "notes": list(self.notes),
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
        }


@dataclass
class FrameResult:
    """Everything produced for one sonar frame by the See → Prove → Decide pipeline."""
    frame_id: str
    width: int
    height: int
    candidates: list[Candidate]
    stage_ms: dict[str, float] = field(default_factory=dict)   # see/prove/decide latency
    nadir: str = "top"                                         # calibrated range/nadir edge

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
            "counts": self.counts,
            "stage_ms": {k: round(v, 2) for k, v in self.stage_ms.items()},
            "candidates": [c.to_dict() for c in self.candidates],
        }
