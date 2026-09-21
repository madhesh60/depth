"""Stage 1 classical-CV pipeline (the COOL core workload)."""
from .config import Stage1Config, QUALITY_PRESET
from .pipeline import Stage1Pipeline, Stage1Result, Candidate, draw_candidates

__all__ = [
    "Stage1Config", "QUALITY_PRESET",
    "Stage1Pipeline", "Stage1Result", "Candidate", "draw_candidates",
]
