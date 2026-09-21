"""
config.py — Stage 1 classical-CV pipeline configuration.

All Stage-1 behaviour is set here (no magic numbers in the pipeline code), so a run is
fully described by a single config object and is reproducible from it.

COOL note
---------
The Cloud-Optimized OpenCV Library (COOL / KleidiCV) accelerates a specific set of ops on
AWS Graviton / Arm — notably ``resize``, adaptive-gaussian thresholding, and contour
detection. The defaults here deliberately bias Stage 1 toward those ops so the Arm-vs-x86
and COOL-vs-stock benchmarks show a real, attributable speedup. The one heavy op that COOL
does *not* accelerate — ``fastNlMeansDenoising`` — is available but OFF by default (it can
blow the <300 ms/frame budget and dilutes COOL's measured contribution). See
``docs/dataset_report.md`` and the module README.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional


@dataclass
class Stage1Config:
    # ---- input normalisation (resize is COOL-accelerated) ----
    resize_long_side: Optional[int] = 1024   # scale so the long side == this; None keeps size

    # ---- speckle denoise ----
    # "gaussian" | "median" | "bilateral" | "nlmeans" | "none"
    # gaussian/median are cheap and COOL-friendly; nlmeans is high-quality but slow + NOT
    # KleidiCV-accelerated — use only when quality matters more than the COOL benchmark.
    denoise: str = "median"
    gaussian_ksize: int = 5
    median_ksize: int = 5
    bilateral_d: int = 7
    bilateral_sigma: float = 50.0
    nlmeans_h: float = 10.0

    # ---- adaptive segmentation (adaptive-gaussian is a COOL sweet-spot op) ----
    thresh_method: str = "adaptive_gaussian"   # "adaptive_gaussian" | "otsu"
    adaptive_block: int = 35                    # odd; local neighbourhood for the threshold
    adaptive_C: int = -5                        # subtracted from the local mean

    # ---- morphological cleanup ----
    morph_op: str = "open"                      # "open" | "close" | "both" | "none"
    morph_ksize: int = 3
    morph_iterations: int = 1

    # ---- geometric contour filtering (contour detection is COOL-accelerated) ----
    # Areas are fractions of the (resized) frame area, so thresholds are resolution-independent.
    min_area_frac: float = 5e-4                 # reject speckle-sized blobs
    max_area_frac: float = 0.50                 # reject near-full-frame regions (shadow bands)
    min_solidity: float = 0.30                  # area / convex-hull area — rejects wispy clutter
    min_extent: float = 0.15                    # area / bbox area — rejects sparse scatter
    max_aspect: float = 15.0                    # bbox long/short — allows elongated pipes/nets
    roi_pad_frac: float = 0.06                  # pad each ROI outward before handing to Stage 2

    # ---- misc ----
    max_candidates: int = 64                    # cap ROIs/frame (largest-area first)

    def validate(self) -> "Stage1Config":
        if self.adaptive_block % 2 == 0 or self.adaptive_block < 3:
            raise ValueError("adaptive_block must be odd and >= 3")
        if self.denoise not in {"gaussian", "median", "bilateral", "nlmeans", "none"}:
            raise ValueError(f"unknown denoise: {self.denoise}")
        if self.thresh_method not in {"adaptive_gaussian", "otsu"}:
            raise ValueError(f"unknown thresh_method: {self.thresh_method}")
        if self.morph_op not in {"open", "close", "both", "none"}:
            raise ValueError(f"unknown morph_op: {self.morph_op}")
        return self

    def to_dict(self) -> dict:
        return asdict(self)


# A slower, higher-quality preset (NLMeans denoise) — kept for the ablation study, NOT for
# the COOL benchmark (NLMeans is not KleidiCV-accelerated).
QUALITY_PRESET = Stage1Config(denoise="nlmeans", nlmeans_h=12.0, median_ksize=3)
