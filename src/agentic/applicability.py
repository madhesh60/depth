"""
applicability.py — does the recall promise apply to THIS recording? (STUDY-19)

The recall promise is fit on the calibration recordings (``calibration.json``). It says nothing about a
sonar the detector has never seen: on the cross-sonar split the validation threshold keeps only 11% of
the pots (``docs/reports/eval_exp003_test_xsonar.md``) while the tiers would still print "≥ 79%". So
before the agent keeps a promise it checks that the frames look like the frames the promise was fit on.

* **Per frame** — eight OpenCV 5 image statistics of the sonogram as the detector sees it (brightness,
  contrast, speckle texture ``cv2.Laplacian``, along- vs across-track gradient ``cv2.Sobel``, dark and
  saturated pixel shares, colour saturation ``cv2.cvtColor`` → HSV, and the Bhattacharyya distance of its
  grey histogram ``cv2.calcHist`` / ``cv2.compareHist`` to the calibration recordings'). The frame's score
  is its largest robust z-score; its **conformal p-value** is its rank among the calibration frames' own
  scores, so a frame like them is called atypical (p ≤ ``alpha_frame``) at most ``alpha_frame`` of the time.
* **Per survey** — the promise is **withdrawn** when the 95% Clopper-Pearson lower bound of the atypical
  share exceeds ``share_limit`` (set from 13 in-distribution training recordings, STUDY-19). One odd frame
  never withdraws it; a different sonar does.

It is a coarse guard against a different sonar or rendering, not a fine-grained drift detector: the
report states what it catches. Fit and verified by ``study_applicability.py``; stored in
``models/<MODEL>/applicability.json``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from src.detection.calibration import DEFAULT_MODEL, MODELS_DIR
from .guarantees import cp_lower

FEATURES = ("mean", "std", "texture", "dark", "bright", "anisotropy", "saturation", "hist_dist")
# the smallest scale a feature's robust z-score divides by: on grayscale frames the saturation and the
# dark / bright shares are ~0 with ~0 spread, and a little JPEG colour noise must not look "atypical"
SCALE_FLOOR = {"mean": 0.01, "std": 0.005, "texture": 0.02, "dark": 0.01, "bright": 0.01,
               "anisotropy": 0.02, "saturation": 0.02, "hist_dist": 0.01}
SIDE = 320              # frames are compared at one size, so resolution alone never makes one atypical
BINS = 32


def frame_stats(bgr: np.ndarray) -> dict:
    """The per-frame OpenCV statistics (+ the normalised grey histogram, used against the reference)."""
    im = bgr if bgr.ndim == 3 else cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    im = cv2.resize(im, (SIDE, SIDE), interpolation=cv2.INTER_AREA)
    sat = float(cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[..., 1].mean() / 255.0)
    g8 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    g = g8.astype(np.float32)
    m = float(g.mean())
    sx = float(np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)).mean())
    sy = float(np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1)).mean())
    h = cv2.calcHist([g8], [0], None, [BINS], [0, 256]).ravel()
    return {"mean": m / 255.0, "std": float(g.std()) / 255.0,
            "texture": float(np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3)).mean()) / (m + 1.0),
            "dark": float((g8 < 16).mean()), "bright": float((g8 > 235).mean()),
            "anisotropy": float(np.log((sx + 1e-3) / (sy + 1e-3))), "saturation": sat,
            "hist": (h / max(float(h.sum()), 1.0)).astype(np.float32)}


class Reference:
    """The calibration recordings' statistics and the rule that compares a survey with them."""

    def __init__(self, data: dict):
        self.data = data
        self.med = np.asarray([data["median"][k] for k in FEATURES], float)
        self.scale = np.asarray([data["scale"][k] for k in FEATURES], float)
        self.hist0 = np.asarray(data["hist"], np.float32)
        self.cal_scores = np.sort(np.asarray(data["cal_scores"], float))
        self.alpha_frame = float(data["alpha_frame"])
        self.share_limit = float(data["share_limit"])
        self.delta = float(data.get("delta", 0.05))

    # -- fit ---------------------------------------------------------------------------------
    @classmethod
    def fit(cls, stats: Sequence[dict], alpha_frame: float = 0.05, share_limit: float = 0.15,
            delta: float = 0.05, **info) -> "Reference":
        hist0 = np.mean([s["hist"] for s in stats], axis=0).astype(np.float32)
        X = np.array([_vec(s, hist0) for s in stats])
        med = np.median(X, axis=0)
        mad = 1.4826 * np.median(np.abs(X - med), axis=0)
        scale = np.maximum(mad, [SCALE_FLOOR[k] for k in FEATURES])
        scores = np.max(np.abs((X - med) / scale), axis=1)
        return cls({"features": list(FEATURES), "median": dict(zip(FEATURES, np.round(med, 6).tolist())),
                    "scale": dict(zip(FEATURES, np.round(scale, 6).tolist())),
                    "hist": np.round(hist0, 6).tolist(), "cal_scores": np.round(np.sort(scores), 5).tolist(),
                    "alpha_frame": alpha_frame, "share_limit": share_limit, "delta": delta,
                    "n_cal": int(len(stats)), **info})

    # -- one frame ---------------------------------------------------------------------------
    def zscores(self, s: dict) -> np.ndarray:
        return (_vec(s, self.hist0) - self.med) / self.scale

    def pvalue(self, s: dict) -> tuple[float, str]:
        """Conformal p-value of a frame (and the feature that made it most unusual)."""
        z = np.abs(self.zscores(s))
        score = float(z.max())
        p = (1 + int(np.sum(self.cal_scores >= score))) / (len(self.cal_scores) + 1)
        return p, FEATURES[int(z.argmax())]

    # -- one survey --------------------------------------------------------------------------
    def check(self, stats: Sequence[dict]) -> dict:
        """The survey verdict: ``applies`` False withdraws the promise for this recording."""
        pv = [self.pvalue(s) for s in stats]
        n = len(pv)
        k = sum(p <= self.alpha_frame for p, _ in pv)
        lcb = cp_lower(k, n, self.delta) if n else 0.0
        applies = not (n and lcb > self.share_limit)
        drivers: dict[str, int] = {}
        for p, f in pv:
            if p <= self.alpha_frame:
                drivers[f] = drivers.get(f, 0) + 1
        top = sorted(drivers.items(), key=lambda kv: -kv[1])[:3]
        why = (f"{k} of {n} frame(s) look unlike the calibration recordings (p <= {self.alpha_frame:g}); the "
               f"95% lower bound of that share is {lcb:.0%} "
               + (f"> the {self.share_limit:.0%} limit -> the recall promise does NOT apply to this recording"
                  if not applies else f"<= the {self.share_limit:.0%} limit -> the promise applies"))
        if top and not applies:
            why += " (most unusual: " + ", ".join(f"{f} in {c}" for f, c in top) + ")"
        return {"applies": bool(applies), "atypical": int(k), "frames": int(n),
                "share": round(k / n, 3) if n else None, "share_lcb": round(lcb, 3),
                "share_limit": self.share_limit, "alpha_frame": self.alpha_frame,
                "drivers": dict(top), "why": why, "p_values": [round(p, 4) for p, _ in pv]}

    # -- storage -----------------------------------------------------------------------------
    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, model: Optional[str] = None) -> Optional["Reference"]:
        p = reference_path(model)
        if not p.exists():
            return None
        try:
            return cls(json.loads(p.read_text(encoding="utf-8")))
        except (KeyError, ValueError, TypeError):
            return None


def reference_path(model: Optional[str] = None) -> Path:
    return MODELS_DIR / (model or DEFAULT_MODEL) / "applicability.json"


def _vec(s: dict, hist0: np.ndarray) -> np.ndarray:
    d = cv2.compareHist(hist0, np.asarray(s["hist"], np.float32), cv2.HISTCMP_BHATTACHARYYA)
    return np.array([s["mean"], s["std"], s["texture"], s["dark"], s["bright"], s["anisotropy"],
                     s["saturation"], float(d)], float)
