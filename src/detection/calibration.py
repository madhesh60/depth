"""
calibration.py — the ONE place every runtime threshold comes from.

Before this module the thresholds lived in four places (``infer.PER_CLASS_CONF``, the val-tuned
values in ``evaluate.py``, ``agentic/policy.py`` and ``infra/lambda_handler._CONF``) and could
drift. Now each model ships a small, versioned ``models/<MODEL>/calibration.json`` that is
**written by the calibration script on VALIDATION data** (``python -m src.agentic.calibrate``) and
**read at startup** by every consumer (detector, agent, dashboard, Lambda, benchmark).

Schema (all keys optional — anything missing falls back to :data:`DEFAULTS`)::

    {
      "model": "EXP-001",
      "names": ["fishing_gear", ...],
      "detector": {"imgsz": 640, "iou_nms": 0.45, "class_aware_nms": true,
                   "conf": {"fishing_gear": 0.10, ...}, "relook_conf": 0.05},
      "tiers": {"method": "...", "score": "...", "tau_review": 0.., "tau_confirm": 0..,
                "guarantees": {...}},
      "fit": {"split": "...", "frames": .., "created": "..."}
    }

The active model is ``$DEPTH_MODEL`` (default ``EXP-001``); a different file can be forced with
``$DEPTH_CALIBRATION``.
"""
from __future__ import annotations

import copy
import json
import os
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO / "models"
DEFAULT_MODEL = os.environ.get("DEPTH_MODEL", "EXP-001")

# Built-in fallback == the values the product shipped with before calibration.json existed, so a
# missing/partial file can never change behaviour silently.
DEFAULTS: dict[str, Any] = {
    "model": DEFAULT_MODEL,
    "names": ["fishing_gear", "pipe_cylinder", "structural_fragment", "natural_formation"],
    "detector": {
        "imgsz": 640,
        "iou_nms": 0.45,
        "class_aware_nms": True,
        "conf": {"fishing_gear": 0.10, "pipe_cylinder": 0.25,
                 "structural_fragment": 0.25, "natural_formation": 0.25},
        "relook_conf": 0.05,
    },
    "tiers": {
        "method": "legacy-rules",
        "score": "max(det_conf, relook_conf)",
        "tau_review": None,
        "tau_confirm": None,
        "guarantees": {},
    },
    "fit": {"split": None, "frames": 0, "created": None},
}


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def calibration_path(model: Optional[str] = None) -> Path:
    forced = os.environ.get("DEPTH_CALIBRATION")
    if forced:
        return Path(forced)
    return MODELS_DIR / (model or DEFAULT_MODEL) / "calibration.json"


class Calibration:
    """Read-only view over a calibration dict with typed accessors."""

    def __init__(self, data: dict, path: Optional[Path] = None):
        self.data = data
        self.path = path

    # -- detector --------------------------------------------------------------------------
    @property
    def names(self) -> list[str]:
        return list(self.data["names"])

    @property
    def conf(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.data["detector"]["conf"].items()}

    @property
    def iou_nms(self) -> float:
        return float(self.data["detector"]["iou_nms"])

    @property
    def class_aware_nms(self) -> bool:
        return bool(self.data["detector"]["class_aware_nms"])

    @property
    def relook_conf(self) -> float:
        return float(self.data["detector"]["relook_conf"])

    @property
    def imgsz(self) -> int:
        return int(self.data["detector"]["imgsz"])

    # -- decision tiers --------------------------------------------------------------------
    @property
    def tiers(self) -> dict:
        return dict(self.data["tiers"])

    @property
    def tau_review(self) -> Optional[float]:
        v = self.data["tiers"].get("tau_review")
        return None if v is None else float(v)

    @property
    def tau_confirm(self) -> Optional[float]:
        v = self.data["tiers"].get("tau_confirm")
        return None if v is None else float(v)

    @property
    def guarantees(self) -> dict:
        return dict(self.data["tiers"].get("guarantees") or {})

    @property
    def is_fitted(self) -> bool:
        return self.tau_review is not None or self.tau_confirm is not None

    def to_dict(self) -> dict:
        return copy.deepcopy(self.data)

    def summary(self) -> dict:
        """Small public summary for /api/health and the UI badges."""
        t = self.data["tiers"]
        return {
            "model": self.data.get("model"),
            "method": t.get("method"),
            "policy": t.get("policy"),
            "tau_review": t.get("tau_review"),
            "tau_confirm": t.get("tau_confirm"),
            "guarantees": t.get("guarantees") or {},
            "fit": self.data.get("fit") or {},
            "source": self.path.relative_to(REPO).as_posix()
            if self.path and self.path.is_relative_to(REPO) else (str(self.path) if self.path else "defaults"),
        }


@lru_cache(maxsize=8)
def _load_cached(path_str: str, mtime: float) -> dict:
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


def load_calibration(model: Optional[str] = None, path: Optional[str | Path] = None) -> Calibration:
    """Load ``calibration.json`` for ``model`` (merged over :data:`DEFAULTS`). Never raises for a
    missing file — it returns the defaults so a fresh checkout still runs."""
    p = Path(path) if path else calibration_path(model)
    if p.exists():
        data = _load_cached(str(p), p.stat().st_mtime)
        return Calibration(_deep_merge(DEFAULTS, data), p)
    return Calibration(copy.deepcopy(DEFAULTS), None)


def save_calibration(data: dict, model: Optional[str] = None, path: Optional[str | Path] = None) -> Path:
    """Write a calibration dict (merged over the current file so partial updates are safe)."""
    p = Path(path) if path else calibration_path(model)
    current = load_calibration(model, p if p.exists() else None).to_dict() if p.exists() else DEFAULTS
    merged = _deep_merge(current, data)
    merged.setdefault("fit", {})["created"] = merged.get("fit", {}).get("created") or date.today().isoformat()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    _load_cached.cache_clear()
    return p
