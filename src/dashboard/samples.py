"""
samples.py — curated sample sonar frames so a judge sees results in one click (no sonar files needed).

Frames are picked deterministically from the local dataset at first use (``SAMPLES_DIR`` env or the
v1 test split). It degrades gracefully: if the dataset is absent (e.g. a fresh deploy before the S3
sync), ``list_samples()`` returns ``[]`` and the dashboard still works for uploads.
"""
from __future__ import annotations

import glob
import os
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SAMPLES_DIR = Path(os.environ.get(
    "SAMPLES_DIR", REPO / "DATASET" / "03_yolo_ready_dataset_v1" / "test" / "images"))
_LABELS_DIR = SAMPLES_DIR.parent / "labels"


def _has_gt(stem: str) -> bool:
    lp = _LABELS_DIR / f"{stem}.txt"
    return lp.exists() and lp.stat().st_size > 0


@lru_cache(maxsize=1)
def _catalog() -> list[dict]:
    """Deterministic list: a few labelled crab-pot frames + a couple with no labelled pot."""
    if not SAMPLES_DIR.exists():
        return []
    wcp = sorted(glob.glob(str(SAMPLES_DIR / "crabpot_*wcp_ss_*.jpg")))
    with_gt = [p for p in wcp if _has_gt(Path(p).stem)][:4]
    without = [p for p in wcp if not _has_gt(Path(p).stem)][:2]
    picks = with_gt + without
    out = []
    for i, p in enumerate(picks, 1):
        stem = Path(p).stem
        side = "port" if "_ss_port" in stem else ("starboard" if "_ss_star" in stem else "?")
        out.append({
            "id": f"sample-{i:02d}",
            "name": f"Rec6 {side} #{i}",
            "kind": "crab-pot sonar" if p in with_gt else "seabed (no labelled pot)",
            "has_gps": False,                        # the HF crab-pot frames carry no GPS
            "path": p,
        })
    return out


def list_samples() -> list[dict]:
    """Public sample list (without the on-disk path)."""
    return [{k: v for k, v in s.items() if k != "path"} for s in _catalog()]


def sample_path(sample_id: str) -> Path | None:
    for s in _catalog():
        if s["id"] == sample_id:
            return Path(s["path"])
    return None
