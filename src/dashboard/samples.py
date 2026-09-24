"""
samples.py — curated sample sonar frames so a judge sees results in one click (no sonar files needed).

Order of preference:
1. **Shipped samples** in ``webui/samples/`` (``manifest.json``) — 8 crab-pot sonograms EXP-001 never
   trained on, one un-rotated copy each, consecutive chunks so stitching shows up, CC-BY-SA-4.0 with
   ``ATTRIBUTION.md``. They make a fresh server (no ``DATASET/``) fully demo-able.
2. ``$SAMPLES_DIR`` or the local v1 test split (developer fallback).

``list_samples()`` never raises: with neither source it returns ``[]`` and uploads still work.
"""
from __future__ import annotations

import glob
import json
import os
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SHIPPED = REPO / "webui" / "samples"
SAMPLES_DIR = Path(os.environ.get(
    "SAMPLES_DIR", REPO / "DATASET" / "03_yolo_ready_dataset_v1" / "test" / "images"))
_LABELS_DIR = SAMPLES_DIR.parent / "labels"


def _has_gt(stem: str) -> bool:
    lp = _LABELS_DIR / f"{stem}.txt"
    return lp.exists() and lp.stat().st_size > 0


def _shipped() -> list[dict]:
    mf = SHIPPED / "manifest.json"
    if not mf.exists():
        return []
    data = json.loads(mf.read_text(encoding="utf-8"))
    out = []
    for s in data.get("samples", []):
        p = SHIPPED / s["file"]
        if p.exists():
            out.append({"id": s["id"], "name": s["name"], "kind": s["kind"],
                        "has_gps": bool(s.get("has_gps")), "license": data.get("license", ""),
                        "attribution": "samples/ATTRIBUTION.md", "path": str(p)})
    return out


def _from_dataset() -> list[dict]:
    if not SAMPLES_DIR.exists():
        return []
    wcp = sorted(glob.glob(str(SAMPLES_DIR / "crabpot_*wcp_ss_*.jpg")))
    with_gt = [p for p in wcp if _has_gt(Path(p).stem)][:4]
    without = [p for p in wcp if not _has_gt(Path(p).stem)][:2]
    out = []
    for i, p in enumerate(with_gt + without, 1):
        stem = Path(p).stem
        side = "port" if "_ss_port" in stem else ("starboard" if "_ss_star" in stem else "?")
        out.append({"id": f"sample-{i:02d}", "name": f"local {side} #{i}",
                    "kind": "crab-pot sonar" if p in with_gt else "seabed (no labelled pot)",
                    "has_gps": False, "license": "CC-BY-SA-4.0", "attribution": "", "path": p})
    return out


@lru_cache(maxsize=1)
def _catalog() -> list[dict]:
    return _shipped() or _from_dataset()


def list_samples() -> list[dict]:
    """Public sample list (without the on-disk path)."""
    return [{k: v for k, v in s.items() if k != "path"} for s in _catalog()]


def sample_path(sample_id: str) -> Path | None:
    for s in _catalog():
        if s["id"] == sample_id:
            return Path(s["path"])
    return None


def frame_id(sample_id: str) -> str | None:
    """The frame id the pipeline sees (drives the orientation source rule + chunk stitching)."""
    p = sample_path(sample_id)
    return p.stem if p else None
