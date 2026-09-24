"""
feedback.py — every human decision becomes a training label (review A-2: "corrections become labels").

The agent is human-gated: REVIEW cards are approved or dismissed, and a person can mark a pot the
detector **missed**. Those decisions are the most valuable labels a survey produces — they sit
exactly where the model is uncertain or blind — so DEPTH keeps them:

* ``decisions.jsonl`` — append-only log, one line per decision:
  ``{id, ts, frame_id, frame_ref, bbox, cls_name, decision, verdict_before, conf, p_pot, source,
  annotator}`` with ``decision ∈ {confirm, reject, missed}`` and ``frame_ref`` = ``{"sample": id}``
  or ``{"upload": sha256}`` (uploaded frames are persisted to ``frames/<sha>.jpg`` only when someone
  gives feedback on them);
* **latest decision wins** per object (same frame, IoU ≥ 0.7);
* ``export()`` → a YOLO fine-tune set: positives = confirmed + missed boxes; explicitly rejected boxes
  are written to ``hard_negatives.json`` (the next training run can weight / mine them); frames where
  the human only saw some candidates are flagged ``partial`` in the manifest, never silently mixed in;
* ``agreement()`` — where ground truth exists (the shipped samples), how often reviewers agree with
  it: reviewer precision/recall is the ceiling on what the human-in-the-loop can deliver.

    python -m src.agentic.feedback stats
    python -m src.agentic.feedback export --out DATASET/feedback_v1
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DECISIONS = ("confirm", "reject", "missed")
SOURCES = ("analyze", "survey", "study", "audit")


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


class FeedbackStore:
    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root or os.environ.get("DEPTH_FEEDBACK_DIR", REPO / "runs" / "feedback"))
        self.log = self.root / "decisions.jsonl"
        self.frames = self.root / "frames"
        self._lock = threading.Lock()

    # -- write -----------------------------------------------------------------------------
    def add(self, rec: dict, image: Optional[np.ndarray] = None) -> dict:
        """Validate + append one decision. ``image`` is persisted for uploaded frames."""
        d = str(rec.get("decision", ""))
        if d not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}")
        bbox = [int(round(float(v))) for v in rec.get("bbox", [])]
        if len(bbox) != 4 or bbox[2] - bbox[0] < 2 or bbox[3] - bbox[1] < 2:
            raise ValueError("bbox must be [x1, y1, x2, y2] with a positive size")
        ref = rec.get("frame_ref") or {}
        if not (ref.get("sample") or ref.get("upload")):
            raise ValueError("frame_ref needs 'sample' or 'upload'")
        out = {
            "id": uuid.uuid4().hex[:12], "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "frame_id": str(rec.get("frame_id", ""))[:200], "frame_ref": {k: str(v)[:80] for k, v in ref.items()},
            "bbox": bbox, "cls_name": str(rec.get("cls_name") or "")[:40], "decision": d,
            "verdict_before": rec.get("verdict_before"), "conf": rec.get("conf"), "p_pot": rec.get("p_pot"),
            "source": rec.get("source") if rec.get("source") in SOURCES else "analyze",
            "annotator": str(rec.get("annotator") or "anon")[:40],
        }
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            if image is not None and ref.get("upload"):
                self.frames.mkdir(parents=True, exist_ok=True)
                fp = self.frames / f"{out['frame_ref']['upload']}.jpg"
                if not fp.exists():
                    cv2.imwrite(str(fp), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(self.log, "a", encoding="utf-8") as f:
                f.write(json.dumps(out) + "\n")
        return out

    # -- read ------------------------------------------------------------------------------
    def all(self) -> list[dict]:
        if not self.log.exists():
            return []
        out = []
        for line in self.log.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def latest(self) -> list[dict]:
        """One decision per object: the most recent one (same frame + IoU ≥ 0.7 ⇒ same object)."""
        kept: list[dict] = []
        for r in self.all():                            # chronological
            key = json.dumps(r["frame_ref"], sort_keys=True)
            for i, k in enumerate(kept):
                if json.dumps(k["frame_ref"], sort_keys=True) == key and _iou(k["bbox"], r["bbox"]) >= 0.7:
                    kept[i] = r
                    break
            else:
                kept.append(r)
        return kept

    def stats(self) -> dict:
        lat = self.latest()
        c = Counter(r["decision"] for r in lat)
        return {"decisions": len(self.all()), "objects": len(lat),
                "frames": len({json.dumps(r["frame_ref"], sort_keys=True) for r in lat}),
                "confirm": c.get("confirm", 0), "reject": c.get("reject", 0), "missed": c.get("missed", 0),
                "by_source": dict(Counter(r["source"] for r in lat))}

    # -- image resolution ------------------------------------------------------------------
    def image_path(self, ref: dict) -> Optional[Path]:
        if ref.get("upload"):
            p = self.frames / f"{ref['upload']}.jpg"
            return p if p.exists() else None
        if ref.get("sample"):
            from src.dashboard import samples as samples_mod
            return samples_mod.sample_path(ref["sample"])
        return None

    # -- export → YOLO fine-tune set ---------------------------------------------------------
    def export(self, out: Path, names: list[str], default_cls: Optional[str] = None) -> dict:
        out = Path(out)
        (out / "images").mkdir(parents=True, exist_ok=True)
        (out / "labels").mkdir(parents=True, exist_ok=True)
        default_cls = default_cls or names[0]
        by_frame: dict[str, list[dict]] = {}
        for r in self.latest():
            by_frame.setdefault(json.dumps(r["frame_ref"], sort_keys=True), []).append(r)
        manifest, hard_neg, n_pos = [], [], 0
        for key, recs in sorted(by_frame.items()):
            ref = json.loads(key)
            ip = self.image_path(ref)
            im = cv2.imread(str(ip)) if ip else None
            if im is None:
                continue
            H, W = im.shape[:2]
            stem = ("s_" + ref["sample"]) if ref.get("sample") else ("u_" + ref["upload"][:16])
            cv2.imwrite(str(out / "images" / f"{stem}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
            lines = []
            for r in recs:
                x1, y1, x2, y2 = r["bbox"]
                if r["decision"] == "reject":
                    hard_neg.append({"image": f"{stem}.jpg", "bbox": r["bbox"], "cls_name": r["cls_name"]})
                    continue
                cls = r["cls_name"] if r["cls_name"] in names else default_cls
                lines.append(f"{names.index(cls)} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                             f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
            (out / "labels" / f"{stem}.txt").write_text("\n".join(lines))
            n_pos += len(lines)
            manifest.append({"image": f"{stem}.jpg", "frame_id": recs[0]["frame_id"], "frame_ref": ref,
                             "positives": len(lines), "rejected": sum(r["decision"] == "reject" for r in recs),
                             "missed_added": sum(r["decision"] == "missed" for r in recs),
                             "partial": True})       # only reviewed objects are labelled — see docstring
        (out / "hard_negatives.json").write_text(json.dumps(hard_neg, indent=1))
        (out / "data.yaml").write_text("# DEPTH human-feedback fine-tune set (src/agentic/feedback.py)\n"
                                       "train: images\nval: images\n"
                                       f"nc: {len(names)}\nnames:\n" + "".join(f"- {n}\n" for n in names),
                                       encoding="utf-8")
        summary = {"frames": len(manifest), "positives": n_pos, "hard_negatives": len(hard_neg),
                   "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "frames_detail": manifest}
        (out / "manifest.json").write_text(json.dumps(summary, indent=1))
        return summary

    # -- reviewer agreement with ground truth (shipped samples carry labels) -----------------------
    def agreement(self, iou: float = 0.3) -> dict:
        from src.dashboard import samples as samples_mod
        tp = fp = fn_conf = n = 0
        for r in self.latest():
            sid = r["frame_ref"].get("sample")
            gts = samples_mod.sample_gt(sid) if sid else None
            if gts is None:
                continue
            n += 1
            real = any(_iou(r["bbox"], g) >= iou for g in gts)
            if r["decision"] in ("confirm", "missed"):
                tp += real; fp += not real
            elif r["decision"] == "reject" and real:
                fn_conf += 1                            # a real pot the reviewer dismissed
        return {"judged_on_gt": n, "confirmed_real": tp, "confirmed_not_in_gt": fp,
                "dismissed_real": fn_conf,
                "reviewer_precision": round(tp / (tp + fp), 3) if tp + fp else None}


def main():
    ap = argparse.ArgumentParser(description="DEPTH human-feedback labels")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats")
    e = sub.add_parser("export")
    e.add_argument("--out", default=str(REPO / "DATASET" / "feedback_v1"))
    a = ap.parse_args()
    st = FeedbackStore()
    if a.cmd == "stats":
        print(json.dumps({**st.stats(), "agreement_with_gt": st.agreement()}, indent=1))
    else:
        from src.detection.calibration import load_calibration
        s = st.export(Path(a.out), load_calibration().names, load_calibration().guaranteed_class)
        print(json.dumps({k: v for k, v in s.items() if k != "frames_detail"}, indent=1), "->", a.out)


if __name__ == "__main__":
    main()
