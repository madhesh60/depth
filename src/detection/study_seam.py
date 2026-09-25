"""
study_seam.py — STUDY-11b: does seam inference recover the pots cut by chunk boundaries?

EXP-001 on its unseen crab-pot sonograms (v1 val Rec19 = calibration split, v1 test = verification
split; unique frames), detector floor 0.05, one-to-one matching at IoU ≥ 0.3:

* ``frame``  — the product today: the frame alone;
* ``seams``  — the frame + windows straddling its left/right chunk boundaries (``seam.py``) where the
  neighbouring chunk exists in the split, merged with class-aware NMS.

Reports recall ceiling (share of all labelled pots any candidate reaches), recall of **edge-touching**
pots, AP@0.3/0.5, candidates and inferences per frame, and a paired frame-bootstrap 95% CI of the
change. Frames without a neighbour are unchanged by construction (counted, and reported separately).

Caveat: the archive's train-origin frames are Roboflow copies; the neighbour used is the least-changed
copy, which can be cropped/zoomed — that can only *hide* a real gain (misaligned halves), not fake one.

    python -m src.detection.study_seam
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from .frames import unique_frames, frame_key
from .infer import YoloOnnxDetector, DEFAULT_ONNX
from .seam import chunk_key, seam_detections, merge

REPO = Path(__file__).resolve().parents[2]
V1 = REPO / "DATASET" / "03_yolo_ready_dataset_v1"
PAT = "crabpot_*wcp_ss_*.jpg"


def _gt(p: Path, W: int, H: int):
    lp = p.parent.parent / "labels" / f"{p.stem}.txt"
    out = []
    if lp.exists():
        for line in lp.read_text().splitlines():
            q = line.split()
            if len(q) >= 5 and int(float(q[0])) == 0:
                cx, cy, bw, bh = map(float, q[1:5])
                out.append((int((cx - bw / 2) * W), int((cy - bh / 2) * H), int((cx + bw / 2) * W), int((cy + bh / 2) * H)))
    return out


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _label(dets, gts, thr=0.3):
    used, pts = set(), []
    for d in sorted(dets, key=lambda d: -d.conf):
        best, bj = 0.0, -1
        for j, g in enumerate(gts):
            if j not in used and _iou(d.bbox, g) > best:
                best, bj = _iou(d.bbox, g), j
        if bj >= 0 and best >= thr:
            used.add(bj); pts.append((d.conf, 1))
        else:
            pts.append((d.conf, 0))
    return pts, used


def run_split(det, split: str) -> list[dict]:
    frames = unique_frames(sorted((V1 / split / "images").glob(PAT)))
    by_key = {}
    for p in frames:
        k = chunk_key(frame_key(p.name))
        if k:
            by_key[k] = p
    recs, t0 = [], time.time()
    for i, p in enumerate(frames):
        img = cv2.imread(str(p))
        if img is None:
            continue
        H, W = img.shape[:2]
        gts = _gt(p, W, H)
        k = chunk_key(frame_key(p.name))
        prev = by_key.get((k[0], k[1] - 1)) if k else None
        nxt = by_key.get((k[0], k[1] + 1)) if k else None
        fd = [d for d in det.detect(img) if d.cls_name == "fishing_gear"]
        sd, n_inf = seam_detections(det, img, cv2.imread(str(prev)) if prev else None, cv2.imread(str(nxt)) if nxt else None)
        sd = [d for d in sd if d.cls_name == "fishing_gear"]
        md = merge(fd, sd)
        edge = [j for j, g in enumerate(gts) if g[0] <= 3 or g[2] >= W - 3]
        rec = {"name": p.name, "n_gt": len(gts), "edge_gt": len(edge), "neighbours": int(prev is not None) + int(nxt is not None),
               "inf": {"frame": 1, "seams": 1 + n_inf}}
        for mode, dets in (("frame", fd), ("seams", md)):
            pts, used = _label(dets, gts)
            rec[mode] = {"pts": pts, "found": len(used), "edge_found": len(set(edge) & used), "cands": len(dets)}
        recs.append(rec)
        if (i + 1) % 30 == 0:
            print(f"  {split}: {i + 1}/{len(frames)} ({time.time() - t0:.0f}s)")
    return recs


def _ap(pts, n):
    from .evaluate import voc_ap
    return voc_ap(pts, n)


def summarise(recs: list[dict], reps: int = 1000, seed: int = 0) -> dict:
    def agg(rs, mode):
        n = sum(r["n_gt"] for r in rs); ne = sum(r["edge_gt"] for r in rs)
        pts = [x for r in rs for x in r[mode]["pts"]]
        return {"pots": n, "ceiling": sum(r[mode]["found"] for r in rs) / max(1, n),
                "edge_pots": ne, "edge_recall": sum(r[mode]["edge_found"] for r in rs) / max(1, ne),
                "ap30": _ap(pts, n), "cands_per_frame": sum(r[mode]["cands"] for r in rs) / max(1, len(rs)),
                "inf_per_frame": sum(r["inf"][mode] for r in rs) / max(1, len(rs))}
    out = {"frames": len(recs), "with_neighbour": sum(1 for r in recs if r["neighbours"]),
           "frame": agg(recs, "frame"), "seams": agg(recs, "seams")}
    rng = np.random.default_rng(seed)
    d_ceil, d_ap = [], []
    for _ in range(reps):
        idx = rng.integers(0, len(recs), len(recs))
        rs = [recs[i] for i in idx]
        a, b = agg(rs, "seams"), agg(rs, "frame")
        d_ceil.append(a["ceiling"] - b["ceiling"]); d_ap.append(a["ap30"] - b["ap30"])
    ci = lambda v: [round(float(np.mean(v)), 4), round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    out["delta_ceiling"], out["delta_ap30"] = ci(d_ceil), ci(d_ap)
    return out


def report(res: dict) -> str:
    L = ["# STUDY-11b — Seam inference across chunk boundaries", "",
         "_`python -m src.detection.study_seam` · EXP-001 via `cv2.dnn` · floor 0.05 · IoU ≥ 0.3 · unique frames · "
         "paired frame-bootstrap 95% CI_", "",
         "Why: STUDY-11a — 32% of the pots EXP-001 never proposes touch the frame edge (11% of found pots).", ""]
    for split, r in res.items():
        f, s = r["frame"], r["seams"]
        L += [f"## {split} — {r['frames']} frames ({r['with_neighbour']} with a neighbouring chunk in the split), {f['pots']} pots "
              f"({f['edge_pots']} touch the edge)", "",
              "| | frame only (today) | frame + seams | Δ (95% CI) |", "|---|--:|--:|:--:|",
              f"| recall ceiling (any candidate reaches the pot) | {f['ceiling']:.3f} | {s['ceiling']:.3f} | "
              f"{r['delta_ceiling'][0]:+.3f} ({r['delta_ceiling'][1]:+.3f} .. {r['delta_ceiling'][2]:+.3f}) |",
              f"| recall of edge-touching pots | {f['edge_recall']:.3f} | {s['edge_recall']:.3f} | |",
              f"| AP@0.3 | {f['ap30']:.3f} | {s['ap30']:.3f} | {r['delta_ap30'][0]:+.3f} ({r['delta_ap30'][1]:+.3f} .. {r['delta_ap30'][2]:+.3f}) |",
              f"| candidates / frame | {f['cands_per_frame']:.2f} | {s['cands_per_frame']:.2f} | |",
              f"| inferences / frame | {f['inf_per_frame']:.2f} | {s['inf_per_frame']:.2f} | (each seam is shared by 2 frames in a survey) |", ""]
    L += ["---", "_Regenerated by the script; do not edit by hand._", ""]
    return "\n".join(L)


def main():
    det = YoloOnnxDetector(DEFAULT_ONNX, conf_thres={"fishing_gear": 0.05})
    det.warmup()
    res = {}
    for split, name in (("val", "calibration split (v1 val, Rec19)"), ("test", "verification split (v1 test)")):
        print(f"{name} …")
        recs = run_split(det, split)
        res[name] = summarise(recs)
        print(json.dumps(res[name], indent=1))
    out = REPO / "runs" / "seam"; out.mkdir(parents=True, exist_ok=True)
    (out / "study.json").write_text(json.dumps(res, indent=1))
    (REPO / "docs" / "seam_inference.md").write_text(report(res), encoding="utf-8")
    print("wrote docs/seam_inference.md")


if __name__ == "__main__":
    main()
