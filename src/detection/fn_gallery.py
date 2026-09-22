"""
fn_gallery.py — false-negative gallery for a target class (default fishing_gear).

EXP-001's confusion matrix shows 63% of true fishing_gear boxes are predicted as
*background* (missed outright, not misclassified). This tool renders those missed GT boxes
so the visual cause (thin/low-contrast/small/occluded) can be characterised and fed into the
next experiment. For each test frame with an unmatched GT box of the target class it writes
the frame with the missed GT (red) and all detections (yellow), plus a tight crop.

Usage:
    python -m src.detection.fn_gallery --cls fishing_gear --limit 40
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO))

from src.detection.infer import YoloOnnxDetector, _iou, DEFAULT_ONNX  # noqa: E402
from src.detection.ablation_fp import load_gt                          # noqa: E402

DEFAULT_DATA = REPO / "DATASET" / "03_yolo_ready_dataset_v1" / "data.yaml"


def main():
    ap = argparse.ArgumentParser(description="Render false-negative examples for a class.")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--onnx", default=str(DEFAULT_ONNX))
    ap.add_argument("--split", default="test")
    ap.add_argument("--cls", default="fishing_gear")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--out", default=str(REPO / "runs" / "fn_gallery"))
    args = ap.parse_args()

    data = yaml.safe_load(Path(args.data).read_text())
    names = data["names"]
    cls_id = names.index(args.cls)
    root = Path(args.data).parent
    img_dir = root / data[args.split]
    lbl_dir = Path(str(img_dir).replace("images", "labels"))
    imgs = sorted(q for q in img_dir.rglob("*") if q.suffix.lower() in {".jpg", ".jpeg", ".png"})

    det = YoloOnnxDetector(args.onnx, names=names, conf_thres=args.conf)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    n_missed = n_total = n_saved = 0
    sizes = []
    for ip in imgs:
        lp = lbl_dir / (ip.stem + ".txt")
        if not lp.exists() or f"{cls_id} " not in " " + lp.read_text().replace("\n", " "):
            continue
        img = cv2.imread(str(ip))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt_px = [g for g in load_gt(lp, w, h) if g[0] == cls_id]
        if not gt_px:
            continue
        dets = [d for d in det.detect(img) if d.cls_id == cls_id]
        vis = img.copy()
        frame_has_miss = False
        for _, gb in gt_px:
            n_total += 1
            matched = any(_iou(gb, d.bbox) >= args.match_iou for d in dets)
            color = (0, 255, 0) if matched else (0, 0, 255)   # green hit, red miss
            cv2.rectangle(vis, gb[:2], gb[2:], color, 2)
            if not matched:
                n_missed += 1
                frame_has_miss = True
                bw, bh = gb[2]-gb[0], gb[3]-gb[1]
                sizes.append(max(bw, bh) / max(w, h))          # long-side frac of frame
        for d in dets:
            cv2.rectangle(vis, d.bbox[:2], d.bbox[2:], (0, 200, 255), 1)
        if frame_has_miss and n_saved < args.limit:
            cv2.imwrite(str(out_dir / f"miss_{ip.stem}.jpg"), vis)
            n_saved += 1

    print(f"class '{args.cls}': {n_missed}/{n_total} GT boxes missed "
          f"({100*n_missed/max(1,n_total):.1f}%)")
    if sizes:
        s = np.array(sizes)
        print(f"missed-box long side (frac of frame): "
              f"median {np.median(s):.3f}  p25 {np.percentile(s,25):.3f}  "
              f"p75 {np.percentile(s,75):.3f}  min {s.min():.3f}  max {s.max():.3f}")
        print(f"  tiny (<0.05): {100*np.mean(s<0.05):.0f}%   small (<0.10): {100*np.mean(s<0.10):.0f}%")
    print(f"{n_saved} frames -> {out_dir}")


if __name__ == "__main__":
    main()
