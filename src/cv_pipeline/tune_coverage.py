"""
tune_coverage.py — Stage-1 recall/coverage tuning against ground truth.

Stage 1's geometry filters were originally set for COOL speed (few ROIs/frame), never
validated for *recall*. The FP-reduction ablation exposed the cost: only ~25% of GT boxes
land in any candidate ROI, so ROI-gating destroys recall. This script sweeps candidate
Stage-1 configs and reports, per config:

    coverage     — fraction of GT boxes covered by >=1 candidate ROI (the recall ceiling)
    avg_rois     — mean candidate ROIs/frame (compute cost of Stage 2 + FP exposure)
    ms/frame     — mean Stage-1 latency

Goal: pick the config with the best coverage/ROI trade-off (target coverage >= 0.90 at a
sane ROI count) to use as the Stage-1 default, then re-run ablation_fp.py.

Usage:
    python -m src.cv_pipeline.tune_coverage --limit 300
"""
from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO))

from src.cv_pipeline.config import Stage1Config          # noqa: E402
from src.cv_pipeline.pipeline import Stage1Pipeline       # noqa: E402
from src.detection.infer import _backed_by_roi            # noqa: E402

DEFAULT_DATA = REPO / "DATASET" / "03_yolo_ready_dataset_v1" / "data.yaml"

BASE = Stage1Config()

# Named candidate configs, from the current default to progressively looser geometry.
# Nets/rope are thin, wispy, elongated -> low solidity/extent, high aspect: loosen those.
CONFIGS: dict[str, Stage1Config] = {
    "baseline": BASE,
    "loose_geom": replace(BASE, min_solidity=0.05, min_extent=0.05, max_aspect=60.0),
    "loose+small": replace(BASE, min_solidity=0.05, min_extent=0.05, max_aspect=60.0,
                           min_area_frac=1e-4),
    "loose+close": replace(BASE, min_solidity=0.05, min_extent=0.05, max_aspect=60.0,
                           min_area_frac=1e-4, morph_op="close"),
    "ceiling": replace(BASE, min_solidity=0.0, min_extent=0.0, max_aspect=1e9,
                       min_area_frac=5e-5, max_area_frac=0.9, max_candidates=200),
}


def load_gt(label_path: Path, w: int, h: int):
    gt = []
    if not label_path.exists():
        return gt
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cls = int(float(p[0]))
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        gt.append((cls, (int((cx-bw/2)*w), int((cy-bh/2)*h),
                         int((cx+bw/2)*w), int((cy+bh/2)*h))))
    return gt


def main():
    ap = argparse.ArgumentParser(description="Sweep Stage-1 configs for GT coverage.")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--gate-iou", type=float, default=0.10)
    args = ap.parse_args()

    data = yaml.safe_load(Path(args.data).read_text())
    names = data["names"]
    root = Path(args.data).parent
    img_dir = root / data[args.split]
    lbl_dir = Path(str(img_dir).replace("images", "labels"))
    imgs = sorted(q for q in img_dir.rglob("*") if q.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit and args.limit < len(imgs):
        step = len(imgs) / args.limit
        imgs = [imgs[int(i*step)] for i in range(args.limit)]

    # cache frames + GT once
    frames = []
    for ip in imgs:
        im = cv2.imread(str(ip))
        if im is None:
            continue
        h, w = im.shape[:2]
        frames.append((im, load_gt(lbl_dir / (ip.stem + ".txt"), w, h)))
    nc = len(names)
    print(f"{len(frames)} frames from {args.split}\n")

    hdr = f"{'config':<14}{'coverage':>10}{'avg_rois':>10}{'ms/frame':>10}   per-class coverage"
    print(hdr); print("-" * len(hdr))
    for cname, cfg in CONFIGS.items():
        pipe = Stage1Pipeline(cfg)
        cov = tot = nroi = 0
        cov_c = [0]*nc; tot_c = [0]*nc
        ms = []
        for im, gt in frames:
            t0 = time.perf_counter()
            res = pipe.process(im)
            ms.append((time.perf_counter()-t0)*1000)
            rois = [c.bbox for c in res.candidates]
            nroi += len(rois)
            for gcls, gbox in gt:
                tot += 1; tot_c[gcls] += 1
                if _backed_by_roi(gbox, rois, args.gate_iou):
                    cov += 1; cov_c[gcls] += 1
        coverage = cov/tot if tot else 0.0
        pc = " ".join(f"{names[c][:4]}:{(cov_c[c]/tot_c[c] if tot_c[c] else 0):.2f}"
                      for c in range(nc))
        print(f"{cname:<14}{coverage:>10.3f}{nroi/len(frames):>10.1f}"
              f"{np.mean(ms):>10.1f}   {pc}")


if __name__ == "__main__":
    main()
