"""
ablation_fp.py — Stage-1 false-positive-reduction ablation (award-critical).

Measures the value of the two-stage architecture by comparing the *same* Stage-2 detector
in two modes over the labelled test split:

    full_frame  — detector on the whole frame (baseline)
    roi_guided  — detector, then gated to Stage-1 candidate ROIs

For each mode we match detections to ground truth (same class, IoU >= --match-iou, greedy)
and tally TP / FP / FN. The headline numbers:

    FP reduction  = (FP_full - FP_roi) / FP_full          (target: >= 60%)
    recall kept   = recall_roi / recall_full              (want ~1.0 — gating shouldn't cost TPs)

Also reports per-class breakdown and mean Stage-1 latency. Writes a CSV summary.

Usage:
    python src/detection/ablation_fp.py --limit 1000
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO))

from src.detection.infer import YoloOnnxDetector, _iou, DEFAULT_ONNX  # noqa: E402
from src.cv_pipeline.pipeline import Stage1Pipeline  # noqa: E402

DEFAULT_DATA = REPO / "DATASET" / "03_yolo_ready_dataset_v1" / "data.yaml"


def load_gt(label_path: Path, w: int, h: int) -> list[tuple[int, tuple[int, int, int, int]]]:
    """Read a YOLO .txt label -> [(cls, (x1,y1,x2,y2) px), ...]."""
    gt = []
    if not label_path.exists():
        return gt
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
        gt.append((cls, (x1, y1, x2, y2)))
    return gt


def match(dets, gt, match_iou: float) -> tuple[int, int]:
    """Greedy per-class IoU matching. Returns (TP, FP). FN = len(gt) - TP."""
    used = [False] * len(gt)
    tp = 0
    for d in sorted(dets, key=lambda x: x.conf, reverse=True):
        best_j, best_iou = -1, match_iou
        for j, (gcls, gbox) in enumerate(gt):
            if used[j] or gcls != d.cls_id:
                continue
            iou = _iou(d.bbox, gbox)
            if iou >= best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0:
            used[best_j] = True
            tp += 1
    return tp, len(dets) - tp


class Tally:
    __slots__ = ("tp", "fp", "n_gt")

    def __init__(self):
        self.tp = self.fp = self.n_gt = 0

    def add(self, tp, fp, n_gt):
        self.tp += tp; self.fp += fp; self.n_gt += n_gt

    @property
    def precision(self):
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self):
        return self.tp / self.n_gt if self.n_gt else 0.0


def main():
    p = argparse.ArgumentParser(description="Stage-1 FP-reduction ablation.")
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--onnx", default=str(DEFAULT_ONNX))
    p.add_argument("--split", default="test")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45, help="detector NMS IoU")
    p.add_argument("--match-iou", type=float, default=0.5, help="TP match IoU vs GT")
    p.add_argument("--gate-iou", type=float, default=0.10, help="ROI gating IoU")
    p.add_argument("--limit", type=int, default=1000, help="max images (0 = all)")
    p.add_argument("--out", default=str(REPO / "runs" / "ablation_fp.csv"))
    args = p.parse_args()

    data = yaml.safe_load(Path(args.data).read_text())
    names = data["names"]
    root = Path(args.data).parent
    img_dir = root / data[args.split]
    lbl_dir = Path(str(img_dir).replace("images", "labels"))
    imgs = sorted(q for q in img_dir.rglob("*") if q.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit and args.limit < len(imgs):
        # evenly stride across the whole split so all sources/classes are represented
        step = len(imgs) / args.limit
        imgs = [imgs[int(i * step)] for i in range(args.limit)]
    print(f"{len(imgs)} images from {args.split} split")

    det = YoloOnnxDetector(args.onnx, names=names, conf_thres=args.conf, iou_thres=args.iou)
    stage1 = Stage1Pipeline()

    nc = len(names)
    full, roi = Tally(), Tally()
    full_c = [Tally() for _ in range(nc)]
    roi_c = [Tally() for _ in range(nc)]
    s1_ms = []
    cov_gt = tot_gt = n_roi = 0

    for k, ip in enumerate(imgs):
        img = cv2.imread(str(ip))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = load_gt(lbl_dir / (ip.stem + ".txt"), w, h)

        dets_full = det.detect(img)
        t0 = time.perf_counter()
        s1 = stage1.process(img)
        s1_ms.append((time.perf_counter() - t0) * 1000)
        rois = [c.bbox for c in s1.candidates]
        n_roi += len(rois)
        # Stage-1 GT coverage (recall ceiling for roi_guided): is each GT box in some ROI?
        for _gcls, gbox in gt:
            tot_gt += 1
            if _backed(gbox, rois, args.gate_iou):
                cov_gt += 1
        dets_roi = [d for d in dets_full if _backed(d.bbox, rois, args.gate_iou)]

        for dets, T, Tc in ((dets_full, full, full_c), (dets_roi, roi, roi_c)):
            tp, fp = match(dets, gt, args.match_iou)
            T.add(tp, fp, len(gt))
            for c in range(nc):
                d_c = [d for d in dets if d.cls_id == c]
                g_c = [g for g in gt if g[0] == c]
                tpc, fpc = match(d_c, g_c, args.match_iou)
                Tc[c].add(tpc, fpc, len(g_c))

        if (k + 1) % 200 == 0:
            print(f"  {k+1}/{len(imgs)}...")

    fp_red = (full.fp - roi.fp) / full.fp if full.fp else 0.0
    rec_keep = roi.recall / full.recall if full.recall else 0.0

    s1_cov = cov_gt / tot_gt if tot_gt else 0.0
    print("\n================ Stage-1 FP-reduction ablation ================")
    print(f"images: {len(imgs)}   mean Stage-1: {np.mean(s1_ms):.1f} ms/frame   "
          f"avg ROIs/frame: {n_roi/max(1,len(imgs)):.1f}")
    print(f"Stage-1 GT coverage: {s1_cov*100:5.1f}%  "
          f"({cov_gt}/{tot_gt} GT boxes in an ROI — recall ceiling for roi_guided)")
    print(f"{'mode':<12}{'TP':>7}{'FP':>7}{'P':>8}{'R':>8}")
    for nm, T in (("full_frame", full), ("roi_guided", roi)):
        print(f"{nm:<12}{T.tp:>7}{T.fp:>7}{T.precision:>8.3f}{T.recall:>8.3f}")
    print(f"\nFP reduction : {fp_red*100:5.1f}%   (target >= 60%)")
    print(f"recall kept  : {rec_keep*100:5.1f}%   (want ~100%)")
    print("\nper-class FP reduction / recall kept:")
    rows = []
    for c in range(nc):
        f, r = full_c[c], roi_c[c]
        cfp = (f.fp - r.fp) / f.fp if f.fp else 0.0
        crk = r.recall / f.recall if f.recall else 0.0
        print(f"  {names[c]:<20} FP {f.fp:>5}->{r.fp:<5} ({cfp*100:5.1f}%)   "
              f"R {f.recall:.3f}->{r.recall:.3f} ({crk*100:5.1f}%)")
        rows.append([names[c], f.tp, f.fp, r.tp, r.fp, round(cfp, 4), round(crk, 4)])

    with open(args.out, "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["class", "full_tp", "full_fp", "roi_tp", "roi_fp", "fp_reduction", "recall_kept"])
        wtr.writerow(["ALL", full.tp, full.fp, roi.tp, roi.fp, round(fp_red, 4), round(rec_keep, 4)])
        wtr.writerows(rows)
    print(f"\nCSV -> {args.out}")


def _backed(box, rois, gate_iou):
    from src.detection.infer import _backed_by_roi
    return _backed_by_roi(box, rois, gate_iou)


if __name__ == "__main__":
    main()
