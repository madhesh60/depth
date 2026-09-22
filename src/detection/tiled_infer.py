"""
tiled_infer.py — SAHI-style tiled inference for small-object recall (strong OpenCV, no retrain).

EXP-001 error analysis: 91% of missed fishing_gear boxes are <10% of frame width — a
small-object problem. Frames are ~640x640 and the detector runs at 640, so a crab-pot
(~38 px) is tiny. Tiled inference slices each frame into an overlapping grid, upscales every
tile back to the detector's input size, and merges the per-tile detections — so a small
object occupies ~grid× more pixels of the detector input. This recovers small-object recall
from the *existing* weights, and the slice/resize/merge work is exactly the CPU (COOL-
accelerated) workload that runs on Graviton for the benchmark.

Pure OpenCV + NumPy on top of the cv2.dnn detector in infer.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO))

from src.detection.infer import (YoloOnnxDetector, Detection, PER_CLASS_CONF,  # noqa: E402
                                 DEFAULT_ONNX, draw_detections)


def _tiles(h: int, w: int, grid: int, overlap: float):
    """Yield (x0, y0, x1, y1) tile boxes for a grid×grid split with fractional overlap."""
    step_x = w / grid
    step_y = h / grid
    ox, oy = int(step_x * overlap), int(step_y * overlap)
    for gy in range(grid):
        for gx in range(grid):
            x0 = max(0, int(gx * step_x) - ox)
            y0 = max(0, int(gy * step_y) - oy)
            x1 = min(w, int((gx + 1) * step_x) + ox)
            y1 = min(h, int((gy + 1) * step_y) + oy)
            yield x0, y0, x1, y1


def _merge_nms(dets: list[Detection], iou_thres: float) -> list[Detection]:
    """Class-wise NMS to merge duplicate detections across overlapping tiles."""
    if not dets:
        return []
    out: list[Detection] = []
    by_cls: dict[int, list[Detection]] = {}
    for d in dets:
        by_cls.setdefault(d.cls_id, []).append(d)
    for cid, group in by_cls.items():
        boxes = [[d.bbox[0], d.bbox[1], d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1]] for d in group]
        confs = [d.conf for d in group]
        idxs = cv2.dnn.NMSBoxes(boxes, confs, 0.0, iou_thres)
        for i in np.array(idxs).flatten() if len(idxs) else []:
            out.append(group[int(i)])
    return out


def tiled_detect(
    det: YoloOnnxDetector,
    img: np.ndarray,
    grid: int = 2,
    overlap: float = 0.2,
    add_full_frame: bool = True,
    merge_iou: float = 0.5,
) -> list[Detection]:
    """Detect on each tile (upscaled by the detector's letterbox), map back, merge."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    collected: list[Detection] = []
    for x0, y0, x1, y1 in _tiles(h, w, grid, overlap):
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        for d in det.detect(crop):                       # detector letterboxes crop -> imgsz
            bx1, by1, bx2, by2 = d.bbox
            collected.append(Detection((bx1 + x0, by1 + y0, bx2 + x0, by2 + y0),
                                       d.cls_id, d.cls_name, d.conf))
    if add_full_frame:
        collected.extend(det.detect(img))                # catch objects that straddle tile seams
    return _merge_nms(collected, merge_iou)


# -- CLI --------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Tiled (SAHI-style) inference on image(s).")
    p.add_argument("source", help="image file or directory")
    p.add_argument("--onnx", default=str(DEFAULT_ONNX))
    p.add_argument("--grid", type=int, default=2, help="grid×grid tiles per frame")
    p.add_argument("--overlap", type=float, default=0.2)
    p.add_argument("--no-full-frame", action="store_true")
    p.add_argument("--out", default=str(REPO / "runs" / "tiled_out"))
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    det = YoloOnnxDetector(args.onnx, conf_thres=PER_CLASS_CONF)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    src = Path(args.source)
    imgs = sorted(q for q in src.rglob("*") if q.suffix.lower() in exts) if src.is_dir() else [src]
    n = 0
    for ip in imgs:
        img = cv2.imread(str(ip))
        if img is None:
            continue
        dets = tiled_detect(det, img, args.grid, args.overlap, not args.no_full_frame)
        cv2.imwrite(str(out_dir / ip.name), draw_detections(img, dets))
        print(f"{ip.name}: {len(dets)} detections")
        n += 1
        if args.limit and n >= args.limit:
            break
    print(f"\n{n} images -> {out_dir}")


if __name__ == "__main__":
    main()
