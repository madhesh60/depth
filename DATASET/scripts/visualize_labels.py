"""
visualize_labels.py — Render YOLO boxes onto sampled images for visual QA (#12).

Draws class-coloured boxes + labels on a random sample of images (stratified per source
prefix) so annotation quality can be eyeballed per dataset. Output PNGs go to
DATASET/exports/qa_<split>/.

Usage:  python DATASET/scripts/visualize_labels.py [DATASET_ROOT] [--split train] [--per-source 6]
        default root: DATASET/03_yolo_ready_dataset_v1
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import cv2

DATASET = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["fishing_gear", "pipe_cylinder", "structural_fragment", "natural_formation"]
COLORS = [(0, 0, 255), (0, 200, 255), (0, 255, 0), (255, 128, 0)]  # BGR per class
PREFIXES = ["crabpot", "uatd", "icra", "mpulse", "seabed", "shipwreck", "vid"]


def source_of(stem: str) -> str:
    low = stem.lower()
    for p in PREFIXES:
        if low.startswith(p):
            return p
    return "other"


def draw(img_path: Path, lbl_path: Path, out_path: Path):
    im = cv2.imread(str(img_path))
    if im is None:
        return False
    h, w = im.shape[:2]
    if lbl_path.exists():
        for ln in lbl_path.read_text().splitlines():
            t = ln.split()
            if len(t) != 5:
                continue
            cid = int(float(t[0]))
            xc, yc, bw, bh = (float(v) for v in t[1:])
            x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
            x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
            color = COLORS[cid % len(COLORS)]
            cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
            cv2.putText(im, CLASS_NAMES[cid], (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), im)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=str(DATASET / "03_yolo_ready_dataset_v1"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--per-source", type=int, default=6)
    args = ap.parse_args()

    root = Path(args.root)
    img_dir = root / args.split / "images"
    lbl_dir = root / args.split / "labels"
    out_dir = DATASET / "exports" / f"qa_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_source = defaultdict(list)
    for img in img_dir.glob("*.*"):
        by_source[source_of(img.stem)].append(img)

    random.seed(0)
    n = 0
    for src, imgs in sorted(by_source.items()):
        random.shuffle(imgs)
        for img in imgs[: args.per_source]:
            if draw(img, lbl_dir / f"{img.stem}.txt", out_dir / f"{src}__{img.stem}.png"):
                n += 1
    print(f"wrote {n} QA images (per_source={args.per_source}) -> {out_dir}")


if __name__ == "__main__":
    main()
