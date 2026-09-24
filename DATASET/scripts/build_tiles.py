"""
build_tiles.py — EXP-002 training data: every TRAIN frame as a full frame **plus** a 2×2 grid of
overlapping tiles (review I-3: "train on 2×2 overlapping tiles and full frames, so training matches
tiled inference"; crab pots are small — 91% of EXP-001's misses were < 10% of frame width).

* train/: full frame (unchanged) + 4 tiles per frame (``tile_frac`` of each side, ``overlap`` between
  neighbours). Boxes are clipped to the tile; a clipped box is kept only if ≥ ``min_vis`` of its area
  is visible (a pot cut in half is a different-looking object — don't teach it). Background tiles
  (no box) are kept at ``bg_keep`` so the detector still sees empty seabed.
* val/ and test*/ are **not tiled** — ``data.yaml`` points at the source dataset's full frames, so
  model selection and evaluation stay deploy-faithful (full-frame inference).
* Tiles are written as JPEG at native resolution; ultralytics resizes them to ``imgsz`` (so a tile
  shows each pot ~2× larger than the full frame does — the small-object lever).

    python DATASET/scripts/build_tiles.py --src DATASET/03_yolo_ready_dataset_v2b \
        --out DATASET/03_yolo_ready_dataset_v2b_tiles
    (on Kaggle: --src /kaggle/input/<v2b> --out /kaggle/working/v2b_tiles)
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2


def read_labels(p: Path) -> list[tuple[int, float, float, float, float]]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        q = line.split()
        if len(q) >= 5:
            out.append((int(float(q[0])), *map(float, q[1:5])))
    return out


def tile_boxes(labels, W, H, x0, y0, tw, th, min_vis):
    """YOLO labels (normalised to the frame) → labels normalised to the tile, clipped."""
    out = []
    for c, cx, cy, bw, bh in labels:
        x1, y1, x2, y2 = (cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H
        area = max(1e-6, (x2 - x1) * (y2 - y1))
        ix1, iy1, ix2, iy2 = max(x1, x0), max(y1, y0), min(x2, x0 + tw), min(y2, y0 + th)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        if (ix2 - ix1) * (iy2 - iy1) / area < min_vis:
            continue
        out.append((c, ((ix1 + ix2) / 2 - x0) / tw, ((iy1 + iy2) / 2 - y0) / th, (ix2 - ix1) / tw, (iy2 - iy1) / th))
    return out


def tiles_for(W, H, tile_frac, overlap):
    tw, th = int(round(W * tile_frac)), int(round(H * tile_frac))
    xs = [0, max(0, W - tw)]
    ys = [0, max(0, H - th)]
    # overlap is implied by tile_frac > 0.5 (e.g. 0.6 → 20% of the frame shared between neighbours)
    return [(x, y, tw, th) for y in ys for x in xs]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--src", required=True, help="YOLO dataset root with train/ val/ test/ and data.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile-frac", type=float, default=0.6, help="tile side as a fraction of the frame side")
    ap.add_argument("--min-vis", type=float, default=0.6, help="min visible fraction to keep a clipped box")
    ap.add_argument("--bg-keep", type=float, default=0.35, help="share of box-free tiles to keep")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    src, out = Path(a.src), Path(a.out)
    (out / "train" / "images").mkdir(parents=True, exist_ok=True)
    (out / "train" / "labels").mkdir(parents=True, exist_ok=True)

    stats = {"frames": 0, "tiles_written": 0, "tiles_bg_dropped": 0, "boxes_full": 0, "boxes_tiles": 0,
             "boxes_clipped_dropped": 0}
    for ip in sorted((src / "train" / "images").iterdir()):
        im = cv2.imread(str(ip))
        if im is None:
            continue
        H, W = im.shape[:2]
        labels = read_labels(src / "train" / "labels" / f"{ip.stem}.txt")
        stats["frames"] += 1
        stats["boxes_full"] += len(labels)
        shutil.copy2(ip, out / "train" / "images" / ip.name)                      # the full frame
        (out / "train" / "labels" / f"{ip.stem}.txt").write_text(
            "\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in labels))
        for k, (x0, y0, tw, th) in enumerate(tiles_for(W, H, a.tile_frac, 0)):
            tl = tile_boxes(labels, W, H, x0, y0, tw, th, a.min_vis)
            touching = sum(1 for c, cx, cy, bw, bh in labels
                           if (cx + bw / 2) * W > x0 and (cx - bw / 2) * W < x0 + tw
                           and (cy + bh / 2) * H > y0 and (cy - bh / 2) * H < y0 + th)
            stats["boxes_clipped_dropped"] += touching - len(tl)
            if not tl and random.random() > a.bg_keep:
                stats["tiles_bg_dropped"] += 1
                continue
            name = f"{ip.stem}__t{k}"
            cv2.imwrite(str(out / "train" / "images" / f"{name}.jpg"), im[y0:y0 + th, x0:x0 + tw],
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out / "train" / "labels" / f"{name}.txt").write_text(
                "\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in tl))
            stats["tiles_written"] += 1
            stats["boxes_tiles"] += len(tl)

    # data.yaml: tiled train, source full-frame val/test (deploy-faithful selection + evaluation)
    names = []
    for line in (src / "data.yaml").read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("- "):
            names.append(line.strip()[2:].strip())
    rel = lambda d: str((src / d / "images").resolve()).replace("\\", "/")
    yaml = [f"# EXP-002 tiled training set (build_tiles.py) — train = full frames + 2x2 tiles; val/test = source full frames",
            f"path: {str(out.resolve()).replace(chr(92), '/')}", "train: train/images",
            f"val: {rel('val')}", f"test: {rel('test')}", f"nc: {len(names)}", "names:"] + [f"- {n}" for n in names]
    (out / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    (out / "tiles_manifest.json").write_text(json.dumps({"src": str(src), "args": vars(a), "stats": stats}, indent=1))
    print(json.dumps(stats, indent=1))
    print(f"-> {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
