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
* ``--paste N`` — **sonar-aware copy-paste** (ultralytics' ``copy_paste`` is a no-op for box-only
  labels). N synthetic training frames: 1–3 real pots, each cut out WITH its acoustic-shadow tail
  (the far-range side), pasted onto an empty seabed frame of the same sonar type at the SAME slant
  range (±8% of the frame; appearance and shadow length depend on range) and below the Stage-1
  tracked seabed, blended with Poisson editing (``cv2.seamlessClone``) so the speckle and gain
  match the new background. Orientation comes from the source rule (PINGMapper ``*_ss_*`` = nadir
  top); frames of unknown orientation are never used.

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

import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # repo root → src.* imports


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


def _box_px(lab, W, H):
    c, cx, cy, bw, bh = lab
    return c, int((cx - bw / 2) * W), int((cy - bh / 2) * H), int((cx + bw / 2) * W), int((cy + bh / 2) * H)


def pot_bank(src: Path, cls_id: int = 0, pad: float = 0.35, tail: float = 2.0) -> list[dict]:
    """Real pots from TRAIN sonograms whose orientation is known (nadir top): the box padded by
    ``pad`` sideways/up and extended ``tail`` × its height DOWN (far range) to keep the shadow."""
    from src.cv_pipeline.orientation import resolve_orientation
    bank = []
    for lp in sorted((src / "train" / "labels").glob("*.txt")):
        if resolve_orientation(lp.stem).nadir != "top":
            continue
        labs = [l for l in read_labels(lp) if l[0] == cls_id]
        if not labs:
            continue
        im = cv2.imread(str(src / "train" / "images" / f"{lp.stem}.jpg"))
        if im is None:
            continue
        H, W = im.shape[:2]
        for lab in labs:
            _, x1, y1, x2, y2 = _box_px(lab, W, H)
            bw, bh = x2 - x1, y2 - y1
            if bw < 4 or bh < 4 or bw > W * 0.25:
                continue
            px1, py1 = max(0, int(x1 - pad * bw)), max(0, int(y1 - pad * bh))
            px2, py2 = min(W, int(x2 + pad * bw)), min(H, int(y2 + tail * bh))
            bank.append({"patch": im[py1:py2, px1:px2].copy(), "box": (x1 - px1, y1 - py1, x2 - px1, y2 - py1),
                         "yfrac": (y1 + y2) / 2 / H, "src": lp.stem})
    return bank


def paste_frames(src: Path, out: Path, n: int, rng: random.Random, stats: dict) -> None:
    from src.cv_pipeline.canonical import Canonicaliser
    from src.cv_pipeline.orientation import resolve_orientation
    bank = pot_bank(src)
    bgs = [ip for ip in sorted((src / "train" / "images").iterdir())
           if resolve_orientation(ip.stem).nadir == "top"
           and not read_labels(src / "train" / "labels" / f"{ip.stem}.txt")]
    stats.update({"paste_bank_pots": len(bank), "paste_backgrounds": len(bgs), "paste_frames": 0, "pots_pasted": 0})
    if not bank or not bgs:
        print("paste: no pot bank or no empty oriented backgrounds - skipped")
        return
    C = Canonicaliser()
    for i in range(n):
        bp = rng.choice(bgs)
        bg = cv2.imread(str(bp))
        if bg is None:
            continue
        H, W = bg.shape[:2]
        cf = C.process(bg, bp.stem)
        seabed = (cf.altitude_px or 0) + 6
        placed, labels = [], []
        for _ in range(rng.choice((1, 1, 2, 3))):
            for _try in range(20):
                pot = rng.choice(bank)
                ph, pw = pot["patch"].shape[:2]
                if ph + 4 >= H or pw + 4 >= W:
                    break
                bx1, by1, bx2, by2 = pot["box"]
                cy = int(np.clip(pot["yfrac"] * H + rng.uniform(-0.08, 0.08) * H, 0, H))   # same range ±8%
                ty = cy - (by1 + by2) // 2                                                   # patch top
                tx = rng.randint(2, W - pw - 2)
                if ty < 2 or ty + ph > H - 2 or ty + by1 < seabed:
                    continue
                box = (tx + bx1, ty + by1, tx + bx2, ty + by2)
                if any(not (box[2] < q[0] - 8 or box[0] > q[2] + 8 or box[3] < q[1] - 8 or box[1] > q[3] + 8)
                       for q in placed):
                    continue
                mask = np.zeros((ph, pw), np.uint8)
                cv2.ellipse(mask, (pw // 2, ph // 2), (max(2, pw // 2 - 1), max(2, ph // 2 - 1)), 0, 0, 360, 255, -1)
                cand = cv2.seamlessClone(pot["patch"], bg, mask, (tx + pw // 2, ty + ph // 2), cv2.NORMAL_CLONE)
                # visibility gate: Poisson blending onto a smooth background can wash a low-contrast
                # pot out; a label on an invisible pot teaches noise, so require the box to stay
                # >= 10% brighter than the seabed ring around it.
                g = cv2.cvtColor(cand, cv2.COLOR_BGR2GRAY).astype(np.float32)
                x1, y1, x2, y2 = box
                bw_, bh_ = x2 - x1, y2 - y1
                ring = g[max(0, y1 - bh_):min(H, y2 + bh_), max(0, x1 - bw_):min(W, x2 + bw_)]
                inner = g[y1:y2, x1:x2]
                ring_mean = (ring.sum() - inner.sum()) / max(1, ring.size - inner.size)
                if inner.size == 0 or np.percentile(inner, 90) < 1.10 * max(ring_mean, 1.0):
                    stats["paste_rejected_faint"] = stats.get("paste_rejected_faint", 0) + 1
                    continue
                bg = cand
                placed.append(box)
                labels.append((0, (box[0] + box[2]) / 2 / W, (box[1] + box[3]) / 2 / H,
                               (box[2] - box[0]) / W, (box[3] - box[1]) / H))
                break
        if not labels:
            continue
        name = f"paste_{i:05d}__{bp.stem[:60]}"
        cv2.imwrite(str(out / "train" / "images" / f"{name}.jpg"), bg, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out / "train" / "labels" / f"{name}.txt").write_text(
            "\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in labels))
        stats["paste_frames"] += 1
        stats["pots_pasted"] += len(labels)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--src", required=True, help="YOLO dataset root with train/ val/ test/ and data.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile-frac", type=float, default=0.6, help="tile side as a fraction of the frame side")
    ap.add_argument("--min-vis", type=float, default=0.6, help="min visible fraction to keep a clipped box")
    ap.add_argument("--bg-keep", type=float, default=0.35, help="share of box-free tiles to keep")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--paste", type=int, default=0, help="N synthetic sonar-aware copy-paste frames (0 = off)")
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

    if a.paste:
        paste_frames(src, out, a.paste, random.Random(a.seed + 1), stats)

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
