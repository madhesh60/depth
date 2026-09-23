"""
build_dataset_v2.py — honest, sonar-only, 2-class dataset (ghost_gear + wreck_debris).

Fixes the WINNING_REPORT.md data bugs that v1 still carries:
  * Marine PULSE pipeline/platform images each got ONE full-frame box in v0, which v1's
    full-frame filter then deleted -> 955 pictures of REAL man-made objects became "empty
    seabed". They have no usable localisation, so v2 DROPS them (never trains them as
    background). Only Marine PULSE natural seabed (`seabed_surface`) is kept, as background.
  * 1,547 empty crab-pot frames were skipped when v0 was built. v2 RECOVERS them straight
    from the raw crab-pot archive as natural-seabed background negatives — the exact cure for
    the biggest EXP-001 error (empty seabed read as fishing gear).
  * Optical camera photos (ICRA19 / TrashCan) and forward-looking UATD inflated EXP-001's
    aggregate mAP. v2 DROPS optical entirely and EXCLUDES UATD from the trained set, exporting
    it instead as a SEPARATE held-out "forward-looking sonar" generalisation eval.
  * Class names now match the data: ghost_gear (side-scan crab pots) + wreck_debris
    (KLSG barges + AI4Shipwrecks sonar).
  * The official crab-pot split (train 5721 / valid 555 / test 398) is respected, and the
    398-image test set is LOCKED and recorded in the manifest so metrics can be reported
    head-to-head with the published GhostVision result on the very same test images.

Sources & mapping
  crab-pot  (RAW archive, side-scan sonar)  -> ghost_gear ; empty frames -> background
                                               (uses the dataset's OWN train/valid/test split)
  shipwreck (v0, KLSG sonar)                -> wreck_debris        } leakage-free 80/10/10
  seabed    (v0, AI4Shipwrecks sonar)       -> wreck_debris (debris boxes);
                                               natural-only frames -> background
  mpulse `seabed_surface` (v0, sonar)       -> background (natural seabed)
  mpulse pipeline/platform                  -> DROPPED (real objects, no usable box)
  ICRA19 / TrashCan / vid (optical)         -> DROPPED (wrong sensor; caused shortcuts)
  UATD (v0, forward-looking sonar)          -> SEPARATE held-out eval (_holdout_uatd_fls/)

Originals only — no baked augmentation is copied; YOLO augments at train time (train.py is
already sonar-aware). Images are hard-linked (instant, ~zero extra disk); labels written fresh.

Usage:  python DATASET/scripts/build_dataset_v2.py
Then:   python DATASET/scripts/audit_dataset.py DATASET/03_yolo_ready_dataset_v2
"""
from __future__ import annotations

import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

# ---- config -----------------------------------------------------------------
DATASET = Path(__file__).resolve().parents[1]
V0 = DATASET / "03_yolo_ready_dataset"
RAW_CRAB = DATASET / "01_raw_archives" / "crab-pot-dataset"
DST = DATASET / "03_yolo_ready_dataset_v2"
SPLITS = ("train", "val", "test")
WRECK_RATIO = (0.80, 0.10, 0.10)   # for the v0 sonar sources (crab-pot uses its own split)
SEED = 42

CLASS_NAMES = ["ghost_gear", "wreck_debris"]
GHOST, WRECK = 0, 1
NC = len(CLASS_NAMES)

# v0 class ids (fishing_gear, pipe_cylinder, structural_fragment, natural_formation)
V0_FISHING, V0_PIPE, V0_STRUCT, V0_NATURAL = 0, 1, 2, 3

FULLFRAME_WH = 0.95     # drop a box if w > this AND h > this (whole-image = no localisation)
MIN_AREA = 1e-5         # drop degenerate boxes
SLIVER = 0.01           # drop a box if width OR height below this (~6px @640)


# ---- shared box cleaning (same rules as v1) ---------------------------------
def clean_boxes(rows):
    """rows: iterable of (cls, cx, cy, w, h) normalised. Returns (kept_lines, drop_stats)."""
    kept = []
    d_ff = d_sliver = d_tiny = 0
    for cls, x, y, w, h in rows:
        x, y = min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)
        w, h = min(max(w, 0.0), 1.0), min(max(h, 0.0), 1.0)
        if w > FULLFRAME_WH and h > FULLFRAME_WH:
            d_ff += 1; continue
        if w < SLIVER or h < SLIVER:
            d_sliver += 1; continue
        if w * h < MIN_AREA:
            d_tiny += 1; continue
        kept.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return kept, (d_ff, d_sliver, d_tiny)


def link_or_copy(src: Path, dst: Path):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def is_aug(stem: str) -> bool:
    return bool(re.search(r"_aug\d+$", stem))


# ---- source A: crab-pot from the RAW archive (own split, pots + empties) -----
def build_crabpot(dst_split_dir, corrupt_counter, drop_counter):
    """Read the raw crab-pot metadata.jsonl per split, convert COCO bbox->YOLO, write
    ghost_gear labels; frames with no annotation become background negatives.
    Returns {raw_split: [written_stems]} and records the locked official test stems."""
    from PIL import Image
    written = {"train": [], "valid": [], "test": []}
    empties = Counter()
    boxes = Counter()
    # raw split -> v2 split
    to_v2 = {"train": "train", "valid": "val", "test": "test"}
    for raw_split in ("train", "valid", "test"):
        meta = RAW_CRAB / raw_split / "metadata.jsonl"
        if not meta.exists():
            continue
        v2split = to_v2[raw_split]
        for line in meta.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            fname = rec["file_name"]
            img = RAW_CRAB / raw_split / fname
            if not img.exists():
                continue
            try:
                with Image.open(img) as im:
                    W, H = im.size
                    im.verify()
            except Exception:
                corrupt_counter[0] += 1
                continue
            objs = rec.get("objects", {}) or {}
            bboxes = objs.get("bbox", []) or []
            rows = []
            for bx in bboxes:                       # COCO [x, y, w, h] in pixels (top-left)
                x, y, w, h = bx
                rows.append((GHOST, (x + w / 2) / W, (y + h / 2) / H, w / W, h / H))
            kept, (d_ff, d_sl, d_ti) = clean_boxes(rows)
            drop_counter[0] += d_ff; drop_counter[1] += d_sl; drop_counter[2] += d_ti
            stem = img.stem
            link_or_copy(img, dst_split_dir[v2split]["images"] / img.name)
            (dst_split_dir[v2split]["labels"] / f"{stem}.txt").write_text("\n".join(kept))
            written[raw_split].append(stem)
            if not kept:
                empties[v2split] += 1
            for ln in kept:
                boxes[v2split] += 1
    return written, empties, boxes


# ---- source B: v0 sonar sources (shipwreck, seabed, mpulse seabed_surface) ---
def gather_v0_sonar():
    """Return {group_key: [(stem, img_path, kept_rows)]} for the trainable v0 sonar sources,
    remapped to the 2-class scheme. Natural-only frames carry an empty kept_rows (background)."""
    groups = defaultdict(list)
    # index v0 images once
    img_of = {}
    for s in SPLITS:
        for img in (V0 / s / "images").glob("*.*"):
            img_of[img.stem] = img

    def read_label(stem):
        for s in SPLITS:
            lf = V0 / s / "labels" / f"{stem}.txt"
            if lf.exists():
                return lf.read_text()
        return ""

    for stem, img in img_of.items():
        if is_aug(stem):
            continue                                  # originals only
        low = stem.lower()
        if low.startswith("shipwreck"):
            src, group = "shipwreck", re.sub(r"_\d+$", "", stem)   # per wreck/object
        elif low.startswith("seabed"):
            m = re.search(r"(\d+)", stem)
            blk = int(m.group(1)) // 100 if m else 0               # 100-frame blocks
            src, group = "seabed", f"seabed_{blk}"
        elif low.startswith("mpulse"):
            if "seabed_surface" in low or "residual_mound" in low or "mound" in low:
                src, group = "mpulse_bg", "mpulse_bg"              # natural seabed -> background
            else:
                continue                                           # pipeline/platform -> DROP
        else:
            continue                                               # crab-pot/uatd/optical handled elsewhere

        rows = []
        for ln in read_label(stem).splitlines():
            t = ln.split()
            if len(t) != 5:
                continue
            try:
                c = int(float(t[0])); x, y, w, h = (float(v) for v in t[1:])
            except ValueError:
                continue
            if c == V0_STRUCT:                        # debris -> wreck_debris
                rows.append((WRECK, x, y, w, h))
            # class NATURAL / anything else -> dropped (frame may become background)
        groups[group].append((stem, img, rows))
    return groups


def split_wreck_groups(groups):
    """Deterministic leakage-free 80/10/10 by group, balanced on image count."""
    random.seed(SEED)
    target = {s: r for s, r in zip(SPLITS, WRECK_RATIO)}
    total = sum(len(v) for v in groups.values()) or 1
    cur = {s: 0 for s in SPLITS}
    split_of = {}
    # largest groups first so the big buckets balance; deterministic tiebreak by key
    for gk in sorted(groups, key=lambda g: (-len(groups[g]), g)):
        n = len(groups[gk])
        # place where the fill-fraction (relative to target share) is currently lowest
        best = min(SPLITS, key=lambda s: (cur[s] + n) / (target[s] * total))
        split_of[gk] = best
        cur[best] += n
    return split_of


# ---- source C: UATD held-out forward-looking eval ---------------------------
def build_uatd_holdout(dst):
    """Copy UATD (forward-looking sonar) as a SEPARATE eval set. struct->wreck_debris;
    cylinders (pipe) dropped. Not part of train/val/test. Documents generalisation."""
    from PIL import Image
    out_i = dst / "_holdout_uatd_fls" / "images"
    out_l = dst / "_holdout_uatd_fls" / "labels"
    out_i.mkdir(parents=True, exist_ok=True)
    out_l.mkdir(parents=True, exist_ok=True)
    img_of = {}
    for s in SPLITS:
        for img in (V0 / s / "images").glob("*.*"):
            if img.stem.lower().startswith("uatd") and not is_aug(img.stem):
                img_of[img.stem] = img
    n_img = n_box = 0
    for stem, img in img_of.items():
        text = ""
        for s in SPLITS:
            lf = V0 / s / "labels" / f"{stem}.txt"
            if lf.exists():
                text = lf.read_text(); break
        rows = []
        for ln in text.splitlines():
            t = ln.split()
            if len(t) != 5:
                continue
            try:
                c = int(float(t[0])); x, y, w, h = (float(v) for v in t[1:])
            except ValueError:
                continue
            if c == V0_STRUCT:
                rows.append((WRECK, x, y, w, h))       # placed debris objects
            # cylinders (pipe) dropped: no analogue in the 2-class side-scan taxonomy
        kept, _ = clean_boxes(rows)
        try:
            with Image.open(img) as im:
                im.verify()
        except Exception:
            continue
        link_or_copy(img, out_i / img.name)
        (out_l / f"{stem}.txt").write_text("\n".join(kept))
        n_img += 1; n_box += len(kept)
    (dst / "_holdout_uatd_fls" / "README.txt").write_text(
        "UATD forward-looking sonar — held-out generalisation eval ONLY (never trained on).\n"
        "Different sonar geometry + placed test objects; structural targets remapped to\n"
        "wreck_debris, cylinders dropped. Use to report cross-sensor generalisation, not\n"
        "as a headline metric.\n"
    )
    return n_img, n_box


# ---- main -------------------------------------------------------------------
def main():
    if DST.exists():
        shutil.rmtree(DST)
    dst_split_dir = {}
    for s in SPLITS:
        di = DST / s / "images"; dl = DST / s / "labels"
        di.mkdir(parents=True, exist_ok=True); dl.mkdir(parents=True, exist_ok=True)
        dst_split_dir[s] = {"images": di, "labels": dl}

    corrupt = [0]
    crab_drops = [0, 0, 0]           # ff, sliver, tiny
    stats = {s: {"images": 0, "background": 0, "boxes": Counter()} for s in SPLITS}

    # A) crab-pot (own split) -------------------------------------------------
    print("building crab-pot (raw archive, official split)...")
    crab_written, crab_empties, crab_boxes = build_crabpot(dst_split_dir, corrupt, crab_drops)
    official_test_stems = sorted(crab_written["test"])
    for s in SPLITS:
        stats[s]["images"] += len(crab_written[{"train": "train", "val": "valid", "test": "test"}[s]])
        stats[s]["background"] += crab_empties[s]
        stats[s]["boxes"][GHOST] += crab_boxes[s]

    # B) v0 sonar sources (wreck_debris + natural-seabed background) ----------
    print("gathering v0 sonar sources (shipwreck / seabed / mpulse)...")
    wreck_groups = gather_v0_sonar()
    split_of = split_wreck_groups(wreck_groups)
    wreck_drops = [0, 0, 0]
    from PIL import Image
    for gk, items in wreck_groups.items():
        s = split_of[gk]
        for stem, img, rows in items:
            try:
                with Image.open(img) as im:
                    im.verify()
            except Exception:
                corrupt[0] += 1; continue
            kept, (d_ff, d_sl, d_ti) = clean_boxes(rows)
            wreck_drops[0] += d_ff; wreck_drops[1] += d_sl; wreck_drops[2] += d_ti
            link_or_copy(img, dst_split_dir[s]["images"] / img.name)
            (dst_split_dir[s]["labels"] / f"{stem}.txt").write_text("\n".join(kept))
            stats[s]["images"] += 1
            if not kept:
                stats[s]["background"] += 1
            for ln in kept:
                stats[s]["boxes"][int(ln.split()[0])] += 1

    # C) UATD held-out eval ---------------------------------------------------
    print("exporting UATD forward-looking held-out eval...")
    uatd_img, uatd_box = build_uatd_holdout(DST)

    # data.yaml ---------------------------------------------------------------
    (DST / "data.yaml").write_text(
        "train: train/images\nval: val/images\ntest: test/images\n"
        f"nc: {NC}\nnames:\n" + "".join(f"- {n}\n" for n in CLASS_NAMES)
    )

    # leakage check (by frame stem, across splits) ----------------------------
    def stems(s):
        return {p.stem for p in (DST / s / "labels").glob("*.txt")}
    kt, kv, ke = stems("train"), stems("val"), stems("test")
    leakage = {"train_val": len(kt & kv), "train_test": len(kt & ke), "val_test": len(kv & ke)}

    manifest = {
        "version": "v2",
        "created": str(date.today()),
        "taxonomy": {i: n for i, n in enumerate(CLASS_NAMES)},
        "design": "sonar-only, 2-class; optical + UATD removed from train/val/test",
        "sources": {
            "ghost_gear": "crab-pot side-scan sonar (raw archive, official split)",
            "wreck_debris": "KLSG shipwreck + AI4Shipwrecks sonar (v0)",
            "background": "recovered empty crab-pot frames + Marine PULSE natural seabed "
                          "+ AI4Shipwrecks natural-only frames",
            "dropped": "Marine PULSE pipeline/platform (full-frame, unlocalised); "
                       "ICRA19/TrashCan optical; vid optical",
            "holdout": "UATD forward-looking sonar -> _holdout_uatd_fls/ (never trained)",
        },
        "crabpot_split": "OFFICIAL dataset split respected (train->train, valid->val, test->test)",
        "official_crabpot_test_locked": len(official_test_stems),
        "wreck_split": "leakage-free 80/10/10 by wreck/frame-block group",
        "images": {s: stats[s]["images"] for s in SPLITS},
        "background_images": {s: stats[s]["background"] for s in SPLITS},
        "class_boxes": {s: dict(sorted(stats[s]["boxes"].items())) for s in SPLITS},
        "holdout_uatd_fls": {"images": uatd_img, "boxes": uatd_box},
        "cleaning": {
            "crabpot_dropped_boxes": {"fullframe": crab_drops[0], "sliver": crab_drops[1],
                                      "tiny": crab_drops[2]},
            "wreck_dropped_boxes": {"fullframe": wreck_drops[0], "sliver": wreck_drops[1],
                                    "tiny": wreck_drops[2]},
            "corrupt_images_skipped": corrupt[0],
        },
        "leakage_shared_frames": leakage,
    }
    (DST / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (DST / "official_crabpot_test.txt").write_text("\n".join(official_test_stems) + "\n")
    (DST / "VERSION").write_text("v2\n")

    # report ------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("DATASET v2 BUILD COMPLETE  (2-class, sonar-only)")
    print("=" * 68)
    for s in SPLITS:
        print(f"[{s:5}] images={stats[s]['images']:>6}  background={stats[s]['background']:>5}  "
              f"boxes={dict(sorted(stats[s]['boxes'].items()))}")
    print(f"held-out UATD FLS eval: images={uatd_img}  boxes={uatd_box}")
    print(f"official crab-pot test LOCKED: {len(official_test_stems)} frames")
    print(f"corrupt skipped: {corrupt[0]}")
    print(f"LEAKAGE (shared frames across splits): {leakage}  "
          f"{'[CLEAN]' if sum(leakage.values()) == 0 else '[STILL LEAKING]'}")
    print(f"\nwrote -> {DST}")
    print("next: python DATASET/scripts/audit_dataset.py " + str(DST))


if __name__ == "__main__":
    main()
