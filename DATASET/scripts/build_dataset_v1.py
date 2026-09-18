"""
build_dataset_v1.py — Produce the clean, training-ready dataset (v1) from the raw v0 split.

Fixes the audit-verified problems (see docs/dataset_report.md):
  #1/#2 Leakage + baked augmentation  -> split at the SOURCE-FRAME level (a frame and all its
        augmentations live in exactly one split). Train keeps augmentations; val/test get
        ONLY the original frame -> clean, honest evaluation, zero leakage by construction.
  #4    Full-frame boxes (w&h>THRESH) -> dropped on write.
  #5    Degenerate/zero-area boxes    -> dropped; coords clamped to [0,1].
        (Plausibly-small sonar boxes are intentionally KEPT — flagged for visual QA.)
  #9    No negatives                  -> images whose boxes all get dropped become background
        (empty-label) negatives, which improves false-positive control.
  #10   Corrupt images                -> every image is opened/verified; corrupt ones skipped + logged.
  #3    Taxonomy                       -> 4 classes (rope_line stays merged into fishing_gear).
  #11   Versioning                     -> writes manifest.json + VERSION.

Images are hard-linked (instant, ~zero extra disk); labels are written fresh (cleaned).
The source v0 dataset is left untouched.

Usage:  python DATASET/scripts/build_dataset_v1.py
Then:   python DATASET/scripts/audit_dataset.py DATASET/03_yolo_ready_dataset_v1
"""
from __future__ import annotations

import hashlib
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
SRC = DATASET / "03_yolo_ready_dataset"
DST = DATASET / "03_yolo_ready_dataset_v1"
SPLITS = ("train", "val", "test")
SPLIT_RATIO = (0.80, 0.10, 0.10)   # applied at the FRAME-GROUP level
SEED = 42

CLASS_NAMES = ["fishing_gear", "pipe_cylinder", "structural_fragment", "natural_formation"]
NC = len(CLASS_NAMES)

FULLFRAME_WH = 0.95     # drop box if w > this AND h > this
MIN_AREA = 1e-5         # drop box if normalized area below this (degenerate)
SLIVER = 0.01           # drop box if width OR height below this (~6px @640; degenerate sliver)

PREFIXES = ["crabpot", "uatd", "icra", "mpulse", "seabed", "shipwreck", "vid"]
SENSOR = {  # source -> sensor modality (for later domain-split experiments)
    "crabpot": "sonar", "uatd": "sonar", "mpulse": "sonar",
    "seabed": "sonar", "shipwreck": "sonar", "icra": "optical", "vid": "optical",
}


def base_key(stem: str) -> str:
    stem = re.sub(r"_aug\d+$", "", stem)
    stem = re.sub(r"\.rf\.[0-9a-f]+$", "", stem)
    return stem


def seq_group(stem: str) -> str:
    """Source-aware split key. Keeps a whole recording / video clip in ONE split so that
    consecutive near-identical frames never leak across train/val/test.
      icra/vid  -> video-clip id (frames of a clip are near-identical)
      crabpot   -> recording+channel (strip trailing frame index)
      shipwreck -> wreck/object (strip trailing view index)
      mpulse    -> survey site (strip trailing _pNN)
      uatd/seabed/other -> per frame (independent index images, no sequence)
    """
    core = re.sub(r"(_png)?(_jpe?g|_bmp)$", "", base_key(stem), flags=re.I)
    s = source_of(core)
    if s in ("icra", "vid"):
        return re.sub(r"_frame\d+.*$", "", core)          # video clip id
    if s == "crabpot":
        core = re.sub(r"_\d+$", "", core)                  # drop frame index
        return re.sub(r"_ss_(port|star)$", "_ss", core)   # merge port/star channels of a run
    if s == "shipwreck":
        return re.sub(r"_\d+$", "", core)                 # wreck/object
    if s == "mpulse":
        return re.sub(r"_p\d+$", "", core)                # survey site
    return core                                            # uatd/seabed/other: per frame


def is_aug(stem: str) -> bool:
    return bool(re.search(r"_aug\d+$", stem))


def source_of(stem: str) -> str:
    low = stem.lower()
    for p in PREFIXES:
        if low.startswith(p):
            return p
    return "other"


def clean_label(text: str):
    """Return (cleaned_lines, stats). Drops full-frame / sliver / degenerate boxes, clamps coords."""
    kept, dropped_ff, dropped_tiny, dropped_sliver, dropped_bad = [], 0, 0, 0, 0
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        toks = ln.split()
        if len(toks) != 5:          # v1 is box-only; anything else is malformed
            dropped_bad += 1
            continue
        try:
            cid = int(float(toks[0]))
            x, y, w, h = (float(v) for v in toks[1:])
        except ValueError:
            dropped_bad += 1
            continue
        if not (0 <= cid < NC):
            dropped_bad += 1
            continue
        # clamp center/size into valid range
        x, y = min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)
        w, h = min(max(w, 0.0), 1.0), min(max(h, 0.0), 1.0)
        if w > FULLFRAME_WH and h > FULLFRAME_WH:
            dropped_ff += 1
            continue
        if w < SLIVER or h < SLIVER:
            dropped_sliver += 1
            continue
        if w * h < MIN_AREA:
            dropped_tiny += 1
            continue
        kept.append(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return kept, (dropped_ff, dropped_tiny, dropped_sliver, dropped_bad)


def dominant_class(label_text: str) -> int:
    counts = Counter()
    for ln in label_text.splitlines():
        toks = ln.split()
        if len(toks) == 5:
            try:
                counts[int(float(toks[0]))] += 1
            except ValueError:
                pass
    return counts.most_common(1)[0][0] if counts else -1   # -1 = background


def link_or_copy(src: Path, dst: Path):
    try:
        os.link(src, dst)          # hard link — instant, no extra disk
    except OSError:
        shutil.copy2(src, dst)     # fallback (cross-volume etc.)


def main():
    random.seed(SEED)

    # ---- 1. index the source: build stem -> image path, and group by frame ----
    img_of = {}
    for s in SPLITS:
        for img in (SRC / s / "images").glob("*.*"):
            img_of[img.stem] = img

    groups = defaultdict(list)      # seq_group (recording/clip) -> list of stems
    label_text = {}                 # stem -> raw label text
    for s in SPLITS:
        for lf in (SRC / s / "labels").glob("*.txt"):
            stem = lf.stem
            if stem not in img_of:
                continue            # orphan label; skip
            groups[seq_group(stem)].append(stem)
            label_text[stem] = lf.read_text()

    print(f"indexed {len(img_of)} images, {len(groups)} recording/clip groups")

    # ---- 2. balanced, leakage-free split at the GROUP (recording/clip) level ----
    # Greedy assignment: place whole groups (no frame ever crosses splits) while keeping
    # each split's per-class box share AND image count close to the target ratio.
    grp_vec, grp_imgs = {}, {}
    for gk, stems in groups.items():
        vec = [0] * NC
        for stem in stems:
            for ln in label_text[stem].splitlines():
                t = ln.split()
                if len(t) == 5:
                    try:
                        c = int(float(t[0]))
                    except ValueError:
                        continue
                    if 0 <= c < NC:
                        vec[c] += 1
        grp_vec[gk] = vec
        grp_imgs[gk] = len(stems)

    G = [sum(grp_vec[gk][k] for gk in groups) for k in range(NC)]
    total_imgs = sum(grp_imgs.values())
    tgt = {s: [SPLIT_RATIO[i] * G[k] for k in range(NC)] for i, s in enumerate(SPLITS)}
    tgt_imgs = {s: SPLIT_RATIO[i] * total_imgs for i, s in enumerate(SPLITS)}
    cur = {s: [0] * NC for s in SPLITS}
    cur_imgs = {s: 0 for s in SPLITS}

    # largest groups first for stable balancing; tie-break random via seeded shuffle
    order = sorted(groups, key=lambda g: (sum(grp_vec[g]), grp_imgs[g]), reverse=True)
    split_of_group = {}
    for gk in order:
        vec, w = grp_vec[gk], grp_imgs[gk]
        best_s, best_cost = None, None
        for s in SPLITS:
            cost = sum(max(0.0, cur[s][k] + vec[k] - tgt[s][k]) / (tgt[s][k] + 1.0)
                       for k in range(NC))
            cost += max(0.0, cur_imgs[s] + w - tgt_imgs[s]) / (tgt_imgs[s] + 1.0)
            if best_cost is None or cost < best_cost:
                best_cost, best_s = cost, s
        split_of_group[gk] = best_s
        for k in range(NC):
            cur[best_s][k] += vec[k]
        cur_imgs[best_s] += w

    # ---- 3. prepare output tree (fresh) ----
    if DST.exists():
        shutil.rmtree(DST)
    for s in SPLITS:
        (DST / s / "images").mkdir(parents=True, exist_ok=True)
        (DST / s / "labels").mkdir(parents=True, exist_ok=True)

    # ---- 4. write files ----
    stats = {s: {"images": 0, "background": 0, "boxes": Counter()} for s in SPLITS}
    tot_ff = tot_tiny = tot_sliver = tot_bad = corrupt = 0
    sensor_counts = {s: Counter() for s in SPLITS}
    from PIL import Image  # local import; Pillow is available

    for gk, stems in groups.items():
        split = split_of_group[gk]
        # train keeps all files of the group; val/test take originals only
        chosen = stems if split == "train" else [s for s in stems if not is_aug(s)]
        if not chosen:                      # safety: no original -> take one
            chosen = stems[:1]
        for stem in chosen:
            img = img_of[stem]
            try:
                with Image.open(img) as im:
                    im.verify()             # corrupt-image check
            except Exception:
                corrupt += 1
                continue
            kept, (d_ff, d_tiny, d_sliver, d_bad) = clean_label(label_text[stem])
            tot_ff += d_ff; tot_tiny += d_tiny; tot_sliver += d_sliver; tot_bad += d_bad
            link_or_copy(img, DST / split / "images" / img.name)
            (DST / split / "labels" / f"{stem}.txt").write_text("\n".join(kept))
            stats[split]["images"] += 1
            sensor_counts[split][SENSOR.get(source_of(stem), "other")] += 1
            if not kept:
                stats[split]["background"] += 1
            for ln in kept:
                stats[split]["boxes"][int(ln.split()[0])] += 1

    # ---- 5. data.yaml ----
    (DST / "data.yaml").write_text(
        "train: train/images\nval: val/images\ntest: test/images\n"
        f"nc: {NC}\nnames:\n" + "".join(f"- {n}\n" for n in CLASS_NAMES)
    )

    # ---- 6. leakage check by construction ----
    def keyset(s):
        return {base_key(p.stem) for p in (DST / s / "labels").glob("*.txt")}
    kt, kv, ke = keyset("train"), keyset("val"), keyset("test")
    leakage = {"train_val": len(kt & kv), "train_test": len(kt & ke), "val_test": len(kv & ke)}

    # ---- 7. manifest ----
    manifest = {
        "version": "v1",
        "created": str(date.today()),
        "built_from": SRC.name,
        "seed": SEED,
        "split_ratio_on_groups": SPLIT_RATIO,
        "split_group_key": "recording/clip (source-aware seq_group)",
        "num_groups": len(groups),
        "classes": {i: n for i, n in enumerate(CLASS_NAMES)},
        "images": {s: stats[s]["images"] for s in SPLITS},
        "background_images": {s: stats[s]["background"] for s in SPLITS},
        "class_boxes": {s: dict(sorted(stats[s]["boxes"].items())) for s in SPLITS},
        "sensor_images": {s: dict(sensor_counts[s]) for s in SPLITS},
        "cleaning": {
            "dropped_fullframe_boxes": tot_ff,
            "dropped_sliver_boxes": tot_sliver,
            "dropped_tiny_boxes": tot_tiny,
            "dropped_malformed_boxes": tot_bad,
            "corrupt_images_skipped": corrupt,
            "fullframe_threshold": FULLFRAME_WH,
            "sliver_threshold": SLIVER,
            "min_area": MIN_AREA,
        },
        "leakage_shared_frames": leakage,
    }
    (DST / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (DST / "VERSION").write_text("v1\n")

    # ---- 8. report ----
    print("\n" + "=" * 68)
    print("DATASET v1 BUILD COMPLETE")
    print("=" * 68)
    print(f"split by recording/clip: {len(groups)} groups")
    for s in SPLITS:
        print(f"[{s:5}] images={stats[s]['images']:>6}  background={stats[s]['background']:>5}  "
              f"boxes={dict(sorted(stats[s]['boxes'].items()))}")
    print(f"\ncleaned boxes: fullframe={tot_ff}  sliver={tot_sliver}  tiny/degenerate={tot_tiny}  "
          f"malformed={tot_bad}  corrupt_images_skipped={corrupt}")
    print(f"LEAKAGE (shared source frames across splits): {leakage}  "
          f"{'[CLEAN]' if sum(leakage.values()) == 0 else '[STILL LEAKING]'}")
    print(f"\nwrote -> {DST}")
    print("next: python DATASET/scripts/audit_dataset.py " + str(DST))


if __name__ == "__main__":
    main()
