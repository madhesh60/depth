"""
build_dataset_v2b.py — v2 with its three evaluation traps removed (review I-2 / C-4).

v2 is honest about *what* is in it (sonar-only, 2 classes, official crab-pot split) but three
things made its numbers untrustworthy:

1. **Baked-in Roboflow augmentation.** The Hugging Face crab-pot archive was exported from Roboflow
   with augmentation already applied: 1,015 unique crab-pot sonogram frames appear as ~4,700 images,
   2-12 copies each, including **rotations with black corners** (rotation breaks sonar geometry:
   range and shadow directions stop being consistent). Ultralytics then augments again.
   → **keep ONE copy per frame** (the least-changed: smallest black border, brightness closest to
   the copies' median) and let ``train.py`` do the augmenting.
2. **Duplicate test frames.** The official test has 398 images but only ~214 unique frames, so every
   metric counted some frames 2-4× and bootstrap ranges came out too narrow.
   → ``test/`` holds **unique frames** (primary metric); ``test_official398/`` keeps the exact
   official 398 images **only** for the GhostVision head-to-head (same split they report on).
3. **The validation set was a different sonar.** The official ``valid`` split is 555 orange
   ``Contact_*_sslo`` crops (horizontal range, clean shadows) while test is greyscale Humminbird
   sonograms (vertical range). Model selection and thresholds would be tuned on the wrong sonar.
   → ``val/`` = **held-out TRAINING RECORDINGS** (Rec10, Rec12, Rec16 — ~16% of unique train frames
   and pots, whole recordings so no neighbouring chunks leak) + v2's wreck/seabed val frames;
   ``test_xsonar/`` = the Contact_sslo crops as a **cross-sonar** generalisation test.

Also written: ``groups.json`` (image → frame key + recording/source group) so evaluation can
bootstrap **by frame group** instead of by image, and ``manifest.json`` with every count.

Pixel-level leakage check (``runs/_patch/leak_probe.py``, 2026-09-24): no unique test frame has a
train twin (max 32×32 thumbnail correlation 0.92, and that pair is from *different* recordings —
similar speckle texture, not the same frame). ``Rec14_Sensor_Depth`` (train) ≠ ``Rec14_wcp`` (test).

Images are hard-linked from v2 (instant, ~zero extra disk); labels are copied.

Usage:  python DATASET/scripts/build_dataset_v2b.py
Then:   python DATASET/scripts/audit_dataset.py DATASET/03_yolo_ready_dataset_v2b
"""
from __future__ import annotations

import json
import os
import re
import shutil
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import cv2
import numpy as np

DATASET = Path(__file__).resolve().parents[1]
SRC = DATASET / "03_yolo_ready_dataset_v2"
DST = DATASET / "03_yolo_ready_dataset_v2b"
CLASS_NAMES = ["ghost_gear", "wreck_debris"]
VAL_RECORDINGS = ("Rec10", "Rec12", "Rec16")      # held-out TRAINING recordings (whole)
BLACK_LEVEL = 4                                    # a pixel this dark in all channels = padding
ROTATED_BLACK_FRAC = 0.02                          # ≥ this black-border share ⇒ likely rotated copy


# ---- naming ------------------------------------------------------------------------------------
def frame_key(name: str) -> str:
    """Roboflow copies share everything before ``.rf.<hash>``; ``_png_jpg`` / ``_jpg`` variants of
    the same chunk are merged too (conservative: fewer unique frames ⇒ wider, honest CIs)."""
    stem = Path(name).stem
    base = stem.split(".rf.")[0]
    return re.sub(r"(_png)?(_jpg)?$", "", base)


def recording(key: str) -> str | None:
    m = re.search(r"rec0*(\d+)", key.lower())
    return f"Rec{int(m.group(1))}" if m else None


def group_of(key: str) -> str:
    """Bootstrap group: the recording+side for sonograms, else the source/site prefix."""
    r = recording(key)
    if r:
        side = "port" if "_port" in key else ("star" if "_star" in key else "")
        return f"{r}_{side}" if side else r
    return re.sub(r"[_\-]?\d+$", "", key) or key


def source_of(key: str) -> str:
    k = key.lower()
    if k.startswith("rec"):
        return "crabpot_sonogram"
    for p in ("contact", "bc_post", "baycove", "ti", "mc", "bb", "seabed", "shipwreck", "mpulse"):
        if k.startswith(p):
            return p
    return k.split("_")[0]


# ---- copy selection ----------------------------------------------------------------------------
def _stats(img_path: Path) -> tuple[float, float]:
    im = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if im is None:
        return 1.0, 0.0
    black = float((im.max(axis=2) <= BLACK_LEVEL).mean())
    return black, float(im.mean())


def pick_copy(paths: list[Path]) -> tuple[Path, dict]:
    """Least-changed copy: smallest black border (un-rotated), then brightness nearest the median."""
    stats = {p: _stats(p) for p in paths}
    med = float(np.median([s[1] for s in stats.values()]))
    best = min(paths, key=lambda p: (round(stats[p][0], 3), abs(stats[p][1] - med), p.name))
    rotated = sum(stats[p][0] >= ROTATED_BLACK_FRAC for p in paths)
    return best, {"copies": len(paths), "rotated_copies": rotated,
                  "kept_black_frac": round(stats[best][0], 4)}


# ---- io ----------------------------------------------------------------------------------------
def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _emit(split: str, img: Path, src_split: str, groups: dict) -> Counter:
    lbl = SRC / src_split / "labels" / f"{img.stem}.txt"
    _link(img, DST / split / "images" / img.name)
    out_lbl = DST / split / "labels" / f"{img.stem}.txt"
    out_lbl.parent.mkdir(parents=True, exist_ok=True)
    text = lbl.read_text() if lbl.exists() else ""
    out_lbl.write_text(text)
    k = frame_key(img.name)
    groups[split][img.name] = {"frame": k, "group": group_of(k), "source": source_of(k)}
    c = Counter()
    for line in text.splitlines():
        p = line.split()
        if len(p) >= 5:
            c[int(float(p[0]))] += 1
    return c


def _images(split: str) -> list[Path]:
    d = SRC / split / "images"
    return sorted(p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def dedupe(paths: list[Path]) -> tuple[list[Path], dict]:
    """One copy per Roboflow frame key; non-Roboflow images pass through untouched."""
    by_key: dict[str, list[Path]] = defaultdict(list)
    passthrough = []
    for p in paths:
        (by_key[frame_key(p.name)] if ".rf." in p.name else passthrough).append(p)
    kept, stats = list(passthrough), Counter()
    for k, ps in sorted(by_key.items()):
        best, info = pick_copy(ps)
        kept.append(best)
        stats["frames"] += 1
        stats["copies"] += info["copies"]
        stats["frames_with_rotated_copies"] += int(info["rotated_copies"] > 0)
        stats["rotated_copies_dropped"] += info["rotated_copies"] - int(info["kept_black_frac"] >= ROTATED_BLACK_FRAC)
        stats["kept_still_black_bordered"] += int(info["kept_black_frac"] >= ROTATED_BLACK_FRAC)
    stats["passthrough"] = len(passthrough)
    return sorted(kept), dict(stats)


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"v2 not found: {SRC} (run build_dataset_v2.py first)")
    if DST.exists():
        shutil.rmtree(DST)
    groups: dict[str, dict] = defaultdict(dict)
    boxes: dict[str, Counter] = defaultdict(Counter)
    counts: dict[str, int] = {}

    # --- train / val: hold out whole training recordings -----------------------------------
    train_all = _images("train")
    held = [p for p in train_all if recording(frame_key(p.name)) in VAL_RECORDINGS]
    train_rest = [p for p in train_all if recording(frame_key(p.name)) not in VAL_RECORDINGS]
    train_kept, train_stats = dedupe(train_rest)
    val_rec_kept, val_rec_stats = dedupe(held)
    v2_val = _images("val")
    xsonar = [p for p in v2_val if frame_key(p.name).lower().startswith("contact")]
    v2_val_other = [p for p in v2_val if p not in xsonar]
    val_other_kept, val_other_stats = dedupe(v2_val_other)
    xsonar_kept, xsonar_stats = dedupe(xsonar)

    for p in train_kept:
        boxes["train"] += _emit("train", p, "train", groups)
    for p in val_rec_kept:
        boxes["val"] += _emit("val", p, "train", groups)
    for p in val_other_kept:
        boxes["val"] += _emit("val", p, "val", groups)
    for p in xsonar_kept:
        boxes["test_xsonar"] += _emit("test_xsonar", p, "val", groups)

    # --- test: unique frames (primary) + the exact official 398 (GhostVision only) ---------
    test_all = _images("test")
    test_kept, test_stats = dedupe(test_all)
    for p in test_kept:
        boxes["test"] += _emit("test", p, "test", groups)
    official = [ln.strip() for ln in (SRC / "official_crabpot_test.txt").read_text().splitlines() if ln.strip()]
    # official entries are bare stems WITHOUT an extension ("…_jpg.rf.<hash>"), so compare the
    # whole string — Path(o).stem would strip ".<hash>" as if it were a suffix.
    off_stems = {o[:-4] if o.lower().endswith((".jpg", ".png")) else o for o in official}
    for p in test_all:
        if p.stem in off_stems:
            boxes["test_official398"] += _emit("test_official398", p, "test", groups)

    for split in ("train", "val", "test", "test_official398", "test_xsonar"):
        d = DST / split / "images"
        counts[split] = sum(1 for _ in d.iterdir()) if d.exists() else 0

    # --- leakage asserts (by frame key AND by held-out recording) ---------------------------
    keys = {s: {v["frame"] for v in groups[s].values()} for s in ("train", "val", "test")}
    leak = {"train_val": len(keys["train"] & keys["val"]), "train_test": len(keys["train"] & keys["test"]),
            "val_test": len(keys["val"] & keys["test"])}
    assert not any(leak.values()), f"frame leakage: {leak}"
    rec_train = {recording(k) for k in keys["train"]} - {None}
    assert not (rec_train & set(VAL_RECORDINGS)), "held-out recording leaked into train"

    (DST / "groups.json").write_text(json.dumps(groups, indent=1))
    (DST / "official_crabpot_test.txt").write_text("\n".join(official) + "\n")
    (DST / "data.yaml").write_text(
        "# dataset v2b — deduplicated, recording-level val, unique-frame test (build_dataset_v2b.py)\n"
        "path: .\ntrain: train/images\nval: val/images\ntest: test/images\n"
        f"nc: {len(CLASS_NAMES)}\nnames:\n" + "".join(f"- {n}\n" for n in CLASS_NAMES))
    manifest = {
        "version": "v2b", "created": date.today().isoformat(), "parent": "v2",
        "taxonomy": dict(enumerate(CLASS_NAMES)),
        "fixes": ["one copy per Roboflow frame (rotated copies dropped)",
                  "val = held-out training recordings (same sonar as test)",
                  "test = unique frames; test_official398 kept for GhostVision only",
                  "Contact_sslo -> test_xsonar (cross-sonar)"],
        "val_recordings": list(VAL_RECORDINGS),
        "images": counts,
        "class_boxes": {s: {CLASS_NAMES[k]: v for k, v in sorted(c.items())} for s, c in boxes.items()},
        "background_images": {s: sum(1 for f in (DST / s / "labels").iterdir() if f.stat().st_size == 0)
                              for s in counts if (DST / s / "labels").exists()},
        "dedupe": {"train": train_stats, "val_recordings": val_rec_stats, "val_other": val_other_stats,
                   "test": test_stats, "test_xsonar": xsonar_stats},
        "leakage_shared_frames": leak,
        "bootstrap_groups": {s: len({v["group"] for v in groups[s].values()}) for s in groups},
    }
    (DST / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in ("images", "class_boxes", "background_images", "leakage_shared_frames",
                                               "bootstrap_groups")}, indent=1))


if __name__ == "__main__":
    main()
