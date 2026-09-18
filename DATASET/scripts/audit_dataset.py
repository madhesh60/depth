"""
audit_dataset.py — Ground-truth audit of the YOLO-ready marine-debris dataset.

Scans DATASET/03_yolo_ready_dataset/{train,val,test}/labels and reports, per split
and overall:
  - image/label file counts and background (empty-label) images
  - class-id distribution
  - annotation format: box (5 tokens) vs polygon (>5 tokens), per class
  - degenerate full-frame boxes (w>0.95 and h>0.95)
  - tiny boxes (area < 0.0005 of the frame)
  - out-of-range / malformed lines
  - source-prefix provenance (crabpot_, uatd_, icra_, vid_, mpulse_, seabed_, shipwreck_)
  - cross-split leakage via a normalised base key (strips aug/hash suffixes)

Read-only. Writes a JSON summary to DATASET/exports/audit_summary.json.

Usage:  python DATASET/scripts/audit_dataset.py [DATASET_ROOT]
        (default root: DATASET/03_yolo_ready_dataset)
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "03_yolo_ready_dataset"
ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else _DEFAULT_ROOT
EXPORTS = Path(__file__).resolve().parents[1] / "exports"
SPLITS = ("train", "val", "test")

# Known source prefixes -> friendly name (order matters: longest/most specific first)
PREFIXES = [
    ("crabpot", "crab_pot"),
    ("uatd", "uatd"),
    ("icra", "icra"),
    ("mpulse", "mpulse"),
    ("seabed", "seabed"),
    ("shipwreck", "shipwreck"),
    ("vid", "trashcan"),
]

FULLFRAME = 0.95          # w and h both above this => degenerate full-frame box
TINY_AREA = 0.0005        # w*h below this => suspiciously tiny box


def source_of(name: str) -> str:
    low = name.lower()
    for prefix, friendly in PREFIXES:
        if low.startswith(prefix):
            return friendly
    return "other"


def base_key(name: str) -> str:
    """Normalise a filename to detect the same underlying image across splits/augs."""
    key = name
    key = re.sub(r"_aug\d+$", "", key)             # strip augmentation suffix
    key = re.sub(r"\.rf\.[0-9a-f]+$", "", key)      # strip roboflow hash
    return key


def audit_split(split: str):
    lbl_dir = ROOT / split / "labels"
    img_dir = ROOT / split / "images"
    labels = sorted(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

    stats = {
        "images": len(list(img_dir.glob("*.*"))) if img_dir.exists() else 0,
        "label_files": len(labels),
        "background_images": 0,          # label file empty
        "total_annotations": 0,
        "class_boxes": Counter(),        # class_id -> box-format count
        "class_polygons": Counter(),     # class_id -> polygon-format count
        "fullframe_boxes": 0,
        "tiny_boxes": 0,
        "malformed_lines": 0,
        "source_images": Counter(),      # source -> image count
        "source_annotations": Counter(), # source -> annotation count
        "base_keys": set(),
    }

    for lf in labels:
        src = source_of(lf.stem)
        stats["source_images"][src] += 1
        stats["base_keys"].add(base_key(lf.stem))
        lines = [ln.strip() for ln in lf.read_text().splitlines() if ln.strip()]
        if not lines:
            stats["background_images"] += 1
            continue
        for ln in lines:
            toks = ln.split()
            stats["total_annotations"] += 1
            stats["source_annotations"][src] += 1
            try:
                cid = int(float(toks[0]))
                vals = [float(v) for v in toks[1:]]
            except (ValueError, IndexError):
                stats["malformed_lines"] += 1
                continue
            if len(vals) == 4:  # standard YOLO box: x y w h
                x, y, w, h = vals
                stats["class_boxes"][cid] += 1
                if w > FULLFRAME and h > FULLFRAME:
                    stats["fullframe_boxes"] += 1
                if w * h < TINY_AREA:
                    stats["tiny_boxes"] += 1
                if not all(0.0 <= v <= 1.0 for v in vals):
                    stats["malformed_lines"] += 1
            elif len(vals) >= 6 and len(vals) % 2 == 0:  # polygon: x1 y1 x2 y2 ...
                stats["class_polygons"][cid] += 1
            else:
                stats["malformed_lines"] += 1
    return stats


def main():
    per_split = {s: audit_split(s) for s in SPLITS}

    # cross-split leakage: base keys shared between splits
    keysets = {s: per_split[s]["base_keys"] for s in SPLITS}
    leakage = {
        "train_val": len(keysets["train"] & keysets["val"]),
        "train_test": len(keysets["train"] & keysets["test"]),
        "val_test": len(keysets["val"] & keysets["test"]),
    }

    # ---- console report ----
    all_class_boxes = Counter()
    all_class_polys = Counter()
    print("=" * 72)
    print("MARINE DEBRIS DATASET AUDIT")
    print("=" * 72)
    for s in SPLITS:
        st = per_split[s]
        all_class_boxes.update(st["class_boxes"])
        all_class_polys.update(st["class_polygons"])
        print(f"\n[{s.upper()}]  images={st['images']}  labels={st['label_files']}  "
              f"background={st['background_images']}  annotations={st['total_annotations']}")
        print(f"  boxes    : {dict(sorted(st['class_boxes'].items()))}")
        print(f"  polygons : {dict(sorted(st['class_polygons'].items()))}")
        print(f"  fullframe boxes={st['fullframe_boxes']}  tiny={st['tiny_boxes']}  "
              f"malformed={st['malformed_lines']}")
        print(f"  sources  : {dict(st['source_images'].most_common())}")

    print("\n" + "-" * 72)
    print("OVERALL class distribution")
    all_ids = sorted(set(all_class_boxes) | set(all_class_polys))
    for cid in all_ids:
        print(f"  class {cid}: boxes={all_class_boxes[cid]:>7}  "
              f"polygons={all_class_polys[cid]:>6}  "
              f"total={all_class_boxes[cid] + all_class_polys[cid]:>7}")
    print(f"\nMax class id present: {max(all_ids) if all_ids else 'n/a'}")
    print(f"Cross-split leakage (shared base keys): {leakage}")

    # ---- JSON export ----
    EXPORTS.mkdir(exist_ok=True)
    out = {
        "splits": {
            s: {
                "images": per_split[s]["images"],
                "label_files": per_split[s]["label_files"],
                "background_images": per_split[s]["background_images"],
                "total_annotations": per_split[s]["total_annotations"],
                "class_boxes": dict(sorted(per_split[s]["class_boxes"].items())),
                "class_polygons": dict(sorted(per_split[s]["class_polygons"].items())),
                "fullframe_boxes": per_split[s]["fullframe_boxes"],
                "tiny_boxes": per_split[s]["tiny_boxes"],
                "malformed_lines": per_split[s]["malformed_lines"],
                "source_images": dict(per_split[s]["source_images"].most_common()),
                "source_annotations": dict(per_split[s]["source_annotations"].most_common()),
            }
            for s in SPLITS
        },
        "overall": {
            "class_boxes": dict(sorted(all_class_boxes.items())),
            "class_polygons": dict(sorted(all_class_polys.items())),
            "max_class_id": max(all_ids) if all_ids else None,
        },
        "leakage": leakage,
    }
    out_path = EXPORTS / f"audit_summary_{ROOT.name}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
